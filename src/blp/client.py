"""TODO: Standardize: Response always stores dataframes (cleaned for values_to_string). to_dictionary returns DF to dictionary.
See HistoricalDataResponse
"""
import atexit
import contextlib
import dataclasses
import datetime
import itertools
import json
import logging
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import blpapi
import pandas as pd
from blpapi.event import Event
from opendate import LCL, UTC, Date, DateTime, Timezone

from blp.parse import Name, Parser, ResponseError, SubscriptionParser
from libb import is_null

try:
    import win32api
    import win32con
except ModuleNotFoundError:
    pass

logger = logging.getLogger(__name__)

__all__ = [
    # Main client
    'Blp',
    # Session
    'Session',
    'SessionFactory',
    'SessionError',
    'SessionCreateError',
    'SessionStartError',
    'SessionTerminatedError',
    'SessionNotAvailableError',
    'RequestTimeoutError',
    'RequestSecurityError',
    'RequestFieldError',
    'ResponseError',
    # Requests
    'HistoricalDataRequest',
    'ReferenceDataRequest',
    'IntradayTickRequest',
    'IntradayBarRequest',
    'EQSRequest',
    'BQLRequest',
    # Responses
    'HistoricalDataResponse',
    'ReferenceDataResponse',
    'IntradayTickResponse',
    'IntradayBarResponse',
    'EQSResponse',
    'BQLResponse',
    # Subscription
    'Subscription',
    'SubscriptionStatus',
    'TopicState',
]


def create_daterange(
    beg: datetime.date | str | None,
    end: datetime.date | str | None) -> tuple[DateTime, DateTime]:
    """UTC start and end instants for an intraday range request.

    Parameters
    ----------
    beg : datetime.date, str or None
        Range start. A naive date, datetime or string is read as UTC.
        None means one day before end.
    end : datetime.date, str or None
        Range end, read as beg is. None means now.

    Returns
    -------
    tuple[DateTime, DateTime]
        (beg, end) in UTC.

    Examples
    --------
    >>> import datetime
    >>> beg, end = datetime.datetime(2024, 1, 1), datetime.datetime(2024, 1, 2)
    >>> create_daterange(beg, end)
    (DateTime(2024, 1, 1, 0, 0, 0, tzinfo=Timezone('UTC')), DateTime(2024, 1, 2, 0, 0, 0, tzinfo=Timezone('UTC')))

    >>> beg, end = datetime.date(2024, 1, 1), datetime.date(2024, 1, 2)
    >>> create_daterange(beg, end)
    (DateTime(2024, 1, 1, 0, 0, 0, tzinfo=Timezone('UTC')), DateTime(2024, 1, 2, 0, 0, 0, tzinfo=Timezone('UTC')))

    >>> beg, end = '1975-05-21T22:00:00-04:00', '1975-05-22T22:00:00-04:00'
    >>> create_daterange(beg, end)
    (DateTime(1975, 5, 22, 2, 0, 0, tzinfo=Timezone('UTC')), DateTime(1975, 5, 23, 2, 0, 0, tzinfo=Timezone('UTC')))

    """
    if is_null(end):
        end = DateTime.now(UTC)
    elif isinstance(end, datetime.date):
        end = DateTime.instance(end).in_timezone(UTC)
    else:
        end = DateTime.parse(end).in_timezone(UTC)
    if is_null(beg):
        beg = end.subtract(days=1)
    elif isinstance(beg, datetime.date):
        beg = DateTime.instance(beg).in_timezone(UTC)
    else:
        beg = DateTime.parse(beg).in_timezone(UTC)
    return beg, end


def calendar_date(value: datetime.date | str) -> Date:
    """The calendar date value names in its own timezone.

    Parameters
    ----------
    value : datetime.date or str
        A date, a datetime or pandas Timestamp, naive or aware, or a string
        Date.parse reads. An offset in the value never shifts the date, so
        '2024-01-02T20:00-05:00' is 2024-01-02.

    Returns
    -------
    Date

    Raises
    ------
    ValueError
        Date.parse cannot read the string.
    """
    if isinstance(value, datetime.date):
        return Date.instance(value)
    date = Date.parse(value)
    if date is None:
        raise ValueError(f'No date in {value!r}')
    return date


def is_not_started(exc: blpapi.Exception) -> bool:
    """True for the error blpapi raises on a session that is not started.
    """
    return 'Session Not Started' in str(exc) or type(exc).__name__ == 'InvalidStateException'


class SessionError(RuntimeError):
    pass


class SessionCreateError(SessionError):
    """Handle create session failed event"""


class SessionStartError(SessionError):
    """Handle start session failed event"""


class SessionTerminatedError(SessionError):
    """Handle start terminated failed event"""


class SessionNotAvailableError(SessionError):
    """No usable Bloomberg session on this host (e.g. terminal not logged in)."""


class RequestTimeoutError(SessionError):
    """A request passed its Blp idle_seconds or total_seconds limit.
    """


class RequestSecurityError(Exception):
    """A request set to raise on a securityError received one.
    """


class RequestFieldError(Exception):
    """A request set to raise on a fieldExceptions entry received one.
    """


REQUEST_IDLE_SECONDS = 120
FIRST_DATA_SECONDS = 60
TICK_WINDOW_SECONDS = 60
SESSION_FAILURE_CATEGORIES = {'CONNECTION_DEAD', 'SERVICE_NOT_AVAILABLE'}
REQUEST_EVENT_TYPES = {Event.RESPONSE, Event.PARTIAL_RESPONSE, Event.REQUEST_STATUS}


class BaseRequest(ABC):
    """Base Request Object"""

    def __init__(
        self,
        service_name,
        raise_security_error=False,
        raise_field_error=False,
        force_string=False,
    ):
        self.field_errors = []
        self.security_errors = []
        self.raise_security_error = raise_security_error
        self.raise_field_error = raise_field_error
        self.service_name = service_name
        self.force_string = force_string
        self.response = None

    @abstractmethod
    def prepare_response(self) -> None:
        """Initialize the response object for this request.
        """

    @property
    def has_exception(self) -> bool:
        """Check if request has security or field errors that should raise exceptions.
        """
        return (self.raise_security_error and self.security_errors) \
            or (self.raise_field_error and self.field_errors)

    def raise_exception(self) -> None:
        """Raise the collected errors the request is set to raise; log the rest.

        Raises
        ------
        RequestSecurityError
            raise_security_error is set and a security error was collected.
            Checked before field errors.
        RequestFieldError
            raise_field_error is set and a field error was collected.
        """
        if self.security_errors:
            msgs = ''.join([str(s) for s in self.security_errors])
            if self.raise_security_error:
                raise RequestSecurityError(msgs)
            logger.debug(f'Security Errors:\n{msgs}')
        if self.field_errors:
            msgs = ''.join([str(s) for s in self.field_errors])
            if self.raise_field_error:
                raise RequestFieldError(msgs)
            logger.debug(f'Field Errors:\n{msgs}')

    @abstractmethod
    def create_request(self, service) -> blpapi.Request:
        """Create Bloomberg API request object from service.
        """

    @abstractmethod
    def process_response(self, event, is_final) -> None:
        """Process response event and update response object.
        """

    def on_admin_event(self, event) -> None:
        """Handle administrative session events.
        """
        for message in event:
            if message.messageType() == Name.SESSION_CONNECTION_UP:
                logger.info('Connected ...')
            elif Name.SESSION_STARTED in message:
                logger.info('Started new Bbg session ...')
            elif 'ServiceOpened' in message:
                logger.info('Opened Bbg service ...')
            elif message.messageType() == 'SessionTerminated':
                logger.info('Session DONE')
                raise SessionTerminatedError('Session DONE')

    @staticmethod
    def apply_overrides(request, overrides) -> None:
        """Apply field overrides to the request.
        """
        if overrides:
            for k, v in overrides.items():
                o = request.getElement('overrides').appendElement()
                o.setElement(Name.FIELD_ID, k)
                o.setElement('value', v)

    def set_flag(self, request, val, fld) -> None:
        """If the specified val is not None, then set the specified field to its boolean value.
        """
        if val is not None:
            val = bool(val)
            request.set(fld, val)

    def set_response(self, response) -> None:
        """Set the response to handle and store the results.
        """
        self.response = response


class BaseResponse(ABC):
    """Base class for Responses"""

    @abstractmethod
    def as_dataframe(self) -> pd.DataFrame:
        """Convert response to pandas DataFrame.
        """

    @abstractmethod
    def as_dict(self) -> dict:
        """Convert response to dictionary.
        """


class HistoricalDataResponse(BaseResponse):

    def __init__(self, request):
        self.request = request
        self.response_map = {}

    def on_security_complete(self, ticker: str, df: pd.DataFrame) -> None:
        """Store completed security data in response map.
        """
        self.response_map[ticker] = df.astype(object).where(df.notna(), None) \
            if self.request.force_string else df

    def as_dict(self):
        return self.response_map

    def as_dataframe(self):
        """:return: Multi-Index DataFrame"""
        data = dict(self.response_map.items())
        if data.values():
            df = pd.concat(data.values(), keys=data.keys(), axis=1)
        else:
            df = pd.DataFrame()
        return df


class HistoricalDataRequest(BaseRequest):
    """Bloomberg HistoricalDataRequest (BDH).

    Parameters
    ----------
    tickers : str or list[str]
        Bloomberg security identifier(s).
    fields : str or list[str]
        Bloomberg field name(s).
    start : datetime.date, str or None
        First date, as calendar_date reads it: an offset in the value never
        shifts the date. None means the day before end.
    end : datetime.date, str or None
        Last date, read as start is. None means this host's date today.
    timezone : str
        Zone for datetime values in the response. The request dates ignore
        it.
    period : str, optional
        DAILY (the default), WEEKLY, MONTHLY, QUARTERLY, SEMI_ANNUALLY or
        YEARLY.
    raise_security_error : bool, default False
        Raise RequestSecurityError for an invalid ticker.
    raise_field_error : bool, default False
        Raise RequestFieldError for an invalid field.
    force_string : bool, default False
        Return every value as a string.
    period_adjustment : str, optional
        ACTUAL, CALENDAR or FISCAL.
    currency : str, optional
        ISO code to convert values into.
    override_option : str, optional
        OVERRIDE_OPTION_CLOSE or OVERRIDE_OPTION_GPA.
    pricing_option : str, optional
        PRICING_OPTION_PRICE or PRICING_OPTION_YIELD.
    non_trading_day_fill_option : str, optional
        NON_TRADING_WEEKDAYS, ALL_CALENDAR_DAYS or ACTIVE_DAYS_ONLY.
    non_trading_day_fill_method : str, optional
        PREVIOUS_VALUE or NIL_VALUE.
    max_data_points : int, optional
        Cap on points per security, counted back from end.
    adjustment_normal, adjustment_abnormal, adjustment_split : bool, optional
        Bloomberg pricing adjustment flags.
    adjustment_follow_DPDF : bool, optional
        Follow the terminal's DPDF adjustment settings.
    calendar_code_override : str, optional
        Two-letter country ISO code.
    **overrides
        Bloomberg field overrides, field id to value.
    """

    def __init__(
        self,
        tickers,
        fields,
        start:datetime.datetime = None,
        end:datetime.datetime = None,
        timezone: str = LCL.name,
        period=None,
        raise_security_error=False,
        raise_field_error=False,
        force_string=False,
        period_adjustment=None,
        currency=None,
        override_option=None,
        pricing_option=None,
        non_trading_day_fill_option=None,
        non_trading_day_fill_method=None,
        max_data_points=None,
        adjustment_normal=None,
        adjustment_abnormal=None,
        adjustment_split=None,
        adjustment_follow_DPDF=None,
        calendar_code_override=None,
        **overrides,
    ):
        super().__init__(
            '//blp/refdata',
            raise_security_error=raise_security_error,
            raise_field_error=raise_field_error,
            force_string=force_string,
        )
        period = period or 'DAILY'
        assert period in {'DAILY', 'WEEKLY', 'MONTHLY', 'QUARTERLY', 'SEMI_ANNUALLY', 'YEARLY'}
        self.is_single_ticker = is_single_ticker = isinstance(tickers, str)
        self.is_single_field = is_single_field = isinstance(fields, str)
        self.tickers = [str(tickers)] if is_single_ticker else [str(t) for t in tickers]
        self.fields = [str(fields)] if is_single_field else [str(f) for f in fields]
        self.end = Date.today() if is_null(end) else calendar_date(end)
        self.start = self.end.subtract(days=1) if is_null(start) else calendar_date(start)
        self.timezone = Timezone(timezone)
        self.parser = Parser(UTC, self.timezone)
        self.period = period
        self.period_adjustment = period_adjustment
        self.currency = currency
        self.override_option = override_option
        self.pricing_option = pricing_option
        self.non_trading_day_fill_option = non_trading_day_fill_option
        self.non_trading_day_fill_method = non_trading_day_fill_method
        self.max_data_points = max_data_points
        self.adjustment_normal = adjustment_normal
        self.adjustment_abnormal = adjustment_abnormal
        self.adjustment_split = adjustment_split
        self.adjustment_follow_DPDF = adjustment_follow_DPDF
        self.calendar_code_override = calendar_code_override
        self.overrides = overrides

    def __repr__(self):
        fmtargs = {
            'clz': self.__class__.__name__,
            'symbols': ','.join(self.tickers),
            'fields': ','.join(self.fields),
            'start': self.start.strftime('%Y%m%d'),
            'end': self.end.strftime('%Y%m%d'),
            'period': self.period,
        }
        # TODO: add self.overrides if defined
        return '<{clz}([{symbols}], [{fields}], start={start}, end={end}, period={period}'.format(**fmtargs)

    def prepare_response(self):
        self.response = HistoricalDataResponse(self)

    def create_request(self, service):
        request = service.createRequest('HistoricalDataRequest')
        [request.append('securities', sec) for sec in self.tickers]
        [request.append('fields', fld) for fld in self.fields]
        request.set('startDate', self.start.strftime('%Y%m%d'))
        request.set('endDate', self.end.strftime('%Y%m%d'))
        request.set('periodicitySelection', self.period)
        self.period_adjustment and request.set('periodicityAdjustment', self.period_adjustment)
        self.currency and request.set('currency', self.currency)
        self.override_option and request.set('overrideOption', self.override_option)
        self.pricing_option and request.set('pricingOption', self.pricing_option)
        self.non_trading_day_fill_option and request.set('nonTradingDayFillOption', self.non_trading_day_fill_option)
        self.non_trading_day_fill_method and request.set('nonTradingDayFillMethod', self.non_trading_day_fill_method)
        self.max_data_points and request.set('maxDataPoints', self.max_data_points)
        self.calendar_code_override and request.set('calendarCodeOverride', self.calendar_code_override)
        self.set_flag(request, self.adjustment_normal, 'adjustmentNormal')
        self.set_flag(request, self.adjustment_abnormal, 'adjustmentAbnormal')
        self.set_flag(request, self.adjustment_split, 'adjustmentSplit')
        self.set_flag(request, self.adjustment_follow_DPDF, 'adjustmentFollowDPDF')

        if hasattr(self, 'overrides') and self.overrides is not None:
            self.apply_overrides(request, self.overrides)
        return request

    def on_security_data_element(self, element) -> None:
        """Process a securityData element - FIXME: currently not handling relateDate element.
        """
        ticker = self.parser.get_subelement_value(element, Name.SECURITY, self.force_string)
        fields = element.getElement(Name.FIELD_DATA)
        dmap = defaultdict(list)
        for pt in fields.values():
            for f in ['date'] + self.fields:
                field_val = self.parser.get_subelement_value(pt, f, self.force_string)
                if isinstance(field_val, datetime.datetime):
                    field_val = field_val.in_timezone(self.timezone)
                dmap[f].append(field_val)
        if not dmap:
            df = pd.DataFrame(columns=self.fields)
        else:
            idx = dmap.pop('date')
            df = pd.DataFrame(dmap, columns=self.fields, index=idx)
            df.index.name = 'date'
        self.response.on_security_complete(ticker, df)

    def process_response(self, event, is_final) -> None:
        """Process response event and update response object.
        """
        for message in self.parser.message_iter(event):
            # single security element in historical request
            element = message.getElement(Name.SECURITY_DATA)
            if Name.SECURITY_ERROR in element:
                ticker = self.parser.get_subelement_value(element, Name.SECURITY, self.force_string)
                self.security_errors.append(self.parser.as_security_error(element.getElement(Name.SECURITY_ERROR), ticker))
            else:
                self.on_security_data_element(element)
                self.field_errors.extend(self.parser.get_field_errors(element))


class ReferenceDataResponse(BaseResponse):

    def __init__(self, request):
        self.request = request
        self.response_map = defaultdict(dict)

    def on_security_data(self, ticker: str, fieldmap: dict) -> None:
        """Store security field data in response map.
        """
        self.response_map[ticker].update(fieldmap)

    def as_dict(self):
        return self.response_map

    def as_dataframe(self):
        """:return: Multi-Index DataFrame"""
        data = {ticker: pd.Series(data) for ticker, data in self.response_map.items()}
        df = pd.DataFrame.from_dict(data, orient='index')
        df = df.reindex(self.request.fields, axis=1)  # layer in any missing fields just in case
        if self.request.force_string:
            df = df.astype(object).where(df.notna(), None)
        return df


class ReferenceDataRequest(BaseRequest):

    def __init__(
        self,
        tickers,
        fields,
        raise_security_error=False,
        raise_field_error=False,
        decimal_places: int = None,
        return_formatted_value=None,
        field_parse_custom: dict | None = None,
        timezone: str = LCL.name,
        force_string=False,
        time_as_datetime=False,
        **overrides,
    ):
        """response_type: (df, map) how to return the results"""
        super().__init__(
            '//blp/refdata',
            raise_security_error=raise_security_error,
            raise_field_error=raise_field_error,
            force_string=force_string,
        )
        self.is_single_ticker = isinstance(tickers, str)
        self.is_single_field = isinstance(fields, str)
        self.tickers = [str(tickers)] if isinstance(tickers, str) else [str(t) for t in tickers]
        self.fields = [str(fields)] if isinstance(fields, str) else [str(f) for f in fields]
        self.return_formatted_value = return_formatted_value
        self.timezone = Timezone(timezone)
        self.parser = Parser(
            assumed_timezone=UTC,
            desired_timezone=self.timezone,
            time_as_datetime=time_as_datetime,
            decimal_places=decimal_places,
            field_parse_custom=field_parse_custom,
        )
        self.overrides = overrides

    def __repr__(self):
        fmtargs = {
            'clz': self.__class__.__name__,
            'tickers': ','.join(self.tickers),
            'fields': ','.join(self.fields),
            'overrides': ','.join([f'{k}={v}' for k, v in self.overrides.items()]),
        }
        return '<{clz}([{tickers}], [{fields}], overrides={overrides})'.format(**fmtargs)

    def prepare_response(self):
        self.response = ReferenceDataResponse(self)

    def create_request(self, service):
        request = service.createRequest('ReferenceDataRequest')
        [request.append('securities', sec) for sec in self.tickers]
        [request.append('fields', fld) for fld in self.fields]
        self.set_flag(request, self.return_formatted_value, 'returnFormattedValue')
        self.set_flag(request, True, 'useUTCTime')
        self.apply_overrides(request, self.overrides)
        return request

    def process_response(self, event, is_final) -> None:
        """Process response event and update response object.
        """
        for msg in self.parser.message_iter(event):
            for element, err in self.parser.security_element_iter(msg.getElement(Name.SECURITY_DATA)):
                if err:
                    self.security_errors.append(err)
                    continue
                self._process_security_element(element)

    def _process_security_element(self, element) -> None:
        """Extract and store field data from security element.
        """
        ticker = self.parser.get_subelement_value(element, Name.SECURITY, self.force_string)
        fields = element.getElement(Name.FIELD_DATA)
        field_data = self.parser.get_subelement_values(fields, self.fields, self.force_string)
        field_data = [x.in_timezone(self.timezone) if isinstance(x, datetime.datetime) else x for x in field_data]
        assert len(field_data) == len(self.fields), 'Field length must match data length'
        self.response.on_security_data(ticker, dict(list(zip(self.fields, field_data))))
        field_errors = self.parser.get_field_errors(element)
        field_errors and self.field_errors.extend(field_errors)


class IntradayTickResponse(BaseResponse):

    def __init__(self, request):
        self.request = request
        self.ticks = []  # iterdict

    def as_dict(self):
        raise NotImplementedError('Needs to be implemented')

    def as_dataframe(self):
        """Return a data frame with no set index"""
        df = pd.DataFrame.from_records(self.ticks)
        return df.astype(object).where(df.notna(), None) \
            if self.request.force_string else df


class IntradayTickRequest(BaseRequest):
    """Intraday tick request.
    Note: returns UTC datetime by default.
    """
    def __init__(
        self,
        ticker,
        start: datetime.datetime = None,
        end: datetime.datetime = None,
        timezone: str = LCL.name,
        events=('TRADE', 'BID', 'ASK', 'BID_BEST', 'ASK_BEST', 'MID_PRICE', 'AT_TRADE', 'BEST_BID', 'BEST_ASK'),
        include_condition_codes=None,
        include_nonplottable_events=None,
        include_exchange_codes=None,
        return_eids=None,
        include_broker_codes=None,
        include_equity_ref_price=None,
        include_action_codes=None,
        include_indicator_codes=None,
        include_rsp_codes=None,
        include_trade_time=None,
        include_bic_mic_codes=None,
        include_spread_price=None,
        include_yield=None,
        adjustment_normal=None,
        adjustment_abnormal=None,
        adjustment_split=None,
        adjustment_follow_DPDF=None,
        include_client_specific_fields=None,
        include_trade_id=None,
        include_upfront_price=None,
        include_trade_date=None,
        include_bloomberg_standard_condition_codes=None,
    ):
        super().__init__('//blp/refdata')
        self.ticker = ticker
        self.events = [events] if isinstance(events, str) else events
        self.include_condition_codes = include_condition_codes
        self.include_nonplottable_events = include_nonplottable_events
        self.include_exchange_codes = include_exchange_codes
        self.return_eids = return_eids
        self.include_broker_codes = include_broker_codes
        self.include_equity_ref_price = include_equity_ref_price
        self.include_rsp_codes = include_rsp_codes
        self.include_bic_mic_codes = include_bic_mic_codes
        self.include_action_codes = include_action_codes
        self.include_indicator_codes = include_indicator_codes
        self.include_trade_time = include_trade_time
        self.include_spread_price = include_spread_price
        self.include_yield = include_yield
        self.adjustment_normal = adjustment_normal
        self.adjustment_abnormal = adjustment_abnormal
        self.adjustment_split = adjustment_split
        self.adjustment_follow_DPDF = adjustment_follow_DPDF
        self.include_client_specific_fields = include_client_specific_fields
        self.include_trade_id = include_trade_id
        self.include_upfront_price = include_upfront_price
        self.include_trade_date = include_trade_date
        self.include_bloomberg_standard_condition_codes = include_bloomberg_standard_condition_codes
        self.start, self.end = create_daterange(start, end)
        self.timezone = Timezone(timezone)
        self.parser = Parser(UTC, self.timezone)

    def __repr__(self):
        fmtargs = {'clz': self.__class__.__name__, 'ticker': self.ticker, 'events': ','.join(self.events)}
        return '<{clz}({ticker}, [{events}])'.format(**fmtargs)

    def prepare_response(self):
        self.response = IntradayTickResponse(self)

    def create_request(self, service):
        request = service.createRequest('IntradayTickRequest')
        request.set(Name.SECURITY, self.ticker)
        [request.append('eventTypes', event) for event in self.events]
        request.set('startDateTime', self.start)
        request.set('endDateTime', self.end)
        self.set_flag(request, self.include_condition_codes, 'includeConditionCodes')
        self.set_flag(request, self.include_nonplottable_events, 'includeNonPlottableEvents')
        self.set_flag(request, self.include_exchange_codes, 'includeExchangeCodes')
        self.set_flag(request, self.return_eids, 'returnEids')
        self.set_flag(request, self.include_broker_codes, 'includeBrokerCodes')
        self.set_flag(request, self.include_spread_price, 'includeSpreadPrice')
        self.set_flag(request, self.include_yield, 'includeYield')
        self.set_flag(request, self.adjustment_normal, 'adjustmentNormal')
        self.set_flag(request, self.adjustment_abnormal, 'adjustmentAbnormal')
        self.set_flag(request, self.adjustment_split, 'adjustmentSplit')
        self.set_flag(request, self.adjustment_follow_DPDF, 'adjustmentFollowDPDF')
        self.set_flag(request, self.include_client_specific_fields, 'includeClientSpecificFields')
        self.set_flag(request, self.include_equity_ref_price, 'includeEqRefPrice')
        self.set_flag(request, self.include_trade_id, 'includeTradeId')
        self.set_flag(request, self.include_rsp_codes, 'includeRpsCodes')
        self.set_flag(request, self.include_bic_mic_codes, 'includeBicMicCodes')
        self.set_flag(request, self.include_upfront_price, 'includeUpfrontPrice')
        self.set_flag(request, self.include_action_codes, 'includeActionCodes')
        self.set_flag(request, self.include_indicator_codes, 'includeIndicatorCodes')
        self.set_flag(request, self.include_trade_date, 'includeTradeDate')
        self.set_flag(request, self.include_trade_time, 'includeTradeTime')
        self.set_flag(request, self.include_bloomberg_standard_condition_codes, 'includeBloombergStandardConditionCodes')
        return request

    def on_tick_data(self, ticks) -> None:
        """Process the incoming tick data array.
        """
        for tick in self.parser.element_iter(ticks):
            names = [str(tick.getElement(_).name()) for _ in range(tick.numElements())]
            tickmap = {n: self.parser.get_subelement_value(tick, n) for n in names}
            self.response.ticks.append(tickmap)

    def process_response(self, event, is_final) -> None:
        """Process response event and update response object.
        """
        for msg in self.parser.message_iter(event):
            tdata = msg.getElement('tickData')
            # tickData will have 0 to 1 tickData[] elements
            if 'tickData' in tdata:
                self.on_tick_data(tdata.getElement('tickData'))


class IntradayBarResponse(BaseResponse):

    def __init__(self, request):
        self.request = request
        self.bars = []  # iterdict

    def as_dict(self):
        raise NotImplementedError('Needs to be implemented')

    def as_dataframe(self):
        df = pd.DataFrame.from_records(self.bars)
        return df.astype(object).where(df.notna(), None) if self.request.force_string else df


class IntradayBarRequest(BaseRequest):

    def __init__(
        self,
        ticker,
        start:datetime.datetime = None,
        end:datetime.datetime = None,
        timezone:str = LCL.name,
        event=('TRADE', 'BID', 'ASK', 'BID_BEST', 'ASK_BEST', 'BEST_BID', 'BEST_ASK'),
        interval=None,
        gap_fill_initial_bar=None,
        return_eids=None,
        adjustment_normal=None,
        adjustment_abnormal=None,
        adjustment_split=None,
        adjustment_follow_DPDF=None,
    ):
        """
        Parameters
        ----------
        interval: int, between 1 and 1440 in minutes. If omitted, defaults to 1 minute
        gap_fill_initial_bar: bool
                            If True, bar contains previous values if not ticks during the interval
        """
        super().__init__('//blp/refdata')
        self.ticker = ticker
        self.event = event
        self.interval = interval
        self.gap_fill_initial_bar = gap_fill_initial_bar
        self.return_eids = return_eids
        self.adjustment_normal = adjustment_normal
        self.adjustment_abnormal = adjustment_abnormal
        self.adjustment_split = adjustment_split
        self.adjustment_follow_DPDF = adjustment_follow_DPDF
        self.start, self.end = create_daterange(start, end)
        self.timezone = Timezone(timezone)
        self.parser = Parser(UTC, self.timezone)

    def __repr__(self):
        fmtargs = {
            'clz': self.__class__.__name__,
            'ticker': self.ticker,
            'event': self.event,
            'start': self.start.in_timezone(self.timezone),
            'end': self.end.in_timezone(self.timezone),
        }
        return '<{clz}({ticker}, {event}, start={start}, end={end})'.format(**fmtargs)

    def prepare_response(self):
        self.response = IntradayBarResponse(self)

    def create_request(self, service):
        request = service.createRequest('IntradayBarRequest')
        request.set(Name.SECURITY, self.ticker)
        request.set('eventType', self.event)
        request.set('startDateTime', self.start)
        request.set('endDateTime', self.end)
        request.set('interval', self.interval or 1)
        self.set_flag(request, self.gap_fill_initial_bar, 'gapFillInitialBar')
        self.set_flag(request, self.return_eids, 'returnEids')
        self.set_flag(request, self.adjustment_normal, 'adjustmentNormal')
        self.set_flag(request, self.adjustment_abnormal, 'adjustmentAbnormal')
        self.set_flag(request, self.adjustment_split, 'adjustmentSplit')
        self.set_flag(request, self.adjustment_follow_DPDF, 'adjustmentFollowDPDF')
        return request

    def on_bar_data(self, bars) -> None:
        """Process the incoming bar data array.
        """
        for tick in self.parser.element_iter(bars):
            names = [str(tick.getElement(_).name()) for _ in range(tick.numElements())]
            barmap = {n: self.parser.get_subelement_value(tick, n) for n in names}
            self.response.bars.append(barmap)

    def process_response(self, event, is_final) -> None:
        """Process response event and update response object.
        """
        for msg in self.parser.message_iter(event):
            data = msg.getElement('barData')
            # tickData will have 0 to 1 tickData[] elements
            if 'barTickData' in data:
                self.on_bar_data(data.getElement('barTickData'))


class EQSResponse(BaseResponse):

    def __init__(self, request):
        self.request = request
        self.response_map = defaultdict(dict)

    def on_security_data(self, ticker, fieldmap):
        self.response_map[ticker].update(fieldmap)

    def as_dict(self):
        return self.response_map

    def as_dataframe(self):
        """:return: Multi-Index DataFrame"""
        data = {ticker: pd.Series(data) for ticker, data in self.response_map.items()}
        df = pd.DataFrame.from_dict(data, orient='index')
        return df.astype(object).where(df.notna(), None) if self.request.force_string else df


class EQSRequest(BaseRequest):

    def __init__(
        self,
        name,
        type='GLOBAL',
        group='General',
        asof=None,
        language=None,
        timezone: str = LCL.name,
    ):
        super().__init__('//blp/refdata')
        self.name = name
        self.group = group
        self.type = type
        self.asof = pd.to_datetime(asof) if asof else None
        self.language = language
        self.timezone = Timezone(timezone)
        self.parser = Parser(UTC, self.timezone)

    def __repr__(self):
        fmtargs = {
            'clz': self.__class__.__name__,
            'name': self.name,
            'type': self.type,
            'group': self.group,
            'asof': self.asof,
        }
        return '<{clz}({name}, type={type}, group={group}, asof={asof})'.format(**fmtargs)

    def prepare_response(self):
        self.response = EQSResponse(self)

    def create_request(self, service):
        request = service.createRequest('BeqsRequest')
        request.set('screenName', self.name)
        self.type and request.set('screenType', self.type)
        self.group and request.set('Group', self.group)
        overrides = {}
        if self.asof:
            overrides['PiTDate'] = self.asof.strftime('%Y%m%d')
        if self.language:
            overrides['languageId'] = self.language
        overrides and self.apply_overrides(request, overrides)
        return request

    def process_response(self, event, is_final) -> None:
        """Process response event and update response object.
        """
        for message in self.parser.message_iter(event):
            data = message.getElement('data')
            security = data.getElement(Name.SECURITY_DATA)
            for element, error in self.parser.security_element_iter(security):
                if error:
                    self.security_errors.append(error)
                    continue
                self._process_security_element(element)

    def _process_security_element(self, element) -> None:
        """Extract and store field data from security element.
        """
        ticker = self.parser.get_subelement_value(element, Name.SECURITY, self.force_string)
        fields = element.getElement(Name.FIELD_DATA)
        fldnames = [str(field.name()) for field in fields.elements()]
        fdata = self.parser.get_subelement_values(fields, fldnames)
        self.response.on_security_data(ticker, dict(list(zip(fldnames, fdata))))
        ferrors = self.parser.get_field_errors(element)
        ferrors and self.field_errors.extend(ferrors)


class BQLResponse(BaseResponse):
    """Response for BQL queries, one DataFrame per field.
    """

    def __init__(self, request):
        self.request = request
        self.response_map = {}

    def on_field_data(self, field_name: str, df: pd.DataFrame) -> None:
        """Store DataFrame for a field.
        """
        self.response_map[field_name] = df

    def as_dict(self) -> dict:
        return self.response_map

    def as_dataframe(self) -> pd.DataFrame:
        """Combine all field DataFrames by joining on common columns.
        """
        if not self.response_map:
            return pd.DataFrame()
        dfs = list(self.response_map.values())
        result = dfs[0]
        for df in dfs[1:]:
            common_cols = list(set(result.columns) & set(df.columns))
            if common_cols:
                result = result.merge(df, on=common_cols, how='outer')
        if self.request.force_string:
            result = result.astype(object).where(result.notna(), None)
        return result


class BQLRequest(BaseRequest):
    """Bloomberg Query Language (BQL) Request using //blp/bqlsvc service.
    """

    def __init__(
        self,
        expression: str = None,
        universe: str = None,
        fields: str = None,
        raise_security_error=False,
        raise_field_error=False,
        force_string=False,
    ):
        super().__init__(
            '//blp/bqlsvc',
            raise_security_error=raise_security_error,
            raise_field_error=raise_field_error,
            force_string=force_string,
        )
        self.expression = self._build_expression(expression, universe, fields)

    def _build_expression(self, expression: str, universe: str, fields: str) -> str:
        """Build BQL expression from components or return provided expression.
        """
        if expression:
            return expression
        if not universe or not fields:
            raise ValueError("Either 'expression' or both 'universe' and 'fields' must be provided")
        return f'get({fields}) for({universe})'

    def __repr__(self):
        fmtargs = {
            'clz': self.__class__.__name__,
            'expression': self.expression[:50] + '...' if len(self.expression) > 50 else self.expression,
        }
        return '<{clz}({expression})'.format(**fmtargs)

    def prepare_response(self):
        self.response = BQLResponse(self)

    def create_request(self, service):
        request = service.createRequest('sendQuery')
        request.set('expression', self.expression)
        try:
            ctx = request.getElement('clientContext')
            ctx.setElement('appName', 'EXCEL')
        except blpapi.NotFoundException:
            pass
        return request

    def process_response(self, event, is_final) -> None:
        """Process BQL response events using message.toPy() for dict conversion.
        """
        for message in event:
            msg_data = message.toPy()
            if isinstance(msg_data, str):
                msg_dict = json.loads(msg_data)
            else:
                msg_dict = msg_data
            self._process_bql_dict(msg_dict)

    def _process_bql_dict(self, msg_dict: dict) -> None:
        """Parse BQL response structure into DataFrames.
        """
        exceptions = msg_dict.get('responseExceptions') or []
        if exceptions:
            for exc in exceptions:
                error_msg = exc.get('message') or exc.get('internalMessage') or 'Unknown BQL error'
                self.security_errors.append(error_msg)
            return

        results = msg_dict.get('results') or {}
        for field_name, content in results.items():
            df = self._parse_field_result(field_name, content)
            self.response.on_field_data(field_name, df)

    def _parse_field_result(self, field_name: str, content: dict) -> pd.DataFrame:
        """Parse a single field's result into a DataFrame.
        """
        data = {}
        id_col = content.get('idColumn', {})
        data['ID'] = id_col.get('values', [])
        val_col = content.get('valuesColumn', {})
        data[field_name] = val_col.get('values', [])
        for sec_col in content.get('secondaryColumns', []):
            col_name = sec_col.get('name', '')
            data[col_name] = sec_col.get('values', [])

        df = pd.DataFrame(data)
        df = df.replace('NaN', pd.NA)
        return df


class Session(blpapi.Session):
    """Wrapper around blpapi.Session with auto-closing"""

    def __init__(self, *args):
        super().__init__(*args)
        atexit.register(self.cleanup)

    def open_service(self, service_name: str) -> None:
        """Open service. Raise Exception if fails.
        """
        if not self.openService(service_name):
            raise SessionStartError(f'Failed to open service {service_name}.')

    def cleanup(self) -> None:
        """Stop and destroy the session, and drop its exit hook so the
        session can be garbage collected. Never raises.
        """
        atexit.unregister(self.cleanup)
        with contextlib.suppress(Exception):
            self.stop()
            self.destroy()
            logger.debug('Closed active Bloomberg session')


def create_session(
    host='localhost',
    port=8194,
    auth='AuthenticationType=OS_LOGON',
    application_identity_key=None,
    event_handler=None,
    event_dispatcher=None,
    event_queue_size=10000,
) -> Session:
    """Vanilla wrapper around Session (blpapi.Session)

    - eventHandler: Handler for events generated by the session.
        Takes two arguments - received event and related session
    - eventDispatcher: An optional dispatcher for events.

    If ``eventHandler`` is not ``None`` then this :class:`Session` will
    operate in asynchronous mode, otherwise the :class:`Session` will
    operate in synchronous mode.

    If ``eventDispatcher`` is ``None`` then the :class:`Session` will
    create a default :class:`EventDispatcher` for this :class:`Session`
    which will use a single thread for dispatching events. For more control
    over event dispatching a specific instance of :class:`EventDispatcher`
    can be supplied. This can be used to share a single
    :class:`EventDispatcher` amongst multiple :class:`Session` objects.

    [see blpapi.Session]
    """
    session_options = blpapi.SessionOptions()
    session_options.setServerHost(host)
    session_options.setServerPort(port)
    session_options.setAuthenticationOptions(auth)
    session_options.setMaxEventQueueSize(event_queue_size)
    if application_identity_key:
        session_options.setApplicationIdentityKey(application_identity_key)
    return Session(session_options, event_handler, event_dispatcher)


class SessionFactory:
    """Session Factory"""

    @staticmethod
    def create(
        host='localhost',
        port=8194,
        auth='AuthenticationType=OS_LOGON',
        application_identity_key=None,
        event_handler=None,
        event_dispatcher=None,
    ) -> Session:
        """Started Bloomberg API session.

        Parameters
        ----------
        host : str, default 'localhost'
            BBComm host. BPIPE hosts are outside the tested scope.
        port : int, default 8194
            BBComm port.
        auth : str, default 'AuthenticationType=OS_LOGON'
            blpapi authentication options string.
        application_identity_key : str, optional
            None sets no application identity.
        event_handler : callable, optional
            Called with (event, session). Setting it makes the session
            asynchronous, as subscriptions need.
        event_dispatcher : blpapi.EventDispatcher, optional
            None uses blpapi's own single-thread dispatch.

        Returns
        -------
        Session

        Raises
        ------
        SessionCreateError
            The session failed to start. It is cleaned up first.
        """
        logger.info('Connecting to Bloomberg BBComm session...')
        session = create_session(
            host=host,
            port=port,
            auth=auth,
            event_handler=event_handler,
            event_dispatcher=event_dispatcher,
            application_identity_key=application_identity_key
        )
        if not session.start():
            session.cleanup()
            raise SessionCreateError('Failed to start session.')
        try:
            envuser = win32api.GetUserNameEx(win32con.NameSamCompatible)
            logger.info(f'Connected to Bloomberg BBComm as {envuser}')
        except (NameError, Exception):
            logger.info('Connected to Bloomberg BBComm')
        return session


class PostInitCaller(type):
    """Call __post_init__ after creation
    """
    def __call__(cls, *args, **kwargs):
        obj = type.__call__(cls, *args, **kwargs)
        obj.__post_init__()
        return obj


class Blp(metaclass=PostInitCaller):
    """Submits requests to the Bloomberg API and dispatches the events back to the request
    object for processing.

    Parameters
    ----------
    host : str, default 'localhost'
    port : int, default 8194
    auth : str, default 'AuthenticationType=OS_LOGON'
        Session authentication options.
    application_identity_key : str, optional
    session : blpapi.Session, optional
        A started session to use. Blp leaves a caller's session running
        when the connectivity test fails. None creates one.
    skip_test : bool, default False
        Skip the connectivity test request at construction.
    idle_seconds : float, default REQUEST_IDLE_SECONDS
        Longest wait for a request's next event. Each partial response
        restarts the count.
    total_seconds : float, optional
        Longest wait for a request's full response, counted from the send.
        Partial responses do not restart it. None sets no limit.
    """

    def __init__(self,
                 host='localhost',
                 port=8194,
                 auth='AuthenticationType=OS_LOGON',
                 application_identity_key=None,
                 session=None,
                 skip_test=False,
                 idle_seconds: float = REQUEST_IDLE_SECONDS,
                 total_seconds: float | None = None):
        self.host = host
        self.port = port
        self.auth = auth
        self.application_identity_key = application_identity_key
        self.skip_test = skip_test
        self.idle_seconds = idle_seconds
        self.total_seconds = total_seconds
        self._owns_session = session is None
        self.session = session or SessionFactory.create(
            host=host,
            port=port,
            auth=auth,
            application_identity_key=application_identity_key,
        )

    def __post_init__(self):
        if self.skip_test:
            return
        try:
            logger.info('Running session connectivity test...')
            resp = self.get_reference_data(flds=['ID_BB_GLOBAL'], tickers=['IBM US Equity'])
            assert resp.as_dict()
        except Exception as exc:
            # A caller's session crashes the process if used after
            # cleanup.
            if self._owns_session:
                self.session.cleanup()
            raise SessionError('Connectivity test failed.') from exc

    def __enter__(self):
        return self

    def __exit__(self, exc_ty, exc_val, tb):
        logger.debug('Exiting Blp context')
        if exc_ty:
            if isinstance(exc_val, SessionError):
                logger.debug(f'Blp context exited with session error: {exc_val}')
            else:
                logger.error(exc_val)
        self.session.cleanup()

    def __repr__(self):
        fmtargs = {
            'clz': self.__class__.__name__,
            'host': self.host,
            'port': self.port,
            'auth': self.auth,
        }
        return '<{clz}({host}:{port}:{auth})'.format(**fmtargs)

    def execute(self, request: BaseRequest) -> BaseResponse:
        """Send a Bloomberg API request and wait for its full response.

        Parameters
        ----------
        request : BaseRequest
            The request to send. It collects the response as it arrives.

        Returns
        -------
        BaseResponse
            The response the request built.

        Raises
        ------
        RequestTimeoutError
            Bloomberg sent nothing for the request for idle_seconds, where
            each partial response restarts the count, or the full response
            took longer than total_seconds after the send. The message
            names the limit and its value.
        SessionNotAvailableError
            The session is not started, or Bloomberg failed the request
            with a category in SESSION_FAILURE_CATEGORIES.
        SessionTerminatedError
            The session terminated while waiting.
        RuntimeError
            Bloomberg failed the request for any other reason, such as bad
            arguments. Unlike the errors above, it is not a SessionError.
        """
        logger.info(f'Sending request: {repr(request)}')
        try:
            self.session.open_service(request.service_name)
            service = self.session.getService(request.service_name)
            _request = request.create_request(service)
            correlation_id = self.session.sendRequest(_request)
        except blpapi.Exception as exc:
            if is_not_started(exc):
                raise SessionNotAvailableError(str(exc)) from exc
            raise
        request.prepare_response()
        return self._wait_for_response(request, correlation_id)

    def _wait_for_response(
        self,
        request: BaseRequest,
        correlation_id: blpapi.CorrelationId) -> BaseResponse:
        """Feed request the events for correlation_id until its RESPONSE.

        Parameters
        ----------
        request : BaseRequest
            The request that processes each response event.
        correlation_id : blpapi.CorrelationId
            The id sendRequest returned. Response and request status events
            for any other id belong to an abandoned request and are skipped.

        Returns
        -------
        BaseResponse
            The response the request built.

        Raises
        ------
        RequestTimeoutError
            The idle_seconds or total_seconds limit passed, as in execute.
        SessionNotAvailableError
        SessionTerminatedError
        RuntimeError
            As listed in execute.
        """
        sent_at = time.monotonic()
        idle_deadline = sent_at + self.idle_seconds
        total_deadline = None
        if self.total_seconds is not None:
            total_deadline = sent_at + self.total_seconds
        while True:
            now = time.monotonic()
            if now > idle_deadline:
                raise RequestTimeoutError(
                    f'No data for {request!r} within the idle limit'
                    f' of {self.idle_seconds} s')
            if total_deadline is not None and now > total_deadline:
                raise RequestTimeoutError(
                    f'No full response for {request!r} within the total limit'
                    f' of {self.total_seconds} s')
            # A short wait keeps Ctrl-C and the deadline check responsive.
            event = self.session.nextEvent(500)
            event_type = event.eventType()
            if (event_type in REQUEST_EVENT_TYPES
                and not any(correlation_id in msg.correlationIds() for msg in event)):
                logger.debug(f'Skipped event type {event_type} of an abandoned request')
                continue
            match event_type:
                case Event.RESPONSE:
                    logger.debug('Processing RESPONSE ...')
                    request.process_response(event, is_final=True)
                    logger.debug('END RESPONSE')
                    break
                case Event.PARTIAL_RESPONSE:
                    logger.debug('Processing PARTIAL_RESPONSE ...')
                    request.process_response(event, is_final=False)
                    idle_deadline = time.monotonic() + self.idle_seconds
                case Event.REQUEST_STATUS:
                    message = next(
                        msg for msg in event if correlation_id in msg.correlationIds())
                    category = ''
                    if message.hasElement('reason'):
                        reason = message.getElement('reason')
                        if reason.hasElement('category'):
                            category = reason.getElementAsString('category')
                    if category in SESSION_FAILURE_CATEGORIES:
                        raise SessionNotAvailableError(f'Request failed: {message}')
                    raise RuntimeError(f'Request failed: {message}')
                case Event.SESSION_STATUS:
                    request.on_admin_event(event)
        if request.has_exception:
            request.raise_exception()
        return request.response

    def get_historical(
        self,
        tickers,
        flds,
        start:datetime.datetime = None,
        end:datetime.datetime = None,
        timezone: str = LCL.name,
        period=None,
        raise_security_error=False,
        raise_field_error=False,
        **overrides,
    ) -> HistoricalDataResponse:
        """Equivalent of Excel BDH Request.

        Parameters
        ----------
        tickers :
        flds :
        start :
        end :
        period :
        raise_security_error :
        raise_field_error :

        Returns
        -------
        HistoricalDataResponse

        """
        req = HistoricalDataRequest(
            tickers,
            flds,
            start=start,
            end=end,
            timezone=timezone,
            period=period,
            raise_security_error=raise_security_error,
            raise_field_error=raise_field_error,
            **overrides,
        )
        return self.execute(req)

    def get_reference_data(
        self,
        tickers,
        flds,
        timezone: str = LCL.name,
        decimal_places: int = None,
        field_parse_custom: dict | None = None,
        raise_security_error=False,
        raise_field_error=False,
        force_string=False,
        time_as_datetime=False,
        **overrides
    ) -> ReferenceDataResponse:
        """Equivalent of Excel BDP Request.

        Parameters
        ----------
        tickers : str or list[str]
            Bloomberg security identifier(s).
        flds : str or list[str]
            Bloomberg field name(s).
        timezone : str
            Zone for datetime and time values in the response.
        decimal_places : int, optional
            Digits kept when force_string formats a number.
        field_parse_custom : dict, optional
            Field name to a function that takes the raw blpapi element.
        raise_security_error : bool, default False
            Raise RequestSecurityError for an invalid ticker.
        raise_field_error : bool, default False
            Raise RequestFieldError for an invalid field.
        force_string : bool, default False
            Return every value as a string, and a bulk field as JSON.
        time_as_datetime : bool, default False
            Return a time-only value as a datetime dated with this host's
            date.
        **overrides
            Bloomberg field overrides, field id to value. A keyword this
            method does not name goes to Bloomberg as an override.

        Returns
        -------
        ReferenceDataResponse
        """
        req = ReferenceDataRequest(
            tickers,
            flds,
            timezone=timezone,
            decimal_places=decimal_places,
            field_parse_custom=field_parse_custom,
            raise_security_error=raise_security_error,
            raise_field_error=raise_field_error,
            force_string=force_string,
            time_as_datetime=time_as_datetime,
            **overrides,
        )
        return self.execute(req)

    def get_intraday_tick(
        self,
        ticker,
        events=None,
        start:datetime.datetime = None,
        end:datetime.datetime = None,
        timezone: str = LCL.name,
        include_condition_codes=None,
        include_nonplottable_events=None,
        include_exchange_codes=None,
        return_eids=None,
        include_broker_codes=None,
        include_equity_ref_price=None,
        include_action_codes=None,
        include_indicator_codes=None,
        include_trade_date=None,
        include_trade_time=None,
        include_rsp_codes=None,
        include_bic_mic_codes=None,
        include_spread_price=None,
        include_yield=None,
        adjustment_normal=None,
        adjustment_abnormal=None,
        adjustment_split=None,
        adjustment_follow_DPDF=None,
        include_client_specific_fields=None,
        include_bloomberg_standard_condition_codes=None,
        include_trade_id=None,
        include_upfront_price=None,
        **overrides,
    ):
        if events is None:
            events = ['TRADE', 'ASK', 'BID']
        req = IntradayTickRequest(
            ticker,
            start=start,
            end=end,
            timezone=timezone,
            events=events,
            include_condition_codes=include_condition_codes,
            include_nonplottable_events=include_nonplottable_events,
            include_exchange_codes=include_exchange_codes,
            return_eids=return_eids,
            include_broker_codes=include_broker_codes,
            include_action_codes=include_action_codes,
            include_equity_ref_price=include_equity_ref_price,
            include_indicator_codes=include_indicator_codes,
            include_trade_time=include_trade_time,
            include_rsp_codes=include_rsp_codes,
            include_bic_mic_codes=include_bic_mic_codes,
            include_spread_price=include_spread_price,
            include_yield=include_yield,
            adjustment_normal=adjustment_normal,
            adjustment_abnormal=adjustment_abnormal,
            adjustment_split=adjustment_split,
            adjustment_follow_DPDF=adjustment_follow_DPDF,
            include_client_specific_fields=include_client_specific_fields,
            include_trade_id=include_trade_id,
            include_upfront_price=include_upfront_price,
            include_trade_date=include_trade_date,
            include_bloomberg_standard_condition_codes=include_bloomberg_standard_condition_codes,
            **overrides,
        )
        return self.execute(req)

    def get_intraday_bar(
        self,
        ticker,
        event='TRADE',
        start:datetime.datetime = None,
        end:datetime.datetime = None,
        timezone: str = LCL.name,
        interval=None,
        gap_fill_initial_bar=None,
        return_eids=None,
        adjustment_normal=None,
        adjustment_abnormal=None,
        adjustment_split=None,
        adjustment_follow_DPDF=None,
    ):
        req = IntradayBarRequest(
            ticker,
            start=start,
            end=end,
            timezone=timezone,
            event=event,
            interval=interval,
            gap_fill_initial_bar=gap_fill_initial_bar,
            return_eids=return_eids,
            adjustment_normal=adjustment_normal,
            adjustment_split=adjustment_split,
            adjustment_abnormal=adjustment_abnormal,
            adjustment_follow_DPDF=adjustment_follow_DPDF,
        )
        return self.execute(req)

    def get_screener(
        self,
        name,
        group='General',
        type='GLOBAL',
        asof=None,
        language=None,
        timezone: str = LCL.name,
    ):
        req = EQSRequest(
            name,
            type=type,
            group=group,
            asof=asof,
            language=language,
            timezone=timezone,
        )
        return self.execute(req)

    def bql(
        self,
        expression: str = None,
        universe: str = None,
        fields: str = None,
        raise_security_error=False,
        raise_field_error=False,
    ) -> BQLResponse:
        """Equivalent of Excel BQL() function.

        Parameters
        ----------
        expression : Complete BQL expression, e.g. "get(px_last) for(['AAPL US Equity'])"
        universe : Universe clause if expression not provided
        fields : Fields clause if expression not provided
        raise_security_error :
        raise_field_error :

        Returns
        -------
        BQLResponse

        """
        req = BQLRequest(
            expression=expression,
            universe=universe,
            fields=fields,
            raise_security_error=raise_security_error,
            raise_field_error=raise_field_error,
        )
        return self.execute(req)


@dataclass
class TopicState:
    """One topic's state, as Subscription.status() copies it.

    Attributes
    ----------
    topic : str
    status : str
        'pending' until Bloomberg answers the subscribe, then 'subscribed'
        on SubscriptionStarted, or 'failed' on SubscriptionFailure or on a
        SubscriptionTerminated that remove() did not cause.
    reason : str or None
        Bloomberg's description, for a failed topic.
    subscribed_at : DateTime or None
        UTC time of SubscriptionStarted.
    first_data_at : DateTime or None
        UTC time of the first tick, which is the initial snapshot.
    last_tick_at : DateTime or None
        UTC time of the latest tick.
    ticks_last_minute : int
        Ticks received less than TICK_WINDOW_SECONDS before the status()
        call.
    """

    topic: str
    status: str = 'pending'
    reason: str | None = None
    subscribed_at: DateTime | None = None
    first_data_at: DateTime | None = None
    last_tick_at: DateTime | None = None
    ticks_last_minute: int = 0


@dataclass
class SubscriptionStatus:
    """A consistent copy of a Subscription's state.

    Attributes
    ----------
    failure : str or None
        Why the subscription can no longer deliver data, in this order: the
        session terminated outside stop(), every held topic failed, or no
        topic ticked within FIRST_DATA_SECONDS of the first add(). None
        while it can. The owner replaces a failed subscription.
    connected : bool
        False between SessionConnectionDown and SessionConnectionUp, while
        blpapi tries to reconnect.
    error_cnt : int
        Exceptions caught on the dispatcher thread, from parsing or on_tick.
    last_error : str or None
        'topic: repr(exception)' of the latest one.
    topics : dict[str, TopicState]
        Held topics, in add order.
    """

    failure: str | None
    connected: bool
    error_cnt: int
    last_error: str | None
    topics: dict[str, TopicState]


class Subscription:
    """Live market data subscription on its own asynchronous session.

    Parameters
    ----------
    fields : str or list[str]
        Fields whose changes raise a tick, such as ['BID', 'ASK']. A row
        holds the ones present in the message.
    on_tick : Callable[[str, dict[str, Any]], None]
        Called with (topic, {field: value}) for each tick, on the
        dispatcher thread, with no lock held, so it may call add, remove,
        status or stop. Subscription catches and counts what it raises.
    interval : float, default 0
        Seconds between conflated updates. 0 leaves it unset.
    host, port, auth, application_identity_key
        Session settings, passed to SessionFactory.create.
    assumed_timezone : Timezone, default LCL
        Zone of the times Bloomberg sends, which is the Terminal's zone.
    desired_timezone : Timezone, default LCL
        Zone of the times in a row.
    time_as_datetime : bool, default False
        Return a time-only value as a datetime.

    Raises
    ------
    SessionCreateError
        The session failed to start.
    SessionStartError
        //blp/mktdata failed to open. The session is cleaned up first.

    Notes
    -----
    - Every method is safe from any thread. stop() is the only way to
      release the session, and an instance left unstopped lives until the
      process exits.
    """

    def __init__(
        self,
        fields: str | list[str],
        on_tick: Callable[[str, dict[str, Any]], None],
        interval: float = 0,
        host: str = 'localhost',
        port: int = 8194,
        auth: str = 'AuthenticationType=OS_LOGON',
        application_identity_key: str | None = None,
        assumed_timezone: Timezone = LCL,
        desired_timezone: Timezone = LCL,
        time_as_datetime: bool = False) -> None:
        self.fields = [fields] if isinstance(fields, str) else list(fields)
        self.on_tick = on_tick
        self.interval = interval
        self.parser = SubscriptionParser(
            assumed_timezone=assumed_timezone,
            desired_timezone=desired_timezone,
            time_as_datetime=time_as_datetime)
        self._lock = threading.Lock()
        self._terminated = threading.Event()
        self._cids = itertools.count(1)
        self._states: dict[str, TopicState] = {}
        self._tick_times: dict[str, deque[float]] = {}
        self._cid_by_topic: dict[str, int] = {}
        self._topic_by_cid: dict[int, str] = {}
        self._first_add_at: float | None = None
        self._first_data_at: float | None = None
        self._session_failure: str | None = None
        self._connected = True
        self._error_cnt = 0
        self._last_error: str | None = None
        self._stopping = False
        self._cleaned = False
        self._dispatch_thread_id: int | None = None
        self.session = SessionFactory.create(
            host=host,
            port=port,
            auth=auth,
            application_identity_key=application_identity_key,
            event_handler=self._on_event)
        try:
            self.session.open_service('//blp/mktdata')
        except Exception:
            self.session.cleanup()
            raise

    def __enter__(self) -> 'Subscription':  # noqa: PYI034
        return self

    def __exit__(self, exc_ty: type | None, exc_val: BaseException | None, tb: Any) -> None:
        self.stop()

    def add(self, topics: str | list[str]) -> None:
        """Subscribe topics on the live session.

        Parameters
        ----------
        topics : str or list[str]
            A held topic is skipped, except a failed one, which is
            subscribed again.

        Raises
        ------
        SessionError
            stop() has begun.
        SessionNotAvailableError
            The session is not started. No topic of the call is held.
        """
        topics = [topics] if isinstance(topics, str) else topics
        options = {'interval': f'{self.interval:.1f}'} if self.interval else {}
        with self._lock:
            if self._stopping:
                raise SessionError('Subscription is stopped')
            new_topics = [
                topic for topic in dict.fromkeys(topics)
                if topic not in self._states or self._states[topic].status == 'failed'
                ]
            if not new_topics:
                return
            prior_by_topic = {
                topic: (self._cid_by_topic[topic], self._states[topic], self._tick_times[topic])
                for topic in new_topics if topic in self._states
                }
            subscriptions = blpapi.SubscriptionList()
            for topic in new_topics:
                self._topic_by_cid.pop(self._cid_by_topic.get(topic), None)
                cid = next(self._cids)
                self._cid_by_topic[topic] = cid
                self._topic_by_cid[cid] = topic
                self._states[topic] = TopicState(topic)
                self._tick_times[topic] = deque()
                subscriptions.add(topic, self.fields, options, blpapi.CorrelationId(cid))
            try:
                self.session.subscribe(subscriptions)
            except blpapi.Exception as exc:
                for topic in new_topics:
                    del self._topic_by_cid[self._cid_by_topic.pop(topic)]
                    del self._states[topic]
                    del self._tick_times[topic]
                for topic, (cid, state, ticks) in prior_by_topic.items():
                    self._cid_by_topic[topic] = cid
                    self._topic_by_cid[cid] = topic
                    self._states[topic] = state
                    self._tick_times[topic] = ticks
                if is_not_started(exc):
                    raise SessionNotAvailableError(str(exc)) from exc
                raise
            if self._first_add_at is None:
                self._first_add_at = time.monotonic()
        logger.info(f'Subscribed {len(new_topics)} topics')

    def remove(self, topics: str | list[str]) -> None:
        """Unsubscribe topics and stop tracking them. Unknown topics are
        skipped.

        Parameters
        ----------
        topics : str or list[str]

        Raises
        ------
        SessionNotAvailableError
            The session is not started. The topics are dropped all the same.
        """
        topics = [topics] if isinstance(topics, str) else topics
        with self._lock:
            held_topics = [topic for topic in dict.fromkeys(topics) if topic in self._states]
            if not held_topics:
                return
            subscriptions = blpapi.SubscriptionList()
            for topic in held_topics:
                cid = self._cid_by_topic.pop(topic)
                # Dropping the mapping first makes the unsubscribe's own
                # SubscriptionTerminated, and any late tick, unknown.
                self._topic_by_cid.pop(cid, None)
                del self._states[topic]
                del self._tick_times[topic]
                subscriptions.add(None, correlationId=blpapi.CorrelationId(cid))
            if self._stopping:
                return
            try:
                self.session.unsubscribe(subscriptions)
            except blpapi.Exception as exc:
                if is_not_started(exc):
                    raise SessionNotAvailableError(str(exc)) from exc
                raise
        logger.info(f'Unsubscribed {len(held_topics)} topics')

    def status(self) -> SubscriptionStatus:
        """A consistent copy of the subscription's state, as of now.
        """
        now = time.monotonic()
        with self._lock:
            topics = {}
            for topic, state in self._states.items():
                ticks = self._tick_times[topic]
                while ticks and now - ticks[0] >= TICK_WINDOW_SECONDS:
                    ticks.popleft()
                topics[topic] = dataclasses.replace(state, ticks_last_minute=len(ticks))
            failure = self._session_failure
            failed_ds = [state for state in topics.values() if state.status == 'failed']
            if not failure and not self._stopping and topics and len(failed_ds) == len(topics):
                failure = f'all {len(failed_ds)} topics failed: {failed_ds[0].reason}'
            if not failure and not self._stopping and self._first_add_at is not None:
                data_at = now if self._first_data_at is None else self._first_data_at
                if data_at - self._first_add_at >= FIRST_DATA_SECONDS:
                    failure = f'no subscription data within {FIRST_DATA_SECONDS} s'
            return SubscriptionStatus(
                failure=failure,
                connected=self._connected,
                error_cnt=self._error_cnt,
                last_error=self._last_error,
                topics=topics)

    def stop(self, timeout: float = 10.0) -> None:
        """Unsubscribe everything and release the session.

        Parameters
        ----------
        timeout : float, default 10.0
            Longest wait, in seconds, for the session to terminate.

        Notes
        -----
        - Called from on_tick, it begins the stop and returns at once. A
          later call from another thread, or after a timeout, waits again
          and releases the session. status() still reads the last state.
        """
        with self._lock:
            if not self._stopping:
                self._stopping = True
                subscriptions = blpapi.SubscriptionList()
                for cid in self._topic_by_cid:
                    subscriptions.add(None, correlationId=blpapi.CorrelationId(cid))
                self._topic_by_cid.clear()
                try:
                    if subscriptions.size():
                        self.session.unsubscribe(subscriptions)
                except blpapi.Exception as exc:
                    logger.warning(f'Could not unsubscribe cleanly: {exc}')
                try:
                    self.session.stopAsync()
                except blpapi.Exception as exc:
                    logger.warning(f'Could not stop session: {exc}')
        if threading.get_ident() == self._dispatch_thread_id:
            return
        if not self._terminated.wait(timeout):
            logger.warning(f'Bloomberg session did not stop within {timeout} s')
            # Session.cleanup joins the dispatcher thread with no limit,
            # which would hang interpreter exit.
            atexit.unregister(self.session.cleanup)
            return
        with self._lock:
            if self._cleaned:
                return
            self._cleaned = True
        self.session.cleanup()

    def _on_event(self, event: blpapi.Event, session: blpapi.Session) -> None:
        """Blpapi's event handler, on the dispatcher thread. Never raises.
        """
        # blpapi ends the process with os._exit(1) when a handler raises,
        # so this guard catches BaseException.
        try:
            self._dispatch_thread_id = threading.get_ident()
            match event.eventType():
                case Event.SUBSCRIPTION_DATA:
                    self._on_data_event(event)
                case Event.SUBSCRIPTION_STATUS:
                    self._on_status_event(event)
                case Event.TIMEOUT:
                    return
                case _:
                    self._on_other_event(event)
        except BaseException as exc:
            self._record_error('', exc)

    def _on_data_event(self, event: blpapi.Event) -> None:
        """Record each message's tick, then pass its row to on_tick.
        """
        for message in event:
            now_mono, now = time.monotonic(), DateTime.now(UTC)
            with self._lock:
                topic = self._topic_by_cid.get(message.correlationId().value())
                if topic is None:
                    continue
                state = self._states[topic]
                state.first_data_at = state.first_data_at or now
                state.last_tick_at = now
                ticks = self._tick_times[topic]
                ticks.append(now_mono)
                while now_mono - ticks[0] >= TICK_WINDOW_SECONDS:
                    ticks.popleft()
                if self._first_data_at is None:
                    self._first_data_at = now_mono
            try:
                row = {
                    field: self.parser.get_subelement_value(message, field.upper())
                    for field in self.fields
                    if field.upper() in message
                    }
                self.on_tick(topic, row)
            except BaseException as exc:
                self._record_error(topic, exc)

    def _on_status_event(self, event: blpapi.Event) -> None:
        """Mark topics subscribed or failed.
        """
        for message in event:
            match message.messageType():
                case blpapi.Names.SUBSCRIPTION_STARTED:
                    started = True
                case blpapi.Names.SUBSCRIPTION_FAILURE | blpapi.Names.SUBSCRIPTION_TERMINATED:
                    started = False
                case _:
                    continue
            desc = None
            if not started:
                desc = 'no description'
                reason = message.getElement('reason')
                if reason.hasElement('description'):
                    desc = reason.getElementAsString('description')
            with self._lock:
                topic = self._topic_by_cid.get(message.correlationId().value())
                if topic is None:
                    continue
                state = self._states[topic]
                if started:
                    state.status = 'subscribed'
                    state.subscribed_at = DateTime.now(UTC)
                else:
                    state.status = 'failed'
                    state.reason = desc
            if not started:
                logger.error(f'Subscription {message.messageType()} topic={topic} desc={desc}')

    def _on_other_event(self, event: blpapi.Event) -> None:
        """Track session termination and connection, and log queue warnings.
        """
        for message in event:
            match message.messageType():
                case blpapi.Names.SESSION_TERMINATED:
                    with self._lock:
                        if not self._stopping:
                            self._session_failure = 'session terminated'
                    # INFO, since every clean stop() delivers one.
                    logger.info('Bloomberg session terminated')
                    # No callback follows SessionTerminated, and the
                    # dispatcher thread's id may be reused once it exits.
                    self._dispatch_thread_id = None
                    self._terminated.set()
                case blpapi.Names.SESSION_CONNECTION_DOWN:
                    with self._lock:
                        self._connected = False
                    logger.warning('Bloomberg connection down')
                case blpapi.Names.SESSION_CONNECTION_UP:
                    with self._lock:
                        self._connected = True
                    logger.info('Bloomberg connection up')
                case blpapi.Names.SLOW_CONSUMER_WARNING:
                    logger.warning('Event queue nearly full; ticks may be dropped')
                case blpapi.Names.SLOW_CONSUMER_WARNING_CLEARED:
                    logger.warning('Event queue no longer nearly full')
                case blpapi.Names.DATA_LOSS:
                    logger.warning(f'Event queue overflowed, data lost: {message}')

    def _record_error(self, topic: str, exc: BaseException) -> None:
        """Count exc and keep it as last_error. Only the first is logged
        with its traceback, since a broken on_tick fails on every tick.
        """
        with self._lock:
            self._error_cnt += 1
            self._last_error = f'{topic}: {exc!r}'
            first = self._error_cnt == 1
        if first:
            logger.error(f'Subscription error on {topic!r}', exc_info=exc)
        else:
            logger.debug(f'Subscription error on {topic!r}: {exc!r}')
