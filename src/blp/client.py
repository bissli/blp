"""TODO: Standardize: Response always stores dataframes (cleaned for values_to_string). to_dictionary returns DF to dictionary.
See HistoricalDataResponse
"""
import atexit
import contextlib
import datetime
import logging
from abc import ABC, abstractmethod
from collections import defaultdict

import blpapi
import pandas as pd
from blpapi.event import Event

from blp.handle import BaseEventHandler, DefaultEventHandler
from blp.parse import Name, Parser
from date import LCL, UTC, DateTime, Timezone
from libb import NonBlockingDelay, is_null

try:
    import win32api
    import win32con
except ModuleNotFoundError:
    pass

logger = logging.getLogger(__name__)

__all__ = ['Blp']


def create_daterange(beg: datetime.datetime, end: datetime.datetime):
    """Create UTC dates for querying range requests.
    """
    end = DateTime.now(UTC) \
        if is_null(end) \
        else DateTime.parse(end).in_timezone(UTC)
    beg = end.subtract(days=1) \
        if is_null(beg) \
        else DateTime.parse(beg).in_timezone(UTC)
    return beg, end


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
    def prepare_response(self):
        pass

    @property
    def has_exception(self):
        return (self.raise_security_error and self.security_errors) \
            or (self.raise_field_error and self.field_errors)

    def raise_exception(self):
        if self.security_errors:
            msgs = ''.join([str(s) for s in self.security_errors])
            if self.raise_security_error:
                raise Exception(msgs)
            logger.debug(f'Security Errors:\n{msgs}')
        if self.field_errors:
            msgs = ''.join([str(s) for s in self.field_errors])
            if self.raise_field_error:
                raise Exception(msgs)
            logger.debug(f'Field Errors:\n{msgs}')

    @abstractmethod
    def create_request(self, service):
        pass

    @abstractmethod
    def process_response(self, event, is_final):
        pass

    def on_admin_event(self, event):
        for message in event:
            if message.messageType() == Name.SESSION_CONNECTION_UP:
                logger.info('Connected ...')
            elif Name.SESSION_STARTED in message:
                logger.info('Started new Bbg session ...')
            elif 'ServiceOpened' in message:
                logger.info('Opened Bbg service ...')
            elif message.messageType() == 'SessionTerminated':
                logger.info('Session DONE')
                raise RuntimeError('Session DONE')

    @staticmethod
    def apply_overrides(request, overrides):
        if overrides:
            for k, v in overrides.items():
                o = request.getElement('overrides').appendElement()
                o.setElement(Name.FIELD_ID, k)
                o.setElement('value', v)

    def set_flag(self, request, val, fld):
        """If the specified val is not None, then set the specified field to its boolean value"""
        if val is not None:
            val = bool(val)
            request.set(fld, val)

    def set_response(self, response):
        """Set the response to handle and store the results"""
        self.response = response


class BaseResponse(ABC):
    """Base class for Responses"""

    @abstractmethod
    def as_dataframe(self):
        pass

    @abstractmethod
    def as_dict(self):
        pass


class HistoricalDataResponse(BaseResponse):

    def __init__(self, request):
        self.request = request
        self.response_map = {}

    def on_security_complete(self, sid, df):
        self.response_map[sid] = df.astype(object).where(df.notna(), None) \
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
    """"
    Manages the creation and processing of Bloomberg HistoricalDataRequest.

    Parameters
    ----------
    sids: Bloomberg security identifier(s).
    fields: Bloomberg field name(s).
    start: Optional. Date or date string. Defaults to 1 year ago if None.
    end: Optional. Date or date string. Defaults to today if None.
    period: Optional. Periodicity of data (DAILY, WEEKLY, MONTHLY, QUARTERLY,
      SEMI_ANNUALLY, YEARLY).
    raise_security_error: If True, raises exceptions for invalid sids.
    raise_field_error: If True, raises exceptions for invalid fields.
    period_adjustment: Frequency and calendar type of the output (ACTUAL,
      CALENDAR, FISCAL).
    currency: ISO Code. Converts value from local to specified currency.
    override_option: OVERRIDE_OPTION_CLOSE or OVERRIDE_OPTION_GPA.
    pricing_option: PRICING_OPTION_PRICE or PRICING_OPTION_YIELD.
    non_trading_day_fill_option: NON_TRADING_WEEKDAYS, ALL_CALENDAR_DAYS, or
      ACTIVE_DAYS_ONLY.
    non_trading_day_fill_method: PREVIOUS_VALUE or NIL_VALUE.
    calendar_code_override: 2-letter country ISO code.
    """

    def __init__(
        self,
        sids,
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
        self.is_single_sid = is_single_sid = isinstance(sids, str)
        self.is_single_field = is_single_field = isinstance(fields, str)
        self.sids = [sids] if is_single_sid else list(sids)
        self.fields = [fields] if is_single_field else list(fields)
        self.start, self.end = create_daterange(start, end)
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
            'symbols': ','.join(self.sids),
            'fields': ','.join(self.fields),
            'start': self.start.in_timezone(self.timezone).strftime('%Y-%m-%d'),
            'end': self.end.in_timezone(self.timezone).strftime('%Y-%m-%d'),
            'period': self.period,
        }
        # TODO: add self.overrides if defined
        return '<{clz}([{symbols}], [{fields}], start={start}, end={end}, period={period}'.format(**fmtargs)

    def prepare_response(self):
        self.response = HistoricalDataResponse(self)

    def create_request(self, service):
        request = service.createRequest('HistoricalDataRequest')
        [request.append('securities', sec) for sec in self.sids]
        [request.append('fields', fld) for fld in self.fields]
        request.set('startDate', self.start)
        request.set('endDate', self.end)
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

    def on_security_data_element(self, element):
        """Process a securityData element - FIXME: currently not handling relateDate element"""
        sid = self.parser.get_subelement_value(element, Name.SECURITY, self.force_string)
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
        self.response.on_security_complete(sid, df)

    def process_response(self, event, is_final):
        for message in self.parser.message_iter(event):
            # single security element in historical request
            element = message.getElement(Name.SECURITY_DATA)
            if Name.SECURITY_ERROR in element:
                sid = self.parser.get_subelement_value(element, Name.SECURITY, self.force_string)
                self.security_errors.append(self.parser.as_security_error(element.getElement(Name.SECURITY_ERROR), sid))
            else:
                self.on_security_data_element(element)


class ReferenceDataResponse(BaseResponse):

    def __init__(self, request):
        self.request = request
        self.response_map = defaultdict(dict)

    def on_security_data(self, sid, fieldmap):
        self.response_map[sid].update(fieldmap)

    def as_dict(self):
        return self.response_map

    def as_dataframe(self):
        """:return: Multi-Index DataFrame"""
        data = {sid: pd.Series(data) for sid, data in self.response_map.items()}
        df = pd.DataFrame.from_dict(data, orient='index')
        df = df.reindex(self.request.fields, axis=1)  # layer in any missing fields just in case
        if self.request.force_string:
            df = df.astype(object).where(df.notna(), None)
        return df


class ReferenceDataRequest(BaseRequest):

    def __init__(
        self,
        sids,
        fields,
        raise_security_error=False,
        raise_field_error=False,
        return_formatted_value=None,
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
        self.is_single_sid = is_single_sid = isinstance(sids, str)
        self.is_single_field = is_single_field = isinstance(fields, str)
        self.sids = [sids] if isinstance(sids, str) else sids
        self.fields = [fields] if isinstance(fields, str) else fields
        self.return_formatted_value = return_formatted_value
        self.timezone = Timezone(timezone)
        self.parser = Parser(
            assumed_timezone=UTC,
            desired_timezone=self.timezone,
            time_as_datetime=time_as_datetime,
        )
        self.overrides = overrides

    def __repr__(self):
        fmtargs = {
            'clz': self.__class__.__name__,
            'sids': ','.join(self.sids),
            'fields': ','.join(self.fields),
            'overrides': ','.join([f'{k}={v}' for k, v in self.overrides.items()]),
        }
        return '<{clz}([{sids}], [{fields}], overrides={overrides})'.format(**fmtargs)

    def prepare_response(self):
        self.response = ReferenceDataResponse(self)

    def create_request(self, service):
        request = service.createRequest('ReferenceDataRequest')
        [request.append('securities', sec) for sec in self.sids]
        [request.append('fields', fld) for fld in self.fields]
        self.set_flag(request, self.return_formatted_value, 'returnFormattedValue')
        self.set_flag(request, True, 'useUTCTime')
        self.apply_overrides(request, self.overrides)
        return request

    def process_response(self, event, is_final):
        for msg in self.parser.message_iter(event):
            for element, err in self.parser.security_element_iter(msg.getElement(Name.SECURITY_DATA)):
                if err:
                    self.security_errors.append(err)
                    continue
                self._process_security_element(element)

    def _process_security_element(self, element):
        sid = self.parser.get_subelement_value(element, Name.SECURITY, self.force_string)
        fields = element.getElement(Name.FIELD_DATA)
        field_data = self.parser.get_subelement_values(fields, self.fields, self.force_string)
        field_data = [x.in_timezone(self.timezone) if isinstance(x, datetime.datetime) else x for x in field_data]
        assert len(field_data) == len(self.fields), 'Field length must match data length'
        self.response.on_security_data(sid, dict(list(zip(self.fields, field_data))))
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
    """Intraday tick request. Can submit to MSG1 as well for bond runs.
    Note: returns UTC datetime by default.
    """
    def __init__(
        self,
        sid,
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
    ):
        super().__init__('//blp/refdata')
        self.sid = sid
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
        self.start, self.end = create_daterange(start, end)
        self.timezone = Timezone(timezone)
        self.parser = Parser(UTC, self.timezone)

    def __repr__(self):
        fmtargs = {'clz': self.__class__.__name__, 'sid': self.sid, 'events': ','.join(self.events)}
        return '<{clz}({sid}, [{events}])'.format(**fmtargs)

    def prepare_response(self):
        self.response = IntradayTickResponse(self)

    def create_request(self, service):
        request = service.createRequest('IntradayTickRequest')
        request.set(Name.SECURITY, self.sid)
        [request.append('eventTypes', event) for event in self.events]
        request.set('startDateTime', self.start)
        request.set('endDateTime', self.end)
        self.set_flag(request, self.include_condition_codes, 'includeConditionCodes')
        self.set_flag(request, self.include_nonplottable_events, 'includeNonPlottableEvents')
        self.set_flag(request, self.include_exchange_codes, 'includeExchangeCodes')
        self.set_flag(request, self.return_eids, 'returnEids')
        self.set_flag(request, self.include_broker_codes, 'includeBrokerCodes')
        self.set_flag(request, self.include_equity_ref_price, 'includeEqRefPrice')
        self.set_flag(request, self.include_rsp_codes, 'includeRpsCodes')
        self.set_flag(request, self.include_bic_mic_codes, 'includeBicMicCodes')
        self.set_flag(request, self.include_action_codes, 'includeActionCodes')
        self.set_flag(request, self.include_indicator_codes, 'includeIndicatorCodes')
        self.set_flag(request, self.include_trade_time, 'includeTradeTime')

        return request

    def on_tick_data(self, ticks):
        """Process the incoming tick data array"""
        for tick in self.parser.element_iter(ticks):
            names = [str(tick.getElement(_).name()) for _ in range(tick.numElements())]
            tickmap = {n: self.parser.get_subelement_value(tick, n) for n in names}
            self.response.ticks.append(tickmap)

    def process_response(self, event, is_final):
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
        sid,
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
        self.sid = sid
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
            'sid': self.sid,
            'event': self.event,
            'start': self.start.in_timezone(self.timezone),
            'end': self.end.in_timezone(self.timezone),
        }
        return '<{clz}({sid}, {event}, start={start}, end={end})'.format(**fmtargs)

    def prepare_response(self):
        self.response = IntradayBarResponse(self)

    def create_request(self, service):
        request = service.createRequest('IntradayBarRequest')
        request.set(Name.SECURITY, self.sid)
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

    def on_bar_data(self, bars):
        """Process the incoming tick data array"""
        for tick in self.parser.element_iter(bars):
            names = [str(tick.getElement(_).name()) for _ in range(tick.numElements())]
            barmap = {n: self.parser.get_subelement_value(tick, n) for n in names}
            self.response.bars.append(barmap)

    def process_response(self, event, is_final):
        for msg in self.parser.message_iter(event):
            data = msg.getElement('barData')
            # tickData will have 0 to 1 tickData[] elements
            if 'barTickData' in data:
                self.on_bar_data(data.getElement('barTickData'))


class EQSResponse(BaseResponse):

    def __init__(self, request):
        self.request = request
        self.response_map = defaultdict(dict)

    def on_security_data(self, sid, fieldmap):
        self.response_map[sid].update(fieldmap)

    def as_dict(self):
        return self.response_map

    def as_dataframe(self):
        """:return: Multi-Index DataFrame"""
        data = {sid: pd.Series(data) for sid, data in self.response_map.items()}
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

    def process_response(self, event, is_final):
        for message in self.parser.message_iter(event):
            data = message.getElement('data')
            security = data.getElement(Name.SECURITY_DATA)
            for element, error in self.parser.security_iter(security):
                if error:
                    self.security_errors.append(error)
                    continue
                self._process_security_element(element)

    def _process_security_element(self, element):
        sid = self.parser.get_subelement_value(element, Name.SECURITY, self.force_string)
        fields = element.getElement(Name.FIELD_DATA)
        fldnames = [str(field.name()) for field in fields.elements()]
        fdata = self.parser.get_subelement_values(fields, fldnames)
        self.response.on_security_data(sid, dict(list(zip(fldnames, fdata))))
        ferrors = self.parser.get_field_errors(element)
        ferrors and self.field_errors.extend(ferrors)


class Session(blpapi.Session):
    """Wrapper around blpapi.Session with auto-closing"""

    def __init__(self, *args):
        super().__init__(*args)
        atexit.register(self.cleanup)

    def open_service(self, service_name):
        """Open service. Raise Exception if fails."""
        if not self.openService(service_name):
            raise RuntimeError(f'Failed to open service {service_name}.')

    def cleanup(self):
        with contextlib.suppress(Exception):
            self.stop()
            self.destroy()
            logger.debug('Closed active Bloomberg session')


def create_session(
    host='localhost',
    port=8194,
    auth='AuthenticationType=OS_LOGON',
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
    return Session(session_options, event_handler, event_dispatcher)


class SessionFactory:
    """Session Factory"""

    @staticmethod
    def create(
        host='localhost',
        port=8194,
        auth='AuthenticationType=OS_LOGON',
        event_handler=None,
        event_dispatcher=None,
    ) -> Session:
        """Create Bloomberg API Session.

        Parameters
        ----------
        host : Not tested for BPIPE.
        port : Not tested for BPIPE.
        auth : Not tested for BPIPE.
        event_handler : Needed only for subscriptions.
        event_dispatcher : Needed only for subscriptions.

        Returns
        -------
        blpapi.Session object.

        Raises
        ------
        Exception : Exception on fail to start session.

        """
        logger.info('Connecting to Bloomberg BBComm session...')
        session = create_session(host, port, auth, event_handler, event_dispatcher)
        if not session.start():
            raise RuntimeError('Failed to start session.')
        envuser = win32api.GetUserNameEx(win32con.NameSamCompatible)
        logger.info(f'Connected to Bloomberg BBComm as {envuser}')
        return session


class Blp:
    """Submits requests to the Bloomberg API and dispatches the events back to the request
    object for processing.
    """

    def __init__(self, host='localhost', port=8194, auth='AuthenticationType=OS_LOGON', session=None):
        self.host = host
        self.port = port
        self.auth = auth
        self.session = session or SessionFactory.create(host, port, auth)

    def __enter__(self):
        return self

    def __exit__(self, exc_ty, exc_val, tb):
        logger.debug('Exiting Blp context')
        if exc_ty:
            logger.exception(exc_val)
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
        logger.info(f'Sending request: {repr(request)}')
        self.session.open_service(request.service_name)
        service = self.session.getService(request.service_name)
        _request = request.create_request(service)
        cid = self.session.sendRequest(_request)
        request.prepare_response()
        return self._wait_for_response(request)

    def _wait_for_response(self, request: BaseRequest) -> BaseResponse:
        """Waits for response after sending the request.

        Success response can come with a number of
        PARTIAL_RESPONSE events followed by a RESPONSE event.
        Failures will be delivered in a REQUEST_STATUS event
        holding a REQUEST_FAILURE message.
        """
        while 1:
            event = self.session.nextEvent(500)  # timeout to gtive the chance to ctrl+c handling
            match event.eventType():
                case Event.RESPONSE:
                    logger.debug('Processing RESPONSE ...')
                    request.process_response(event, is_final=True)
                    logger.debug('END RESPONSE')
                    break
                case Event.PARTIAL_RESPONSE:
                    logger.debug('Processing PARTIAL_RESPONSE ...')
                    request.process_response(event, is_final=False)
                case Event.SESSION_STATUS:
                    try:
                        request.on_admin_event(event)
                    except:
                        break
                case _:
                    pass
        request.has_exception and request.raise_exception()
        return request.response

    def get_historical(
        self,
        sids,
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
        sids :
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
            sids,
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
        sids,
        flds,
        timezone: str = LCL.name,
        raise_security_error=False,
        raise_field_error=False,
        **overrides
    ) -> ReferenceDataResponse:
        """Equivalent of Excel BDP Request.

        Parameters
        ----------
        sids :
        flds :
        raise_security_error :
        raise_field_error :

        Returns
        -------
        ReferenceDataResponse

        """
        req = ReferenceDataRequest(
            sids,
            flds,
            timezone=timezone,
            raise_security_error=raise_security_error,
            raise_field_error=raise_field_error,
            **overrides,
        )
        return self.execute(req)

    def get_intraday_tick(
        self,
        sid,
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
        include_trade_time=None,
        include_rsp_codes=None,
        include_bic_mic_codes=None,
        **overrides,
    ):
        if events is None:
            events = ['TRADE', 'ASK', 'BID']
        req = IntradayTickRequest(
            sid,
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
            **overrides,
        )
        return self.execute(req)

    def get_intraday_bar(
        self,
        sid,
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
            sid,
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

    def subscribe(
        self,
        topics,
        fields,
        interval=0,
        host='localhost',
        port=8194,
        auth='AuthenticationType=OS_LOGON',
        dispatcher=None,
        runtime=24*60*60,
        handler=DefaultEventHandler,
        **kwargs
    ):
        """Create subscription request"""
        sub = Subscription(
            topics=topics,
            fields=fields,
            interval=interval,
            host=host,
            port=port,
            auth=auth,
            dispatcher=dispatcher,
        )
        sub.subscribe(runtime=runtime, handler=handler, **kwargs)


class Subscription:
    """
    Fields: Bloomberg will always return all subscribable fields in reach response.
        What `fields` does is specify which fields (when they change) should trigger an event.
        Internally though our event handler will filter for the fields we request.
    """

    def __init__(
        self,
        topics,
        fields,
        interval=0,
        host='localhost',
        port=8194,
        auth='AuthenticationType=OS_LOGON',
        dispatcher=None,
    ):
        """

        Parameters
        ----------
        topics :  (Sids, but also custom Bloomberg mnemonics IE @MSG1). IE ['IBM US Equity', 'TSLA US Equity' ]
        fields :  IE `['BID', 'ASK', 'TRADE']`
        interval :  Time in seconds to intervalize the subscriptions
        host : For session creation. See SessionFactory.
        port : For session creation. See SessionFactory.
        auth : For session creation. See SessionFactory.
        dispatcher :  if None, will create a defualt Dispatcher (single thread)

        """
        self.fields = [fields] if isinstance(fields, str) else fields
        self.topics = [topics] if isinstance(topics, str) else topics
        self.interval = interval
        self.host = host
        self.port = port
        self.auth = auth
        self.dispatcher = dispatcher

    def subscribe(self, handler: BaseEventHandler, runtime=24 * 60 * 60, **kwargs):
        r"""Subscribe with a given handler

        Parameters
        ----------
        handler : Async handler instance of BaseHandler. Expects class, not object instance
        runtime : Duration to run service in seconds. Generally best to run between 0 and time to market close..

        Form of created subscription string (subscriptions.add):

        "//blp/mktdata/ticker/IBM US Equity?fields=BID,ASK&interval=2"
        \\-----------/\\------/\\-----------/\\------------------------/
        |          |         |                  |
        Service    Prefix   Instrument           Suffix

        All subscription date/times results are in the default timezone of the terminal.

        """
        _handler = handler(self.topics, self.fields, **kwargs)
        session = SessionFactory.create(self.host, self.port, self.auth, _handler, self.dispatcher)
        session.open_service('//blp/mktdata')

        subscriptions = blpapi.SubscriptionList()
        options = {'interval': f'{self.interval:.1f}'} if self.interval else {}
        for topic in self.topics:
            subscriptions.add(topic, self.fields, options, blpapi.CorrelationId(topic))
        try:
            logger.info('Starting subscription...')
            session.subscribe(subscriptions)
            delay = NonBlockingDelay()
            delay.delay(runtime)
            while not delay.timeout():
                continue
        finally:
            logger.info('Ending subscription...')
            session.unsubscribe(subscriptions)
