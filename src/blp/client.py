"""TODO: Standardize: Response always stores dataframes (cleaned for values_to_string). to_dictionary returns DF to dictionary.
See HistoricalDataResponse
"""
import atexit
import datetime
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable

import blpapi
import numpy as np
import pandas as pd
import pytz
import win32api
import win32con
from blp.handle import BaseEventHandler
from blp.parse import Name, Parser
from blpapi.event import Event
from dateutil.relativedelta import relativedelta

import libb

logger = logging.getLogger(__name__)

__all__ = ['Blp']


def empty(obj):
    if isinstance(obj, Iterable):
        try:
            return np.all(obj.isna())
        except:
            return bool(obj)
    return bool(obj)


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
    """Base class for Responses
    """
    @abstractmethod
    def as_dataframe(self):
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
        sids, dfs = list(self.response_map.keys()), list(self.response_map.values())
        df = pd.concat(dfs, keys=sids, axis=1)
        return df


class HistoricalDataRequest(BaseRequest):
    """A class which manages the creation of the Bloomberg
    HistoricalDataRequest and the processing of the associated Response.

    Parameters
    ----------
    sids: bbg security identifier(s)
    fields: bbg field name(s)
    start: (optional) date, date string , or None. If None, defaults to 1 year ago.
    end: (optional) date, date string, or None. If None, defaults to today.
    period: (optional) periodicity of data [DAILY, WEEKLY, MONTHLY, QUARTERLY, SEMI_ANNUALLY, YEARLY]
    raise_security_error: If True, raise exceptions caused by invalid sids
    raise_field_error: If True, raise exceptions caused by invalid fields
    period_adjustment: (ACTUAL, CALENDAR, FISCAL)
                        Set the frequency and calendar type of the output
    currency: ISO Code
              Amends the value from local to desired currency
    override_option: (OVERRIDE_OPTION_CLOSE | OVERRIDE_OPTION_GPA)
    pricing_option: (PRICING_OPTION_PRICE | PRICING_OPTION_YIELD)
    non_trading_day_fill_option: (NON_TRADING_WEEKDAYS | ALL_CALENDAR_DAYS | ACTIVE_DAYS_ONLY)
    non_trading_day_fill_method: (PREVIOUS_VALUE | NIL_VALUE)
    calendar_code_override: 2 letter county iso code
    """

    def __init__(
        self,
        sids,
        fields,
        start=None,
        end=None,
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
        self.end = end = pd.to_datetime(end) if empty(end) else pd.Timestamp.now()
        self.start = pd.to_datetime(start) if empty(start) else end + relativedelta(years=-1)
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
            'start': self.start.strftime('%Y-%m-%d'),
            'end': self.end.strftime('%Y-%m-%d'),
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

    def on_security_data_element(self, element):
        """Process a securityData element - FIXME: currently not handling relateDate element"""
        sid = Parser.get_subelement_value(element, Name.SECURITY, self.force_string)
        fields = element.getElement(Name.FIELD_DATA)
        dmap = defaultdict(list)
        for pt in fields.values():
            for f in ['date'] + self.fields:
                field_val = Parser.get_subelement_value(pt, f, self.force_string)
                if isinstance(field_val, datetime.datetime):
                    field_val = field_val.astimezone(self.timezone)
                dmap[f].append(field_val)
        if not dmap:
            df = pd.DataFrame(columns=self.fields)
        else:
            idx = dmap.pop('date')
            df = pd.DataFrame(dmap, columns=self.fields, index=idx)
            df.index.name = 'date'
        self.response.on_security_complete(sid, df)

    def process_response(self, event, is_final):
        for message in Parser.message_iter(event):
            # single security element in historical request
            element = message.getElement(Name.SECURITY_DATA)
            if Name.SECURITY_ERROR in element:
                sid = Parser.get_subelement_value(element, Name.SECURITY, self.force_string)
                self.security_errors.append(Parser.as_security_error(element.getElement(Name.SECURITY_ERROR), sid))
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
        timezone='UTC',
        force_string=False,
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
        self.timezone = pytz.timezone(timezone)
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
        for msg in Parser.message_iter(event):
            for element, err in Parser.security_element_iter(msg.getElement(Name.SECURITY_DATA)):
                if err:
                    self.security_errors.append(err)
                    continue
                self._process_security_element(element)

    def _process_security_element(self, element):
        sid = Parser.get_subelement_value(element, Name.SECURITY, self.force_string)
        fields = element.getElement(Name.FIELD_DATA)
        field_data = Parser.get_subelement_values(fields, self.fields, self.force_string)
        field_data = [x.astimezone(self.timezone) if isinstance(x, datetime.datetime) else x for x in field_data]
        assert len(field_data) == len(self.fields), 'Field length must match data length'
        self.response.on_security_data(sid, dict(list(zip(self.fields, field_data))))
        field_errors = Parser.get_field_errors(element)
        field_errors and self.field_errors.extend(field_errors)


class IntradayTickResponse(BaseResponse):

    def __init__(self, request):
        self.request = request
        self.ticks = []  # iterdict

    def as_dataframe(self):
        """Return a data frame with no set index"""
        df = pd.DataFrame.from_records(self.ticks)
        return df.astype(object).where(df.notna(), None) \
            if self.request.force_string else df


class IntradayTickRequest(BaseRequest):
    """Intraday tick request. Can submit to MSG1 as well for bond runs.
    """
    def __init__(
        self,
        sid,
        start=None,
        end=None,
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
        self.end = end = pd.to_datetime(end) if end else pd.Timestamp.now()
        self.start = pd.to_datetime(start) if start else end + relativedelta(days=-1)

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
        for tick in Parser.element_iter(ticks):
            names = [str(tick.getElement(_).name()) for _ in range(tick.numElements())]
            tickmap = {n: Parser.get_subelement_value(tick, n) for n in names}
            self.response.ticks.append(tickmap)

    def process_response(self, event, is_final):
        for msg in Parser.message_iter(event):
            tdata = msg.getElement('tickData')
            # tickData will have 0 to 1 tickData[] elements
            if 'tickData' in tdata:
                self.on_tick_data(tdata.getElement('tickData'))


class IntradayBarResponse(BaseResponse):

    def __init__(self, request):
        self.request = request
        self.bars = []  # iterdict

    def as_dataframe(self):
        df = pd.DataFrame.from_records(self.bars)
        return df.astype(object).where(df.notna(), None) if self.request.force_string else df


class IntradayBarRequest(BaseRequest):

    def __init__(
        self,
        sid,
        start=None,
        end=None,
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
        self.end = end = pd.to_datetime(end) if end else pd.Timestamp.now()
        self.start = pd.to_datetime(start) if start else end + relativedelta(hours=-1)

    def __repr__(self):
        fmtargs = {
            'clz': self.__class__.__name__,
            'sid': self.sid,
            'event': self.event,
            'start': self.start,
            'end': self.end,
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
        for tick in Parser.element_iter(bars):
            names = [str(tick.getElement(_).name()) for _ in range(tick.numElements())]
            barmap = {n: Parser.get_subelement_value(tick, n) for n in names}
            self.response.bars.append(barmap)

    def process_response(self, event, is_final):
        for msg in Parser.message_iter(event):
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

    def __init__(self, name, type='GLOBAL', group='General', asof=None, language=None):
        super().__init__('//blp/refdata')
        self.name = name
        self.group = group
        self.type = type
        self.asof = pd.to_datetime(asof) if asof else None
        self.language = language

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
        for message in Parser.message_iter(event):
            data = message.getElement('data')
            security = data.getElement(Name.SECURITY_DATA)
            for element, error in Parser.security_iter(security):
                if error:
                    self.security_errors.append(error)
                    continue
                self._process_security_element(element)

    def _process_security_element(self, element):
        sid = Parser.get_subelement_value(element, Name.SECURITY, self.force_string)
        fields = element.getElement(Name.FIELD_DATA)
        fldnames = [str(field.name()) for field in fields.elements()]
        fdata = Parser.get_subelement_values(fields, fldnames)
        self.response.on_security_data(sid, dict(list(zip(fldnames, fdata))))
        ferrors = Parser.get_field_errors(element)
        ferrors and self.field_errors.extend(ferrors)


class Session(blpapi.Session):
    """Wrapper around blpapi.Session with auto-closing"""

    def __init__(self, *args):
        super().__init__(*args)
        atexit.register(self.__cleanup)

    def open_service(self, service_name):
        """Open service. Raise Exception if fails."""
        if not self.openService(service_name):
            raise RuntimeError(f'Failed to open service {service_name}.')

    def __cleanup(self):
        try:
            self.stop()
            self.destroy()
        except:
            pass


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
        start=None,
        end=None,
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
            raise_security_error=raise_security_error,
            raise_field_error=raise_field_error,
            **overrides,
        )
        return self.execute(req)

    def get_intraday_tick(
        self,
        sid,
        events=None,
        start=None,
        end=None,
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
        start=None,
        end=None,
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

    def get_screener(self, name, group='General', type='GLOBAL', asof=None, language=None):
        req = EQSRequest(name, type=type, group=group, asof=asof, language=language)
        return self.execute(req)

    def destroy(self):
        try:
            self.session.stop()
            self.session.destroy()
        except:
            pass


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

    def subscribe(self, handler: BaseEventHandler, runtime=24 * 60 * 60, *args, **kwargs):
        r"""Open subscription.

        Parameters
        ----------
        handler : Async handler instance of BaseHandler. Expects class, not object instance
        runtime : Duration to run service in seconds. Generally best to run between 0 and time to market close..

        Form of created subscription string (subscriptions.add):

        "//blp/mktdata/ticker/IBM US Equity?fields=BID,ASK&interval=2"
        \\-----------/\\------/\\-----------/\\------------------------/
        |          |         |                  |
        Service    Prefix   Instrument           Suffix

        """
        _handler = handler(self.topics, self.fields, *args, **kwargs)
        session = SessionFactory.create(self.host, self.port, self.auth, _handler, self.dispatcher)
        session.open_service('//blp/mktdata')

        subscriptions = blpapi.SubscriptionList()
        options = {'interval': f'{self.interval:.1f}'} if self.interval else {}
        for topic in self.topics:
            subscriptions.add(topic, self.fields, options, blpapi.CorrelationId(topic))
        session.subscribe(subscriptions)

        _runtime = libb.NonBlockingDelay()
        _runtime.delay(runtime)

        while not _runtime.timeout():
            continue

        logger.warning('Subscription runtime expired. Unsubscribing...')
        session.unsubscribe(subscriptions)
