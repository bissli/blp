import atexit
import logging
import time
import warnings
from abc import ABCMeta, abstractmethod
from collections import defaultdict, namedtuple
from datetime import datetime

import blpapi
import numpy as np
import pandas as pd
import pytz
import win32api
import win32con
from bbg.util import Name, NonBlockingDelay
from blpapi.event import Event
from dateutil.relativedelta import relativedelta

logger = logging.getLogger(__name__)


SecurityErrorAttrs = [Name.SECURITY, Name.SOURCE, Name.CODE, Name.CATEGORY, Name.MESSAGE, Name.SUBCATEGORY]
SecurityError = namedtuple(Name.SECURITY_ERROR, SecurityErrorAttrs)
FieldErrorAttrs = [
    Name.SECURITY,
    Name.FIELD,
    Name.SOURCE,
    Name.CODE,
    Name.CATEGORY,
    Name.MESSAGE,
    Name.SUBCATEGORY,
]
FieldError = namedtuple(Name.FIELD_ERROR, FieldErrorAttrs)

UTC = pytz.timezone('UTC')
GMT = pytz.timezone('GMT')
EST = pytz.timezone('US/Eastern')


class XmlHelper:
    """Interpreter class for Bloomberg responses"""

    @staticmethod
    def security_iter(nodearr):
        """Provide a security data iterator by returning a tuple of (Element, SecurityError) which are mutually exclusive"""
        assert nodearr.name() == Name.SECURITY_DATA and nodearr.isArray()
        for i in range(nodearr.numValues()):
            node = nodearr.getValue(i)
            err = XmlHelper.get_security_error(node)
            result = (None, err) if err else (node, None)
            yield result

    @staticmethod
    def node_iter(nodearr):
        assert nodearr.isArray()
        for i in range(nodearr.numValues()):
            yield nodearr.getValue(i)

    @staticmethod
    def message_iter(event):
        """Provide a message iterator which checks for a response error prior to returning"""
        for msg in event:
            logger.info(f'Received response to request {msg.getRequestId()}')
            logger.debug(msg.toString())
            if Name.RESPONSE_ERROR in msg:
                responseError = msg[Name.RESPONSE_ERROR]
                raise Exception(f'REQUEST FAILED: {responseError}')
            yield msg

    @staticmethod
    def get_sequence_value(node):
        """Convert an element with DataType Sequence to a DataFrame.
        Note this may be a naive implementation as I assume that bulk data is always a table
        """
        assert node.datatype() == 15
        data = defaultdict(list)
        cols = []
        for i in range(node.numValues()):
            row = node.getValue(i)
            if i == 0:  # Get the ordered cols and assume they are constant
                cols = [str(row.getElement(_).name()) for _ in range(row.numElements())]

            for cidx in range(row.numElements()):
                col = row.getElement(cidx)
                data[str(col.name())].append(XmlHelper.as_value(col))
        return pd.DataFrame(data, columns=cols)

    @staticmethod
    def as_value(ele):
        """Convert the specified element as a python value"""
        dtype = ele.datatype()
        if dtype in (1, 2, 3, 4, 5, 6, 7, 9, 12):
            # BOOL, CHAR, BYTE, INT32, INT64, FLOAT32, FLOAT64, BYTEARRAY, DECIMAL)
            try:
                return ele.getValue()
            except Exception as exc:
                print(type(exc))
                return np.nan
        if dtype == 8:  # String
            val = ele.getValue()
            return str(val)
        if dtype == 10:  # Date
            if ele.isNull():
                return pd.NaT
            v = ele.getValue()
            if not v:
                return pd.NaT
            dt = datetime(year=v.year, month=v.month, day=v.day)
            return dt.astimezone(EST)
        if dtype in (11, 13):  # Datetime
            if ele.isNull():
                return pd.NaT
            v = ele.getValue()
            now = datetime.now()
            dt = datetime(year=now.year, month=now.month, day=now.day, hour=v.hour, minute=v.minute, second=v.second)
            return dt.astimezone(EST)
        if dtype == 14:  # Enumeration
            return str(ele.getValue())
        if dtype == 16:  # Choice
            raise NotImplementedError('CHOICE data type needs implemented')
        if dtype == 15:  # SEQUENCE
            return XmlHelper.get_sequence_value(ele)
        raise NotImplementedError(f'Unexpected data type {dtype}. Check documentation')

    @staticmethod
    def get_child_value(parent, name, allow_missing=0):
        """Return the value of the child element with name in the parent Element"""
        if not parent.hasElement(name):
            if allow_missing:
                return np.nan
            raise Exception(f'failed to find child element {name} in parent')
        return XmlHelper.as_value(parent.getElement(name))

    @staticmethod
    def get_child_values(parent, names):
        """Return a list of values for the specified child fields. If field not in Element then replace with nan."""
        vals = []
        for name in names:
            if parent.hasElement(name):
                vals.append(XmlHelper.as_value(parent.getElement(name)))
            else:
                vals.append(np.nan)
        return vals

    @staticmethod
    def as_security_error(node, secid):
        """Convert the securityError element to a SecurityError"""
        assert node.name() == Name.SECURITY_ERROR
        src = XmlHelper.get_child_value(node, Name.SOURCE)
        code = XmlHelper.get_child_value(node, Name.CODE)
        cat = XmlHelper.get_child_value(node, Name.CATEGORY)
        msg = XmlHelper.get_child_value(node, Name.MESSAGE)
        subcat = XmlHelper.get_child_value(node, Name.SUBCATEGORY)
        return SecurityError(security=secid, source=src, code=code, category=cat, message=msg, subcategory=subcat)

    @staticmethod
    def as_field_error(node, secid):
        """Convert a fieldExceptions element to a FieldError or FieldError array"""
        assert node.name() == Name.FIELD_EXCEPTIONS
        if node.isArray():
            return [XmlHelper.as_field_error(node.getValue(_), secid) for _ in range(node.numValues())]
        fld = XmlHelper.get_child_value(node, Name.FIELD_ID)
        info = node.getElement(Name.ERROR_INFO)
        src = XmlHelper.get_child_value(info, Name.SOURCE)
        code = XmlHelper.get_child_value(info, Name.CODE)
        cat = XmlHelper.get_child_value(info, Name.CATEGORY)
        msg = XmlHelper.get_child_value(info, Name.MESSAGE)
        subcat = XmlHelper.get_child_value(info, Name.SUBCATEGORY)
        return FieldError(
            security=secid, field=fld, source=src, code=code, category=cat, message=msg, subcategory=subcat
        )

    @staticmethod
    def get_security_error(node):
        """Return a SecurityError if the specified securityData element has one, else return None"""
        assert node.name() == Name.SECURITY_DATA and not node.isArray()
        if node.hasElement(Name.SECURITY_ERROR):
            secid = XmlHelper.get_child_value(node, Name.SECURITY)
            err = XmlHelper.as_security_error(node.getElement(Name.SECURITY_ERROR), secid)
            return err

    @staticmethod
    def get_field_errors(node):
        """Return a list of FieldErrors if the specified securityData element has field errors"""
        assert node.name() == Name.SECURITY_DATA and not node.isArray()
        nodearr = node.getElement(Name.FIELD_EXCEPTIONS)
        if nodearr.numValues() > 0:
            secid = XmlHelper.get_child_value(node, Name.SECURITY)
            errors = XmlHelper.as_field_error(nodearr, secid)
            return errors


def debug_event(event):
    print(f'unhandled event: {event.EventType}')
    if event.EventType in [Event.RESPONSE, Event.PARTIAL_RESPONSE]:
        print('messages:')
        for msg in XmlHelper.message_iter(event):
            print(msg.Print)


class BaseRequest(metaclass=ABCMeta):
    """Base Request Object"""

    def __init__(self, service_name, ignore_security_error=False, ignore_field_error=False):
        self.field_errors = []
        self.security_errors = []
        self.ignore_security_error = ignore_security_error
        self.ignore_field_error = ignore_field_error
        self.service_name = service_name
        self.response = None

    @abstractmethod
    def prepare_response(self):
        pass

    @property
    def has_exception(self):
        if not self.ignore_security_error and len(self.security_errors) > 0:
            return True
        if not self.ignore_field_error and len(self.field_errors) > 0:
            return True

    def raise_exception(self):
        if not self.ignore_security_error and len(self.security_errors) > 0:
            msgs = [f'({s.security}, {s.category}, {s.message})' for s in self.security_errors]
            raise Exception(f"SecurityError: {','.join(msgs)}")
        if not self.ignore_field_error and len(self.field_errors) > 0:
            msgs = [f'({s.security}, {s.field}, {s.category}, {s.message})' for s in self.field_errors]
            raise Exception(f"FieldError: {','.join(msgs)}")
        raise Exception('Programmer Error: No exception to raise')

    @abstractmethod
    def create_request(self, service):
        pass

    @abstractmethod
    def process_response(self, event, is_final):
        pass

    def on_admin_event(self, event):
        for msg in event:
            if msg.messageType() == Name.SESSION_CONNECTION_UP:
                logger.info('Connected ...')
            elif msg.hasElement(Name.SESSION_STARTED):
                logger.info('Started new Bbg session ...')
            elif msg.hasElement('ServiceOpened'):
                logger.info('Opened Bbg service ...')
            elif msg.messageType() == 'SessionTerminated':
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


class BaseResponse(metaclass=ABCMeta):
    """Base class for Responses"""

    @abstractmethod
    def as_frame(self):
        pass


class HistoricalDataResponse(BaseResponse):
    def __init__(self, request):
        self.request = request
        self.response_map = {}

    def on_security_complete(self, sid, frame):
        self.response_map[sid] = frame

    def as_map(self):
        return self.response_map

    def as_frame(self):
        """:return: Multi-Index DataFrame"""
        sids, frames = list(self.response_map.keys()), list(self.response_map.values())
        frame = pd.concat(frames, keys=sids, axis=1)
        return frame


class HistoricalDataRequest(BaseRequest):
    """A class which manages the creation of the Bloomberg HistoricalDataRequest and
    the processing of the associated Response.

    Parameters
    ----------
    sids: bbg security identifier(s)
    fields: bbg field name(s)
    start: (optional) date, date string , or None. If None, defaults to 1 year ago.
    end: (optional) date, date string, or None. If None, defaults to today.
    period: (optional) periodicity of data [DAILY, WEEKLY, MONTHLY, QUARTERLY, SEMI_ANNUALLY, YEARLY]
    ignore_security_error: If True, ignore exceptions caused by invalid sids
    ignore_field_error: If True, ignore exceptions caused by invalid fields
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
        ignore_security_error=False,
        ignore_field_error=False,
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
            ignore_security_error=ignore_security_error,
            ignore_field_error=ignore_field_error,
        )
        period = period or 'DAILY'
        assert period in ('DAILY', 'WEEKLY', 'MONTHLY', 'QUARTERLY', 'SEMI_ANNUALLY', 'YEARLY')
        self.is_single_sid = is_single_sid = isinstance(sids, str)
        self.is_single_field = is_single_field = isinstance(fields, str)
        self.sids = is_single_sid and [sids] or list(sids)
        self.fields = is_single_field and [fields] or list(fields)
        self.end = end = pd.to_datetime(end) if end else pd.Timestamp.now()
        self.start = pd.to_datetime(start) if start else end + relativedelta(years=-1)
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

    def on_security_data_node(self, node):
        """Process a securityData node - FIXME: currently not handling relateDate node"""
        sid = XmlHelper.get_child_value(node, Name.SECURITY)
        farr = node.getElement(Name.FIELD_DATA)
        dmap = defaultdict(list)
        for i in range(farr.numValues()):
            pt = farr.getValue(i)
            [dmap[f].append(XmlHelper.get_child_value(pt, f, allow_missing=1)) for f in ['date'] + self.fields]

        if not dmap:
            frame = pd.DataFrame(columns=self.fields)
        else:
            idx = dmap.pop('date')
            frame = pd.DataFrame(dmap, columns=self.fields, index=idx)
            frame.index.name = 'date'
        self.response.on_security_complete(sid, frame)

    def process_response(self, event, is_final):
        for msg in XmlHelper.message_iter(event):
            # Single security element in historical request
            node = msg.getElement(Name.SECURITY_DATA)
            if node.hasElement(Name.SECURITY_ERROR):
                sid = XmlHelper.get_child_value(node, Name.SECURITY)
                self.security_errors.append(XmlHelper.as_security_error(node.getElement(Name.SECURITY_ERROR), sid))
            else:
                self.on_security_data_node(node)


class ReferenceDataResponse(BaseResponse):
    def __init__(self, request):
        self.request = request
        self.response_map = defaultdict(dict)

    def on_security_data(self, sid, fieldmap):
        self.response_map[sid].update(fieldmap)

    def as_map(self):
        return self.response_map

    def as_frame(self):
        """:return: Multi-Index DataFrame"""
        data = {sid: pd.Series(data) for sid, data in self.response_map.items()}
        frame = pd.DataFrame.from_dict(data, orient='index')
        # layer in any missing fields just in case
        frame = frame.reindex(self.request.fields, axis=1)
        return frame


class ReferenceDataRequest(BaseRequest):
    def __init__(
        self,
        sids,
        fields,
        ignore_security_error=False,
        ignore_field_error=False,
        return_formatted_value=None,
        use_utc_time=None,
        **overrides,
    ):
        """response_type: (frame, map) how to return the results"""
        super().__init__(
            '//blp/refdata',
            ignore_security_error=ignore_security_error,
            ignore_field_error=ignore_field_error,
        )
        self.is_single_sid = is_single_sid = isinstance(sids, str)
        self.is_single_field = is_single_field = isinstance(fields, str)
        self.sids = isinstance(sids, str) and [sids] or sids
        self.fields = isinstance(fields, str) and [fields] or fields
        self.return_formatted_value = return_formatted_value
        self.use_utc_time = use_utc_time
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
        self.set_flag(request, self.use_utc_time, 'useUTCTime')
        self.apply_overrides(request, self.overrides)
        return request

    def process_response(self, event, is_final):
        for msg in XmlHelper.message_iter(event):
            for node, error in XmlHelper.security_iter(msg.getElement(Name.SECURITY_DATA)):
                if error:
                    self.security_errors.append(error)
                else:
                    self._process_security_node(node)

    def _process_security_node(self, node):
        sid = XmlHelper.get_child_value(node, Name.SECURITY)
        farr = node.getElement(Name.FIELD_DATA)
        fdata = XmlHelper.get_child_values(farr, self.fields)
        assert len(fdata) == len(self.fields), 'field length must match data length'
        self.response.on_security_data(sid, dict(list(zip(self.fields, fdata))))
        ferrors = XmlHelper.get_field_errors(node)
        ferrors and self.field_errors.extend(ferrors)


class IntradayTickResponse(BaseResponse):
    def __init__(self, request):
        self.request = request
        self.ticks = []  # array of dicts

    def as_frame(self):
        """Return a data frame with no set index"""
        return pd.DataFrame.from_records(self.ticks)


class IntradayTickRequest(BaseRequest):
    """Intraday tick request. Can submit to MSG1 as well for bond runs."""

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
        self.events = isinstance(events, str) and [events] or events
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
        for tick in XmlHelper.node_iter(ticks):
            names = [str(tick.getElement(_).name()) for _ in range(tick.numElements())]
            tickmap = {n: XmlHelper.get_child_value(tick, n) for n in names}
            self.response.ticks.append(tickmap)

    def process_response(self, event, is_final):
        for msg in XmlHelper.message_iter(event):
            tdata = msg.getElement('tickData')
            # tickData will have 0 to 1 tickData[] elements
            if tdata.hasElement('tickData'):
                self.on_tick_data(tdata.getElement('tickData'))


class IntradayBarResponse(BaseResponse):
    def __init__(self, request):
        self.request = request
        self.bars = []  # array of dicts

    def as_frame(self):
        return pd.DataFrame.from_records(self.bars)


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
        for tick in XmlHelper.node_iter(bars):
            names = [str(tick.getElement(_).name()) for _ in range(tick.numElements())]
            barmap = {n: XmlHelper.get_child_value(tick, n) for n in names}
            self.response.bars.append(barmap)

    def process_response(self, event, is_final):
        for msg in XmlHelper.message_iter(event):
            data = msg.getElement('barData')
            # tickData will have 0 to 1 tickData[] elements
            if data.hasElement('barTickData'):
                self.on_bar_data(data.getElement('barTickData'))


class EQSResponse(BaseResponse):
    def __init__(self, request):
        self.request = request
        self.response_map = defaultdict(dict)

    def on_security_data(self, sid, fieldmap):
        self.response_map[sid].update(fieldmap)

    def as_map(self):
        return self.response_map

    def as_frame(self):
        """:return: Multi-Index DataFrame"""
        data = {sid: pd.Series(data) for sid, data in self.response_map.items()}
        return pd.DataFrame.from_dict(data, orient='index')


class EQSRequest(BaseRequest):
    def __init__(self, name, type='GLOBAL', group='General', asof=None, language=None):
        super(EQSRequest, self).__init__('//blp/refdata')
        self.name = name
        self.group = group
        self.type = type
        self.asof = asof and pd.to_datetime(asof) or None
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
        for msg in XmlHelper.message_iter(event):
            data = msg.getElement('data')
            for node, error in XmlHelper.security_iter(data.getElement(Name.SECURITY_DATA)):
                if error:
                    self.security_errors.append(error)
                else:
                    self._process_security_node(node)

    def _process_security_node(self, node):
        sid = XmlHelper.get_child_value(node, Name.SECURITY)
        farr = node.getElement(Name.FIELD_DATA)
        fldnames = [str(farr.getElement(_).name()) for _ in range(farr.numElements())]
        fdata = XmlHelper.get_child_values(farr, fldnames)
        self.response.on_security_data(sid, dict(list(zip(fldnames, fdata))))
        ferrors = XmlHelper.get_field_errors(node)
        ferrors and self.field_errors.extend(ferrors)


class Session(blpapi.Session):
    """Wrapper around blpapi.Session with auto-closing"""

    def __init__(self, *args):
        super().__init__(*args)
        atexit.register(self.__cleanup__)

    def open_service(self, service_name):
        """Open service. Raise Exception if fails."""
        if not self.openService(service_name):
            raise Exception(f'Failed to open service {service_name}')

    def __cleanup__(self):
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
        print('Connecting to Bloomberg BBComm session...')
        session = create_session(host, port, auth, event_handler, event_dispatcher)
        if not session.start():
            raise Exception('Failed to create session')
        envuser = win32api.GetUserNameEx(win32con.NameSamCompatible)
        print(f'Connected to Bloomberg BBComm as {envuser}')
        return session


class Context:
    """Submits requests to the Bloomberg API and dispatches the events back to the request
    object for processing.
    """

    def __init__(self, host='localhost', port=8194, auth='AuthenticationType=OS_LOGON', session=None):
        self.host = host
        self.port = port
        self.auth = auth
        self.session = session and session or SessionFactory.create(host, port, auth)

    def __repr__(self):
        fmtargs = {
            'clz': self.__class__.__name__,
            'host': self.host,
            'port': self.port,
            'auth': self.auth,
        }
        return '<{clz}({host}:{port}:{auth})'.format(**fmtargs)

    def execute(self, request: BaseRequest):
        logger.info(f'Sending request: {repr(request)}')
        self.session.open_service(request.service_name)
        service = self.session.getService(request.service_name)
        _request = request.create_request(service)
        cid = self.session.sendRequest(_request)
        request.prepare_response()
        return self._wait_for_response(request)

    def _wait_for_response(self, request: BaseRequest):
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
                    logger.info('Processing RESPONSE ...')
                    request.process_response(event, is_final=True)
                    logger.info('END RESPONSE')
                    break
                case Event.PARTIAL_RESPONSE:
                    logger.info('Processing PARTIAL_RESPONSE ...')
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
        ignore_security_error=False,
        ignore_field_error=False,
        **overrides,
    ):
        req = HistoricalDataRequest(
            sids,
            flds,
            start=start,
            end=end,
            period=period,
            ignore_security_error=ignore_security_error,
            ignore_field_error=ignore_field_error,
            **overrides,
        )
        return self.execute(req)

    def get_reference_data(self, sids, flds, ignore_security_error=False, ignore_field_error=False, **overrides):
        req = ReferenceDataRequest(
            sids,
            flds,
            ignore_security_error=ignore_security_error,
            ignore_field_error=ignore_field_error,
            **overrides,
        )
        return self.execute(req)

    def get_intraday_tick(
        self,
        sid,
        events=['TRADE', 'ASK', 'BID'],
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


class BaseEventHandler(metaclass=ABCMeta):
    """Base event handler"""

    def __init__(self, topics, fields):
        self.topics = topics
        self.fields = fields

    def __call__(self, event, session):
        """This method is called from Bloomberg session in a separate thread
        for each incoming event.
        """
        try:
            event_type = event.eventType()
            if event_type == Event.SUBSCRIPTION_DATA:
                logger.info('next(): subscription data')
                self.on_data_event(event, session)
                return
            if event_type == Event.SUBSCRIPTION_STATUS:
                logger.info('next(): subscription status')
                self.on_status_event(event, session)
                return
            if event_type == Event.TIMEOUT:
                return
            self.on_misc_event(event, session)
        except blpapi.Exception as exception:
            print(f'Failed to process event {event}: {exception}')

    @abstractmethod
    def on_status_event(self, event, _):
        pass

    @abstractmethod
    def on_data_event(self, event, _):
        pass

    @abstractmethod
    def on_misc_event(self, event, _):
        pass


class LoggingEventHandler(BaseEventHandler):
    """Default event handler"""

    def __init__(self, topics, fields, override={}):
        super().__init__(topics, fields)
        self.override = override
        # create dataframe grid
        nrows, ncols = len(self.topics), len(self.fields)
        vals = np.repeat(np.nan, nrows * ncols).reshape((nrows, ncols))
        self.frame = pd.DataFrame(vals, columns=self.fields, index=[self.override.get(t, t) for t in self.topics])

    def on_status_event(self, event, session):
        for msg in XmlHelper.message_iter(event):
            topic = msg.correlationId().value()
            match msg.messageType():
                case Name.SUBSCRIPTION_FAILURE:
                    desc = msg.getElement('reason').getElementAsString('description')
                    raise Exception(f'Subscription failed topic={topic} desc={desc}')
                case Name.SUBSCRIPTION_TERMINATED:
                    # Subscription can be terminated if the session identity is revoked.
                    print(f'Subscription for {topic} TERMINATED')

    def on_data_event(self, event, session):
        for msg in XmlHelper.message_iter(event):
            topic = msg.correlationId().value()
            print(f'Received event for {get_timestamp()}: {self.override.get(topic, topic)}')
            ridx = self.topics.index(topic)
            for cidx, field in enumerate(self.fields):
                if field.upper() in msg:
                    val = XmlHelper.get_child_value(msg, field.upper())
                    with warnings.catch_warnings():
                        warnings.simplefilter(action='ignore', category=FutureWarning)
                        self.frame.iloc[ridx, cidx] = val

    def on_misc_event(self, event, _):
        for msg in event:
            match msg.messageType():
                case Name.SLOW_CONSUMER_WARNING:
                    print(
                        f'{Name.SLOW_CONSUMER_WARNING} - The event queue is '
                        + 'beginning to approach its maximum capacity and '
                        + 'the application is not processing the data fast '
                        + 'enough. This could lead to ticks being dropped'
                        + ' (DataLoss).\n'
                    )
                case Name.SLOW_CONSUMER_WARNING_CLEARED:
                    print(
                        f'{Name.SLOW_CONSUMER_WARNING_CLEARED} - the event '
                        + 'queue has shrunk enough that there is no '
                        + 'longer any immediate danger of overflowing the '
                        + 'queue. If any precautionary actions were taken '
                        + 'when SlowConsumerWarning message was delivered, '
                        + 'it is now safe to continue as normal.\n'
                    )
                case Name.DATA_LOSS:
                    print(msg)
                    topic = msg.correlationId().value()
                    print(
                        f'{Name.DATA_LOSS} - The application is too slow to '
                        + 'process events and the event queue is overflowing. '
                        + f'Data is lost for topic {topic}.\n'
                    )
                case Name.SESSION_TERMINATED:
                    # SESSION_STATUS events can happen at any time and
                    # should be handled as the session can be terminated,
                    # e.g. session identity can be revoked at a later
                    # time, which terminates the session.
                    print('Session terminated')


class Subscription:
    """
    Fields: Bloomberg will always return all subscribable fields in reach response.
        What `fields` does is specify which fields (when they change) should trigger an event.
        Internally though our event handler will filter for the fields we request.

    """

    def __init__(
        self,
        topics,  # IE ['IBM US Equity', 'TSLA US Equity' ]
        fields,  # IE `['BID', 'ASK', 'TRADE']`
        interval=0,  # Time in seconds to intervalize the subscriptions
        host='localhost',
        port=8194,
        auth='AuthenticationType=OS_LOGON',
        dispatcher=None,  # if None, will create a defualt Dispatcher (single thread)
    ):
        self.fields = isinstance(fields, str) and [fields] or fields
        self.topics = isinstance(topics, str) and [topics] or topics
        self.interval = interval
        self.host = host
        self.port = port
        self.auth = auth
        self.dispatcher = dispatcher

    def subscribe(
        self,
        runtime=24 * 60 * 60,
        handler=LoggingEventHandler,
        override={},
    ):
        """Form of created subscription string (subscriptions.add):

        "//blp/mktdata/ticker/IBM US Equity?fields=BID,ASK&interval=2"
        \\-----------/\\------/\\-----------/\\------------------------/
        |          |         |                  |
        Service    Prefix   Instrument           Suffix

        handler: async handler of events: primary driver of event routing.
        """
        _handler = handler(self.topics, self.fields, override=override)
        session = SessionFactory.create(self.host, self.port, self.auth, _handler, self.dispatcher)
        session.open_service('//blp/mktdata')

        subscriptions = blpapi.SubscriptionList()
        options = {'interval': f'{self.interval:.1f}'} if self.interval else {}
        for topic in self.topics:
            subscriptions.add(topic, self.fields, options, blpapi.CorrelationId(topic))
        session.subscribe(subscriptions)

        _runtime = NonBlockingDelay()
        _runtime.delay(runtime)

        while not _runtime.timeout():
            continue

        print('Subscription runtime expired. Unsubscribing...')

        session.unsubscribe(subscriptions)


def get_timestamp():
    return time.strftime('%Y/%m/%d %X')
