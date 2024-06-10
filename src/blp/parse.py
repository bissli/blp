import contextlib
import datetime
import json
import logging
from collections import defaultdict, namedtuple

import blpapi
import numpy as np
import pandas as pd
from blpapi.datatype import DataType

from date import LCL, UTC, Date, DateTime, Time, Timezone
from libb import round_digit_string, underscore_to_camelcase

logger = logging.getLogger(__name__)


class NameType(type):
    def __getattribute__(cls, name):
        _name = underscore_to_camelcase(name)
        return blpapi.Name.findName(_name) or blpapi.Name(_name)


class Name(metaclass=NameType):
    """Blpapi Name class wrapper"""


SecurityError = namedtuple(
    Name.SECURITY_ERROR,
    [Name.SECURITY, Name.CATEGORY, Name.MESSAGE, Name.SUBCATEGORY],
)
FieldError = namedtuple(
    Name.FIELD_ERROR,
    [Name.SECURITY, Name.FIELD, Name.CATEGORY, Name.MESSAGE, Name.SUBCATEGORY],
)

NUMERIC_TYPES = (
    DataType.BOOL, DataType.CHAR, DataType.BYTE, DataType.INT32,
    DataType.INT64, DataType.FLOAT32, DataType.FLOAT64, DataType.BYTEARRAY,
    DataType.DECIMAL
)


class Parser:
    """Interpreter class for Bloomberg Events

    One Event -> one or more Message -> one or more Element

    """

    def __init__(
        self,
        assumed_timezone: Timezone = UTC,
        desired_timezone: Timezone = LCL,
        decimal_places: int = None,
        time_as_datetime: bool = False
    ):
        self.assumed_timezone = assumed_timezone or UTC
        self.desired_timezone = desired_timezone or LCL
        self.time_as_datetime = time_as_datetime
        self.decimal_places = decimal_places

    #
    # iterator wrappers to handle errors in elements
    #

    def security_element_iter(self, elements):
        """Provide a security data iterator by returning a tuple of (Element, SecurityError) which are mutually exclusive"""
        if elements.name() != Name.SECURITY_DATA:
            return None, None
        assert elements.isArray()
        for element in elements.values():  # same as element_iter
            err = self.get_security_error(element)
            result = (None, err) if err else (element, None)
            yield result

    @staticmethod
    def element_iter(elements) -> list[blpapi.element.Element]:
        yield from elements.values() if elements.isArray() else []

    @staticmethod
    def message_iter(event) -> list[blpapi.message.Message]:
        """Provide a message iterator which checks for a response error prior to returning"""
        for message in event:
            if Name.RESPONSE_ERROR in message:
                raise Exception(f'REQUEST FAILED: {str(message[Name.RESPONSE_ERROR])}')
            yield message

    #
    # value getters
    #

    def get_subelement_value(self, element, name, force_string=False):
        """Return the value of the child element with name in the parent Element"""
        if name not in element:
            logger.debug(f'Response did not contain field {name}')
            return np.nan
        return self.element_as_value(element.getElement(name), force_string)

    def get_subelement_values(self, element, names, force_string=False):
        """Return a list of values for the specified child fields. If field not in Element then replace with nan."""
        return [self.get_subelement_value(element, name, force_string) for name in names]

    def element_as_value(self, element=None, force_string=False):
        """Convert the specified element as a python value. Aware of instance
        timezone if instantiated as instance.
        """
        dtype = element.datatype()
        if dtype == DataType.SEQUENCE:
            if not force_string:
                with contextlib.suppress(blpapi.exception.UnsupportedOperationException):
                    return self._sequence_as_dataframe(element)
            return self._sequence_as_json(element)
        if force_string:
            if element.isNull():
                return ''
            return round_digit_string(element.getValueAsString(), self.decimal_places)
        if dtype in NUMERIC_TYPES:
            if element.isNull():
                return np.nan
            return element.getValue()
        if dtype == DataType.DATE:
            if element.isNull():
                return pd.NaT
            # parsing a datetime.date object
            return Date.parse(element.getValue())
        if dtype in {DataType.DATETIME, DataType.TIME}:
            if element.isNull():
                return pd.NaT
            obj = element.getValue()
            if isinstance(obj, datetime.time):
                # parsing datetime.time with no tzinfo
                _date = Date.today()
                _time = Time.parse(obj).replace(tzinfo=self.assumed_timezone)
                _datetime = DateTime\
                    .combine(_date, _time, self.assumed_timezone)\
                    .in_timezone(self.desired_timezone)
                if self.time_as_datetime:
                    return _datetime
                return _datetime.time()
            if isinstance(obj, datetime.datetime):
                # parsing datetime.datetime with no tzinfo
                return DateTime\
                    .parse(obj)\
                    .replace(tzinfo=self.assumed_timezone)\
                    .in_timezone(self.desired_timezone)
        if dtype == DataType.CHOICE:
            logger.warning('CHOICE data type needs implemented')
        if element.isNull():
            return ''
        return round_digit_string(element.getValueAsString(), self.decimal_places)

    #
    # error getters
    #

    def get_security_error(self, element) -> SecurityError | None:
        """Return a SecurityError if the specified securityData element has one, else return None"""
        if element.name() != Name.SECURITY_DATA:
            return
        assert not element.isArray()
        if Name.SECURITY_ERROR in element:
            secid = self.get_subelement_value(element, Name.SECURITY)
            error = self.as_security_error(element.getElement(Name.SECURITY_ERROR), secid)
            return error

    def get_field_errors(self, element) -> list[FieldError] | None:
        """Return a list of FieldErrors if the specified securityData element has field errors"""
        if element.name() != Name.SECURITY_DATA:
            return []
        assert not element.isArray()
        if Name.FIELD_EXCEPTIONS in element:
            secid = self.get_subelement_value(element, Name.SECURITY)
            errors = self.as_field_error(element.getElement(Name.FIELD_EXCEPTIONS), secid)
            return errors
        return []

    def as_security_error(self, element, secid):
        """Convert the securityError element to a SecurityError"""
        if element.name() != Name.SECURITY_ERROR:
            return
        cat = self.get_subelement_value(element, Name.CATEGORY)
        msg = self.get_subelement_value(element, Name.MESSAGE)
        subcat = self.get_subelement_value(element, Name.SUBCATEGORY)
        return SecurityError(security=secid, category=cat, message=msg, subcategory=subcat)

    def as_field_error(self, element, secid):
        """Convert a fieldExceptions element to a FieldError or FieldError array"""
        if element.name() != Name.FIELD_EXCEPTIONS:
            return []
        if element.isArray():
            return [self.as_field_error(_, secid) for _ in element.values()]
        fld = self.get_subelement_value(element, Name.FIELD_ID)
        info = element.getElement(Name.ERROR_INFO)
        cat = self.get_subelement_value(info, Name.CATEGORY)
        msg = self.get_subelement_value(info, Name.MESSAGE)
        subcat = self.get_subelement_value(info, Name.SUBCATEGORY)
        return FieldError(security=secid, field=fld, category=cat, message=msg, subcategory=subcat)

    #
    # private methods
    #

    def _sequence_as_dataframe(self, elements):
        """Convert an element with DataType Sequence to a DataFrame."""
        data = defaultdict(list)
        cols = []
        for i, element in enumerate(elements.values()):
            if i == 0:  # Get the ordered cols and assume they are constant
                cols = [str(_.name()) for _ in element.elements()]
            for subelement in element.elements():
                data[str(subelement.name())].append(self.element_as_value(subelement))
        return pd.DataFrame(data, columns=cols)

    def _sequence_as_json(self, elements):
        data = []
        for element in elements.values():
            for subelement in element.elements():
                d = {str(subelement.name()):
                     round_digit_string(subelement.getValueAsString(), self.decimal_places)}
                data += [d]
        return json.dumps(data) if data else ''
