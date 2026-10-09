import contextlib
import datetime
import json
import logging
from collections import namedtuple
from typing import Any

import blpapi
import numpy as np
import pandas as pd
from blpapi.datatype import DataType
from opendate import LCL, UTC, Date, DateTime, Time, Timezone

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

CLOCK_SKEW_ALLOWANCE = datetime.timedelta(hours=1)


class Parser:
    """Interpreter class for Bloomberg Events

    One Event -> one or more Message -> one or more Element

    """

    def __init__(
        self,
        assumed_timezone: Timezone = UTC,
        desired_timezone: Timezone = LCL,
        decimal_places: int = None,
        time_as_datetime: bool = False,
        include_ticker_field=False,
        field_parse_custom: dict = {}

    ):
        self.assumed_timezone = assumed_timezone or UTC
        self.desired_timezone = desired_timezone or LCL
        self.time_as_datetime = time_as_datetime
        self.include_ticker_field = include_ticker_field
        self.decimal_places = decimal_places
        self.field_parse_custom = field_parse_custom or {}

    #
    # iterator wrappers to handle errors in elements
    #

    def security_element_iter(self, elements):
        """Provide a security data iterator by returning a tuple of (Element, SecurityError) which are mutually exclusive.
        """
        if elements.name() != Name.SECURITY_DATA:
            return
        assert elements.isArray()
        for element in elements.values():  # same as element_iter
            err = self.get_security_error(element)
            result = (None, err) if err else (element, None)
            yield result

    @staticmethod
    def element_iter(elements):
        """Iterate over array elements or return empty if not array.
        """
        yield from elements.values() if elements.isArray() else []

    @staticmethod
    def message_iter(event):
        """Provide a message iterator which checks for a response error prior to returning.
        """
        for message in event:
            if Name.RESPONSE_ERROR in message:
                raise Exception(f'REQUEST FAILED: {str(message[Name.RESPONSE_ERROR])}')
            yield message

    #
    # value getters
    #

    def get_subelement_value(self, element, name, force_string=False):
        """Return the value of the child element with name in the parent Element.
        """
        if name not in element:
            logger.debug(f'Response did not contain field {name}')
            return np.nan
        _element = element.getElement(name)
        if name in self.field_parse_custom:
            return self.field_parse_custom[name](_element)
        return self.element_as_value(_element, force_string)

    def get_subelement_values(self, element, names, force_string=False) -> list:
        """Return a list of values for the specified child fields. If field not in Element then replace with nan.
        """
        return [self.get_subelement_value(element, name, force_string) for name in names]

    def element_as_value(
        self,
        element: blpapi.Element = None,
        force_string: bool = False
    ) -> Any:
        """Python value of a Bloomberg element, with timezone awareness.

        Parameters
        ----------
        element : blpapi.Element
            Element to read.
        force_string : bool, default False
            Return every value as a string, and a bulk field as JSON.
            A value Bloomberg types as a number goes through
            round_digit_string.

        Returns
        -------
        Any
            The element's value. A string type comes back as Bloomberg sent
            it, with surrounding whitespace stripped, so a CUSIP keeps its
            leading zeros.
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
            if dtype in NUMERIC_TYPES:
                return round_digit_string(element.getValueAsString(), self.decimal_places)
            return element.getValueAsString().strip()
        if dtype in NUMERIC_TYPES:
            if element.isNull():
                return np.nan
            return element.getValue()
        if dtype == DataType.DATE:
            if element.isNull():
                return pd.NaT
            # parsing a datetime.date object
            return Date.instance(element.getValue())
        if dtype in {DataType.DATETIME, DataType.TIME}:
            if element.isNull():
                return pd.NaT
            obj = element.getValue()
            if isinstance(obj, datetime.time):
                dated = self._assign_date(obj).in_timezone(self.desired_timezone)
                if self.time_as_datetime:
                    return dated
                return dated.time()
            if isinstance(obj, datetime.datetime):
                # parsing datetime.datetime with no tzinfo
                return DateTime\
                    .instance(obj)\
                    .replace(tzinfo=self.assumed_timezone)\
                    .in_timezone(self.desired_timezone)
        if dtype == DataType.CHOICE:
            logger.warning('CHOICE data type needs implemented')
        if element.isNull():
            return ''
        return element.getValueAsString().strip()

    def _assign_date(self, time_of_day: datetime.time) -> DateTime:
        """time_of_day in assumed_timezone on this machine's local date.
        """
        time_in_zone = Time.instance(time_of_day).replace(tzinfo=self.assumed_timezone)
        return DateTime.combine(Date.today(), time_in_zone, self.assumed_timezone)

    #
    # error getters
    #

    def get_security_error(self, element) -> SecurityError | None:
        """Return a SecurityError if the specified securityData element has one, else return None.
        """
        if element.name() != Name.SECURITY_DATA:
            return
        assert not element.isArray()
        if Name.SECURITY_ERROR in element:
            secid = self.get_subelement_value(element, Name.SECURITY)
            error = self.as_security_error(element.getElement(Name.SECURITY_ERROR), secid)
            return error

    def get_field_errors(self, element) -> list[FieldError]:
        """Return a list of FieldErrors if the specified securityData element has field errors.
        """
        if element.name() != Name.SECURITY_DATA:
            return []
        assert not element.isArray()
        if Name.FIELD_EXCEPTIONS in element:
            secid = self.get_subelement_value(element, Name.SECURITY)
            errors = self.as_field_error(element.getElement(Name.FIELD_EXCEPTIONS), secid)
            return errors
        return []

    def as_security_error(self, element, secid) -> SecurityError | None:
        """Convert the securityError element to a SecurityError.
        """
        if element.name() != Name.SECURITY_ERROR:
            return
        cat = self.get_subelement_value(element, Name.CATEGORY)
        msg = self.get_subelement_value(element, Name.MESSAGE)
        subcat = self.get_subelement_value(element, Name.SUBCATEGORY)
        return SecurityError(security=secid, category=cat, message=msg, subcategory=subcat)

    def as_field_error(self, element, secid) -> FieldError | list[FieldError]:
        """Convert a fieldExceptions element to a FieldError or FieldError array.
        """
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

    def _sequence_as_dataframe(self, elements: blpapi.Element) -> pd.DataFrame:
        """One row per sequence entry, one column per subelement name.

        Parameters
        ----------
        elements : blpapi.Element
            Element of DataType SEQUENCE.

        Returns
        -------
        pd.DataFrame
            Columns in first-seen order across all rows. A row that omits
            an optional field holds None there, which pandas stores as NaN
            or NaT in a numeric or datetime column.
        """
        rows = [
            {str(subelement.name()): self.element_as_value(subelement)
             for subelement in element.elements()}
            for element in elements.values()
            ]
        cols = list(dict.fromkeys(name for row in rows for name in row))
        data = {name: [row.get(name) for row in rows] for name in cols}
        if self.include_ticker_field:
            data['ticker'] = None
            data['field'] = None
        return pd.DataFrame(data, columns=cols)

    def _sequence_as_json(self, elements: blpapi.Element) -> str:
        """JSON list of one {name: value} object per subelement.

        Parameters
        ----------
        elements : blpapi.Element
            Element of DataType SEQUENCE.

        Returns
        -------
        str
            JSON text, or '' for an empty sequence. Only a subelement
            Bloomberg types as a number goes through round_digit_string.
        """
        data = []
        for element in elements.values():
            for subelement in element.elements():
                value = subelement.getValueAsString()
                if subelement.datatype() in NUMERIC_TYPES:
                    value = round_digit_string(value, self.decimal_places)
                d = {str(subelement.name()): value.strip()}
                data += [d]
        return json.dumps(data) if data else ''


class SubscriptionParser(Parser):
    """Parser that dates a time-only value at its latest past occurrence.

    A subscription sends a time with no date only as a last-update stamp.
    A stamp up to CLOCK_SKEW_ALLOWANCE ahead of this machine's clock keeps
    that instant. A stamp a day or more old still lands in the last day.
    """

    def _assign_date(self, time_of_day: datetime.time) -> DateTime:
        """Latest instant of time_of_day up to CLOCK_SKEW_ALLOWANCE past now.
        """
        latest = DateTime.now(self.assumed_timezone) + CLOCK_SKEW_ALLOWANCE
        time_in_zone = Time.instance(time_of_day).replace(tzinfo=self.assumed_timezone)
        stamp = DateTime.combine(latest.date(), time_in_zone, self.assumed_timezone)
        # Python compares same-zone datetimes by wall time, which is wrong
        # across a fall-back change.
        if stamp.in_timezone(UTC) > latest.in_timezone(UTC):
            return stamp.subtract(days=1)
        return stamp
