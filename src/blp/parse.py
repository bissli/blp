import contextlib
import datetime
import json
import logging
from collections import defaultdict, namedtuple

import blpapi
import numpy as np
import pandas as pd
import pytz
from blpapi.datatype import DataType

import libb

logger = logging.getLogger(__name__)


class NameType(type):
    def __getattribute__(cls, name):
        _name = libb.underscore_to_camelcase(name)
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

UTC = pytz.timezone('UTC')

NUMERIC_TYPES = (
    DataType.BOOL, DataType.CHAR, DataType.BYTE, DataType.INT32,
    DataType.INT64, DataType.FLOAT32, DataType.FLOAT64, DataType.BYTEARRAY,
    DataType.DECIMAL
)


class Parser:
    """Interpreter class for Bloomberg Events

    One Event -> one or more Message -> one or more Element

    """

    #
    # iterator wrappers to handle errors in elements
    #

    @staticmethod
    def security_element_iter(elements):
        """Provide a security data iterator by returning a tuple of (Element, SecurityError) which are mutually exclusive"""
        if elements.name() != Name.SECURITY_DATA:
            return None, None
        assert elements.isArray()
        for element in elements.values():  # same as element_iter
            err = Parser.get_security_error(element)
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

    @staticmethod
    def get_subelement_value(element, name, force_string=False):
        """Return the value of the child element with name in the parent Element"""
        if name not in element:
            logger.error(f'Failed to find child element {name} in parent {element}')
            return np.nan
        return Parser.element_as_value(element.getElement(name), force_string)

    @staticmethod
    def get_subelement_values(element, names, force_string=False):
        """Return a list of values for the specified child fields. If field not in Element then replace with nan."""
        return [Parser.get_subelement_value(element, name, force_string) for name in names]

    @staticmethod
    def element_as_value(element, force_string=False):
        """Convert the specified element as a python value"""
        typ = element.datatype()
        if typ == DataType.SEQUENCE:
            if not force_string:
                with contextlib.suppress(blpapi.exception.UnsupportedOperationException):
                    return Parser._sequence_as_dataframe(element)
            return Parser._sequence_as_json(element)
        if force_string:
            return libb.round_digit_string(element.getValueAsString())
        if typ in NUMERIC_TYPES:
            return element.getValue() or np.nan
        if typ in {DataType.DATE, DataType.DATETIME, DataType.TIME}:
            if element.isNull():
                return pd.NaT
            v = element.getValue()
            if isinstance(v, datetime.date):
                return v
            if isinstance(v, datetime.datetime):
                return v.astimezone(UTC)
            if isinstance(v, datetime.time):
                t = datetime.date.today()
                dt = datetime.datetime(t.year, t.month, t.day, v.hour, v.minute, v.second)
                return dt.astimezone(UTC)
        if typ == DataType.CHOICE:
            logger.error('CHOICE data type needs implemented')
        return libb.round_digit_string(element.getValueAsString())

    #
    # error getters
    #

    @staticmethod
    def get_security_error(element) -> SecurityError | None:
        """Return a SecurityError if the specified securityData element has one, else return None"""
        if element.name() != Name.SECURITY_DATA:
            return
        assert not element.isArray()
        if Name.SECURITY_ERROR in element:
            secid = Parser.get_subelement_value(element, Name.SECURITY)
            error = Parser._as_security_error(element.getElement(Name.SECURITY_ERROR), secid)
            return error

    @staticmethod
    def get_field_errors(element) -> list[FieldError] | None:
        """Return a list of FieldErrors if the specified securityData element has field errors"""
        if element.name() != Name.SECURITY_DATA:
            return []
        assert not element.isArray()
        if Name.FIELD_EXCEPTIONS in element:
            secid = Parser.get_subelement_value(element, Name.SECURITY)
            errors = Parser._as_field_error(element.getElement(Name.FIELD_EXCEPTIONS), secid)
            return errors
        return []

    #
    # private methods
    #

    @staticmethod
    def _sequence_as_dataframe(elements):
        """Convert an element with DataType Sequence to a DataFrame."""
        data = defaultdict(list)
        cols = []
        for i, element in enumerate(elements.values()):
            if i == 0:  # Get the ordered cols and assume they are constant
                cols = [str(_.name()) for _ in element.elements()]
            for cidx, _ in enumerate(element.elements()):
                element = element.getElement(cidx)
                data[str(element.name())].append(Parser.element_as_value(element))
        return pd.DataFrame(data, columns=cols)

    @staticmethod
    def _sequence_as_json(elements):
        data = []
        for k, _ in enumerate(elements.values()):
            element = elements.getValueAsElement(k)
            for subelement in element.elements():
                d = {str(subelement.name()): libb.round_digit_string(subelement.getValueAsString())}
                data += [d]
        return json.dumps(data) if data else ''

    @staticmethod
    def _as_security_error(element, secid):
        """Convert the securityError element to a SecurityError"""
        if element.name() != Name.SECURITY_ERROR:
            return
        cat = Parser.get_subelement_value(element, Name.CATEGORY)
        msg = Parser.get_subelement_value(element, Name.MESSAGE)
        subcat = Parser.get_subelement_value(element, Name.SUBCATEGORY)
        return SecurityError(security=secid, category=cat, message=msg, subcategory=subcat)

    @staticmethod
    def _as_field_error(element, secid):
        """Convert a fieldExceptions element to a FieldError or FieldError array"""
        if element.name() != Name.FIELD_EXCEPTIONS:
            return []
        if element.isArray():
            return [Parser._as_field_error(_, secid) for _ in element.values()]
        fld = Parser.get_subelement_value(element, Name.FIELD_ID)
        info = element.getElement(Name.ERROR_INFO)
        cat = Parser.get_subelement_value(info, Name.CATEGORY)
        msg = Parser.get_subelement_value(info, Name.MESSAGE)
        subcat = Parser.get_subelement_value(info, Name.SUBCATEGORY)
        return FieldError(security=secid, field=fld, category=cat, message=msg, subcategory=subcat)
