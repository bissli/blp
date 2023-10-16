import logging
from collections import defaultdict, namedtuple
from datetime import datetime

import blpapi
import numpy as np
import pandas as pd
import pytz

logger = logging.getLogger(__name__)


def underscore_to_camelcase(text):
    """Converts underscore_delimited_text to camelCase"""
    return ''.join(word.title() if i else word.lower() for i, word in enumerate(text.split('_')))


class NameType(type):
    """Blpapi Name class wrapper"""

    def __getattribute__(cls, name):
        _name = underscore_to_camelcase(name)
        return blpapi.Name.findName(_name) or blpapi.Name(_name)


class Name(metaclass=NameType):
    pass


SecurityError = namedtuple(
    Name.SECURITY_ERROR,
    [Name.SECURITY, Name.SOURCE, Name.CODE, Name.CATEGORY, Name.MESSAGE, Name.SUBCATEGORY],
)
FieldError = namedtuple(
    Name.FIELD_ERROR,
    [Name.SECURITY, Name.FIELD, Name.SOURCE, Name.CODE, Name.CATEGORY, Name.MESSAGE, Name.SUBCATEGORY],
)

UTC = pytz.timezone('UTC')
GMT = pytz.timezone('GMT')
EST = pytz.timezone('US/Eastern')


class Parser:
    """Interpreter class for Bloomberg Events"""

    @staticmethod
    def security_iter(nodearr):
        """Provide a security data iterator by returning a tuple of (Element, SecurityError) which are mutually exclusive"""
        assert nodearr.name() == Name.SECURITY_DATA and nodearr.isArray()
        for i in range(nodearr.numValues()):
            node = nodearr.getValue(i)
            err = Parser.get_security_error(node)
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
            # logger.debug(f'Received response to request {msg.getRequestId()}')
            # logger.debug(msg.toString())
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
                data[str(col.name())].append(Parser.as_value(col))
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
            return Parser.get_sequence_value(ele)
        raise NotImplementedError(f'Unexpected data type {dtype}. Check documentation')

    @staticmethod
    def get_child_value(parent, name, allow_missing=0):
        """Return the value of the child element with name in the parent Element"""
        if not parent.hasElement(name):
            if allow_missing:
                return np.nan
            raise Exception(f'failed to find child element {name} in parent')
        return Parser.as_value(parent.getElement(name))

    @staticmethod
    def get_child_values(parent, names):
        """Return a list of values for the specified child fields. If field not in Element then replace with nan."""
        vals = []
        for name in names:
            if parent.hasElement(name):
                vals.append(Parser.as_value(parent.getElement(name)))
            else:
                vals.append(np.nan)
        return vals

    @staticmethod
    def as_security_error(node, secid):
        """Convert the securityError element to a SecurityError"""
        assert node.name() == Name.SECURITY_ERROR
        src = Parser.get_child_value(node, Name.SOURCE)
        code = Parser.get_child_value(node, Name.CODE)
        cat = Parser.get_child_value(node, Name.CATEGORY)
        msg = Parser.get_child_value(node, Name.MESSAGE)
        subcat = Parser.get_child_value(node, Name.SUBCATEGORY)
        return SecurityError(security=secid, source=src, code=code, category=cat, message=msg, subcategory=subcat)

    @staticmethod
    def as_field_error(node, secid):
        """Convert a fieldExceptions element to a FieldError or FieldError array"""
        assert node.name() == Name.FIELD_EXCEPTIONS
        if node.isArray():
            return [Parser.as_field_error(node.getValue(_), secid) for _ in range(node.numValues())]
        fld = Parser.get_child_value(node, Name.FIELD_ID)
        info = node.getElement(Name.ERROR_INFO)
        src = Parser.get_child_value(info, Name.SOURCE)
        code = Parser.get_child_value(info, Name.CODE)
        cat = Parser.get_child_value(info, Name.CATEGORY)
        msg = Parser.get_child_value(info, Name.MESSAGE)
        subcat = Parser.get_child_value(info, Name.SUBCATEGORY)
        return FieldError(
            security=secid, field=fld, source=src, code=code, category=cat, message=msg, subcategory=subcat
        )

    @staticmethod
    def get_security_error(node):
        """Return a SecurityError if the specified securityData element has one, else return None"""
        assert node.name() == Name.SECURITY_DATA and not node.isArray()
        if node.hasElement(Name.SECURITY_ERROR):
            secid = Parser.get_child_value(node, Name.SECURITY)
            err = Parser.as_security_error(node.getElement(Name.SECURITY_ERROR), secid)
            return err

    @staticmethod
    def get_field_errors(node):
        """Return a list of FieldErrors if the specified securityData element has field errors"""
        assert node.name() == Name.SECURITY_DATA and not node.isArray()
        nodearr = node.getElement(Name.FIELD_EXCEPTIONS)
        if nodearr.numValues() > 0:
            secid = Parser.get_child_value(node, Name.SECURITY)
            errors = Parser.as_field_error(nodearr, secid)
            return errors
