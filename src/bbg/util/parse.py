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
    [Name.SECURITY, Name.CATEGORY, Name.MESSAGE, Name.SUBCATEGORY],
)
FieldError = namedtuple(
    Name.FIELD_ERROR,
    [Name.SECURITY, Name.FIELD, Name.CATEGORY, Name.MESSAGE, Name.SUBCATEGORY],
)

UTC = pytz.timezone('UTC')
GMT = pytz.timezone('GMT')
EST = pytz.timezone('US/Eastern')

TYPE_MAP = {
    'NUMERIC': (1, 2, 3, 4, 5, 6, 7, 9, 12),  # BOOL, CHAR, BYTE, INT32, INT64, FLOAT32, FLOAT64, BYTEARRAY, DECIMAL)
    'STRING': (8,),
    'DATE': (10,),
    'DATETIME': (11, 13),
    'ENUM': (14,),
    'SEQUENCE': (15,),
    'CHOICE': (16,),
}


class Parser:
    """Interpreter class for Bloomberg Events"""

    #
    # iterator wrappers to handle errors in nodes
    #

    @staticmethod
    def security_iter(nodes):
        """Provide a security data iterator by returning a tuple of (Element, SecurityError) which are mutually exclusive"""
        if nodes.name() != Name.SECURITY_DATA:
            return None, None
        assert nodes.isArray()
        for node in nodes.values():
            err = Parser.get_security_error(node)
            result = (None, err) if err else (node, None)
            yield result

    @staticmethod
    def node_iter(nodes):
        yield from nodes.values() if nodes.isArray() else []

    @staticmethod
    def message_iter(event):
        """Provide a message iterator which checks for a response error prior to returning"""
        for msg in event:
            if Name.RESPONSE_ERROR in msg:
                raise Exception(f'REQUEST FAILED: {str(msg[Name.RESPONSE_ERROR])}')
            yield msg

    #
    # value getters
    #

    @staticmethod
    def get_sequence_value(nodes):
        """Convert an element with DataType Sequence to a DataFrame.
        Assume that bulk data is always a table
        """
        assert nodes.datatype() in TYPE_MAP['SEQUENCE']
        data = defaultdict(list)
        cols = []
        for i, node in enumerate(nodes.values()):
            if i == 0:  # Get the ordered cols and assume they are constant
                cols = [str(_.name()) for _ in node.elements()]
            for cidx, _ in enumerate(node.elements()):
                col = node.getElement(cidx)
                data[str(col.name())].append(Parser.as_value(col))
        return pd.DataFrame(data, columns=cols)

    @staticmethod
    def as_value(el):
        """Convert the specified element as a python value"""
        try:
            v = el.getValue()
        except:
            v = None
        typ = el.datatype()
        if typ in TYPE_MAP['NUMERIC']:
            return v or np.nan
        if typ in TYPE_MAP['STRING']:
            return str(v)
        if typ in TYPE_MAP['DATE']:
            if el.isNull() or not v:
                return pd.NaT
            dt = datetime(year=v.year, month=v.month, day=v.day)
            return dt.astimezone(EST)
        if typ in TYPE_MAP['DATETIME']:
            if el.isNull() or not v:
                return pd.NaT
            now = datetime.now()
            dt = datetime(year=now.year, month=now.month, day=now.day, hour=v.hour, minute=v.minute, second=v.second)
            return dt.astimezone(EST)
        if typ in TYPE_MAP['ENUM']:
            return str(v)
        if typ in TYPE_MAP['CHOICE']:
            raise NotImplementedError('CHOICE data type needs implemented')
        if typ in TYPE_MAP['SEQUENCE']:
            return Parser.get_sequence_value(el)
        raise NotImplementedError(f'Unexpected data type {dtype}. Check documentation')

    @staticmethod
    def get_child_value(parent, name):
        """Return the value of the child element with name in the parent Element"""
        if name not in parent:
            logger.error(f'Failed to find child element {name} in parent {parent}')
            return np.nan
        return Parser.as_value(parent.getElement(name))

    @staticmethod
    def get_child_values(parent, names):
        """Return a list of values for the specified child fields. If field not in Element then replace with nan."""
        return [Parser.get_child_value(parent, name) for name in names]

    @staticmethod
    def as_security_error(node, secid):
        """Convert the securityError element to a SecurityError"""
        if node.name() != Name.SECURITY_ERROR:
            return
        cat = Parser.get_child_value(node, Name.CATEGORY)
        msg = Parser.get_child_value(node, Name.MESSAGE)
        subcat = Parser.get_child_value(node, Name.SUBCATEGORY)
        return SecurityError(security=secid, category=cat, message=msg, subcategory=subcat)

    @staticmethod
    def as_field_error(node, secid):
        """Convert a fieldExceptions element to a FieldError or FieldError array"""
        if node.name() != Name.FIELD_EXCEPTIONS:
            return []
        if node.isArray():
            return [Parser.as_field_error(_, secid) for _ in node.values()]
        fld = Parser.get_child_value(node, Name.FIELD_ID)
        info = node.getElement(Name.ERROR_INFO)
        cat = Parser.get_child_value(info, Name.CATEGORY)
        msg = Parser.get_child_value(info, Name.MESSAGE)
        subcat = Parser.get_child_value(info, Name.SUBCATEGORY)
        return FieldError(security=secid, field=fld, category=cat, message=msg, subcategory=subcat)

    @staticmethod
    def get_security_error(node):
        """Return a SecurityError if the specified securityData element has one, else return None"""
        if node.name() != Name.SECURITY_DATA:
            return
        assert not node.isArray()
        if Name.SECURITY_ERROR in node:
            secid = Parser.get_child_value(node, Name.SECURITY)
            error = Parser.as_security_error(node.getElement(Name.SECURITY_ERROR), secid)
            return error

    @staticmethod
    def get_field_errors(node):
        """Return a list of FieldErrors if the specified securityData element has field errors"""
        if node.name() != Name.SECURITY_DATA:
            return []
        assert not node.isArray()
        if Name.FIELD_EXCEPTIONS in node:
            secid = Parser.get_child_value(node, Name.SECURITY)
            errors = Parser.as_field_error(node.getElement(Name.FIELD_EXCEPTIONS), secid)
            return errors
        return []
