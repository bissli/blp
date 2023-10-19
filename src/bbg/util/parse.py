import contextlib
import json
import logging
from collections import defaultdict, namedtuple
from datetime import datetime
from typing import List, Optional

import blpapi
import numpy as np
import pandas as pd
import pytz

logger = logging.getLogger(__name__)


def clean_string_value(v: str) -> str:
    """Clean string comprised of digits"""
    v = v.strip()

    with contextlib.suppress(ValueError):
        v_float = float(v)
        v_int = int(v_float)
        v = v_int if v_float == v_int else round(v_float, 3)
        return str(v)

    return v


def underscore_to_camelcase(text):
    """Converts underscore_delimited_text to camelCase"""
    return ''.join(word.title() if i else word.lower() for i, word in enumerate(text.split('_')))


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

    def __init__(self, values_as_string=False, use_timezone=EST):
        self.values_as_string = values_as_string
        self.tz = use_timezone

    #
    # iterator wrappers to handle errors in nodes
    #

    def security_iter(self, nodes):
        """Provide a security data iterator by returning a tuple of (Element, SecurityError) which are mutually exclusive"""
        if nodes.name() != Name.SECURITY_DATA:
            return None, None
        assert nodes.isArray()
        for node in nodes.values():
            err = self.get_security_error(node)
            result = (None, err) if err else (node, None)
            yield result

    def node_iter(self, nodes):
        yield from nodes.values() if nodes.isArray() else []

    def message_iter(self, event):
        """Provide a message iterator which checks for a response error prior to returning"""
        for msg in event:
            if Name.RESPONSE_ERROR in msg:
                raise Exception(f'REQUEST FAILED: {str(msg[Name.RESPONSE_ERROR])}')
            yield msg

    #
    # value getters
    #

    def get_child_value(self, parent, name):
        """Return the value of the child element with name in the parent Element"""
        if name not in parent:
            logger.error(f'Failed to find child element {name} in parent {parent}')
            return np.nan
        return self.as_value(parent.getElement(name))

    def get_child_values(self, parent, names):
        """Return a list of values for the specified child fields. If field not in Element then replace with nan."""
        return [self.get_child_value(parent, name) for name in names]

    def as_value(self, el):
        """Convert the specified element as a python value"""
        typ = el.datatype()
        if typ in TYPE_MAP['SEQUENCE']:
            if self.values_as_string:
                return self._get_sequence_value_as_json(el)
            else:
                return self._get_sequence_value_as_dataframe(el)
        if self.values_as_string:
            return clean_string_value(el.getValueAsString())
        if typ in TYPE_MAP['NUMERIC']:
            return el.getValue() or np.nan
        if typ in TYPE_MAP['DATE']:
            if el.isNull():
                return pd.NaT
            v = el.getValue()
            dt = datetime(year=v.year, month=v.month, day=v.day)
            return dt.astimezone(self.tz)
        if typ in TYPE_MAP['DATETIME']:
            if el.isNull():
                return pd.NaT
            v = el.getValue()
            now = datetime.now()
            dt = datetime(year=now.year, month=now.month, day=now.day, hour=v.hour, minute=v.minute, second=v.second)
            return dt.astimezone(self.tz)
        if typ in TYPE_MAP['CHOICE']:
            logger.error('CHOICE data type needs implemented')
        return clean_string_value(el.getValueAsString())

    #
    # error getters
    #

    def get_security_error(self, node) -> Optional[SecurityError]:
        """Return a SecurityError if the specified securityData element has one, else return None"""
        if node.name() != Name.SECURITY_DATA:
            return
        assert not node.isArray()
        if Name.SECURITY_ERROR in node:
            secid = self.get_child_value(node, Name.SECURITY)
            error = self._as_security_error(node.getElement(Name.SECURITY_ERROR), secid)
            return error

    def get_field_errors(self, node) -> Optional[List[FieldError]]:
        """Return a list of FieldErrors if the specified securityData element has field errors"""
        if node.name() != Name.SECURITY_DATA:
            return []
        assert not node.isArray()
        if Name.FIELD_EXCEPTIONS in node:
            secid = self.get_child_value(node, Name.SECURITY)
            errors = self._as_field_error(node.getElement(Name.FIELD_EXCEPTIONS), secid)
            return errors
        return []

    #
    # private methods
    #

    def _get_sequence_value_as_dataframe(self, nodes):
        data = defaultdict(list)
        cols = []
        for i, node in enumerate(nodes.values()):
            if i == 0:  # Get the ordered cols and assume they are constant
                cols = [str(_.name()) for _ in node.elements()]
            for cidx, _ in enumerate(node.elements()):
                el = node.getElement(cidx)
                data[str(el.name())].append(self.as_value(el))
        return pd.DataFrame(data, columns=cols)

    def _get_sequence_value_as_json(self, nodes):
        data = []
        for k, _ in enumerate(nodes.values()):
            node = nodes.getValueAsElement(k)
            for subnode in node.elements():
                d = {str(subnode.name()): clean_string_value(subnode.getValueAsString())}
                data += [d]
        return json.dumps(data) if data else ''

    def _as_security_error(self, node, secid):
        """Convert the securityError element to a SecurityError"""
        if node.name() != Name.SECURITY_ERROR:
            return
        cat = self.get_child_value(node, Name.CATEGORY)
        msg = self.get_child_value(node, Name.MESSAGE)
        subcat = self.get_child_value(node, Name.SUBCATEGORY)
        return SecurityError(security=secid, category=cat, message=msg, subcategory=subcat)

    def _as_field_error(self, node, secid):
        """Convert a fieldExceptions element to a FieldError or FieldError array"""
        if node.name() != Name.FIELD_EXCEPTIONS:
            return []
        if node.isArray():
            return [self._as_field_error(_, secid) for _ in node.values()]
        fld = self.get_child_value(node, Name.FIELD_ID)
        info = node.getElement(Name.ERROR_INFO)
        cat = self.get_child_value(info, Name.CATEGORY)
        msg = self.get_child_value(info, Name.MESSAGE)
        subcat = self.get_child_value(info, Name.SUBCATEGORY)
        return FieldError(security=secid, field=fld, category=cat, message=msg, subcategory=subcat)
