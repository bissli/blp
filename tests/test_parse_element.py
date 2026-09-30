import json
from unittest.mock import MagicMock

import pandas as pd
import pytest
from blpapi.datatype import DataType

from blp.parse import Parser


def make_element(dtype: int, text: str, name: str = 'VALUE') -> MagicMock:
    """Mock non-null element whose getValueAsString returns text.

    Parameters
    ----------
    dtype : int
        A blpapi DataType constant, returned by datatype().
    text : str
        Returned by getValueAsString().
    name : str, default 'VALUE'
        Returned by name(), the key in bulk-field JSON.

    Returns
    -------
    MagicMock
        Stand-in for a blpapi.Element.
    """
    element = MagicMock()
    element.datatype.return_value = dtype
    element.isNull.return_value = False
    element.getValueAsString.return_value = text
    element.name.return_value = name
    return element


@pytest.mark.parametrize('force_string', [True, False])
@pytest.mark.parametrize('cusip', ['046433108', '36831E108'])
def test_string_identifier_kept_verbatim(cusip, force_string):
    """Verify a STRING element that parses as a number comes back unchanged.

    Mutation: the NUMERIC_TYPES test dropped from the force_string branch,
        or round_digit_string put back on the fallback return.
    Oracle: the literal input CUSIPs.
    """
    element = make_element(DataType.STRING, cusip)
    assert Parser().element_as_value(element, force_string) == cusip


@pytest.mark.parametrize('force_string', [True, False])
def test_string_value_stripped_and_not_rounded(force_string):
    """Verify a STRING element loses surrounding whitespace and keeps its digits.

    Mutation: .strip() dropped from the STRING path, or round_digit_string
        put back on it, which returns '1.5'.
    Oracle: hand-trimmed ' 1.50 ' -> '1.50'.
    """
    element = make_element(DataType.STRING, ' 1.50 ')
    assert Parser().element_as_value(element, force_string) == '1.50'


def test_numeric_element_still_rounded_under_force_string():
    """Verify a FLOAT64 element under force_string still goes through rounding.

    Mutation: rounding dropped for every type, so '7283.0' keeps its '.0'.
    Oracle: hand-computed '7283.0' -> '7283' and '1.23456' -> '1.23'.
    """
    parser = Parser(decimal_places=2)
    whole = make_element(DataType.FLOAT64, '7283.0')
    fraction = make_element(DataType.FLOAT64, '1.23456')
    assert parser.element_as_value(whole, True) == '7283'
    assert parser.element_as_value(fraction, True) == '1.23'


def test_bulk_field_json_rounds_only_numeric_subelements():
    """Verify bulk-field JSON keeps a string subelement and rounds a number.

    Mutation: the NUMERIC_TYPES test dropped from _sequence_as_json, which
        turns the CUSIP into '46433108', or rounding dropped, which leaves
        '0.2500'.
    Oracle: the literal CUSIP, and hand-computed '0.2500' -> '0.25'.
    """
    row = MagicMock()
    row.elements.return_value = [
        make_element(DataType.STRING, '046433108', 'ID_CUSIP'),
        make_element(DataType.FLOAT64, '0.2500', 'AMOUNT'),
        ]
    sequence = MagicMock()
    sequence.datatype.return_value = DataType.SEQUENCE
    sequence.values.return_value = [row]
    result = json.loads(Parser().element_as_value(sequence, True))
    assert result == [{'ID_CUSIP': '046433108'}, {'AMOUNT': '0.25'}]


def test_bulk_field_dataframe_rows_with_different_fields():
    """Verify a bulk field whose rows omit different optional fields.

    Mutation: columns taken from the first row only, which raises
        ValueError on the short AMOUNT column or drops CURRENCY, first
        seen in the last row.
    Oracle: hand-built frame, None where a row omits a field.
    """
    fields_by_row = [
        [('ID_CUSIP', '046433108'), ('AMOUNT', '0.25')],
        [('ID_CUSIP', '009066101')],
        [('ID_CUSIP', '00164V103'), ('AMOUNT', '0.30'), ('CURRENCY', 'USD')],
        ]
    rows = []
    for fields in fields_by_row:
        row = MagicMock()
        row.elements.return_value = [
            make_element(DataType.STRING, text, name) for name, text in fields
            ]
        rows.append(row)
    sequence = MagicMock()
    sequence.datatype.return_value = DataType.SEQUENCE
    sequence.values.return_value = rows
    result = Parser().element_as_value(sequence)
    expected = pd.DataFrame({
        'ID_CUSIP': ['046433108', '009066101', '00164V103'],
        'AMOUNT': ['0.25', None, '0.30'],
        'CURRENCY': [None, None, 'USD'],
        })
    pd.testing.assert_frame_equal(result, expected)
