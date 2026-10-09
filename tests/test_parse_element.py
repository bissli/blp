import datetime
import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from blpapi.datatype import DataType
from dateutil.tz import gettz
from opendate import UTC, Date, DateTime, Timezone

from blp.handle import LoggingEventHandler
from blp.parse import Parser, SubscriptionParser


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


NEW_YORK = Timezone('America/New_York')
SYDNEY = Timezone('Australia/Sydney')
MORNING = DateTime(2026, 6, 16, 8, 44, tzinfo=NEW_YORK)
FALL_BACK_MORNING = DateTime(2026, 11, 1, 8, 44, tzinfo=NEW_YORK)
MIDDAY = DateTime(2026, 6, 16, 12, 0, tzinfo=NEW_YORK)
LATE_EVENING = DateTime(2026, 6, 16, 22, 30, tzinfo=NEW_YORK)
FALL_BACK_NIGHT = DateTime(2026, 11, 1, 1, 10, tzinfo=NEW_YORK)
NEW_YORK_BEFORE_MIDNIGHT = DateTime(2026, 6, 16, 23, 30, tzinfo=NEW_YORK)
UTC_BEFORE_MIDNIGHT = DateTime(2026, 6, 16, 23, 30, tzinfo=UTC)


def parse_time_only(parser_class, stamp, now, assumed_timezone, as_datetime=True):
    """element_as_value of stamp's time of day, with the clock stopped at now.

    Parameters
    ----------
    parser_class : type[Parser]
        Parser or SubscriptionParser.
    stamp : DateTime
        Its naive time of day in assumed_timezone is the element value.
    now : DateTime
        What DateTime.now returns, in the zone asked for.
    assumed_timezone : Timezone
        Parser assumed_timezone. desired_timezone is New York.
    as_datetime : bool, default True
        Parser time_as_datetime.

    Returns
    -------
    DateTime or Time
    """
    element = make_element(DataType.DATETIME, '')
    local_stamp = stamp.in_timezone(assumed_timezone)
    element.getValue.return_value = local_stamp.time().replace(tzinfo=None)
    parser = parser_class(
        assumed_timezone=assumed_timezone,
        desired_timezone=NEW_YORK,
        time_as_datetime=as_datetime)
    with patch('blp.parse.DateTime.now', side_effect=now.in_timezone):
        return parser.element_as_value(element)


@pytest.mark.parametrize('assumed_timezone', [
    UTC,
    NEW_YORK,
    SYDNEY,
    gettz('Australia/Sydney'),
    ])
@pytest.mark.parametrize(('now', 'stamp'), [
    (MORNING, DateTime(2026, 6, 15, 21, 40, 51, tzinfo=NEW_YORK)),
    (MORNING, DateTime(2026, 6, 16, 3, 41, 56, tzinfo=NEW_YORK)),
    (MORNING, MORNING.add(minutes=59)),
    (MORNING, MORNING.add(hours=1)),
    (MORNING, MORNING.add(minutes=61).subtract(days=1)),
    (FALL_BACK_MORNING, DateTime(2026, 10, 31, 21, 40, 51, tzinfo=NEW_YORK)),
    (MIDDAY, MIDDAY.subtract(hours=1)),
    (LATE_EVENING, DateTime(2026, 6, 16, 22, 0, tzinfo=NEW_YORK)),
    (FALL_BACK_NIGHT, DateTime(2026, 11, 1, 1, 30, tzinfo=NEW_YORK)),
    (NEW_YORK_BEFORE_MIDNIGHT, NEW_YORK_BEFORE_MIDNIGHT.add(minutes=40)),
    (UTC_BEFORE_MIDNIGHT, UTC_BEFORE_MIDNIGHT.add(minutes=40)),
    ])
def test_subscription_time_only_at_latest_past_instant(now, stamp, assumed_timezone):
    """Verify a subscription dates a time-only value at its latest past instant.

    Mutation: now's date, or the UTC date, for the date of now plus the
        allowance, the previous-day step dropped, >= for >, a wall-time
        comparison, subtract(hours=24) for subtract(days=1), another
        allowance, or opendate before 0.1.50, which reads a dateutil zone
        as UTC.
    Oracle: hand-picked instants either side of and at one hour ahead of
        now, fed in as their time of day in assumed_timezone.
    """
    result = parse_time_only(SubscriptionParser, stamp, now, assumed_timezone)
    assert result.isoformat() == stamp.in_timezone(NEW_YORK).isoformat()


@pytest.mark.parametrize(('now', 'stamp', 'assumed_timezone'), [
    (LATE_EVENING, DateTime(2026, 6, 16, 23, 59, 58, tzinfo=UTC), UTC),
    (MORNING, DateTime(2026, 6, 16, 21, 40, 51, tzinfo=NEW_YORK), NEW_YORK),
    ])
def test_request_time_only_takes_machine_date(now, stamp, assumed_timezone):
    """Verify Parser dates a time-only value with the machine's date.

    Mutation: today in assumed_timezone for Date.today(), which dates the
        UTC stamp a day ahead, or the subscription rule, which dates the
        21:40:51 stamp a day early.
    Oracle: a New York machine sees 2026-06-16 at both instants.
    """
    with patch('blp.parse.Date.today', return_value=Date(2026, 6, 16)):
        result = parse_time_only(Parser, stamp, now, assumed_timezone)
    assert result.isoformat() == stamp.in_timezone(NEW_YORK).isoformat()


def test_time_only_as_time_in_desired_timezone():
    """Verify time_as_datetime=False returns the time in desired_timezone.

    Mutation: the in_timezone conversion dropped, which returns 01:40:51.
    Oracle: 2026-11-01 01:40:51Z is 2026-10-31 21:40:51 EDT.
    """
    stamp = DateTime(2026, 10, 31, 21, 40, 51, tzinfo=NEW_YORK)
    result = parse_time_only(SubscriptionParser, stamp, FALL_BACK_MORNING, UTC, False)
    assert result.replace(tzinfo=None) == datetime.time(21, 40, 51)


def test_event_handler_parses_with_subscription_parser():
    """Verify a subscription handler parses with SubscriptionParser.

    Mutation: BaseEventHandler builds a plain Parser.
    Oracle: the class of the handler's parser.
    """
    handler = LoggingEventHandler(['IBM US Equity'], ['TIME'])
    assert type(handler.parser) is SubscriptionParser


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
