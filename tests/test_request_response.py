import datetime
import re
from unittest.mock import MagicMock, patch

import blpapi
import pytest
from blpapi.datatype import DataType
from opendate import UTC, Date, DateTime

from blp.client import Blp, EQSRequest, HistoricalDataRequest
from blp.client import RequestFieldError, RequestSecurityError, ResponseError
from blp.client import SessionError
from blp.parse import FieldError, Parser, SecurityError


def make_node(
    name: str,
    value: str | float | None = None,
    children: list | None = None,
    entries: list | None = None) -> MagicMock:
    """Stand-in blpapi.Element, since blpapi ships no //blp/refdata schema.

    Parameters
    ----------
    name : str
        Returned by name().
    value : str or float, optional
        A str or float leaf value. None for a non-leaf node.
    children : list, optional
        Child nodes, reached by name through `in` and getElement.
    entries : list, optional
        Array entries returned by values(). Given, the node is an array.
    """
    child_by_name = {str(child.name()): child for child in children or []}
    node = MagicMock()
    node.name.return_value = blpapi.Name(name)
    node.isArray.return_value = entries is not None
    node.values.return_value = entries or []
    node.elements.return_value = children or []
    node.__contains__.side_effect = lambda key: str(key) in child_by_name
    node.getElement.side_effect = lambda key: child_by_name[str(key)]
    node.isNull.return_value = False
    if isinstance(value, float):
        node.datatype.return_value = DataType.FLOAT64
    else:
        node.datatype.return_value = DataType.STRING
    node.getValue.return_value = value
    node.getValueAsString.return_value = str(value)
    return node


def make_event(*messages: MagicMock) -> MagicMock:
    """Stand-in event that yields messages on each iteration.
    """
    event = MagicMock()
    event.__iter__.side_effect = lambda: iter(messages)
    return event


def make_field_exceptions(field: str) -> MagicMock:
    """FieldExceptions array with one BAD_FLD entry for field.
    """
    error_info = make_node('errorInfo', children=[
        make_node('category', 'BAD_FLD'),
        make_node('message', 'Field not valid'),
        make_node('subcategory', 'INVALID_FIELD'),
        ])
    entry = make_node('fieldExceptions', children=[make_node('fieldId', field), error_info])
    return make_node('fieldExceptions', entries=[entry])


def make_historical_message(field_exceptions: MagicMock | None = None) -> MagicMock:
    """HistoricalDataResponse message for 'IBM US Equity' with one PX_LAST bar.
    """
    bar = make_node('fieldData', children=[
        make_node('date', '2024-01-02'),
        make_node('PX_LAST', 161.5),
        ])
    children = [
        make_node('security', 'IBM US Equity'),
        make_node('fieldData', entries=[bar]),
        ]
    if field_exceptions is not None:
        children.append(field_exceptions)
    security_data = make_node('securityData', children=children)
    return make_node('HistoricalDataResponse', children=[security_data])


def test_screener_response_fills_rows():
    """Verify EQSRequest.process_response stores one row per security.

    Mutation: the call to the Parser.security_iter name Parser no longer
        defines, which raises AttributeError on every response.
    Oracle: a hand-built securityData array of two securities.
    """
    rows = [
        make_node('securityData', children=[
            make_node('security', ticker),
            make_node('fieldData', children=[make_node('Ticker', ticker.split()[0])]),
            ])
        for ticker in ('IBM US Equity', 'AAPL US Equity')
        ]
    data = make_node('data', children=[make_node('securityData', entries=rows)])
    request = EQSRequest('Screen')
    request.prepare_response()
    request.process_response(make_event(make_node('BeqsResponse', children=[data])), True)
    assert request.response.as_dict() == {
        'IBM US Equity': {'Ticker': 'IBM'},
        'AAPL US Equity': {'Ticker': 'AAPL'},
        }


def test_historical_field_error_raises():
    """Verify a historical fieldExceptions entry raises under raise_field_error.

    Mutation: the field errors of a historical securityData element
        dropped, so a bad field passes in silence.
    Oracle: one hand-built BAD_FLD entry for the field BAD_FIELD.
    """
    request = HistoricalDataRequest(
        'IBM US Equity',
        ['PX_LAST', 'BAD_FIELD'],
        raise_field_error=True)
    request.prepare_response()
    message = make_historical_message(make_field_exceptions('BAD_FIELD'))
    request.process_response(make_event(message), True)
    assert request.has_exception
    with pytest.raises(RequestFieldError, match='BAD_FIELD'):
        request.raise_exception()


def test_historical_field_error_recorded():
    """Verify a historical fieldExceptions entry is kept without raising.

    Mutation: the field errors of a historical securityData element
        dropped, or raised with raise_field_error off.
    Oracle: the FieldError built from the hand-built entry's values.
    """
    request = HistoricalDataRequest('IBM US Equity', ['PX_LAST', 'BAD_FIELD'])
    request.prepare_response()
    message = make_historical_message(make_field_exceptions('BAD_FIELD'))
    request.process_response(make_event(message), True)
    assert request.field_errors == [FieldError(
        security='IBM US Equity',
        field='BAD_FIELD',
        category='BAD_FLD',
        message='Field not valid',
        subcategory='INVALID_FIELD')]
    assert not request.has_exception
    request.raise_exception()
    assert request.response.as_dict()['IBM US Equity']['PX_LAST'].tolist() == [161.5]


def test_response_error_is_typed():
    """Verify a responseError message raises ResponseError outside SessionError.

    Mutation: a bare Exception raised, or ResponseError under SessionError,
        which a caller would read as a lost session.
    Oracle: a message holding a responseError element.
    """
    message = make_node('Response', children=[make_node('responseError')])
    message.__getitem__.return_value = 'BAD_SEC'
    with pytest.raises(ResponseError, match='^REQUEST FAILED: BAD_SEC$') as excinfo:
        list(Parser.message_iter(make_event(message)))
    assert not isinstance(excinfo.value, SessionError)


@pytest.mark.parametrize(('errors', 'record', 'exc_class'), [
    (
        'security_errors',
        SecurityError('IBM US Equity', 'BAD_SEC', 'Unknown', 'INVALID_SECURITY'),
        RequestSecurityError,
        ),
    (
        'field_errors',
        FieldError('IBM US Equity', 'BAD_FIELD', 'BAD_FLD', 'Field not valid', 'INVALID_FIELD'),
        RequestFieldError,
        ),
    ])
def test_request_errors_are_typed(errors, record, exc_class):
    """Verify raise_exception raises the class for each error kind.

    Mutation: a bare Exception raised, the two classes swapped, or either
        class placed under SessionError.
    Oracle: one hand-built error record per kind; the message is its str.
    """
    request = HistoricalDataRequest(
        'IBM US Equity',
        'PX_LAST',
        raise_security_error=True,
        raise_field_error=True)
    getattr(request, errors).append(record)
    with pytest.raises(exc_class, match=f'^{re.escape(str(record))}$') as excinfo:
        request.raise_exception()
    assert not isinstance(excinfo.value, SessionError)


def sent_dates(start: object, end: object) -> tuple[str, str]:
    """StartDate and endDate a HistoricalDataRequest sends for start and end.
    """
    service = MagicMock()
    HistoricalDataRequest('IBM US Equity', 'PX_LAST', start=start, end=end).create_request(service)
    request = service.createRequest.return_value
    sent = {call.args[0]: call.args[1] for call in request.set.call_args_list}
    return sent['startDate'], sent['endDate']


@pytest.mark.parametrize('value', [
    '2024-01-02T20:00:00-05:00',
    '2024-01-02T08:00+09:00',
    datetime.date(2024, 1, 2),
    datetime.datetime(2024, 1, 2, 23, 0),
    ])
def test_historical_date_in_own_timezone(value):
    """Verify each input sends the calendar date it names in its own zone.

    Mutation: the date taken after conversion to UTC, which sends
        20240103 for the -05:00 input and 20240101 for the +09:00 input.
    Oracle: every input reads January 2 where it was written.
    """
    assert sent_dates(value, value) == ('20240102', '20240102')


def test_historical_unreadable_date_raises():
    """Verify a string with no date in it raises at construction.

    Mutation: Date.parse's None kept as the date, which fails later in
        __repr__ with an AttributeError.
    Oracle: 'garbage', a string Date.parse returns None for.
    """
    with pytest.raises(ValueError, match="'garbage'"):
        HistoricalDataRequest('IBM US Equity', 'PX_LAST', start='garbage', end='2024-01-02')


def test_historical_missing_start_is_day_before_end():
    """Verify a missing start is the day before the end's own date.

    Mutation: the start taken from the end in UTC, which sends 20240102.
    Oracle: the end '2024-01-02T20:00-05:00' reads January 2 in New York.
    """
    assert sent_dates(None, '2024-01-02T20:00:00-05:00') == ('20240101', '20240102')


def test_historical_missing_end_is_host_date():
    """Verify a missing end is the host's date when UTC is a day behind.

    Mutation: the end taken from now(UTC), which sends 20240102 on a host
        east of UTC between local midnight and 00:00 UTC.
    Oracle: a Tokyo host at 08:00 on January 3, 23:00 UTC on January 2.
    """
    with patch('blp.client.Date.today', return_value=Date(2024, 1, 3)), \
            patch('blp.client.DateTime.now', return_value=DateTime(2024, 1, 2, 23, tzinfo=UTC)):
        assert sent_dates(None, None) == ('20240102', '20240103')


def test_reference_data_passes_string_options():
    """Verify get_reference_data hands force_string and time_as_datetime to
    the request and sends neither to Bloomberg as an override.

    Mutation: either keyword named in the signature but not passed on.
    Oracle: the ReferenceDataRequest a recording execute receives.
    """
    client = Blp(session=MagicMock(), skip_test=True)
    with patch.object(Blp, 'execute') as execute:
        client.get_reference_data('IBM US Equity', 'PX_LAST', force_string=True, time_as_datetime=True)
    request = execute.call_args.args[0]
    assert request.force_string
    assert request.parser.time_as_datetime
    assert request.overrides == {}
