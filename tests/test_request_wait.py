import gc
import weakref
from unittest.mock import MagicMock, patch

import blpapi
import blpapi.test
import pytest
from blpapi.event import Event

import blp.client
from blp.client import BaseRequest, Blp, RequestTimeoutError, SessionError
from blp.client import SessionNotAvailableError

OWN_ID = blpapi.CorrelationId(1)
STALE_ID = blpapi.CorrelationId(2)


def make_response_event(
    event_type: int,
    correlation_id: blpapi.CorrelationId) -> MagicMock:
    """Stand-in RESPONSE or PARTIAL_RESPONSE event with one message for
    correlation_id, since blpapi ships no //blp/refdata schema to build one.
    """
    message = MagicMock()
    message.correlationIds.return_value = [correlation_id]
    event = MagicMock()
    event.eventType.return_value = event_type
    event.__iter__.side_effect = lambda: iter([message])
    return event


def make_failure_event(
    correlation_id: blpapi.CorrelationId,
    category: str) -> blpapi.Event:
    """Real REQUEST_STATUS event holding one RequestFailure of category.
    """
    event = blpapi.test.createEvent(Event.REQUEST_STATUS)
    definition = blpapi.test.getAdminMessageDefinition(blpapi.Name('RequestFailure'))
    properties = blpapi.test.MessageProperties()
    properties.setCorrelationIds([correlation_id])
    blpapi.test.appendMessage(event, definition, properties).formatMessageDict(
        {'reason': {'category': category, 'description': 'test failure'}})
    return event


class FakeClock:
    """Clock that advances 0.5 s per nextEvent call, as a quiet session
    does, and returns each scheduled event once it falls due.

    Parameters
    ----------
    schedule : list[tuple]
        (seconds after send, event) pairs.
    """

    def __init__(self, schedule: list[tuple]) -> None:
        self.now = 0.0
        self.schedule = sorted(schedule, key=lambda item: item[0])

    def next_event(self, timeout: int) -> MagicMock:
        """The next due event, else a TIMEOUT event after timeout ms.
        Raises AssertionError past 1000 s, so a wait with no deadline fails
        its test instead of hanging the suite.
        """
        assert self.now < 1000, 'request wait never ended'
        if self.schedule and self.schedule[0][0] <= self.now:
            return self.schedule.pop(0)[1]
        self.now += timeout / 1000
        event = MagicMock()
        event.eventType.return_value = Event.TIMEOUT
        event.__iter__.side_effect = lambda: iter([])
        return event


def run_request(clock: FakeClock) -> MagicMock:
    """Run Blp.execute on a mock session timed by clock; return the mock
    request so a test can read its process_response calls.
    """
    session = MagicMock()
    session.sendRequest.return_value = OWN_ID
    session.nextEvent.side_effect = clock.next_event
    request = MagicMock(has_exception=False)
    request.on_admin_event.side_effect = (
        lambda event: BaseRequest.on_admin_event(request, event))
    fake_time = MagicMock()
    fake_time.monotonic.side_effect = lambda: clock.now
    with patch.object(blp.client, 'time', fake_time):
        Blp(session=session, skip_test=True).execute(request)
    return request


def test_silent_server_raises_timeout_at_idle_limit():
    """Verify a request with no reply raises RequestTimeoutError just past
    120 s instead of looping forever.

    Mutation: the deadline check dropped, or a limit other than 120 s.
    Oracle: the fake clock reads 120.5 at the raise, the first 0.5 s tick
        past the limit.
    """
    clock = FakeClock([])
    with pytest.raises(RequestTimeoutError):
        run_request(clock)
    assert clock.now == 120.5


def test_partial_response_restarts_idle_limit():
    """Verify a partial at 100 s lets a RESPONSE at 210 s complete.

    Mutation: the deadline not reset on PARTIAL_RESPONSE.
    Oracle: 210 s is past one 120 s limit but 110 s after the partial.
    """
    clock = FakeClock([
        (100, make_response_event(Event.PARTIAL_RESPONSE, OWN_ID)),
        (210, make_response_event(Event.RESPONSE, OWN_ID)),
        ])
    request = run_request(clock)
    calls = request.process_response.call_args_list
    assert [call.kwargs['is_final'] for call in calls] == [False, True]


def test_stale_partial_does_not_restart_idle_limit():
    """Verify a partial for another request leaves the deadline alone.

    Mutation: the deadline reset before the correlation id check.
    Oracle: a stale partial at 100 s; the raise still comes at 120.5 s.
    """
    clock = FakeClock([(100, make_response_event(Event.PARTIAL_RESPONSE, STALE_ID))])
    with pytest.raises(RequestTimeoutError):
        run_request(clock)
    assert clock.now == 120.5


def test_stale_response_is_skipped():
    """Verify a late RESPONSE from an abandoned request neither ends nor
    feeds the current request.

    Mutation: the correlation id filter dropped, so the stale RESPONSE
        ends the request.
    Oracle: the request processes exactly its own RESPONSE event.
    """
    own = make_response_event(Event.RESPONSE, OWN_ID)
    clock = FakeClock([(1, make_response_event(Event.RESPONSE, STALE_ID)), (2, own)])
    request = run_request(clock)
    assert request.process_response.call_args_list[0].args == (own,)
    assert request.process_response.call_count == 1


@pytest.mark.parametrize('category', ['CONNECTION_DEAD', 'SERVICE_NOT_AVAILABLE'])
def test_connection_failure_raises_session_error(category):
    """Verify a RequestFailure for a dead connection raises a SessionError.

    Mutation: REQUEST_STATUS ignored (the old hang), or the category set
        missing a member.
    Oracle: real RequestFailure events built from the admin schema.
    """
    clock = FakeClock([(1, make_failure_event(OWN_ID, category))])
    with pytest.raises(SessionNotAvailableError):
        run_request(clock)
    assert clock.now == 1


def test_bad_request_failure_is_not_session_error():
    """Verify a RequestFailure for a bad request raises a plain error, so
    a caller that treats SessionError as a dead session keeps going.

    Mutation: every RequestFailure raised as a SessionError.
    Oracle: a real BAD_ARGS RequestFailure event.
    """
    clock = FakeClock([(1, make_failure_event(OWN_ID, 'BAD_ARGS'))])
    with pytest.raises(RuntimeError, match='BAD_ARGS') as excinfo:
        run_request(clock)
    assert not isinstance(excinfo.value, SessionError)


def test_stale_failure_is_skipped():
    """Verify a RequestFailure for another request does not fail this one.

    Mutation: REQUEST_STATUS handled before the correlation id check.
    Oracle: a stale CONNECTION_DEAD failure, then this request's RESPONSE.
    """
    clock = FakeClock([
        (1, make_failure_event(STALE_ID, 'CONNECTION_DEAD')),
        (2, make_response_event(Event.RESPONSE, OWN_ID)),
        ])
    request = run_request(clock)
    assert request.process_response.call_count == 1


def test_failed_connectivity_test_stops_own_session():
    """Verify a failed connectivity test stops the session Blp created.

    Mutation: the cleanup call dropped from __post_init__.
    Oracle: a recording mock session from a patched SessionFactory.
    """
    session = MagicMock()
    with patch.object(blp.client.SessionFactory, 'create', return_value=session), \
            patch.object(Blp, 'get_reference_data', side_effect=RuntimeError('down')):
        with pytest.raises(SessionError, match='^Connectivity test failed.$'):
            Blp()
    session.cleanup.assert_called_once()


def test_failed_connectivity_test_leaves_caller_session():
    """Verify a failed connectivity test leaves a caller's session alone.

    Mutation: the ownership guard dropped, so a caller's session is
        destroyed and crashes the process on its next use.
    Oracle: a recording mock session passed in by the caller.
    """
    session = MagicMock()
    with patch.object(Blp, 'get_reference_data', side_effect=RuntimeError('down')):
        with pytest.raises(SessionError):
            Blp(session=session)
    session.cleanup.assert_not_called()


def test_failed_start_cleans_up_session():
    """Verify SessionFactory.create cleans up a session that fails to start.

    Mutation: the cleanup call dropped before SessionCreateError.
    Oracle: a recording mock session whose start() returns False.
    """
    session = MagicMock()
    session.start.return_value = False
    with patch.object(blp.client, 'create_session', return_value=session):
        with pytest.raises(blp.client.SessionCreateError):
            blp.client.SessionFactory.create()
    session.cleanup.assert_called_once()


def test_cleanup_lets_session_be_collected():
    """Verify a cleaned-up Session is garbage collected before exit.

    Mutation: the atexit.unregister call dropped from cleanup, so the
        exit hook keeps every Session alive.
    Oracle: a weakref to a real, never-started blpapi Session.
    """
    session = blp.client.create_session(port=1)
    session_ref = weakref.ref(session)
    session.cleanup()
    del session
    gc.collect()
    assert session_ref() is None


def test_session_terminated_raises_while_waiting():
    """Verify SessionTerminated ends the wait with SessionTerminatedError.

    Mutation: SESSION_STATUS events ignored, so the wait runs to the idle
        limit, or the arm returning partial data instead of raising.
    Oracle: a real SessionTerminated event at 5 s; the raise lands at 5.
    """
    ended = blpapi.test.createEvent(Event.SESSION_STATUS)
    definition = blpapi.test.getAdminMessageDefinition(blpapi.Name('SessionTerminated'))
    blpapi.test.appendMessage(ended, definition, blpapi.test.MessageProperties())
    clock = FakeClock([(5, ended)])
    with pytest.raises(blp.client.SessionTerminatedError):
        run_request(clock)
    assert clock.now == 5
