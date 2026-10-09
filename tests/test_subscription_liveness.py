import threading
from unittest.mock import MagicMock, patch

import blpapi
import blpapi.test
import pytest
from blpapi.event import Event

import blp.client
from blp.client import Subscription, SubscriptionDeadError
from blp.handle import LoggingEventHandler

TOPICS = ['AAA US Equity', 'BBB US Equity']
ALL_DEAD = '^all 2 topics failed or were terminated: '


def make_admin_event(event_type: int, messages: list[tuple]) -> blpapi.Event:
    """Real blpapi event built from the admin schema, no Terminal needed.

    Parameters
    ----------
    event_type : int
        A blpapi Event constant, such as Event.SUBSCRIPTION_STATUS.
    messages : list[tuple]
        One (message type, topic or None, body dict) per message.

    Returns
    -------
    blpapi.Event
    """
    event = blpapi.test.createEvent(event_type)
    for message_type, topic, body in messages:
        definition = blpapi.test.getAdminMessageDefinition(blpapi.Name(message_type))
        properties = blpapi.test.MessageProperties()
        if topic is not None:
            properties.setCorrelationIds([blpapi.CorrelationId(topic)])
        blpapi.test.appendMessage(event, definition, properties).formatMessageDict(body)
    return event


def make_status_event(*messages: tuple) -> blpapi.Event:
    """SUBSCRIPTION_STATUS event from (message type, topic, description)
    triples. A description of None leaves the reason empty.
    """
    return make_admin_event(Event.SUBSCRIPTION_STATUS, [
        (message_type, topic, {'reason': {} if desc is None else {'description': desc}})
        for message_type, topic, desc in messages
        ])


def make_data_event(topic: str) -> MagicMock:
    """Stand-in SUBSCRIPTION_DATA event, since blpapi ships no //blp/mktdata
    schema to build a real one.
    """
    message = MagicMock()
    message.correlationId.return_value.value.return_value = topic
    event = MagicMock()
    event.eventType.return_value = Event.SUBSCRIPTION_DATA
    event.__iter__.side_effect = lambda: iter([message])
    return event


class FakeShutdown(threading.Event):
    """Shutdown event whose wait() advances a fake clock and delivers the
    events that fall due.

    Parameters
    ----------
    schedule : list[tuple]
        (seconds after subscribe, event) pairs.
    stop_at : float, optional
        Clock time from which wait() reports the shutdown as set.
    """

    def __init__(self, schedule: list[tuple], stop_at: float | None = None) -> None:
        super().__init__()
        self.now = 0.0
        self.schedule = sorted(schedule, key=lambda item: item[0])
        self.stop_at = stop_at
        self.handler = None

    def wait(self, timeout: float | None = None) -> bool:
        """Advance the clock by timeout; True once stop_at is reached.
        """
        self.now += timeout
        while self.schedule and self.schedule[0][0] <= self.now:
            self.handler(self.schedule.pop(0)[1], None)
        return self.stop_at is not None and self.now >= self.stop_at


def run_subscription(shutdown: FakeShutdown, session: MagicMock | None = None) -> None:
    """Run a 120 s Subscription.subscribe over TOPICS on a mock session,
    timed by shutdown's fake clock.
    """
    session = session or MagicMock()

    def create_session(**kwargs: dict) -> MagicMock:
        shutdown.handler = kwargs['event_handler']
        return session

    fake_time = MagicMock()
    fake_time.monotonic.side_effect = lambda: shutdown.now
    with patch.object(blp.client.SessionFactory, 'create', side_effect=create_session), \
            patch.object(blp.client, 'time', fake_time):
        Subscription(TOPICS, ['LAST_PRICE']).subscribe(
            LoggingEventHandler,
            runtime=120,
            shutdown_event=shutdown)


def test_session_terminated_raises_and_unsubscribes():
    """Verify SessionTerminated raises at once and the session is still
    cleaned up.

    Mutation: matching blp's lower camel Name.SESSION_TERMINATED, dropping
        the failure check, or raising outside the try.
    Oracle: a real SessionTerminated event at 5 s, after data, so the raise
        lands at 5 s.
    """
    ended = make_admin_event(Event.SESSION_STATUS, [('SessionTerminated', None, {})])
    shutdown = FakeShutdown([(1, make_data_event(TOPICS[0])), (5, ended)])
    session = MagicMock()
    with pytest.raises(SubscriptionDeadError, match='^session terminated$'):
        run_subscription(shutdown, session)
    assert shutdown.now == 5
    assert session.unsubscribe.called


def test_all_topics_failed_raises_with_first_reason():
    """Verify failure on every topic raises, naming the first reason.

    Mutation: matching blp's lower camel Name.SUBSCRIPTION_FAILURE, or
        keeping the latest reason.
    Oracle: real failures at 1 s and 2 s; the raise lands at 2 s.
    """
    first = make_status_event(('SubscriptionFailure', TOPICS[0], 'first'))
    second = make_status_event(('SubscriptionFailure', TOPICS[1], 'second'))
    shutdown = FakeShutdown([(1, first), (2, second)])
    with pytest.raises(SubscriptionDeadError, match=ALL_DEAD + 'first$'):
        run_subscription(shutdown)
    assert shutdown.now == 2


def test_all_but_one_failed_runs_to_end():
    """Verify a failure on some topics does not raise.

    Mutation: raising on the first dead topic, or an off-by-one in the
        all-dead count.
    Oracle: one of two topics fails and the other streams; the clock ends
        at runtime.
    """
    failed = make_status_event(('SubscriptionFailure', TOPICS[0], 'bad'))
    shutdown = FakeShutdown([(1, failed), (2, make_data_event(TOPICS[1]))])
    run_subscription(shutdown)
    assert shutdown.now == 120


def test_failed_then_terminated_raises():
    """Verify topics failed at subscribe plus the rest terminated later raise.

    Mutation: tracking only SubscriptionFailure, or matching blp's lower
        camel Name.SUBSCRIPTION_TERMINATED.
    Oracle: the raise lands at the 30 s termination, after data arrived.
    """
    failed = make_status_event(('SubscriptionFailure', TOPICS[0], 'bad'))
    ended = make_status_event(('SubscriptionTerminated', TOPICS[1], 'revoked'))
    shutdown = FakeShutdown([(1, failed), (2, make_data_event(TOPICS[1])), (30, ended)])
    with pytest.raises(SubscriptionDeadError, match=ALL_DEAD + 'bad$'):
        run_subscription(shutdown)
    assert shutdown.now == 30


def test_failure_without_description_counts_every_topic():
    """Verify a failure lacking its optional description still lets the
    rest of the event's topics count as dead.

    Mutation: reading description unguarded, whose exception skips the
        event's second message.
    Oracle: one real event holds both failures, the first with an empty
        reason.
    """
    both = make_status_event(
        ('SubscriptionFailure', TOPICS[0], None),
        ('SubscriptionFailure', TOPICS[1], 'bad'))
    with pytest.raises(SubscriptionDeadError, match=ALL_DEAD + 'no description$'):
        run_subscription(FakeShutdown([(1, both)]))


def test_no_data_within_60s_raises():
    """Verify a run with no data raises at 60 s.

    Mutation: the no-data check dropped, or a threshold other than 60 s.
    Oracle: the fake clock reads 60 at the raise.
    """
    shutdown = FakeShutdown([])
    with pytest.raises(SubscriptionDeadError, match='^no subscription data within 60 s$'):
        run_subscription(shutdown)
    assert shutdown.now == 60


def test_data_at_59s_runs_to_end():
    """Verify data one second inside the limit lets the run finish.

    Mutation: first_data never set, or a threshold under 59 s.
    Oracle: data at 59 s; the clock ends at runtime.
    """
    shutdown = FakeShutdown([(59, make_data_event(TOPICS[0]))])
    run_subscription(shutdown)
    assert shutdown.now == 120


def test_shutdown_event_returns_normally():
    """Verify a caller's shutdown ends the run without error.

    Mutation: the shutdown wait's result ignored.
    Oracle: shutdown at 5 s with no data; the clock ends at 5.
    """
    shutdown = FakeShutdown([], stop_at=5)
    run_subscription(shutdown)
    assert shutdown.now == 5
