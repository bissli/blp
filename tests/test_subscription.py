import threading
from unittest.mock import MagicMock, patch

import blpapi
import blpapi.test
import pytest
from blpapi.event import Event

import blp.client
from blp.client import SessionError, SessionNotAvailableError
from blp.client import SessionStartError, Subscription
from blp.parse import SubscriptionParser

MKTDATA_SCHEMA = """<?xml version="1.0" encoding="UTF-8" ?>
<ServiceDefinition name="blp.mktdata" version="1.0.0.0">
  <service name="//blp/mktdata" version="1.0.0.0">
    <event name="MarketDataEvents" eventType="MarketDataUpdate">
      <cacheable>true</cacheable>
      <cachingMode>full</cachingMode>
    </event>
    <defaultServiceId>1</defaultServiceId>
    <publisherSupportsRecap>true</publisherSupportsRecap>
    <authoritativeSourceSupportsRecap>true</authoritativeSourceSupportsRecap>
    <isInfrastructureService>false</isInfrastructureService>
    <isMetered>false</isMetered>
    <appendMtrId>false</appendMtrId>
  </service>
  <schema>
    <sequenceType name="MarketDataUpdate">
      <element name="LAST_PRICE" type="Float64" minOccurs="0" maxOccurs="1"/>
      <element name="BID" type="Float64" minOccurs="0" maxOccurs="1"/>
    </sequenceType>
  </schema>
</ServiceDefinition>
"""
MKTDATA_EVENT = blpapi.test.deserializeService(MKTDATA_SCHEMA).getEventDefinition('MarketDataEvents')
NOT_STARTED = blpapi.InvalidStateException('Session Not Started', 0)


def admin_event(event_type: int, *messages: tuple) -> blpapi.Event:
    """Real blpapi event built from the admin schema.

    Parameters
    ----------
    event_type : int
        A blpapi Event constant, such as Event.SUBSCRIPTION_STATUS.
    *messages : tuple
        One (message type, correlation id int or None, body dict) each.
    """
    event = blpapi.test.createEvent(event_type)
    for message_type, cid, body in messages:
        definition = blpapi.test.getAdminMessageDefinition(blpapi.Name(message_type))
        properties = blpapi.test.MessageProperties()
        if cid is not None:
            properties.setCorrelationIds([blpapi.CorrelationId(cid)])
        blpapi.test.appendMessage(event, definition, properties).formatMessageDict(body)
    return event


def started(cid: int) -> tuple:
    """SubscriptionStarted message tuple for admin_event.
    """
    body = {'streamIds': ['1'], 'receivedFrom': {'address': 'localhost:8194'}, 'reason': 'ok'}
    return 'SubscriptionStarted', cid, body


def failed(cid: int, desc: str | None, message_type: str = 'SubscriptionFailure') -> tuple:
    """SubscriptionFailure (or Terminated) tuple. None leaves no description.
    """
    return message_type, cid, {'reason': {} if desc is None else {'description': desc}}


def status_event(*messages: tuple) -> blpapi.Event:
    return admin_event(Event.SUBSCRIPTION_STATUS, *messages)


def session_event(message_type: str) -> blpapi.Event:
    body = {} if message_type == 'SessionTerminated' else {'server': 'localhost:8194'}
    return admin_event(Event.SESSION_STATUS, (message_type, None, body))


def data_event(*ticks: tuple) -> blpapi.Event:
    """Real SUBSCRIPTION_DATA event from (correlation id int, body dict) pairs.
    """
    event = blpapi.test.createEvent(Event.SUBSCRIPTION_DATA)
    for cid, body in ticks:
        properties = blpapi.test.MessageProperties()
        properties.setCorrelationIds([blpapi.CorrelationId(cid)])
        blpapi.test.appendMessage(event, MKTDATA_EVENT, properties).formatMessageDict(body)
    return event


class Harness:
    """A Subscription on a mock session, with a fake monotonic clock and a
    dispatcher thread that delivers events.
    """

    def __init__(self, on_tick=None, session=None) -> None:
        self.now = 0.0
        self.ticks = []
        self.session = session or MagicMock()
        clock = MagicMock()
        clock.monotonic.side_effect = lambda: self.now
        self._patches = [
            patch.object(blp.client.SessionFactory, 'create', side_effect=self._create),
            patch.object(blp.client, 'time', clock),
            ]
        for p in self._patches:
            p.start()
        self.sub = Subscription(['LAST_PRICE', 'BID'], on_tick or self._record_tick)

    def _create(self, **kwargs) -> MagicMock:
        self.handler = kwargs['event_handler']
        return self.session

    def _record_tick(self, topic, row) -> None:
        self.ticks.append((topic, row))

    def deliver(self, *events) -> None:
        """Run the handler on one worker thread; fail on a hang.
        """
        thread = threading.Thread(
            target=lambda: [self.handler(e, self.session) for e in events],
            daemon=True)
        thread.start()
        thread.join(timeout=5)
        assert not thread.is_alive(), 'handler hung'

    def cids(self, call_index: int = -1) -> dict[str, int]:
        """Topic to correlation id of one session.subscribe call.
        """
        subscriptions = self.session.subscribe.call_args_list[call_index].args[0]
        return {
            subscriptions.topicStringAt(i).split('?')[0]: subscriptions.correlationIdAt(i).value()
            for i in range(subscriptions.size())
            }

    def close(self) -> None:
        for p in self._patches:
            p.stop()


@pytest.fixture
def harness():
    h = Harness()
    yield h
    h.close()


def test_add_skips_held_topics_and_uses_fresh_cids(harness):
    """Verify add() subscribes only new topics, each with its own int id.

    Mutation: add() resubscribing held topics, or one cid for every topic.
    Oracle: two calls, the second naming one held and one new topic.
    """
    harness.sub.add(['AAA US Equity', 'BBB US Equity'])
    harness.sub.add(['AAA US Equity', 'CCC US Equity'])
    first, second = harness.cids(0), harness.cids(1)
    assert list(first) == ['AAA US Equity', 'BBB US Equity']
    assert list(second) == ['CCC US Equity']
    assert len({*first.values(), *second.values()}) == 3


def test_removed_topic_ignores_stale_events(harness):
    """Verify events for a removed topic's old id never reach the re-added
    topic.

    Mutation: the cid mapping dropped after unsubscribe, or a topic-string
        correlation id reused on re-add.
    Oracle: the old id's termination and tick arrive after the re-add.
    """
    harness.sub.add('AAA US Equity')
    old_cid = harness.cids()['AAA US Equity']
    harness.sub.remove('AAA US Equity')
    unsubscribed = harness.session.unsubscribe.call_args.args[0]
    assert unsubscribed.correlationIdAt(0).value() == old_cid
    harness.sub.add('AAA US Equity')
    assert harness.cids()['AAA US Equity'] != old_cid
    harness.deliver(
        status_event(failed(old_cid, 'gone', 'SubscriptionTerminated')),
        data_event((old_cid, {'LAST_PRICE': 1.0})))
    state = harness.sub.status().topics['AAA US Equity']
    assert state.status == 'pending'
    assert state.first_data_at is None
    assert harness.ticks == []


def test_status_transitions_and_reason(harness):
    """Verify SubscriptionStarted marks subscribed and SubscriptionFailure
    marks failed with Bloomberg's description.

    Mutation: matching blp's lower camel Name, or reading description off
        the message root.
    Oracle: real admin events for two topics.
    """
    harness.sub.add(['AAA US Equity', 'BBB US Equity'])
    cids = harness.cids()
    harness.deliver(status_event(started(cids['AAA US Equity']), failed(cids['BBB US Equity'], 'bad')))
    topics = harness.sub.status().topics
    assert topics['AAA US Equity'].status == 'subscribed'
    assert topics['AAA US Equity'].subscribed_at is not None
    assert (topics['BBB US Equity'].status, topics['BBB US Equity'].reason) == ('failed', 'bad')


def test_add_resubscribes_failed_topic(harness):
    """Verify add() subscribes a failed topic again under a new id.

    Mutation: add() skipping every held topic.
    Oracle: the failed topic appears in a second subscribe call.
    """
    harness.sub.add('AAA US Equity')
    cid = harness.cids()['AAA US Equity']
    harness.deliver(status_event(failed(cid, 'bad')))
    harness.sub.add('AAA US Equity')
    assert harness.session.subscribe.call_count == 2
    assert harness.cids()['AAA US Equity'] != cid
    assert harness.sub.status().topics['AAA US Equity'].status == 'pending'


def test_tick_row_holds_present_fields(harness):
    """Verify on_tick gets the topic and the requested fields present.

    Mutation: rows built from every requested field, or keyed by cid.
    Oracle: LAST_PRICE 101.5 set by hand; BID absent.
    """
    harness.sub.add('AAA US Equity')
    harness.deliver(data_event((harness.cids()['AAA US Equity'], {'LAST_PRICE': 101.5})))
    assert harness.ticks == [('AAA US Equity', {'LAST_PRICE': 101.5})]
    state = harness.sub.status().topics['AAA US Equity']
    assert state.first_data_at is not None
    assert state.last_tick_at >= state.first_data_at


def test_ticks_last_minute_counts_window_at_read_time(harness):
    """Verify ticks_last_minute counts ticks under 60 s old at the read.

    Mutation: pruning only on a new tick, or a <= 60 s window.
    Oracle: ticks at 0 s and 30 s; reads at 59.9, 60 and 90 s.
    """
    harness.sub.add('AAA US Equity')
    cid = harness.cids()['AAA US Equity']
    for at in (0, 30):
        harness.now = at
        harness.deliver(data_event((cid, {'LAST_PRICE': 1.0})))
    counts = []
    for at in (59.9, 60, 90):
        harness.now = at
        counts.append(harness.sub.status().topics['AAA US Equity'].ticks_last_minute)
    assert counts == [2, 1, 0]


def test_on_tick_error_keeps_rest_of_event(harness):
    """Verify an on_tick exception is counted and the event's next message
    still arrives.

    Mutation: one try around the whole event, or errors not counted.
    Oracle: one event with ticks for two topics; on_tick raises on the first.
    """
    seen = []

    def on_tick(topic, row):
        seen.append(topic)
        if topic == 'AAA US Equity':
            raise ValueError('boom')

    harness.sub.on_tick = on_tick
    harness.sub.add(['AAA US Equity', 'BBB US Equity'])
    cids = harness.cids()
    harness.deliver(data_event(
        (cids['AAA US Equity'], {'LAST_PRICE': 1.0}),
        (cids['BBB US Equity'], {'LAST_PRICE': 2.0})))
    status = harness.sub.status()
    assert seen == ['AAA US Equity', 'BBB US Equity']
    assert status.error_cnt == 1
    assert status.last_error == "AAA US Equity: ValueError('boom')"


def test_parse_error_keeps_rest_of_event(harness):
    """Verify a parse exception is counted and the next message still ticks.

    Mutation: the per-message try placed around on_tick alone.
    Oracle: the parser raises on the first value and returns 20.0 next.
    """
    harness.sub.add(['AAA US Equity', 'BBB US Equity'])
    cids = harness.cids()
    with patch.object(harness.sub.parser, 'get_subelement_value', side_effect=[ValueError('bad'), 20.0]):
        harness.deliver(data_event(
            (cids['AAA US Equity'], {'LAST_PRICE': 1.0}),
            (cids['BBB US Equity'], {'LAST_PRICE': 2.0})))
    assert harness.ticks == [('BBB US Equity', {'LAST_PRICE': 20.0})]
    assert harness.sub.status().last_error.startswith('AAA US Equity: ')


def test_system_exit_in_on_tick_never_escapes():
    """Verify SystemExit from on_tick stays inside the handler, which
    blpapi would answer with os._exit(1).

    Mutation: the handler's guard catching Exception only.
    Oracle: the handler returns normally and the error is counted.
    """
    def on_tick(topic, row):
        raise SystemExit(3)

    harness = Harness(on_tick)
    try:
        harness.sub.add('AAA US Equity')
        harness.deliver(data_event((harness.cids()['AAA US Equity'], {'LAST_PRICE': 1.0})))
        assert harness.sub.status().error_cnt == 1
    finally:
        harness.close()


def test_on_tick_may_call_back_into_subscription():
    """Verify on_tick runs with no lock held.

    Mutation: on_tick called inside the state lock, which deadlocks here.
    Oracle: the dispatcher thread finishes, and status() saw the tick.
    """
    seen = []
    harness = Harness(lambda topic, row: seen.append(harness.sub.status().topics[topic].last_tick_at))
    try:
        harness.sub.add('AAA US Equity')
        harness.deliver(data_event((harness.cids()['AAA US Equity'], {'LAST_PRICE': 1.0})))
        assert len(seen) == 1
        assert seen[0] is not None
    finally:
        harness.close()


def test_session_terminated_outside_stop_is_failure(harness):
    """Verify SessionTerminated outside stop() sets the failure.

    Mutation: matching blp's lower camel Name.SESSION_TERMINATED.
    Oracle: a real SessionTerminated event after data.
    """
    harness.sub.add('AAA US Equity')
    harness.deliver(data_event((harness.cids()['AAA US Equity'], {'LAST_PRICE': 1.0})))
    harness.deliver(session_event('SessionTerminated'))
    assert harness.sub.status().failure == 'session terminated'


def test_connection_down_is_not_failure(harness):
    """Verify SessionConnectionDown clears connected and sets no failure,
    and SessionConnectionUp restores it.

    Mutation: connection down treated as session death, or connected never
        restored.
    Oracle: real down and up events, read between them.
    """
    harness.sub.add('AAA US Equity')
    harness.deliver(data_event((harness.cids()['AAA US Equity'], {'LAST_PRICE': 1.0})))
    harness.deliver(session_event('SessionConnectionDown'))
    down = harness.sub.status()
    harness.deliver(session_event('SessionConnectionUp'))
    assert (down.connected, down.failure) == (False, None)
    assert harness.sub.status().connected is True


def test_all_topics_failed_names_first_reason(harness):
    """Verify failure on every held topic is a failure naming the first
    topic's reason, and failure on some is not.

    Mutation: failure on the first dead topic, or an off-by-one in the count.
    Oracle: two topics fail one after the other, read between.
    """
    harness.sub.add(['AAA US Equity', 'BBB US Equity'])
    cids = harness.cids()
    harness.deliver(status_event(failed(cids['AAA US Equity'], 'first')))
    partial = harness.sub.status().failure
    harness.deliver(status_event(failed(cids['BBB US Equity'], 'second', 'SubscriptionTerminated')))
    assert partial is None
    assert harness.sub.status().failure == 'all 2 topics failed: first'


def test_removing_good_topic_leaves_failed_ones_as_failure(harness):
    """Verify the all-failed rule holds over the topics held at the read.

    Mutation: the rule evaluated only when a status event arrives.
    Oracle: one topic failed, one ticking; the ticking one is removed.
    """
    harness.sub.add(['AAA US Equity', 'BBB US Equity'])
    cids = harness.cids()
    harness.deliver(
        status_event(failed(cids['AAA US Equity'], 'bad')),
        data_event((cids['BBB US Equity'], {'LAST_PRICE': 1.0})))
    harness.sub.remove('BBB US Equity')
    assert harness.sub.status().failure == 'all 1 topics failed: bad'


def test_failure_without_description_counts_every_topic(harness):
    """Verify a failure lacking its optional description still lets the
    rest of the event's topics count.

    Mutation: reading description unguarded, whose exception skips the
        event's second message.
    Oracle: one real event holds both failures, the first with an empty
        reason.
    """
    harness.sub.add(['AAA US Equity', 'BBB US Equity'])
    cids = harness.cids()
    harness.deliver(status_event(failed(cids['AAA US Equity'], None), failed(cids['BBB US Equity'], 'bad')))
    assert harness.sub.status().failure == 'all 2 topics failed: no description'


def test_no_data_within_60s_is_failure(harness):
    """Verify no tick within 60 s of the first add is a failure.

    Mutation: the no-data rule dropped, or a threshold other than 60 s.
    Oracle: reads at 59.99 s and 60 s after an add at 0 s.
    """
    harness.sub.add('AAA US Equity')
    harness.now = 59.99
    before = harness.sub.status().failure
    harness.now = 60
    assert before is None
    assert harness.sub.status().failure == 'no subscription data within 60 s'


@pytest.mark.parametrize(('data_at', 'failure'), [
    (59, None),
    (60, 'no subscription data within 60 s'),
    ])
def test_no_data_rule_ignores_read_time(harness, data_at, failure):
    """Verify the no-data rule judges the first tick's time, whenever
    status() is read.

    Mutation: the rule evaluated against the read time, or first data
        disarming it outright.
    Oracle: a tick one second either side of the limit, read at 120 s.
    """
    harness.sub.add('AAA US Equity')
    harness.now = data_at
    harness.deliver(data_event((harness.cids()['AAA US Equity'], {'LAST_PRICE': 1.0})))
    harness.now = 120
    assert harness.sub.status().failure == failure


def test_add_rolls_back_when_session_not_started(harness):
    """Verify a failed subscribe leaves none of the call's topics held.

    Mutation: rollback skipped, or the blpapi error raised unmapped.
    Oracle: the real error an unstarted session raises; a later add works.
    """
    harness.session.subscribe.side_effect = [NOT_STARTED, None]
    with pytest.raises(SessionNotAvailableError):
        harness.sub.add('AAA US Equity')
    assert harness.sub.status().topics == {}
    harness.sub.add('AAA US Equity')
    assert list(harness.sub.status().topics) == ['AAA US Equity']


def test_stop_waits_for_termination_then_cleans_up(harness):
    """Verify stop() unsubscribes, waits for SessionTerminated, cleans up
    once, and reads as no failure.

    Mutation: cleanup before termination, a clean stop read as a failure,
        or a second stop() cleaning up again.
    Oracle: the mock stopAsync delivers SessionTerminated on another thread.
    """
    harness.sub.add('AAA US Equity')
    cid = harness.cids()['AAA US Equity']
    harness.deliver(data_event((cid, {'LAST_PRICE': 1.0})))
    harness.session.stopAsync.side_effect = lambda: threading.Thread(
        target=harness.handler, args=(session_event('SessionTerminated'), None)).start()
    harness.sub.stop(timeout=5)
    harness.sub.stop(timeout=5)
    unsubscribed = harness.session.unsubscribe.call_args.args[0]
    assert unsubscribed.correlationIdAt(0).value() == cid
    harness.session.cleanup.assert_called_once()
    assert harness.sub.status().failure is None


def test_stop_timeout_unhooks_exit_and_retry_cleans_up(harness):
    """Verify a stop() that times out leaves the session with its exit hook
    removed, and a later stop() cleans up.

    Mutation: cleanup on timeout, the atexit hook left in place, or a second
        stop() returning at once.
    Oracle: stopAsync delivers nothing; SessionTerminated comes later.
    """
    with patch.object(blp.client, 'atexit') as atexit:
        harness.sub.stop(timeout=0)
    harness.session.cleanup.assert_not_called()
    atexit.unregister.assert_called_once_with(harness.session.cleanup)
    harness.deliver(session_event('SessionTerminated'))
    harness.sub.stop(timeout=0)
    harness.session.cleanup.assert_called_once()


def test_stop_from_on_tick_returns_without_waiting():
    """Verify stop() called on the dispatcher thread begins the stop and
    returns, and a later stop() from the owner cleans up.

    Mutation: the dispatcher-thread check dropped, which waits out the
        timeout on the thread that must deliver SessionTerminated.
    Oracle: a 60 s timeout against the harness's 5 s hang check.
    """
    harness = Harness(lambda topic, row: harness.sub.stop(timeout=60))
    try:
        harness.sub.add('AAA US Equity')
        harness.deliver(data_event((harness.cids()['AAA US Equity'], {'LAST_PRICE': 1.0})))
        harness.session.stopAsync.assert_called_once()
        harness.session.cleanup.assert_not_called()
        harness.deliver(session_event('SessionTerminated'))
        harness.sub.stop(timeout=0)
        harness.session.cleanup.assert_called_once()
    finally:
        harness.close()


def test_stop_cleans_up_when_unsubscribe_raises(harness):
    """Verify stop() still stops and releases a live session when
    unsubscribe raises.

    Mutation: the blpapi error left to escape stop(), or stopAsync in the
        same try as unsubscribe.
    Oracle: stopAsync delivers SessionTerminated; nothing else does.
    """
    harness.sub.add('AAA US Equity')
    harness.session.unsubscribe.side_effect = blpapi.InvalidArgumentException('bad', 0)
    harness.session.stopAsync.side_effect = lambda: threading.Thread(
        target=harness.handler, args=(session_event('SessionTerminated'), None)).start()
    harness.sub.stop(timeout=5)
    harness.session.cleanup.assert_called_once()


def test_add_after_stop_raises(harness):
    """Verify add() refuses once stop() has begun.

    Mutation: the stopping check dropped from add().
    Oracle: stop() with no termination, then add().
    """
    with patch.object(blp.client, 'atexit'):
        harness.sub.stop(timeout=0)
    with pytest.raises(SessionError, match='stopped'):
        harness.sub.add('AAA US Equity')
    harness.session.subscribe.assert_not_called()


def test_failed_open_service_cleans_up():
    """Verify the session is cleaned up when opening //blp/mktdata fails.

    Mutation: open_service outside the try.
    Oracle: a mock session whose open_service raises; cleanup is recorded.
    """
    session = MagicMock()
    session.open_service.side_effect = SessionStartError('no service')
    try:
        with pytest.raises(SessionStartError):
            Harness(session=session)
    finally:
        patch.stopall()
    session.cleanup.assert_called_once()


def test_subscription_parses_with_subscription_parser(harness):
    """Verify a subscription parses with SubscriptionParser.

    Mutation: Subscription builds a plain Parser.
    Oracle: the class of the subscription's parser.
    """
    assert type(harness.sub.parser) is SubscriptionParser


def test_clean_stop_never_reads_as_failure(harness):
    """Verify a stopped subscription reads no failure later.

    Mutation: the no-data or all-failed rule applied after stop().
    Oracle: add at 0 s with no tick, stop at 5 s, read at 61 s; then the
        good topic removed, leaving only the failed one.
    """
    harness.sub.add(['AAA US Equity', 'BBB US Equity'])
    harness.deliver(status_event(failed(harness.cids()['AAA US Equity'], 'bad')))
    harness.now = 5
    harness.session.stopAsync.side_effect = lambda: threading.Thread(
        target=harness.handler, args=(session_event('SessionTerminated'), None)).start()
    harness.sub.stop(timeout=5)
    harness.now = 61
    late = harness.sub.status().failure
    harness.sub.remove('BBB US Equity')
    assert late is None
    assert harness.sub.status().failure is None


def test_stop_after_termination_on_reused_thread_cleans_up(harness):
    """Verify stop() cleans up when called on a thread whose id matches the
    ended dispatcher thread.

    Mutation: the dispatcher thread id kept after SessionTerminated.
    Oracle: termination delivered on the main thread, then stop() there.
    """
    harness.handler(session_event('SessionTerminated'), None)
    harness.sub.stop(timeout=0)
    harness.session.cleanup.assert_called_once()


def test_failed_readd_rollback_keeps_failure(harness):
    """Verify a failed re-add of a failed topic restores its failed state.

    Mutation: rollback deleting the topic's prior state.
    Oracle: the only topic failed; the re-add's subscribe raises.
    """
    harness.sub.add('AAA US Equity')
    cid = harness.cids()['AAA US Equity']
    harness.deliver(status_event(failed(cid, 'bad')))
    harness.session.subscribe.side_effect = NOT_STARTED
    with pytest.raises(SessionNotAvailableError):
        harness.sub.add('AAA US Equity')
    assert harness.sub.status().failure == 'all 1 topics failed: bad'
    harness.deliver(status_event(failed(cid, 'again')))
    assert harness.sub.status().topics['AAA US Equity'].reason == 'again'


def test_system_exit_keeps_rest_of_event():
    """Verify SystemExit from on_tick on one message leaves the next one.

    Mutation: the per-message catch narrowed to Exception.
    Oracle: one event, two topics; on_tick exits on the first.
    """
    seen = []

    def on_tick(topic, row):
        seen.append(topic)
        if topic == 'AAA US Equity':
            raise SystemExit(3)

    harness = Harness(on_tick)
    try:
        harness.sub.add(['AAA US Equity', 'BBB US Equity'])
        cids = harness.cids()
        harness.deliver(data_event(
            (cids['AAA US Equity'], {'LAST_PRICE': 1.0}),
            (cids['BBB US Equity'], {'LAST_PRICE': 2.0})))
        assert seen == ['AAA US Equity', 'BBB US Equity']
    finally:
        harness.close()


def test_subscribe_sends_fields_and_interval():
    """Verify each topic is subscribed with the fields and the interval.

    Mutation: fields or the interval option dropped from the list.
    Oracle: blpapi's own topic string for interval 2.
    """
    harness = Harness()
    try:
        harness.sub.interval = 2
        harness.sub.add('AAA US Equity')
        subscriptions = harness.session.subscribe.call_args.args[0]
        topic = subscriptions.topicStringAt(0)
        assert topic == 'AAA US Equity?fields=LAST_PRICE,BID&interval=2.0'
    finally:
        harness.close()


def test_add_dedups_topics_in_one_call(harness):
    """Verify a topic named twice in one add() is subscribed once.

    Mutation: dict.fromkeys dropped from add().
    Oracle: one call naming the same topic twice.
    """
    harness.sub.add(['AAA US Equity', 'AAA US Equity'])
    assert harness.session.subscribe.call_args.args[0].size() == 1


def test_no_data_clock_starts_at_first_good_add(harness):
    """Verify the no-data clock starts at the first add that subscribed and
    never restarts.

    Mutation: the clock set before subscribe succeeds, or reset per add.
    Oracle: failed add at 0 s, good add at 30 s, another at 60 s.
    """
    harness.session.subscribe.side_effect = [NOT_STARTED, None, None]
    with pytest.raises(SessionNotAvailableError):
        harness.sub.add('AAA US Equity')
    harness.now = 30
    harness.sub.add('AAA US Equity')
    harness.now = 60
    harness.sub.add('BBB US Equity')
    harness.now = 89.9
    early = harness.sub.status().failure
    harness.now = 90
    assert early is None
    assert harness.sub.status().failure == 'no subscription data within 60 s'


def test_first_data_times_are_kept(harness):
    """Verify later ticks keep both first-data times.

    Mutation: first_data_at, or the run-wide first tick time, overwritten.
    Oracle: ticks at 10 s and 70 s, read at 120 s.
    """
    harness.sub.add('AAA US Equity')
    cid = harness.cids()['AAA US Equity']
    harness.now = 10
    harness.deliver(data_event((cid, {'LAST_PRICE': 1.0})))
    first = harness.sub.status().topics['AAA US Equity'].first_data_at
    harness.now = 70
    harness.deliver(data_event((cid, {'LAST_PRICE': 2.0})))
    harness.now = 120
    status = harness.sub.status()
    assert status.failure is None
    assert status.topics['AAA US Equity'].first_data_at == first
    assert status.topics['AAA US Equity'].last_tick_at > first


def test_session_terminated_outranks_all_failed(harness):
    """Verify session termination is the failure reported over all-failed.

    Mutation: the failure rules checked in another order.
    Oracle: the only topic fails, then the session terminates.
    """
    harness.sub.add('AAA US Equity')
    harness.deliver(
        status_event(failed(harness.cids()['AAA US Equity'], 'bad')),
        session_event('SessionTerminated'))
    assert harness.sub.status().failure == 'session terminated'


def test_stop_drops_later_ticks_and_stops_empty_session(harness):
    """Verify stop() ignores later ticks and stops a session with no topics.

    Mutation: the cid map kept in stop(), or stopAsync skipped when nothing
        is held.
    Oracle: a tick after stop(); an empty subscription stopped.
    """
    harness.sub.add('AAA US Equity')
    cid = harness.cids()['AAA US Equity']
    harness.sub.remove('AAA US Equity')
    with patch.object(blp.client, 'atexit'):
        harness.sub.stop(timeout=0)
    harness.deliver(data_event((cid, {'LAST_PRICE': 1.0})))
    assert harness.ticks == []
    harness.session.stopAsync.assert_called_once()


def test_status_topics_are_copies(harness):
    """Verify a status() result does not change after later events.

    Mutation: status() returning the live TopicState objects.
    Oracle: a held copy read after SubscriptionStarted.
    """
    harness.sub.add('AAA US Equity')
    before = harness.sub.status().topics['AAA US Equity']
    harness.deliver(status_event(started(harness.cids()['AAA US Equity'])))
    assert before.status == 'pending'


def test_remove_maps_not_started_error(harness):
    """Verify remove() raises SessionNotAvailableError on a dead session and
    drops the topics anyway.

    Mutation: the blpapi error raised unmapped, or topics kept on error.
    Oracle: the real error an unstarted session raises.
    """
    harness.sub.add('AAA US Equity')
    harness.session.unsubscribe.side_effect = NOT_STARTED
    with pytest.raises(SessionNotAvailableError):
        harness.sub.remove('AAA US Equity')
    assert harness.sub.status().topics == {}
