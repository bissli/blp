"""Subscription event handlers. Baseline backend model for all handlers here
is pandas.DataFrame.
"""
import logging
import warnings
from abc import ABC, abstractmethod
from typing import Any

import blpapi
import numpy as np
import pandas as pd
from blpapi.event import Event
from opendate import LCL

from blp.parse import Name, SubscriptionParser

logger = logging.getLogger(__name__)

warnings.simplefilter(action='ignore', category=FutureWarning)


class BaseEventHandler(ABC):
    """Turns subscription events into emit() calls and tracks whether the
    subscription can still deliver data.

    Attributes
    ----------
    first_data : bool
        True once any SUBSCRIPTION_DATA event arrives.
    dead_topics : set[str]
        Topics that got SubscriptionFailure or SubscriptionTerminated.
    failure : str or None
        Why the subscription can no longer deliver data: the session
        terminated, or dead_topics covers every topic. None while it can.

    Notes
    -----
    - Read first_data, dead_topics and failure only while subscribe() runs.
      Its cleanup stops the session, which sets failure even after a clean
      run.
    """

    def __init__(self, topics: list[str], fields: list[str], **kwargs):
        self.topics = topics
        self.fields = fields
        self.first_data = False
        self.dead_topics: set[str] = set()
        self.failure: str | None = None
        self._first_reason: str | None = None

        assumed_timezone = kwargs.pop('assumed_timezone', LCL)
        desired_timezone = kwargs.pop('desired_timezone', LCL)
        time_as_datetime = kwargs.pop('time_as_datetime', False)

        self.parser = SubscriptionParser(
            assumed_timezone=assumed_timezone,
            desired_timezone=desired_timezone,
            time_as_datetime=time_as_datetime,
        )

    @abstractmethod
    def emit(self, topic: str, row: dict[str, Any]) -> None:
        """Triggered by BaseEventHandler on data event.

        Topic: topic from topics
        Row: {field: field value}

        Implement any handling logic here.
        """

    def __call__(self, event, *args) -> None:
        """This method is called from Bloomberg session in a separate thread for each incoming event.
        """
        try:
            match event.eventType():
                case Event.SUBSCRIPTION_DATA:
                    self.first_data = True
                    self._on_data_event(event)
                case Event.SUBSCRIPTION_STATUS:
                    self._on_status_event(event)
                case Event.TIMEOUT:
                    return
                case _:
                    self._on_other_event(event)
        except blpapi.Exception as exception:
            logger.error(f'Failed to process event {event}: {exception}')

    def _on_status_event(self, event) -> None:
        """Handle subscription status events.
        """
        logger.debug('Event triggered: subscription status')
        for message in self.parser.message_iter(event):
            topic = message.correlationId().value()
            # blp's Name spells message types in lower camel case, which
            # never matches.
            match message.messageType():
                case blpapi.Names.SUBSCRIPTION_FAILURE:
                    status, level = 'failed', logging.ERROR
                case blpapi.Names.SUBSCRIPTION_TERMINATED:
                    # INFO, since every clean unsubscribe delivers one per
                    # topic.
                    status, level = 'terminated', logging.INFO
                case _:
                    continue
            reason = message.getElement('reason')
            desc = 'no description'
            if reason.hasElement('description'):
                desc = reason.getElementAsString('description')
            logger.log(level, f'Subscription {status} topic={topic} desc={desc}')
            self._first_reason = self._first_reason or desc
            self.dead_topics.add(topic)
            if not self.failure and self.dead_topics >= set(self.topics):
                self.failure = (
                    f'all {len(self.dead_topics)} topics failed or were '
                    f'terminated: {self._first_reason}')

    def _on_data_event(self, event) -> None:
        """Process data events and emit field values.
        """
        logger.debug('Event triggered: subscription data')
        for message in self.parser.message_iter(event):
            row = {}
            topic = message.correlationId().value()
            for field in self.fields:
                if field.upper() in message:
                    val = self.parser.get_subelement_value(message, field.upper())
                    row[field] = val
            self.emit(topic, row)

    def _on_other_event(self, event) -> None:
        """Handle internal warning events.
        """
        logger.debug('Event triggered: internal warning event')
        for message in event:
            match message.messageType():
                case Name.SLOW_CONSUMER_WARNING:
                    logger.warning(
                        f'{Name.SLOW_CONSUMER_WARNING} - The event queue is '
                        'beginning to approach its maximum capacity and '
                        'the application is not processing the data fast '
                        'enough. This could lead to ticks being dropped'
                        ' (DataLoss).\n'
                    )
                case Name.SLOW_CONSUMER_WARNING_CLEARED:
                    logger.warning(
                        f'{Name.SLOW_CONSUMER_WARNING_CLEARED} - the event '
                        'queue has shrunk enough that there is no '
                        'longer any immediate danger of overflowing the '
                        'queue. If any precautionary actions were taken '
                        'when SlowConsumerWarning message was delivered, '
                        'it is now safe to continue as normal.\n'
                    )
                case Name.DATA_LOSS:
                    logger.warning(message)
                    topic = message.correlationId().value()
                    logger.warning(
                        f'{Name.DATA_LOSS} - The application is too slow to '
                        'process events and the event queue is overflowing. '
                        f'Data is lost for topic {topic}.\n'
                    )
                case blpapi.Names.SESSION_TERMINATED:
                    # INFO, since every clean stop() delivers one.
                    logger.info('Session terminated')
                    self.failure = self.failure or 'session terminated'


class LoggingEventHandler(BaseEventHandler):
    """Log to debug the emit message.
    """

    def emit(self, topic: str, row: dict[str, Any]) -> None:
        """Emit event data with debug logging.
        """
        logger.debug(f'Event: {topic}: {row}')


class DefaultEventHandler(LoggingEventHandler):
    """Creates DataFrame and update as events are registered.

    Topics: list[str]: subscribable topics, i.e. IBM US Equity
    Fields: list[str]: fields to query
    Index:  list[str]: list of column names representing the DataFrame index.

    """

    def __init__(
        self,
        topics: list[str],
        fields: list[str],
        /,
        index: list[str] = None,
        **kwargs
    ):
        super().__init__(topics, fields, **kwargs)

        nrows, ncols = len(self.topics), len(self.fields)
        vals = np.repeat(np.nan, nrows * ncols).reshape((nrows, ncols))
        self.frame = pd.DataFrame(vals, columns=self.fields, index=self.topics)
        self.frame = self.frame.astype(object).where(pd.notnull(self.frame), None)

        self.index = index
        if self.index:
            self.frame.index = self.index

    def emit(self, topic: str, row: dict[str, Any]) -> None:
        """Update DataFrame with new field values for the given topic.
        """
        super().emit(topic, row)

        ridx = self.frame.index.get_loc(topic)
        for cidx, field in enumerate(self.fields):
            if field in row:
                self.frame.iloc[ridx, cidx] = row[field]
