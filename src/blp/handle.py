"""Subscription event handlers
"""
import datetime
import logging
import time
import warnings
from abc import ABC, abstractmethod
from typing import Any

import blpapi
import numpy as np
import pandas as pd
from blp.parse import Name, Parser
from blpapi.event import Event
from natsort import index_natsorted

from date import LCL

logger = logging.getLogger(__name__)

warnings.simplefilter(action='ignore', category=FutureWarning)


class BaseEventHandler(ABC):
    """Base Event Handler."""

    def __init__(self, topics: list[str], fields: list[str]):
        self.topics = topics
        self.fields = fields
        self.parser = Parser()

    @abstractmethod
    def emit(self, topic: str, row: dict[str: Any]):
        """Triggerd by BaseEventHandler on data event.

        Topic: topic from topics
        Row: {field: field value}

        Implement any handling logic here.
        """

    def __call__(self, event, *args):
        """This method is called from Bloomberg session in a separate thread
        for each incoming event.
        """
        try:
            match event.eventType():
                case Event.SUBSCRIPTION_DATA:
                    self._on_data_event(event)
                case Event.SUBSCRIPTION_STATUS:
                    self._on_status_event(event)
                case Event.TIMEOUT:
                    return
                case _:
                    self._on_other_event(event)
        except blpapi.Exception as exception:
            logger.error(f'Failed to process event {event}: {exception}')

    def _on_status_event(self, event):
        logger.debug('Event triggered: subscription status')
        for message in self.parser.message_iter(event):
            topic = message.correlationId().value()
            match message.messageType():
                case Name.SUBSCRIPTION_FAILURE:
                    desc = message.getElement('reason').getElementAsString('description')
                    raise Exception(f'Subscription failed topic={topic} desc={desc}')
                case Name.SUBSCRIPTION_TERMINATED:
                    # Subscription can be terminated if the session identity is revoked.
                    logger.error(f'Subscription for {topic} TERMINATED')

    def _on_data_event(self, event):
        """Return a full mapping of fields to parsed values"""
        logger.debug('Event triggered: subscription data')
        for message in self.parser.message_iter(event):
            row = {}
            topic = message.correlationId().value()
            for field in self.fields:
                if field.upper() in message:
                    val = self.parser.get_subelement_value(message, field.upper())
                    if isinstance(val, datetime.datetime):
                        val = val.replace(tzinfo=LCL)
                    row[field] = val
            self.emit(topic, row)

    def _on_other_event(self, event):
        logger.debug('Event triggered: other')
        for message in event:
            match message.messageType():
                case Name.SLOW_CONSUMER_WARNING:
                    logger.warning(
                        f'{Name.SLOW_CONSUMER_WARNING} - The event queue is '
                        + 'beginning to approach its maximum capacity and '
                        + 'the application is not processing the data fast '
                        + 'enough. This could lead to ticks being dropped'
                        + ' (DataLoss).\n'
                    )
                case Name.SLOW_CONSUMER_WARNING_CLEARED:
                    logger.warning(
                        f'{Name.SLOW_CONSUMER_WARNING_CLEARED} - the event '
                        + 'queue has shrunk enough that there is no '
                        + 'longer any immediate danger of overflowing the '
                        + 'queue. If any precautionary actions were taken '
                        + 'when SlowConsumerWarning message was delivered, '
                        + 'it is now safe to continue as normal.\n'
                    )
                case Name.DATA_LOSS:
                    logger.warning(message)
                    topic = message.correlationId().value()
                    logger.warning(
                        f'{Name.DATA_LOSS} - The application is too slow to '
                        + 'process events and the event queue is overflowing. '
                        + f'Data is lost for topic {topic}.\n'
                    )
                case Name.SESSION_TERMINATED:
                    # SESSION_STATUS events can happen at any time and
                    # should be handled as the session can be terminated,
                    # e.g. session identity can be revoked at a later
                    # time, which terminates the session.
                    logger.error('Session terminated')


class SimpleLoggingEventHandler(BaseEventHandler):
    """Simple event handler."""

    def emit(self, topic, parsed):
        logger.info(f'{topic}: {parsed}')


class BaseDataFrameEventHandler(BaseEventHandler):
    """Store as DataFrame"""

    def __init__(self, topics: list, fields: list, /, index: list = None):
        super().__init__(topics, fields)
        self.index = index
        nrows, ncols = len(self.topics), len(self.fields)
        vals = np.repeat(np.nan, nrows * ncols).reshape((nrows, ncols))
        self.frame = pd.DataFrame(vals, columns=self.fields)

    def emit(self, topic, parsed):
        logger.debug(f'Received event for {time.strftime("%Y/%m/%d %X")}: {topic}')
        ridx = self.topics.index(topic)
        for cidx, field in enumerate(self.fields):
            if field in parsed:
                self.frame.iloc[ridx, cidx] = parsed[field]


class LoggingDataFrameEventHandler(BaseDataFrameEventHandler):
    """Basic dataset logging event handler"""

    def ordering_datetime_modifier(self, x):
        """Place anything not parsed at the beginning
        Replace sort_mod perhaps with this if type is datetime
        """
        parsed = DateTime.parse(x)
        if not isinstance(parsed, DateTime):
            parsed = DateTime.now().start_of('day')
        return parsed

    def sorted(self):
        df = self.frame.copy(deep=True)
        for col in (self.index or []):
            if isinstance(df.dtypes[col], pd.Timestamp):
                sort_mod = self.ordering_datetime_modifier
            else:
                sort_mod = lambda x: x
            sortable = [sort_mod(x) for x in df[col]]
            sort_key = lambda x: np.argsort(index_natsorted(sortable))
            df = df.sort_values(by=col, key=sort_key)
        return df

    def emit(self, topic, parsed):
        super().emit(topic, parsed)
        logger.info(self.sorted().to_string())
