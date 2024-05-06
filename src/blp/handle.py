import datetime
import logging
import time
import warnings
from abc import ABC, abstractmethod

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

    def __init__(self, topics, fields):
        self.topics = topics
        self.fields = fields

    @abstractmethod
    def emit(self, topic, parsed):
        """Parsed will be provided by the BaseEventHandler.

        Consists of a tuple: (topic, dictionary of type {field: parsed_value} )
        """

    def __call__(self, event, _):
        """This method is called from Bloomberg session in a separate thread
        for each incoming event.
        """
        try:
            event_type = event.eventType()
            if event_type == Event.SUBSCRIPTION_DATA:
                logger.debug('next(): subscription data')
                self._data_event(event, _)
                return
            if event_type == Event.SUBSCRIPTION_STATUS:
                logger.debug('next(): subscription status')
                self._status_event(event, _)
                return
            if event_type == Event.TIMEOUT:
                return
            self._misc_event(event, _)
        except blpapi.Exception as exception:
            logger.error(f'Failed to process event {event}: {exception}')

    def _status_event(self, event, _):
        for message in Parser.message_iter(event):
            topic = message.correlationId().value()
            match message.messageType():
                case Name.SUBSCRIPTION_FAILURE:
                    desc = message.getElement('reason').getElementAsString('description')
                    raise Exception(f'Subscription failed topic={topic} desc={desc}')
                case Name.SUBSCRIPTION_TERMINATED:
                    # Subscription can be terminated if the session identity is revoked.
                    logger.error(f'Subscription for {topic} TERMINATED')

    def _data_event(self, event, _):
        """Return a full mapping of fields to parsed values"""
        for message in Parser.message_iter(event):
            parsed = {}
            topic = message.correlationId().value()
            for field in self.fields:
                if field.upper() in message:
                    val = Parser.get_subelement_value(message, field.upper())
                    if isinstance(val, datetime.datetime):
                        val = val.replace(tzinfo=LCL)
                    parsed[field] = val
            self.emit(topic, parsed)

    def _misc_event(self, event, _):
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


class SimpleEventHandler(BaseEventHandler):
    """Simple event handler."""

    def emit(self, topic, parsed):
        logger.info(f'{topic}: {parsed}')


class LoggingEventHandler(BaseEventHandler):
    """Basic dataset logging event handler"""

    def __init__(self, topics, fields, index: dict = None, sort_by: str =
                 None, sort_mod: callable = lambda x: x):
        super().__init__(topics, fields)
        nrows, ncols = len(self.topics), len(self.fields)
        vals = np.repeat(np.nan, nrows * ncols).reshape((nrows, ncols))
        self.frame = pd.DataFrame(vals, columns=self.fields,
                                  index=[(index or {}).get(t, t) for t in self.topics])
        self.sort_by = sort_by
        self.sort_mod = sort_mod

    def emit(self, topic, parsed):
        logger.debug(f'Received event for {time.strftime("%Y/%m/%d %X")}: {topic}')
        ridx = self.topics.index(topic)
        for cidx, field in enumerate(self.fields):
            if field in parsed:
                self.frame.iloc[ridx, cidx] = parsed[field]
        df = self.frame
        if self.sort_by:
            sortable = [self.sort_mod(x) for x in df[self.sort_by]]
            sort_key = lambda x: np.argsort(index_natsorted(sortable))
            df = self.frame.sort_values(by=self.sort_by, key=sort_key)
        logger.info(df.to_string())
