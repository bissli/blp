import logging
import time
import warnings
from abc import ABCMeta, abstractmethod

import blpapi
import numpy as np
import pandas as pd
from bbg.util import Name, Parser
from blpapi.event import Event

logger = logging.getLogger(__name__)


class BaseEventHandler(metaclass=ABCMeta):
    """Base Event Handler."""

    def __init__(self, topics, fields, parser: Parser = None):
        self.topics = topics
        self.fields = fields
        self.parser = parser or Parser()

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
                self.__data_event(event, _)
                return
            if event_type == Event.SUBSCRIPTION_STATUS:
                logger.debug('next(): subscription status')
                self.__status_event(event, _)
                return
            if event_type == Event.TIMEOUT:
                return
            self.__misc_event(event, _)
        except blpapi.Exception as exception:
            logger.error(f'Failed to process event {event}: {exception}')

    def __status_event(self, event, _):
        for msg in self.parser.message_iter(event):
            topic = msg.correlationId().value()
            match msg.messageType():
                case Name.SUBSCRIPTION_FAILURE:
                    desc = msg.getElement('reason').getElementAsString('description')
                    raise Exception(f'Subscription failed topic={topic} desc={desc}')
                case Name.SUBSCRIPTION_TERMINATED:
                    # Subscription can be terminated if the session identity is revoked.
                    logger.error(f'Subscription for {topic} TERMINATED')

    def __data_event(self, event, _):
        """Return a full mapping of fields to parsed values"""
        for msg in self.parser.message_iter(event):
            parsed = {}
            topic = msg.correlationId().value()
            for field in self.fields:
                if field.upper() in msg:
                    val = self.parser.get_child_value(msg, field.upper())
                    parsed[field] = val
            self.emit(topic, parsed)

    def __misc_event(self, event, _):
        for msg in event:
            match msg.messageType():
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
                    logger.warning(msg)
                    topic = msg.correlationId().value()
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

    def __init__(self, topics, fields):
        super().__init__(topics, fields)

        # create dataframe grid
        nrows, ncols = len(self.topics), len(self.fields)
        vals = np.repeat(np.nan, nrows * ncols).reshape((nrows, ncols))
        self.frame = pd.DataFrame(vals, columns=self.fields, index=self.topics)

    def emit(self, topic, parsed):
        logger.debug(f'Received event for {time.strftime("%Y/%m/%d %X")}: {topic}')
        ridx = self.topics.index(topic)
        for cidx, field in enumerate(self.fields):
            if field in parsed:
                with warnings.catch_warnings():
                    warnings.simplefilter(action='ignore', category=FutureWarning)
                    self.frame.iloc[ridx, cidx] = parsed[field]
        logger.info(self.frame)
