import logging
import warnings
from abc import ABC, abstractmethod
from typing import Any

import blpapi
import numpy as np
import pandas as pd
from blpapi.event import Event
from natsort import index_natsorted
from prettytable.colortable import ColorTable

from blp import Blp
from blp.parse import Name, Parser
from opendate import LCL, DateTime

logger = logging.getLogger(__name__)

warnings.simplefilter(action='ignore', category=FutureWarning)


def ordering_datetime_modifier(x):
    """Place anything not parsed at the beginning
    Replace sort_mod perhaps with this if type is datetime
    """
    parsed = DateTime.parse(x)
    if not isinstance(parsed, DateTime):
        parsed = DateTime.now().start_of('day')
    return pd.to_datetime(parsed)


def sort(df, by):
    """Sorts dataframe inplace. Debounce N seconds."""
    if isinstance(df.dtypes[by], pd.Timestamp):
        sort_mod = ordering_datetime_modifier
    else:
        sort_mod = lambda x: x

    sortable = [sort_mod(x) for x in df[by]]
    sort_key = lambda x: np.argsort(index_natsorted(sortable))
    df.sort_values(by=by, key=sort_key, inplace=True)  # noqa


def pretty_dataframe(df, header=True, colmap=None):
    if colmap:
        df = df.rename(columns=colmap)
    t = ColorTable()
    t.field_names = list(df.columns)
    for row in df.to_dict('records'):
        try:
            t.add_row(list(row.values()))
        except:
            print('==Pretty Table Error==')
            print(row)
            print(t.field_names)
            print('======================')
            return
    print(t.get_string(header=header))


class BaseEventHandler(ABC):
    """Base Event Handler."""

    def __init__(self, topics: list[str], fields: list[str], **kwargs):
        self.topics = topics
        self.fields = fields

        assumed_timezone = kwargs.pop('assumed_timezone', LCL)
        desired_timezone = kwargs.pop('desired_timezone', LCL)
        time_as_datetime = kwargs.pop('time_as_datetime', False)

        self.parser = Parser(
            assumed_timezone=assumed_timezone,
            desired_timezone=desired_timezone,
            time_as_datetime=time_as_datetime,
        )

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
                    # subscription can be terminated if the session identity is revoked.
                    logger.error(f'Subscription for {topic} TERMINATED')

    def _on_data_event(self, event):
        """Return a full mapping of fields to parsed values"""
        logger.debug('Event triggered: subscription data')
        for message in self.parser.message_iter(event):
            print(message)
            row = {}
            topic = message.correlationId().value()
            for field in self.fields:
                if field.upper() in message:
                    val = self.parser.get_subelement_value(message, field.upper())
                    row[field] = val
            self.emit(topic, row)

    def _on_other_event(self, event):
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
                case Name.SESSION_TERMINATED:
                    # SESSION_STATUS events can happen at any time and
                    # should be handled as the session can be terminated,
                    # e.g. session identity can be revoked at a later
                    # time, which terminates the session.
                    logger.error('Session terminated')


class LoggingEventHandler(BaseEventHandler):
    """Log to debug the emit message"""

    def emit(self, topic, row, **kwargs):
        super().emit(topic, row, **kwargs)
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
        time_field: str = None,
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

        if time_field is None:
            this = [field for field in self.fields if 'TIME' in field]
            if not this:
                raise ValueError('A time-like for sorting not found in fields')
            time_field = this[0]
        self.time_field = time_field

    def emit(self, topic, row):
        super().emit(topic, row)

        ridx = self.frame.index.get_loc(topic)
        for cidx, field in enumerate(self.fields):
            if field in row:
                self.frame.iloc[ridx, cidx] = row[field]

        sort(self.frame, self.time_field)


class PrintingEventHandler(DefaultEventHandler):
    """Basic dataset logging event handler"""

    def __init__(self, topics: list[str], fields: list[str], field_label: dict):
        super().__init__(topics, fields)
        self.field_label = field_label

    def emit(self, topic, row):
        """Warning, do not filter or reassign blp frame object
        """
        super().emit(topic, row)
        # pretty_dataframe(self.frame, colmap=self.field_label)


def test_subscribe_timezone():
    with Blp() as blp:
        topics = ['BBG00S46V7M6@MSG1 Corp']
        fields = [
        'TIME',
        'BID_SENDERS_FIRM_RT',
        'ASK_SENDERS_FIRM_RT',
        'CVT_BOND_EQTY_REF_PRICE_RT',
        'BID',
        'ASK',
        'BID_SIZE',
        'ASK_SIZE',
        'ID_BB_GLOBAL_RT',
            ]
        blp.subscribe(topics, fields=fields, handler=PrintingEventHandler,
                      runtime=10, field_label={k:k for k in fields})


if __name__ == '__main__':
    __import__('pytest').main([__file__])
