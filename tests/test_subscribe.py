import atexit

import asserts

from blp import Blp
from blp.handle import DefaultEventHandler
from date import DateTime


class AssertionEventHandler(DefaultEventHandler):

    def __init__(self, topics: list[str], fields: list[str]):
        super().__init__(topics, fields)
        self.match = False
        atexit.register(lambda: asserts.assert_true(self.match))

    def emit(self, topic, row):
        now = DateTime.now()
        if 'TIME' in row:
            asserts.assert_less(now.subtract(minutes=30), row['TIME'])
            asserts.assert_greater(now.add(minutes=10), row['TIME'])
        if 'PRICE_LAST_TIME_RT' in row:
            asserts.assert_less(now.subtract(minutes=30), row['PRICE_LAST_TIME_RT'])
            asserts.assert_greater(now.add(minutes=10), row['PRICE_LAST_TIME_RT'])
            self.match = True


def test_subscribe_timezone():
    with Blp() as blp:
        topics = ['IBM US Equity']
        fields = ['TIME', 'PRICE_LAST_TIME_RT', 'PX_LAST']
        blp.subscribe(topics, fields=fields, handler=AssertionEventHandler, runtime=5)


if __name__ == '__main__':
    __import__('pytest').main([__file__])
