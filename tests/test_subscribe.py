import logging
import threading

from blp import Subscription

logger = logging.getLogger(__name__)


def test_subscribe_live():
    """Verify a live subscription ticks and stops cleanly on a Terminal.

    Mutation: ticks never reaching on_tick, or a clean stop read as failure.
    Oracle: Bloomberg sends a listed equity's initial snapshot on subscribe.
    """
    fields = ['LAST_PRICE', 'BID', 'ASK']
    ticked = threading.Event()

    def on_tick(topic, row):
        logger.info(f'{topic}: {row}')
        ticked.set()

    with Subscription(fields, on_tick) as sub:
        sub.add('IBM US Equity')
        assert ticked.wait(timeout=30)
        assert sub.status().failure is None
    assert sub.status().failure is None


if __name__ == '__main__':
    __import__('pytest').main([__file__])
