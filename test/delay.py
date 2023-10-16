import sys
import time

from bbg.util import NonBlockingDelay


def seconds():
    """Get seconds"""
    return int(time.time())


def delay_s(delay):
    """Blocking delay in seconds"""
    t0 = seconds()
    while (seconds() - t0) < delay:
        pass


print('With blocking delay')
t0 = seconds()
delay_s(1)
print(seconds() - t0)

# non blocking delay
d0, d1 = NonBlockingDelay(), NonBlockingDelay()

print('With non-blocking delay')
while 1:
    try:
        if d0.timeout():
            print('d0')
            d0.delay(1)

        if d1.timeout():
            print('d1')
            d1.delay(10)

    except KeyboardInterrupt:
        print('Ctrl-C pressed...')
        sys.exit()
