import functools
import time

from wasabi import Printer

wait_time = 10


def monitor(f):
    @functools.wraps(f)
    def wrapper_monitor(*args, **kwargs):
        msg = Printer()
        tic = time.time()
        with msg.loading("Working hard..."):
            try:
                value = f(*args, **kwargs)
            except Exception as e:
                print(e)
                value = None
        toc = time.time()
        msg.good(f"Done! (Took {round(toc - tic)}s)")
        return value

    return wrapper_monitor


def timer(f):
    @functools.wraps(f)
    def wrapper_timer(*args, **kwargs):
        msg = Printer()
        with msg.loading("Take a deep breath..."):
            time.sleep(wait_time)
        msg.good("")
        try:
            value = f(*args, **kwargs)
        except Exception as e:
            print(e)
            value = None
        return value

    return wrapper_timer
