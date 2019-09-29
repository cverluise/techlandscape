import functools
from time import time

from wasabi import Printer


def monitor(f):
    @functools.wraps(f)
    def wrapper_monitor(*args, **kwargs):
        msg = Printer()
        tic = time()
        with msg.loading("working hard..."):
            value = f(*args, **kwargs)
        toc = time()
        msg.good(f"Done! (Took {round(toc - tic)}s)")
        return value

    return wrapper_monitor
