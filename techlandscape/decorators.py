import time
import os
from decorator import decorator
import pandas as pd

from wasabi import Printer

wait_time = 10


@decorator
def monitor(f, *args, **kwargs):
    msg = Printer()
    tic = time.time()
    with msg.loading("Working hard..."):
        try:
            value = f(*args, **kwargs)
            toc = time.time()
            msg.good(f"Done! (Took {round(toc - tic)}s)")
        except Exception as e:
            raise (e)
    return value


@decorator
def timer(f, wait=wait_time, *args, **kwargs):
    msg = Printer()
    with msg.loading("Take a deep breath..."):
        time.sleep(wait)
    msg.good("")
    try:
        value = f(*args, **kwargs)
    except Exception as e:
        raise (e)
    return value


@decorator
def load_or_persist(f, fio=None, *args, **kwargs):
    msg = Printer()
    if os.path.isfile(fio):
        result = pd.read_csv(fio, index_col=0)
        msg.info(f"Already persisted. Loaded from file {fio}")
    else:
        result = f(*args, **kwargs)
        result.to_csv(fio)
        msg.info(f"Loaded from the cloud. Persisted in {fio}")
    return result
