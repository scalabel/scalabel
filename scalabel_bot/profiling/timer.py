import time

from scalabel_bot.common.consts import Timers

from scalabel_bot.common.logger import logger


def timer(t: Timers):
    """This function shows the execution time of the function object passed"""

    def timer_wrapper(func):
        if t == Timers.PERF_COUNTER:
            execute_timer = time.perf_counter_ns
        elif t == Timers.PROCESS_TIMER:
            execute_timer = time.process_time_ns
        elif t == Timers.THREAD_TIMER:
            execute_timer = time.thread_time_ns
        else:
            logger.error(f"{func.__qualname__!r}: Timer type not supported")

        def timer_func(*args, **kwargs):
            t1 = execute_timer()
            result = func(*args, **kwargs)
            t2 = execute_timer()
            execution_time = (t2 - t1) / 1000000
            func_name = f"'{args[0].__class__.__name__}.{func.__name__}'"
            logger.verbose(f"{func_name:<45} {execution_time:>17.6f} ms")
            return result

        return timer_func

    return timer_wrapper
