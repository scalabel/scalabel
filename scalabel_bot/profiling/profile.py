import os
import time
from argparse import ArgumentParser
from typing import OrderedDict
from colorama import Back, Style

from scalabel_bot.common.consts import (
    LATENCY_THRESHOLD,
    TIMING_LOG_FILE,
)


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description="FsDet demo for builtin models")

    parser.add_argument(
        "--sort",
        type=str,
        default="chrono",
        help=(
            "How profiling results are sorted: 'asc', 'desc', or 'chrono'."
            " Default is 'chrono'"
        ),
    )
    parser.add_argument(
        "--agg",
        action="store_true",
        default=False,
        help=(
            "Whether to aggregate profiling results based on function. Default"
            " is False."
        ),
    )
    return parser


def launch():
    try:
        args: ArgumentParser = get_parser().parse_args()
        last_modified_time = -1

        while True:
            stats = os.stat(TIMING_LOG_FILE)
            if last_modified_time != stats.st_mtime:
                last_modified_time = stats.st_mtime

                with open(
                    file=TIMING_LOG_FILE, mode="r", encoding="utf-8"
                ) as f:
                    timing_list = [line.split("'") for line in f.readlines()]
                    if args.agg:
                        timing_dict = OrderedDict()
                        for timing in timing_list:
                            t = float(timing[2].strip().split(" ")[0])
                            if timing[1] not in timing_dict:
                                timing_dict[timing[1]] = (1, t)
                            else:
                                timing_dict[timing[1]] = (
                                    timing_dict[timing[1]][0] + 1,
                                    timing_dict[timing[1]][1] + t,
                                )

                        for fn, (count, total_time) in timing_dict.items():
                            timing_dict[fn] = (
                                round((total_time / count), 6),
                                count,
                            )

                        if args.sort == "asc":
                            sorted_keys = sorted(
                                timing_dict, key=timing_dict.get
                            )
                        elif args.sort == "desc":
                            sorted_keys = sorted(
                                timing_dict, key=timing_dict.get, reverse=True
                            )
                        sorted_list = (
                            [
                                (key, timing_dict[key][0], timing_dict[key][1])
                                for key in sorted_keys
                            ]
                            if args.sort != "chrono"
                            else [
                                (key, val[0], val[1])
                                for key, val in timing_dict.items()
                            ]
                        )

                        os.system("clear")
                        print(
                            f"{'Function':<45}{'Avg Time (ms)':>17}"
                            f"{'Occurrences':>20}\n"
                        )

                        for fn, timing, occ in sorted_list:
                            if timing <= LATENCY_THRESHOLD:
                                color = Back.GREEN
                            elif (
                                LATENCY_THRESHOLD
                                < timing
                                <= LATENCY_THRESHOLD * 5
                            ):
                                color = Back.YELLOW
                            else:
                                color = Back.RED

                            timing = f"{(color + str(timing) + Style.RESET_ALL):>25}"
                            print(f"{fn:<45} {timing:>17} {occ:>19}")
                    else:
                        if args.sort == "asc":
                            timing_list.sort(
                                key=lambda x: float(
                                    x[2].strip().split(" ")[0]
                                ),
                            )
                        elif args.sort == "desc":
                            timing_list.sort(
                                key=lambda x: float(
                                    x[2].strip().split(" ")[0]
                                ),
                                reverse=True,
                            )

                        os.system("clear")
                        print(
                            f"{'Timestamp':<41}{'Function':<45}"
                            f"{'Time (ms)':>17}\n"
                        )

                        for timing in timing_list:
                            t = float(timing[2].strip().split(" ")[0])

                            if t <= LATENCY_THRESHOLD:
                                color = Back.GREEN
                            elif (
                                LATENCY_THRESHOLD < t <= LATENCY_THRESHOLD * 5
                            ):
                                color = Back.YELLOW
                            else:
                                color = Back.RED

                            timing[
                                2
                            ] = f"{(color + str(t) + Style.RESET_ALL):>25}"
                            print(f"{timing[0]} {timing[1]:<45} {timing[2]}")

            time.sleep(1)

    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    launch()
