import os
import time

from scalabel_bot.common.consts import DEBUG_LOG_FILE


def launch():
    try:
        last_modified_time = -1

        while True:
            stats = os.stat(DEBUG_LOG_FILE)
            if last_modified_time != stats.st_mtime:
                os.system("clear")
                last_modified_time = stats.st_mtime

                with open(
                    file=DEBUG_LOG_FILE, mode="r", encoding="utf-8"
                ) as f:
                    for line in f.readlines():
                        if "timer" not in line:
                            print(line, end="")

            time.sleep(1)

    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    launch()
