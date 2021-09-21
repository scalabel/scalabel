"""Set / get scalabel logging status."""
import logging

from .logger import logger


class Quiet:
    """Define a singleton to support a global Quiet attribute."""

    _is_quiet = False

    @classmethod
    def set(cls, value: bool) -> None:
        """Set quiet value."""
        if value:
            cls._is_quiet = True
            logger.setLevel(logging.CRITICAL)
        else:
            cls._is_quiet = False
            logger.setLevel(logging.INFO)

    @classmethod
    def get(cls) -> bool:
        """Get quiet value."""
        return cls._is_quiet


def mute(yes_or_no: bool) -> None:
    """Enable quiet option."""
    Quiet.set(yes_or_no)


def is_quiet() -> bool:
    """Get quiet state."""
    return Quiet.get()


if is_quiet():
    logger.setLevel(logging.CRITICAL)
else:
    logger.setLevel(logging.INFO)
