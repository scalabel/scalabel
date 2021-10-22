"""tqdm wrapper that can be quiet."""
from typing import Any, Iterable

from tqdm import tqdm as tqdm_original

from .quiet import is_quiet


def tqdm(iterable: Iterable[Any], **kwargs: Any) -> Any:  # type: ignore
    """Wrap tqdm to enable quiet execution."""
    if is_quiet():
        return iterable
    return tqdm_original(iterable, **kwargs)
