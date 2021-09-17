"""tqdm wrapper that can be quiet."""
from typing import Any, Iterable

from tqdm import tqdm as tqdm_original

from .quiet import Quiet


def tqdm(iterable: Iterable[Any], **kwargs: Any) -> Any:  # type: ignore
    """Wrap tqdm to enable quiet execution."""
    if Quiet.get():
        return iterable
    return tqdm_original(iterable, **kwargs)
