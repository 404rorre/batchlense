"""Timing utilities for profiling."""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from time import time
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def timer_func(func: F) -> F:
    """Log execution time of the wrapped callable."""

    @functools.wraps(func)
    def wrap_func(*args: Any, **kwargs: Any) -> Any:
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        logger.info("Function %r executed in %.4fs", func.__name__, t2 - t1)
        return result

    return wrap_func  # type: ignore[return-value]
