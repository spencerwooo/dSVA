from logging import Logger
from typing import Iterable, Iterator, TypeVar

T = TypeVar("T")


def progress(
    sequence: Iterable[T], description: str, total: int | None = None
) -> Iterator[T]:
    """
    Create a progress bar for a sequence.

    Args:
        sequence: Iterable to track progress of.
        description: Description of the task.
        total: Total number of items in the sequence.

    Yields:
        Items from the tracked sequence.
    """

    from rich.progress import (
        BarColumn,
        Progress,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

    bar_items = [
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        "· ETA",
        TimeRemainingColumn(),
    ]

    progress = Progress(*bar_items)
    with progress:
        yield from progress.track(sequence, description=description, total=total)


def logger(level: str = "INFO") -> Logger:
    """Shared logger instance with rich formatting."""
    import logging

    from rich.logging import RichHandler

    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler()],
    )
    return logging.getLogger("dsva")
