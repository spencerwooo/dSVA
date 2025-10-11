import json
import logging
import os
import random
import socket
from datetime import datetime
from typing import Iterable, Iterator, TypeVar

import numpy as np
import torch
from rich.logging import RichHandler

T = TypeVar("T")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler()],
    )
    return logging.getLogger("dsva")


def make_run_dir(save_dir: str, run_id: str) -> str:
    run_dir = os.path.join(save_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def record_env(run_dir: str, run_id: str) -> None:
    info = {
        "run_id": run_id,
        "hostname": socket.gethostname(),
        "time": datetime.now().isoformat(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    with open(os.path.join(run_dir, "env.json"), "w") as f:
        json.dump(info, f, indent=2)


def progress(seq: Iterable[T], desc: str, total: int | None = None) -> Iterator[T]:
    """
    Create a progress bar for a sequence.

    Args:
        seq: Iterable to track progress of.
        desc: Description of the task.
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
        yield from progress.track(seq, total=total, description=desc)
