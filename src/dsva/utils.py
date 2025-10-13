import json
import logging
import os
import random
import socket
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any, Iterable, Iterator, Mapping, TypeVar

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


def get_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


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


def record_env(
    run_dir: str, run_id: str, run_cfg: Mapping[str, Any] | None = None
) -> None:
    info = {
        "run_id": run_id,
        "run_dir": run_dir,
        "hostname": socket.gethostname(),
        "time": get_timestamp(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if run_cfg:
        info.update(dict(run_cfg))
    with open(os.path.join(run_dir, "env.json"), "w") as f:
        json.dump(info, f, indent=2)


def setup_run(seed: int, save_dir: str, run_id: str, args: dict[str, Any]) -> str:
    """
    Setup fixed seed, experiment run directory, and logger.

    Args:
        seed: Random seed for reproducibility.
        save_dir: Directory to save model checkpoints and other training intermediates.
        run_id: Unique identifier for the experiment run.
        args: Additional arguments to record in the environment file.
    """
    set_seed(seed)

    # make run directory if not exist
    run_dir = os.path.join(save_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    # record run configuration
    run_cfg = {
        key: asdict(value) if is_dataclass(value) else value
        for key, value in args.items()
    }

    # record environment information
    record_env(run_dir, run_id, run_cfg)

    # setup logger
    logger = get_logger()
    logger.info(f'Starting run "{run_id}"')
    return run_dir, logger


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
