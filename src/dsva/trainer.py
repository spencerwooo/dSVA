import os
from typing import Callable

import torch
import torch.nn as nn

from dsva.generator import Generator
from dsva.utils import get_logger, progress

logger = get_logger(__name__)


class Trainer:
    """
    A minimal trainer for the dSVA generator.

    N.B.: No validation is performed during the training process.

    Args:
        generator: The dSVA generator to be trained.
        normalize: Image normalization callable.
        eps: The Linf adversarial perturbation budget.
        optimizer: The optimizer for training the generator.
        criterion: The loss function for training the generator.
        device: The device to run the training on.
    """

    def __init__(
        self,
        generator: Generator,
        normalize: Callable[[torch.Tensor], torch.Tensor],
        eps: float,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
    ):
        # training components
        self.generator = generator
        self.normalize = normalize

        # adversarial perturbation budget
        self.eps = eps

        # optimizer and loss function
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def _train_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
        epoch: int,
        num_epochs: int,
        log_every_n_steps: int,
    ) -> float:
        """
        Train for one epoch.

        Args:
            dataloader: Training dataset dataloader.
            epoch: Current running epoch.
            num_epochs: Total number of epochs.
            log_every_n_steps: Log running loss every number of steps.

        Returns:
            Average loss over the current epoch.
        """

        self.generator.train()

        total_loss = 0.0  # accumulate loss over the epoch
        description = f"Epoch {epoch + 1}/{num_epochs}"
        for i, (img, _) in enumerate(progress(dataloader, desc=description)):
            self.optimizer.zero_grad()

            # forward pass
            img = img.to(self.device)
            adv = self.generator(img)

            # project adversarial perturbation to Linf ball
            delta = torch.clamp(adv - img, min=-self.eps, max=self.eps)
            adv = torch.clamp(img + delta, min=0, max=1)

            # compute loss
            loss = self.criterion(
                self.normalize(img),  # feed normalized images
                self.normalize(adv),
            )
            loss.backward()
            self.optimizer.step()

            # log running loss
            total_loss += loss.item()  # track running sum for epoch average
            if (i % log_every_n_steps == 0) or (i == len(dataloader) - 1):
                loss_log = (
                    f"epoch {epoch + 1}/{num_epochs}, "
                    f"step {i + 1}/{len(dataloader)}: loss {loss.item():.6f}"
                )
                logger.info(loss_log)

        # return epoch average loss
        return total_loss / max(1, len(dataloader))

    def _save_checkpoint(self, path: str) -> None:
        torch.save(self.generator.state_dict(), path)
        logger.info(f'Saved model to "{path}"')

    def train(
        self,
        dataloader: torch.utils.data.DataLoader,
        run_dir: str,
        num_epochs: int,
        log_every_n_steps: int,
    ) -> None:
        """
        Training loop.

        Args:
            dataloader: Training dataset dataloader.
            run_dir: Directory to save model checkpoints.
            num_epochs: Total number of epochs.
            log_every_n_steps: Log running loss every number of steps.
        """

        for epoch in range(num_epochs):
            self._train_epoch(dataloader, epoch, num_epochs, log_every_n_steps)

        # save model checkpoint each epoch
        checkpoint_path = os.path.join(run_dir, f"generator_epoch{epoch + 1}.pth")
        self._save_checkpoint(checkpoint_path)
