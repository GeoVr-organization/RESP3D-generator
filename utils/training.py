import sys
from typing import Any, Generator

import torch
from torch import Tensor
from torch.optim import Optimizer
from tqdm import trange

from .plot import plot_values


class Trainer:
    def __init__(
        self,
        epochs: int,
        optimizer_cls: type[Optimizer],
        optimizer_params: dict[str, Any],
        scheduler_cls: type = None,
        scheduler_params: dict[str, Any] = None,
        plot_losses: bool = True,
        desc: str = None
    ):
        self._epochs = epochs
        self._params = optimizer_params["params"]
        self._optimizer = optimizer_cls(**optimizer_params)

        if scheduler_cls is not None and scheduler_params is not None:
            self._scheduler = scheduler_cls(
                self._optimizer,
                **scheduler_params
            )
        else:
            self._scheduler = None

        self.loss = None
        self._plot_losses = plot_losses
        self._iter_history = []
        self._loss_history = []
        self._res = max(1, epochs // 300)
        self._desc = desc

    def __iter__(self) -> Generator[int, Tensor, None]:
        for p in self._params:
            p.requires_grad = True

        # Enable gradient computation for the parameters
        with torch.enable_grad():
            # Training
            for i in trange(self._epochs, desc=self._desc, file=sys.stdout):
                # Reset gradients
                self._optimizer.zero_grad()

                # User code entry-point
                yield i

                # Update parameters and learning rates
                self.loss.backward(retain_graph=True)
                if self._scheduler is not None:
                    self._scheduler.step(self.loss)
                self._optimizer.step()

                if self._plot_losses and i % self._res == 0:
                    # Update loss history
                    self._iter_history.append(i)
                    self._loss_history.append(self.loss.item())

        for p in self._params:
            p.requires_grad = False

        # Plot the loss history
        if self._plot_losses:
            x = torch.tensor(self._iter_history)
            y = torch.tensor(self._loss_history)
            plot_values(
                x=x, y=y,
                ylim=[0, x.max()],
                xlim=[0, self._epochs],
                title="Optimization loss"
            )
