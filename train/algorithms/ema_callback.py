# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""
Weight Averaging Callback
^^^^^^^^^^^^^^^^^^^^^^^^^
"""

import itertools
from copy import deepcopy
from typing import Any, Optional, Union

import torch
from torch.optim.swa_utils import AveragedModel
from typing_extensions import override

import lightning.pytorch as pl
from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.utilities.model_helpers import is_overridden
from lightning.pytorch.utilities.rank_zero import rank_zero_info, rank_zero_warn
from lightning.pytorch.utilities.types import STEP_OUTPUT


class WeightAveraging(Callback):
    """Copied from https://github.com/Lightning-AI/pytorch-lightning/blob/master/src/lightning/pytorch/callbacks/weight_averaging.py A callback that updates an averaged model for Stochastic Weight Averaging (SWA) or Exponential Moving Average (EMA) after each training step."""

    def __init__(
        self,
        device: Optional[Union[torch.device, str, int]] = None,
        use_buffers: bool = True,
        **kwargs: Any,
    ) -> None:
        # The default value is a string so that jsonargparse knows how to serialize it.
        """Lightning callback maintaining an averaged copy of the model weights."""
        if isinstance(device, str):
            self._device: Optional[Union[torch.device, int]] = torch.device(device)
        else:
            self._device = device
        self._use_buffers = use_buffers
        self._kwargs = kwargs

        self._average_model: Optional[AveragedModel] = None

        # Number of optimizer steps taken, when the average model was last updated. Initializing this with zero ensures
        # that self.should_update() will be first called after the first optimizer step, which takes place after N
        # batches when using accumulate_grad_batches=N.
        self._latest_update_step = 0
        # The epoch after which the average model was last updated. The first epoch is 0, so initializing this to a
        # negative value means that if self.should_update(epoch_idx=0) returns True, the first update is after the first
        # epoch.
        self._latest_update_epoch = -1

    def should_update(self, step_idx: Optional[int] = None, epoch_idx: Optional[int] = None) -> bool:
        """Called after every optimizer step and after every training epoch to check whether the average model should be updated."""
        return step_idx is not None

    @override
    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        """Called when fit, validate, test, predict, or tune begins."""
        if stage == "fit":
            device = self._device or pl_module.device

            # If the configure_model hook is overridden, call it to create the layers before constructing the
            # AveragedModel. However, sharding will not be done and a warning will be issued.
            if is_overridden("configure_model", pl_module):
                rank_zero_warn(
                    "You're using the WeightAveraging callback with a model that overrides the configure_model "
                    "callback. WeightAveraging doesn't support sharding model layers, so you may run out of memory."
                )
                pl_module.configure_model()

            self._average_model = AveragedModel(
                model=pl_module, device=device, use_buffers=self._use_buffers, **self._kwargs
            )

    @override
    def on_train_batch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        """Called when a training batch ends."""
        # trainer.global_step is the number of optimizer steps taken so far, i.e. 1 after the first optimizer step. To
        # make step_idx consistent with epoch_idx, we'll pass a zero-based index.
        step_idx = trainer.global_step - 1
        if (trainer.global_step > self._latest_update_step) and self.should_update(step_idx=step_idx):
            assert self._average_model is not None
            self._average_model.update_parameters(pl_module)
            self._latest_update_step = trainer.global_step

    @override
    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when a training epoch ends."""
        if (trainer.current_epoch > self._latest_update_epoch) and self.should_update(epoch_idx=trainer.current_epoch):
            assert self._average_model is not None
            self._average_model.update_parameters(pl_module)
            self._latest_update_epoch = trainer.current_epoch

    @override
    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when training ends."""
        assert self._average_model is not None
        self._copy_average_to_current(pl_module)

    @override
    def on_validation_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when a validation epoch begins."""
        if self._average_model is not None:
            self._swap_models(pl_module)

    @override
    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when a validation epoch ends."""
        if self._average_model is not None:
            self._swap_models(pl_module)

    @override
    def state_dict(self) -> dict[str, Any]:
        """Called when saving a checkpoint."""
        return {"latest_update_step": self._latest_update_step}

    @override
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Called when loading a checkpoint."""
        self._latest_update_step = state_dict["latest_update_step"]

    @override
    def on_save_checkpoint(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: dict[str, Any]
    ) -> None:
        """Called when saving a checkpoint."""
        if self._average_model is None:
            rank_zero_info(
                "You're using the WeightAveraging callback, but saving a checkpoint outside the 'fit' stage. The state "
                "of the WeightAveraging callback won't be saved in the checkpoint. If training has finished, the "
                "average model parameters will be saved to the state_dict in the checkpoint."
            )
        else:
            average_model_state = self._average_model.state_dict()
            checkpoint["current_model_state"] = checkpoint["state_dict"]
            # Truncate the "module." prefix (the first 7 characters) from the names of the variables in the
            # AveragedModel state.
            checkpoint["state_dict"] = {
                name[7:]: value for name, value in average_model_state.items() if name.startswith("module.")
            }
            checkpoint["averaging_state"] = {
                name: value for name, value in average_model_state.items() if not name.startswith("module.")
            }

    @override
    def on_load_checkpoint(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: dict[str, Any]
    ) -> None:
        """Called when loading a model checkpoint."""
        if self._average_model is None:
            rank_zero_warn(
                "You're using the WeightAveraging callback, but loading a checkpoint outside the 'fit' stage. The "
                "WeightAveraging state cannot be restored. If you're using the checkpoint for prediction or testing, "
                "you can ignore this warning. To disable the warning, remove the WeightAveraging callback."
            )
        elif ("current_model_state" in checkpoint) and ("averaging_state" in checkpoint):
            rank_zero_info("Found current_model_state in the checkpoint. This will be used to initialize the model.")
            average_model_state = {"module." + name: value for name, value in checkpoint["state_dict"].items()}
            average_model_state |= checkpoint["averaging_state"]
            self._average_model.load_state_dict(average_model_state)
            # The current model state has already been loaded from "state_dict" (which contains the average model
            # weights) at this point, so overwriting "state_dict" in the checkpoint dictionary makes no difference. We
            # have to reload the model state from "current_model_state".
            pl_module.load_state_dict(checkpoint["current_model_state"])
        else:
            rank_zero_warn(
                "The checkpoint was not created with WeightAveraging. Both the current and the average model will be "
                "initialized with state_dict."
            )
            self._average_model.module.load_state_dict(deepcopy(checkpoint["state_dict"]), strict=False)

    def _swap_models(self, pl_module: "pl.LightningModule") -> None:
        """Swaps the parameter values of the current model and the :class:`AveragedModel`."""
        assert self._average_model is not None
        average_params = itertools.chain(self._average_model.module.parameters(), self._average_model.module.buffers())
        current_params = itertools.chain(pl_module.parameters(), pl_module.buffers())
        for average_param, current_param in zip(average_params, current_params):
            tmp = average_param.data.clone()
            average_param.data.copy_(current_param.data)
            current_param.data.copy_(tmp)

    def _copy_average_to_current(self, pl_module: "pl.LightningModule") -> None:
        """Copies the parameter values from the :class:`AveragedModel` to the current model."""
        assert self._average_model is not None
        average_params = itertools.chain(self._average_model.module.parameters(), self._average_model.module.buffers())
        current_params = itertools.chain(pl_module.parameters(), pl_module.buffers())
        for average_param, current_param in zip(average_params, current_params):
            current_param.data.copy_(average_param.data)


class EMAWeightAveraging(WeightAveraging):
    """EMA weight averaging with Crowson warmup schedule."""

    def __init__(
        self,
        update_after_step=0,
        update_every=1,
        inv_gamma=1.0,
        power=2/3,
        min_value=0.0,
        max_value=0.9999,
        use_buffers=False,
    ):
        """Weight averaging with an EMA decay schedule (warmup via inv_gamma/power)."""
        super().__init__(avg_fn=self.ema_avg_fn, use_buffers=use_buffers)

        self.update_after_step = update_after_step
        self.update_every = max(1, int(update_every))
        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value
        self.max_value = max_value

    def get_decay(self, step):
        """
        Crowson EMA warmup schedule
        """
        step = max(0, step - self.update_after_step - 1)
        val = 1 - (1 + step / self.inv_gamma) ** -self.power
        if step <= 0:
            return 0.0
        return max(self.min_value, min(val, self.max_value))

    @torch.no_grad()
    def ema_avg_fn(self, averaged_param, new_param, step):
        """
        Custom EMA averaging function
        """
        decay = self.get_decay(step)
        return averaged_param * decay + new_param * (1 - decay)

    def should_update(self, step_idx=None, **kwargs):
        """
        Control when EMA starts updating
        """
        return (
            step_idx is not None
            and step_idx >= self.update_after_step
            and (step_idx - self.update_after_step) % self.update_every == 0
        )
