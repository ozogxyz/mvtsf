import logging
from typing import Any, List, Optional, Sequence, Tuple, Union

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
from torch.optim import Optimizer
from torchmetrics import MeanMetric, MinMetric
from torchmetrics.regression.mse import MeanSquaredError

from nn_core.common import PROJECT_ROOT
from nn_core.model_logging import NNLogger

from mvtsf.data.datamodule import MetaData
from mvtsf.modules.module import Module

pylogger = logging.getLogger(__name__)


class PLModule(pl.LightningModule):
    """Example of LightningModule for CMAPSS-RUL estimation.

        A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init)
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """
    _logger: NNLogger

    def __init__(self, metadata: Optional[MetaData] = None, *args, **kwargs) -> None:
        super().__init__()
        # Populate self.hparams with args and kwargs automagically!
        # this line allows to access init params with 'self.hparams' attribute
        # We want to skip metadata since it is saved separately by the NNCheckpointIO object.
        # Be careful when modifying this instruction. If in doubt, don't do it :]
        self.save_hyperparameters(logger=False, ignore=("metadata",)) # TODO metric name none

        # model
        self.model = Module(num_features=metadata.num_features, conv_out=32, lstm_hidden=32)

        # loss function
        self.criterion = torch.nn.MSELoss()

        # metric objects for calculating and averaging RMSE across batches
        self.train_rmse = MeanSquaredError(squared=False)
        self.val_rmse = MeanSquaredError(squared=False)
        self.test_rmse = MeanSquaredError(squared=False)

        # for averaging loss across bathces
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best validation metric (lowest RMSE) so far
        self.val_rmse_best = MinMetric()

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def on_train_start(self) -> None:
        # reset metrics after sanity check
        self.val_rmse_best.reset()

    def _step(self, batch: Any):
        x, y = batch
        preds = self.forward(x)
        loss = self.criterion(preds, y)
        self.train_loss(loss.item())
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        loss, preds, targets = self._step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_rmse(preds, targets)
        self.log_dict(
            {"train/loss": self.train_loss, "train/rmse": self.train_rmse},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_step(self, batch: Any, batch_idx: int) -> Any:
        loss, preds, targets = self._step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_rmse(preds, targets)
        self.log_dict(
            {"val/loss": self.val_loss, "val/rmse": self.val_rmse},
        )

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        rmse = self.val_rmse.compute()
        self.val_rmse_best(rmse)

        # use `.compute()` otherwise it would be reset by lightning after each epoch
        self.log("val/rmse_best", self.val_rmse_best.compute(), prog_bar=False)

    def test_step(self, batch: Any, batch_idx: int) -> Any:
        loss, preds, targets = self._step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_rmse(preds, targets)
        self.log_dict(
            {"test/loss": self.test_loss, "test/rmse": self.test_rmse},
        )

        return {"loss": loss, "preds": preds, "targets": targets}

    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Tuple[Sequence[Optimizer], Sequence[Any]]]:
        opt = hydra.utils.instantiate(self.hparams.optimizer, params=self.parameters(), _convert_="partial")
        if "lr_scheduler" not in self.hparams:
            return [opt]
        scheduler = hydra.utils.instantiate(self.hparams.lr_scheduler, optimizer=opt)
        return [opt], [scheduler]



@hydra.main(version_base=None, config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig) -> None:
    """Debug main to quickly develop the Lightning Module.

    Args:
        cfg: the hydra configuration
    """
    _: pl.LightningModule = hydra.utils.instantiate(
        cfg.model,
        optim=cfg.optim,
        _recursive_=False,
    )


if __name__ == "__main__":
    main()
