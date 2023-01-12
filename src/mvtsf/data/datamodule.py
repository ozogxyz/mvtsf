from cmath import log
import logging
from functools import cached_property
from pathlib import Path
from typing import Mapping

import hydra
import omegaconf
import pytorch_lightning as pl
from omegaconf import DictConfig
from rul_datasets.core import RulDataModule

from nn_core.common import PROJECT_ROOT

pylogger = logging.getLogger(__name__)


class MetaData:
    def __init__(self, num_features: Mapping[str, int]):
        """The data information the Lightning Module will be provided with.

        This is a "bridge" between the Lightning DataModule and the Lightning Module.
        There is no constraint on the class name nor in the stored information, as long as it exposes the
        `save` and `load` methods.

        The Lightning Module will receive an instance of MetaData when instantiated,
        both in the train loop or when restored from a checkpoint.

        This decoupling allows the architecture to be parametric (e.g. in the number of classes) and
        DataModule/Trainer independent (useful in prediction scenarios).
        MetaData should contain all the information needed at test time, derived from its train dataset.

        Examples are the class names in a classification task or the vocabulary in NLP tasks.
        MetaData exposes `save` and `load`. Those are two user-defined methods that specify
        how to serialize and de-serialize the information contained in its attributes.
        This is needed for the checkpointing restore to work properly.

        Args:
            class_vocab: association between class names and their indices
        """
        # example
        self.num_features: Mapping[str, int] = num_features

    def save(self, dst_path: Path) -> None:
        """Serialize the MetaData attributes into the zipped checkpoint in dst_path.

        Args:
            dst_path: the root folder of the metadata inside the zipped checkpoint
        """
        pylogger.debug(f"Saving MetaData to '{dst_path}'")

        # example
        (dst_path / "num_features.tsv").write_text("\n".join(f"{key}\t{value}" for key, value in self.num_features.items()))

    @staticmethod
    def load(src_path: Path) -> "MetaData":
        """Deserialize the MetaData from the information contained inside the zipped checkpoint in src_path.

        Args:
            src_path: the root folder of the metadata inside the zipped checkpoint

        Returns:
            an instance of MetaData containing the information in the checkpoint
        """
        pylogger.debug(f"Loading MetaData from '{src_path}'")

        # example
        lines = (src_path / "num_features.tsv").read_text(encoding="utf-8").splitlines()

        num_features = {}
        for line in lines:
            key, value = line.strip().split("\t")
            num_features[key] = value

        return MetaData(
            num_features=num_features,
        )


class CMAPSSDataModule(RulDataModule):
    def __init__(
        self,
        dataset: DictConfig,
        batch_size: DictConfig,
        num_features: DictConfig,
    ):
        print(omegaconf.OmegaConf.to_yaml(dataset))
        reader = hydra.utils.instantiate(dataset.reader)
        self.num_features = num_features
        batch_size = batch_size
        super().__init__(reader=reader, batch_size=batch_size)  # type: ignore
        self.save_hyperparameters(logger=False)
    @cached_property
    def metadata(self) -> MetaData:
        """Data information to be fed to the Lightning Module as parameter.

        Examples are vocabularies, number of classes...

        Returns:
            metadata: everything the model should know about the data, wrapped in a MetaData object.
        """
        return MetaData(num_features=self.num_features)


@hydra.main(version_base=None, config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig) -> None:
    """Debug main to quickly develop the DataModule.

    Args:
        cfg: the hydra configuration
    """
    _: pl.LightningDataModule = hydra.utils.instantiate(cfg.nn.data.dataset, _recursive_=False)


if __name__ == "__main__":
    main()
