from typing import Any, Dict
import hydra
import omegaconf
from rul_datasets.reader.cmapss import CmapssReader
from torch.utils.data import Dataset
from nn_core.common import PROJECT_ROOT


class CMAPSSDataset(Dataset):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.kwargs = kwargs
        # CMAPSS reader
        # self.cmapss = CmapssReader(
        #     fd=kwargs["fd"],
        #     window_size=kwargs["window_size"],
        #     max_rul=kwargs["max_rul"],
        #     percent_broken= kwargs["percent_broken"],
        #     percent_fail_runs=kwargs["percent_fail_runs"],
        #     feature_select=kwargs["feature_select"],
        #     truncate_val=kwargs["truncate_val"],
        # )
        self.cmapss = CmapssReader(**kwargs)

    @property
    def num_features(self) -> Dict[str, Any]:
        return {"num_features": len(self.cmapss._DEFAULT_CHANNELS)}

    def prepare_data(self) -> None:
        self.cmapss.prepare_data()
        print(f"{self.__class__.__name__}: prepared data")

    def __repr__(self) -> str:
        return f"CMAPSS Dataset(fd={self.kwargs['fd']}, window_size={self.kwargs['window_size']}, \
             max_rul={self.kwargs['max_rul']})"


@hydra.main(version_base=None, config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig) -> None:
    """Debug main to quickly develop the Dataset.

    Args:
        cfg: the hydra configuration
    """
    _: Dataset = hydra.utils.instantiate(cfg.nn.data.datasets.train, split="train", _recursive_=False)


if __name__ == "__main__":
    main()
