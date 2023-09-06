from gluonts.dataset.repository.datasets import dataset_names, get_dataset
from gluonts.dataset.common import ListDataset
from pathlib import Path

if __name__ == "__main__":
    dataset = get_dataset(
        dataset_name="nn5_weekly",
        path=Path("/storage/user/roschman/datasets/gluonts"),
        regenerate=True,
        prediction_length=24,
    )

    dataset.train[0]
