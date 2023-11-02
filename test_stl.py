from statsmodels.tsa.seasonal import STL, seasonal_decompose
from data.dataset import load_dataset
from torch.utils.data import DataLoader
from utils import setup
from options import Options
from factory import setup_pipeline
import matplotlib.pyplot as plt
from utils import seed_everything
from models.patch_tst_decomposition.decomposition import series_decomp

seed_everything()
args = Options().parse()
config = setup(args)

config.use_time_features = True
config.seq_len = 1024

if config.seq_len is not None:
    max_len = config.seq_len
else:
    max_len = config.window * config.fs

train_dataset, val_dataset, test_dataset = load_dataset(config)

dataset_class, collate_fn, runner_class = setup_pipeline(config)

train_dataset = dataset_class(train_dataset)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=config.batch_size,
    shuffle=False if config.debug else True,
    num_workers=config.num_workers,
    pin_memory=True,
    collate_fn=lambda x: collate_fn(x, max_len=max_len),
)

# series_decomp_week = series_decomp(7 * 24 + 1)
series_decomp_day = series_decomp(25)

for i, batch in enumerate(train_loader):
    X, targets, padding_masks, X_time, y_time = batch

    for d in range(X.shape[-1]):
        # decomposition = seasonal_decompose(
        #     X[0, :, d], model="additive", period=7 * 24, extrapolate_trend=0
        # )

        # fig, axs = plt.subplots(4, 1, figsize=(25, 10), layout="constrained")

        # axs[0].plot(decomposition.trend)
        # axs[0].set_title("Trend")
        # axs[1].plot(decomposition.seasonal)
        # axs[1].set_title("Seasonal")
        # axs[2].plot(decomposition.resid)
        # axs[2].set_title("Residual")
        # axs[3].plot(decomposition.observed)
        # axs[3].set_title("Observed")

        # plt.savefig(f"stl_{config.dataset}_{i}_{d}.png")

        # plt.close()

        # trend_week, residual = series_decomp_week(X)
        trend_day, residual = series_decomp_day(X)

        fig, axs = plt.subplots(4, 1, figsize=(25, 10), layout="constrained")

        # axs[0].plot(trend_week[0, :, d])
        # axs[0].set_title("Trend")
        axs[1].plot(trend_day[0, :, d])
        axs[1].set_title("Seasonal")
        axs[2].plot(residual[0, :, d])
        axs[2].set_title("Residual")
        axs[3].plot(X[0, :, d])
        axs[3].set_title("Observed")

        plt.savefig(f"stl_{config.dataset}_{i}_{d}.png")

        plt.close()
