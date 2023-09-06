from data.dataset import load_dataset
from torch.utils.data import DataLoader
from utils import setup
from options import Options
import torch
from factory import setup_pipeline

args = Options().parse()
config = setup(args)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if config.debug:
    config.batch_size = 5
    config.val_interval = 5
    config.augment = False
    config.dropout = 0

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

for batch in train_loader:
    if config.use_time_features:
        X, targets, padding_masks, X_time, y_time = batch
        X_time = X_time.to(device)
        y_time = y_time.to(device)
