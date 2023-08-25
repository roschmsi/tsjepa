import torch
from models.patch_tst.model import PatchTST
import matplotlib.pyplot as plt
import math

model = PatchTST(
    c_in=12,
    c_out=27,
    num_patch=62,
    patch_len=16,
    num_layers=8,
    num_heads=8,
    d_model=128,
    d_ff=256,
    dropout=0.1,
    shared_embedding=True,
    norm="BatchNorm",
    activation="gelu",
    pe="sincos",
    learn_pe=False,
    ch_token=False,
    cls_token=False,
    task="pretraining",
    head_dropout=0,
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0001,
    weight_decay=0.01,
)

# scheduler = torch.optim.lr_scheduler.StepLR(
#     optimizer=optimizer, step_size=10, gamma=0.1
# )

# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#     optimizer=optimizer, T_max=100  # config.epochs * iters_per_epoch
# )

scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer=optimizer, T_0=50
)

steps = []
lr_values = []

for i in range(200):
    steps.append(i)
    lr_values.append(scheduler.get_last_lr()[0])
    scheduler.step()

plt.plot(steps, lr_values)
plt.savefig("LRscheduler.png")

warmup_epochs = 10
arg_lr = 0.0001
epochs = 100

steps = []
lr_values = []

for epoch in range(100):
    if epoch < warmup_epochs:
        lr = arg_lr * epoch / warmup_epochs
    else:
        lr = 0 + (arg_lr - 0) * 0.5 * (
            1.0 + math.cos(math.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs))
        )
    steps.append(epoch)
    lr_values.append(lr)

plt.plot(steps, lr_values)
plt.savefig("LRscheduler.png")
