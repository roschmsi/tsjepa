import torch


def init_optimizer(model, lr, weight_decay, epochs):
    param_groups = [
        {
            "params": (
                p
                for n, p in model.named_parameters()
                if ("bias" not in n) and (len(p.shape) != 1)
            )
        },
        {
            "params": (
                p
                for n, p in model.named_parameters()
                if ("bias" in n) or (len(p.shape) == 1)
            ),
            "WD_exclude": True,
            "weight_decay": 0,
        },
    ]

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=lr,
        weight_decay=weight_decay,
    )

    return optimizer


def init_scheduler(optimizer, config):
    if config.scheduler == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=config.epochs
        )
    elif config.scheduler == "CosineAnnealingLRWithWarmup":
        scheduler_1 = torch.optim.lr_scheduler.LinearLR(
            optimizer=optimizer,
            start_factor=config.start_factor,
            total_iters=config.warmup,
        )
        scheduler_2 = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=config.epochs - config.warmup
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[scheduler_1, scheduler_2],
            milestones=[config.warmup],
        )
    elif config.scheduler == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer, step_size=config.step_size, gamma=0.1
        )

    else:
        scheduler = None

    return scheduler
