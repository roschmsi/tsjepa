from ray import tune
import numpy as np


def get_patch_tst_search_space(config):
    config.training.lr = tune.loguniform(1e-5, 1e-3)
    config.training.weight_decay = tune.loguniform(1e-5, 1e-2)

    config.model.d_model = tune.sample_from(lambda _: 2 ** np.random.randint(6, 10))
    config.model.d_ff = tune.sample_from(lambda spec: 2 * spec.config.model.d_model)
    config.model.num_layers = tune.randint(4, 17)
    config.model.num_heads = tune.choice([8, 16])
    # config.model.dropout = tune.uniform(0, 0.2)
    # config.model.head_dropout = tune.uniform(0, 0.2)

    config.model.patch_len = tune.sample_from(lambda _: 2 ** np.random.randint(3, 6))
    config.model.stride = tune.sample_from(
        lambda spec: int(spec.config.model.patch_len // 2)
    )

    return config
