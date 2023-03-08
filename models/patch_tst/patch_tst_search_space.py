from ray import tune
import numpy as np


def get_patch_tst_search_space(config):
    config.lr = tune.loguniform(1e-5, 1e-3)
    config.weight_decay = tune.loguniform(1e-5, 1e-2)

    config.d_model = tune.sample_from(lambda _: 2 ** np.random.randint(6, 10))
    config.d_ff = tune.sample_from(lambda spec: 2 * spec.config.d_model)
    config.num_layers = tune.randint(4, 17)
    config.num_heads = tune.choice([8, 16])
    # config.dropout = tune.uniform(0, 0.2)
    # config.head_dropout = tune.uniform(0, 0.2)

    config.patch_len = tune.sample_from(lambda _: 2 ** np.random.randint(3, 6))
    config.stride = tune.sample_from(lambda spec: int(spec.config.patch_len // 2))

    return config
