from ray import tune
import numpy as np


def get_pretraining_masked_autoencoder_search_space(config):
    config.lr = tune.loguniform(1e-5, 1e-6)
    config.weight_decay = tune.loguniform(1e-5, 1e-3)

    config.enc_d_model = tune.sample_from(lambda _: 2 ** np.random.randint(6, 10))
    config.enc_d_ff = tune.sample_from(lambda spec: 2 * spec.config.enc_d_model)
    config.enc_num_layers = tune.randint(8, 17)
    config.enc_num_heads = tune.choice([8, 16])

    config.dec_d_model = tune.sample_from(lambda _: 2 ** np.random.randint(6, 10))
    config.dec_d_ff = tune.sample_from(lambda spec: 2 * spec.config.dec_d_model)
    config.dec_num_layers = tune.randint(8, 17)
    config.dec_num_heads = tune.choice([8, 16])

    # config.dropout = tune.uniform(0, 0.2)
    # config.head_dropout = tune.uniform(0, 0.2)

    config.patch_len = tune.sample_from(lambda _: 2 ** np.random.randint(3, 6))
    config.stride = tune.sample_from(lambda spec: int(spec.config.patch_len))

    config.masking_ratio = tune.quniform(0.2, 0.7, 0.1)

    return config
