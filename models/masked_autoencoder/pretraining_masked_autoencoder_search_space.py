from ray import tune
import numpy as np


def get_pretraining_masked_autoencoder_search_space(config):
    config.training.lr = tune.loguniform(1e-5, 1e-6)
    config.training.weight_decay = tune.loguniform(1e-5, 1e-3)

    config.model.enc_d_model = tune.sample_from(lambda _: 2 ** np.random.randint(6, 10))
    config.model.enc_d_ff = tune.sample_from(
        lambda spec: 2 * spec.config.model.enc_d_model
    )
    config.model.enc_num_layers = tune.randint(8, 17)
    config.model.enc_num_heads = tune.choice([8, 16])

    config.model.dec_d_model = tune.sample_from(lambda _: 2 ** np.random.randint(6, 10))
    config.model.dec_d_ff = tune.sample_from(
        lambda spec: 2 * spec.config.model.dec_d_model
    )
    config.model.dec_num_layers = tune.randint(8, 17)
    config.model.dec_num_heads = tune.choice([8, 16])

    # config.model.dropout = tune.uniform(0, 0.2)
    # config.model.head_dropout = tune.uniform(0, 0.2)

    config.model.patch_len = tune.sample_from(lambda _: 2 ** np.random.randint(3, 6))
    config.model.stride = tune.sample_from(
        lambda spec: int(spec.config.model.patch_len)
    )

    config.model.masking_ratio = tune.quniform(0.2, 0.7, 0.1)

    return config
