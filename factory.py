def optimizer_factory(config, model):
    if config.model.name == "unsupervised_transformer":
        # initialize optimizer and regularization
        if config["global_reg"]:
            weight_decay = config["l2_reg"]
            output_reg = None
        else:
            weight_decay = 0
            output_reg = config["l2_reg"]

        optim_class = get_optimizer(config["optimizer"])
        optimizer = optim_class(
            model.parameters(), lr=config["lr"], weight_decay=weight_decay
        )
        return optimizer

    elif config.model.name == "supervised_transformer":
        optimizer = NoamOpt(
            model_size=config.model.d_model,
            factor=1,
            warmup=4000,
            optimizer=torch.optim.Adam(
                model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9
            ),
        )
        return optimizer

    elif config.model.name == "supervised_fedformer":
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
        return optimizer

    else:
        raise ValueError("No optimizer specified for this configuration.")


def pipeline_factory(config):
    """For the task specified in the configuration returns the corresponding combination of
    Dataset class, collate function and Runner class."""

    task = config["task"]

    if task == "imputation":
        return (
            partial(
                ImputationDataset,
                mean_mask_length=config["mean_mask_length"],
                masking_ratio=config["masking_ratio"],
                mode=config["mask_mode"],
                distribution=config["mask_distribution"],
                exclude_feats=config["exclude_feats"],
            ),
            collate_unsuperv,
            UnsupervisedRunner,
        )
    if task == "classification":
        return ClassificationDataset, collate_superv, SupervisedRunner
    else:
        raise NotImplementedError("Task '{}' not implemented".format(task))


def model_factory(config):
    task = config["task"]
    feat_dim = 12  # dimensionality of data features
    # data windowing is used when samples don't have a predefined length or the length is too long
    max_seq_len = (
        config["data_window_len"]
        if config["data_window_len"] is not None
        else config["max_seq_len"]
    )
    if max_seq_len is None:
        try:
            max_seq_len = 5000
        except AttributeError as x:
            print(
                "Data class does not define a maximum sequence length, so it must be defined with the script argument `max_seq_len`"
            )
            raise x

    if task == "imputation":
        if config.model.name == "transformer":
            return TSTransformerEncoder(
                feat_dim,
                max_seq_len,
                config["d_model"],
                config["num_heads"],
                config["num_layers"],
                config["dim_feedforward"],
                dropout=config["dropout"],
                pos_encoding=config["pos_encoding"],
                activation=config["activation"],
                norm=config["normalization_layer"],
                freeze=config["freeze"],
            )

    if task == "classification":
        num_labels = (
            27 if task == "classification" else None
        )  # dimensionality of labels
        if config.model.name == "unsupervised_transformer":
            return TSTransformerEncoderClassifier(
                feat_dim,
                max_seq_len,
                config["d_model"],
                config["num_heads"],
                config["num_layers"],
                config["dim_feedforward"],
                num_classes=num_labels,
                dropout=config["dropout"],
                pos_encoding=config["pos_encoding"],
                activation=config["activation"],
                norm=config["normalization_layer"],
                freeze=config["freeze"],
            )
        elif config.model.name == "supervised_transformer":
            return CTN(
                d_model=config.model.d_model,
                nhead=config.model.nhead,
                d_ff=config.model.d_ff,
                num_layers=config.model.num_layers,
                num_classes=config.data.num_classes,
            )
        elif config.model.name == "supervised_fedformer":
            return FEDformer(config)
    else:
        raise ValueError("Model class for task '{}' does not exist".format(task))
