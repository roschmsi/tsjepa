from data.dataset import create_patch


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_classwise_distribution(
    encoder,
    data_loader,
    device,
    d_model,
    num_classes,
    revin=None,
    tb_writer=None,
    mode=None,
    epoch=None,
    supervised=False,
    patch_len=0,
    stride=0,
):
    encoder.eval()
    df = None

    with torch.no_grad():
        for X in data_loader:
            if df is not None and df.shape[0] > 100000:
                break

            X = X.to(device)
            if revin is not None:
                X = revin(X, "norm")

            X = create_patch(X, patch_len=patch_len, stride=stride)
            X = X.squeeze()
            X_enc = encoder(X)["encoder_out"]
            bs, seq_len, d_model = X_enc.shape
            X_enc = X_enc.reshape(bs * seq_len, d_model)
            # X_enc = X_enc.mean(dim=1)

            df_z = pd.DataFrame(X_enc.cpu()).astype("float")

            if num_classes == 1:
                df_y = pd.DataFrame(torch.zeros(X.shape[0]), columns=["y"]).astype(
                    "int"
                )
            else:
                df_y = pd.DataFrame(y.cpu(), columns=["y"]).astype("int")

            df_batch = pd.concat([df_y, df_z], axis=1)
            df = pd.concat([df, df_batch], axis=0)

    # plot per label
    for dim in range(d_model):
        plt.figure(figsize=(8, 4))

        for cls in range(num_classes):
            # Plot KDE plot for each class
            fig = sns.kdeplot(
                x=dim,
                data=df[df["y"] == cls].reset_index(),
                label=cls,
            )
            fig.set(xlabel=None)
            fig.set(ylabel=None)

        plt.xlim([-2.0, 2.0])
        plt.legend()
        plt.tight_layout()

        if tb_writer is not None:
            tb_writer.add_figure(
                f"classwise distribution {dim}/{mode}", plt.gcf(), epoch
            )

        plt.close()

    encoder.train()


def plot_cov_matrix(cov_matrix):
    plt.close()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cov_matrix.detach(), vmin=-1, vmax=1)
    fig.colorbar(cax)
    return fig