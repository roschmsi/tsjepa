import matplotlib.pyplot as plt

ecg_results_finetuning = {
    "PatchTST": [
        [0.1, 0.3, 0.5, 0.7, 0.9],
        [0.9564, 0.9574, 0.9580, 0.9551, 0.9511],
        "x",
    ],
    "PatchTST-T": [[0.3, 0.5, 0.7, 0.9], [0.9550, 0.9539, 0.9555, 0.9481], "x"],
    "PatchTST-TC": [[0.3, 0.5, 0.7], [0.9519, 0.9545, 0.9524], "x"],
    "MAE": [[0.1, 0.3, 0.5, 0.7, 0.9], [0.9386, 0.9463, 0.9439, 0.9466, 0.9444], "o"],
    "MAE-T": [[0.3, 0.5, 0.7, 0.9], [0.9370, 0.9390, 0.9349, 0.9329], "o"],
    "MAE-TC": [[0.5, 0.7], [0.9342, 0.9413], "o"],
}

plt.figure(figsize=(10, 4))
for k, v in ecg_results_finetuning.items():
    plt.plot(v[0], v[1], marker=v[2], linestyle="-", label=k)

plt.xlabel("masking ratio")
plt.ylabel("AUROC")
plt.legend(loc="right")
plt.tight_layout()
plt.savefig("ecg_results_finetuning.png")
plt.close()

# ettm1_results_finetuning = {
#     "PatchTST": [[0.1, 0.3, 0.5, 0.7, 0.9], [0.9, 0.95, 0.95, 0.95, 0.95], "x"],
#     "PatchTST-T": [[0.1, 0.3, 0.5, 0.7, 0.9], [0.9, 0.95, 0.95, 0.95, 0.9], "x"],
#     "PatchTST-TC": [[0.1, 0.3, 0.5, 0.7, 0.9], [0.9, 0.95, 0.95, 0.9, 0.95], "x"],
#     "MAE": [[0.1, 0.3, 0.5, 0.7, 0.9], [0.9386, 0.9463, 0.9439, 0.9466, 0.9444], "o"],
#     "MAE-T": [[0.3, 0.5, 0.7], [0.9370, 0.9390, 0.9349], "o"],
#     "MAE-TC": [[0.5, 0.7], [0.9342, 0.9413], "o"],
# }


# plt.figure(figsize=(10, 4))
# for k, v in ettm1_results_finetuning.items():
#     plt.plot(v[0], v[1], marker=v[2], linestyle="-", label=k)

# plt.xlabel("masking ratio")
# plt.ylabel("AUROC")
# plt.legend(loc="right")
# plt.tight_layout()
# plt.savefig("ettm1_results_finetuning.png")
# plt.close()
