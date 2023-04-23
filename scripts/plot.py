import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({"font.size": 16})
plt.rcParams["figure.dpi"] = 300

ecg_results_finetuning = {
    "PatchTST": [
        [0.1, 0.3, 0.5, 0.7, 0.9],
        [0.9564, 0.9574, 0.9580, 0.9551, 0.9511],
        "x",
    ],
    "PatchTST-T": [[0.3, 0.5, 0.7, 0.9], [0.9550, 0.9539, 0.9555, 0.9481], "x"],
    "PatchTST-TC": [[0.3, 0.5, 0.7, 0.9], [0.9519, 0.9545, 0.9524, 0.9486], "x"],
    "MAE": [[0.1, 0.3, 0.5, 0.7, 0.9], [0.9386, 0.9463, 0.9439, 0.9466, 0.9444], "o"],
    "MAE-T": [[0.3, 0.5, 0.7, 0.9], [0.9370, 0.9390, 0.9349, 0.9329], "o"],
    "MAE-TC": [[0.3, 0.5, 0.7, 0.9], [0.9170, 0.9342, 0.9413, 0.9395], "o"],
}

plt.figure(figsize=(10, 5))
for k, v in ecg_results_finetuning.items():
    plt.plot(v[0], v[1], marker=v[2], linestyle="-", label=k)

plt.xlabel("Masking ratio")
plt.ylabel("AUROC")
plt.legend(loc="lower right", ncol=2)
plt.tight_layout()
plt.savefig("ecg_results_finetuning.png")
plt.close()

barWidth = 0.15
fig = plt.subplots(figsize=(8, 6))

mae = [0.9414, 0.9439, 0.9408]
mae_t = [0.9400, 0.9390, 0.9360]
mae_tc = [0.9371, 0.9413, 0.9220]

br1 = [0, 1, 2]
br2 = [x + barWidth + 0.05 for x in br1]
br3 = [x + barWidth + 0.05 for x in br2]

plt.bar(br1, mae, color="#bad7e7", width=barWidth, edgecolor="black", label="MAE")
plt.bar(br2, mae_t, color="#ffffff", width=barWidth, edgecolor="black", label="MAE-T")
plt.bar(br3, mae_tc, color="#f8d984", width=barWidth, edgecolor="black", label="MAE-TC")

plt.xlabel("Number of decoder layers")
plt.ylabel("AUROC")
plt.xticks([r + barWidth for r in range(len(mae))], ["1", "2", "4"])

x1, x2, y1, y2 = plt.axis()
plt.axis((x1, x2, 0.90, 0.955))

plt.legend()
plt.tight_layout()
plt.savefig("mae_decoder_layers.png")
plt.show()
