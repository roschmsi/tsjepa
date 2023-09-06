from models.patch_tst.layers.pos_encoding import positional_encoding
import matplotlib.pyplot as plt
import torch
from torch.nn import AvgPool1d

pe = positional_encoding(pe="sincos", learn_pe=False, q_len=512, d_model=128)

plt.imshow(pe)
plt.savefig("pe_128.png")

pe = positional_encoding(pe="sincos", learn_pe=False, q_len=256, d_model=128)

plt.imshow(pe)
plt.savefig("pe_256.png")

avg_pool = AvgPool1d(kernel_size=2, stride=2, padding=0)
pe = avg_pool(pe.unsqueeze(0))
plt.imshow(pe.squeeze())
pe.squeeze()
plt.savefig("pe_256_interpolated.png")
