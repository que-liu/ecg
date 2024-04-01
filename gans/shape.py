import numpy as np

# 加载.npy文件
# generated_signals = np.load(r'/home/zyw/gans/P2P/p2p/generated3000_test_signals.npy')  #(2203, 12, 1000)
generated_signals = np.load(r'/home/zyw/gans/WAVEGAN_COND/all_labels_uncond_star/checkpoints/generated3000_val_signals.npy')    #(2193, 12, 1000)
# 查看形状
print("生成的信号数组形状为:", type(generated_signals))
print("生成的信号数组形状为:", generated_signals.shape)
