import scipy.io
import numpy as np

# 读取 MATLAB 模型
mat = scipy.io.loadmat("./SEDS/seds_model_for_python.mat")

# 保存为 .npz 格式
np.savez("seds_model.npz",
         Priors=mat["Priors"].flatten(),
         Mu=mat["Mu"],
         Sigma=mat["Sigma"],
         att=mat["att"].flatten(),
         M=int(mat["M"][0][0]))
