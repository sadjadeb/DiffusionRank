from scipy.stats import wasserstein_distance
from scipy.spatial import distance
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import pandas as pd
from dython.nominal import compute_associations


real_data_path = '/home/sajadeb/Gen_LTR/tab-ddpm/data/california/X_num_train.npy'
fake_data_path = '/home/sajadeb/Gen_LTR/tab-ddpm/exp/california/exp_2024-10-14_21-18-19/X_num_train.npy'

real_data_path = '/home/sajadeb/Gen_LTR/tab-ddpm/data/MQ2007/X_num_train.npy'
fake_data_path = '/home/sajadeb/Gen_LTR/tab-ddpm/exp/MQ2007/exp_2024-11-04_18-38-43/X_num_train.npy'

real_data_path = '/home/sajadeb/Gen_LTR/tab-ddpm/data/MSLR-Web10K/X_num_train.npy'
fake_data_path = '/home/sajadeb/Gen_LTR/tab-ddpm/exp/MSLR-Web10K/exp_2024-11-05_15-37-24/X_num_train.npy'
fake_data_path = '/home/sajadeb/Gen_LTR/tab-ddpm/exp/MSLR-Web10K/exp_2024-11-07_12-58-30/X_num_train.npy'

real_arr = np.load(real_data_path)
fake_arr = np.load(fake_data_path)

# make both arrays have the same shape
if real_arr.shape[0] > fake_arr.shape[0]:
    real_arr = real_arr[:fake_arr.shape[0], :]
elif real_arr.shape[0] < fake_arr.shape[0]:
    fake_arr = fake_arr[:real_arr.shape[0], :]

real = pd.DataFrame(real_arr)
fake = pd.DataFrame(fake_arr)

real_corr = compute_associations(real)

fake_corr = compute_associations(fake)

corr_dist = np.linalg.norm(real_corr - fake_corr)
print(f"Correlation distance: {corr_dist}")

Stat_dict={}

cat_stat = np.zeros(real_arr.shape[1])
num_stat = []

for column in real.columns:
    scaler = StandardScaler()
    scaler.fit(real[column].values.reshape(-1,1))
    l1 = scaler.transform(real[column].values.reshape(-1,1)).flatten()
    l2 = scaler.transform(fake[column].values.reshape(-1,1)).flatten()
    Stat_dict[column]= (wasserstein_distance(l1,l2))
    num_stat.append(Stat_dict[column])

print("Wasserstein distance:")
print(f"{np.mean(num_stat)}, {np.mean(cat_stat)}")
print("Normalized Wasserstein distance:")
print(f"{np.mean(num_stat)/real_arr.shape[1]}, {np.mean(cat_stat)}")