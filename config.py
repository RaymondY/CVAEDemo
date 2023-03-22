import os
import torch
from datetime import datetime

cluster_num = -1

# general
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("mps" if torch.backends.mps.is_available() else device)

# dir path
current_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_path, 'data/')
model_dir = os.path.join(current_path, 'model')
model_path = os.path.join(model_dir, f'test.pth')
# model_path = os.path.join(model_dir, f'{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.pth')
# sample_num = 1000
# input_size = 32 * 17
# input_size = 32
input_size = 32
batch_size = 128
# other_result_path_test = '../result/1109/test1/'
# entropy_threshold = 3.5

eps_threshold = 4
eps_ratio = 0.01

block_num = 4
feature_num = 32
intermediate_size = 16
latent_size = 5

epoch_num = 40
lr = 1e-3

# lr_decay = 0.95  # when val_loss increase, lr = lr * lr_decay
# weight_decay = 1e-10

current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = 'logs/gradient_tape/' + current_time
