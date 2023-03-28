import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils import load_train_data, get_cluster_num
from cvae import CVAE
from config import DefaultConfig

config = DefaultConfig()
device = config.device


def loss_func(pred_x, x, z_mu, z_log_var):
    # reconstruction loss:
    px_z = -nn.MSELoss()(pred_x, x) / 2
    # px_z = -nn.MSELoss()(pred_x, x)
    # use kl divergence: -kl
    kl_item = -0.5 * config.beta * torch.sum(1 + z_log_var - z_mu.pow(2) - z_log_var.exp())
    return -(px_z - kl_item)


def train(train_loader, model, prefix):
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    model.train()
    for epoch in range(config.epoch_num):
        running_loss = 0.0
        with tqdm(train_loader, desc=f"Epoch: {epoch + 1}", unit="batch") as tepoch:
            for batch, (x, c) in enumerate(tepoch):
                x = x.unsqueeze(1).float().to(device)
                c = c.float().to(device)
                # forward
                pred_x, z_mu, z_log_var = model(x, c)
                # backward
                loss = loss_func(pred_x, x, z_mu, z_log_var)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # tepoch.set_postfix(loss=loss.item())
                running_loss += loss.item()
                tepoch.set_postfix(loss=running_loss / (batch + 1))
        scheduler.step()

    # save model
    torch.save(model.state_dict(), config.model_path + f"{prefix}.pth")


def train_specific_model(prefix):
    print(f"Training model for prefix {prefix}...")
    train_loader = load_train_data(prefix)
    cluster_num = get_cluster_num(prefix)
    model = CVAE(cluster_num).to(device)
    train(train_loader, model, prefix)


def train_multiple_model(prefix_list):
    for prefix in prefix_list:
        train_specific_model(prefix)


def train_all_model():
    file_list = os.listdir(config.data_path)
    prefix_num = len(file_list)
    for prefix in range(prefix_num):
        train_specific_model(prefix)
