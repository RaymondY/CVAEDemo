import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils import load_train_data, get_cluster_num
from cvae import CVAE
from config import DefaultConfig

config = DefaultConfig()
device = config.device


def log_normal_pdf(sample, mean, log_var, dim=1):
    log_2pi = torch.log(torch.tensor(2. * np.pi))
    return torch.sum(-.5 * ((sample - mean) ** 2. * torch.exp(-log_var) + log_var + log_2pi), dim=dim)


def loss_func(pred_x, x, z_mu, z_log_var, z):
    log_p_x_z = log_normal_pdf(x, pred_x, torch.zeros_like(pred_x))
    # log_p_x_z = torch.sum(-.5 * (x - pred_x) ** 2, dim=1)
    log_p_z = log_normal_pdf(z, torch.zeros_like(z), torch.zeros_like(z))
    log_q_z_x = log_normal_pdf(z, z_mu, z_log_var)
    return -torch.mean(log_p_x_z + config.beta * (log_p_z - log_q_z_x))


def train(train_loader, model, prefix):
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.25)
    model.train()
    for epoch in range(config.epoch_num):
        running_loss = 0.0
        with tqdm(train_loader, desc=f"Epoch: {epoch + 1}", unit="batch") as tepoch:
            for batch, (x, c) in enumerate(tepoch):
                x = x.unsqueeze(1).float().to(device)
                c = c.float().to(device)
                # forward
                pred_x, z_mu, z_log_var, z = model(x, c)
                # backward
                loss = loss_func(pred_x.squeeze(1), x.squeeze(1), z_mu, z_log_var, z)
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
