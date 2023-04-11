import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils import *
from cvae import CVAE
from config import DefaultConfig

config = DefaultConfig()
device = config.device


def log_laplace_pdf(sample, mean, log_b, dim=1):
    log_2 = torch.log(torch.tensor(2.))
    return torch.sum(-log_2 - log_b - torch.abs(sample - mean) / torch.exp(log_b), dim=dim)


def log_normal_pdf(sample, mean, log_var, dim=1):
    log_2pi = torch.log(torch.tensor(2. * np.pi))
    return torch.sum(-.5 * ((sample - mean) ** 2. * torch.exp(-log_var) + log_var + log_2pi), dim=dim)


def loss_func(pred_x, x, z_mu, z_log_var, z):
    log_p_x_z = log_normal_pdf(x, pred_x, torch.zeros_like(pred_x))
    # log_p_x_z = log_laplace_pdf(x, pred_x, torch.zeros_like(pred_x))
    # log_p_x_z = torch.sum(-.5 * (x - pred_x) ** 2, dim=1)
    log_p_z = log_normal_pdf(z, torch.zeros_like(z), torch.zeros_like(z))
    log_q_z_x = log_normal_pdf(z, z_mu, z_log_var)
    return -torch.mean(log_p_x_z + config.beta * (log_p_z - log_q_z_x))


def train(train_loader, model, prefix,
          lr=config.lr, epoch_num=config.epoch_num, is_init=False):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    model.train()

    if is_init:
        warm_up_optimizer = optim.Adam(model.parameters(), lr=config.warmup_lr)
        for warm_up_epoch in range(config.warmup_epoch_num):
            for batch, (x, c) in enumerate(train_loader):
                x = x.unsqueeze(1).float().to(device)
                c = c.float().to(device)
                pred_x, z_mu, z_log_var, z = model(x, c)
                loss = loss_func(pred_x.squeeze(1), x.squeeze(1), z_mu, z_log_var, z)
                warm_up_optimizer.zero_grad()
                loss.backward()
                warm_up_optimizer.step()

    for epoch in range(epoch_num):
        running_loss = 0.0
        with tqdm(train_loader, desc=f"Epoch: {epoch + 1}", unit="batch") as tepoch:
            for batch, (x, c) in enumerate(tepoch):
                x = x.unsqueeze(1).float().to(device)
                c = c.float().to(device)
                pred_x, z_mu, z_log_var, z = model(x, c)
                loss = loss_func(pred_x.squeeze(1), x.squeeze(1), z_mu, z_log_var, z)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                tepoch.set_postfix(loss=running_loss / (batch + 1))
        scheduler.step()

    # save model
    if not is_init:
        torch.save(model.state_dict(), config.model_path + f"{prefix}.pth")
    torch.save(model.state_dict(), config.iter_model_path + f"{prefix}.pth")


def init_train_specific_prefix(prefix):
    print(f"Training model for prefix {prefix}...")
    train_loader = load_init_train_data(prefix)
    cluster_num = train_loader.dataset.get_cluster_num()
    model = CVAE(cluster_num).to(device)
    train(train_loader, model, prefix, is_init=True)


def iter_train_specific_prefix(lr=config.lr * 0.1, epoch_num=config.epoch_num):
    # iterate over prefixes under config.iter_model_path
    # the prefixes under config.iter_model_path should be the same as those under config.new_address_path
    prefix_list = os.listdir(config.iter_model_path)
    # remove .pth suffix
    # prefix_list = [re.sub(r"\.pth$", "", prefix) for prefix in prefix_list]
    prefix_list = [int(re.sub(r"\.pth$", "", prefix)) for prefix in prefix_list]
    for prefix in prefix_list:
        print(f"Fine-tuning model for prefix {prefix}...")
        train_loader = load_iter_train_data(prefix)
        cluster_num = train_loader.dataset.get_cluster_num()
        model = CVAE(cluster_num).to(device)
        model.load_state_dict(torch.load(config.iter_model_path + f"{prefix}.pth"))
        train(train_loader, model, prefix, lr=lr, epoch_num=epoch_num)
