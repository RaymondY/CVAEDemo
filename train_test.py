import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils import load_data
from config import DefaultConfig

config = DefaultConfig()
device = config.device


def loss_func(pred_x, x, z_mean, z_log_var):
    # reconstruction loss: hamming distance in torchmetrics
    l1_loss = nn.L1Loss()(pred_x, x)
    # use kl divergence
    kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
    return l1_loss + kl_loss


def train(train_loader, model):
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)
    model.train()
    for epoch in range(config.epoch_num):
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
                tepoch.set_postfix(loss=loss.item())

        # scheduler.step()
    # save model
    torch.save(model.state_dict(), config.model_path)


def test(model, test_loader):
    model.eval()
    with torch.no_grad():
        for batch, (x, c) in enumerate(test_loader):
            x = x.unsqueeze(1).float().to(device)
            c = c.float().to(device)
            # forward
            pred_x, z_mu, z_log_var = model(x, c)
            # backward
            loss = loss_func(pred_x, x, z_mu, z_log_var)
            print(f"Test loss: {loss.item():.4f}")
            for i in range(10):
                print(pred_x[i].cpu().numpy())
                print(pred_x[i].cpu().numpy().round())
                print(x[i].cpu().numpy())
                print("---------------------------------")
            print("=================================")
            break
