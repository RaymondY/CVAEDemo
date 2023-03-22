import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils import load_data
import config

device = config.device


def loss_func(pred_x, x, z_mu, z_log_var):
    # reconstruction loss:
    l1_loss = nn.MSELoss()(pred_x, x)
    # use kl divergence
    kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mu.pow(2) - z_log_var.exp())
    return l1_loss + kl_loss


def train(train_loader, model):
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
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

        scheduler.step()
    # save model
    torch.save(model.state_dict(), config.model_path)


# def test(model, test_loader):
#     model.eval()
#     loss_sum = 0
#     with torch.no_grad():
#         for batch, (x, c) in enumerate(test_loader):
#             x = x.unsqueeze(1).float().to(device)
#             c = c.float().to(device)
#             # forward
#             pred_x, z_mu, z_log_var = model(x, c)
#             # define hamming loss (pre_x, x)
#             x = x.squeeze(1).int()
#             pred_x = pred_x.squeeze(1)
#             pred_x = pred_x.round().int()
#             # if pred_x > 15: pred_x = 15
#             # if pred_x < 0: pred_x = 0
#             pred_x[pred_x > 15] = 15
#             pred_x[pred_x < 0] = 0
#             for i in range(x.shape[0]):
#                 hamming_loss = 0
#                 for j in range(x.shape[1]):
#                     if x[i][j] != pred_x[i][j]:
#                         hamming_loss += 1
#                 # if hamming_loss > 4:
#                 #     print(f"Hamming loss: {hamming_loss}")
#                 #     print(f"pred_x: {pred_x[i]}")
#                 #     print(f"x: {x[i]}")
#                 #     print("---------------------------------")
#                 loss_sum += hamming_loss
#
#             # loss = loss_func(pred_x, x, z_mu, z_log_var)
#             # print(f"Test loss: {loss.item():.4f}")
#             # for i in range(10):
#             #     print(pred_x[i].cpu().numpy())
#             #     print(pred_x[i].cpu().numpy().round())
#             #     print(x[i].cpu().numpy())
#             #     print("---------------------------------")
#             # print("=================================")
#             # break
#
#     # loss_sum = loss_sum / (x.shape[0] * x.shape[1] * x.shape[2])
#
#     print(f"Test loss: {loss_sum}")
