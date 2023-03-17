import torch
from utils import load_data
from cvae import CVAE
from train_test import train, test
from config import DefaultConfig

config = DefaultConfig()
device = config.device


def read_model_and_test():
    prefix = 0
    train_loader = load_data(prefix)
    model = CVAE().to(device)
    # load block_num.pth
    print("Loading model...")
    model.load_state_dict(torch.load(f"model/test.pth"))
    test(model, train_loader)


def main():
    prefix = 0
    train_loader = load_data(prefix)
    model = CVAE().to(device)
    train(train_loader, model)


if __name__ == '__main__':
    main()
    # read_model_and_test()
