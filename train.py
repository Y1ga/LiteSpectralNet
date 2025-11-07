import torch
import torch.nn as nn
import scipy.io as scio
import time
import math
import torch.nn.functional as F
import scipy.io as sio
import numpy as np
import os
from model import *


dtype = torch.float
device_data = torch.device("cpu")
device_train = torch.device("cuda:0")
device_test = torch.device("cpu")

BATCH_SIZE = 2000
INFERENCE_BATCH_SIZE = 400000

TRAINING_DATA_RATIO = 0.8
DATA_SIZE = 1 * 1000 * 1000
EPOCH_NUM = 501
TEST_INTERVAL = 20
LR = 1e-3
LR_DECAY_STEP = 2000
LR_DECAY_GAMMA = 0.8
L1_LAMBDA = 0e-1
TV_WEIGHT = 1e-2
MRAE_WEIGHT = 1e-2
DROPOUT_RATE = 0.1
WEIGHT_DECAY = 3e-3
FILE_PATH = "./data/train_dataset.mat"


def main():
    data = scio.loadmat(FILE_PATH)
    Input_data = torch.tensor(data["input_data"][:, :], device=device_data, dtype=dtype)
    Output_data = torch.tensor(
        data["output_data"][:, :], device=device_data, dtype=dtype
    )
    idx = torch.randperm(Input_data.shape[0])
    Input_data = Input_data[idx, :]
    Output_data = Output_data[idx, :]

    if DATA_SIZE > Input_data.shape[0]:
        data_size = Input_data.shape[0]
        batch_size = int(data_size * TRAINING_DATA_RATIO)
    else:
        data_size = DATA_SIZE
        batch_size = BATCH_SIZE

    TrainingDataSize = int(data_size * TRAINING_DATA_RATIO)
    TestingDataSize = data_size - TrainingDataSize

    Input_train = Input_data[0:TrainingDataSize, :]
    Output_train = Output_data[0:TrainingDataSize, :]
    Input_test = Input_data[TrainingDataSize : TrainingDataSize + TestingDataSize, :]
    Output_test = Output_data[TrainingDataSize : TrainingDataSize + TestingDataSize, :]

    OutputNum = Output_train.shape[1]

    del data, Input_data, Output_data

    output_channel = OutputNum
    fnet = LiteSpectralNet(output_channel=output_channel)
    fnet.to(device_train)
    fnet.train()
    total_params = sum(p.numel() for p in fnet.parameters())

    LossFcn = Loss(tv_weight=TV_WEIGHT, mrae_weight=MRAE_WEIGHT)
    optimizer = torch.optim.AdamW(fnet.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=LR_DECAY_STEP, gamma=LR_DECAY_GAMMA
    )

    loss_train = torch.zeros(math.ceil(EPOCH_NUM / TEST_INTERVAL))
    loss_test = torch.zeros(math.ceil(EPOCH_NUM / TEST_INTERVAL))

    os.makedirs(".\\nets", exist_ok=True)
    final_model_path = os.path.join(".\\nets", "lsnet.pth")
    time_start = time.time()
    time_epoch0 = time_start

    for epoch in range(EPOCH_NUM):
        idx = torch.randperm(TrainingDataSize, device=device_data)
        Input_train = Input_train[idx, :]
        Output_train = Output_train[idx, :]
        for i in range(0, TrainingDataSize // batch_size):
            InputBatch = Input_train[i * batch_size : i * batch_size + batch_size, :]
            OutputBatch = Output_train[i * batch_size : i * batch_size + batch_size, :]
            Output_pred = fnet(InputBatch.to(device_train))

            loss = LossFcn(OutputBatch.to(device_train), Output_pred)

            l1_reg = torch.tensor(0.0, device=device_train)
            for param in fnet.parameters():
                l1_reg += torch.norm(param, p=1)
            total_loss = loss + L1_LAMBDA * l1_reg

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        scheduler.step()

        if epoch % TEST_INTERVAL == 0:
            fnet.to(device_test)
            fnet.eval()
            with torch.no_grad():
                Out_test_pred = fnet(Input_test)
                current_test_loss = LossFcn(Output_test, Out_test_pred).item()
            fnet.to(device_train)
            fnet.train()

            loss_train[epoch // TEST_INTERVAL] = loss.data
            loss_test[epoch // TEST_INTERVAL] = current_test_loss

            if epoch == 0:
                time_epoch0 = time.time()
                time_remain = (time_epoch0 - time_start) * EPOCH_NUM
            else:
                time_remain = (time.time() - time_epoch0) / epoch * (EPOCH_NUM - epoch)

            log_str = (
                f"Epoch: {epoch} | train loss: {loss.item():.5f} "
                f"| test loss: {torch.tensor(current_test_loss).item():.5f} "
                f"| lr: {scheduler.get_last_lr()[0]:.8f} "
                f'| remaining: {time_remain:.0f}s (to {time.strftime("%H:%M:%S", time.localtime(time.time() + time_remain))})'
            )
            print(log_str)

    time_end = time.time()
    time_total = time_end - time_start
    m, s = divmod(time_total, 60)
    h, m = divmod(m, 60)
    print("Training time: %.0fs (%dh%02dm%02ds)" % (time_total, h, m, s))

    torch.save(
        {
            "epoch": EPOCH_NUM - 1,
            "model_state_dict": fnet.state_dict(),
            "output_channel": output_channel,
        },
        final_model_path,
    )
    print(f"Final model saved to: {final_model_path}")


if __name__ == "__main__":
    main()
