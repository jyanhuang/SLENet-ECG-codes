import os
import datetime
import time

import numpy as np
import torch
import torch.nn as nn
from thop import profile
from torch.nn.utils import prune
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from utils import load_data, plot_heat_map, model_select
from FPGM import prune

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
pre_trained_path = ""
save_path = '0320model.pt'

# the device to use
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))


# define the dataset class
class ECGDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        x = torch.tensor(self.x[index], dtype=torch.float32)
        y = torch.tensor(self.y[index], dtype=torch.long)
        return x, y

    def __len__(self):
        return len(self.x)

# define the training function and validation function
def train_steps(loop, model, criterion, optimizer):
    train_loss = []
    train_acc = []
    model.train()

    for step_index, (X, y) in loop:

        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = criterion(pred, y)

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        loss = loss.item()
        train_loss.append(loss)
        pred_result = torch.argmax(pred, dim=1).detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        acc = accuracy_score(y, pred_result)
        train_acc.append(acc)
        loop.set_postfix(loss=loss, acc=acc)
    return {"loss": np.mean(train_loss),
            "acc": np.mean(train_acc)}


def test_steps(loop, model, criterion):
    test_loss = []
    test_acc = []
    model.eval()
    with torch.no_grad():
        for step_index, (X, y) in loop:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = criterion(pred, y).item()

            test_loss.append(loss)
            pred_result = torch.argmax(pred, dim=1).detach().cpu().numpy()
            y = y.detach().cpu().numpy()
            acc = accuracy_score(y, pred_result)
            test_acc.append(acc)
            loop.set_postfix(loss=loss, acc=acc)
    return {"loss": np.mean(test_loss),
            "acc": np.mean(test_acc)}


def train_epochs(train_dataloader, test_dataloader, model, criterion, optimizer, config, writer):
    num_epochs = config['num_epochs']
    train_loss_ls = []
    train_loss_acc = []
    test_loss_ls = []
    test_loss_acc = []
    for epoch in range(num_epochs):
        train_loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        test_loop = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
        train_loop.set_description('Epoch [{epoch + 1}/{num_epochs}]')
        test_loop.set_description('Epoch [{epoch + 1}/{num_epochs}]')

        train_metrix = train_steps(train_loop, model, criterion, optimizer)
        test_metrix = test_steps(test_loop, model, criterion)

        train_loss_ls.append(train_metrix['loss'])
        train_loss_acc.append(train_metrix['acc'])
        test_loss_ls.append(test_metrix['loss'])
        test_loss_acc.append(test_metrix['acc'])

        print('Epoch {epoch + 1}: '
              'train loss: {train_metrix["loss"]}; '
              'train acc: {train_metrix["acc"]}; ')
        print('Epoch {epoch + 1}: '
              'test loss: {test_metrix["loss"]}; '
              'test acc: {test_metrix["acc"]}')

        writer.add_scalar('train/loss', train_metrix['loss'], epoch)
        writer.add_scalar('train/accuracy', train_metrix['acc'], epoch)
        writer.add_scalar('validation/loss', test_metrix['loss'], epoch)
        writer.add_scalar('validation/accuracy', test_metrix['acc'], epoch)

    return {'train_loss': train_loss_ls,
            'train_acc': train_loss_acc,
            'test_loss': test_loss_ls,
            'test_acc': test_loss_acc}


def main():
    config = {
        'seed': 42,  # the random seed
        'test_ratio': 0.2,  # the ratio of the test set
        'num_epochs': 30,
        'batch_size': 128,
        'lr': 0.001,
        'prune': False,
        'model_name': 'ecgTransForm'  # LightX3ECG, ecgTransForm, MSDNN, Ours
    }

    # X_train,y_train is the training set
    # X_test,y_test is the test set
    X_train, X_test, y_train, y_test = load_data(config['test_ratio'], config['seed'])
    train_dataset, test_dataset = ECGDataset(X_train, y_train), ECGDataset(X_test, y_test)
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)


    model = model_select(config['model_name'])

    if os.path.exists(pre_trained_path):
        # 导入预训练模型，跳过训练过程
        print('Import the pre-trained model, skip the training process')
        model.load_state_dict(torch.load(pre_trained_path))
        model.to(device)
        model.eval()
    else:
        # 构建CNN模型
        model = model.to(device)

    if config['prune']:
        prune(model, 'conv1', 0.9, 0.8)
        prune(model, 'fc1.0', 0.9, 0.5)
        prune(model, 'fc2', 0.9, 0.3)


    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    # # print the model structure
    # summary(model, (config['batch_size'], X_train.shape[1]), col_names=["input_size", "kernel_size", "output_size"],
    #         verbose=2)

    # define the Tensorboard SummaryWriter
    writer = SummaryWriter(log_dir=log_dir)
    # train and evaluate model
    train_epochs(train_dataloader, test_dataloader, model, criterion, optimizer, config, writer)
    writer.close()

    # save the model
    torch.save(model.state_dict(), save_path)
    # plot the training history
    # plot_history_torch(history)

    # predict the class of test data
    y_pred = []
    model.eval()
    model.to(device)  # 将模型移动到GPU上
    torch.cuda.reset_max_memory_allocated()
    max_memory_allocated = torch.cuda.max_memory_allocated()
    print("最大显存消耗（bytes）：", max_memory_allocated)
    print("最大显存消耗（MB）：", max_memory_allocated / 1024 / 1024)
    start_time = time.time()
    total_test = 1
    for i in range(total_test):
        with torch.no_grad():
            # 记录开始时间

            for step_index, (X, y) in enumerate(test_dataloader):
                X, y = X.to(device), y.to(device)
                pred = model(X)
                pred_result = torch.argmax(pred, dim=1).cpu().numpy()
                y_pred.extend(pred_result)
    max_memory_allocated = torch.cuda.max_memory_allocated()
    print("最大显存消耗（bytes）：", max_memory_allocated)
    print("最大显存消耗（MB）：", max_memory_allocated / 1024 / 1024)

    end_time = time.time()  # 记录结束时间
    inference_time = (end_time - start_time)/total_test  # 计算推理时间
    print('Inference Time: {inference_time} seconds')

    # 将预测结果转移到CPU上，并将其转换为NumPy数组


    y_true = y_test  # 不需要使用.cpu()方法，直接使用NumPy数组
    y_pred = np.array(y_pred)  # 将预测标签转换为NumPy数组
    y_pred = y_pred[:y_true.shape[0]]
    plot_heat_map(y_true, y_pred)

    correct_predictions = np.sum(y_pred == y_true)
    total_predictions = len(y_pred)
    accuracy = correct_predictions / total_predictions
    print("Total accuracy: ", accuracy)

    class_accuracy = {}

    for i in range(5):  # Assuming there are 5 classes
        mask_true = (y_true == i)
        mask_pred = (y_pred == i)

        correct_predictions = np.sum(mask_true & mask_pred)
        total_predictions = np.sum(mask_true)

        if total_predictions > 0:
            acc = correct_predictions / total_predictions
            class_accuracy["Class {i}"] = acc
        else:
            class_accuracy["Class {i}"] = 0.0  # Handle the case where there are no samples for a class

    # Print class-wise accuracy
    for class_label, acc in class_accuracy.items():
        print("{class_label} Accuracy: {acc:.4f}")

    # Calculate overall accuracy
    overall_accuracy = accuracy_score(y_true, y_pred)

    # Print overall accuracy
    print("Overall Accuracy: {overall_accuracy:.4f}")

    # Calculate average accuracy
    average_accuracy = np.mean(list(class_accuracy.values()))

    # Print average accuracy
    print("Average Accuracy: {average_accuracy:.4f}")
    with torch.no_grad():
        flops, params = profile(model, inputs=(torch.ones((128,300)).to(device),))
        print("Total parameters:", params)
        print("Total FLOPs:", flops)

if __name__ == '__main__':
    main()
