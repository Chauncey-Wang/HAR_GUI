from torch.autograd import Variable
import torch
import time
import numpy as np


def adjust_learning_rate(optimizer, epoch, learning_rate):
    lr = learning_rate * (0.1 ** (epoch // 40))  # 30  40
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# def adjust_learning_rate(optimizer, epoch, learning_rate):
#     lr = learning_rate * (0.2 ** (epoch // 35))  # 30  40
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr


def train(train_model, train_loader, criterion, optimizer, learning_rate, epoch):
    adjust_learning_rate(optimizer, epoch, learning_rate)

    train_model.train()
    start = time.time()
    # print(model)
    epoch_loss = 0
    epoch_acc = 0
    loss_list = []
    for step, (batch_x, batch_y) in enumerate(train_loader):
        inputs, labels = Variable(batch_x).float(), Variable(batch_y).long()

        optimizer.zero_grad()
        outputs = train_model(inputs)
        _, preds = torch.max(outputs.data, 1)
        # with open('BN_running_mean_0_1_1', 'ab') as f:
        #     pickle.dump(np.mean(list(train_model.module.layer1[1].running_mean.cpu())), f)
        epoch_acc += torch.sum(preds == labels.data).item()
        loss = criterion(outputs, labels)
        epoch_loss += loss.detach().item()

        loss.backward()
        optimizer.step()

        print("train iteration {}, loss {}, acc {}, lr {}".format(step, loss.item(),
                                                                  torch.sum(preds == labels.data).item() / len(batch_x),
                                                                  optimizer.param_groups[0]['lr']))

    end = time.time()
    time_elapsed = end - start
    return epoch_loss, epoch_acc, time_elapsed


def test(test_model, test_loader, criterion):
    # total_num = sum(p.numel() for p in test_model.parameters())
    # trainable_num = sum(p.numel() for p in test_model.parameters() if p.requires_grad)
    # print('test_model Total_Number of params: {} |Trainable_num of params: {}'.format(total_num, trainable_num))

    epoch_loss = 0
    epoch_acc = 0

    test_time = []

    test_model.eval()
    with torch.no_grad():
        for step, (batch_x, batch_y) in enumerate(test_loader):
            inputs, labels = Variable(batch_x).float(), Variable(batch_y).long()
            start = time.time()
            outputs = test_model(inputs)
            preds_prob = torch.nn.functional.softmax(outputs.data, dim=1)  # ####################
            end = time.time()
            test_time.append(end - start)
            _, preds = torch.max(outputs.data, 1)
            # print(preds)
            epoch_acc += torch.sum(preds == labels.data).item()
            loss = criterion(outputs, labels)
            epoch_loss += loss.detach().item()


    time_elapsed = test_time

    return epoch_loss, epoch_acc, time_elapsed, preds_prob, preds


def test_one_tensor(test_model, test_loader):
    test_model.eval()
    with torch.no_grad():
        for step, (batch_x, batch_y) in enumerate(test_loader):
            inputs, labels = Variable(batch_x).float(), Variable(batch_y).long()
            # print(inputs.shape)
            outputs = test_model(inputs)
            preds_prob = torch.nn.functional.softmax(outputs.data, dim=1)  # ####################
            _, preds = torch.max(outputs.data, 1)
            # print(preds)

    return preds_prob, preds



