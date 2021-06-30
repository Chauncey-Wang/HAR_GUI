from dataset.data_proc import *


import torch.nn as nn
from utils.model_profiling import *
# from utils.model_profiling_base import *
from train_test.train_test_proc import *

if __name__ == "__main__":  # 这个工程里的model_profiling未调用gpu
    test_loss_list = []
    test_acc_list = []
    test_time_elapsed_list = []

    # test_x_list = "./dataset/UCI/x_test.npy"
    # test_y_list = "./dataset/UCI/y_test.npy"
    #
    # data_test = HAR(test_x_list, test_y_list)
    # har_test_tensor = data_test.HAR_data()
    #
    # test_loader = Data.DataLoader(dataset=har_test_tensor, batch_size=1, shuffle=True, num_workers=5, )

    # model = torch.load('./model_save/UCI/no_awm/net0.965412004069176_199.pth', map_location='cpu')
    # model = torch.load(r'D:\PycharmProjects\awn\awn_test\model_save\UCI\no_awm\net0.965412004069176_199.pth', map_location='cpu')
    model = torch.load('./model_save/UCI/16/net0.9684638860630722_199.pth', map_location='cpu')
    # model = torch.load('./model_save/UCI/32/net0.9684638860630722_199.pth', map_location='cpu')
    # model = torch.load('./model_save/UCI/48/net0.9684638860630722_199.pth', map_location='cpu')
    # model = torch.load('./model_save/UCI/64/net0.9684638860630722_199.pth', map_location='cpu')
    # model = torch.load('./model_save/UCI/80/net0.9684638860630722_199.pth', map_location='cpu')
    # model = torch.load('./model_save/UCI/96/net0.9684638860630722_199.pth', map_location='cpu')
    # model = torch.load('./model_save/UCI/112/net0.9684638860630722_199.pth', map_location='cpu')
    # model = torch.load('./model_save/UCI/128/net0.9613428280773143_199.pth', map_location='cpu')
    # model = torch.load('./model_save/wisdm/128/net0.9745222929936306_199.pth', map_location='cpu')
    # model = torch.load('./model_save/pamap2/128/net0.9108469539375929_199.pth', map_location='cpu')
    # model = torch.load('./model_save/unimib/128/net0.7570579494799405_199.pth', map_location='cpu')
    # model = torch.load('./model_save/oppo/1/net0.9170812603648425_199.pth', map_location='cpu')

    model = model.module.to(torch.device("cpu"))

    loss_func = torch.nn.CrossEntropyLoss()

    # model = model.cuda()
    # model = nn.DataParallel(model)
    # loss_func = loss_func.cuda()


    profiling(model, 'cpu', [128, 9],  # UCI
              1, [1],
              True)

    # profiling(model, 'cpu', [200, 3],  # wisdm
    #           1, [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1],
    #           True)

    # profiling(model, 'cpu', [171, 40],  # pamap2
    #           1, [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1],
    #           True)

    # profiling(model, 'cpu', [151, 3],  # unimib
    #           1, [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1],
    #           True)
    #
    # profiling(model, 'cpu', [64, 113],  # oppo
    #           1, [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1],
    #           True)



    # for e in range(1):
    #     model.set_width_mult(0.25)
    #
    #     test_loss, test_acc, test_time_elapsed, cm_test, pred_result = test(model, test_loader, loss_func)
    #     test_acc = float(test_acc) / len(test_loader.dataset)
    #
    #     test_loss_list.append(test_loss / len(test_loader.dataset))
    #     test_acc_list.append(test_acc)
    #     test_time_elapsed_sum = sum(test_time_elapsed)
    #
    #     print("Test set: Epoch {}, Loss {}, Accuracy {}, Time Elapsed {}".format(e, test_loss / len(
    #         test_loader.dataset), test_acc, test_time_elapsed_sum))
    # # # plot_line("uci_test_acc", test_acc_list, label='test_acc')
    # # print(sum(test_time_elapsed_list))




