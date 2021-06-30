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


    model = torch.load('./model_save/UCI/no_awm/net0.965412004069176_199.pth', map_location='cpu')
    

    model = model.module.to(torch.device("cpu"))

    loss_func = torch.nn.CrossEntropyLoss()

    # model = model.cuda()
    # model = nn.DataParallel(model)
    # loss_func = loss_func.cuda()


    profiling(model, 'cpu', [128, 9],  # UCI
              1, [1],
              True)

   



