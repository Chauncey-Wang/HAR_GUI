"""
Source: https://github.com/JiahuiYu/slimmable_networks
"""
import numpy as np
import time
import torch
import torch.nn as nn

model_profiling_hooks = []
model_profiling_speed_hooks = []

name_space = 95
params_space = 15
macs_space = 15
seconds_space = 15

num_forwards = 10


class Timer(object):
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.start = None
        self.end = None

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.time = self.end - self.start
        if self.verbose:
            print('Elapsed time: %f ms.' % self.time)


def get_params(self):
    """get number of params in module"""
    return np.sum(
        [np.prod(list(w.size())) for w in self.parameters()])


def run_forward(self, input):
    with Timer() as t:
        for _ in range(num_forwards):
            self.forward(*input)
            # torch.cuda.synchronize()
    return int(t.time * 1e9 / num_forwards)


def conv_module_name_filter(name):
    """filter module name to have a short view"""
    filters = {
        'kernel_size': 'k',
        'stride': 's',
        'padding': 'pad',
        'bias': 'b',
        'groups': 'g',
    }
    for k in filters:
        name = name.replace(k, filters[k])
    return name


def module_profiling(self, input, output, verbose):
    ins = input[0].size()
    outs = output.size()
    # NOTE: There are some difference between type and isinstance, thus please
    # be careful.
    t = type(self)
    if isinstance(self, nn.Conv2d):
        self.n_macs = (ins[1] * outs[1] *
                       self.kernel_size[0] * self.kernel_size[1] *
                       outs[2] * outs[3] // self.groups) * outs[0]
        self.n_params = get_params(self)
        self.n_seconds = run_forward(self, input)
        self.name = conv_module_name_filter(self.__repr__())
    elif isinstance(self, nn.ConvTranspose2d):
        self.n_macs = (ins[1] * outs[1] *
                       self.kernel_size[0] * self.kernel_size[1] *
                       outs[2] * outs[3] // self.groups) * outs[0]
        self.n_params = get_params(self)
        self.n_seconds = run_forward(self, input)
        self.name = conv_module_name_filter(self.__repr__())
    elif isinstance(self, nn.Linear):
        self.n_macs = ins[1] * outs[1] * outs[0]
        self.n_params = get_params(self)
        self.n_seconds = run_forward(self, input)
        self.name = self.__repr__()
    elif isinstance(self, nn.AvgPool2d):
        # NOTE: this function is correct only when stride == kernel size
        self.n_macs = ins[1] * ins[2] * ins[3] * ins[0]
        self.n_params = 0
        self.n_seconds = run_forward(self, input)
        self.name = self.__repr__()
    elif isinstance(self, nn.AdaptiveAvgPool2d):
        # NOTE: this function is correct only when stride == kernel size
        self.n_macs = ins[1] * ins[2] * ins[3] * ins[0]
        self.n_params = 0
        self.n_seconds = run_forward(self, input)
        self.name = self.__repr__()
    else:
        # This works only in depth-first travel of modules.
        self.n_macs = 0
        self.n_params = 0
        self.n_seconds = 0
        num_children = 0
        for m in self.children():
            self.n_macs += getattr(m, 'n_macs', 0)
            self.n_params += getattr(m, 'n_params', 0)
            self.n_seconds += getattr(m, 'n_seconds', 0)
            num_children += 1
        ignore_zeros_t = [
            nn.BatchNorm2d, nn.Dropout2d, nn.Dropout, nn.Sequential,
            nn.ReLU6, nn.ReLU, nn.MaxPool2d,
            nn.modules.padding.ZeroPad2d, nn.modules.activation.Sigmoid,
        ]
        # if (not getattr(self, 'ignore_model_profiling', False) and
        #         self.n_macs == 0 and
        #         t not in ignore_zeros_t):
        #     print(
        #         'WARNING: leaf module {} has zero n_macs.'.format(type(self)))
        return
    if verbose:
        print(
            self.name.ljust(name_space, ' ') +
            '{:,}'.format(self.n_params).rjust(params_space, ' ') +
            '{:,}'.format(self.n_macs).rjust(macs_space, ' ') +
            '{:,}'.format(self.n_seconds).rjust(seconds_space, ' '))
    return


def add_profiling_hooks(m, verbose):
    global model_profiling_hooks
    model_profiling_hooks.append(
        m.register_forward_hook(
            lambda m, input, output: module_profiling(
                m, input, output, verbose=verbose)))


def remove_profiling_hooks():
    global model_profiling_hooks
    for h in model_profiling_hooks:
        h.remove()
    model_profiling_hooks = []


def model_profiling(model, height, width, channel=3, batch=1, use_cuda=False,
                    verbose=True):
    """ Pytorch model profiling with input image size
    (batch, channel, height, width).
    The function exams the number of multiply-accumulates (n_macs).

    Args:
        model: pytorch model
        height: int
        width: int
        batch: int
        channel: int
        use_cuda: bool

    Returns:
        macs: int
        params: int

    """
    model.eval()
    # data = torch.rand(batch, channel, height, width)
    data = torch.rand(batch, channel, height, width)
    # data = torch.tensor(np.load("./dataset/UCI/x_test.npy")[0], dtype=torch.float).reshape(batch, channel,128, 9) ###

    device = torch.device("cuda:0" if use_cuda else "cpu")
    model = model.to(device)
    data = data.to(device)
    model.apply(lambda m: add_profiling_hooks(m, verbose=verbose))
    print(
        'Item'.ljust(name_space, ' ') +
        'params'.rjust(macs_space, ' ') +
        'macs'.rjust(macs_space, ' ') +
        'nanosecs'.rjust(seconds_space, ' '))
    if verbose:
        print(''.center(name_space + params_space + macs_space + seconds_space, '-'))
    model(data)
    if verbose:
        print(''.center(name_space + params_space + macs_space + seconds_space, '-'))
    print(
        'Total'.ljust(name_space, ' ') +
        '{:,}'.format(model.n_params).rjust(params_space, ' ') +
        '{:,}'.format(model.n_macs).rjust(macs_space, ' ') +
        '{:,}'.format(model.n_seconds).rjust(seconds_space, ' '))
    remove_profiling_hooks()
    return model.n_macs, model.n_params


def _profiling(model, use_cuda, image_size, channel=3, verbose=False,
               slimmable=False, width_mult_list=None):
    """profiling on either gpu or cpu"""
    print('Start model profiling, use_cuda:{}.'.format(use_cuda))
    if slimmable:
        for width_mult in sorted(width_mult_list, reverse=True):
            # model.apply(lambda m: setattr(m, 'width_mult', width_mult))
            if isinstance(model, nn.DataParallel):
                model = model.module
            # model.set_width_mult(width_mult)
            print('Model profiling with width mult {}x:'.format(width_mult))
            model_profiling(
                model, image_size[0], image_size[1], channel=channel, use_cuda=False,
                verbose=verbose or width_mult == max(width_mult_list))
    else:
        model_profiling(
            model, image_size, image_size, channel=channel, verbose=verbose)
    return


def profiling(model, machines, image_size, channel, width_mult_list, verbose=False):
    slimmable = True
    # if 'gpu' in machines:
    #     use_cuda = True
    #     _profiling(model, use_cuda, image_size, channel, verbose,
    #                slimmable, width_mult_list)
    if 'cpu' in machines:
        use_cuda = False
        _profiling(model, use_cuda, image_size, channel, verbose,
                   slimmable, width_mult_list)
    return


# if __name__ == "__main__":
    # profiling(model, 'cpu', [128, 9],  # #######################################
    #           1, [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1],
    #           True)


