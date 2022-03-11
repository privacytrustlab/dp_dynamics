# adapted from https://github.com/ftramer/Handcrafted-DP/blob/main/log.py
# MIT License

# Copyright (c) 2020 ftramer

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.



import numpy as np
import os
import shutil
import sys
from torch.utils.tensorboard import SummaryWriter
import torch


def model_input(data, device):
    datum = data.data[0:1]
    if isinstance(datum, np.ndarray):
        return torch.from_numpy(datum).float().to(device)
    else:
        return datum.float().to(device)


def get_script():
    py_script = os.path.basename(sys.argv[0])
    return os.path.splitext(py_script)[0]


def get_specified_params(hparams):
    keys = [k.split("=")[0][2:] for k in sys.argv[1:]]
    specified = {k: hparams[k] for k in keys}
    return specified


def make_hparam_str(hparams, exclude):
    return ",".join([f"{key}_{value}"
                     for key, value in sorted(hparams.items())
                     if key not in exclude])


class Logger(object):
    def __init__(self, logdir):

        if logdir is None:
            self.writer = None
        else:
            if os.path.exists(logdir) and os.path.isdir(logdir):
                shutil.rmtree(logdir)

            self.writer = SummaryWriter(log_dir=logdir)

    def log_model(self, model, input_to_model):
        if self.writer is None:
            return
        self.writer.add_graph(model, input_to_model)

    def log_epoch(self, epoch, train_loss, train_acc, test_loss, test_acc, epsilon=None, epsilon3=None):
        if self.writer is None:
            return
        self.writer.add_scalar("Loss/train", train_loss, epoch)
        self.writer.add_scalar("Loss/test", test_loss, epoch)
        self.writer.add_scalar("Accuracy/train", train_acc, epoch)
        self.writer.add_scalar("Accuracy/test", test_acc, epoch)

        if epsilon is not None:
            self.writer.add_scalar("Acc@Eps/train", train_acc, 100*epsilon)
            self.writer.add_scalar("Acc@Eps/test", test_acc, 100*epsilon)

        if epsilon3 is not None:
            self.writer.add_scalar("Acc@Eps3/train", train_acc, 100*epsilon3)
            self.writer.add_scalar("Acc@Eps3/test", test_acc, 100*epsilon3)

    def log_scalar(self, tag, scalar_value, global_step):
        if self.writer is None or scalar_value is None:
            return
        self.writer.add_scalar(tag, scalar_value, global_step)
