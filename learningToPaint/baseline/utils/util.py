"""
util file
"""


import os
import torch
from torch.autograd import Variable

USE_CUDA = torch.cuda.is_available()


def prRed(prt):
    """
    打印红色
    :param prt:
    :return:
    """
    print(f"\033[91m {prt}\033[00m")


def prGreen(prt):
    """
    打印绿色
    :param prt:
    :return:
    """
    print(f"\033[92m {prt}\033[00m")


def prYellow(prt):
    """
    打印黄色
    :param prt:
    :return:
    """
    print(f"\033[93m {prt}\033[00m")


def prLightPurple(prt):
    """
    打印淡紫色
    :param prt:
    :return:
    """
    print("\033[94m {}\033[00m" .format(prt))


def prPurple(prt):
    """
    打印紫色
    :param prt:
    :return:
    """
    print("\033[95m {}\033[00m" .format(prt))


def prCyan(prt):
    """
    打印青色
    :param prt:
    :return:
    """
    print("\033[96m {}\033[00m" .format(prt))


def prLightGray(prt):
    """
    打印浅灰色
    :param prt:
    :return:
    """
    print("\033[97m {}\033[00m" .format(prt))


def prBlack(prt):
    """
    打印黑色
    :param prt:
    :return:
    """
    print("\033[98m {}\033[00m" .format(prt))


def to_numpy(var):
    """
    将数据转换为 numpy 格式，在 cpu 上计算
    :param var:
    :return:
    """
    return var.cpu().data.numpy() if USE_CUDA else var.data.numpy()


def to_tensor(ndarray, device):
    """
    将 ndarray 转换为 tensor 并在 gpu 上计算
    :param ndarray:
    :param device:
    :return:
    """
    return torch.tensor(ndarray, dtype=torch.float, device=device)


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


def hard_update(target, source):
    for m1, m2 in zip(target.modules(), source.modules()):
        m1._buffers = m2._buffers.copy()
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def get_output_folder(parent_dir, env_name):
    """
    Return save folder.
    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.
    Parameters
    ----------
    parent_dir: str
      Path of the directory containing all experiment runs.
    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    """
    os.makedirs(parent_dir, exist_ok=True)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)
    os.makedirs(parent_dir, exist_ok=True)
    return parent_dir