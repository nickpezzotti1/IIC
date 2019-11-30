import argparse
import itertools
import os
import pickle
import sys
from datetime import datetime

import matplotlib
import numpy as np
import torch
import torchvision

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import code.archs as archs
from code.utils.cluster.general import config_to_str, get_opt, update_lr, nice
from code.utils.cluster.data import cluster_twohead_create_dataloaders
from code.utils.cluster.cluster_eval import cluster_eval, get_subhead_using_loss
from code.utils.cluster.IID_losses import IID_loss
from code.utils.cluster.render import save_progress

"""
  Fully unsupervised clustering ("IIC" = "IID").
  Train and test script (greyscale datasets).
  Network has two heads, for overclustering and final clustering.
"""

# Options ----------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--model_ind", type=int, required=True)
parser.add_argument("--arch", type=str, default="ClusterNet4h")
parser.add_argument("--opt", type=str, default="Adam")
parser.add_argument("--mode", type=str, default="IID")

parser.add_argument("--dataset", type=str, default="MNIST")
parser.add_argument("--dataset_root", type=str,
                    default="/scratch/local/ssd/xuji/MNIST")

parser.add_argument("--gt_k", type=int, default=10)
parser.add_argument("--output_k_A", type=int, required=True)
parser.add_argument("--output_k_B", type=int, required=True)

parser.add_argument("--lamb_A", type=float, default=1.0)
parser.add_argument("--lamb_B", type=float, default=1.0)

parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--lr_schedule", type=int, nargs="+", default=[])
parser.add_argument("--lr_mult", type=float, default=0.1)

parser.add_argument("--num_epochs", type=int, default=1000)
parser.add_argument("--batch_sz", type=int, default=240)  # num pairs
parser.add_argument("--num_dataloaders", type=int, default=3)
parser.add_argument("--num_sub_heads", type=int, default=5)

parser.add_argument("--out_root", type=str,
                    default="/scratch/shared/slow/xuji/iid_private")
parser.add_argument("--restart", dest="restart", default=False,
                    action="store_true")
parser.add_argument("--restart_from_best", dest="restart_from_best",
                    default=False, action="store_true")
parser.add_argument("--test_code", dest="test_code", default=False,
                    action="store_true")

parser.add_argument("--save_freq", type=int, default=20)

parser.add_argument("--double_eval", default=False, action="store_true")

parser.add_argument("--head_A_first", default=False, action="store_true")
parser.add_argument("--head_A_epochs", type=int, default=1)
parser.add_argument("--head_B_epochs", type=int, default=1)

parser.add_argument("--batchnorm_track", default=False, action="store_true")

parser.add_argument("--save_progression", default=False, action="store_true")

parser.add_argument("--select_sub_head_on_loss", default=False,
                    action="store_true")

# transforms
parser.add_argument("--demean", dest="demean", default=False,
                    action="store_true")
parser.add_argument("--per_img_demean", dest="per_img_demean", default=False,
                    action="store_true")
parser.add_argument("--data_mean", type=float, nargs="+",
                    default=[0.5, 0.5, 0.5])
parser.add_argument("--data_std", type=float, nargs="+",
                    default=[0.5, 0.5, 0.5])

parser.add_argument("--crop_orig", dest="crop_orig", default=False,
                    action="store_true")
parser.add_argument("--crop_other", dest="crop_other", default=False,
                    action="store_true")
parser.add_argument("--tf1_crop", type=str, default="random")  # type name
parser.add_argument("--tf2_crop", type=str, default="random")
parser.add_argument("--tf1_crop_sz", type=int, default=84)
parser.add_argument("--tf2_crop_szs", type=int, nargs="+",
                    default=[84])  # allow diff crop for imgs_tf
parser.add_argument("--tf3_crop_diff", dest="tf3_crop_diff", default=False,
                    action="store_true")
parser.add_argument("--tf3_crop_sz", type=int, default=0)
parser.add_argument("--input_sz", type=int, default=96)

parser.add_argument("--rot_val", type=float, default=0.)
parser.add_argument("--always_rot", dest="always_rot", default=False,
                    action="store_true")
parser.add_argument("--no_jitter", dest="no_jitter", default=False,
                    action="store_true")
parser.add_argument("--no_flip", dest="no_flip", default=False,
                    action="store_true")

config = parser.parse_args()

# Setup ------------------------------------------------------------------------

config.twohead = True
config.in_channels = 1
config.out_dir = os.path.join(config.out_root, str(config.model_ind))
assert (config.batch_sz % config.num_dataloaders == 0)
config.dataloader_batch_sz = config.batch_sz / config.num_dataloaders

assert (config.mode == "IID")
assert ("TwoHead" in config.arch)
assert (config.output_k_B == config.gt_k)
config.output_k = config.output_k_B  # for eval code
assert (config.output_k_A >= config.gt_k)
config.eval_mode = "hung"

assert ("MNIST" == config.dataset)
dataset_class = torchvision.datasets.MNIST
config.train_partitions = [True, False]
config.mapping_assignment_partitions = [True, False]
config.mapping_test_partitions = [True, False]

if not os.path.exists(config.out_dir):
  os.makedirs(config.out_dir)

if config.restart:
  config_name = "config.pickle"
  net_name = "latest_net.pytorch"
  opt_name = "latest_optimiser.pytorch"

  if config.restart_from_best:
    config_name = "best_config.pickle"
    net_name = "best_net.pytorch"
    opt_name = "best_optimiser.pytorch"

  given_config = config
  reloaded_config_path = os.path.join(given_config.out_dir, config_name)
  print("Loading restarting config from: %s" % reloaded_config_path)
  with open(reloaded_config_path, "rb") as config_f:
    config = pickle.load(config_f)
  assert (config.model_ind == given_config.model_ind)
  config.restart = True

  # copy over new num_epochs and lr schedule
  config.num_epochs = given_config.num_epochs
  config.lr_schedule = given_config.lr_schedule
  config.save_progression = given_config.save_progression

  if not hasattr(config, "batchnorm_track"):
    config.batchnorm_track = True  # before we added in false option

  if not hasattr(config, "lamb_A"):
    config.lamb_A = config.lamb
    config.lamb_B = config.lamb

else:
  print("Config: %s" % config_to_str(config))



# Create the model
net = archs.__dict__[config.arch](config)
net.load_state_dict(torch.load("/content/code/best_net.pytorch"))

# Get example of MNIST
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, TensorDataset

dl = DataLoader(torchvision.datasets.MNIST('/data/mnist', train=True, download=True, transform=transforms.Compose([
                       transforms.ToTensor()
                   ])))

example = enumerate(dl).next()

# Predict
predicted = torch.argmax(net(example[1][0])[0]).item()
label = example[1][1][0].item()
print("Predicted " + str(predicted) + " for " + str(label))

