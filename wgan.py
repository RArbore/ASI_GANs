import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import math
import time
import pickle

manualSeed = int(torch.rand(1).item() * 1000)
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

ngpu = 1

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

def current_milli_time():
    return int(round(time.time() * 1000))

before_time = current_milli_time()

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

files = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]

tensor_images_list = []

for f in files:
    dict = unpickle("cifar-10-batches-py/"+f)
    np_images = dict[b"data"]
    tensor_images = torch.from_numpy(np_images)
    tensor_images = tensor_images.view(10000, 3, 32, 32)
    tensor_images_list.append(tensor_images)

data = torch.cat(tensor_images_list, dim=0)
print(data.size())