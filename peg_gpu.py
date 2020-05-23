import torch
import time

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

def current_milli_time():
    return int(round(time.time() * 1000))

while True:
    before_time = current_milli_time()

    a = torch.rand(100, 100, 100).to(device)
    b = torch.rand(100, 100, 100).to(device)
    c = torch.rand(100, 100, 100).to(device)
    d = a*b+c

    after_time = current_milli_time()
    seconds = (after_time - before_time) / 1000
    print(seconds)