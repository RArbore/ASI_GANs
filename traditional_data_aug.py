import torch
from torchvision import transforms
import torch.nn as nn
import numpy as np
import elasticdeform

def save_image(tensor, filename):
    ndarr = tensor.mul(255).clamp(0, 255).int().byte().cpu()
    image = transforms.ToPILImage()(ndarr)
    image.save(filename)

data = torch.load("TRIMMED64.pt")[0:2]

flipped_data = torch.flip(data[:, 0:300, :, :, :], [3])

data = torch.cat((flipped_data, data), dim=1)

np_data = data.numpy()

deformed_tensors = []

for x in range(0, 1):
    for i in range(600):
        print(x, i)
        scan_image = np_data[0, i, :, :, :]
        seg_image = np_data[1, i, :, :, :]
        [scan_deformed, seg_deformed] = elasticdeform.deform_random_grid([scan_image, seg_image], sigma=1.0, points=4)
        scan_deformed = torch.from_numpy(scan_deformed)
        seg_deformed = torch.from_numpy(seg_deformed)
        deformed_tensors.append(torch.stack([scan_deformed, seg_deformed]))

deformed_data = torch.stack(deformed_tensors).permute(1, 0, 2, 3, 4)

write_data = torch.cat((deformed_data, data), dim=1)

torch.save(write_data, "DATA_AUG64.pt")