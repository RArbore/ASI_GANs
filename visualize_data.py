import torch
import torchvision.transforms as transforms
import os

data = torch.load("TRIMMED64.pt")

print(data.size())

def save_image(tensor, filename):
    ndarr = tensor.mul(256).clamp(0, 255).int().byte().cpu()
    image = transforms.ToPILImage()(ndarr)
    image.save(filename)

for num_blocks in range(1, 6):
    data_i = torch.nn.functional.interpolate(data, size=(2 * (2 ** num_blocks)))
    for type in range(0, 5):
        for image in range(0, 335):
            for dim in range(0, 2 * (2 ** num_blocks)):
                save_image(data_i[type, image, dim, :, :], "visualized_data/size_"+str(num_blocks)+"_type_" + str(type) + "_image_" + str(image + 1) + "_num_" + str(dim + 1) + ".png")