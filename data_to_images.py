import torch
from torchvision import transforms

def save_image(tensor, filename):
    ndarr = tensor.mul(255).clamp(0, 255).int().byte().cpu()
    image = transforms.ToPILImage()(ndarr)
    image.save(filename)

data = torch.load("WGAN_5000_DATA.pt")

for dp in range(335):
    for c in range(2):
        save_image(data[dp, c, 32, :, :], "dataset_images/"+str(dp)+"_"+str(c)+".png")

