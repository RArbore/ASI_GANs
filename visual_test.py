import torch
import torch.nn as nn
import random
import torchvision.transforms as transforms

image_size = 64

ngpu = 1

nz = 100

nc = 2

ngf = 64

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose3d(nz, ngf * 8, 4, 1, 0),
            #nn.Conv3d(ngf * 8, ngf * 8, 3, 1, 1),
            nn.BatchNorm3d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose3d(ngf * 8, ngf * 4, 4, 2, 1),
            #nn.Conv3d(ngf * 4, ngf * 4, 3, 1, 1),
            nn.BatchNorm3d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose3d(ngf * 4, ngf * 2, 4, 2, 1),
            #nn.Conv3d(ngf * 2, ngf * 2, 3, 1, 1),
            nn.BatchNorm3d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose3d(ngf * 2, ngf, 4, 2, 1),
            #nn.Conv3d(ngf, ngf, 3, 1, 1),
            nn.BatchNorm3d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose3d(ngf, nc, 4, 2, 1),
            #nn.Conv3d(nc, nc, 3, 1, 1),
        )

    def forward(self, input):
        out = self.layers(input)
        return torch.tanh(out)

netG = Generator().to(device)
netG.load_state_dict(torch.load("gantrial148/gan_models/gen_at_e4950.pt"))

data = torch.load("TRIMMED64.pt")
data = data.permute(1, 0, 2, 3, 4)[:, 0:2, :, :, :]
data = data.view(335, 2, 64, 64, 64).to(device)

noise = torch.randn(8, nz, 1, 1, 1, device=device)
fake = netG(noise)

image_list = []
alr_selected = []
for i in range(8):
    slice = int(torch.rand(1).item() * 335)
    while slice in alr_selected:
        slice = int(torch.rand(1).item() * 335)
    a = data[slice]
    a[1] = 1
    a = a[:, 32, :, :]
    image_list.append(a)
    b = fake[i]
    b[1] = 0
    b = b[:, 32, :, :]
    image_list.append(b)
random.shuffle(image_list)

def save_image(tensor, filename):
    ndarr = tensor.mul(256).clamp(0, 255).int().byte().cpu()
    image = transforms.ToPILImage()(ndarr)
    image.save(filename)

image = None
for y in range(4):
    row = None
    for x in range(4):
        if row == None:
            row = image_list[x + y * 4]
        else:
            row = torch.cat((row, image_list[x + y * 4]), dim=2)
    if image == None:
        image = row
    else:
        image = torch.cat((image, row), dim=1)
save_image(image[0], "scans.png")
save_image(image[1], "labels.png")