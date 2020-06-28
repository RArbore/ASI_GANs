import torch
import torch.nn as nn

ngf = 64

nz = 256

nc = 2

device = torch.device("cpu")

class DCGenerator(nn.Module):
    def __init__(self):
        super(DCGenerator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose3d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm3d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose3d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose3d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose3d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose3d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

class WGenerator(nn.Module):

    def __init__(self):
        super(WGenerator, self).__init__()
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
'''
dcg = DCGenerator().to(device)
dcg.load_state_dict(torch.load("gantrial2/gan_models/gen_at_e1250.pt"))

wg = WGenerator().to(device)
wg.load_state_dict(torch.load("gantrial1/gan_models/gen_at_e1500.pt"))
'''

wg = WGenerator().to(device)
wg.load_state_dict(torch.load("gantrial3/gan_models/gen_at_e5000.pt"))

#dcouts = []
wouts = []

for batch in range(0, 660):
    noise = torch.randn(5, nz, 1, 1, 1, device=device).float()
    #dcout = dcg(noise)
    wout = wg(noise)
    #dcouts.append(dcout)
    wouts.append(wout)
    print(batch)

#dcout = torch.cat(dcouts, dim=0)
wout = torch.cat(wouts, dim=0)

#print(dcout.size())
print(wout.size())

#torch.save(dcout, "DCGAN_DATA.pt")
#torch.save(wout, "WGAN_DATA.pt")
torch.save(wout, "WGAN_5000_DATA.pt")
