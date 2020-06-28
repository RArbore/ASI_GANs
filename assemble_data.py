import torch
import sys

truth = torch.load("TRIMMED64.pt")

truth = truth[0:2]

if sys.argv[1] == "wgan":
    wgan = torch.load("WGAN_DATA.pt").view(2, 300, 64, 64, 64).clamp(0, 1)
    torch.save(torch.cat((wgan, truth), dim=1), "TRUTH_AND_WGAN.pt")
elif sys.argv[1] == "dcgan":
    dcgan = torch.load("DCGAN_DATA.pt").view(2, 300, 64, 64, 64).clamp(0, 1)
    torch.save(torch.cat((dcgan, truth), dim=1), "TRUTH_AND_DCGAN.pt")