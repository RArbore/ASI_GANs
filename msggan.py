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

train_data_size = 335

batch_size = 5

num_epochs = 5000

image_size = 64

ngpu = 1

nz = 100

nc = 2

ngf = 8

ndf = 8

lr = 0.0002

LAMBDA = 10

b1 = 0.5
b2 = 0.999

critic_iter = 2

gen_iter = 1

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

def current_milli_time():
    return int(round(time.time() * 1000))

before_time = current_milli_time()

files = os.listdir(".")
m = [int(f[8:]) for f in files if len(f) > 8 and f[0:8] == "gantrial"]
if len(m) > 0:
    folder = "gantrial" + str(max(m) + 1)
else:
    folder = "gantrial1"
os.mkdir(folder)

print("Created session folder " + folder)


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

files = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]

tensor_images_list = []

'''
for f in files:
    dict = unpickle("cifar-10-batches-py/"+f)
    np_images = dict[b"data"]
    tensor_images = torch.from_numpy(np_images)
    tensor_images = tensor_images.view(10000, 3, 32, 32)
    tensor_images_list.append(tensor_images)

data = torch.cat(tensor_images_list, dim=0).float()
data = torch.mean(data, dim=1).view(50000, 1, 32, 32)

'''
data = torch.load("TRIMMED64.pt")
data = data.permute(1, 0, 2, 3, 4)[:, 0:2, :, :, :]*256
data = data.view(335, 2, 64, 64, 64)
#data = nn.functional.interpolate(data, scale_factor = 0.5)


after_time = current_milli_time()
seconds = math.floor((after_time - before_time) / 1000)
minutes = math.floor(seconds / 60)
seconds = seconds % 60
print("Data loading took " + str(minutes) + " minute(s) " + str(seconds) + " second(s).")

#def weights_init(m):
#    classname = m.__class__.__name__
#    if "Conv3d" in classname:
#        torch.nn.init.xavier_uniform_(m.weight.data)

def check_nan(x):
    return torch.sum(x != x) > 0

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        self.block1 = nn.Sequential(
            nn.ConvTranspose3d(nz, ngf * 8, 4, 1, 0),
            nn.BatchNorm3d(ngf * 8),
            nn.ReLU(True),
            nn.Conv3d(ngf * 8, ngf * 8, 3, 1, 1),
            nn.BatchNorm3d(ngf * 8),
            nn.ReLU(True),
        )
        self.block2 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv3d(ngf * 8, ngf * 4, 3, 1, 1),
            nn.BatchNorm3d(ngf * 4),
            nn.ReLU(True),
            nn.Conv3d(ngf * 4, ngf * 4, 3, 1, 1),
            nn.BatchNorm3d(ngf * 4),
            nn.ReLU(True),
        )
        self.block3 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv3d(ngf * 4, ngf * 2, 3, 1, 1),
            nn.BatchNorm3d(ngf * 2),
            nn.ReLU(True),
            nn.Conv3d(ngf * 2, ngf * 2, 3, 1, 1),
            nn.BatchNorm3d(ngf * 2),
            nn.ReLU(True),
        )
        self.block4 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv3d(ngf * 2, ngf, 3, 1, 1),
            nn.BatchNorm3d(ngf),
            nn.ReLU(True),
            nn.Conv3d(ngf, ngf, 3, 1, 1),
            nn.BatchNorm3d(ngf),
            nn.ReLU(True),
        )
        self.block5 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv3d(ngf, nc, 3, 1, 1),
            nn.BatchNorm3d(nc),
            nn.ReLU(True),
            nn.Conv3d(nc, nc, 3, 1, 1),
        )
        self.blocks = [self.block1, self.block2, self.block3, self.block4, self.block5]
        self.rgb1 = nn.Sequential(
            nn.Conv3d(ngf * 8, nc, 1)
        )
        self.rgb2 = nn.Sequential(
            nn.Conv3d(ngf * 4, nc, 1)
        )
        self.rgb3 = nn.Sequential(
            nn.Conv3d(ngf * 2, nc, 1)
        )
        self.rgb4 = nn.Sequential(
            nn.Conv3d(ngf, nc, 1)
        )
        self.rgbs = [self.rgb1, self.rgb2, self.rgb3, self.rgb4]

    def forward(self, input):
        out = input
        h_reps = []
        i = 0
        for block in self.blocks:
            out = block(out)
            h_reps.append(out)
            if i < len(self.blocks) - 1:
                h_reps[i] = self.rgbs[i](h_reps[i])
                i += 1
        return h_reps

netG = Generator().to(device)

#netG.apply(weights_init)

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv3d(nc, ndf, 3, 1, 1),
            nn.InstanceNorm3d(ndf, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(ndf, ndf, 3, 1, 1),
            nn.InstanceNorm3d(ndf, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool3d(kernel_size=2),
        )
        self.block2 = nn.Sequential(
            nn.Conv3d(ndf + nc, ndf * 2, 3, 1, 1),
            nn.InstanceNorm3d(ndf * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(ndf * 2, ndf * 2, 3, 1, 1),
            nn.InstanceNorm3d(ndf * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool3d(kernel_size=2),
        )
        self.block3 = nn.Sequential(
            nn.Conv3d(ndf * 2 + nc, ndf * 4, 3, 1, 1),
            nn.InstanceNorm3d(ndf * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(ndf * 4, ndf * 4, 3, 1, 1),
            nn.InstanceNorm3d(ndf * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool3d(kernel_size=2),
        )
        self.block4 = nn.Sequential(
            nn.Conv3d(ndf * 4 + nc, ndf * 8, 3, 1, 1),
            nn.InstanceNorm3d(ndf * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(ndf * 8, ndf * 8, 3, 1, 1),
            nn.InstanceNorm3d(ndf * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool3d(kernel_size=2),
        )
        self.block5 = nn.Sequential(
            nn.Conv3d(ndf * 8 + nc, 1, 3, 1, 1),
            nn.InstanceNorm3d(1, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(1, 1, 4, 1, 0),
        )
        self.blocks = [self.block1, self.block2, self.block3, self.block4, self.block5]

    def forward(self, input):
        out = input[4]
        i = 0
        for block in self.blocks:
            if i > 0:
                out = torch.cat((out, input[4-i]), dim=1)
            out = block(out)
            i += 1
        return out

netD = Discriminator().to(device)

#netD.apply(weights_init)

optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(b1, b2))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(b1, b2))

#optimizerD = optim.Adadelta(netD.parameters(), lr=lr)
#ptimizerG = optim.Adadelta(netG.parameters(), lr=lr)

G_losses = []
D_losses = []

one = torch.tensor(1, dtype=torch.float).to(device)
mone = one * -1

def save_image(tensor, filename):
    ndarr = tensor.mul(256).clamp(0, 255).int().byte().cpu()
    image = transforms.ToPILImage()(ndarr)
    image.save(filename)

if not os.path.isdir(folder + "/gan_output"):
    os.mkdir(folder + "/gan_output")
if not os.path.isdir(folder + "/gan_models"):
    os.mkdir(folder + "/gan_models")
f = open(folder + "/gan_performance.txt", "a")
gengradf = open(folder + "/gen_grad.txt", "a")
disgradf = open(folder + "/dis_grad.txt", "a")

def calc_gradient_penalty(D, r, f):
    interpolated = []
    for i in range(5):
        real_images = r[i]
        fake_images = f[i]
        eta = torch.FloatTensor(batch_size, 1, 1, 1, 1).uniform_(0, 1).to(device)
        eta = eta.expand(batch_size, real_images.size(1), real_images.size(2), real_images.size(3), real_images.size(4))

        interpolated.append(torch.autograd.Variable((eta * real_images + ((1 - eta) * fake_images)).to(device), requires_grad=True))

    # calculate probability of interpolated examples
    prob_interpolated = D(interpolated)

    # calculate gradients of probabilities with respect to examples
    gradients = torch.autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                              grad_outputs=torch.ones(
                                  prob_interpolated.size()).to(device),
                              create_graph=True, retain_graph=True)[0]

    grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return grad_penalty

print("Starting Training Loop...")
for epoch in range(num_epochs):
    epoch_before_time = current_milli_time()
    if not os.path.isdir(folder + "/gan_output/epoch_" + str(epoch)):
        os.mkdir(folder + "/gan_output/epoch_" + str(epoch))
    for batch in range(int(train_data_size / batch_size)):
        noise = torch.randn(batch_size, nz, 1, 1, 1, device=device)
        fake = netG(noise)

        for p in netD.parameters():
            p.requires_grad = True

        D_x = 0
        D_G_z1 = 0
        errD = torch.tensor(0)

        for i in range(critic_iter):
            #if (epoch > 10) :
            #    break
            netD.zero_grad()
            real = []
            for i in range(5):
                real.append(nn.functional.avg_pool3d(data[batch * batch_size:(batch + 1) * batch_size].float().to(device)/256.0, kernel_size=2**(4-i)))
            output = netD(real).view(-1)
            errD_real = output.mean()
            errD_real.backward(mone)
            D_x = errD_real.item()

            output = netD(list(map(lambda x : x.detach(), fake))).view(-1)
            errD_fake = output.mean()
            errD_fake.backward(one)
            D_G_z1 = errD_fake.item()

            gradient_penalty = calc_gradient_penalty(netD, real, fake)
            gradient_penalty.backward()

            errD = errD_fake - errD_real# + gradient_penalty
            w_d = errD_real - errD_fake

            optimizerD.step()

        for i in range(gen_iter):
            for p in netD.parameters():
                p.requires_grad = False

            fake = netG(noise)

            # errD = torch.tensor(0)
            # D_x = 0
            # D_G_z1 = 0
            netG.zero_grad()
            output = netD(fake).view(-1)
            errG = output.mean()
            # errG = criterion(fake, torch.ones(fake.size()).to(device)*0.2)
            errG.backward(mone)
            D_G_z2 = output.mean().item()
            optimizerG.step()

        if batch == (train_data_size / batch_size) - 1:
            D_G_z2_epoch = D_G_z2
            epoch_after_time = current_milli_time()
            seconds = math.floor((epoch_after_time - epoch_before_time) / 1000)
            minutes = math.floor(seconds / 60)
            seconds = seconds % 60

            # print(errG.item())
            # print(torch.mean(fake))
            # print("")
            f.write(str(epoch + 1) + " " + str(errD.item()) + " " + str(errG.item()) + " " + str(D_x) + " " + str(
                D_G_z1) + " " + str(D_G_z2) + "\n")

            gen_abs_mean = 0
            gen_std = 0
            count = 0
            for block in netG.blocks:
                for layer in block:
                    if "Conv" in str(layer) and not layer.weight.grad is None:
                        gen_abs_mean += torch.mean(torch.abs(layer.weight.grad)).item()
                        gen_std += torch.std(layer.weight.grad).item()
                        count += 1
            gen_abs_mean /= count
            gen_std /= count
            gengradf.write(str(gen_abs_mean) + " " + str(gen_std) + "\n")

            dis_abs_mean = 0
            dis_std = 0
            for block in netD.blocks:
                for layer in block:
                    if "Conv" in str(layer) and not layer.weight.grad is None:
                        dis_abs_mean += torch.mean(torch.abs(layer.weight.grad)).item()
                        dis_std += torch.std(layer.weight.grad).item()
                        count += 1
            dis_abs_mean /= count
            dis_std /= count
            disgradf.write(str(dis_abs_mean) + " " + str(dis_std) + "\n")
            print((
                              '[%d] LD: %.4f LG: %.4f D(x): %.4f D(G(z)): %.4f / %.4f Gen Grad Abs Avg: %.2E Gen Grad Std: %.2E Dis Grad Abs Avg: %.2E Dis Grad Std: %.2E' % (
                      epoch, errD.item(), errG.item(), D_x, D_G_z1, D_G_z2, gen_abs_mean, gen_std,
                      dis_abs_mean, dis_std)) + " Time: " + str(minutes) + "M " + str(
                seconds) + "S")
            if epoch % 50 == 49:
                print("Writing models...")
                torch.save(netD.state_dict(), folder + "/gan_models/dis_at_e" + str(epoch + 1) + ".pt")
                torch.save(netG.state_dict(), folder + "/gan_models/gen_at_e" + str(epoch + 1) + ".pt")
            for image in range(0, batch_size):
                for dim in range(0, image_size):
                    save_image(fake[4][image, 0, dim, :, :], folder + "/gan_output/epoch_" + str(epoch) + "/scanimage" + str(image + 1) + "_dim" + str(dim + 1) + ".png")
                    save_image(fake[4][image, 1, dim, :, :], folder + "/gan_output/epoch_" + str(epoch) + "/segimage" + str(image + 1) + "_dim" + str(dim + 1) + ".png")
            #save_image(real[0, 0, :, :],folder + "/gan_output/epoch_" + str(epoch) + "/real_image" + str(image + 1) + ".png")
        G_losses.append(errG.item())
        D_losses.append(errD.item())
f.close()
gengradf.close()
disgradf.close()





















