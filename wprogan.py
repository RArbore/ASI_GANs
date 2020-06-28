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

manualSeed = int(torch.rand(1).item() * 1000)
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

train_data_size = 300

batch_size = 3

image_size = 64

nc = 2

nz = 256

ngf = 64

ndf = 64

num_epochs = 600

epochs_per_block = 100.0

lr = 0.0002

beta1 = 0.5

ngpu = 1

num_blocks = 1

total_blocks = 5

alpha = 0

lambda_term = 10

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

print("Loading data...")
data = torch.load("TRIMMED64.pt")

after_time = current_milli_time()
seconds = math.floor((after_time - before_time) / 1000)
minutes = math.floor(seconds / 60)
seconds = seconds % 60
print("Data loading took " + str(minutes) + " minute(s) " + str(seconds) + " second(s).")


def weights_init(m):
    classname = m.__class__.__name__
    if "Conv3d" in classname:
        torch.nn.init.xavier_uniform_(m.weight.data)


def check_nan(x):
    return torch.sum(x != x) > 0


class FeatureNormLayer(nn.Module):
    def __init__(self, eps=1e-8):
        super(FeatureNormLayer, self).__init__()
        self.eps = eps

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)


class BlockedGenerator(nn.Module):
    def __init__(self, ngpu):
        super(BlockedGenerator, self).__init__()
        self.ngpu = ngpu
        self.block0 = nn.Sequential(
            nn.Conv3d(nz, ngf * 8, 4, 1, 3, bias=False),
            FeatureNormLayer(),
            #nn.BatchNorm3d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(ngf * 8, ngf * 8, 3, 1, 1, bias=False),
            FeatureNormLayer(),
        )
        self.block1 = nn.Sequential(
            #nn.BatchNorm3d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv3d(ngf * 8, ngf * 4, 3, 1, 1, bias=False),
            FeatureNormLayer(),
            #nn.BatchNorm3d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(ngf * 4, ngf * 4, 3, 1, 1, bias=False),
            FeatureNormLayer(),
        )
        self.block2 = nn.Sequential(
            #nn.BatchNorm3d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv3d(ngf * 4, ngf * 2, 3, 1, 1, bias=False),
            FeatureNormLayer(),
            #nn.BatchNorm3d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(ngf * 2, ngf * 2, 3, 1, 1, bias=False),
            FeatureNormLayer(),
        )
        self.block3 = nn.Sequential(
            #nn.BatchNorm3d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv3d(ngf * 2, ngf, 3, 1, 1, bias=False),
            FeatureNormLayer(),
            #nn.BatchNorm3d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(ngf, ngf, 3, 1, 1, bias=False),
            FeatureNormLayer(),
        )
        self.block4 = nn.Sequential(
            #nn.BatchNorm3d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv3d(ngf, nc, 3, 1, 1, bias=False),
            FeatureNormLayer(),
            #nn.BatchNorm3d(nc),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(nc, nc, 3, 1, 1, bias=False),
            FeatureNormLayer(),
        )

        self.rgb8 = nn.Conv3d(ngf * 8, nc, 1)
        self.rgb4 = nn.Conv3d(ngf * 4, nc, 1)
        self.rgb2 = nn.Conv3d(ngf * 2, nc, 1)
        self.rgb1 = nn.Conv3d(ngf * 1, nc, 1)
        self.rgb0 = nn.Conv3d(nc, nc, 1)

        self.rgbs = [self.rgb8, self.rgb4, self.rgb2, self.rgb1, self.rgb0]

        self.blocks = [self.block0, self.block1, self.block2, self.block3, self.block4]

    def forward(self, input):
        '''
        #if num_blocks == 1 or alpha == 1:
        #    for block in range(0, num_blocks):
        #        layer = self.blocks[block]
        #        out = layer(out)
        #    if num_blocks < total_blocks:
        #        out = self.rgbs[num_blocks-1](out)
        #else:
        for block in range(0, num_blocks-1):
            layer = self.blocks[block]
            out = layer(out)
        before = out
        after = self.blocks[num_blocks-1](out)
        before = nn.functional.interpolate(before, scale_factor=2)
        if num_blocks < total_blocks:
            after = self.rgbs[num_blocks-1](after)
        if num_blocks-1 < total_blocks:
            before = self.rgbs[num_blocks-2](before)
        out = (1.0-alpha)*before+alpha*after
        '''

        depth = num_blocks - 1

        y = self.blocks[0](input)

        if depth > 0:
            for block in self.blocks[1:depth]:
                y = block(y)

            residual = self.rgbs[depth - 1](nn.functional.interpolate(y, scale_factor=2))
            straight = self.rgbs[depth](self.blocks[depth](y))

            out = (alpha * straight) + ((1.0 - alpha) * residual)

        else:
            out = self.rgbs[0](y)

        return (torch.tanh(out)+1)/2


netG = BlockedGenerator(ngpu).to(device)

if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

netG.apply(weights_init)

def get_noise(d_loss):
    #return math.exp(-8 * d_loss)
    return 0

class BlockedDiscriminator(nn.Module):
    def __init__(self, ngpu):
        super(BlockedDiscriminator, self).__init__()
        self.ngpu = ngpu
        self.block0 = nn.Sequential(
            nn.Conv3d(nc, ndf, 3, 1, 1, bias=False),
            FeatureNormLayer(),
            #nn.InstanceNorm3d(ndf, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(ndf, ndf, 3, 1, 1, bias=False),
            FeatureNormLayer(),
            #nn.InstanceNorm3d(ndf, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool3d(kernel_size=2),
        )
        self.block1 = nn.Sequential(
            nn.Conv3d(ndf, ndf * 2, 3, 1, 1, bias=False),
            FeatureNormLayer(),
            #nn.InstanceNorm3d(ndf * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(ndf * 2, ndf * 2, 3, 1, 1, bias=False),
            FeatureNormLayer(),
            #nn.InstanceNorm3d(ndf * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool3d(kernel_size=2),
        )
        self.block2 = nn.Sequential(
            nn.Conv3d(ndf * 2, ndf * 4, 3, 1, 1, bias=False),
            FeatureNormLayer(),
            #nn.InstanceNorm3d(ndf * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(ndf * 4, ndf * 4, 3, 1, 1, bias=False),
            FeatureNormLayer(),
            #nn.InstanceNorm3d(ndf * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool3d(kernel_size=2),
        )
        self.block3 = nn.Sequential(
            nn.Conv3d(ndf * 4, ndf * 8, 3, 1, 1, bias=False),
            FeatureNormLayer(),
            #nn.InstanceNorm3d(ndf * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(ndf * 8, ndf * 8, 3, 1, 1, bias=False),
            FeatureNormLayer(),
            #nn.InstanceNorm3d(ndf * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool3d(kernel_size=2),
        )
        self.block4 = nn.Sequential(
            nn.Conv3d(ndf * 8, ndf * 16, 3, 1, 1, bias=False),
            FeatureNormLayer(),
            #nn.InstanceNorm3d(ndf * 16, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(ndf * 16, 1, 4, 1, 0, bias=False),
            FeatureNormLayer(),
        )

        self.rgb8 = nn.Conv3d(nc, ndf * 8, 1)
        self.rgb4 = nn.Conv3d(nc, ndf * 4, 1)
        self.rgb2 = nn.Conv3d(nc, ndf * 2, 1)
        self.rgb1 = nn.Conv3d(nc, ndf * 1, 1)
        self.rgb0 = nn.Conv3d(nc, nc, 1)

        self.rgbs = [self.rgb8, self.rgb4, self.rgb2, self.rgb1, self.rgb0]

        self.blocks = [self.block0, self.block1, self.block2, self.block3, self.block4]

    def forward(self, input, d_loss):
        '''
        #if num_blocks == 1 or alpha == 0 or alpha == 1:
        #    if num_blocks < total_blocks:
        #        out = self.rgbs[num_blocks-1](out)
        #    for block in range(total_blocks-num_blocks, total_blocks):
        #        layer = self.blocks[block]
        #        out = layer(out)
        #else:
        before = out
        after = out
        before = nn.functional.interpolate(before, scale_factor=0.5)
        if num_blocks < total_blocks:
            after = self.rgbs[num_blocks - 1](after)
        if num_blocks - 1 < total_blocks:
            before = self.rgbs[num_blocks - 2](before)
        after = self.blocks[total_blocks - num_blocks](after)
        out = (1.0 - alpha) * before + alpha * after
        for block in range(total_blocks - num_blocks + 1, total_blocks):
            layer = self.blocks[block]
            out = layer(out)
        '''

        height = num_blocks - 1

        loss_noise_prop = get_noise(d_loss)

        if height > 0:
            residual = self.rgbs[height - 1](nn.functional.interpolate(input, scale_factor=0.5))

            block_input = self.rgbs[height](input);

            block_input += (torch.randn(block_input.size()) * loss_noise_prop).to(device);

            straight = self.blocks[total_blocks - height - 1](block_input);

            y = (alpha * straight) + ((1.0 - alpha) * residual)

            for block in self.blocks[total_blocks - height:total_blocks - 1]:
                y += (torch.randn(y.size()) * loss_noise_prop).to(device);

                y = block(y)

        else:
            y = self.rgbs[0](input)

        y += (torch.randn(y.size()) * loss_noise_prop).to(device);

        out = self.blocks[total_blocks - 1](y)

        return torch.sigmoid(out)

def calculate_gradient_penalty(netD, real_images, fake_images, d_loss):
    eta = torch.FloatTensor(batch_size, 2, 1, 1, 1).uniform_(0, 1).to(device)
    eta = eta.expand(batch_size, 2, 2 * (2 ** num_blocks), 2 * (2 ** num_blocks), 2 * (2 ** num_blocks))

    interpolated = (eta * real_images + ((1 - eta) * fake_images)).to(device)

    # define it to calculate gradient
    interpolated = torch.autograd.Variable(interpolated, requires_grad=True)

    # calculate probability of interpolated examples
    prob_interpolated = netD(interpolated, d_loss)

    # calculate gradients of probabilities with respect to examples
    gradients = torch.autograd.grad(outputs=prob_interpolated, inputs=interpolated, grad_outputs=torch.ones(prob_interpolated.size()).to(device), create_graph=True, retain_graph=True)[0]

    grad_penalty = (((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_term).to(device)
    return grad_penalty


netD = BlockedDiscriminator(ngpu).to(device)

if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

netD.apply(weights_init)

optimizerD = optim.Adadelta(netD.parameters(), lr=lr)
optimizerG = optim.Adadelta(netG.parameters(), lr=lr)

G_losses = []
D_losses = []

one = torch.tensor(1, dtype=torch.float).to(device)
mone = one * -1
last_d_loss = 0;

def cross_entropy(pred, label):
    return -torch.mean(label*torch.log(pred)+(1-label)*torch.log((1-pred)))

def save_image(tensor, filename):
    ndarr = tensor.mul(256).clamp(0, 255).int().byte().cpu()
    image = transforms.ToPILImage()(ndarr)
    image.save(filename)


if not os.path.isdir(folder + "/dcgan_output"):
    os.mkdir(folder + "/dcgan_output")
if not os.path.isdir(folder + "/gan_models"):
    os.mkdir(folder + "/gan_models")
f = open(folder + "/gan_performance.txt", "a")
gengradf = open(folder + "/gen_grad.txt", "a")
disgradf = open(folder + "/dis_grad.txt", "a")

print("Starting Training Loop...")
for epoch in range(num_epochs):
    epoch_before_time = current_milli_time()
    if not os.path.isdir(folder + "/dcgan_output/epoch_" + str(epoch)):
        os.mkdir(folder + "/dcgan_output/epoch_" + str(epoch))
    for batch in range(int(train_data_size / batch_size)):
        noise = torch.randn(batch_size, nz, 1, 1, 1, device=device)
        fake = netG(noise)
        if (check_nan(fake)):
            print(0)

        for p in netD.parameters():
            p.requires_grad = True

        nonseg_tensor = data[0][batch * batch_size:(batch + 1) * batch_size].view(batch_size, 1, image_size, image_size, image_size).to(device)
        seg_tensor = data[1][batch * batch_size:(batch + 1) * batch_size].view(batch_size, 1, image_size, image_size, image_size).to(device)
        nonseg_tensor = nn.functional.interpolate(nonseg_tensor, size=(2 * (2 ** num_blocks)))
        seg_tensor = nn.functional.interpolate(seg_tensor, size=(2 * (2 ** num_blocks)))
        netD.zero_grad()
        real = torch.cat((nonseg_tensor, seg_tensor), dim=1).to(device)
        output = netD(real, last_d_loss).view(-1)
        errD_real = output.mean()
        errD_real.backward(mone)
        D_x = errD_real.item()

        output = netD(fake.detach(), last_d_loss).view(-1)
        errD_fake = output.mean()
        errD_fake.backward(one)
        D_G_z1 = errD_fake.item()

        gradient_penalty = calculate_gradient_penalty(netD, real, fake, last_d_loss)
        gradient_penalty.backward()

        errD = errD_fake - errD_real + gradient_penalty
        w_d = errD_real - errD_fake
        last_d_loss = errD_fake.item() - errD_real.item()
        optimizerD.step()

        for p in netD.parameters():
            p.requires_grad = False

        #errD = torch.tensor(0)
        #D_x = 0
        #D_G_z1 = 0
        netG.zero_grad()
        output = netD(fake, last_d_loss).view(-1)
        errG = output.mean()
        #errG = criterion(fake, torch.ones(fake.size()).to(device)*0.2)
        errG.backward(mone)
        D_G_z2 = output.mean().item()
        optimizerG.step()

        if batch == (train_data_size / batch_size) - 1:
            D_G_z2_epoch = D_G_z2
            epoch_after_time = current_milli_time()
            seconds = math.floor((epoch_after_time - epoch_before_time) / 1000)
            minutes = math.floor(seconds / 60)
            seconds = seconds % 60

            #print(errG.item())
            #print(torch.mean(fake))
            #print("")
            f.write(str(epoch + 1) + " " + str(errD.item()) + " " + str(errG.item()) + " " + str(D_x) + " " + str(
                D_G_z1) + " " + str(D_G_z2) + "\n")
            gen_abs_mean = 0
            gen_std = 0
            for b in range(0, 5):
                for layer in netG.blocks[b]:
                    if "Conv3d" in str(layer) and not layer.weight.grad is None:
                        gen_abs_mean = torch.mean(torch.abs(layer.weight.grad)).item()
                        gen_std = torch.std(layer.weight.grad).item()
            gengradf.write(str(gen_abs_mean) + " " + str(gen_std) + "\n")
            dis_abs_mean = 0
            dis_std = 0
            for b in range(0, 5):
                for layer in netD.blocks[b]:
                    if "Conv3d" in str(layer) and not layer.weight.grad is None:
                        dis_abs_mean = torch.mean(torch.abs(layer.weight.grad)).item()
                        dis_std = torch.std(layer.weight.grad).item()
            disgradf.write(str(dis_abs_mean) + " " + str(dis_std) + "\n")
            print(('[%d] LD: %.4f LG: %.4f D(x): %.4f D(G(z)): %.4f / %.4f Alpha: %.4f Number of Blocks: %.4f Gen Grad Abs Avg: %.2E Gen Grad Std: %.2E Dis Grad Abs Avg: %.2E Dis Grad Std: %.2E D_Noise: %.2E' % (epoch, errD.item(), errG.item(), D_x, D_G_z1, D_G_z2, alpha, num_blocks, gen_abs_mean, gen_std, dis_abs_mean, dis_std, get_noise(last_d_loss))) + " Time: " + str(minutes) + "M " + str(seconds) + "S")
            if epoch % 50 == 49:
                print("Writing models...")
                torch.save(netD.state_dict(), folder + "/gan_models/dis_at_e" + str(epoch + 1) + ".pt")
                torch.save(netG.state_dict(), folder + "/gan_models/gen_at_e" + str(epoch + 1) + ".pt")
            for image in range(0, batch_size):
                for dim in range(0, fake.size()[2]):
                    save_image(fake[image, 0, dim, :, :],
                               folder + "/dcgan_output/epoch_" + str(epoch) + "/nonseg_image" + str(
                                   image + 1) + "_num" + str(dim + 1) + ".png")
                    save_image(fake[image, 1, dim, :, :],
                               folder + "/dcgan_output/epoch_" + str(epoch) + "/seg_image" + str(
                                   image + 1) + "_num" + str(dim + 1) + ".png")
        G_losses.append(errG.item())
        D_losses.append(errD.item())

    alpha += 1.0 / epochs_per_block
    #'''
    if alpha >= 1:
        alpha = 1
    if alpha == 1 and num_blocks < total_blocks:
        num_blocks += 1
        alpha = 0
    #'''
f.close()
gengradf.close()
disgradf.close()