import torch
from torchvision import transforms
import random
import time
import math
import os

manualSeed = int(torch.rand(1).item() * 1000)
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

#Define constants.

DATA_SIZE = 64*64*64

DATA_DIMENSIONS = [64, 64, 64]

TRAIN_DATA_SIZE = 3600

NUM_BATCHES = 360

BATCH_SIZE = int(TRAIN_DATA_SIZE/NUM_BATCHES)

VALIDATION_DATA_SIZE = 20

TESTING_DATA_SIZE = 15

VALIDATION_BATCH_SIZE = 5

TESTING_BATCH_SIZE = 5

NUM_EPOCHS = 500

BCE_COEFFICIENT = 10

nf = 16

kernel = 3

padding = 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cpu = torch.device("cpu")

folder = ""


#Define the network.

class UNet(torch.nn.Module):
    
    def __init__(self):
        super(UNet, self).__init__()
        self.s1 = torch.nn.Sequential(
            torch.nn.Conv3d(1, nf, kernel, 1, padding),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv3d(nf, nf, kernel, 1, padding),
            torch.nn.LeakyReLU(0.2),
        )
        self.s2 = torch.nn.Sequential(
            torch.nn.AvgPool3d(2),
            torch.nn.Conv3d(nf, nf * 2, kernel, 1, padding),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv3d(nf * 2, nf * 2, kernel, 1, padding),
            torch.nn.LeakyReLU(0.2),
        )
        self.s3 = torch.nn.Sequential(
            torch.nn.AvgPool3d(2),
            torch.nn.Conv3d(nf * 2, nf * 4, kernel, 1, padding),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv3d(nf * 4, nf * 4, kernel, 1, padding),
            torch.nn.LeakyReLU(0.2),
        )
        self.s4 = torch.nn.Sequential(
            torch.nn.AvgPool3d(2),
            torch.nn.Conv3d(nf * 4, nf * 8, kernel, 1, padding),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv3d(nf * 8, nf * 8, kernel, 1, padding),
            torch.nn.LeakyReLU(0.2),
        )
        self.s5 = torch.nn.Sequential(
            torch.nn.AvgPool3d(2),
            torch.nn.Conv3d(nf * 8, nf * 16, kernel, 1, padding),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv3d(nf * 16, nf * 16, kernel, 1, padding),
            torch.nn.LeakyReLU(0.2),
        )
        self.s6 = torch.nn.Sequential(
            torch.nn.Conv3d(nf * 16, nf * 8, kernel, 1, padding),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv3d(nf * 8, nf * 8, kernel, 1, padding),
            torch.nn.LeakyReLU(0.2),
        )
        self.s7 = torch.nn.Sequential(
            torch.nn.Conv3d(nf * 8, nf * 4, kernel, 1, padding),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv3d(nf * 4, nf * 4, kernel, 1, padding),
            torch.nn.LeakyReLU(0.2),
        )
        self.s8 = torch.nn.Sequential(
            torch.nn.Conv3d(nf * 4, nf * 2, kernel, 1, padding),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv3d(nf * 2, nf * 2, kernel, 1, padding),
            torch.nn.LeakyReLU(0.2),
        )
        self.s9 = torch.nn.Sequential(
            torch.nn.Conv3d(nf * 2, 2, kernel, 1, padding),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv3d(2, 2, kernel, 1, padding),
            torch.nn.Softmax(dim=1),
        )
        self.upconv1 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(nf * 16, nf * 8, 2, 2)
        )
        self.upconv2 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(nf * 8, nf * 4, 2, 2)
        )
        self.upconv3 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(nf * 4, nf * 2, 2, 2)
        )
        self.upconv4 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(nf * 2, nf, 2, 2)
        )
    
    def forward(self, input):
        before = input.view(input.size(0), 1, DATA_DIMENSIONS[0], DATA_DIMENSIONS[1], DATA_DIMENSIONS[2]).float()/256.0 #1, 64, 64, 64
        
        s1 = self.s1(before)
        s2 = self.s2(s1)
        s3 = self.s3(s2)
        s4 = self.s4(s3)
        s5 = self.s5(s4)
        s6 = self.s6(torch.cat((s4, self.upconv1(s5)), dim=1))
        s7 = self.s7(torch.cat((s3, self.upconv2(s6)), dim=1))
        s8 = self.s8(torch.cat((s2, self.upconv3(s7)), dim=1))
        final = self.s9(torch.cat((s1, self.upconv4(s8)), dim=1))
        
        out = final[:, 0, :, :, :].view(input.size(0), 1, DATA_DIMENSIONS[0], DATA_DIMENSIONS[1], DATA_DIMENSIONS[2])
        return out*torch.tensor(0.998)+torch.tensor(0.001)


#Define a function for saving images.
    
def save_image(tensor, filename):
    ndarr = tensor.mul(255).clamp(0, 255).int().byte().cpu()
    image = transforms.ToPILImage()(ndarr)
    image.save(filename)


# Define pixel-wise binary cross entropy.

def pixel_BCE(output_tensor, label_tensor):
    label_tensor = torch.min(label_tensor, torch.ones(label_tensor.size()).float().to(device))

    loss_tensor = BCE_COEFFICIENT * label_tensor * torch.log(output_tensor) + (
                torch.ones(label_tensor.size()).to(device) - label_tensor) * torch.log(
        torch.ones(output_tensor.size()).to(device) - output_tensor)

    return torch.mean(loss_tensor) * torch.tensor(-1).to(device)


#Define dice loss

def dice_loss(pred, label):
    return 2*torch.sum(pred*label)/(torch.sum(pred*pred)+torch.sum(label*label))

    
#Declare and train the network.
    
def train_model(data):
    model = UNet().to(device)
    
    opt = torch.optim.Adadelta(model.parameters())

    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    current_milli_time = lambda: int(round(time.time() * 1000))
    
    before_time = current_milli_time()

    print("Beginning Training with nf of "+str(nf)+".")
    print("")

    if not os.path.isdir(folder+"/control_u_net_image_output"):
        os.mkdir(folder+"/control_u_net_image_output")
    if not os.path.isdir(folder+"/during_training_models"):
        os.mkdir(folder+"/during_training_models")
        
    f = open(folder+"/during_training_performance.txt", "a")
    
    for epoch in range(0, NUM_EPOCHS):
        batch_loss = 0
        e_dice_l = 0
        epoch_before_time = current_milli_time()
        for batch in (range(0, int(TRAIN_DATA_SIZE/BATCH_SIZE))):
            input_tensor = data[0][batch*BATCH_SIZE:(batch+1)*BATCH_SIZE].to(device)
            label_tensor = data[1][batch*BATCH_SIZE:(batch+1)*BATCH_SIZE].to(device)
            label_tensor = torch.clamp(torch.ceil(label_tensor.float()), 0, 1).float().to(device).view(BATCH_SIZE, 1, DATA_DIMENSIONS[0], DATA_DIMENSIONS[1], DATA_DIMENSIONS[2])
            if (torch.sum(torch.min(input_tensor, torch.zeros(input_tensor.size()).to(device))) < 0):
                print("Input!!!!!!!!!!!!!!!!T")
            if (torch.sum(torch.min(label_tensor, torch.zeros(label_tensor.size()).to(device))) < 0):
                print("Label!!!!!!!!!!!!!!!!T")
            opt.zero_grad()
            output_tensor = model(input_tensor.float())
            if batch == 0:
                if not os.path.isdir(folder+"/control_u_net_image_output/epoch_"+str(epoch)):
                    os.mkdir(folder+"/control_u_net_image_output/epoch_"+str(epoch))
                for i in range(0, DATA_DIMENSIONS[0]):
                    save_image(output_tensor[0, 0, i, :, :], folder+"/control_u_net_image_output/epoch_"+str(epoch)+"/"+str(i+1)+".png")
                    save_image(input_tensor[0, i, :, :], folder + "/control_u_net_image_output/epoch_" + str(epoch) + "/" + str(i + 1) + "_data.png")
            train_loss = pixel_BCE(output_tensor, label_tensor.float())
            dice_l = dice_loss(output_tensor, label_tensor.float())
            train_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.02)
            opt.step()
            batch_loss += train_loss.item()
            e_dice_l += dice_l.item()
            #print("Batch "+str(batch+1)+" Loss : "+str(train_loss_item)+" Took "+str(minutes)+" minute(s) "+str(seconds)+" second(s).")
        valid_loss = 0
        v_dice_l = 0
        for i in range(0, int(VALIDATION_DATA_SIZE/VALIDATION_BATCH_SIZE)):
            valid_data = data[0:2, TRAIN_DATA_SIZE+i*VALIDATION_BATCH_SIZE:TRAIN_DATA_SIZE+(i+1)*VALIDATION_BATCH_SIZE, :, :, :]
            input_tensor = valid_data[0].to(device)
            label_tensor = valid_data[1].to(device)
            label_tensor = torch.clamp(torch.ceil(label_tensor.float()), 0, 1).float().to(device).view(VALIDATION_BATCH_SIZE, 1, DATA_DIMENSIONS[0], DATA_DIMENSIONS[1], DATA_DIMENSIONS[2])
            if (torch.sum(torch.min(input_tensor, torch.zeros(input_tensor.size()).to(device))) < 0):
                print("Input!!!!!!!!!!!!!!!!V")
            if (torch.sum(torch.min(label_tensor, torch.zeros(label_tensor.size()).to(device))) < 0):
                print("Label!!!!!!!!!!!!!!!!V")
            output_tensor = model(input_tensor.float())
            loss = pixel_BCE(output_tensor, label_tensor.float())
            dice_l = dice_loss(output_tensor, label_tensor.float())
            valid_loss += loss.item()
            v_dice_l += dice_l.item()
        valid_loss /= VALIDATION_DATA_SIZE/VALIDATION_BATCH_SIZE
        epoch_loss = batch_loss/(NUM_BATCHES)
        v_dice_l /= VALIDATION_DATA_SIZE/VALIDATION_BATCH_SIZE
        e_dice_l /= NUM_BATCHES

        gen_abs_mean = 0
        gen_std = 0
        count = 0
        for section in [model.s1, model.s2, model.s3, model.s4, model.s5, model.s6, model.s7, model.s8, model.s9]:
            for layer in section:
                if "Conv" in str(layer) and not layer.weight.grad is None:
                    gen_abs_mean += torch.mean(torch.abs(layer.weight.grad)).item()
                    gen_std += torch.std(layer.weight.grad).item()
                    count += 1
        gen_abs_mean /= count
        gen_std /= count

        epoch_after_time = current_milli_time()
        seconds = math.floor((epoch_after_time - epoch_before_time) / 1000)
        minutes = math.floor(seconds / 60)
        seconds = seconds % 60
        print("Epoch "+str(epoch+1)+" Loss : "+str(epoch_loss)+"   Validation Loss : "+str(valid_loss)+"   Dice Loss : "+str(e_dice_l)+"   Validation Dice Loss : "+str(v_dice_l)+" Took "+str(minutes)+" minute(s) "+str(seconds)+" second(s). "+str(gen_abs_mean)+" "+str(gen_std))
        f.write(str(epoch+1)+" "+str(epoch_loss)+" "+str(valid_loss)+" "+str(e_dice_l)+" "+str(v_dice_l)+" "+str(gen_abs_mean)+" "+str(gen_std)+"\n")
        if epoch % 50 == 49:
            print("Writing models...")
            torch.save(model.state_dict(), folder+"/during_training_models/model_at_e"+str(epoch+1)+".pt")
        if epoch+1 == NUM_EPOCHS:
            f.write("\n")

    after_time = current_milli_time()
    
    torch.save(model.state_dict(), folder+"/model.pt")
    
    t_test_loss = 0

    t_dice_l = 0

    for i in range(0, int(TESTING_DATA_SIZE/TESTING_BATCH_SIZE)):
        test_data = data[0:2, TRAIN_DATA_SIZE+VALIDATION_DATA_SIZE+i*TESTING_BATCH_SIZE:TRAIN_DATA_SIZE+VALIDATION_DATA_SIZE+(i+1)*TESTING_BATCH_SIZE, :, :, :]
        input_tensor = test_data[0].to(device)
        label_tensor = test_data[1].to(device)
        label_tensor = torch.min(torch.ceil(label_tensor.float()), torch.ones(label_tensor.size()).to(device)).to(device).view(TESTING_BATCH_SIZE, 1, DATA_DIMENSIONS[0], DATA_DIMENSIONS[1], DATA_DIMENSIONS[2])
        output_tensor = model(input_tensor.float())
        test_loss = pixel_BCE(output_tensor, label_tensor.float())
        dice_l = dice_loss(output_tensor, label_tensor.float())
        t_test_loss += test_loss.item()
        t_dice_l += dice_l.item()
    t_dice_l /= TESTING_DATA_SIZE/TESTING_BATCH_SIZE
    print("")
    print("Testing Loss : "+str(t_test_loss/(TESTING_DATA_SIZE/TESTING_BATCH_SIZE))+"   Dice Loss : "+str(t_dice_l))
    f.write("Testing Loss : "+str(t_test_loss/(TESTING_DATA_SIZE/TESTING_BATCH_SIZE))+"   Dice Loss : "+str(t_dice_l)+"   nf : "+str(nf))
    f.close()

    seconds = math.floor((after_time-before_time)/1000)
    minutes = math.floor(seconds/60)
    seconds = seconds % 60
    
    print(str(NUM_EPOCHS)+" epochs took "+str(minutes)+" minute(s) "+str(seconds)+" second(s).")
    
    return model
    

if __name__ == "__main__":
    print("Start!")
    current_milli_time = lambda: int(round(time.time() * 1000))
    before_time = current_milli_time()

    files = os.listdir(".")
    m = [int(f[9:]) for f in files if len(f) > 9 and f[0:9] == "unettrial"]
    if len(m) > 0:
        folder = "unettrial" + str(max(m) + 1)
    else:
        folder = "unettrial1"
    os.mkdir(folder)

    print("Created session folder " + folder)

    print("Loading data...")
    data = torch.load("TRIMMED64.pt").clamp(0, 1)[0:2].detach()
    # data = data.permute(1, 0, 2, 3, 4).clamp(0, 1).detach()
    # #data[1, :, :, :, :] = torch.round(data[1, :, :, :, :])
    #
    # # good_wgan_images = torch.LongTensor([0, 1, 4, 7, 8, 9, 10, 12, 13, 14, 18, 23, 24, 25, 26, 27, 33, 37, 45, 48, 49, 52, 53, 54, 62, 63, 65, 68, 69, 70, 71, 78, 78, 79, 80, 81, 84, 85, 86, 89, 90, 91, 92, 94, 95, 96, 97, 99, 100, 101, 102, 103, 107, 108, 111, 112, 114, 117, 130, 139])
    wgan_data = torch.load("WGAN_5000_DATA.pt")
    # # data = torch.load("TRIMMED64.pt")
    wgan_data = wgan_data.permute(1, 0, 2, 3, 4).clamp(0, 1).detach()
    wgan_data[1, :, :, :, :] = torch.round(wgan_data[1, :, :, :, :])
    # # wgan_data = wgan_data[:, good_wgan_images, :, :, :]
    # # data = data[0:2, 60:335]
    data = torch.cat((wgan_data[:, :, :, :, :], data), dim=1)
    #
    # data = torch.cat((wgan_data, data[:, 300:335, :, :, :]), dim=1)

    print(data.size())

    after_time = current_milli_time()
    seconds = math.floor((after_time-before_time)/1000)
    minutes = math.floor(seconds/60)
    seconds = seconds % 60
    print("Data loading took "+str(minutes)+" minute(s) "+str(seconds)+" second(s).")  
    
    model = train_model(data)
    
    '''
    
    1. Tweak BCE coefficient
    2. Dropout
    3. Cutoff at certain epoch
    4. Record data
    
    
    
    
    
    
    
    '''