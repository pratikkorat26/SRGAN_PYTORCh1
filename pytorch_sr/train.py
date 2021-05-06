import torch
import torchvision
from models import Generator , Discriminator
from utils import SRGAN_loss
from dataset import SRGANDATA
from glob import glob
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
def train(args):
    """
    args : neccesary agrument neede for the project
    """
    # Model Loading

    # 1) Generator
    generator = Generator(res_block = 8)
    gen_optim = torch.optim.AdamW(generator.parameters() , lr = 1e-4)
    gen_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(gen_optim , min_lr = 2e-6 , patience = 2)
    print("generator built successfully")

    #  Discriminator
    discriminator = Discriminator(in_channels = 3)
    disc_optim = torch.optim.AdamW(discriminator.parameters(), lr=1e-4)
    disc_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(disc_optim, min_lr=2e-6, patience=2)
    print("discriminator built successfully")

    # Loss defining
    loss_fn = SRGAN_loss(alpha = 1 , beta = 1e-3)
    print("loss function loaded")

    # Dataset loading
    files = glob("dataset/*.jpg")

    dataset = SRGANDATA(txt_file = files)
    dataloader = DataLoader(dataset,
                            batch_size = 4,
                            shuffle = True)
    # Data loading done and model building also done
    total_batch = int(len(dataset)/4)
    print("dataset loaded successfully")

    hr_img , lr_img = next(iter(dataloader))

    # Setting up summary writer
    summarywriter = SummaryWriter()
    summarywriter.add_graph(generator , lr_img)
    #summarywriter.add_graph(discriminator , hr_img)
    # Done

    print("trainint start")
    global_var = 0
    for epoch in range(args.epoch):
        loss = 0
        for batch_idx , (hr_img , lr_img) in enumerate(dataloader):
            # fake image generation from lower resolution images
            gen_img = generator(lr_img)
            true_img = hr_img

            # logit generation
            true_logit = discriminator(true_img)
            false_logit = discriminator(gen_img)

            total_loss = loss_fn(true_img , gen_img , true_logit , false_logit)
            loss += total_loss

            total_loss.backward(retain_graph = True)
            gen_optim.step()
            disc_optim.step()
            gen_scheduler.step(total_loss)
            disc_scheduler.step(total_loss)
            summarywriter.add_scalar("batch loss/train", total_loss, global_step=global_var)
            global_var += 1

            print(f"batch_idx:[{batch_idx}/{total_batch}]  Loss value : {total_loss}")

        true_img_grid = torchvision.utils.make_grid(hr_img)
        gen_img_grid = torchvision.utils.make_grid(gen_img)
        lr_img_grid = torchvision.utils.make_grid(lr_img)

        def plot_image(image, title=""):
            """
              Plots images from image tensors.
              Args:
                image: 3D image tensor. [height, width, channels].
                title: Title to display in the plot.
            """

            image = torch.clip_(image, 0, 255)

            image = transforms.ToPILImage()(image).convert("RGB")
            plt.imshow(image)
            plt.axis("off")
            plt.title(title)
            plt.show()

        plot_image(true_img_grid , "true_image")
        plot_image(gen_img_grid , "geneated img")
        plot_image(lr_img_grid , "low resolution img")
        # adding values to summary writer
        summarywriter.add_scalar("Total Loss/train" , loss , global_step = epoch)
        # torchvision.utils.save_image(true_img_grid, f"gen/img_high{epoch}.jpg")
        # torchvision.utils.save_image(gen_img_grid, f"gen/img_gen{epoch}.jpg")
        # torchvision.utils.save_image(lr_img_grid, f"gen/img_lr{epoch}.jpg")

    torch.save(generator , f = "generator.pth")


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument("--epoch" , type=int , help = "add number of epochs for training")
    args.add_argument("--log_dir", type=str, help="please enter path for logging directory")

    args = args.parse_args()
    print(args)

    train(args)








