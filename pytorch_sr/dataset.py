from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch
import matplotlib.pyplot as plt

class SRGANDATA(Dataset):
    '''
    input : text_file is a list of path of images
    output : low resolution image patch , and high resolution image patch
    '''
    def __init__(self , txt_file):
        self.text_file = txt_file
        self.transform = transforms.Compose([
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
    ])

    def __len__(self):
        return len(self.text_file)

    def __getitem__(self, item):
        img = Image.open(self.text_file[item])

        hr_img = img.resize((256 , 256))
        lr_img = img.resize((64 , 64))

        # hr_img.show()
        # lr_img.show()

        #We can do transform if wanna do but that's not neccesary
        hr_img = self.transform(hr_img)
        lr_img = self.transform(lr_img)

        return hr_img , lr_img ,


if __name__ == '__main__':
    from glob import glob
    # from torchvision.models import vgg19
    # from models import Generator , Discriminator
    #
    # gen = Generator()
    # disc = Discriminator(in_channels = 3)
    #
    data = glob(pathname = "dataset/*.jpg")
    dataset = SRGANDATA(data)
    #
    # dataloader = DataLoader(dataset , batch_size = 4 , shuffle = True)
    #
    exmp = next(iter(dataset))
    #
    # fake_gen = gen(exmp[1])
    # true_gen = exmp[0]
    #
    # true_logit = disc(true_gen)
    # false_logit = disc(fake_gen)
    # from utils import SRGAN_loss
    #
    # loss = SRGAN_loss()
    #
    # final_loss = loss(true_gen , fake_gen , true_logit , false_logit)
    # print(f"final_loss : {final_loss}")
    # print(fake_gen.shape , true_logit , false_logit)
    #
    from torchvision import transforms
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

    plot_image(exmp[0] , "hr image")


