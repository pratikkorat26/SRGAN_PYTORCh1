import torch
from torch.utils.data import Dataset
import torchvision
from skimage import io

class SRGANDATA(Dataset):
    '''
    text_file is a list of path of images
    '''
    def __init__(self , txt_file):
        self.text_file = txt_file
        self.transform = torchvision.transforms.ToTensor()

        self.lr_transform = torchvision.transforms.Resize(size = (64 , 64))
        self.hr_transform = torchvision.transforms.Resize(size=(256, 256))

    def __len__(self):
        return len(self.text_file)

    def __getitem__(self, item):
        img = torchvision.io.read_image(self.text_file[item])

        hr_img = self.hr_transform(img)
        lr_img = self.lr_transform(img)

        # hr_img = self.transform(hr_img)
        # lr_img = self.transform(lr_img)

        return hr_img , lr_img


if __name__ == '__main__':
    from glob import glob

    data = glob(pathname = "dataset/*.jpg")
    dataset = SRGANDATA(data)
    print(dataset)

    exmp = dataset.__getitem__(11)

    print(exmp[0].shape , exmp[1].shape)

