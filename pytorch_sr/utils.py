import torch
import torch.nn as nn
from torchvision.models import alexnet

class SRGAN_loss(nn.Module):
    def __init__(self , alpha = 1 , beta = 1e-3):
        super(SRGAN_loss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.vgg = alexnet(pretrained = False).features
        self.vgg.eval()
        self.mse_loss = nn.MSELoss(reduction = "sum")
        self.cross_entropy_loss = nn.BCELoss(reduction = "sum")

    def forward(self , hr_img , lr_img , logit_real , logit_fake):

        content_loss = self.content_loss(hr_img , lr_img)
        adverserial_loss = self.adverserial_loss(logit_real , logit_fake)

        final_loss = self.alpha*content_loss + self.beta*adverserial_loss

        return final_loss


    def content_loss(self , hr_img , gen_lr_to_hr_img):
        hr_features = self.vgg(hr_img)
        lr_gen_features = self.vgg(gen_lr_to_hr_img)

        mse_loss = self.mse_loss(hr_features , lr_gen_features)

        return mse_loss


    def adverserial_loss(self , logit_real , logit_fake):
        true_logit_real = torch.ones_like(logit_real)
        false_logit_fake = torch.zeros_like(logit_fake)

        real_loss = self.cross_entropy_loss(logit_real, true_logit_real)
        fake_loss = self.cross_entropy_loss(logit_fake, false_logit_fake)

        total_loss = real_loss + fake_loss

        return total_loss



if __name__ == '__main__':
    model = alexnet(True)
    print(model)

