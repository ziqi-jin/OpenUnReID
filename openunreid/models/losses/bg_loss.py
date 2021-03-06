import torch
import torch.nn as nn
import torch.nn.functional as F


class BG_loss(nn.Module):


    def __init__(self):
        super(BG_loss, self).__init__()
        # L1 or L2 
        self.loss = nn.L1Loss()
    def forward(self, real_imgs, fake_imgs, masks):
        real_imgs_ = real_imgs.clone()
        fake_imgs_ = fake_imgs.clone()
        for index in range(len(real_imgs)):
            real_imgs_[index] = masks[index]*real_imgs[index]
            fake_imgs_[index] = masks[index]*fake_imgs[index]
        loss = self.loss(real_imgs_,fake_imgs_)
        return loss
