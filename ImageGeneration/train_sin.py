import numpy as np
from model import UNetModel
#from u_vit_model import UViT
#from u_vit_numclass import UViT
import torch as th
from PIL import Image
import torchvision as tv
from torch.optim import AdamW
import random
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':

    input = tv.transforms.ToTensor()(Image.open("data/img_128.png").convert('RGB'))[None]
    #adjust_scales2image_UViT(input)
    # print(real.shape)

    #logger.configure()
    # writer = SummaryWriter("logs")
    # writer.add_images("UViT", real)
    # print(real.shape)
#扩散模型参数设置
    betas=th.linspace(0.0001,0.02,1000).to("cpu")   #设置 betas参数，线性增长
    alphas = 1 - betas
    alphas_cumprod = th.cumprod(alphas, dim=0)  # 返回alphas连乘  [a0, a0*a1, a0*a1*a2, ..., a0*a1*...*at]
#损失函数
    loss_fn = th.nn.MSELoss()
    #loss_fn = th.nn.L1Loss()
#模型加载
    #model=UViT()
    model=UNetModel(img_size=128,in_channels=3,model_channels=128,out_channels=3)
    model.to("cpu")
    model.train()
#优化器
    opt = AdamW(model.parameters() , lr=0.0001, weight_decay=0.0)

#利用模型训练
    for epoch in range(10001):
        x = input
        #print(input.size())
        #梯度清零
        opt.zero_grad()
    #改变宽高
        curr_h = round(input.shape[2] * random.uniform(0.75, 1.25))     #input.shape[2]对应 H
        curr_w = round(input.shape[3] * random.uniform(0.75, 1.25))     #对应 W
        curr_h, curr_w = 8 * (curr_h // 8), 8 * (curr_w // 8)       #后面要下采样3次，防止H，w中出现除不尽的情况
        x = F.interpolate( x, (curr_h, curr_w), mode="bicubic")     #改宽高


    # 在0 和 1000 中随机产生一个t用来训练
        t = th.randint(0, 1000,[1]).to("cpu")   #比如 tensor([545])
        #print(t)

        alphas_bar = alphas_cumprod.gather(-1, t)
        noise = th.randn_like(x)   #随机采样添加的噪声
        x_t = th.sqrt(alphas_bar) * x + th.sqrt(1 - alphas_bar) * noise  # 直接由X0计算出Xt
    #过模型
        pred_noise = model(x_t, t)
    #计算损失
        loss = loss_fn(pred_noise, noise)
        # writer = SummaryWriter("logs")
        # writer.add_scalar("loss",loss,epoch)
    #回传
        loss.backward()
        opt.step()
        print(f"Epoch: {epoch},  Loss: {loss.item()}")
        if epoch % 5000 == 0 :  # 每10轮记一次模型
            th.save(model.state_dict(), f"./saved_models/model{str(epoch)}.pth")


