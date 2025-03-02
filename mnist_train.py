from model import *
import torch
from galore_torch import GaLoreAdamW8bit
from torch import nn
import torchvision
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter # type: ignore

# 准备训练集
dtst_tf = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])   # 把图片转化为tensor

train_set = torchvision.datasets.MNIST('',train=True,transform=dtst_tf,download=False)
test_set = torchvision.datasets.MNIST('',train=False,transform=dtst_tf,download=False)

train_dataloader = DataLoader(train_set, batch_size=128, drop_last=True, shuffle=True)
test_dataloader = DataLoader(test_set, batch_size=128, drop_last=True, shuffle=False)

model = Mix_ViT(
    d=32,
    dk=16,
    head_num=3,
    encoder_num=6,
    img_size=28,
    patch_size=7,
    in_chans=1,
    class_num=10,
    use_dropout=True,
    device='cuda'
)   # 创建模型

rating = 0.02
# opz = GaLoreAdamW8bit(model.parameters(), lr=rating)
opz = torch.optim.AdamW(model.parameters(), lr=rating)
# 优化器

loss_fn = CrossEntropyLoss()
# 损失函数

train_step = 0
# 记录训练次数

test_step = 0
# 记录测试次数

epoch = 10
# 训练轮数

writer = SummaryWriter('logs\\mnist')
# 日志记录器

for i in range(epoch):
    print(f'epoch:{i + 1}')
    for x in train_dataloader:

        imgs, targets = x
        imgs = imgs.to('cuda')
        targets = targets.to('cuda')

        output = model(imgs)
        loss = loss_fn(output, targets)
        # 计算损失

        opz.zero_grad()
        loss.backward()
        opz.step()
        # 优化器优化模型

        train_step += 1
        writer.add_scalar('train.loss2', loss.item(), train_step)

        if train_step % 100 == 0:
            print(f'train_step:{train_step},loss:{loss}')

        total_loss = 0
        total_acry = 0
        # 测试模型

        with torch.no_grad():   # 禁用梯度计算
            for m in test_dataloader:
                imgs, targets = m
                imgs = imgs.to('cuda')
                targets = targets.to('cuda')

                output = model(imgs)
                loss = loss_fn(output, targets)
                total_loss += loss
                acry = (output.argmax(1) == targets).sum()

                total_acry += acry   # 累加成功次数

        writer.add_scalar('test.loss2', total_loss, test_step)
        writer.add_scalar('test.acry2', total_acry/len(test_set), test_step)

        test_step += 1

        if test_step % 100 == 0:
            print(f'test_loss:{total_loss}')
            print(f'accuracy_rate:{total_acry/len(test_set)}')
            torch.save(model, 'vit_transformer+.pth')
