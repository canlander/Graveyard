# -*- coding: utf-8 -*-
"""
Adversarial Example Generation
==============================

**Author：** ‘Tong Jinyang canlander@outlook.com'

FGSM's reproduction

I will code following tips below
"""
# FGSM对抗样本生成方法：计算损失函数对于输入的梯度，将其乘以一个系数作为扰动值，加入进原输入中形成对抗样本

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# 设置实验系数值，初始化模型参数，确定系统GPU设置
epsilons = [0, 0.15, 0.2, 0.3, 0.35, 0.5]
model_path = "lenet_mnist_model.pth.pt"
use_cuda = True

device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
print(f'Using {device} device')
torch.manual_seed(40)   # 设置随机数种子，确保实验可重复性

# 定义 Model 和 DataLoader，然后初始化 Model 并加载预训练权重，加载测试集，定义损失函数等等
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

model = Net().to(device)
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.eval()

# # 定义数据转换
# transform = transforms.Compose([
#     transforms.ToTensor(),  # 将图像转换为PyTorch张量
#     transforms.Normalize((0.1307,), (0.3081,))  # 归一化处理（相对于MNIST数据集的标准参数）
# ])
#
# # 加载MNIST测试数据集
# test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 创建DataLoader
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            ])),
        batch_size=1, shuffle=True)

# 定义FGSM攻击函数
def fgsm_attack(image, epsilon, data_grad):
    # 收集数据梯度
    sign_data_grad = data_grad.sign()
    # 通过调整输入图像来更新对抗性样本
    perturbed_image = image + epsilon * sign_data_grad
    # 添加剪切以维持[0,1]范围
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image  # 缺少归一化

def denorm(batch, mean=[0.1307], std=[0.3081]):
    """
    Convert a batch of tensors to their original scale.

    Args:
        batch (torch.Tensor): Batch of normalized tensors.
        mean (torch.Tensor or list): Mean used for normalization.
        std (torch.Tensor or list): Standard deviation used for normalization.

    Returns:
        torch.Tensor: batch of tensors without normalization applied to them.
    """
    if isinstance(mean, list):
        mean = torch.tensor(mean).to(device)
    if isinstance(std, list):
        std = torch.tensor(std).to(device)

    return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)

# 定义攻击测试函数
def test_fgsm(model, device, test_loader, epsilon):
    correct = 0
    adv_examples = []  # 存储对抗性样本

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data.requires_grad = True  # 需要梯度，以便计算损失对输入的梯度

        # 前向传播
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]  # 获取初始预测结果

        if init_pred.item() != target.item():
            continue

        # 计算损失
        loss = F.nll_loss(output, target)
        model.zero_grad()  # 清除之前的梯度
        loss.backward()  # 反向传播，计算梯度

        # 生成对抗性样本
        data_grad = data.grad.data
        perturbed_data_normalize = denorm(data)
        perturbed_data = fgsm_attack(perturbed_data_normalize, epsilon, data_grad)
        perturbed_data_normalized = transforms.Normalize((0.1307,), (0.3081,))(perturbed_data)

        # 再次前向传播，获取对抗性样本的预测结果
        output = model(perturbed_data_normalized)

        final_pred = output.max(1, keepdim=True)[1]  # 获取对抗性样本的预测结果

        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if epsilon == 0 and len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))

    final_acc = correct / float(len(test_loader))
    print(f"Epsilon: {epsilon}\tTest Accuracy = {correct} / {len(test_loader)} = {final_acc}")

    return final_acc, adv_examples

# 运行攻击进行实验
accuracies = []
examples = []

# Run test for each epsilon
for eps in epsilons:
    acc, ex = test_fgsm(model, device, test_loader, eps)
    accuracies.append(acc)
    examples.append(ex)

# 统计实验结果，作准确率与参数值的关系图表
cnt = 0
plt.figure(figsize=(8,10))
for i in range(len(epsilons)):
    for j in range(len(examples[i])):
        cnt += 1
        plt.subplot(len(epsilons),len(examples[0]),cnt)
        plt.xticks([], [])
        plt.yticks([], [])
        if j == 0:
            plt.ylabel(f"Eps: {epsilons[i]}", fontsize=14)
        orig,adv,ex = examples[i][j]
        plt.title(f"{orig} -> {adv}")
        plt.imshow(ex, cmap="gray")
plt.tight_layout()
plt.show()

# 显示对抗攻击图像