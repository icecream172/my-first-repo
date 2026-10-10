import torch
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv=torch.nn.Sequential(
            # 卷积操作卷积层
            torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3,padding=1),
            # 归一化BN层
            torch.nn.BatchNorm2d(32),
            # 激活层 Relu函数
            torch.nn.ReLU(),
            # 最大池化
            torch.nn.MaxPool2d(2)
        );
        self.fc=torch.nn.Linear(in_features=14*14*32,
                                out_features=10)
    def forward(self,x):
        out=self.conv(x)
        # 将图像展开为一维
        out=out.view(out.size(0),-1)
        out=self.fc(out)
        return out