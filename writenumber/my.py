

import torch
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
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
cnn=CNN()
cnn.load_state_dict(torch.load("model/mnist_model_state.pth"))
cnn.eval()
print("模型加载成功")
def process_image(image_path):
    transform=transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28,28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    image=Image.open(image_path)
    image=transform(image)
    debug_img=image.clone()
    debug_img=debug_img*0.3081+0.1307
    debug_img=debug_img.squeeze().numpy()
    plt.imshow(debug_img,cmap='gray')
    plt.title("What model sees (after normalization)")
    plt.show()
    image=image.unsqueeze(0)
    return image
image_path="./mnist/seven.png"
input_data=process_image(image_path)
with torch.no_grad():
    output=cnn(input_data)
    prediction=torch.argmax(output,dim=1).item()
    probabilities=torch.softmax(output,dim=1)[0]*100
    print(f"\n预测结果：{prediction}")
    print(f"置信度{probabilities[prediction]}")