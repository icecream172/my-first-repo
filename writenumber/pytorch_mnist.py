import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.utils.data as data_utils # 内含分批加载数据的类
from CNN import CNN
# 数据加载   加载mnist数据集
# 训练数据
train_data = dsets.MNIST(
    root="mnist",
    train=True,
    transform=transforms.ToTensor(),
    download=True
);
#测试数据
test_data = dsets.MNIST(
    root="mnist",
    train=False,
    transform=transforms.ToTensor(),
    download=True
);
# 分批加载数据
train_loader = data_utils.DataLoader(dataset=train_data,
                                     batch_size=64,
                                     shuffle=True
                                     )
test_loader = data_utils.DataLoader(dataset=test_data,
                                     batch_size=64,
                                     shuffle=True
                                     )
# print(train_loader)
# print(test_loader)
cnn = CNN()
# 损失函数：交叉熵
loss_func=torch.nn.CrossEntropyLoss()

# 优化函数
optimizer=torch.optim.Adam(cnn.parameters(),lr=0.01)
# epoch 通常指一次训练数据全部训练一遍
for epoch in range(10):

# 用enumerate函数访问数据
    for index, (images, labels) in enumerate(train_loader):
        # print(index)
        # print(value)
        # 前向传播
        outputs = cnn(images)
        # 传入输出层节点和真实标签来计算损失函数
        loss = loss_func(outputs, labels)
        # 清空梯度
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        optimizer.step()
        print("当前为第{}轮，当前批次为{}/{},loss为{}".format(epoch + 1, index+1, len(train_data) // 64, loss.item()))

    # 测试集验证
    loss_test=0
    rightValue=0
    for index2, (images, labels) in enumerate(test_loader):
        outputs = cnn(images)
        # print(outputs)
        # print(labels)
        loss_test+=loss_func(outputs, labels)
        _,pred=outputs.max(1)
        rightValue+=(pred==labels).sum().item()
        print("当前为第{}轮测试集验证，当前批次为{}/{},loss为{}，准确率是{}".format(epoch + 1, index2+1, len(train_data) // 64, loss_test,rightValue/len(test_data)))
torch.save(cnn.state_dict(),"model/mnist_model_state.pth")