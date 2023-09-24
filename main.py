from model import Unet
# from torchsummary import summary
import torch
import torch.nn as nn
from utils import MakeData
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import torchvision.transforms.functional as TF
import cv2
import matplotlib.pyplot as plt



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'


print(device)

def training(model, data, num_epochs = 1):
    
    criterian = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    loop = tqdm(data)

    for epoch in range(num_epochs):
        for i, (x,y) in enumerate(loop):
            x = x.to(device)
            y = y.to(device)

            pred = model(x)

            loss = criterian(pred, y)
            loss.backward()

            optimizer.step()

            optimizer.zero_grad()

            loop.set_postfix(loss = loss.item())

    

def main():

    train_path = "data/train/"
    val_path = "data/val/"
    try_path = "data/try/"

    # x_train, y_train = get_data(train_path)

    train_data = MakeData(train_path)

    train_loader = DataLoader(train_data, batch_size=8, shuffle=True)

    model = Unet(in_challen=3, out_channel=3).to(device)
    # model = UNet(3,3).to(device)
    
    # training(model, data=train_loader)

    criterian = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    num_epochs = 30

    

    for epoch in range(num_epochs):
        loop = tqdm(train_loader)
        for i, (x,y) in enumerate(loop):
            x = x.to(device)
            y = y.to(device)

            pred = model(x)

            loss = criterian(pred, y)
            loss.backward()

            optimizer.step()

            optimizer.zero_grad()

            loop.set_postfix(loss = loss.item())

    
    # model_path = 'checkpoint/trained-Unet.pt'
    model_path = f'checkpoint/trained-Unet{num_epochs}.pt'
    torch.save(model.state_dict(), model_path)



def test():
    # x = torch.randn((3,160,160))
    # model = Unet(3, 1).to(device)
    # pred = model(x)
    # # print(summary(model, (1,160,160), device=str(device).lower()))
    # print(x.shape)
    # print(pred.shape)
    # assert x.shape == pred.shape
    # pass
    model1 = Unet(3,3)
    model_test = torch.load('checkpoint/trained-Unet30.pt')
    model1.load_state_dict(model_test)
    model1.eval()


    img_dir = 'data/val/'
    images_test = '2.jpg'
    img1 = cv2.imread(img_dir+images_test)

    x_test = img1[0:256, 0:256]
    y_test = img1[0:256, 256:512]

    x_test_tensor = TF.to_tensor(x_test)
    y_test_tensor = TF.to_tensor(y_test)

    pred_model1 = model1(x_test_tensor.unsqueeze(0))

    pred_print = torch.reshape(pred_model1, (3,256,256))

    pred_print = pred_print.permute(1, 2, 0).detach().numpy()

    plt.imshow(pred_print)
    plt.savefig(f'output/val_{images_test}')
    plt.show()


if __name__ == "__main__":
    test()
