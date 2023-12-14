from torchvision import models, transforms
from PIL import Image
import torch
import torch.nn as nn
from cyclegan import ResNetGenerator
from getPath import get_base_path

class Chapter3:
    def __init__(self, tensor_dim=0):
        self.tensor_dim = tensor_dim
        self.base_path = get_base_path()
        pass

    def firstTensor(self):
        points = torch.tensor([[4.0, 1.0], [5.0, 3.0],[2.0, 1.0]])
        print(points)

        # Slice
        # print(points[1:])
        # print(points[1:, :])
        # print(points[1:, 0])
        # print(points[None])
        return

    def img2GreyScale(self):
        resnet = models.resnet101(pretrained=True)
        preprocess = transforms.Compose([
            transforms.Resize(256),  # Resizes the image scaling the input image to 256x256
            transforms.CenterCrop(224),  # Crops the image to 224x224 around the center
            transforms.ToTensor(),  # transforms it into a tensor
            transforms.Normalize(  # Normalize its RGB components so that they have defines
                # means and standards deviations
                mean=[0.485, 0.456, 0],
                std=[0.229, 0.224, 0.225]
            )
        ])
        img = Image.open(f"{self.base_path}/dlwpt-code/data/p1ch2/bobby.jpg")

        # we want to converto the image to grey scale
        img_t = torch.randn(3, 5, 5)  # shape [channel, rows, columns]
        weigths = torch.tensor([0.2126, 0.7152, 0.0722])
        batch_t = torch.randn(2, 3, 5, 5)  # shape [batch, channel, rows, columns]

        img_gray_naive = img_t.mean(-3)
        batch_gray_naive = batch_t.mean(-3)
        print(img_gray_naive.shape, batch_gray_naive.shape)
        return

def main():
    chapter3 = Chapter3()

    # chapter3.firstTensor()
    chapter3.img2GreyScale()

    return


if __name__ == "__main__":
    main()
