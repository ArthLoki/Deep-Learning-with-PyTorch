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
        # Loading image from local filesystem
        # img = Image.open(f"{self.base_path}/dlwpt-code/data/p1ch2/bobby.jpg")

        # we want to converto the image to grey scale
        img_t = torch.randn(3, 5, 5)  # shape [channel, rows, columns]
        weights = torch.tensor([0.2126, 0.7152, 0.0722])

        # want our code to generalize—for example, from grayscale images represented
        # as 2D tensors with height and width dimensions to color images adding a third
        # channel dimension (as in RGB), or from a single image to a batch of images
        batch_t = torch.randn(2, 3, 5, 5)  # shape [batch, channel, rows, columns]

        # sometimes the RGB channels are in dimension 0, and sometimes they are in dimension
        # 1. But we can generalize by counting from the end: they are always in dimension
        # –3, the third from the end
        img_gray_naive = img_t.mean(-3)
        batch_gray_naive = batch_t.mean(-3)
        # print(img_gray_naive.shape, batch_gray_naive.shape)

        # multiply things that are the
        # same shape, as well as shapes where one operand is of size 1 in a given dimension
        # It also appends leading dimensions of size 1 automatically. It's called broadcasting
        unsqueezed_weights = weights.unsqueeze(-1).unsqueeze_(-1)

        # batch_t of shape (2, 3, 5, 5) is multiplied by unsqueezed_weights of shape (3,
        # 1, 1), resulting in a tensor of shape (2, 3, 5, 5), from which we can then sum the third
        # dimension from the end (the three channels)
        img_weights = (img_t * unsqueezed_weights)
        batch_weights = (batch_t * unsqueezed_weights)

        # img_grey_weighted = img_weights.sum(-3)
        # batch_grey_weighted = batch_weights.sum(-3)
        # print(batch_weights.shape, batch_t.shape, unsqueezed_weights.shape)

        # the PyTorch function
        # einsum (adapted from NumPy) specifies an indexing mini-language giving index
        # names to dimensions for sums of such products
        img_gray_weighted_fancy = torch.einsum('...chw,c->...hw', img_t, weights)
        batch_gray_weighted_fancy = torch.einsum('...chw,c->...hw', batch_t, weights)
        # print(batch_gray_weighted_fancy.shape)

        weights_named = torch.tensor(data=[0.2126, 0.7152, 0.0722], names=['channels'])
        # print(weights_named)

        img_named = img_t.refine_names(..., 'channels', 'rows', 'columns')
        batch_named = batch_t.refine_names(..., 'channels', 'rows', 'columns')
        print("img named:", img_named.shape, img_named.names)
        print("batch named:", batch_named.shape, batch_named.names)

        # two inputs, in addition to the usual dimension checks—whether
        # sizes are the same, or if one is 1 and can be broadcast to the other
        weights_aligned = weights_named.align_as(img_named)
        gray_named = (img_named * weights_aligned).sum('channels')
        return


def main():
    chapter3 = Chapter3()

    print(torch.cuda.is_available())

    # chapter3.firstTensor()
    chapter3.img2GreyScale()

    return


if __name__ == "__main__":
    main()
