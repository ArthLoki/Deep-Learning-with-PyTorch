from torchvision import models, transforms
from torchvision.models import ResNet101_Weights
from PIL import Image
import torch

alex_net = models.AlexNet()

'''
resnet = models.resnet101(pretrained=True)

Warning:
1. The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, 
please use 'weights' instead

2. Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in 
the future. The current behavior is equivalent to passing `weights=ResNet101_Weights.IMAGENET1K_V1`. You can also use 
`weights=ResNet101_Weights.DEFAULT` to get the most up-to-date weights.
'''

# The argument instructs the function to download the weights of resnet101 trained on the ImageNet dataset
resnet = models.resnet101(pretrained=True)  # Model is 94,8% certain that it's a golden retriever
# resnet = models.resnet101(weights=ResNet101_Weights.DEFAULT)  # Model is 69,04% certain that it's a golden retriever

# we have to preprocess the input images, so they are the right size and so that their values (colors) sit roughly
# in the same numerical range.
# In that case, we use the transformers which defines pipelines of basic preprocessing functions
preprocess = transforms.Compose([
    transforms.Resize(256),  # Resizes the image scaling the input image to 256x256
    transforms.CenterCrop(224),  # Crops the image to 224x224 around the center
    transforms.ToTensor(),  # transforms it into a tensor
    transforms.Normalize(  # Normalize its RGB components so that they have defines means and standards deviations
        mean=[0.485, 0.456, 0],
        std=[0.229, 0.224, 0.225]
    )
])

# Loading an image from the local filesystem using Pillow
img = Image.open("images/bobby.jpg")
# img.show()

# Passing image through our preprocessing pipeline
img_t = preprocess(img)

# we can reshape, crop, and normalize the input tensor in a way that the network
# expects
batch_t = torch.unsqueeze(img_t, 0)

# We can reshape, crop, and normalize the input tensor in a way that the network
# expects. In order to do inference, we need to put the network in eval mode
resnet_eval = resnet.eval()

# If we forget to do that, some pretrained models, like batch normalization and dropout,
# will not produce meaningful answers
out = resnet_eval(batch_t)  # Returns a tensor of 1000 scores

# load the file containing the 1000 labels for the ImageNet dataset classes
with open('imagenet_classes.txt') as f:
    labels = [line.strip() for line in f.readlines()]

# Using the max function in PyTorch, which outputs the maximum value in a tensor as well as the indices where
# that maximum value occurred, we need to determine the index corresponding to the maximum score in the out tensor
# we obtained previously
_, index = torch.max(out, 1)

# Using index to access the label. Index is a one-dimensional tensor, not a number, so we
# need to get the actual numerical value to use as an index into our labels list using index[0]
# Using torch.nn.functional.softmax to normalize our outputs to the range [0, 1], and divide by the sum
# something
# That gives us something roughly akin to the confidence that the model has in its prediction
percentage = torch.nn.functional.softmax(out, 1)[0] * 100
# print(labels[index[0]], percentage[index[0]].item())

# we can use the sort function, which sorts the values
# in ascending or descending order and also provides the indices of the sorted values in the original array
_, indices = torch.sort(out, descending=True)
# for el in [(labels[idx], percentage[idx].item()) for idx in indices[0][:5]]:
#     print(el)




