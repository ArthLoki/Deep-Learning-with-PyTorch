import torch
from getPath import get_base_path
from cpu2cuda import print_device_timer, timer_data, get_cuda_device_data
# from cpu2cuda import get_device, get_device_name, get_device_data, check_device_change, dev2dev

class Chapter3:
    def __init__(self, tensor_dim=0):
        self.tensor_dim = tensor_dim
        self.base_path = get_base_path()
        self.dev_code = 1  # Default: Running on CPU. OBS: code == 0 => running on GPU/cuda
        pass

    def firstTensor(self):
        points = torch.tensor([[4.0, 1.0], [5.0, 3.0],[2.0, 1.0]])
        # print(points)

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
        # print("img named:", img_named.shape, img_named.names)
        # print("batch named:", batch_named.shape, batch_named.names)

        # two inputs, in addition to the usual dimension checks—whether
        # sizes are the same, or if one is 1 and can be broadcast to the other
        # The method align_as returns a tensor with missing dimensions
        # added and existing ones permuted to the right order
        weights_aligned = weights_named.align_as(img_named)
        # print(weights_aligned.shape, weights_aligned.names)

        # Functions accepting dimension arguments, like sum, also take named dimensions
        gray_named = (img_named * weights_aligned).sum('channels')
        # print(gray_named.shape, gray_named.names)

        gray_plain = gray_named.rename(None)
        # print(gray_plain.shape, gray_plain.names)
        return

    def tensorDataTypes(self):
        double_points = torch.ones(10, 2, dtype=torch.double)
        short_points = torch.tensor([[1, 2], [3, 4]], dtype=torch.short)
        # print(short_points.dtype, double_points)

        double_points_cast = torch.zeros(10, 2).double()
        short_points_cast = torch.ones(10, 2).short()

        double_points_toMethod = torch.ones(10, 2).to(torch.double)
        short_points_toMethod = torch.ones(10, 2).to(dtype=torch.short)

        points_64 = torch.rand(5, dtype=torch.double)
        points_short = points_64.to(dtype=torch.short)
        # print(points_64 * points_short)  # works from PyTorch 1.3 onwards
        return

    def tensorAPI(self):
        # transpose (example 1)
        a = torch.ones(3, 2)
        a_t1 = torch.transpose(a, 0, 1)
        # print(a.stride(), a_t1.stride())
        # print(f'Transpose of tensor: {a.shape}  ---------->  {a_t1.shape}\n')

        # transpose (example 2)
        a_t2 = a.transpose(0, 1)
        # print(a_t2.stride())
        # print(f'Transpose of tensor: {a.shape}  ---------->  {a_t2.shape}')

        # Transposing without copying: using t function, a shorthand alternative to transpose function for 2D tensors
        points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])

        points_t = points.t()

        '''
        Stride is the jump necessary to go from one element to the next one in the specified dimension dim. 
        A tuple of all strides is returned when no argument is passed in. Otherwise, an integer value is returned 
        as the stride in the particular dimension dim
        '''

        # print(points.is_contiguous())
        # print(points_t.is_contiguous())

#         print(f'\n{points}\n')
#         print(f'\n{points_t}\n')
#         print(f'\n{id(points.storage()) == id(points_t.storage())}\n')
#
        print(f'''\npoints:\n\t-> stride: {points.stride()}\n\t-> shape: {points.shape}\n
points_t:\n\t-> stride: {points_t.stride()}\n\t-> shape: {points_t.shape}\n''')

        # transposing in higher dimensions
        some_t = torch.ones(3, 4, 5)
        transpose_t = some_t.transpose(0, 2)
        print(f'''\nsome_t:\n\t-> shape: {some_t.shape}\n\t-> stride: {some_t.stride()}\n
transpose_t:\n\t-> shape: {transpose_t.shape}\n\t-> stride: {transpose_t.stride()}''')
        return

    def indexingStorage(self):
        points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
        points_storage = points.storage()  # always return one-dimensional array/list of storage data
        points_storage[0] = 2.0  # It changes the value in the tensor just like a list/array

        second_point = points[1].clone()
        second_point[0] = 10.0
        # print(points)
        return


    def modifyingStoredValues(self):
        a = torch.ones(3, 2)
        # print(a)
        a.zero_()
        # print(a)
        return


def main():
    chapter3 = Chapter3()

    # chapter3.firstTensor()
    # chapter3.img2GreyScale()
    # chapter3.tensorDataTypes()
    chapter3.tensorAPI()
    # chapter3.indexingStorage()
    # chapter3.modifyingStoredValues()
    return


if __name__ == "__main__":
    # main()
    timer_data(main)
    # get_cuda_device_data()
