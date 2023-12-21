import os
import csv
import torch
import numpy as np
import imageio.v2 as imageio
from auxiliary_func.getPath import get_base_path, get_chapter_data_path
from auxiliary_func.getDevice import get_device_name, get_device_data
from auxiliary_func.getTimer import timer_data

'''
WARNING:
DeprecationWarning: Starting with ImageIO v3 the behavior of this function will switch to that of iio.v3.imread. 
To keep the current behavior (and make this warning disappear) use `import imageio.v2 as imageio` or call 
`imageio.v2.imread` directly. img_arr = imageio.imread(f'{self.base_path_image}/p1ch4/image-dog/bobby.jpg')
'''


class Chapter4:

    def __init__(self):
        self.base_path = get_base_path()
        self.base_path_data = get_chapter_data_path(1, 4)
        self.dev_code = -1  # Default: code = -1 => running on CPU. OBS: code = 0 => Running on GPU/cuda
        self.dev_name = get_device_name(self.dev_code)
        return

    def working_with_images(self):
        # Loading image: img_arr is a numpy array
        # an input tensor H × W × C is obtained. We want a pytorch tensor C x H x W
        # img_arr = imageio.imread(f'{self.base_path_data}/image-dog/bobby.jpg')
        # print(img_arr.shape)

        # Changing layout to Pytorch tensor
        # We'll use the tensor’s permute method with the old dimensions for each new dimension
        # to get to an appropriate layout
        # we get a proper layout by having channel 2 first and then channels 0 and 1
        # img = torch.from_numpy(img_arr)
        # out = img.permute(2, 0, 1)
        # print(out.shape)

        # Multiple images
        # We need to add batch dimension in the input tensor. Our input tensor dim are N x C x H x W
        batch_size = 3
        batch = torch.zeros(batch_size, 3, 256, 256, dtype=torch.uint8, device=self.dev_name)
        get_device_data(batch)

        data_dir = f'{self.base_path_data}/image-cats/'
        filenames = [name for name in os.listdir(data_dir) if os.path.splitext(name)[-1] == '.png']
        for i, filename in enumerate(filenames):
            img_arr = imageio.imread(os.path.join(data_dir, filename))
            img_t = torch.from_numpy(img_arr)
            img_t = img_t.permute(2, 0, 1)
            img_t = img_t[:3]
            batch[i] = img_t

        batch = batch.float()
        batch /= 255.0

        n_channels = batch.shape[1]
        for c in range(n_channels):
            mean = torch.mean(batch[:, c])
            std = torch.std(batch[:, c])
            batch[:, c] = (batch[:, c] - mean) / std
        return

    def volumetric_data_3d_image(self):
        # 3D images: Volumetric data
        # there’s no fundamental difference between a tensor storing volumetric
        # data versus image data
        # have an extra dimension, depth, after the channel
        # dimension, leading to a 5D tensor of shape N × C × D × H × W

        # load a sample CT scan using the volread function in the imageio module, which
        # takes a directory as an argument and assembles all Digital Imaging and Communications
        # in Medicine (DICOM) files in a series in a NumPy 3D array
        dir_path = f'{self.base_path_data}/volumetric-dicom/2-LUNG 3.0  B70f-04083'
        vol_arr = imageio.volread(dir_path, 'DICOM')
        # print(vol_arr.shape)
        return

    def representing_tabular_data(self):
        # Loading a csv file and turning the result NumPy array into a PyTorch tensor
        wine_path = f'{self.base_path_data}/tabular-wine/winequality-white.csv'
        wineq_numpy = np.loadtxt(wine_path, dtype=np.float32, delimiter=";", skiprows=1)
        # print(wineq_numpy)

        # check that all the data has been read
        csv_wine = csv.reader(open(wine_path), delimiter=";")
        col_list = next(csv_wine)
        # print(wineq_numpy.shape, col_list)

        # convert the NumPy array to a PyTorch tensor
        wineq = torch.from_numpy(wineq_numpy)
        # print(wineq.shape, wineq.dtype)

        # Representing Scores

        # we will typically remove the
        # score from the tensor of input data and keep it in a separate tensor, so that we can use
        # the score as the ground truth without it being input to our model
        data = wineq[:, :-1]  # Selects all rows and all columns except the last
        # print(data, data.shape)

        target = wineq[:, -1]  # Selects all rows and the last column
        # print(target, target.shape)

        # transform the target tensor in a tensor of labels, we have two options,
        # depending on the strategy or what we use the categorical data for
        # One is simply to treat labels as an integer vector of scores
        target = wineq[:, -1].long()
        # print(target)

        # If targets were string labels, like wine color, assigning an integer number to each string
        # would let us follow the same approach

        # One-Hot Encoding

        # We can achieve one-hot encoding using the scatter_ method, which fills the tensor
        # with values from a source tensor along the indices provided as arguments
        target_onehot = torch.zeros(target.shape[0], 10)
        # print(target_onehot)
        target_onehot.scatter_(1, target.unsqueeze(1), 1.0)
        print(target_onehot)
        return

def main():
    chapter4 = Chapter4()

    # chapter4.working_with_images()
    # chapter4.volumetric_data_3d_image()  # Bugfix
    chapter4.representing_tabular_data()

    return


if __name__ == '__main__':
    # main()
    timer_data(main)
