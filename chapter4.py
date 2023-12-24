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
        self.dev_code = 0  # Default: code = -1 => running on CPU. OBS: code = 0 => Running on GPU/cuda
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

        # Loading a specialized format
        # load a sample CT scan using the volread function in the imageio module, which
        # takes a directory as an argument and assembles all Digital Imaging and Communications
        # in Medicine (DICOM) files in a series in a NumPy 3D array
        dir_path = f'{self.base_path_data}/volumetric-dicom/2-LUNG 3.0  B70f-04083'
        vol_arr = imageio.volread(dir_path, 'DICOM')
        # print(vol_arr.shape)

        # the layout is different from what PyTorch expects, due to
        # having no channel information. we’ll have to make room for the channel dimension
        # using unsqueeze
        vol = torch.from_numpy(vol_arr).float()  # NumPy array to PyThorch tensor
        vol = torch.unsqueeze(vol, 0)
        print(vol.shape)

        '''
        When you are unsqueezing a tensor, it is ambiguous which dimension you wish to 'unsqueeze' it across 
        (as a row or column etc). The dim argument dictates this - i.e. position of the new dimension to be added
        
        torch.unsqueeze(input, dim) → Tensor
        '''

        batch_size = vol.shape[0]
        batch = torch.zeros(batch_size, *vol.shape[1:], device=self.dev_name)
        vol = torch.unsqueeze(batch, 0)
        print(vol.shape)

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
        # print(target_onehot)

        '''
        Let’s see what scatter_ does. First, we notice that its name ends with an underscore.
        As you learned in the previous chapter, this is a convention in PyTorch that indicates
        the method will not return a new tensor, but will instead modify the tensor in place.
        The arguments for scatter_ are as follows:
             The dimension along which the following two arguments are specified
             A column tensor indicating the indices of the elements to scatter
             A tensor containing the elements to scatter or a single scalar to scatter (1, in
            this case)

        In other words, the previous invocation reads, “For each row, take the index of the target
        label (which coincides with the score in our case) and use it as the column index
        to set the value 1.0.” The end result is a tensor encoding categorical information.

        The second argument of scatter_, the index tensor, is required to have the same
        number of dimensions as the tensor we scatter into. Since target_onehot has two
        dimensions (4,898 × 10), we need to add an extra dummy dimension to target using
        unsqueeze.
        '''

        target_unsqueezed = target.unsqueeze(1)
        # print(target_unsqueezed)

        '''
        The call to unsqueeze adds a singleton dimension, from a 1D tensor of 4,898 elements
        to a 2D tensor of size (4,898 × 1), without changing its contents—no extra elements
        are added; we just decided to use an extra index to access the elements. That is, we
        access the first element of target as target[0] and the first element of its
        unsqueezed counterpart as target_unsqueezed[0,0].

        PyTorch allows us to use class indices directly as targets while training neural networks.
        However, if we wanted to use the score as a categorical input to the network, we
        would have to transform it to a one-hot-encoded tensor.
        '''

        # When to categorize

        # obtain the mean and standard deviations for each column
        # dim=0 indicates that the reduction is performed along dimension 0
        # we can normalize the data by subtracting the mean and dividing by the
        # standard deviation, which helps with the learning process
        data_mean = torch.mean(data, dim=0)
        # print(data_mean)

        data_var = torch.var(data, dim=0)
        # print(data_var)

        data_normalized = (data - data_mean) / torch.sqrt(data_var)
        # print(data_normalized)

        # Finding thresholds

        # determine which rows in
        # target correspond to a score less than or equal to 3

        bad_indexes = target <= 3

        # PyTorch also provides comparison functions, here torch.le(target, 3), but using operators seems to be a
        # good standard
        # print(bad_indexes.shape, bad_indexes.dtype, bad_indexes.sum())

        # PyTorch called advanced indexing, we can use a tensor with data type torch.bool to
        # index the data tensor
        # This will essentially filter data to be only items (or rows) corresponding
        # to True in the indexing tensor
        # The bad_indexes tensor has the same shape as target, with values of False or True depending on
        # the outcome of the comparison between our threshold and each element in the original target tensor

        bad_data = data[bad_indexes]
        # print(bad_data.shape)

        # the new bad_data tensor has 20 rows, the same as the number of rows with
        # True in the bad_indexes tensor
        # Now we can start to get information about wines grouped into good, middling, and bad categories
        # taking the .mean() of each column
        bad_data = data[target <= 3]
        mid_data = data[(target > 3) & (target < 7)]
        good_data = data[target >= 7]

        bad_mean = torch.mean(bad_data, dim=0)
        mid_mean = torch.mean(mid_data, dim=0)
        good_mean = torch.mean(good_data, dim=0)

        for i, args in enumerate(zip(col_list, bad_mean, mid_mean, good_mean)):
            print('{:2} {:20} {:6.2f} {:6.2f} {:6.2f}'.format(i, *args))

        # the bad wines seem to have higher total sulfur dioxide, among other differences. We could use a threshold on
        # total sulfur dioxide as a crude criterion for discriminating good wines from bad ones.
        # Let’s get the indexes where the total sulfur dioxide column is below the midpoint we
        # calculated earlier

        total_sulfur_threshold = 141.83
        total_sulfur_data = data[:, 6]
        predicted_indexes = torch.lt(total_sulfur_data, total_sulfur_threshold)
        # print(predicted_indexes.shape, predicted_indexes.dtype, predicted_indexes.sum())

        # This means our threshold implies that just over half of all the wines are going to be
        # high quality. Next, we’ll need to get the indexes of the actually good wines

        actual_indexes = target > 5
        # print(actual_indexes.shape, actual_indexes.dtype, actual_indexes.sum())

        # We will perform a logical “and” between our
        # prediction indexes and the actual good indexes (remember that each is just an array
        # of zeros and ones) and use that intersection of wines-in-agreement to determine how
        # well we did
        n_matches = torch.sum(actual_indexes & predicted_indexes).item()
        n_predicted = torch.sum(predicted_indexes).item()
        n_actual = torch.sum(actual_indexes).item()
        # print(n_matches, n_matches / n_predicted, n_matches / n_actual)
        return

    def working_with_time_series(self):
        # Loading data from csv
        bikes_numpy = np.loadtxt(
            f'{self.base_path_data}/bike-sharing-dataset/hour-fixed.csv',
            dtype=np.float32,
            delimiter=',',
            skiprows=1,
            converters={1: lambda x: float(x[8:10])}
            # converters converts date strings to numbers corresponding to the day of the month in column 1
        )
        bikes = torch.from_numpy(bikes_numpy)
        # print(bikes.shape, bikes.stride())  # OUTPUT: torch.Size([17520, 17]) (17, 1)

        # let’s reshape the data to have 3 axes — day, hour, and then our 17 columns
        daily_bikes = bikes.view(-1, 24, bikes.shape[1])
        print(daily_bikes.shape, daily_bikes.stride())  # OUTPUT: torch.Size([730, 24, 17]) (408, 17, 1)

        # First, bikes.shape[1] is 17, the number of columns in the
        # bikes tensor. But the real crux of this code is the call to view, which is really important:
        # it changes the way the tensor looks at the same data as contained in storage.

        # calling view on a tensor returns a new tensor
        # that changes the number of dimensions and the striding information, without
        # changing the storage

        # This means we can rearrange our tensor at basically zero cost,
        # because no data will be copied. Our call to view requires us to provide the new shape
        # for the returned tensor. We use -1 as a placeholder for “however many indexes are
        # left, given the other dimensions and the original number of elements.”

        # Our bikes tensor will have each row
        # stored one after the other in its corresponding storage. This is confirmed by the output
        # from the call to bikes.stride() earlier.

        # For daily_bikes, the stride is telling us that advancing by 1 along the hour dimension
        # (the second dimension) requires us to advance by 17 places in the storage (or
        # one set of columns);

        # advancing along the day dimension (the first dimension)
        # requires us to advance by a number of elements equal to the length of a row in
        # the storage times 24 (here, 408, which is 17 × 24).

        # We see that the rightmost dimension is the number of columns in the original
        # dataset. Then, in the middle dimension, we have time, split into chunks of 24 sequential
        # hours. In other words, we now have N sequences of L hours in a day, for C channels.
        # To get to our desired N × C × L ordering, we need to transpose the tensor

        daily_bikes = daily_bikes.transpose(1, 2)
        print(daily_bikes.shape, daily_bikes.stride())
        return


def main():
    chapter4 = Chapter4()

    # chapter4.working_with_images()
    # chapter4.volumetric_data_3d_image()
    # chapter4.representing_tabular_data()
    chapter4.working_with_time_series()

    return


if __name__ == '__main__':
    # main()
    timer_data(main)
