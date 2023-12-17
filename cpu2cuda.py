import torch


def getIndexCurrentCudaDevice():
    return torch.cuda.current_device()

# def setCudaDevice(device):
#     torch.cuda.set_device(device)


def get_device(code=1):
    dev_name = ''
    if torch.cuda.is_available():
        if code == 0:
            index_current_cuda_device = getIndexCurrentCudaDevice()
            dev_name = f'cuda:{index_current_cuda_device}'
        elif code == 1:
            dev_name = 'cpu'
    else:
        dev_name = 'cpu'

    device = torch.device(dev_name)
    return device


def cpu2cuda_or_cuda2cpu(tensor, code):
    device = get_device(code)
    assignment = tensor.to(device)
    return assignment
