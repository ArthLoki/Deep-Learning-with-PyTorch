import torch


def get_index_current_cuda_device():
    return torch.cuda.current_device()


def get_cuda_device_data():
    available = torch.cuda.is_available()
    index = torch.cuda.current_device()
    dev_name = torch.cuda.get_device_name(index)
    torch_dev_name = get_device_name(index)

    print('\nCUDA DEVICE DATA:')
    print(f'\tCUDA AVAILABLE: {available}')
    print(f'\tCUDA INDEX: {index}')  # gets the index of the current device
    print(f'\tTORCH DEVICE NAME: {torch_dev_name}')
    print(f'\tDEVICE NAME: {dev_name}\n')
    return


def get_device_name(code=-1):
    dev_name = ''
    if torch.cuda.is_available():
        if code == 0:
            index_current_cuda_device = get_index_current_cuda_device()
            dev_name = f'cuda:{index_current_cuda_device}'
        elif code == -1:
            dev_name = 'cpu'
    else:
        dev_name = 'cpu'
    return dev_name


def get_device(code=1):
    dev_name = get_device_name(code)
    device = 'cpu'  # default device

    if dev_name != '':
        device = torch.device(dev_name)
    return device


def get_device_data(tensor):
    index_dev_tensor = tensor.get_device()
    dev_name = get_device_name(index_dev_tensor)
    print(f'''\nDEVICE DATA\n\tDEVICE NAME: {dev_name}\n\tDEVICE INDEX: {index_dev_tensor}\nEND DEVICE DATA\n''')


def dev2dev(tensor, code):
    print('\nASSIGNMENT: CHANGING DEVICE')
    device = get_device(code)
    assignment = tensor.to(device=device)
    # get_device_data(assignment)
    return assignment


def check_device_change(tensor, code):
    print('\nPREVIOUS DEVICE')
    get_device_data(tensor)

    print('\nNEW DEVICE')
    tensor_change = dev2dev(tensor, code)
    get_device_data(tensor_change)
    return
