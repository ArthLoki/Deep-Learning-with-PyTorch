import torch
import time

# DEVICE


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


def get_device_name(code=1):
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
    device = ''

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
    assignment = tensor.to(device)
    get_device_data(assignment)
    return


def check_device_change(tensor, code):
    print('\nPREVIOUS DEVICE')
    get_device_data(tensor)

    print('\nNEW DEVICE')
    dev2dev(tensor, code)
    return

# TIMER


def get_timer():
    return time.time()


def print_device_timer(start, end):
    elapsed_time = end - start
    print(f'TIME ELAPSED: {elapsed_time} seconds\n')
    return


def timer_data(func):
    print('\n-----> START TIMER <-----')
    start_time = get_timer()

    # param function here
    func()

    print('\n-----> END TIMER <-----\n')
    end_time = get_timer()
    print_device_timer(start_time, end_time)
    return
