import os
import torch


# GET PATH


def get_current_path():
    return os.getcwd().replace('\\', '/')

def get_base_path():
    og_path = get_current_path()
    l_path = og_path.split('/')
    base_path = l_path[0]
    for i in range(1, len(l_path)-1):
        base_path += f'/{l_path[i]}'
    return base_path


# LOAD/SAVE MODEL


def loadModel(path, filename='model.t'):
    with open(path+filename, 'rb') as f:
        model = torch.load(f)
    return model


def saveModel(path, model, filename='model.t'):
    with open(path+filename, 'wb') as f:
        torch.save(model, f)
    return


def getModel(path, filename='model.t', file_mode='load', model=''):
    if file_mode == 'save':
        saveModel(path, model, filename)
        return
    elif file_mode == 'load':
        model = loadModel(path, filename)
        return model
    else:
        print("Invalid file mode")
        return
