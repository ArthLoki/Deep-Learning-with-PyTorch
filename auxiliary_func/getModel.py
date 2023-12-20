import torch


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