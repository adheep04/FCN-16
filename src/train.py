import torch
import torch.nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import SGD
from model import FCN

def train():
    fcn8 = FCN(n_class=30, net='8')
    








if __name__ == '__main__':
    
    
    
    
    train_dataloader = trainer.get_dataloader(data_path='./data/mapillary-vista/training')
    optimizer = Optimizer()
