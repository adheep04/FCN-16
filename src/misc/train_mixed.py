import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import SGD
from model import FCN
import math
from dataset import CityscapesDataset
from config import config
from torch.amp import autocast, GradScaler




def train():
    
    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # initialize model
    fcn8 = FCN(n_class=20, net='8')
    fcn8.to(device) # bring it to CUDA:) or cpu if u dont have cuda ig
    
    # get datasets
    train_data = CityscapesDataset(config.DATA_TRAIN_DIR, config.LABEL_TRAIN_DIR)
    val_data = CityscapesDataset(config.DATA_VAL_DIR, config.LABEL_VAL_DIR) 
    
    # initialize dataloader
    # batch size of 1
    train_dataloader = DataLoader(
        dataset = train_data,
        batch_size = config.BATCH_SIZE,
        shuffle = True
    )

    val_dataloader = DataLoader(
        dataset = val_data,
        batch_size = config.BATCH_SIZE,
        shuffle = True
    )
    
    loss_fn = nn.CrossEntropyLoss()
    
    # scales the loss/gradients after switching to float16 to avoid underflow/overflow
    scaler = GradScaler('cuda')
    
    training_stats = []
    
    log_step = 4
    
    ''' authors use 2 x learning_rate for biases'''
    
    
    # initialize lists to hold model parameters
    bias = []
    weight = []
    
    # seperate model parameters into weights and biases
    for name, param in fcn8.named_parameters():
        if 'bias' in name:
            bias.append(param)
        else:
            weight.append(param)
        
    # initialize optimizer with momentum (add accumulated past gradients to smoothen updates)
    # set weight decay = 0.0005 (l2 regularization to prevent overfitting / large weight values)
    optimizer = SGD(params=[
        {'params' : bias, 'lr' : 2 * config.LEARNING_RATE}, # 2 x lr
        {'params' : weight, 'lr' : config.LEARNING_RATE}
        ], momentum=0.9, weight_decay=5e-4)

    
    ''' train '''
    
    fcn8.train()
    

    for epoch in range(config.NUM_EPOCHS):
        if epoch > 1:
            break
        print(f'starting epoch {epoch}')
        for step, (data, label) in enumerate(train_dataloader):
            if step > 1000:
                break
            
            # initialize gradients (so they don't accumulate)
            optimizer.zero_grad()
            
            # send data to device
            data = data.to(device)
            label = label.to(device)
            # writer.add_graph(fcn8, data)
            
            # use automatic mixed precision (float32 vs float16) for efficiency
            with autocast('cuda', enabled=True):
                # forward pass  
                pred = fcn8(data)
                loss = loss_fn(pred, label)
            
            # print and log loss
            if step%log_step == 0:
                print(loss.item())
                
            if step%log_step == 0:
                training_stats.append(loss.item())
            
            # backward pass (calculate gradients)
            # scales loss if needed to prevent underflow
            scaler.scale(loss).backward()
            
            # update parameters
            # unscales gradients back to original scale
            scaler.step(optimizer)
            
            # adjusts scale factor
            scaler.update()

if __name__ == '__main__':
    train()