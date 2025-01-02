import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import SGD
from torch.amp import autocast, GradScaler

import numpy as np
from math import inf

import time

from pathlib import Path

from misc.model_older import FCN
from dataset import CityscapesDataset
from config import config


def train(resume=False, resume_file_path=None):
    ''' training setup '''
    
    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # initialize model
    fcn8 = FCN(n_class=20, net='8')
    # bring it to CUDA:) or cpu if u dont have cuda ig
    fcn8.to(device) 
    
    # tensorboard
    writer = SummaryWriter(log_dir=config.LOG_DIR / f'run_{config.RUN}')    
    
    if resume:
        state_dict = torch.load(resume_file_path)
        fcn8.load_state_dict(state_dict)
    
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
        shuffle = False
    )
    
    # labels with value 19 are ignored
    loss_fn = nn.CrossEntropyLoss(ignore_index=19)
    
    # scales the loss/gradients after switching to float16 to avoid underflow/overflow
    scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu')
    
    log_step = 30
    global_step = 0  #counter for tensorboard

    
    ''' use 2 x learning_rate for biases (like authors)'''
    
    # initialize lists to hold model parameters
    bias = []
    weight = []
    
    # seperate model parameters into weights and biases
    for name, param in fcn8.named_parameters():
        if 'bias' in name:
            bias.append(param)
        else:
            weight.append(param)
    
    # ensure bias list is populated
    assert len(bias) > 0
        
    # initialize optimizer with momentum (add accumulated past gradients to smoothen updates)
    # set weight decay = 0.0005 (l2 regularization to prevent overfitting / large weight values)
    optimizer = SGD(params=[
        {'params' : bias, 'lr' : 2 * config.LEARNING_RATE}, # 2 x lr
        {'params' : weight, 'lr' : config.LEARNING_RATE}
        ], momentum=0.9, weight_decay=5e-6)

    
    ''' training loop '''
    
    fcn8.train()

    try:
        for epoch in range(config.NUM_EPOCHS):
            
            # run validation for last epoch
            if epoch != 0:
                # run validation
                val_stats = validation(fcn8, val_dataloader, epoch=epoch-1)
                
                # log stats for tensorboard
                writer.add_scalar('Loss/val', val_stats['mean_loss'], epoch)
                writer.add_scalar('mIoU/val', val_stats['mean_iou'], epoch)
                
                # log per-class IoU
                for class_id, iou in enumerate(val_stats['miou_per_class']):
                    writer.add_scalar(f'IoU/class_{class_id}', iou, epoch)

            print(f'starting epoch {epoch}')
            for step, (data, label) in enumerate(train_dataloader):
       
                # initialize gradients (so they don't accumulate)
                optimizer.zero_grad(set_to_none=True)
                
                # send data to device
                data = data.to(device)
                label = label.to(device)
                
                # use automatic mixed precision (float32 vs float16) for efficiency
                with autocast('cuda' if torch.cuda.is_available() else 'cpu', enabled=True):
                    # forward pass  
                    output = fcn8(data)
                    loss = loss_fn(output, label)
                
                # log loss
                if step % log_step == 0:
                    
                    # miou is the mean of the class ious
                    miou = np.mean(class_iou(output, label)[0])
                    print(f'{loss}, {miou}')
                    
                    # log loss and miou 
                    writer.add_scalar('Loss/train', loss.item(), global_step)
                    writer.add_scalar('mIoU/train', miou, global_step)
                    
                    # log learning rates
                    writer.add_scalar('LR/bias', optimizer.param_groups[0]['lr'], global_step)
                    writer.add_scalar('LR/weight', optimizer.param_groups[1]['lr'], global_step)
                    
                    # log sample predictions
                    if step % (log_step * 10) == 0:
                        pred = output.softmax(dim=1).argmax(dim=1)
                        writer.add_images('Predictions', pred.unsqueeze(1).float(), global_step)
                        writer.add_images('Ground Truth', label.unsqueeze(1).float(), global_step)
                
                # backward pass (calculate gradients)
                # scales loss if needed to prevent underflow
                scaler.scale(loss).backward()
                
                # update parameters
                # unscales gradients back to original scale
                scaler.step(optimizer)
                
                # adjusts scale factor
                scaler.update()
                
                # update for tb
                global_step += 1
                
    except Exception as e:
        print(f"Error during training: {e}")
        writer.close() 
        torch.save(fcn8.state_dict(),
            f=config.CHECKPOINT_DIR / f'paused_s_dict_{config.RUN}_{int(time.time())}')
         
    writer.close() 
    torch.save(fcn8.state_dict(),
        f=config.CHECKPOINT_DIR / f'finished_s_dict_{config.RUN}_{int(time.time())}')
    return "training done!"
           
def validation(
    model, 
    val_dataloader, 
    loss_fn=nn.CrossEntropyLoss(ignore_index=19), 
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    epoch=None):
    '''
    run a single round of validation using loss and mean-intersection/union

    args:
    - they r obvious :D
    
    '''
    
    # initialize metric accumulation variables
    ious = np.zeros(19)
    loss = 0
    iou = 0
    
    # move model to device and set to evalulation mode
    model = model.to(device)
    model.eval() 
    
    print('starting validation round')
    with torch.no_grad():      # no gradient 
        for step, (data, label) in enumerate(val_dataloader):
            
            # move to device
            data = data.to(device)
            label = label.to(device)
            
            # forward pass  
            output = model(data)
            step_loss = loss_fn(output, label)
            
            # get iou for each class and overall mean
            step_ious, class_ids = class_iou(output, label)
            step_iou = np.mean(step_ious)
                
            # update running average metrics
            loss += (step_loss - loss) / (step + 1) if step != 0 else step_loss
            iou += (step_iou - iou) / (step + 1) if step != 0 else step_iou
            ious[class_ids] += (step_ious - ious[class_ids]) / (step + 1) if step != 0 else step_ious
        
                
        stats = {
            'mean_loss' : loss,
            'mean_iou' : iou,
            'miou_per_class' : ious,
        }
        
        torch.save(stats, f=config.CHECKPOINT_DIR / f'val_{config.RUN}_{epoch}')
        print(stats)
        return stats       
                         
def class_iou(model_out, label):
        '''
        metric for evaluating image segmentation tasks by dividing
        the intersection area by the union area of a given object in 
        both label and prediction images (measuring overlapp)
        
        args:
        - model_out: tensor shape (1, n_class, h, w)
        
        output:
        - (np.array(n_class), mean_iou float)
            - iou per class
        '''
        
        # gets a set of all class labels in the sample
        label_class_ids = label.unique().tolist()
        
        # convert from predictions for all classes to single prediction per pixel
        # (1, n_class, h, w) -> (1, 1, h, w)
        pred = model_out.softmax(dim=1).argmax(dim=1).to(dtype=torch.uint8)
        
        # get set of prediction classes by model
        pred_class_ids = pred.unique().tolist()
        
        # set of all predictions
        class_ids = set(label_class_ids + pred_class_ids)
        class_ids.discard(19) # remove ignore class
        
        assert len(class_ids) < 20
        
        # initialize per class iou score list
        scores = []
        
        # iterate through all types
        for id in class_ids:
            # if both pred and label contain type object
            if id in pred_class_ids and id in label_class_ids:
                
                # get boolean masks that are True where the pixel value == the type
                pred_mask = (pred == id)
                label_mask = (label == id)
                
                # get the boolean mask for the union and intersection of pred and label 
                union = pred_mask | label_mask          # using or operator for union
                intersection = pred_mask & label_mask   # using and operator for intersection
                type_iou = float(torch.sum(intersection))/float(torch.sum(union))
                scores.append(type_iou)
            else:
                # if a type is in label but isn't in pred, it's a false positive
                # if a type is in pred but isn't in label, it's a false negative
                # either case it's a 0
                scores.append(0)
        
        return scores, np.array(list(class_ids), dtype=int)
