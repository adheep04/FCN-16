import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import SGD
from torch.amp import autocast, GradScaler
import torch.optim.lr_scheduler as schedule

import numpy as np
from math import inf

import time

from pathlib import Path

from model import FCN
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
    
    # initialize focal loss with ignore class 19
    loss_fn = FocalLoss(ignore_index=19)
    
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
    optimizer = SGD(params=[
        # lr = 8e-4
        {'params' : bias, 'lr' : 2 * config.LEARNING_RATE}, # 2 x lr
        {'params' : weight, 'lr' : config.LEARNING_RATE}
        #  momentum = 0.9, wd = 5e-6
        ], momentum=config.MOMENTUM, weight_decay=config.WEIGHT_DECAY)
    
    scheduler = schedule.CyclicLR(
        optimizer,
        base_lr=4e-5,
        max_lr=3e-3,
        step_size_up=300,  # Steps per half cycle
        mode='triangular2'  # Decreasing amplitude over time
    )
        
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
    
                
                # send data to device
                data = data.to(device)
                label = label.to(device)
                
                # use automatic mixed precision (float32 vs float16) for efficiency
                with autocast('cuda' if torch.cuda.is_available() else 'cpu', enabled=True):
                    # forward pass  
                    output = fcn8(data)
                    loss = loss_fn(output, label) / config.ACCUM_STEPS
                
                # log loss
                if step % log_step == 0:
                    
                    # miou is the mean of the class ious
                    miou = np.mean(class_iou(output, label)[0])
                    acc = pixel_acc(output, label)
                    print(f'{loss}, {miou}, {acc}')
                    
                    # log loss and miou 
                    writer.add_scalar('Loss/train', loss.item(), global_step)
                    writer.add_scalar('mIoU/train', miou, global_step)
                    writer.add_scalar('accuracy/train', acc, global_step)
                    
                    # log sample predictions
                    if step % (log_step * 10) == 0:
                        pred = output.softmax(dim=1).argmax(dim=1)
                        writer.add_images('Predictions', pred.unsqueeze(1).float() // 19, global_step)
                        writer.add_images('Ground Truth', label.unsqueeze(1).float() // 19, global_step)
                
                # backward pass (calculate gradients)
                # scales loss if needed to prevent underflow
                scaler.scale(loss).backward()
                
                
                if (step + 1) % config.ACCUM_STEPS == 0:
                    # update parameters
                    # unscales gradients back to original scale
                    scaler.step(optimizer)
                    
                    scheduler.step()
                    
                    # adjusts scale factor
                    scaler.update()
                    
                    # reset gradients to 0 (so they don't accumulate past step size)
                    optimizer.zero_grad(set_to_none=True)
                
                # update for tb
                global_step += 1
                    
    except KeyboardInterrupt:
        print("training interrupted. Saving checkpoint...")
        writer.close()
        torch.save(fcn8.state_dict(), 
            f=config.CHECKPOINT_DIR / f'interrupted_s_dict_{config.RUN}_{int(time.time())}')
        print(f'epoch: {epoch}, step: {step}')
        return
        
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
    acc = 0
    
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
            
            # get metrics
            step_ious, class_ids = class_iou(output, label)
            step_iou = np.mean(step_ious)
            step_acc = pixel_acc(output, label)
                
            # update running average metrics
            loss += (step_loss - loss) / (step + 1) if step != 0 else step_loss
            iou += (step_iou - iou) / (step + 1) if step != 0 else step_iou
            acc += (step_acc - acc) / (step + 1) if step != 0 else acc
            ious[class_ids] += (step_ious - ious[class_ids]) / (step + 1) if step != 0 else step_ious
            
        
                
        stats = {
            'mean_loss' : loss,
            'mean_iou' : iou,
            'miou_per_class' : ious,
            'acc' : acc
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
        - label: tensor shape (1, 1, h, w)
        
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

def pixel_acc(model_out, label):
    # convert from predictions for all classes to single prediction per pixel
    # (1, n_class, h, w) -> (1, h, w)
    pred = model_out.softmax(dim=1).argmax(dim=1).to(dtype=torch.uint8)
            
    # ensure tensors have the same shape
    assert pred.shape == label.shape, "Predictions and labels must have the same shape"
    
    accuracy = (pred.cpu().numpy() == label.cpu().numpy()).mean()
    
    return accuracy.item()

class FocalLoss(nn.Module):
   def __init__(self, alpha=config.ALPHA, gamma=config.GAMMA, ignore_index=19):
       super().__init__()
       self.alpha = alpha
       self.gamma = gamma
       self.ignore_index = ignore_index

   def forward(self, inputs, targets):
       ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
       pt = torch.exp(-ce_loss)
       focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
       
       return focal_loss.mean()


if __name__ == '__main__':
    train()