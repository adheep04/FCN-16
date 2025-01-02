from pathlib import Path


class Config:
    
    # training/validation run num/type
    RUN = '3__lr_up_wd_d'
    
    # Paths
    DATA_TRAIN_DIR = Path("data/cityscapes/train/img") 
    LABEL_TRAIN_DIR = Path("data/cityscapes/train/label")
    
    DATA_VAL_DIR = Path("data/cityscapes/val/img")
    LABEL_VAL_DIR = Path("data/cityscapes/val/label")
    
    CHECKPOINT_DIR = Path("checkpoints/")
    
    LOG_DIR = Path("logs")

    # Model hyperparameters
    BATCH_SIZE = 1
    LEARNING_RATE = 8e-4
    NUM_EPOCHS = 10
    
    # image crop to save memory during training
    MAX_PIXELS = 1024*1024
    
    # 'class', true_id, color
    LABELS = {  
        ('road'         , 0 , (128, 64,128)),
        ('sidewalk'     , 1 , (244, 35,232) ),
        ('building'     , 2 , ( 70, 70, 70) ),
        ('wall'         , 3 ,  (102,102,156) ),
        ('fence'        , 4 ,  (190,153,153) ),
        ('pole'         , 5 ,  (153,153,153) ),
        ('traffic light', 6 ,  (250,170, 30) ),
        ('traffic sign' , 7 ,  (220,220,  0) ),
        ('vegetation'   , 8 ,  (107,142, 35) ),
        ('terrain'      , 9 ,  (152,251,152) ),
        ('sky'          , 10 , ( 70,130,180) ),
        ('person'       , 11 , (220, 20, 60) ),
        ('rider'        , 12 , (255,  0,  0) ),
        ('car'          , 13 , (  0,  0,142) ),
        ('truck'        , 14 ,  (  0,  0, 70) ),
        ('bus'          , 15 ,  (  0, 60,100) ),
        ('train'        , 16 ,  (  0, 80,100) ),
        ('motorcycle'   , 17 ,  (  0,  0,230) ),
        ('bicycle'      , 18 ,  (119, 11, 32) ),
        ('null'         , 19 ,  None),
    }
    
    # maps true_id -> train_id
    # 19 is ignore class
    LABEL_MAP = {
    0 : 19,
    1 : 19,
    2 : 19,
    3 : 19,
    4 : 19,
    5 : 19,
    6 : 19,
    7 : 0,
    8 : 1,
    9 : 19,
    10 : 19,
    11 : 2,
    12 : 3,
    13 : 4,
    14 : 19,
    15 : 19,
    16 : 19,
    17 : 5,
    18 : 19,
    19 : 6,
    20 : 7,
    21 : 8,
    22 : 9,
    23 : 10,
    24 : 11,
    25 : 12,
    26 : 13,
    27 : 14,
    28 : 15,
    29 : 19,
    30 : 19,
    31 : 16,
    32 : 17,
    33 : 18,
    34 : 19,
    
}

    # 19 classes + 1 ignore class
    CLASS_SIZE = len(LABELS)

config = Config()