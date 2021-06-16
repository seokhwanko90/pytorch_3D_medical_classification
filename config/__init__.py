from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()
_C.CKPT_DIR = "ckpt/tmp"

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.root_dataset = "./data/"
_C.DATASET.list_train = "./data/training.odgt"
_C.DATASET.list_val = "./data/validation.odgt"
_C.DATASET.list_test = "./data/test.odgt"
_C.DATASET.md_classes = ['NOR','ABN']
_C.DATASET.num_class = 2
_C.DATASET.spatial_size = 128
_C.DATASET.sample_count = 128
_C.DATASET.random_flip = False
_C.DATASET.stain_norm = ""

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.type = "Classification"
_C.MODEL.arch = "resnet50"
_C.MODEL.pretrained_path = ""
_C.MODEL.pretrained_num_class = 0

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.tr_batchsize = 2
_C.TRAIN.tr_num_epochs = 20
_C.TRAIN.tr_epoch_num_iters = 1
_C.TRAIN.ckpt_interval = 1
_C.TRAIN.tr_optim = "SGD"
_C.TRAIN.tr_lr = 0.001
_C.TRAIN.tr_lr_pow = 0.9
_C.TRAIN.tr_momentum = 0.9
_C.TRAIN.tr_weight_decay = 1e-4
_C.TRAIN.workers = 1

# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------
_C.VAL = CN()
_C.VAL.vl_batchsize = 1
_C.VAL.visualize = False
_C.VAL.checkpoint = "model_epoch_10.pth"

# -----------------------------------------------------------------------------
# Testing
# -----------------------------------------------------------------------------
_C.TEST = CN()
_C.TEST.ts_batchsize = 1
_C.TEST.checkpoint = "model_epoch_10.pth"
_C.TEST.result = "./tmp"

