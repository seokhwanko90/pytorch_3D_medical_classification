import torch
import torch.nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from models._3d import resnet
import utils
from dataset import DatasetList3d
import argparse
from config import _C as cfg
import time
import os

device = torch.device('cuda')


def main(cfg):
    # Data loading code

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    # '''
    trainset = DatasetList3d(cfg.DATASET.list_train, cfg.DATASET.md_classes, transforms.Compose([
        transforms.RandomResizedCrop(cfg.DATASET.spatial_size),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))
    valset = DatasetList3d(cfg.DATASET.list_val, cfg.DATASET.md_classes, transforms.Compose([
        transforms.RandomResizedCrop(cfg.DATASET.spatial_size),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))
    # '''
    '''
    classes = {}
    trainset = ImageFolder(traindir, classes, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    '''

    dataloader = dict()
    dataloader['train'] = torch.utils.data.DataLoader(trainset,
                                                      batch_size=cfg.TRAIN.tr_batchsize,
                                                      shuffle=True,
                                                      num_workers=cfg.TRAIN.workers,
                                                      pin_memory=True)
    dataloader['val'] = torch.utils.data.DataLoader(valset,
                                                    batch_size=cfg.VAL.vl_batchsize,
                                                    shuffle=True,
                                                    num_workers=cfg.TRAIN.workers,
                                                    pin_memory=True)

    num_classes = cfg.DATASET.num_class
    num_tr_data = len(dataloader['train'])

    cfg.TRAIN.tr_epoch_iters = num_tr_data
    cfg.TRAIN.max_iters = cfg.TRAIN.tr_epoch_iters * cfg.TRAIN.tr_num_epochs \
                          * cfg.TRAIN.tr_epoch_num_iters
    cfg.TRAIN.running_lr = cfg.TRAIN.tr_lr
    cfg.TRAIN.best_acc = [0.0, 0.0]
    cfg.TRAIN.tmp_acc = [0.0, 0.0]
    cfg.TRAIN.best_acc_cfg = [False, False, 1]

    if cfg.MODEL.pretrained_path:
        init_num_classes = cfg.MODEL.pretrained_num_class
    else:
        init_num_classes = num_classes

    net = resnet.resnet50(num_classes=init_num_classes, shortcut_type='B', spatial_size=cfg.DATASET.spatial_size,
                          sample_count=cfg.DATASET.sample_count).to(device)
    # net = resnet.resnet101(num_classes=2, shortcut_type='B', spatial_size=spatial_size, sample_count=sample_count).to(device)
    net = torch.nn.DataParallel(net, device_ids=None)

    if cfg.MODEL.pretrained_path:
        print('loading pretrained model {}'.format(cfg.MODEL.pretrained_path))
        pretrain = torch.load(cfg.MODEL.pretrained_path)
        net.load_state_dict(pretrain['state_dict'])

    net.module.fc = torch.nn.Linear(net.module.fc.in_features, num_classes)
    net.module.fc = net.module.fc.cuda()

    criterion = torch.nn.CrossEntropyLoss()
    # criterion = torch.nn.NLLLoss(ignore_index=-1)
    optimizer = get_optimizer(net, cfg)

    dataloader_tr = dataloader['train']
    dataloader_val = dataloader['val']

    for epoch in range(cfg.TRAIN.tr_num_epochs):
        train(net, dataloader_tr, optimizer, criterion, epoch + 1, cfg)
        val(net, dataloader_val, criterion, epoch + 1, cfg)

        #print(cfg.TRAIN.best_acc_cfg)
        logger.info(cfg.TRAIN.best_acc_cfg)
        logger.info(cfg.TRAIN.best_acc)
        checkpoint(net, cfg, epoch + 1)

    print('Finished Training')


def get_optimizer(net, cfg):
    optim = ''
    if cfg.TRAIN.tr_optim.lower() == 'sgd':
        optim = torch.optim.SGD(
            net.parameters(),
            lr=cfg.TRAIN.tr_lr,
            momentum=cfg.TRAIN.tr_momentum,
            weight_decay=cfg.TRAIN.tr_weight_decay)
    elif cfg.TRAIN.tr_optim.lower() == 'adam':
        optim = torch.optim.Adam(
            net.parameters(),
            lr=cfg.TRAIN.tr_lr,
            weight_decay=cfg.TRAIN.tr_weight_decay)

    return optim


def adjust_learning_rate(optimizer, cur_iter, cfg):
    scale_running_lr = ((1. - float(cur_iter) / cfg.TRAIN.max_iters) ** cfg.TRAIN.tr_lr_pow)
    cfg.TRAIN.running_lr = cfg.TRAIN.tr_lr * scale_running_lr

    optimizer_encoder = optimizer
    for param_group in optimizer_encoder.param_groups:
        param_group['lr'] = cfg.TRAIN.running_lr


def train(net, dataloader, optimizer, criterion, epoch, tmp_cfg):
    net.train()

    cum_loss = 0.0
    cum_acc = 0.0
    c = 0
    for eni in range(cfg.TRAIN.tr_epoch_num_iters):
        for i, data in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels, _ = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            cur_iter = c + i + (epoch - 1) \
                       * cfg.TRAIN.tr_epoch_iters * cfg.TRAIN.tr_epoch_num_iters
            adjust_learning_rate(optimizer, cur_iter, tmp_cfg)

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            softmax = torch.nn.Softmax()
            outputs = softmax(outputs)

            acc1, acc5 = utils.compute_accuracy(
                outputs,
                labels,
                augmentation=False,
                topk=(1, 2))
            cum_loss = cum_loss + loss
            cum_acc = cum_acc + acc1
        c = i

    avg_loss = cum_loss / ((i + 1) * cfg.TRAIN.tr_epoch_num_iters)
    avg_acc = cum_acc / ((i + 1) * cfg.TRAIN.tr_epoch_num_iters)
    # print statistics
    tr_log = '[%d, %5d] TRAIN lr: %.8f loss: %.5f acc: %.5f' %\
             (epoch, i + 1, cfg.TRAIN.running_lr, avg_loss, avg_acc)
    logger.info(tr_log)

    get_best_acc(float(avg_acc), cfg, 0)


def val(net, dataloader, criterion, epoch, cfg):
    net.eval()

    cum_loss = 0.0
    cum_acc = 0.0
    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels, _ = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            softmax = torch.nn.Softmax()
            outputs = softmax(outputs)

            acc1, acc5 = utils.compute_accuracy(
                outputs,
                labels,
                augmentation=False,
                topk=(1, 2))
            cum_loss = cum_loss + loss
            cum_acc = cum_acc + acc1
    # print statistics
    val_log = '[%d, %5d] VAL loss: %.5f acc: %.5f' %\
              (epoch, i + 1, cum_loss / (i + 1), cum_acc / (i + 1))
    logger.info(val_log)

    get_best_acc(float(cum_acc/(i+1)), cfg, 1)

def get_best_acc(acc, cfg, idx):
    if acc >= cfg.TRAIN.best_acc[idx]:
        cfg.TRAIN.tmp_acc[idx] = acc
        cfg.TRAIN.best_acc_cfg[idx] = True

def checkpoint(net, cfg, epoch):
    if cfg.TRAIN.ckpt_interval < 0:
        if cfg.TRAIN.best_acc_cfg[0] and cfg.TRAIN.best_acc_cfg[1] and\
                -cfg.TRAIN.ckpt_interval <= epoch:
            if -cfg.TRAIN.ckpt_interval != epoch:
                print('removing previous model...')
                os.remove('{}/model_epoch_{}_best.pth'\
                          .format(cfg.CKPT_DIR, cfg.TRAIN.best_acc_cfg[2]))
                cfg.TRAIN.best_acc_cfg[2] = epoch

            print('Saving best model...')
            dict_model = net.state_dict()
            torch.save(
                dict_model,
                '{}/model_epoch_{}_best.pth'.format(cfg.CKPT_DIR, epoch))

            cfg.TRAIN.best_acc = cfg.TRAIN.tmp_acc

        cfg.TRAIN.best_acc_cfg[0] = False
        cfg.TRAIN.best_acc_cfg[1] = False

    elif epoch % cfg.TRAIN.ckpt_interval == 0 or epoch == cfg.TRAIN.tr_num_epochs:
        print('Saving checkpoints...')
        dict_model = net.state_dict()
        torch.save(
            dict_model,
            '{}/model_epoch_{}.pth'.format(cfg.CKPT_DIR, epoch))


# functions to show an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="PyTorch 3D Classification Training"
    )
    parser.add_argument(
        "--cfg",
        default="config/DCM_RP2-resnet50_3d_CV5_rsp224.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--gpus",
        default="0",
        help="gpus to use, e.g. 0-3 or 0,1,2,3"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--logs",
        default="./logs",
        help="path to logs dir"
    )

    args = parser.parse_args()
    cfg.merge_from_file(args.cfg)

    if not os.path.exists(args.logs):
        os.makedirs(args.logs)
    cur_time = time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())
    log_file = os.path.join(args.logs, cur_time+'.log')

    logger = utils.setup_logger(distributed_rank=0, filename=log_file)

    if not os.path.isdir(cfg.CKPT_DIR):
        os.makedirs(cfg.CKPT_DIR)

    with open(os.path.join(cfg.CKPT_DIR, 'config.yaml'), 'w') as f:
        f.write("{}".format(cfg))

    main(cfg)
