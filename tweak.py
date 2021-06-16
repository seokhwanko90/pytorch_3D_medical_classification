import torch
import torch.nn
import torchvision.transforms as transforms
import numpy as np
from models._3d import resnet
import utils
from dataset import DatasetList3d
import argparse
from config import _C as cfg
import glob
import math
import os

'''
Tweak method by concatenating additional parameters in FC layers
'''

device = torch.device('cuda')
TRAIN_TEST_CFG = 'test'

def main(cfg):
    # Data loading code

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    #'''
    trainset = DatasetList3d(cfg.DATASET.list_train, cfg.DATASET.md_classes, transforms.Compose([
        transforms.RandomResizedCrop(cfg.DATASET.spatial_size),
        transforms.ToTensor(),
        normalize,
    ]))
    valset = DatasetList3d(cfg.DATASET.list_val, cfg.DATASET.md_classes, transforms.Compose([
        transforms.RandomResizedCrop(cfg.DATASET.spatial_size),
        transforms.ToTensor(),
        normalize,
    ]))
    testset = DatasetList3d(cfg.DATASET.list_test, cfg.DATASET.md_classes, transforms.Compose([
        transforms.RandomResizedCrop(cfg.DATASET.spatial_size),
        transforms.ToTensor(),
        normalize,
    ]))
    #'''
    dataloader = dict()

    dataloader['train'] = torch.utils.data.DataLoader(trainset,
                                                      batch_size=cfg.TRAIN.tr_batchsize,
                                                      shuffle=False,
                                                      num_workers=cfg.TRAIN.workers,
                                                      pin_memory=True)
    dataloader['val'] = torch.utils.data.DataLoader(valset,
                                                    batch_size=cfg.VAL.vl_batchsize,
                                                    shuffle=False,
                                                    num_workers=cfg.TRAIN.workers,
                                                    pin_memory=True)
    dataloader['test'] = torch.utils.data.DataLoader(testset,
                                                     batch_size=cfg.TEST.ts_batchsize,
                                                     shuffle=False,
                                                     num_workers=cfg.TRAIN.workers,
                                                     pin_memory=True)


    cfg.TRAIN.min_loss = [1000.0, 1000.0]
    cfg.TRAIN.tmp_loss = [1000.0, 1000.0]
    cfg.TRAIN.min_loss_cfg = [False, False, 1]

    cfg.TRAIN.best_acc = [0.0, 0.0]
    cfg.TRAIN.tmp_acc = [0.0, 0.0]
    cfg.TRAIN.best_acc_cfg = [False, False, 1]

    cfg.TRAIN.best_loss_acc = [1000.0, 0.0]
    cfg.TRAIN.tmp_loss_acc = [1000.0, 0.0]
    cfg.TRAIN.best_loss_acc_cfg = [False, False, 1]

    cfg.TRAIN.get_loss = [0.0, 0.0]

    org_net = resnet.resnet50(num_classes=2,
                          shortcut_type='B',
                          spatial_size=cfg.DATASET.spatial_size,
                          sample_count=cfg.DATASET.sample_count).to(device)

    #t_net = t_Resnet(org_net, spatial_size=cfg.DATASET.spatial_size,
    #                      sample_count=cfg.DATASET.sample_count)
    fc_net = fc_Resnet(org_net, spatial_size=cfg.DATASET.spatial_size,
                          sample_count=cfg.DATASET.sample_count)


    #t_net = torch.nn.DataParallel(t_net, device_ids=None)
    fc_net = torch.nn.DataParallel(fc_net, device_ids=None)

    model_path = os.path.join(cfg.CKPT_DIR, cfg.TEST.checkpoint)
    if cfg.TRAIN.ckpt_interval < 0:
        model_path = glob.glob(os.path.join(cfg.CKPT_DIR, 'model_*_best.pth'))[0]
    print(model_path)
    #t_net.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    fc_net.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    d = 0

    #add_params = (torch.ones(512).cuda() * 1)/2


    if TRAIN_TEST_CFG == 'train':
        print ('===> Train mode')

        add_params_path = './data/params.npy'
        params_len = 25
        add_params = np.load(add_params_path, allow_pickle=True).tolist()

        dataloader_tr = dataloader['train']
        dataloader_vl = dataloader['val']

        fc_dataloader_tr = get_fc_data(fc_net, dataloader_tr, add_params)
        fc_dataloader_vl = get_fc_data(fc_net, dataloader_vl, add_params)

        rg_model = Regression(2048+params_len, 2)
        rg_model.cuda()
        rg_criterion = torch.nn.CrossEntropyLoss()
        rg_optimizer = torch.optim.SGD(rg_model.parameters(),
                                       lr=0.0001,
                                       momentum=0.9,
                                       weight_decay=1e-4)

        rg_set = (rg_model, rg_criterion, rg_optimizer)

        for epoch in range(2000):
            tweak(rg_set, fc_dataloader_tr, epoch+1, cfg)
            val(rg_set, fc_dataloader_vl, epoch+1, cfg)

            #print(cfg.TRAIN.best_loss_acc_cfg)
            print(cfg.TRAIN.best_acc_cfg)
            print(cfg.TRAIN.best_acc)

            checkpoint(rg_set[0], cfg, epoch + 1)


    elif TRAIN_TEST_CFG == 'test':
        print ('===> Test mode')

        add_params = []

        dataloader_ts = dataloader['test']
        fc_dataloader_ts = get_fc_data(fc_net, dataloader_ts, add_params)

        #"""
        rg_model = t25_Regression(2048+25, 2)
        rg_criterion = torch.nn.CrossEntropyLoss()
        rg_optimizer = torch.optim.SGD(rg_model.parameters(),
                                       lr=0.0001,
                                       momentum=0.9,
                                       weight_decay=1e-4)

        test_model_path = os.path.join(cfg.CKPT_DIR, cfg.TEST.checkpoint)
        if cfg.TRAIN.ckpt_interval < 0:
            test_model_path = glob.glob(os.path.join(cfg.CKPT_DIR, 'tw_*_best.pth'))[0]
        print(test_model_path)

        rg_model.load_state_dict(torch.load(test_model_path, map_location=device), strict=False)
        rg_model.cuda()

        rg_set = (rg_model, rg_criterion, rg_optimizer)

        test(rg_set, fc_dataloader_ts, cfg)
        #"""

        d = 0

    print('Finished Test')

def deactivate_batchnorm(m):
    if isinstance(m, torch.nn.BatchNorm3d):
        m.eval()

def get_fc_data(net, dataloader, add_params=[]):
    net.eval()
    fc_data = []
    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels, info = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward
            outputs = net(inputs)

            new_fc = []

            idx = info[0][info[0][:info[0].rfind('/nifti')].rfind('/')+1:info[0].rfind('/nifti')]
            if len(add_params):
                params = torch.as_tensor(add_params[idx].astype(np.float32)).cuda()
            else:
                params = torch.as_tensor([]).cuda()

            for c in outputs:
                new_fc.append(torch.cat((c, params)))
            new_fc = torch.stack(new_fc, dim=0)

            fc_data.append((new_fc, labels, info))

    return fc_data

def make_npy(fc_data, save_path):
    d = 0
    new_arr = []
    for data, label, info in fc_data:
        new_arr.append(data[0].tolist())
        print(info[0] + '\t' + str(data[0].tolist()))

    np.save(save_path, np.array(new_arr))

def get_loss(loss, cfg, idx):
    cfg.TRAIN.get_loss[idx] = loss

def get_min_loss(loss, cfg, idx):
    if loss <= cfg.TRAIN.min_loss[idx]:
        cfg.TRAIN.tmp_loss[idx] = loss
        cfg.TRAIN.min_loss_cfg[idx] = True

def get_best_acc(acc, cfg, idx):
    if acc >= cfg.TRAIN.best_acc[idx]:
        cfg.TRAIN.tmp_acc[idx] = acc
        cfg.TRAIN.best_acc_cfg[idx] = True

def get_best_loss_acc(loss_acc, cfg, idx):
    loss, acc = loss_acc
    if loss < cfg.TRAIN.best_loss_acc[0]:
        cfg.TRAIN.tmp_loss_acc[0] = loss
        cfg.TRAIN.best_loss_acc_cfg[0] = True

    if acc > cfg.TRAIN.best_loss_acc[1]:
        cfg.TRAIN.tmp_loss_acc[1] = acc
        cfg.TRAIN.best_loss_acc_cfg[1] = True

def checkpoint(net, cfg, epoch):
    if cfg.TRAIN.ckpt_interval < 0:
        #"""
        if abs(cfg.TRAIN.get_loss[0]-cfg.TRAIN.get_loss[1]) <= 0.2 and \
                cfg.TRAIN.best_acc_cfg[0] and cfg.TRAIN.best_acc_cfg[1] and \
                -cfg.TRAIN.ckpt_interval <= epoch:
            if -cfg.TRAIN.ckpt_interval != epoch:
                print('removing previous model...')
                os.remove('{}/tw_model_epoch_{}_best.pth'\
                          .format(cfg.CKPT_DIR, cfg.TRAIN.best_acc_cfg[2]))
                cfg.TRAIN.best_acc_cfg[2] = epoch

            print('Saving best model...')
            dict_model = net.state_dict()
            torch.save(
                dict_model,
                '{}/tw_model_epoch_{}_best.pth'.format(cfg.CKPT_DIR, epoch))

            cfg.TRAIN.best_acc[0] = cfg.TRAIN.tmp_acc[0]
            cfg.TRAIN.best_acc[1] = cfg.TRAIN.tmp_acc[1]


        cfg.TRAIN.best_acc_cfg[0] = False
        cfg.TRAIN.best_acc_cfg[1] = False
        #"""

    elif epoch % cfg.TRAIN.ckpt_interval == 0 or epoch == cfg.TRAIN.tr_num_epochs:
        print('Saving checkpoints...')
        dict_model = net.state_dict()
        torch.save(
            dict_model,
            '{}/tw_model_epoch_{}.pth'.format(cfg.CKPT_DIR, epoch))


def tweak(rg_set, dataloader, epoch, tmp_cfg):
    rg_model, rg_criterion, rg_optimizer = rg_set
    rg_model.train()

    cum_loss = 0.0
    cum_acc1 = 0.0
    for i, data in enumerate(dataloader, 0):
        inputs, labels, info = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        rg_optimizer.zero_grad()  # Manually zero the gradient buffers
        new_outputs = rg_model(inputs)
        loss = rg_criterion(new_outputs, labels)  # Compute the loss given the predicted label
        # and actual label
        loss.backward(retain_graph=True)  # Compute the error gradients
        rg_optimizer.step()

        softmax = torch.nn.Softmax()
        new_outputs = softmax(new_outputs)
        s_outputs = ''
        _, predicted = torch.max(new_outputs, 1)

        #'''
        acc1, acc5 = utils.compute_accuracy(
                                      new_outputs,
                                      labels,
                                      augmentation=False,
                                      topk=(1, 2))
        #'''

        if epoch == -1:
            print('[%d] TWEAK %s\toutputs: %s s_outputs: %s pred1: %d gth: %d' %
                  (i + 1, info, str(new_outputs),s_outputs, predicted, labels))

        debug = 0
        cum_loss = cum_loss + loss
        cum_acc1 = cum_acc1 + acc1

    avg_loss = cum_loss / (i+1)
    avg_acc = cum_acc1 / (i+1)
    print("[TRAIN] Epoch {}, avg_loss :{}, avg_acc1: {}".format(
        epoch, float(avg_loss), float(avg_acc)))

    #get_min_loss(float(avg_loss), cfg, 0)
    get_best_acc(float(avg_acc), cfg, 0)
    get_loss(float(cum_loss/(i+1)), cfg, 0)


def val (rg_set, dataloader, epoch, tmp_cfg):
    rg_model, rg_criterion, rg_optimizer = rg_set
    rg_model.eval()

    cum_loss = 0.0
    cum_acc1 = 0.0
    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            inputs, labels, info = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            new_outputs = rg_model(inputs)
            loss = rg_criterion(new_outputs, labels)  # Compute the loss given the predicted label

            softmax = torch.nn.Softmax()
            new_outputs = softmax(new_outputs)
            s_outputs = ''
            _, predicted = torch.max(new_outputs, 1)

            #'''
            acc1, acc5 = utils.compute_accuracy(
                                          new_outputs,
                                          labels,
                                          augmentation=False,
                                          topk=(1, 2))
            #print (acc1)
            #'''

            if epoch == -1:
                print('[%d] TWEAK %s\toutputs: %s s_outputs: %s pred1: %d gth: %d' %
                      (i + 1, info, str(new_outputs),s_outputs, predicted, labels))

            debug = 0
            cum_loss = cum_loss + loss
            cum_acc1 = cum_acc1 + acc1

    print("[VAL] Epoch {}, avg_loss :{}, avg_acc1: {}".format(
        epoch, float(cum_loss/(i+1)), float(cum_acc1/(i+1))))

    #get_min_loss(float(cum_loss/(i+1)), cfg, 1)
    get_best_acc(float(cum_acc1/(i+1)), cfg, 1)
    get_loss(float(cum_loss/(i+1)), cfg, 1)
    #get_best_loss_acc((float(cum_loss/(i+1)), float(cum_acc1/(i+1))), cfg, 1)

def test(rg_set, dataloader, tmp_cfg):
    rg_model, rg_criterion, rg_optimizer = rg_set
    rg_model.eval()

    corr = 0
    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            inputs, labels, info = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            new_outputs = rg_model(inputs)

            softmax = torch.nn.Softmax()
            new_outputs = softmax(new_outputs)

            _, predicted = torch.max(new_outputs, 1)

            if predicted == labels:
                corr += 1
            '''
            acc1, acc5 = utils.compute_accuracy(
                                          new_outputs,
                                          labels,
                                          augmentation=False,
                                          topk=(1, 2))
            #print (acc1)
            '''

            print('[%d] TEST %s\toutputs: %s pred1: %d gth: %d' %
                  (i + 1, info, new_outputs.tolist(), predicted, labels))

    print (str(corr)+'/'+str(i+1))
    print (str(corr/(i+1)))


class t_Resnet(torch.nn.Module):
    def __init__(self, orig_resnet, spatial_size, sample_count):
        super(t_Resnet, self).__init__()

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu = orig_resnet.relu
        self.maxpool = orig_resnet.maxpool

        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

        last_count = int(math.ceil(sample_count / 16))
        last_size = int(math.ceil(spatial_size / 32))
        self.avgpool = torch.nn.AvgPool3d(
            (last_count, last_size, last_size), stride=1)
        self.fc = torch.nn.Linear(512 * 4, 2)
        self.fc_append = torch.nn.Linear(512 * 4 + 512, 2)

        self.add_params = torch.ones(512).cuda() * 1

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        d = 0

        return x

class fc_Resnet(torch.nn.Module):
    def __init__(self, orig_resnet, spatial_size, sample_count):
        super(fc_Resnet, self).__init__()

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu = orig_resnet.relu
        self.maxpool = orig_resnet.maxpool

        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

        last_count = int(math.ceil(sample_count / 16))
        last_size = int(math.ceil(spatial_size / 32))
        self.avgpool = torch.nn.AvgPool3d(
            (last_count, last_size, last_size), stride=1)

        self.fc = torch.nn.Linear(512 * 4, 2)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x


class Regression(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.linear_1 = torch.nn.Linear(input_size, int(input_size/2))
        self.linear_2 = torch.nn.Linear(int(input_size/2), num_classes)

    def forward(self, x):
        x = self.linear_1(x)
        out = self.linear_2(x)

        return out

class t25_Regression(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.linear_0 = torch.nn.Linear(input_size-25, input_size)
        self.linear_1 = torch.nn.Linear(input_size, int(input_size/2))
        self.linear_2 = torch.nn.Linear(int(input_size/2), num_classes)

    def forward(self, x):
        x = self.linear_0(x)
        x = self.linear_1(x)
        out = self.linear_2(x)

        return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="PyTorch 3D Classification tweak"
    )
    parser.add_argument(
        "--cfg",
        default="config/DCM_RP2-resnet50_3d_CV5_ps_rsp224.yaml",
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

    main(cfg)
