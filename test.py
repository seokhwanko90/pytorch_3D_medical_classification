import torch
import torch.nn
import torchvision.transforms as transforms
import numpy as np
from models._3d import resnet
from dataset import DatasetList3d
import argparse
from config import _C as cfg
import glob
import os

device = torch.device('cuda')

def main(cfg):
    # Data loading code

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    #'''
    testset = DatasetList3d(cfg.DATASET.list_test, cfg.DATASET.md_classes, transforms.Compose([
            transforms.RandomResizedCrop(cfg.DATASET.spatial_size),
            transforms.ToTensor(),
            normalize,
        ]))
    #'''
    dataloader = dict()
    dataloader['test'] = torch.utils.data.DataLoader(testset,
        batch_size=1, shuffle=False,
        num_workers=1, pin_memory=True)

    num_classes = cfg.DATASET.num_class

    #"""
    net = resnet.resnet50(num_classes=num_classes, shortcut_type='B', spatial_size=cfg.DATASET.spatial_size, sample_count=cfg.DATASET.sample_count).to(device)
    #net = resnet.resnet101(num_classes=2, shortcut_type='B', spatial_size=spatial_size, sample_count=sample_count).to(device)
    net = torch.nn.DataParallel(net, device_ids=None)
    #"""


    model_path = os.path.join(cfg.CKPT_DIR, cfg.TEST.checkpoint)
    if cfg.TRAIN.ckpt_interval < 0:
        model_path = glob.glob(os.path.join(cfg.CKPT_DIR,'model_*_best.pth'))[0]

    print (model_path)

    net.load_state_dict(torch.load(model_path, map_location=device))
    # move model to the right device
    net.to(device)

    dataloader_ts = dataloader['test']
    test(net, dataloader_ts)

    print('Finished Test')

def make_npy(arr, save_path):
    d = 0
    np.save(save_path, np.array(arr))

def test(net, dataloader):
    net.eval()

    corr = 0
    arr_prd = []
    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels, info = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            # forward
            outputs = net(inputs)

            softmax = torch.nn.Softmax()
            s_outputs = softmax(outputs)

            _, predicted = torch.max(outputs, 1)

            if predicted == labels:
                corr += 1

            arr_prd += s_outputs.tolist()
            '''
            acc1 = utils.compute_accuracy(
                                          outputs,
                                          labels,
                                          augmentation=False,
                                          topk=(1, ))
            print (acc1)
            '''

            print('[%d] TEST %s\ts_outputs: %s pred1: %d gth: %d' %
                  (i + 1, info, s_outputs.tolist(), predicted, labels))

            debug = 0

    print (str(corr)+'/'+str(i+1))
    print (str(corr/(i+1)))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="PyTorch 3D Classification Testing"
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

    main(cfg)
