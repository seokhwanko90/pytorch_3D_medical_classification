import torch


def compute_accuracy(outputs, targets, augmentation, topk=(1, )):
    if augmentation:
        accs = accuracy(outputs, targets, topk)
    else:
        accs = accuracy(outputs, targets, topk)
    return accs


def accuracy(outputs, targets, topk=(1, )):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = targets.size(0)

        _, pred = outputs.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1 / batch_size))
    return res
