import torch
import torch.nn as nn
import torch.optim as optim

from args import args
import utils.common as utils

import os
import time
import copy
import sys
import random
import numpy as np
import heapq
from data import cifar,svhn
from importlib import import_module
import models
checkpoint = utils.checkpoint(args)
device = torch.device(f"cuda:{args.multigpu[0]}") if torch.cuda.is_available() else 'cpu'
logger = utils.get_logger(os.path.join(args.job_dir + 'logger.log'))
loss_func = nn.CrossEntropyLoss()

conv_num_cfg = {
    'vgg16': 13,
    'resnet18' : 8
    }


# Data
print('==> Loading Data..')
if args.set == 'CIFAR10':
    loader = cifar.CIFAR10(args)
else:
    loader = svhn.svhn(args)


# Model
print('==> Loading Model..')
if args.arch == 'VGG_Bee':
    origin_model = models.__dict__[args.arch]("vgg16",[10]*13).to(device)
elif args.arch == 'resnet':
    origin_model = models.__dict__[args.arch]("resnet18",[10]*8).to(device)




ckpt = torch.load(args.ori_model, map_location=device)
origin_model.load_state_dict(ckpt['state_dict'])
oristate_dict = origin_model.state_dict()

print_freq = (256*50)//256

def generate_random_integers(_sum, n):  
    mean = _sum / n
    variance = int(0.25 * mean)

    min_v = 1
    max_v = 9
    array = [1] * n

    diff = _sum - n
    while diff > 0:
        a = random.randint(0, n - 1)
        if array[a] >= max_v:
            continue
        array[a] += 1
        diff -= 1
    return array



#load pre-train params
def load_vgg_honey_model(model, random_rule):
    #print(ckpt['state_dict'])
    global oristate_dict
    state_dict = model.state_dict()
    last_select_index = None #Conv index selected in the previous layer

    for name, module in model.named_modules():

        if isinstance(module, nn.Conv2d):

            oriweight = oristate_dict[name + '.weight']
            curweight = state_dict[name + '.weight']
            orifilter_num = oriweight.size(0)
            currentfilter_num = curweight.size(0)

            if orifilter_num != currentfilter_num and (random_rule == 'random_pretrain' or random_rule == 'l1_pretrain'):

                select_num = currentfilter_num
                if random_rule == 'random_pretrain':
                    select_index = random.sample(range(0, orifilter_num-1), select_num)
                    select_index.sort()
                else:
                    l1_sum = list(torch.sum(torch.abs(oriweight), [1, 2, 3]))
                    select_index = list(map(l1_sum.index, heapq.nlargest(currentfilter_num, l1_sum)))
                    select_index.sort()
                if last_select_index is not None:
                    for index_i, i in enumerate(select_index):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[name + '.weight'][index_i][index_j] = \
                                oristate_dict[name + '.weight'][i][j]
                else:
                    for index_i, i in enumerate(select_index):
                        state_dict[name + '.weight'][index_i] = \
                            oristate_dict[name + '.weight'][i]

                last_select_index = select_index

            else:
                state_dict[name + '.weight'] = oriweight
                last_select_index = None

    model.load_state_dict(state_dict)




def load_resnet_honey_model(model, random_rule):

    cfg = {'resnet18': [2,2,2,2],
           'resnet34': [3,4,6,3],
           'resnet50': [3,4,6,3],
           'resnet101': [3,4,23,3],
           'resnet152': [3,8,36,3]}

    
    state_dict = model.state_dict()
        
    current_cfg = cfg[args.cfg]
    last_select_index = None

    all_honey_conv_weight = []

    for layer, num in enumerate(current_cfg):
        layer_name = 'layer' + str(layer + 1) + '.'
        for k in range(num):
            if args.cfg == 'resnet18' or args.cfg == 'resnet34':
                iter = 2 #the number of convolution layers in a block, except for shortcut
            else:
                iter = 3
            for l in range(iter):
                conv_name = layer_name + str(k) + '.conv' + str(l+1)
                conv_weight_name = conv_name + '.weight'
                all_honey_conv_weight.append(conv_weight_name)
                oriweight = oristate_dict[conv_weight_name]
                curweight = state_dict[conv_weight_name]
                orifilter_num = oriweight.size(0)
                currentfilter_num = curweight.size(0)
                #logger.info('weight_num {}'.format(conv_weight_name))
                #logger.info('orifilter_num {}\tcurrentnum {}\n'.format(orifilter_num,currentfilter_num))
                #logger.info('orifilter  {}\tcurrent {}\n'.format(oristate_dict[conv_weight_name].size(),state_dict[conv_weight_name].size()))

                if orifilter_num != currentfilter_num and (random_rule == 'random_pretrain' or random_rule == 'l1_pretrain'):

                    select_num = currentfilter_num
                    if random_rule == 'random_pretrain':
                        select_index = random.sample(range(0, orifilter_num-1), select_num)
                        select_index.sort()
                    else:
                        l1_sum = list(torch.sum(torch.abs(oriweight), [1, 2, 3]))
                        select_index = list(map(l1_sum.index, heapq.nlargest(currentfilter_num, l1_sum)))
                        select_index.sort()
                    if last_select_index is not None:
                        for index_i, i in enumerate(select_index):
                            for index_j, j in enumerate(last_select_index):
                                state_dict[conv_weight_name][index_i][index_j] = \
                                    oristate_dict[conv_weight_name][i][j]
                    else:
                        for index_i, i in enumerate(select_index):
                            state_dict[conv_weight_name][index_i] = \
                                oristate_dict[conv_weight_name][i]  

                    last_select_index = select_index
                    #logger.info('last_select_index{}'.format(last_select_index)) 

                elif last_select_index != None:
                    for index_i in range(orifilter_num):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[conv_weight_name][index_i][index_j] = \
                                oristate_dict[conv_weight_name][index_i][j]
                    last_select_index = None

                else:
                    state_dict[conv_weight_name] = oriweight
                    last_select_index = None

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_name = name + '.weight'
            if conv_name not in all_honey_conv_weight:
                state_dict[conv_name] = oristate_dict[conv_name]

        elif isinstance(module, nn.Linear):
            state_dict[name + '.weight'] = oristate_dict[name + '.weight']
            state_dict[name + '.bias'] = oristate_dict[name + '.bias']

    #for param_tensor in state_dict:
        #logger.info('param_tensor {}\tType {}\n'.format(param_tensor,state_dict[param_tensor].size()))
    #for param_tensor in model.state_dict():
        #logger.info('param_tensor {}\tType {}\n'.format(param_tensor,model.state_dict()[param_tensor].size()))
 

    model.load_state_dict(state_dict)


def train(epoch, train_loader, model, criterion, optimizer, scheduler):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    data_time = utils.AverageMeter('Data', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')

    model.train()
    end = time.time()

    for param_group in optimizer.param_groups:
        cur_lr = param_group['lr']
    logger.info('learning_rate: ' + str(cur_lr))

    num_iter = len(train_loader)
    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        images = images.to(device)
        target = target.to(device)

        # compute outputy
        logits = model(images)
        loss = criterion(logits, target)

        # measure accuracy and record loss
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = images.size(0)
        losses.update(loss.item(), n)   #accumulated loss
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            logger.info(
                'Epoch[{0}]({1}/{2}): '
                'Loss {loss.avg:.4f} '
                'Prec@1(1,5) {top1.avg:.2f}, {top5.avg:.2f}'.format(
                    epoch, i, num_iter, loss=losses,
                    top1=top1, top5=top5))

    scheduler.step()

    return losses.avg, top1.avg, top5.avg

def validate(epoch, val_loader, model, criterion, args):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')

    # switch to evaluation mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            # compute output
            logits = model(images)
            loss = criterion(logits, target)

            # measure accuracy and record loss
            pred1, pred5 = utils.accuracy(logits, target, topk=(1, 5))
            n = images.size(0)
            losses.update(loss.item(), n)
            top1.update(pred1[0], n)
            top5.update(pred5[0], n)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        logger.info(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg





def main():
   
    start_epoch = 0
    best_acc = 0.0
    best_acc_top1 = 0.0

    best_top1_acc= 0
    

    # Model
    print('==> Building model..')
    if args.arch == 'VGG_Bee':
        honey1 = generate_random_integers(10,2)
        honey2 = generate_random_integers(10,2)
        honey3 = generate_random_integers(15,3)
        honey4 = generate_random_integers(30,6)
        honey  = honey1 + honey2 + honey3 + honey4
        model = models.__dict__[args.arch]("vgg16",honey).to(device)
        load_vgg_honey_model(model,args.random_rule)
    elif args.arch == 'resnet':
        honey1 = generate_random_integers(10,2)
        honey2 = generate_random_integers(10,2)
        honey3 = generate_random_integers(10,2)
        honey4 = generate_random_integers(10,2)
        honey  = honey1 + honey2 + honey3 + honey4
        model = models.__dict__[args.arch]("resnet",honey).to(device)
        load_resnet_honey_model(model,args.random_rule)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=30, gamma=0.1)
    
    for epoch in range(start_epoch):
        scheduler.step()

    # train the model
    epoch = start_epoch
    while epoch < args.epochs:
        train_obj, train_top1_acc,  train_top5_acc = train(epoch,  loader.train_loader, model, criterion, optimizer, scheduler)
        valid_obj, valid_top1_acc, valid_top5_acc = validate(epoch, loader.val_loader, model, criterion, args)

        # is_best = False
        if valid_top1_acc > best_top1_acc:
            best_top1_acc = valid_top1_acc
            is_best = True

            utils.save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_top1_acc': best_top1_acc,
                'optimizer' : optimizer.state_dict(),
                }, True, args.job_dir,0)

        epoch += 1
        logger.info("=>Best accuracy {:.3f}".format(best_top1_acc))

    np.save("honey.npy",honey)

if __name__ == '__main__':
    main()
