import torch

import sys
sys.path.append('../')
import numpy as np

import torch

import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed


from args import args

import models

def set_gpu(args, model):
    

    # DataParallel will divide and allocate batch_size to all available GPUs
    print(f"=> Parallelizing on {args.multigpu} gpus")
    torch.cuda.set_device(args.multigpu[0])

    model = torch.nn.DataParallel(model, device_ids=args.multigpu).cuda(
            args.multigpu[0]
        )

    cudnn.benchmark = True

    return model

def get_model(args):
    if args.first_layer_dense:
        args.first_layer_type = "DenseConv"

    if args.structure == False:
        model = models.__dict__[args.arch]()

        # applying sparsity to the network
        if (
            args.conv_type != "DenseConv"
            and args.conv_type != "SampleSubnetConv"
            and args.conv_type != "ContinuousSparseConv"
        ):
        

            set_model_prune_rate(model, prune_rate=0.5)
            print(
                f"=> Rough estimate model params {sum(int(p.numel() * (1-args.prune_rate)) for n, p in model.named_parameters() if not n.endswith('scores'))}"
            )

        # freezing the weights if we are only doing subnet training
        # if args.freeze_weights:
        #     freeze_model_weights(model)
    else:
        model = models.__dict__[args.arch](args.cfg,args.honey)

    return model

def set_model_prune_rate(model, prune_rate):
    print(f"==> Setting prune rate of network to {prune_rate}")

    for n, m in model.named_modules():
        if hasattr(m, "set_prune_rate"):
            m.set_prune_rate(prune_rate)
            print(f"==> Setting prune rate of {n} to {prune_rate}")

def fetch_models(path, device):
    '''
    :param models_folder:
    :param num_models: the number of models to be load
    :param start_no: the start serial number from which loading  "num_models" models. 1-index
    :return: the top [num_models] models in models_folder
    '''
    target_models = []

    for j in range(100):

        if args.structure == False:
            model1 = path + '{}'.format(j) + '/checkpoints/model_best.pth'
            
            args.pretrained = model1
            model = get_model(args) 
    

            pretrained = torch.load(
                model1,
                map_location=torch.device("cuda:{}".format(args.multigpu[0])),
            )["state_dict"]
            print('===> successful loading the pretrained')
            model_state_dict = model.state_dict()

            for k, v in pretrained.items():
                if k not in model_state_dict or v.size() != model_state_dict[k].size():
                    print("IGNORE:", k)
            pretrained = {
                k: v
                for k, v in pretrained.items()
                if (k in model_state_dict and v.size() == model_state_dict[k].size())
            }

            model_state_dict.update(pretrained)
            model.load_state_dict(model_state_dict)
        else:
            model = get_model(args) 
            checkpoint = torch.load(path+"model_best{}.pth.tar".format(j))
            model.load_state_dict(checkpoint['state_dict'])
        target_models.append(model.to(device))
    return target_models



