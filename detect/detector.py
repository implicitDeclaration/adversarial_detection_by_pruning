import sys
sys.path.append('../')
import torch
import numpy as np

from attacks.attack_util import *
from args import args
import models

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

def fetch_single_model(path, device,t):

    if args.structure == False:
            model1 = path + '{}'.format(t) + '/checkpoints/model_best.pth'
            
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
        checkpoint = torch.load(path+"model_best{}.pth.tar".format(t))
        model.load_state_dict(checkpoint['state_dict'])

    return model.to(device)



class Detector(object):
    '''
        A statistical detector for adversaries
        :param alpha : error bound of false negative
        :param beta : error bound of false positive
        :param sigma : size of indifference region
        :param kappa_nor : ratio of label change of a normal input
        :param mu : hyper parameter reflecting the difference between kappa_nor and kappa_adv
    '''

    def __init__(self, threshold, sigma, beta, alpha, max_mutated_numbers, data_type,
                 device='cpu', models_folder=None):
        self.threshold = threshold
        self.sigma = sigma
        self.beta = beta
        self.alpha = alpha
        self.device = device
        self.data_type = data_type
        self.models_folder = models_folder
        self.max_mutated_numbers = max_mutated_numbers
        self.start_no = 1

        if data_type == DATA_MNIST:
            self.max_models_in_memory = self.max_mutated_numbers
            # self.mutated_models = fetch_models(models_folder, device=device)
        else:
            self.max_models_in_memory = 20
            # self.mutated_models = fetch_models(models_folder, device=device)
            self.start_no += self.max_models_in_memory

    def calculate_sprt_ratio(self, c, n):
        '''
        :param c: number of model which lead to label changes
        :param n: total number of mutations
        :return: the sprt ratio
        '''
        p1 = self.threshold + self.sigma
        p0 = self.threshold - self.sigma

        return c * np.log(p1 / p0) + (n - c) * np.log((1 - p1) / (1 - p0))


    def detect(self, img, origi_label):
        '''
        just judge img is an adversarial sample or not
        :param img: the adv sample
        :param origi_label: the adv label of the img
        :return:
        '''
        img = img.to(self.device)
        accept_pr = np.log((1 - self.beta) / self.alpha)
        deny_pr = np.log(self.beta / (1 - self.alpha))

        if isinstance(origi_label, torch.Tensor):
            origi_label = origi_label.item()
        stop = False
        deflected_mutated_model_count = 0

        total_mutated_model_count = 0
        while (not stop):
            total_mutated_model_count += 1
            if total_mutated_model_count > self.max_mutated_numbers:
                return False, deflected_mutated_model_count, total_mutated_model_count
            mutated_model = fetch_single_model(self.models_folder,device=self.device,t=total_mutated_model_count-1)
            mutated_model.eval()
            new_score = mutated_model(img)
            new_lable = torch.argmax(new_score.cpu()).item()
            pr = self.calculate_sprt_ratio(deflected_mutated_model_count, total_mutated_model_count)
            if new_lable != origi_label:
                deflected_mutated_model_count += 1
                # pr = self.calculate_sprt_ratio(deflected_mutated_model_count, total_mutated_model_count)
                if pr >= accept_pr:
                    return True, deflected_mutated_model_count, total_mutated_model_count
                if pr <= deny_pr:
                    return False, deflected_mutated_model_count, total_mutated_model_count

