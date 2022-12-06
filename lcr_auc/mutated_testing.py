import sys
sys.path.append("../")
import torch
import os
from torch.utils.data import DataLoader

from ensemble_model import EnsembleModel
from utils import logging_util
from attacks.craft_adversarial_img import get_model,set_gpu,pretrained
import logging
from utils.data_manger import *
from attacks.attack_util import *
from importlib import import_module
from args import args

MAX_NUM_SAMPLES = 1000

from utils.model_manager import fetch_models



def step_mutated_vote(models_folder, model_name_list, target_samples, samples_folder, useAttackSeed=True,
                      dataloader=None):
    '''
    step=10,up to 100
    :param model_folder:
    :param model_name_list:
    :param target_samples:
    :param samples_folder:
    :return:
    '''

    for i, targe_sample in enumerate(target_samples):

        # i += 3 just for mnist4, mnist5

        if not dataloader:
            adv_file_path = os.path.join(samples_folder, targe_sample)
            torch.manual_seed(random_seed)
            dataset = MyDataset(root=adv_file_path,
                                transform=transforms.Compose([transforms.ToTensor(), normalize_mnist]))
            dataloader = DataLoader(dataset=dataset, shuffle=True)

        print('>>>Progress: Test attacked samples of {} '.format(targe_sample))
        logging.info('>>>Progress: Test attacked samples of {} '.format(targe_sample))

        # for num_models in range(10, 110, 10):
        for num_models in [100]:
            # to do
            # 1. for each seed model, select the top [num_models] models
            # 2. ensemble 5*[num_models] models

            num_seed_models = len(model_name_list)
            models_list = []
            for i2, seed_name in enumerate(model_name_list):
                if useAttackSeed:
                    models_list.extend(fetch_models(models_folder, num_models, seed_name))
                elif i != i2:
                    models_list.extend(fetch_models(models_folder, num_models, seed_name))

            logging.info('>>>Progress: {} models for {}'.format(len(models_list), targe_sample))
            print('>>>Progress: {} models for {}'.format(len(models_list), targe_sample))

            vote_model = EnsembleModel(models_list)
            logging.info('>>Test-Details-start-{}>>>{}'.format(num_seed_models * num_models, targe_sample))
            samples_filter(vote_model, dataloader, '{} >> {} '.format(len(models_list), targe_sample), size=-1,
                           show_progress=True)
            logging.info('>>Test-Details-end-{}>>>{}'.format(num_seed_models * num_models, targe_sample))


def inspect_adv_lale(real_labels, adv_labels, img_files):
    for real_label, adv_label, file_name in zip(real_labels, adv_labels, img_files):
        print ('real:{},adv:{},files:{}'.format(real_label, adv_label, file_name))


def batch_adv_tetsing(device, seed_data,
                      adv_folder,
                      mutated_models_path):
    if seed_data == 'mnist':
        normalization = normalize_mnist
        img_mode = 'L'  # 8-bit pixels, black and white
    elif seed_data == 'cifar10':
        normalization = normalize_cifar10
        img_mode = None
    elif seed_data == 'ilsvrc12':
        normalization = normalize_imgNet
        img_mode = None
    elif seed_data == 'svhn':
        normalization = normalize_svhn
        img_mode = None
    else:
        raise Exception('Unknown data soure!')

    # adv_type = parseAdvType(adv_folder)

    tf = transforms.Compose([transforms.ToTensor(), normalization])

    logging.info('>>>>>>>>>>>seed data:{},mutated_models:{}<<<<<<<<<<'.format(seed_data, mutated_models_path))
    mutated_models = fetch_models(mutated_models_path, device=device)
    ensemble_model = EnsembleModel(mutated_models)

    dataset = MyDataset(root=adv_folder, transform=tf, img_mode=img_mode)
    dataloader = DataLoader(dataset=dataset)
    logging.info(
        '>>>Progress: {} mutated models for {}, samples {}'.format(len(mutated_models), 'deepfool',
                                                                   adv_folder))
    logging.info('>>Test-Details-start-{}>>> {}>>>{}'.format(100, 'deepfool',seed_data))
    samples_filter(ensemble_model, dataloader, '{} >> {} '.format(100, 'deepfool'), size=-1,
                   show_progress=False, device=device, is_verbose=True)
    logging.info('>>Test-Details-end-{}>>>{}>>>{}'.format(100, 'deepfool',seed_data))


def batch_legitimate_testing(device, num_models, seed_data, raw_data_path, seed_model,
                             mutated_models_path,  use_train=True):
    if seed_data == 'mnist':
        data_type = DATA_MNIST
    elif seed_data == 'cifar10':
        data_type = DATA_CIFAR10
    elif seed_data == 'svhn':
        data_type = DATA_svhn
    data = load_natural_data(True,data_type, raw_data_path, use_train=use_train, seed_model=seed_model, device=device, MAX_NUM_SAMPLES=MAX_NUM_SAMPLES)

    logging.info(
        '>>>>>>>>>>>For {}({}) randomly choose {} with randomseed {}. mutated_models:{}<<<<<<<<<<'.format(
            seed_data,"Training" if use_train else "Testing",MAX_NUM_SAMPLES,random_seed,mutated_models_path))

    mutated_models = fetch_models(mutated_models_path, device=device)

    ensemble_model = EnsembleModel(mutated_models)
    logging.info(
        '>>>Progress: {} mutated models for normal samples, samples path: {}'.format(len(mutated_models),
                                                                                     raw_data_path))
    logging.info('>>Test-Details-start-{}>>>{}'.format(num_models, seed_data))
    samples_filter(ensemble_model, DataLoader(dataset=data), 'legitimate {} >>'.format(seed_data), size=-1,
                   show_progress=False, device=device, is_verbose=True)
    logging.info('>>Test-Details-end-{}>>>{}'.format(num_models, seed_data))


def batch_wl_testing(device, num_models, seed_data, raw_data_path, seed_model, mutated_models_path,
                        use_train=True):

    if seed_data == 'mnist':
        data_type = DATA_MNIST
    elif seed_data == 'cifar10':
        data_type = DATA_CIFAR10
    elif seed_data == 'svhn':
        data_type = DATA_svhn

    if data_type == DATA_svhn:
        _, dataset = load_svhn(raw_data_path, split=True, normalize=normalize_svhn)
    else:
        dataset, channel = load_data_set(data_type, raw_data_path, train=use_train)

    dataloader = DataLoader(dataset=dataset)

    wrong_labeled = samples_filter(seed_model, dataloader, return_type='adv', name='seed model', device=device,show_accuracy=False)
    data = datasetMutiIndx(dataset, [idx for idx, _, _ in wrong_labeled])
    wrong_labels = [wrong_label for idx, true_label, wrong_label in wrong_labeled]
    data = TensorDataset(data.tensors[0], data.tensors[1], torch.LongTensor(wrong_labels))

    logging.info(
        '>>>>>>>>>>>For {}({}),mutated Models Path: {} <<<<<<<<<<'.format(
            seed_data,"Training" if use_train else "Testing",mutated_models_path))

    mutated_models = fetch_models(mutated_models_path, device=device)

    ensemble_model = EnsembleModel(mutated_models)
    logging.info(
        '>>>Progress: {} mutated models for wl samples, '.format(len(mutated_models)))
    logging.info('>>Test-Details-start-{}>>> wrong labeled of {}'.format(num_models, seed_data))
    samples_filter(ensemble_model, DataLoader(dataset=data), 'legitimate {} >>'.format(seed_data), size=-1,
                   show_progress=False, device=device, is_verbose=True)
    logging.info('>>Test-Details-end-{}>>> wrong labeled of {}'.format(num_models, seed_data))



def run():


    device = args.multigpu[0]

    if args.arch == 'vgg_cifar':
        model = import_module(f'model.{args.arch}').VGG(args.cfg).to(device)
    elif args.arch == 'resnet_cifar':
        model = import_module(f'model.{args.arch}').resnet(args.cfg).to(device)
    ckpt = torch.load(args.ori_model, map_location=device)
    model.load_state_dict(ckpt['state_dict'])

        
    if args.testType == 'adv':
        batch_adv_tetsing(device=device,
                          seed_data=args.set,
                          adv_folder=args.testSamplesPath,
                          mutated_models_path=args.prunedModelsPath,
                          )

    elif args.testType == 'normal':
        
        testSamplesPath ='../dataset/{}'.format(args.set)

        batch_legitimate_testing(device=device,num_models=100, seed_data=args.set, raw_data_path=testSamplesPath,
                                 seed_model = model,mutated_models_path=args.prunedModelsPath,use_train=False)


    elif args.testType == 'wl':

        testSamplesPath = '../dataset/{}'.format(args.set)
        batch_wl_testing(device=device,num_models=100, seed_data=args.set, raw_data_path=testSamplesPath,
                        seed_model = model,mutated_models_path=args.prunedModelsPath,use_train=False)
    else:
        raise Exception('Unknown test type:{}'.format(args.testType))


if __name__ == '__main__':
    logging_util.setup_logging()
    run()
