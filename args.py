import argparse
import sys
import yaml

from configs import parser as _parser

args = None


def parse_arguments():
    parser = argparse.ArgumentParser(description="PyTorch Training and Detecting")

    # Parameters for training
    parser.add_argument(
        "--data", help="path to dataset base directory", default="/dataset"
    )
    parser.add_argument("--optimizer", help="Which optimizer to use", default="sgd")
    parser.add_argument("--set", help="name of dataset", type=str, default="CIFAR10")
    parser.add_argument(
        "-a", "--arch", metavar="ARCH", default="cResNet18", help="model architecture"
    )
    parser.add_argument(
        "--config", help="Config file to use (see configs dir)", default=None
    )
    parser.add_argument(
        "--log-dir", help="Where to save the runs. If None use ./runs", default=None
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=20,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 20)",
    )
    parser.add_argument(
        "--epochs",
        default=100,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--start-epoch",
        default=None,
        type=int,
        metavar="N",
        help="manual epoch number (useful on restarts)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=256,
        type=int,
        metavar="N",
        help="mini-batch size (default: 256), this is the total "
        "batch size of all GPUs on the current node when "
        "using Data Parallel or Distributed Data Parallel",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.1,
        type=float,
        metavar="LR",
        help="initial learning rate",
        dest="lr",
    )
    parser.add_argument(
        "--warmup_length", default=0, type=int, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="momentum"
    )
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "-p",
        "--print-freq",
        default=10,
        type=int,
        metavar="N",
        help="print frequency (default: 10)",
    )
    parser.add_argument("--num-classes", default=10, type=int)
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "-e",
        "--evaluate",
        dest="evaluate",
        action="store_true",
        help="evaluate model on validation set",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        default=None,
        type=str,
        help="use pre-trained model",
    )
    parser.add_argument(
        "--seed", default=None, type=int, help="seed for initializing training. "
    )
    parser.add_argument(
        "--multigpu",
        default=None,
        type=lambda x: [int(a) for a in x.split(",")],
        help="Which GPUs to use for multigpu training",
    )

    # Learning Rate Policy Specific
    parser.add_argument(
        "--lr-policy", default="constant_lr", help="Policy for the learning rate."
    )
    parser.add_argument(
        "--multistep-lr-adjust", default=30, type=int, help="Interval to drop lr"
    )
    parser.add_argument(
        "--multistep-lr-gamma", default=0.1, type=int, help="Multistep multiplier"
    )
    parser.add_argument(
        "--name", default=None, type=str, help="Experiment name to append to filepath"
    )
    parser.add_argument(
        "--save_every", default=-1, type=int, help="Save every ___ epochs"
    )
    parser.add_argument(
        "--prune-rate",
        default=0.5,
        help="Amount of pruning to do during sparse training",
        type=float,
    )
    parser.add_argument(
        "--low-data", default=1, help="Amount of data to use", type=float
    )
    parser.add_argument(
        "--width-mult",
        default=1.0,
        help="How much to vary the width of the network.",
        type=float,
    )
    parser.add_argument(
        "--nesterov",
        default=False,
        action="store_true",
        help="Whether or not to use nesterov for SGD",
    )
    parser.add_argument(
        "--random-subnet",
        action="store_true",
        help="Whether or not to use a random subnet when fine tuning for lottery experiments",
    )
    parser.add_argument(
        "--one-batch",
        action="store_true",
        help="One batch train set for debugging purposes (test overfitting)",
    )
    parser.add_argument(
        "--conv-type", type=str, default=None, help="What kind of sparsity to use"
    )
    parser.add_argument(
        "--freeze-weights",
        action="store_true",
        help="Whether or not to train only subnet (this freezes weights)",
    )
    parser.add_argument("--mode", default="fan_in", help="Weight initialization mode")
    parser.add_argument(
        "--nonlinearity", default="relu", help="Nonlinearity used by initialization"
    )
    parser.add_argument("--bn-type", default=None, help="BatchNorm type")
    parser.add_argument(
        "--init", default="kaiming_normal", help="Weight initialization modifications"
    )
    parser.add_argument(
        "--no-bn-decay", action="store_true", default=False, help="No batchnorm decay"
    )
    parser.add_argument(
        "--scale-fan", action="store_true", default=False, help="scale fan"
    )
    parser.add_argument(
        "--first-layer-dense", action="store_true", help="First layer dense or sparse"
    )
    parser.add_argument(
        "--last-layer-dense", action="store_true", help="Last layer dense or sparse"
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        help="Label smoothing to use, default 0.0",
        default=None,
    )
    parser.add_argument(
        "--first-layer-type", type=str, default=None, help="Conv type of first layer"
    )
    parser.add_argument(
        "--trainer", type=str, default="default", help="cs, ss, or standard training"
    )
    parser.add_argument(
        "--score-init-constant",
        type=float,
        default=None,
        help="Sample Baseline Subnet Init",
    )

    parser.add_argument(
        "--iter_num", help="iter_num used in optimal graph training times during the running ", default=100, type=int
    )  

    parser.add_argument(
    '--cfg',
    type=str,
    default='vgg16',
    help='Detail architecuture of model. default:vgg16'
    )

    parser.add_argument(
    '--ori_model',
    type=str,
    help='The directory where your ori model stored.'
    )

    parser.add_argument(
    '--job_dir',
    type=str,
    default='experiments/',
    help='The directory where the summaries will be stored. default:./experiments'
    )

    parser.add_argument(
    '--random_rule',
    type=str,
    default='random_pretrain',
    help='Weight initialization criterion after random clipping. default:default optional:default,random_pretrain,l1_pretrain'
    )

    # Parameters for generating adversarial samples
    parser.add_argument(
        "--savePath", help="The path where the adversarial samples to be stored", type=str
    )

    parser.add_argument("--attackType", type=str,
                        help="four attacks are available: fgsm, jsma, deepfool, cw, localsearch, ILA, FIA"
    )

    # Parameters for calculating lcr
    parser.add_argument("--honey", type=str, help="The files path of honey")

    parser.add_argument("--testType", type=str,
                        help="Tree types are available: [adv], advesarial data; [normal], test on normal data; [wl],test on wrong labeled data",
    )

    parser.add_argument("--prunedModelsPath", type=str,
                        help="The path of pruned models",
    )

    parser.add_argument("--testSamplesPath", type=str,
                        help="The path of adversarial samples",
    )

    parser.add_argument("--logPath", type=str, help="The files path of batch testing results")

    parser.add_argument("--maxModelsUsed", type=int,
                        help="Total mutated models are used to yield the label change rate(lcr)")

    parser.add_argument("--isAdv", type=str, help="True if the samples are adversarial, otherwise,false")

    parser.add_argument("--nrLcrPath", type=str,
                        help="The lcr list of normal samples. This is just for the auc computing")

    parser.add_argument("--lcrSavePath", type=str, help="The path to save the lcr list")
    
    # Parameter for detecting
    parser.add_argument("--threshold", type=float,
                        help="The lcr_auc of normal samples. The value is equal to: avg+99%confidence.")

    args = parser.parse_args()

    # Allow for use from notebook without config file
    if len(sys.argv) > 1:
        get_config(args)

    return args


def get_config(args):
    # get commands from command line
    override_args = _parser.argv_to_vars(sys.argv)

    # load yaml file
    yaml_txt = open(args.config).read()

    # override args
    loaded_yaml = yaml.load(yaml_txt, Loader=yaml.FullLoader)
    for v in override_args:
        loaded_yaml[v] = getattr(args, v)

    print(f"=> Reading YAML config from {args.config}")
    args.__dict__.update(loaded_yaml)


def run_args():
    global args
    if args is None:
        args = parse_arguments()


run_args()
