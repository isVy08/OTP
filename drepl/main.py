import os
import argparse
from configs.defaults import get_cfgs_defaults
import torch

from trainer import OTP_Trainer
from util import set_seeds, get_loader


def arg_parse():
    parser = argparse.ArgumentParser(
            description="main.py")
    parser.add_argument(
        "-c", "--config_file", default="", help="config file")
    parser.add_argument(
        "-ts", "--timestamp", default="", help="saved path (random seed + date)")
    parser.add_argument(
        "-rs", "--resume", default="", help="saved path (random seed + date)")
    parser.add_argument(
        "--save", action="store_true", help="save trained model")
    parser.add_argument(
        "--dbg", action="store_true", help="print losses per epoch")
    parser.add_argument(
        "--gpu", default="0", help="index of gpu to be used")
    parser.add_argument(
        "--seed", type=int, default=0, help="seed number for randomness")
    args = parser.parse_args()
    return args


def load_config(args):
    cfgs = get_cfgs_defaults()
    config_path = os.path.join(os.path.dirname(__file__), "configs", args.config_file)
    print(config_path)
    cfgs.merge_from_file(config_path)
    cfgs.train.seed = args.seed
    cfgs.flags.save = args.save
    cfgs.flags.noprint = not args.dbg
    cfgs.path_data = cfgs.path
    cfgs.path = os.path.join(cfgs.path, cfgs.path_specific)

    cfgs.freeze()
    flgs = cfgs.flags
    return cfgs, flgs


if __name__ == "__main__":
    print("main.py")
    
    ## Experimental setup
    args = arg_parse()
    if args.gpu != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    cfgs, flgs = load_config(args)
    print("[Checkpoint path] "+cfgs.path)
    print(cfgs)
    
    ## Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seeds(args.seed)

    ## Data loader
    train_loader, val_loader, test_loader = get_loader(
        cfgs.dataset.name, cfgs.path_dataset, cfgs.train.bs, cfgs.nworker)
    print("Complete dataload")

    ## Trainer
    print("=== {} ===".format(cfgs.model.name.upper()))
    trainer = OTP_Trainer(cfgs, flgs, train_loader, val_loader, test_loader)

    ## Main
    if args.timestamp == "":
        if args.resume != "":
            trainer.load(args.resume)
            res_test = trainer.test()
        trainer.main_loop()
    if flgs.save:
        trainer.load(args.timestamp)
        print("Best models were loaded!!")
        #res_test = trainer.test(mode="train")
        res_test = trainer.test(mode="test")
        # print("Plotting")
        # trainer.tsne('{}_{}_{}.pdf'.format(cfgs.dataset.name, cfgs.model.param_var_q, cfgs.quantization.size_dict))

