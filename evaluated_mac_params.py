import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4,7'
import argparse
import json
import time
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from look2hear.utils.parser_utils import prepare_parser_from_dict, parse_args_as_dict
import look2hear.models
import look2hear.videomodels
import yaml
from ptflops import get_model_complexity_info
from rich import print
from tqdm import tqdm

def check_parameters(net):
    """
        Returns module parameters. Mb
    """
    parameters = sum(param.numel() for param in net.parameters())
    return parameters / 10 ** 6


parser = argparse.ArgumentParser()
parser.add_argument(
    "--exp_dir", default="exp/tmp", help="Full path to save best validation model"
)

with open("configs/resepformer_causal.yml") as f:
    def_conf = yaml.safe_load(f)
parser = prepare_parser_from_dict(def_conf, parser=parser)

arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True)
audiomodel = getattr(look2hear.models, arg_dic["audionet"]["audionet_name"])(
    sample_rate=arg_dic["datamodule"]["data_config"]["sample_rate"],
    **arg_dic["audionet"]["audionet_config"]
)

with torch.cuda.device(0):
    a = torch.randn(1, 1, 8000)
    total_macs = 0
    total_params = 0
    with torch.no_grad():
        audiomodel = audiomodel.cpu()
        audiomodel.eval()
        start_time = time.time()
        with torch.no_grad():
            for i in tqdm(range(100)):
                audiomodel(a)
        print("Inference CPU Time: ", (time.time() - start_time) / 100)
    # DPRNN
    # model = audiomodel.cuda()
    # with torch.no_grad():
    #     macs, params = get_model_complexity_info(
    #         model, (8000,), as_strings=False, print_per_layer_stat=True, verbose=False
    #     )
    # print(model(a).shape)
    # model = audiomodel.cuda()
    # macs, params = get_model_complexity_info(
    #     model, (1,16000), as_strings=False, print_per_layer_stat=True, verbose=False
    # )
    # total_macs += macs
    # total_params += params
    # model = nn.Conv1d(1, 512, 64, 16).cuda()
    # macs, params = get_model_complexity_info(model, (1, 32000), as_strings=False,
    #                                                 print_per_layer_stat=True, verbose=False)
    # total_macs += macs
    # total_params += params

    # model = nn.ConvTranspose1d(512, 1, 64, 16).cuda()
    # macs, params = get_model_complexity_info(model, (512, 2000), as_strings=False,
    #                                                 print_per_layer_stat=True, verbose=False)
    # total_macs += macs*2
    # total_params += params*2
    # print(model(a, v).shape)
    # print("MACs: ", total_macs / 10.0 ** 9)
    # print("Params: ", total_params / 10.0 ** 6)
    # for i in range(1000):
    #     model(a)
