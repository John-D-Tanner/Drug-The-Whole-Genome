#!/usr/bin/env python3 -u
# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
import pickle
import torch
from unicore import checkpoint_utils, distributed_utils, options, utils
from unicore.logging import progress_bar
from unicore import tasks
import numpy as np
from tqdm import tqdm
import unicore

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("unimol.inference")


#from skchem.metrics import bedroc_score
from rdkit.ML.Scoring.Scoring import CalcBEDROC, CalcAUC, CalcEnrichment
from sklearn.metrics import roc_curve



def main(args):

    use_fp16 = args.fp16
    use_cuda = torch.cuda.is_available() and not args.cpu

    if use_cuda:
        torch.cuda.set_device(args.device_id)


    # Load model
    logger.info("loading model(s) from {}".format(args.path))
    #state = checkpoint_utils.load_checkpoint_to_cpu(args.path)
    task = tasks.setup_task(args)
    model = task.build_model(args)
    #model.load_state_dict(state["model"], strict=False)

    # Move models to GPU
    if use_cuda:
        if use_fp16:
            model.half()
        if use_cuda:
            model.cuda()

    # Print args
    logger.info(args)


    model.eval()

    task.perform_virtual_screen(model,
                                 args.pocket_data_path,
                                   args.mol_data_path,
                                     args.output_fpath,
                                       fold_version=args.fold_version,
                                         topN_percent=args.topN_percent,
                                           torch_lig_batch_size=args.torch_lig_batch_size,
                                             torch_pocket_batch_size=args.torch_pocket_batch_size,
                                               write_encodings=args.write_encodings,
                                                 verbose=args.verbose,
                                                     use_cuda=use_cuda)


def cli_main():
    # add args
    

    parser = options.get_validation_parser()
    parser.add_argument("--mol-data-path", type=str, default="", help=" A filepath str to an .lmdb or .pkl file containing ligand / chemical library data. The pkl should be of the preencoded vectors, and if passed, the encoding step will be skipped.")
    parser.add_argument("--pocket-data-path", type=str, default="", help="A filepath str to an .lmdb or .pkl file containing protein pocket data. The pkl should be of the preencoded vectors, and if passed, the encoding step will be skipped.")
    parser.add_argument("--fold-version", type=str, default="6_folds", help="Which set of model parameters should be used. Value must be a one of the following: \"6_folds\", \"8_folds\", \"6_folds_filtered\".")
    parser.add_argument("--output-fpath", type=str, default="", help="filepath string for the outputted text file with the ligand smiles and scores")
    parser.add_argument("--topN-percent", type=float, default=1.0, help="A float, N, between 0 and 1 that determines which top N % of ligands get outputted. Set to 1 for all ligands to be outputted. 0.01 for top 1%, etc.")
    parser.add_argument("--torch-lig-batch-size", type=int, default=64, help="When uploading the ligand dataset, upload it in batches of this size. Passed to Dataloader, see: https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader")
    parser.add_argument("--torch-pocket-batch-size", type=int, default=16, help="Same as for torch_lig_batch_size, but for the uploading of the pocket information. ")
    parser.add_argument("--write-encodings", type=bool, default=True, help="Write the encoded vectors to a file")
    parser.add_argument("--verbose", type=bool, default=True, help="Print performance and stats once the function has finished")
    options.add_model_args(parser)
    args = options.parse_args_and_arch(parser)

    distributed_utils.call_main(args, main)


if __name__ == "__main__":
    cli_main()
