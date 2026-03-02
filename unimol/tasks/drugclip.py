# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from IPython import embed as debug_embedded
import logging
import os
from collections.abc import Iterable
from sklearn.metrics import roc_auc_score
from xmlrpc.client import Boolean
import numpy as np
import torch
import pickle
from tqdm import tqdm
from unicore import checkpoint_utils
import unicore
from unicore.data import (AppendTokenDataset, Dictionary, EpochShuffleDataset,
                          FromNumpyDataset, NestedDictionaryDataset,
                          PrependTokenDataset, RawArrayDataset,LMDBDataset, RawLabelDataset,
                          RightPadDataset, RightPadDataset2D, TokenizeDataset,SortDataset,data_utils)
from unicore.tasks import UnicoreTask, register_task
from unimol.data import (AffinityDataset, CroppingPocketDataset,
                         CrossDistanceDataset, DistanceDataset,
                         EdgeTypeDataset, KeyDataset, LengthDataset,
                         NormalizeDataset, NormalizeDockingPoseDataset,
                         PrependAndAppend2DDataset, RemoveHydrogenDataset,
                         RemoveHydrogenPocketDataset, RightPadDatasetCoord, LMDBDatasetV2, LMDBKeyDataset,
                         RightPadDatasetCross2D, TTADockingPoseDataset, AffinityTestDataset, AffinityValidDataset, AffinityMolDataset, AffinityPocketDataset, ResamplingDataset)
#from skchem.metrics import bedroc_score
from rdkit.ML.Scoring.Scoring import CalcBEDROC, CalcAUC, CalcEnrichment
from sklearn.metrics import roc_curve
import h5py
import time



logger = logging.getLogger(__name__)


def re_new(y_true, y_score, ratio):
    fp = 0
    tp = 0
    p = sum(y_true)
    n = len(y_true) - p
    num = ratio*n
    sort_index = np.argsort(y_score)[::-1]
    for i in range(len(sort_index)):
        index = sort_index[i]
        if y_true[index] == 1:
            tp += 1
        else:
            fp += 1
            if fp>= num:
                break
    return (tp*n)/(p*fp)


def calc_re(y_true, y_score, ratio_list):
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    #print(fpr, tpr)
    res = {}
    res2 = {}
    total_active_compounds = sum(y_true)
    total_compounds = len(y_true)

    # for ratio in ratio_list:
    #     for i, t in enumerate(fpr):
    #         if t > ratio:
    #             #print(fpr[i], tpr[i])
    #             if fpr[i-1]==0:
    #                 res[str(ratio)]=tpr[i]/fpr[i]
    #             else:
    #                 res[str(ratio)]=tpr[i-1]/fpr[i-1]
    #             break
    
    for ratio in ratio_list:
        res2[str(ratio)] = re_new(y_true, y_score, ratio)

    #print(res)
    #print(res2)
    return res2

def cal_metrics(y_true, y_score, alpha):
    """
    Calculate BEDROC score.

    Parameters:
    - y_true: true binary labels (0 or 1)
    - y_score: predicted scores or probabilities
    - alpha: parameter controlling the degree of early retrieval emphasis

    Returns:
    - BEDROC score
    """
    
        # concate res_single and labels
    scores = np.expand_dims(y_score, axis=1)
    y_true = np.expand_dims(y_true, axis=1)
    scores = np.concatenate((scores, y_true), axis=1)
    # inverse sort scores based on first column
    scores = scores[scores[:,0].argsort()[::-1]]
    bedroc = CalcBEDROC(scores, 1, 80.5)
    count = 0
    # sort y_score, return index
    index  = np.argsort(y_score)[::-1]
    for i in range(int(len(index)*0.005)):
        if y_true[index[i]] == 1:
            count += 1
    auc = CalcAUC(scores, 1)
    ef_list = CalcEnrichment(scores, 1, [0.005, 0.01, 0.02, 0.05])
    ef = {
        "0.005": ef_list[0],
        "0.01": ef_list[1],
        "0.02": ef_list[2],
        "0.05": ef_list[3]
    }
    re_list = calc_re(y_true, y_score, [0.005, 0.01, 0.02, 0.05])
    return auc, bedroc, ef, re_list



@register_task("drugclip")
class DrugCLIP(UnicoreTask):
    """Task for training transformer auto-encoder models."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument(
            "data",
            help="downstream data path",
        )
        parser.add_argument(
            "--finetune-mol-model",
            default=None,
            type=str,
            help="pretrained molecular model path",
        )
        parser.add_argument(
            "--finetune-pocket-model",
            default=None,
            type=str,
            help="pretrained pocket model path",
        )
        parser.add_argument(
            "--dist-threshold",
            type=float,
            default=6.0,
            help="threshold for the distance between the molecule and the pocket",
        )
        parser.add_argument(
            "--max-pocket-atoms",
            type=int,
            default=256,
            help="selected maximum number of atoms in a pocket",
        )
        parser.add_argument(
            "--test-model",
            default=False,
            type=Boolean,
            help="whether test model",
        )
        parser.add_argument("--reg", action="store_true", help="regression task")

    def __init__(self, args, dictionary, pocket_dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        self.pocket_dictionary = pocket_dictionary
        self.seed = args.seed
        # add mask token
        self.mask_idx = dictionary.add_symbol("[MASK]", is_special=True)
        self.pocket_mask_idx = pocket_dictionary.add_symbol("[MASK]", is_special=True)
        self.mol_reps = None
        self.keys = None

    @classmethod
    def setup_task(cls, args, **kwargs):
        mol_dictionary = Dictionary.load(os.path.join(args.data, "dict_mol.txt"))
        pocket_dictionary = Dictionary.load(os.path.join(args.data, "dict_pkt.txt"))
        logger.info("ligand dictionary: {} types".format(len(mol_dictionary)))
        logger.info("pocket dictionary: {} types".format(len(pocket_dictionary)))
        return cls(args, mol_dictionary, pocket_dictionary)

    def load_dataset(self, split, **kwargs):
        """Load a given dataset split.
        'smi','pocket','atoms','coordinates','pocket_atoms','pocket_coordinates'
        Args:
            split (str): name of the data scoure (e.g., bppp)
        """
        data_path = os.path.join(self.args.data, split + ".lmdb")
        dataset = LMDBDataset(data_path)
        if split.startswith("train"):
            smi_dataset = KeyDataset(dataset, "smi")
            poc_dataset = KeyDataset(dataset, "pocket")
            
            dataset = AffinityDataset(
                dataset,
                self.args.seed,
                "atoms",
                "coordinates",
                "pocket_atoms",
                "pocket_coordinates",
                "label",
                True,
            )
            tgt_dataset = KeyDataset(dataset, "affinity")
            
        else:
            
            dataset = AffinityDataset(
                dataset,
                self.args.seed,
                "atoms",
                "coordinates",
                "pocket_atoms",
                "pocket_coordinates",
                "label",
            )
            tgt_dataset = KeyDataset(dataset, "affinity")
            smi_dataset = KeyDataset(dataset, "smi")
            poc_dataset = KeyDataset(dataset, "pocket")


        def PrependAndAppend(dataset, pre_token, app_token):
            dataset = PrependTokenDataset(dataset, pre_token)
            return AppendTokenDataset(dataset, app_token)

        dataset = RemoveHydrogenPocketDataset(
            dataset,
            "pocket_atoms",
            "pocket_coordinates",
            True,
            True,
        )
        dataset = CroppingPocketDataset(
            dataset,
            self.seed,
            "pocket_atoms",
            "pocket_coordinates",
            self.args.max_pocket_atoms,
        )

        dataset = RemoveHydrogenDataset(dataset, "atoms", "coordinates", True, True)


        apo_dataset = NormalizeDataset(dataset, "coordinates")
        apo_dataset = NormalizeDataset(apo_dataset, "pocket_coordinates")

        src_dataset = KeyDataset(apo_dataset, "atoms")
        mol_len_dataset = LengthDataset(src_dataset)
        src_dataset = TokenizeDataset(
            src_dataset, self.dictionary, max_seq_len=self.args.max_seq_len
        )
        coord_dataset = KeyDataset(apo_dataset, "coordinates")
        src_dataset = PrependAndAppend(
            src_dataset, self.dictionary.bos(), self.dictionary.eos()
        )
        edge_type = EdgeTypeDataset(src_dataset, len(self.dictionary))
        coord_dataset = FromNumpyDataset(coord_dataset)
        distance_dataset = DistanceDataset(coord_dataset)
        coord_dataset = PrependAndAppend(coord_dataset, 0.0, 0.0)
        distance_dataset = PrependAndAppend2DDataset(distance_dataset, 0.0)

        src_pocket_dataset = KeyDataset(apo_dataset, "pocket_atoms")
        pocket_len_dataset = LengthDataset(src_pocket_dataset)
        src_pocket_dataset = TokenizeDataset(
            src_pocket_dataset,
            self.pocket_dictionary,
            max_seq_len=self.args.max_seq_len,
        )
        coord_pocket_dataset = KeyDataset(apo_dataset, "pocket_coordinates")
        src_pocket_dataset = PrependAndAppend(
            src_pocket_dataset,
            self.pocket_dictionary.bos(),
            self.pocket_dictionary.eos(),
        )
        pocket_edge_type = EdgeTypeDataset(
            src_pocket_dataset, len(self.pocket_dictionary)
        )
        coord_pocket_dataset = FromNumpyDataset(coord_pocket_dataset)
        distance_pocket_dataset = DistanceDataset(coord_pocket_dataset)
        coord_pocket_dataset = PrependAndAppend(coord_pocket_dataset, 0.0, 0.0)
        distance_pocket_dataset = PrependAndAppend2DDataset(
            distance_pocket_dataset, 0.0
        )

        nest_dataset = NestedDictionaryDataset(
            {
                "net_input": {
                    "mol_src_tokens": RightPadDataset(
                        src_dataset,
                        pad_idx=self.dictionary.pad(),
                    ),
                    "mol_src_distance": RightPadDataset2D(
                        distance_dataset,
                        pad_idx=0,
                    ),
                    "mol_src_edge_type": RightPadDataset2D(
                        edge_type,
                        pad_idx=0,
                    ),
                    "pocket_src_tokens": RightPadDataset(
                        src_pocket_dataset,
                        pad_idx=self.pocket_dictionary.pad(),
                    ),
                    "pocket_src_distance": RightPadDataset2D(
                        distance_pocket_dataset,
                        pad_idx=0,
                    ),
                    "pocket_src_edge_type": RightPadDataset2D(
                        pocket_edge_type,
                        pad_idx=0,
                    ),
                    "pocket_src_coord": RightPadDatasetCoord(
                        coord_pocket_dataset,
                        pad_idx=0,
                    ),
                    "mol_len": RawArrayDataset(mol_len_dataset),
                    "pocket_len": RawArrayDataset(pocket_len_dataset)
                },
                "target": {
                    "finetune_target": RawLabelDataset(tgt_dataset),
                },
                "smi_name": RawArrayDataset(smi_dataset),
                "pocket_name": RawArrayDataset(poc_dataset),
            },
        )
        if split == "train":
            with data_utils.numpy_seed(self.args.seed):
                shuffle = np.random.permutation(len(src_dataset))

            self.datasets[split] = SortDataset(
                nest_dataset,
                sort_order=[shuffle],
            )
            self.datasets[split] = ResamplingDataset(
                self.datasets[split]
            )
        else:
            self.datasets[split] = nest_dataset


    

    def load_mols_dataset(self, data_path,atoms,coords, **kwargs):
 
        dataset = LMDBDataset(data_path)
        label_dataset = KeyDataset(dataset, "label")
        dataset = AffinityMolDataset(
            dataset,
            self.args.seed,
            atoms,
            coords,
            False,
        )
        
        smi_dataset = KeyDataset(dataset, "smi")

        def PrependAndAppend(dataset, pre_token, app_token):
            dataset = PrependTokenDataset(dataset, pre_token)
            return AppendTokenDataset(dataset, app_token)



        dataset = RemoveHydrogenDataset(dataset, "atoms", "coordinates", True, True)


        apo_dataset = NormalizeDataset(dataset, "coordinates")

        src_dataset = KeyDataset(apo_dataset, "atoms")
        len_dataset = LengthDataset(src_dataset)
        src_dataset = TokenizeDataset(
            src_dataset, self.dictionary, max_seq_len=self.args.max_seq_len
        )
        coord_dataset = KeyDataset(apo_dataset, "coordinates")
        src_dataset = PrependAndAppend(
            src_dataset, self.dictionary.bos(), self.dictionary.eos()
        )
        edge_type = EdgeTypeDataset(src_dataset, len(self.dictionary))
        coord_dataset = FromNumpyDataset(coord_dataset)
        distance_dataset = DistanceDataset(coord_dataset)
        coord_dataset = PrependAndAppend(coord_dataset, 0.0, 0.0)
        distance_dataset = PrependAndAppend2DDataset(distance_dataset, 0.0)


        nest_dataset = NestedDictionaryDataset(
            {
                "net_input": {
                    "mol_src_tokens": RightPadDataset(
                        src_dataset,
                        pad_idx=self.dictionary.pad(),
                    ),
                    "mol_src_distance": RightPadDataset2D(
                        distance_dataset,
                        pad_idx=0,
                    ),
                    "mol_src_edge_type": RightPadDataset2D(
                        edge_type,
                        pad_idx=0,
                    ),
                },
                "smi_name": RawArrayDataset(smi_dataset),
                "target":  RawArrayDataset(label_dataset),
                "mol_len": RawArrayDataset(len_dataset),
            },
        )
        return nest_dataset
    

    def load_mols_dataset_new(self, data_path,atoms,coords, **kwargs):
        #atom_key = 'atoms'
        #atom_key = 'atom_types'

        """Load a given dataset split.
        'smi','pocket','atoms','coordinates','pocket_atoms','pocket_coordinates','holo_coordinates','holo_pocket_coordinates','scaffold'
        Args:
            split (str): name of the data scoure (e.g., bppp)
        """

 
        dataset = LMDBDataset(data_path)
        subset_dataset = KeyDataset(dataset, "subset")
        id_dataset = KeyDataset(dataset, "IDs")
        smi_dataset = KeyDataset(dataset, "smi")
        name_dataset = KeyDataset(dataset, "name")
        dataset = AffinityMolDataset(
            dataset,
            self.args.seed,
            atoms,
            coords,
            False,
        )
        
        

        def PrependAndAppend(dataset, pre_token, app_token):
            dataset = PrependTokenDataset(dataset, pre_token)
            return AppendTokenDataset(dataset, app_token)



        dataset = RemoveHydrogenDataset(dataset, "atoms", "coordinates", True, True)


        apo_dataset = NormalizeDataset(dataset, "coordinates")

        src_dataset = KeyDataset(apo_dataset, "atoms")
        len_dataset = LengthDataset(src_dataset)
        src_dataset = TokenizeDataset(
            src_dataset, self.dictionary, max_seq_len=self.args.max_seq_len
        )
        coord_dataset = KeyDataset(apo_dataset, "coordinates")
        src_dataset = PrependAndAppend(
            src_dataset, self.dictionary.bos(), self.dictionary.eos()
        )
        edge_type = EdgeTypeDataset(src_dataset, len(self.dictionary))
        coord_dataset = FromNumpyDataset(coord_dataset)
        distance_dataset = DistanceDataset(coord_dataset)
        coord_dataset = PrependAndAppend(coord_dataset, 0.0, 0.0)
        distance_dataset = PrependAndAppend2DDataset(distance_dataset, 0.0)


        nest_dataset = NestedDictionaryDataset(
            {
                "net_input": {
                    "mol_src_tokens": RightPadDataset(
                        src_dataset,
                        pad_idx=self.dictionary.pad(),
                    ),
                    "mol_src_distance": RightPadDataset2D(
                        distance_dataset,
                        pad_idx=0,
                    ),
                    "mol_src_edge_type": RightPadDataset2D(
                        edge_type,
                        pad_idx=0,
                    ),
                },
                "smi_name": RawArrayDataset(smi_dataset),
                #"target":  RawArrayDataset(label_dataset),
                "mol_len": RawArrayDataset(len_dataset),
                #"key": RawArrayDataset(key_dataset),
                "id": RawArrayDataset(id_dataset),
                "subset": RawArrayDataset(subset_dataset),
            },
        )
        return nest_dataset

    def load_mols_dataset_dtwg(self, data_path,atoms,coords,dataset_type=1, **kwargs):
        #atom_key = 'atoms'
        #atom_key = 'atom_types'

        """Load a given dataset split.
        'smi','pocket','atoms','coordinates','pocket_atoms','pocket_coordinates','holo_coordinates','holo_pocket_coordinates','scaffold'
        Args:
            split (str): name of the data scoure (e.g., bppp)
        """

        if dataset_type == 2:
            dataset = LMDBDatasetV2(data_path)
            keys = dataset.get_split("success")
            keys = list(sorted(list(set(keys))))
            start = kwargs.get("start", 0)
            end = kwargs.get("end")
            if end is None:
                end = len(keys)
            if start >= len(keys):
                raise ValueError("start should be less than len(keys) = {}".format(len(keys)))
            logger.info("chunk dataset, start: {}, end: {}".format(start, end))
            dataset.set_split("chunk", keys[start:end], deduplicate=False, temporary=True)
            dataset.set_default_split("chunk")
            # keydataset = LMDBKeyDataset(data_path)
            # keydataset.set_split("chunk", keys[start:end], deduplicate=False, temporary=True)
            # keydataset.set_default_split("chunk")
        else:
            if kwargs.get("start", 0) != 0 or kwargs.get("end", None) is not None:
                logger.info("chuck is not supported when using default lmdb, ignore start and end")
            dataset = LMDBDataset(data_path)
            # keydataset = KeyDataset(dataset, "smiles")
        
        dataset = AffinityMolDataset(
            dataset,
            self.args.seed,
            atoms,
            coords,
            False,
        )

        # smi_dataset = KeyDataset(dataset, "smi")
        
        

        def PrependAndAppend(dataset, pre_token, app_token):
            dataset = PrependTokenDataset(dataset, pre_token)
            return AppendTokenDataset(dataset, app_token)



        dataset = RemoveHydrogenDataset(dataset, "atoms", "coordinates", True, True)


        apo_dataset = NormalizeDataset(dataset, "coordinates")

        src_dataset = KeyDataset(apo_dataset, "atoms")
        len_dataset = LengthDataset(src_dataset)
        src_dataset = TokenizeDataset(
            src_dataset, self.dictionary, max_seq_len=self.args.max_seq_len
        )
        coord_dataset = KeyDataset(apo_dataset, "coordinates")
        src_dataset = PrependAndAppend(
            src_dataset, self.dictionary.bos(), self.dictionary.eos()
        )
        edge_type = EdgeTypeDataset(src_dataset, len(self.dictionary))
        coord_dataset = FromNumpyDataset(coord_dataset)
        distance_dataset = DistanceDataset(coord_dataset)
        coord_dataset = PrependAndAppend(coord_dataset, 0.0, 0.0)
        distance_dataset = PrependAndAppend2DDataset(distance_dataset, 0.0)


        nest_dataset = {
            "net_input": {
                "mol_src_tokens": RightPadDataset(
                    src_dataset,
                    pad_idx=self.dictionary.pad(),
                ),
                "mol_src_distance": RightPadDataset2D(
                    distance_dataset,
                    pad_idx=0,
                ),
                "mol_src_edge_type": RightPadDataset2D(
                    edge_type,
                    pad_idx=0,
                ),
            },
            # "smi_name": RawArrayDataset(smi_dataset),
            #"target":  RawArrayDataset(label_dataset),
            "mol_len": RawArrayDataset(len_dataset),
        }
        # if keydataset is not None:
        #     nest_dataset["key"] = RawArrayDataset(keydataset)
        return NestedDictionaryDataset(nest_dataset)


    def load_retrieval_mols_dataset(self, data_path,atoms,coords, **kwargs):
 
        dataset = LMDBDataset(data_path)
        dataset = AffinityMolDataset(
            dataset,
            self.args.seed,
            atoms,
            coords,
            False,
        )
        
        smi_dataset = KeyDataset(dataset, "name")

        def PrependAndAppend(dataset, pre_token, app_token):
            dataset = PrependTokenDataset(dataset, pre_token)
            return AppendTokenDataset(dataset, app_token)



        dataset = RemoveHydrogenDataset(dataset, "atoms", "coordinates", True, True)


        apo_dataset = NormalizeDataset(dataset, "coordinates")

        src_dataset = KeyDataset(apo_dataset, "atoms")
        len_dataset = LengthDataset(src_dataset)
        src_dataset = TokenizeDataset(
            src_dataset, self.dictionary, max_seq_len=self.args.max_seq_len
        )
        coord_dataset = KeyDataset(apo_dataset, "coordinates")
        src_dataset = PrependAndAppend(
            src_dataset, self.dictionary.bos(), self.dictionary.eos()
        )
        edge_type = EdgeTypeDataset(src_dataset, len(self.dictionary))
        coord_dataset = FromNumpyDataset(coord_dataset)
        distance_dataset = DistanceDataset(coord_dataset)
        coord_dataset = PrependAndAppend(coord_dataset, 0.0, 0.0)
        distance_dataset = PrependAndAppend2DDataset(distance_dataset, 0.0)


        nest_dataset = NestedDictionaryDataset(
            {
                "net_input": {
                    "mol_src_tokens": RightPadDataset(
                        src_dataset,
                        pad_idx=self.dictionary.pad(),
                    ),
                    "mol_src_distance": RightPadDataset2D(
                        distance_dataset,
                        pad_idx=0,
                    ),
                    "mol_src_edge_type": RightPadDataset2D(
                        edge_type,
                        pad_idx=0,
                    ),
                },
                "smi_name": RawArrayDataset(smi_dataset),
                "mol_len": RawArrayDataset(len_dataset),
            },
        )
        return nest_dataset

    def load_pockets_dataset(self, data_path, **kwargs):

        dataset = LMDBDataset(data_path)
 
        dataset = AffinityPocketDataset(
            dataset,
            self.args.seed,
            "pocket_atoms",
            "pocket_coordinates",
            False,
            "pocket"
        )
        poc_dataset = KeyDataset(dataset, "pocket")

        def PrependAndAppend(dataset, pre_token, app_token):
            dataset = PrependTokenDataset(dataset, pre_token)
            return AppendTokenDataset(dataset, app_token)

        dataset = RemoveHydrogenPocketDataset(
            dataset,
            "pocket_atoms",
            "pocket_coordinates",
            True,
            True,
        )
        dataset = CroppingPocketDataset(
            dataset,
            self.seed,
            "pocket_atoms",
            "pocket_coordinates",
            self.args.max_pocket_atoms,
        )




        apo_dataset = NormalizeDataset(dataset, "pocket_coordinates")



        src_pocket_dataset = KeyDataset(apo_dataset, "pocket_atoms")
        len_dataset = LengthDataset(src_pocket_dataset)
        src_pocket_dataset = TokenizeDataset(
            src_pocket_dataset,
            self.pocket_dictionary,
            max_seq_len=self.args.max_seq_len,
        )
        coord_pocket_dataset = KeyDataset(apo_dataset, "pocket_coordinates")
        src_pocket_dataset = PrependAndAppend(
            src_pocket_dataset,
            self.pocket_dictionary.bos(),
            self.pocket_dictionary.eos(),
        )
        pocket_edge_type = EdgeTypeDataset(
            src_pocket_dataset, len(self.pocket_dictionary)
        )
        coord_pocket_dataset = FromNumpyDataset(coord_pocket_dataset)
        distance_pocket_dataset = DistanceDataset(coord_pocket_dataset)
        coord_pocket_dataset = PrependAndAppend(coord_pocket_dataset, 0.0, 0.0)
        distance_pocket_dataset = PrependAndAppend2DDataset(
            distance_pocket_dataset, 0.0
        )

        nest_dataset = NestedDictionaryDataset(
            {
                "net_input": {
                    "pocket_src_tokens": RightPadDataset(
                        src_pocket_dataset,
                        pad_idx=self.pocket_dictionary.pad(),
                    ),
                    "pocket_src_distance": RightPadDataset2D(
                        distance_pocket_dataset,
                        pad_idx=0,
                    ),
                    "pocket_src_edge_type": RightPadDataset2D(
                        pocket_edge_type,
                        pad_idx=0,
                    ),
                    "pocket_src_coord": RightPadDatasetCoord(
                        coord_pocket_dataset,
                        pad_idx=0,
                    ),
                },
                "pocket_name": RawArrayDataset(poc_dataset),
                "pocket_len": RawArrayDataset(len_dataset),
            },
        )
        return nest_dataset

    

    def build_model(self, args):
        from unicore import models

        model = models.build_model(args, self)
        
        if args.finetune_mol_model is not None:
            print("load pretrain model weight from...", args.finetune_mol_model)
            state = checkpoint_utils.load_checkpoint_to_cpu(
                args.finetune_mol_model,
            )
            model.mol_model.load_state_dict(state["model"], strict=False)
            
        if args.finetune_pocket_model is not None:
            print("load pretrain model weight from...", args.finetune_pocket_model)
            state = checkpoint_utils.load_checkpoint_to_cpu(
                args.finetune_pocket_model,
            )
            model.pocket_model.load_state_dict(state["model"], strict=False)

        return model

    def train_step(
        self, sample, model, loss, optimizer, update_num, ignore_grad=False
    ):
        """
        Do forward and backward, and return the loss as computed by *loss*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~unicore.data.UnicoreDataset`.
            model (~unicore.models.BaseUnicoreModel): the model
            loss (~unicore.losses.UnicoreLoss): the loss
            optimizer (~unicore.optim.UnicoreOptimizer): the optimizer
            update_num (int): the current update
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """

        model.train()
        model.set_num_updates(update_num)
        with torch.autograd.profiler.record_function("forward"):
            loss, sample_size, logging_output = loss(model, sample)
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        return loss, sample_size, logging_output
    
    def valid_step(self, sample, model, loss, test=False):
        model.eval()
        with torch.no_grad():
            loss, sample_size, logging_output = loss(model, sample)
        return loss, sample_size, logging_output

    def test_pcba_target_ensemble(self, target, model, **kwargs):


        data_path = "./data/lit_pcba/" + target + "/mols.lmdb"
        mol_dataset = self.load_mols_dataset(data_path, "atoms", "coordinates")
        num_data = len(mol_dataset)
        bsz=512
        print(num_data//bsz)
        
        
        # generate mol data
        
        mol_data = torch.utils.data.DataLoader(mol_dataset, batch_size=bsz, collate_fn=mol_dataset.collater)


        # 6 folds

        ckpts = [f"./data/model_weights/6_folds/fold_{i}.pt" for i in range(6)]

        res_list = []
        for fold, ckpt in enumerate(ckpts[:6]):
            # random generate mol_resps with size (num_data, 128)
            mol_reps = np.random.randn(num_data, 128)
            state = checkpoint_utils.load_checkpoint_to_cpu(ckpt)
            model.load_state_dict(state["model"], strict=False)
            mol_reps = []
            mol_names = []
            labels = []
            for _, sample in enumerate(tqdm(mol_data)):
                sample = unicore.utils.move_to_cuda(sample)
                dist = sample["net_input"]["mol_src_distance"]
                et = sample["net_input"]["mol_src_edge_type"]
                st = sample["net_input"]["mol_src_tokens"]
                mol_padding_mask = st.eq(model.mol_model.padding_idx)
                mol_x = model.mol_model.embed_tokens(st)
                n_node = dist.size(-1)
                gbf_feature = model.mol_model.gbf(dist, et)
                gbf_result = model.mol_model.gbf_proj(gbf_feature)
                graph_attn_bias = gbf_result
                graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
                graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
                mol_outputs = model.mol_model.encoder(
                    mol_x, padding_mask=mol_padding_mask, attn_mask=graph_attn_bias
                )
                mol_encoder_rep = mol_outputs[0][:,0,:]
                mol_emb = mol_encoder_rep
                mol_emb = model.mol_project(mol_encoder_rep)
                mol_emb = mol_emb / mol_emb.norm(dim=-1, keepdim=True)
                #print(mol_emb.dtype)
                mol_emb = mol_emb.detach().cpu().numpy()
                #print(mol_emb.dtype)
                mol_reps.append(mol_emb)
                mol_names.extend(sample["smi_name"])
                labels.extend(sample["target"].detach().cpu().numpy())
            mol_reps = np.concatenate(mol_reps, axis=0)
            labels = np.array(labels, dtype=np.int32)
            # labels = np.zeros(num_data)
            # generate pocket data
            data_path = "./data/lit_pcba/" + target + "/pockets.lmdb"
            if not os.path.exists(data_path):
                return None
            pocket_dataset = self.load_pockets_dataset(data_path)
            pocket_data = torch.utils.data.DataLoader(pocket_dataset, batch_size=bsz, collate_fn=pocket_dataset.collater)
            #pocket_reps = np.random.randn(len(pocket_data), 128)
            pocket_reps = []

            for _, sample in enumerate(tqdm(pocket_data)):
                sample = unicore.utils.move_to_cuda(sample)
                dist = sample["net_input"]["pocket_src_distance"]
                et = sample["net_input"]["pocket_src_edge_type"]
                st = sample["net_input"]["pocket_src_tokens"]
                pocket_padding_mask = st.eq(model.pocket_model.padding_idx)
                pocket_x = model.pocket_model.embed_tokens(st)
                n_node = dist.size(-1)
                gbf_feature = model.pocket_model.gbf(dist, et)
                gbf_result = model.pocket_model.gbf_proj(gbf_feature)
                graph_attn_bias = gbf_result
                graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
                graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
                pocket_outputs = model.pocket_model.encoder(
                    pocket_x, padding_mask=pocket_padding_mask, attn_mask=graph_attn_bias
                )
                pocket_encoder_rep = pocket_outputs[0][:,0,:]
                #pocket_emb = pocket_encoder_rep
                pocket_emb = model.pocket_project(pocket_encoder_rep)
                pocket_emb = pocket_emb / pocket_emb.norm(dim=-1, keepdim=True)
                pocket_emb = pocket_emb.detach().cpu().numpy()
                pocket_reps.append(pocket_emb)
            pocket_reps = np.concatenate(pocket_reps, axis=0)
            print(pocket_reps.shape)
            res = pocket_reps @ mol_reps.T
            print(res.shape)
            #res = np.expand_dims(res, axis=0)
            #print(res.shape)
            res_list.append(res)
        
        
        # get mean value of values in res_list without reduce dimension

        #res = np.concatenate(res_list, axis=0)

        res = np.array(res_list)

        res = np.mean(res, axis=0)
       
        print(res.shape)

        medians = np.median(res, axis=1, keepdims=True)
            # get mad for each row
        mads = np.median(np.abs(res - medians), axis=1, keepdims=True)
        # get z score
        res = 0.6745 * (res - medians) / (mads + 1e-6)
        # get max for each column


        
        res_single = res.max(axis=0)

        auc, bedroc, ef_list, re_list = cal_metrics(labels, res_single, 80.5)
        
        
        # print(target)

        # print(np.sum(labels), len(labels)-np.sum(labels))

        # print("auc", auc)
        # print("bedroc", bedroc)
        # print("ef", ef_list)

        return auc, bedroc, ef_list, re_list


    def test_pcba_target(self, name, model, **kwargs):
        """Encode a dataset with the molecule encoder."""

        #names = "PPARG"
        data_path = "./data/lit_pcba/" + name + "/mols.lmdb"
        mol_dataset = self.load_mols_dataset(data_path, "atoms", "coordinates")
        num_data = len(mol_dataset)
        bsz=512
        #print(num_data//bsz)
        mol_reps = []
        mol_names = []
        labels = []
        
        # generate mol data
        
        mol_data = torch.utils.data.DataLoader(mol_dataset, batch_size=bsz, collate_fn=mol_dataset.collater)
        for _, sample in enumerate(tqdm(mol_data)):
            sample = unicore.utils.move_to_cuda(sample)
            dist = sample["net_input"]["mol_src_distance"]
            et = sample["net_input"]["mol_src_edge_type"]
            st = sample["net_input"]["mol_src_tokens"]
            mol_padding_mask = st.eq(model.mol_model.padding_idx)
            mol_x = model.mol_model.embed_tokens(st)
            
            n_node = dist.size(-1)
            gbf_feature = model.mol_model.gbf(dist, et)

            gbf_result = model.mol_model.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            mol_outputs = model.mol_model.encoder(
                mol_x, padding_mask=mol_padding_mask, attn_mask=graph_attn_bias
            )
            mol_encoder_rep = mol_outputs[0][:,0,:]
            mol_emb = model.mol_project(mol_encoder_rep)
            mol_emb = mol_emb / mol_emb.norm(dim=1, keepdim=True)
            mol_emb = mol_emb.detach().cpu().numpy()
            mol_reps.append(mol_emb)
            mol_names.extend(sample["smi_name"])
            labels.extend(sample["target"].detach().cpu().numpy())
        mol_reps = np.concatenate(mol_reps, axis=0)
        labels = np.array(labels, dtype=np.int32)
        # generate pocket data
        data_path = "./data/lit_pcba/" + name + "/pockets.lmdb"
        pocket_dataset = self.load_pockets_dataset(data_path)
        pocket_data = torch.utils.data.DataLoader(pocket_dataset, batch_size=bsz, collate_fn=pocket_dataset.collater)
        pocket_reps = []

        for _, sample in enumerate(tqdm(pocket_data)):
            sample = unicore.utils.move_to_cuda(sample)
            dist = sample["net_input"]["pocket_src_distance"]
            et = sample["net_input"]["pocket_src_edge_type"]
            st = sample["net_input"]["pocket_src_tokens"]
            pocket_padding_mask = st.eq(model.pocket_model.padding_idx)
            pocket_x = model.pocket_model.embed_tokens(st)
            n_node = dist.size(-1)
            gbf_feature = model.pocket_model.gbf(dist, et)
            gbf_result = model.pocket_model.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            pocket_outputs = model.pocket_model.encoder(
                pocket_x, padding_mask=pocket_padding_mask, attn_mask=graph_attn_bias
            )
            pocket_encoder_rep = pocket_outputs[0][:,0,:]
            pocket_emb = model.pocket_project(pocket_encoder_rep)
            pocket_emb = pocket_emb / pocket_emb.norm(dim=1, keepdim=True)
            pocket_emb = pocket_emb.detach().cpu().numpy()
            pocket_names = sample["pocket_name"]
            pocket_reps.append(pocket_emb)
        pocket_reps = np.concatenate(pocket_reps, axis=0)

        res = pocket_reps @ mol_reps.T

        # medians = np.median(res, axis=1, keepdims=True)
        #     # get mad for each row
        # mads = np.median(np.abs(res - medians), axis=1, keepdims=True)
        # # get z score
        # res = 0.6745 * (res - medians) / (mads + 1e-6)
        
        res_single = res.max(axis=0)
        auc, bedroc, ef_list, re_list = cal_metrics(labels, res_single, 80.5)

        return auc, bedroc, ef_list, re_list
    
    
    

    def test_pcba(self, model, use_folds=True, **kwargs):

        use_folds = False
        targets = os.listdir("./data/lit_pcba/")

        #print(targets)
        auc_list = []
        ef_list = []
        bedroc_list = []

        re_list = {
            "0.005": [],
            "0.01": [],
            "0.02": [],
            "0.05": []
        }
        ef_list = {
            "0.005": [],
            "0.01": [],
            "0.02": [],
            "0.05": []
        }
        for target in targets:
            if use_folds:
                auc, bedroc, ef, re = self.test_pcba_target_ensemble(target, model)
            else:
                auc, bedroc, ef, re = self.test_pcba_target(target, model)
            auc_list.append(auc)
            bedroc_list.append(bedroc)
            for key in ef:
                ef_list[key].append(ef[key])
            # print("re", re)
            # print("ef", ef)
            for key in re:
                re_list[key].append(re[key])
        print(auc_list)
        print(ef_list)

        print("auc mean", np.mean(auc_list))
        print("bedroc mean", np.mean(bedroc_list))
        for key in ef_list:
            print("ef",key, "mean", np.mean(ef_list[key]))
        for key in re_list:
            print("re",key, "mean", np.mean(re_list[key]))

        return 
    
    def test_dude_target(self, target, model, **kwargs):

        data_path = "./data/DUD-E/" + target + "/mols.lmdb"
        if not os.path.exists(data_path):
            return None
        mol_dataset = self.load_mols_dataset(data_path, "atoms", "coordinates")
        num_data = len(mol_dataset)
        bsz=64
        print(num_data//bsz)
        mol_reps = []
        mol_names = []
        labels = []
        
        # generate mol data
        
        mol_data = torch.utils.data.DataLoader(mol_dataset, batch_size=bsz, collate_fn=mol_dataset.collater)
        for _, sample in enumerate(tqdm(mol_data)):
            sample = unicore.utils.move_to_cuda(sample)
            dist = sample["net_input"]["mol_src_distance"]
            et = sample["net_input"]["mol_src_edge_type"]
            st = sample["net_input"]["mol_src_tokens"]
            mol_padding_mask = st.eq(model.mol_model.padding_idx)
            mol_x = model.mol_model.embed_tokens(st)
            n_node = dist.size(-1)
            gbf_feature = model.mol_model.gbf(dist, et)
            gbf_result = model.mol_model.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            mol_outputs = model.mol_model.encoder(
                mol_x, padding_mask=mol_padding_mask, attn_mask=graph_attn_bias
            )
            mol_encoder_rep = mol_outputs[0][:,0,:]
            mol_emb = mol_encoder_rep
            mol_emb = model.mol_project(mol_encoder_rep)
            mol_emb = mol_emb / mol_emb.norm(dim=-1, keepdim=True)
            #print(mol_emb.dtype)
            mol_emb = mol_emb.detach().cpu().numpy()
            #print(mol_emb.dtype)
            mol_reps.append(mol_emb)
            mol_names.extend(sample["smi_name"])
            labels.extend(sample["target"].detach().cpu().numpy())
        mol_reps = np.concatenate(mol_reps, axis=0)
        labels = np.array(labels, dtype=np.int32)
        # generate pocket data
        data_path = "./data/DUD-E/" + target + "/pocket.lmdb"
        #data_path = f"/drug/schrodinger_DUD-E/{target}/Holo_RealPocket_GenPack/pocket.lmdb"
        if not os.path.exists(data_path):
            return None
        pocket_dataset = self.load_pockets_dataset(data_path)
        pocket_data = torch.utils.data.DataLoader(pocket_dataset, batch_size=bsz, collate_fn=pocket_dataset.collater)
        pocket_reps = []

        for _, sample in enumerate(tqdm(pocket_data)):
            sample = unicore.utils.move_to_cuda(sample)
            dist = sample["net_input"]["pocket_src_distance"]
            et = sample["net_input"]["pocket_src_edge_type"]
            st = sample["net_input"]["pocket_src_tokens"]
            pocket_padding_mask = st.eq(model.pocket_model.padding_idx)
            pocket_x = model.pocket_model.embed_tokens(st)
            n_node = dist.size(-1)
            gbf_feature = model.pocket_model.gbf(dist, et)
            gbf_result = model.pocket_model.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            pocket_outputs = model.pocket_model.encoder(
                pocket_x, padding_mask=pocket_padding_mask, attn_mask=graph_attn_bias
            )
            pocket_encoder_rep = pocket_outputs[0][:,0,:]
            #pocket_emb = pocket_encoder_rep
            pocket_emb = model.pocket_project(pocket_encoder_rep)
            pocket_emb = pocket_emb / pocket_emb.norm(dim=-1, keepdim=True)
            pocket_emb = pocket_emb.detach().cpu().numpy()
            pocket_reps.append(pocket_emb)
        pocket_reps = np.concatenate(pocket_reps, axis=0)
        print(pocket_reps.shape)
        res = pocket_reps @ mol_reps.T

        medians = np.median(res, axis=1, keepdims=True)
            # get mad for each row
        mads = np.median(np.abs(res - medians), axis=1, keepdims=True)
        # get z score
        res = 0.6745 * (res - medians) / (mads + 1e-6)
        # get max for each column
        #res_max = np.max(res_cur, axis=0)
        #res = np.max(res, axis=0)

        res_single = res.max(axis=0)

        auc, bedroc, ef_list, re_list = cal_metrics(labels, res_single, 80.5)
        
        
        print(target)

        print(np.sum(labels), len(labels)-np.sum(labels))

        print(auc, bedroc)

        return auc, bedroc, ef_list, re_list, res_single, labels

    def test_dude_target_ensemble(self, target, model, **kwargs):

       
   
        data_path = "./data/DUD-E/" + target + "/mols.lmdb"
        if not os.path.exists(data_path):
            return None
        mol_dataset = self.load_mols_dataset(data_path, "atoms", "coordinates")
        num_data = len(mol_dataset)
        bsz=512
        print(num_data//bsz)
        
        
        # generate mol data
        
        mol_data = torch.utils.data.DataLoader(mol_dataset, batch_size=bsz, collate_fn=mol_dataset.collater)

        # 6 folds

        ckpts = [f"./data/model_weights/6_folds/fold_{i}.pt" for i in range(6)]




        res_list = []
        for fold, ckpt in enumerate(ckpts):

            state = checkpoint_utils.load_checkpoint_to_cpu(ckpt)
            model.load_state_dict(state["model"], strict=False)
            mol_reps = []
            mol_names = []
            labels = []
            for _, sample in enumerate(tqdm(mol_data)):
                sample = unicore.utils.move_to_cuda(sample)
                dist = sample["net_input"]["mol_src_distance"]
                et = sample["net_input"]["mol_src_edge_type"]
                st = sample["net_input"]["mol_src_tokens"]
                mol_padding_mask = st.eq(model.mol_model.padding_idx)
                mol_x = model.mol_model.embed_tokens(st)
                n_node = dist.size(-1)
                gbf_feature = model.mol_model.gbf(dist, et)
                gbf_result = model.mol_model.gbf_proj(gbf_feature)
                graph_attn_bias = gbf_result
                graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
                graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
                mol_outputs = model.mol_model.encoder(
                    mol_x, padding_mask=mol_padding_mask, attn_mask=graph_attn_bias
                )
                mol_encoder_rep = mol_outputs[0][:,0,:]
                mol_emb = mol_encoder_rep
                mol_emb = model.mol_project(mol_encoder_rep)
                mol_emb = mol_emb / mol_emb.norm(dim=-1, keepdim=True)
                #print(mol_emb.dtype)
                mol_emb = mol_emb.detach().cpu().numpy()
                #print(mol_emb.dtype)
                mol_reps.append(mol_emb)
                mol_names.extend(sample["smi_name"])
                labels.extend(sample["target"].detach().cpu().numpy())
            mol_reps = np.concatenate(mol_reps, axis=0)
            labels = np.array(labels, dtype=np.int32)
            # generate pocket data
            #data_path = "./data/DUD-E/" + target + "/RealProtein_RealPocket/pockets.lmdb"
            data_path = "./data/DUD-E/" + target + "/pocket.lmdb"
            if not os.path.exists(data_path):
                return None
            pocket_dataset = self.load_pockets_dataset(data_path)
            pocket_data = torch.utils.data.DataLoader(pocket_dataset, batch_size=bsz, collate_fn=pocket_dataset.collater)
            pocket_reps = []

            for _, sample in enumerate(tqdm(pocket_data)):
                sample = unicore.utils.move_to_cuda(sample)
                dist = sample["net_input"]["pocket_src_distance"]
                et = sample["net_input"]["pocket_src_edge_type"]
                st = sample["net_input"]["pocket_src_tokens"]
                pocket_padding_mask = st.eq(model.pocket_model.padding_idx)
                pocket_x = model.pocket_model.embed_tokens(st)
                n_node = dist.size(-1)
                gbf_feature = model.pocket_model.gbf(dist, et)
                gbf_result = model.pocket_model.gbf_proj(gbf_feature)
                graph_attn_bias = gbf_result
                graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
                graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
                pocket_outputs = model.pocket_model.encoder(
                    pocket_x, padding_mask=pocket_padding_mask, attn_mask=graph_attn_bias
                )
                pocket_encoder_rep = pocket_outputs[0][:,0,:]
                #pocket_emb = pocket_encoder_rep
                pocket_emb = model.pocket_project(pocket_encoder_rep)
                pocket_emb = pocket_emb / pocket_emb.norm(dim=-1, keepdim=True)
                pocket_emb = pocket_emb.detach().cpu().numpy()
                pocket_reps.append(pocket_emb)
            pocket_reps = np.concatenate(pocket_reps, axis=0)
            print(pocket_reps.shape)
            res = pocket_reps @ mol_reps.T
            #res_list.append(np.expand_dims(res, axis=0))
            res_list.append(res)
        
        #res = np.concatenate(res_list, axis=0)

        res = np.array(res_list)
        print(res.shape)
        res = res.mean(axis=0)

        print(res.shape)

        medians = np.median(res, axis=1, keepdims=True)
            # get mad for each row
        mads = np.median(np.abs(res - medians), axis=1, keepdims=True)
        # get z score
        res = 0.6745 * (res - medians) / (mads + 1e-6)
        # get max for each column
        #res_max = np.max(res_cur, axis=0)
        #res = np.max(res, axis=0)

        res_single = res.max(axis=0)

        auc, bedroc, ef_list, re_list = cal_metrics(labels, res_single, 80.5)
        
        
        print(target)

        print(np.sum(labels), len(labels)-np.sum(labels))

        print("auc", auc)
        print("bedroc", bedroc)
        print("ef", ef_list)

        return auc, bedroc, ef_list, re_list, res_single, labels
    

    

    def test_dude(self, model, use_folds=True, **kwargs):


        targets = os.listdir("./data/DUD-E")
        #targets = os.listdir("/drug/schrodinger_DUD-E/")
        auc_list = []
        bedroc_list = []
        ef_list = []
        res_list= []
        labels_list = []
        re_list = {
            "0.005": [],
            "0.01": [],
            "0.02": [],
            "0.05": [],
        }
        ef_list = {
            "0.005": [],
            "0.01": [],
            "0.02": [],
            "0.05": [],
        }
        ef_dic = {}
        bedroc_dic = {}
        for i,target in enumerate(targets):
            print(i)
            #try:
            use_folds = False
            if use_folds:
                res = self.test_dude_target_ensemble(target, model)
            else:
                res = self.test_dude_target(target, model)
            #except:
            #    continue
            if res is None:
                continue
            auc, bedroc, ef, re, res_single, labels = res#self.test_dude_target(target, model)

            auc_list.append(auc)
            bedroc_list.append(bedroc)
            for key in ef:
                ef_list[key].append(ef[key])
            for key in re_list:
                re_list[key].append(re[key])
            res_list.append(res_single)
            labels_list.append(labels)
            ef_dic[target] = ef["0.01"]
            bedroc_dic[target] = bedroc
        res = np.concatenate(res_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)
        print(len(auc_list))
        print("auc mean", np.mean(auc_list))
        print("bedroc mean", np.mean(bedroc_list))

        for key in ef_list:
            print("ef", key, "mean", np.mean(ef_list[key]))

        for key in re_list:
            print("re", key, "mean",  np.mean(re_list[key]))

        # save printed results to csv
        
        targets = targets
        ef = ef_list["0.01"]

        # save to csv
        import pandas as pd
        df = pd.DataFrame({"target": targets, "ef": ef})
        df.to_csv("Holo_RealPocket_GenPack.csv", index=False)
            
        
        
        return
    
    
    
    
    
    def encode_mols_once(self, model, data_path, cache_path, atoms, coords, **kwargs):
        
        # cache path is embdir/data_path.pkl

        # cache_path = os.path.join(emb_dir, data_path.split("/")[-1] + ".pkl")

        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                mol_reps, mol_names = pickle.load(f)
            return mol_reps, mol_names

        mol_dataset = self.load_retrieval_mols_dataset(data_path,atoms,coords)
        mol_reps = []
        mol_names = []
        bsz=32
        mol_data = torch.utils.data.DataLoader(mol_dataset, batch_size=bsz, collate_fn=mol_dataset.collater)
        for _, sample in enumerate(tqdm(mol_data)):
            sample = unicore.utils.move_to_cuda(sample)
            dist = sample["net_input"]["mol_src_distance"]
            et = sample["net_input"]["mol_src_edge_type"]
            st = sample["net_input"]["mol_src_tokens"]
            mol_padding_mask = st.eq(model.mol_model.padding_idx)
            mol_x = model.mol_model.embed_tokens(st)
            n_node = dist.size(-1)
            gbf_feature = model.mol_model.gbf(dist, et)
            gbf_result = model.mol_model.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            mol_outputs = model.mol_model.encoder(
                mol_x, padding_mask=mol_padding_mask, attn_mask=graph_attn_bias
            )
            mol_encoder_rep = mol_outputs[0][:,0,:]
            mol_emb = model.mol_project(mol_encoder_rep)
            mol_emb = mol_emb / mol_emb.norm(dim=-1, keepdim=True)
            mol_emb = mol_emb.detach().cpu().numpy()
            mol_reps.append(mol_emb)
            mol_names.extend(sample["smi_name"])

        mol_reps = np.concatenate(mol_reps, axis=0)

        # save the results
        
        with open(cache_path, "wb") as f:
            pickle.dump([mol_reps, mol_names], f)

        return mol_reps, mol_names
    
    def retrieve_mols(self, model, mol_path, pocket_path, emb_dir, k, **kwargs):
 
        os.makedirs(emb_dir, exist_ok=True)        
        mol_reps, mol_names = self.encode_mols_once(model, mol_path, emb_dir,  "atoms", "coordinates")
        
        pocket_dataset = self.load_pockets_dataset(pocket_path)
        pocket_data = torch.utils.data.DataLoader(pocket_dataset, batch_size=16, collate_fn=pocket_dataset.collater)
        pocket_reps = []
        pocket_names = []
        for _, sample in enumerate(tqdm(pocket_data)):
            sample = unicore.utils.move_to_cuda(sample)
            dist = sample["net_input"]["pocket_src_distance"]
            et = sample["net_input"]["pocket_src_edge_type"]
            st = sample["net_input"]["pocket_src_tokens"]
            pocket_padding_mask = st.eq(model.pocket_model.padding_idx)
            pocket_x = model.pocket_model.embed_tokens(st)
            n_node = dist.size(-1)
            gbf_feature = model.pocket_model.gbf(dist, et)
            gbf_result = model.pocket_model.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            pocket_outputs = model.pocket_model.encoder(
                pocket_x, padding_mask=pocket_padding_mask, attn_mask=graph_attn_bias
            )
            pocket_encoder_rep = pocket_outputs[0][:,0,:]
            pocket_emb = model.pocket_project(pocket_encoder_rep)
            pocket_emb = pocket_emb / pocket_emb.norm(dim=-1, keepdim=True)
            pocket_emb = pocket_emb.detach().cpu().numpy()
            pocket_reps.append(pocket_emb)
            pocket_names.extend(sample["pocket_name"])
        pocket_reps = np.concatenate(pocket_reps, axis=0)
        
        res = pocket_reps @ mol_reps.T
        res = res.max(axis=0)


        # get top k results

        
        top_k = np.argsort(res)[::-1][:k]

        # return names and scores
        
        return [mol_names[i] for i in top_k], res[top_k]

    

    def encode_mols_multi_folds(self, model, batch_size, mol_path, save_dir, use_cuda, dataset_type=None, write_npy=True, write_h5=True, flush_interval=60, **kwargs):

        # 6 folds
        
        ckpts = [f"./data/model_weights/6_folds/fold_{i}.pt" for i in range(6)]

        if dataset_type is None:
            dataset_type = 2 if os.path.isdir(mol_path) else 1
        logger.info(f"dataset_type: {dataset_type}")
        if write_h5:
            h5_path = os.path.join(save_dir,f"mol_reps{kwargs.get('start', '')}{kwargs.get('end', '')}.h5")
            logger.info(f"encoding write to {h5_path}, resume is supported")
        if write_npy:
            npy_path = os.path.join(save_dir,f"mol_reps{kwargs.get('start', '')}{kwargs.get('end', '')}.npy")
            logger.info(f"encoding write to {npy_path} in one shot, embeddings will accumulate in the memory")
            if not write_h5: logger.info("resume is not supported for npy")

        #ckpts = ckpts[:1]

        # prefix = "/drug/DrugCLIP_chemdata_v2024/embs/"

        # prefix = save_dir

        #os.makedirs(prefix, exist_ok=True)

        # caches = ["fold0.pkl", "fold1.pkl", "fold2.pkl", "fold3.pkl", "fold4.pkl", "fold5.pkl"]
        # caches = [prefix + cache for cache in caches]

        mol_reps_all = []

        for fold, ckpt in enumerate(ckpts):

            

            state = checkpoint_utils.load_checkpoint_to_cpu(ckpt)
            model.load_state_dict(state["model"], strict=False)

            
            #mol_data_path = "/drug/DrugCLIP_chemdata_v2024/DrugCLIP_mols_v2024.lmdb"

            mol_data_path = mol_path

            
            mol_dataset = self.load_mols_dataset_dtwg(mol_data_path, "atoms", "coordinates", dataset_type=dataset_type, **kwargs)
            collate_fn=mol_dataset.collater
            bsz=batch_size
            mol_reps = []
            mol_names = []
            mol_ids_subsets = []
            
            # generate mol data
            try:
                if write_h5:
                    hdf5 = h5py.File(os.path.join(save_dir,f"mol_reps{kwargs.get('start', '')}{kwargs.get('end', '')}.h5"), "a")
                    dset = hdf5.require_dataset("mol_reps", shape=(len(mol_dataset), 768), dtype=np.float32, chunks=True)
                    kset = hdf5.require_dataset("fold{}".format(fold), shape=(len(mol_dataset),), dtype=np.bool_, chunks=True, compression="lzf")
                    written_mask = kset[:]
                    num_written = np.sum(written_mask)
                    if num_written == len(mol_dataset):
                        if write_npy:
                            mol_reps = dset[:, fold*128:(fold+1)*128]
                            mol_reps = np.expand_dims(mol_reps, axis=1)
                            mol_reps_all.append(mol_reps)
                        print("Already written fold {} mols".format(fold))
                        continue
                    elif num_written > 0:
                        mol_dataset = torch.utils.data.Subset(mol_dataset, range(num_written, len(mol_dataset)))
                        if write_npy:
                            mol_reps.append(dset[:num_written, fold*128:(fold+1)*128])
                        print("Already written {} mols in fold {}, will skip them".format(num_written, fold))
                logger.info(f"dataloader workers: {self.args.num_workers}")
                mol_data = torch.utils.data.DataLoader(mol_dataset, batch_size=bsz, collate_fn=collate_fn, num_workers=self.args.num_workers)
                for batch, sample in enumerate(tqdm(mol_data)):
                    if use_cuda:
                        sample = unicore.utils.move_to_cuda(sample)
                    
                    dist = sample["net_input"]["mol_src_distance"]
                    et = sample["net_input"]["mol_src_edge_type"]
                    st = sample["net_input"]["mol_src_tokens"]
                    mol_padding_mask = st.eq(model.mol_model.padding_idx)
                    mol_x = model.mol_model.embed_tokens(st)
                    n_node = dist.size(-1)
                    gbf_feature = model.mol_model.gbf(dist, et)
                    gbf_result = model.mol_model.gbf_proj(gbf_feature)
                    graph_attn_bias = gbf_result
                    graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
                    graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
                    mol_outputs = model.mol_model.encoder(
                        mol_x, padding_mask=mol_padding_mask, attn_mask=graph_attn_bias
                    )
                    mol_encoder_rep = mol_outputs[0][:,0,:]
                    mol_emb = model.mol_project(mol_encoder_rep)
                    mol_emb = mol_emb / mol_emb.norm(dim=-1, keepdim=True)
                    mol_emb = mol_emb.detach().cpu().numpy()
                    if write_h5:
                        dset[num_written+batch*bsz:num_written+batch*bsz+len(mol_emb), fold*128:(fold+1)*128] = mol_emb
                        kset[num_written+batch*bsz:num_written+batch*bsz+len(mol_emb)] = 1
                        if batch % flush_interval == 0:
                            hdf5.flush()
                    #mol_reps.append(mol_emb)
                    #index = st.squeeze(0) > 3
                    #cur_mol_reps = mol_outputs[0]
                    #cur_mol_reps = cur_mol_reps[:, index, :]
                    if write_npy:
                        mol_reps.append(mol_emb)
                    #print(mol_emb.detach().cpu().numpy().shape)
                    # mol_names.extend(sample["smi_name"])

                    #ids = sample["id"]
                    #subsets = sample["subset"]
                    #ids_subsets = [ids[i] + ";" + subsets[i] for i in range(len(ids))]
                    #mol_ids_subsets.extend(ids_subsets)
                if write_npy:
                    mol_reps = np.concatenate(mol_reps, axis=0)
                    # add a dimension to mol_reps
                    mol_reps = np.expand_dims(mol_reps, axis=1)
                    mol_reps_all.append(mol_reps)
            except Exception as e:
                print(e)
                if write_h5:
                    hdf5.close()
                raise e
            finally:
                if write_h5:
                    hdf5.flush()
                    hdf5.close()
        
        # concate
        if write_npy:
            mol_reps_all = np.concatenate(mol_reps_all, axis=1)

            # mol_names = np.array(mol_names)

            

            # convert to float 32
            mol_reps_all = mol_reps_all.astype(np.float32)

            # save the reps to npy file
            print(mol_reps_all.shape)
            np.save(os.path.join(save_dir,f"mol_reps{kwargs.get('start', '')}{kwargs.get('end', '')}.npy"), mol_reps_all)

    
    def encode_pockets_multi_folds(self, model, pocket_dir, pocket_path, **kwargs):
        print(pocket_path)
        # 6 folds
        ckpts = [f"./data/model_weights/6_folds/fold_{i}.pt" for i in range(6)]


        #ckpts = ckpts[:1]




        pocket_reps_all = []
        pocket_names_all = []

        for fold, ckpt in enumerate(ckpts):

            state = checkpoint_utils.load_checkpoint_to_cpu(ckpt)
            model.load_state_dict(state["model"], strict=False)

            # generate pocket data
            pocket_dataset = self.load_pockets_dataset(pocket_path)
            logger.info(f"dataloader workers: {self.args.num_workers}")
            pocket_data = torch.utils.data.DataLoader(pocket_dataset, batch_size=32, collate_fn=pocket_dataset.collater, num_workers=self.args.num_workers)
            pocket_reps = []
            pocket_names = []
            for _, sample in enumerate(tqdm(pocket_data)):

                sample = unicore.utils.move_to_cuda(sample)
                dist = sample["net_input"]["pocket_src_distance"]
                et = sample["net_input"]["pocket_src_edge_type"]
                st = sample["net_input"]["pocket_src_tokens"]
                print(st)
                pocket_padding_mask = st.eq(model.pocket_model.padding_idx)
                pocket_x = model.pocket_model.embed_tokens(st)
                n_node = dist.size(-1)
                gbf_feature = model.pocket_model.gbf(dist, et)
                gbf_result = model.pocket_model.gbf_proj(gbf_feature)
                graph_attn_bias = gbf_result
                graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
                graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
                pocket_outputs = model.pocket_model.encoder(
                    pocket_x, padding_mask=pocket_padding_mask, attn_mask=graph_attn_bias
                )
                pocket_encoder_rep = pocket_outputs[0][:,0,:]
                pocket_emb = model.pocket_project(pocket_encoder_rep)
                pocket_emb = pocket_emb / pocket_emb.norm(dim=-1, keepdim=True)
                pocket_emb = pocket_emb.detach().cpu().numpy()
                pocket_name = sample["pocket_name"]
                pocket_names.extend(pocket_name)
                #pocket_reps.append(pocket_emb)
                # find all index that st is > 3
                index = st > 3

                index = index.squeeze(0)

                print(index.shape)

                # index is the second 

                cur_pocket_reps = pocket_outputs[0]

                cur_pocket_reps = cur_pocket_reps[:, index, :]

                pocket_reps.append(cur_pocket_reps.detach().cpu().numpy())

               
            pocket_reps = np.concatenate(pocket_reps, axis=0)
            pocket_reps = pocket_reps.astype(np.float32)
            pocket_reps_all.append(pocket_reps)

        
        pocket_reps_all = np.concatenate(pocket_reps_all, axis=0)

        # change the first and second dimension

        pocket_reps_all = pocket_reps_all.transpose(1, 0, 2)

        # merge the second and third dimension



        print(pocket_reps_all.shape)

        #print(pocket_reps_all.shape)

        # save the reps and names

        return pocket_reps_all, pocket_names



    def retrieval_multi_folds(self, model, pocket_path, save_path, mol_data_path, fold_version, use_cache=True, use_cuda=True, **kwargs):
        

        if fold_version=="6_folds":
            # 6 folds
            ckpts = [f"./data/model_weights/6_folds/fold_{i}.pt" for i in range(6)]

            caches = [f"./data/encoded_mol_embs/6_folds/fold{i}.pkl" for i in range(6)]
        
        elif fold_version=="8_folds":

            ckpts = [f"./data/model_weights/8_folds/fold_{i}.pt" for i in range(8)]

            caches = [f"./data/encoded_mol_embs/8_folds/fold{i}.pkl" for i in range(8)]
        elif fold_version=="6_folds_filtered":
            ckpts = [f"./data/model_weights/6_folds/fold_{i}.pt" for i in range(6)]

            caches = [f"./data/encoded_mol_embs/6_folds_filtered/fold{i}.pkl" for i in range(6)]


        res_list = []

        pocket_data_path = pocket_path


        for fold, ckpt in enumerate(ckpts):
            state = checkpoint_utils.load_checkpoint_to_cpu(ckpt)
            model.load_state_dict(state["model"], strict=False)

            # generate mol data

            mol_cache_path=caches[fold]
#            if use_cache:
            if True:
                #JT test:
                mol_cache_path = 'data/targets/Test_4P42/testsdf.pkl'
                with open(mol_cache_path, "rb") as f:
                    mol_reps, mol_names = pickle.load(f)
                mol_reps = mol_reps[fold] # JT test

            else:            

                
                mol_dataset = self.load_mols_dataset(mol_data_path, "atoms", "coordinates")
                num_data = len(mol_dataset)
                bsz=64
                mol_reps = []
                mol_names = []
                labels = []
                mol_ids_subsets = []
                mol_data = torch.utils.data.DataLoader(mol_dataset, batch_size=bsz, collate_fn=mol_dataset.collater)
                for _, sample in enumerate(tqdm(mol_data)):
                    if use_cuda:
                        sample = unicore.utils.move_to_cuda(sample)
                    dist = sample["net_input"]["mol_src_distance"]
                    et = sample["net_input"]["mol_src_edge_type"]
                    st = sample["net_input"]["mol_src_tokens"]
                    mol_padding_mask = st.eq(model.mol_model.padding_idx)
                    mol_x = model.mol_model.embed_tokens(st)
                    n_node = dist.size(-1)
                    gbf_feature = model.mol_model.gbf(dist, et)
                    gbf_result = model.mol_model.gbf_proj(gbf_feature)
                    graph_attn_bias = gbf_result
                    graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
                    graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
                    mol_outputs = model.mol_model.encoder(
                        mol_x, padding_mask=mol_padding_mask, attn_mask=graph_attn_bias
                    )
                    mol_encoder_rep = mol_outputs[0][:,0,:]
                    mol_emb = model.mol_project(mol_encoder_rep)
                    mol_emb = mol_emb / mol_emb.norm(dim=-1, keepdim=True)
                    mol_emb = mol_emb.detach().cpu().numpy()
                    mol_reps.append(mol_emb)
                    mol_names.extend(sample["smi_name"])
                mol_reps = np.concatenate(mol_reps, axis=0)
                with open(mol_cache_path, "wb") as f:
                    pickle.dump([mol_reps, mol_ids_subsets], f)

            

            # generate pocket data
            pocket_dataset = self.load_pockets_dataset(pocket_data_path)
            pocket_data = torch.utils.data.DataLoader(pocket_dataset, batch_size=16, collate_fn=pocket_dataset.collater)
            pocket_reps = []

            #JT test:
            with open('data/targets/GPR35/PDB/pocket.pkl', 'rb') as f:
                pocket_repz, pocket_names = pickle.load(f)
            pocket_reps.append(pocket_repz[fold])


            for _, sample in enumerate(tqdm(pocket_data)):
                break #JT test
                if use_cuda:
                    sample = unicore.utils.move_to_cuda(sample)
                #sample = unicore.utils.move_to_cuda(sample)
                dist = sample["net_input"]["pocket_src_distance"]
                et = sample["net_input"]["pocket_src_edge_type"]
                st = sample["net_input"]["pocket_src_tokens"]
                pocket_padding_mask = st.eq(model.pocket_model.padding_idx)
                pocket_x = model.pocket_model.embed_tokens(st)
                n_node = dist.size(-1)
                gbf_feature = model.pocket_model.gbf(dist, et)
                gbf_result = model.pocket_model.gbf_proj(gbf_feature)
                graph_attn_bias = gbf_result
                graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
                graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
                pocket_outputs = model.pocket_model.encoder(
                    pocket_x, padding_mask=pocket_padding_mask, attn_mask=graph_attn_bias
                )
                pocket_encoder_rep = pocket_outputs[0][:,0,:]
                pocket_emb = model.pocket_project(pocket_encoder_rep)
                pocket_emb = pocket_emb / pocket_emb.norm(dim=-1, keepdim=True)
                pocket_emb = pocket_emb.detach().cpu().numpy()
                pocket_reps.append(pocket_emb)
            print(len(pocket_reps))
            pocket_reps = np.concatenate(pocket_reps, axis=0)
            # change reps to fp32
            mol_reps = mol_reps.astype(np.float32)
            pocket_reps = pocket_reps.astype(np.float32)

            res_list.append(pocket_reps @ mol_reps.T)



        res_new = np.array(res_list)
        
        res_new = np.mean(res_new, axis=0)

        if fold_version.startswith("6_folds"):
            medians = np.median(res_new, axis=1, keepdims=True)
            # get mad for each row
            mads = np.median(np.abs(res_new - medians), axis=1, keepdims=True)
            # get z score
            res_new = 0.6745 * (res_new - medians) / (mads + 1e-6)

        res_max = np.max(res_new, axis=0)
        # JT testing
#        ret_names = mol_names
        ret_names = ['test'] * 6596

        lis = []
        for i, score in enumerate(res_max):
            lis.append((score, ret_names[i]))
        lis.sort(key=lambda x:x[0], reverse=True)

        # get top 1%

        lis = lis[:int(len(lis) * 0.02)] 

        
        res_path = save_path
        with open(res_path, "w") as f:
            for score, name in lis:
                f.write(f"{name},{score}\n")

        return
        

        
    def perform_virtual_screen(self,
                            model,
                            pocket_data_path: str,
                            mol_data_path: str,
                            output_fpath: str,

                            fold_version: str='6_folds',
                            topN_percent: float=1,
                            use_cuda: bool=True,
                            torch_lig_batch_size: int=64,
                            torch_pocket_batch_size: int=16,
                            write_encodings: bool=True,
                            verbose: bool=True,
                            ) -> None:

        """Refactored retrieval_multi_folds() for easier usage
        Args: 
            model: model object produced by unicore. Let retrieval.py handle its input.
            pocket_data_path: A filepath str to an .lmdb or .pkl file containing protein pocket data. The pkl should be of the preencoded vectors, and if passed, the encoding step will be skipped.
            mol_data_path: A filepath str to an .lmdb or .pkl file containing ligand / chemical library data. The pkl should be of the preencoded vectors, and if passed, the encoding step will be skipped.
            output_fpath: filepath string for the outputted text file with the ligand smiles and scores.

        Kwargs:
            fold_version:  Which set of model parameters should be used. Value must be a one of the following: "6_folds", "8_folds", "6_folds_filtered".
            topN_percent:  A float, N, between 0 and 1 that determines which top N % of ligands get outputted. Set to 1 for all ligands to be outputted. 0.01 for top 1%, etc.
            torch_lig_batch_size:  When uploading the ligand dataset, upload it in batches of this size. Passed to Dataloader, see: https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
            torch_pocket_batch_size:  Same as for torch_lig_batch_size, but for the uploading of the pocket information. 
            write_encodings: Write the encoded vectors to a file
            verbose: Print performance and stats once the function has finished


        add checks on mol and pocket numbers
        print statements
        add timers
        checkpoints?
        encoding function
        dimensionality checks. fold dimensions? 
        does using the zscore on a per pocket basis, mean that we won't be able to compare score across pockets?
        """

        func_start = time.time()

        # Get the filepath strs of the model weights of for each fold (output is concensus of each fold (ithink))
        # Also get the filepath of the pre-encoded encodings of the default chem library
        if fold_version == "6_folds":
            fold_weights = [f"./data/model_weights/6_folds/fold_{i}.pt" for i in range(6)]
        
        elif fold_version == "8_folds":
            fold_weights = [f"./data/model_weights/8_folds/fold_{i}.pt" for i in range(8)]

        elif fold_version == "6_folds_filtered":
            fold_weights = [f"./data/model_weights/6_folds/fold_{i}.pt" for i in range(6)]

        else:
            raise Exception('The fold_version argument specfied is not recognised by drugclip it should be one of the following ["6_folds", "8_folds", "6_folds_filtered",]')


        # Check inputted paths to pocket and ligand data are the right format
        if not mol_data_path.endswith(('.pkl', '.lmdb')):
            raise Exception(f'The mol_data_path: {mol_data_path} does not specifcy an .lmdb or .pkl file')
        
        if not pocket_data_path.endswith(('.pkl', '.lmdb')):
            raise Exception(f'The pocket_data_path: {pocket_data_path} does not specify an .lmdb or .pkl file')
       

        # Get ligand data, and if needed, encode it. 
        mol_emb_start = time.time()
        if mol_data_path.endswith('.pkl'):
            print('Loading pre-encoded ligands')
            with open(mol_data_path, "rb") as f:
                mol_reps, mol_names = pickle.load(f)

        else:
            print('encoding ligands')
            # loop through folds and encode. pickle after
            # print new fpaths
            mol_reps = []
            mol_names = []
            for i, weights in enumerate(fold_weights):
                # Load the weights into the model
                state = checkpoint_utils.load_checkpoint_to_cpu(weights)
                model.load_state_dict(state["model"], strict=False)

                # load the .lmdb file                
                mol_dataset = self.load_mols_dataset(mol_data_path, "atoms", "coordinates")
                mol_data = torch.utils.data.DataLoader(mol_dataset, batch_size=torch_lig_batch_size, collate_fn=mol_dataset.collater)

                #Create embeddings
                embeddings, names = self._encode_mols_or_pockets(model, mol_data, use_cuda, 'mol', 'smi_name')

                print(f'fold-{i}, emb type, emb size, names type, n_names:', type(embeddings), len(embeddings), type(names), len(names),) 
         #       print(embeddings[0])

                mol_reps.append(embeddings)
                mol_names.extend(names)

            # Convert list of arrays to single array:
            print('mol reps len:', len(mol_reps))
        #    mol_reps = np.concatenate(mol_reps, axis=0)
            
            if write_encodings:
                outname = mol_data_path.replace('.lmdb', '.pkl')
                with open(outname, "wb") as f:
                    pickle.dump([mol_reps, mol_names], f) 

        mol_emb_end = time.time()

        #Get pocket data, and if needed, encode it. 
        if pocket_data_path.endswith('.pkl'):
            print('Using pre-encoded pocket file')
            with open(pocket_data_path, "rb") as f:
                pocket_reps, pocket_names = pickle.load(f)
        # Code is essentially the same as for mol, but a different uploader is used
        else:
            print('Encoding pockets ... ')
            pocket_reps = []
            pocket_names = []
            for weights in fold_weights:
                # Load the weights into the model
                state = checkpoint_utils.load_checkpoint_to_cpu(weights)
                model.load_state_dict(state["model"], strict=False)

                # load the .lmdb file                
                pocket_dataset = self.load_pockets_dataset(pocket_data_path)
                pocket_data = torch.utils.data.DataLoader(pocket_dataset, batch_size=torch_pocket_batch_size, collate_fn=pocket_dataset.collater)

                #Create embeddings
                embeddings, names = self._encode_mols_or_pockets(model, pocket_data, use_cuda, 'pocket', 'pocket_name')

                pocket_reps.append(embeddings)
                pocket_names.extend(names)

            # Convert list of arrays to single array:
#            pocket_reps = np.concatenate(pocket_reps, axis=0)
            
            if write_encodings:
                outname = pocket_data_path.replace('.lmdb', '.pkl')
                with open(outname, "wb") as f:
                    pickle.dump([pocket_reps, pocket_names], f) 

        pocket_emb_end = time.time()
        print('Encoding finished. Beginning post-processing... ')

        # Calculate distances between encoded pockets and ligands
        # change reps to fp32 #Reduced float precision, but why? Less memory usage?
        dist_calc_start = time.time()
        latent_space_distances = []
        for pocket_rep, mol_rep in zip(pocket_reps, mol_reps):
            latent_space_distances.append(pocket_rep.astype(np.float32) @  mol_rep.astype(np.float32).T)
        latent_space_distances = np.array(latent_space_distances)
        dist_calc_end = time.time()
        print('distances arr', latent_space_distances.shape)


        np.save('refactor_test/distances.npy', latent_space_distances)

        
        # Get mean dist of a ligand to each pocket structure
        latent_space_distance_means = np.mean(latent_space_distances, axis=0)

        # Normalise distances / convert to standardised z-scores 
        # Not sure why this is not done for 8folds (or why they need 8 folds for that matter)
        if fold_version.startswith("6_folds"):
            # Get the median distance across all ligands to pockets (i think - check dims)
            medians = np.median(latent_space_distance_means, axis=1, keepdims=True)
            # Get the median absolute deviation for each row (each ligand or pocket?)
            mads = np.median(np.abs(latent_space_distance_means - medians), axis=1, keepdims=True)
            # get z score (while the formula is correct, I do not know wehre the 0.6745 or the 1e-6 comes from)
            zscores = 0.6745 * (latent_space_distance_means - medians) / (mads + 1e-6)

        # Get the maximum score of each ligand across each pocket structure
        max_scores = np.max(zscores, axis=0)

        # Pair scores to their ligand names, and then sort by the lowest (closest distance) first
        ligand_scores_list = []
        for i, score in enumerate(max_scores):
            ligand_scores_list.append((score, mol_names[i]))
        ligand_scores_list.sort(key=lambda x:x[0], reverse=True)

        # Get the top N percent
        ligand_scores_list = ligand_scores_list[:int(len(ligand_scores_list) * topN_percent)] 

        # Output ligand scores to file
        write_start = time.time()
        with open(output_fpath, "w") as f:
            for score, name in ligand_scores_list:
                f.write(f"{name},{score}\n")
        write_end = time.time()

        # Print a report on what has been done and performance
        print('Done!')
        if verbose:
            print(f"""##### Virtual Screen Report #####
Total Interactions predicted:  {len(mol_reps) * len(pocket_reps)}
Number of ligands:             {len(mol_reps)}
Number of pockets:             {len(pocket_reps)}
Total function runtime:        {write_end - func_start}
Time per interaction:          {(write_end - write_start) / len(mol_reps) * len(pocket_reps)}
Ligand embedding time:         {mol_emb_end - mol_emb_start}
Pocket embedding time:         {pocket_emb_end - mol_emb_end}
distance calculation time:     {dist_calc_end - dist_calc_start}
output file write time:        {write_end - write_start}
####################################
""")
        return



    

    
    def _encode_mols_or_pockets(self, model, data, use_cuda, pocket_or_mol, label_str):

        """
        Generalised function for encoding either pockets or ligands

        The original drugclip code has the same encoding process for both pockets and ligands,
        but because they had type specific attribute names, they were not able to create a generic 
        function that they could call for either pockets or ligands. This causes a lot of repeated code
        that will be terrible to maintain in the future. Here's my clean up of it. 

        Args:
            model: model object that was originally passed to perform_virtual_screen()
            data: The data object produced by the pytorch DataLoader function 
            use_cuda: boolean. If true, send data to the gpu.
            pocket_or_mol: a string that is literally "pocket" or "mol". Used to call the right attr names
            label_str: string that corresponds to that used to label a pocket or ligand in the original .lmdb file.

        """





        data_model = getattr(model, f'{pocket_or_mol}_model')
        names = []
        embeddings = []

        for sample in tqdm(data):
            # move structure to gpu, if using
            if use_cuda:
                sample = unicore.utils.move_to_cuda(sample)

            # Get structure associated data
            dist = sample["net_input"][f"{pocket_or_mol}_src_distance"]
            et = sample["net_input"][f"{pocket_or_mol}_src_edge_type"]
            st = sample["net_input"][f"{pocket_or_mol}_src_tokens"]

            # Pad and do other preparations
            padding_mask = st.eq(data_model.padding_idx)
            x = data_model.embed_tokens(st)
            n_node = dist.size(-1)
            gbf_feature = data_model.gbf(dist, et)
            gbf_result = data_model.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)

            # Perform the encoding
            outputs = data_model.encoder(
                x, padding_mask=padding_mask, attn_mask=graph_attn_bias
            )
            encoder_rep = outputs[0][:,0,:]
            emb = getattr(model, f'{pocket_or_mol}_project')(encoder_rep)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            emb = emb.detach().cpu().numpy()

            embeddings.append(emb)
            print('sample type, len, keys', type(sample), len(sample), sample.keys())
            names.extend(sample[label_str])

        embeddings = np.concatenate(embeddings, axis=0)
        return embeddings, names

        
            
         

        
    
    
