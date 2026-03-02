#!/bin/bash
#PBS -N refactor_test2
#PBS -lngpus=1
#PBS -lncpus=12
#PBS -q gpuvolta
#PBS -lmem=50GB
#PBS -ljobfs=50GB
#PBS -lwalltime=00:20:00
#PBS -P gc56
#PBS -l wd
#PBS -l storage=scratch/gc56+gdata/km21


# DrugCLIP inputs
#MOL_PATH="./data/targets/Test_4P42/testsdf.pkl" # path to the molecule file
MOL_PATH="./data/encoded_mol_embs/6_folds/combined.pkl"
POCKET_PATH="./data/targets/GPR35/PDB/pocket.pkl"
FOLD_VERSION=6_folds
use_cache=False 
save_path="refactor_test/refactor_output.txt"


# Python env setup
export PIP_CACHE_DIR=/scratch/gc56/jt2189/Software/.cache/pip/
export PYTHONUSERBASE=/scratch/gc56/jt2189/Software/pip
export PATH="${PATH}:/scratch/gc56/jt2189/Software/pip/bin" # for torch modules
export PYTHONPATH="${PYTHONPATH}:/scratch/gc56/jt2189/Software/DrugCLIP/Unicore/Uni-Core-main" #location of the unicore module

# Load conda env
source /scratch/gc56/jt2189/Software/miniconda3/bin/activate
conda init --all
conda activate drugclip

echo "data path: "
echo $data_path
echo "running drug clip .... "
# Run drugclip
CUDA_VISIBLE_DEVICES="0" python3 ./unimol/perform_virtual_screen.py --user-dir ./unimol $data_path "./dict" --valid-subset test \
       --num-workers 8 --ddp-backend=c10d --batch-size 4 \
       --task drugclip --loss in_batch_softmax --arch drugclip  \
       --max-pocket-atoms 511 \
       --cpu \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256  --seed 1 \
       --log-interval 100 --log-format simple \
       --mol-data-path $MOL_PATH \
       --pocket-data-path $POCKET_PATH \
       --fold-version $FOLD_VERSION \
       --topN-percent 0.02 \
       --output-fpath $save_path




