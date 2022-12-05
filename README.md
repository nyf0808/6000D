

This repository is the final project of DSAA6000D,  which enables a conditional molecular optimization with deepinversion. 

- src
  - chemutils.py (Chemical tools
  - clean.py (Clean data
  - denovo.py (Generate molecule
  - download.py (Download Dataset
  - dpp.py  (Dpp algorithm
  - evaluate.py (test
  - gnn_layer.py (gnn
  - inference_utils.py (Tools related to deepinversion
  - labelling.py (label data
  - module.py (gnn
  - train.py (train gnn
  - utils.py
  - vocabulary.py (Decomposition of molecules into vocabularies
## Installation 

```bash
conda create -n di python=3.7 
conda activate di
pip install torch 
pip install PyTDC 
conda install -c rdkit rdkit 
```

Activate conda environment. 
```bash
conda activate di
mkdir -p save_model result 
```

## Raw Data 
Download `ZINC`.
```bash
python src/download.py
```
Download Oracle.

## Generate Vocabulary and labelling
```bash 
python src/vocabulary.py
python src/clean.py 
python src/labelling.py
```

## training

```bash 
python src/train.py qed 500
```

## molecule generation

```bash 
python src/denovo.py qed 500
```

## evaluate 

```bash 
python src/evaluate.py qed
```
