

This repository is the final project of DSAA6000D,  which enables a conditional molecular optimization with deepinversion. 


## Installation 

```bash
conda create -n dst python=3.7 
conda activate dst
pip install torch 
pip install PyTDC 
conda install -c rdkit rdkit 
```

Activate conda environment. 
```bash
conda activate dst
```

make directory
```bash
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







# 6000D
