# BGCA

This repo contains the data and code for our paper "Bidirectional Generative Framework for Cross-domain Aspect-based Sentiment Analysis" (BGCA) in ACL 2023.

## Requirements

This repo is developed using the following packages:
- transformers==4.18.0
- sentencepiece==0.1.96
- pytorch_lightning==0.8.1
- editdistance==0.6.0
- scikit-learn==0.24.2
- numpy==1.22.3
- tqdm==4.64.0

## Usage
We conduct experiments on four ABSA tasks:
1. ATE
```
cd code
bash ../scripts/run_ate.sh
```
2. UABSA
```
cd code
bash ../scripts/run_uasa.sh
```
3. AOPE
```
cd code
bash ../scripts/run_aope.sh
```
4. ASTE
```
cd code
bash ../scripts/run_aste.sh
```

## Code Sturcture
* ```constants.py``` Contains constant variables
* ```data_utils.py``` Contains code to prepare input & output for generative model
* ```eval_utils.py``` Contains code to extract sentiment elements and calculate metric
* ```main.py``` Contains code for the main function
* ```model_utils.py``` Contains code for model initialization
* ```preprocess.py``` Contains code to preprocess different task's data (ATE and UABSA share the same data)
* ```run_utils.py``` Contains code for training, where **data_gene()** is the key method for this repo.
* ```setup.py``` Contains code for setup such as args parsing

## Note
1. Extract model in the code refers to text-to-label stage in the paper, and Gene model refers to label-to-text stage.


## Citation
If the code is used in your research, please star our repo and cite our paper as follows:
```
@misc{deng2023bidirectional,
      title={Bidirectional Generative Framework for Cross-domain Aspect-based Sentiment Analysis}, 
      author={Yue Deng and Wenxuan Zhang and Sinno Jialin Pan and Lidong Bing},
      year={2023},
      eprint={2305.09509},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
