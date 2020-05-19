# 2019 N2C2 Track-1 Clinical Semantic Textual Similarity


## best RoBERTa model (0.9065)
https://transformer-models.s3.amazonaws.com/2019n2c2_tack1_roberta_pt_stsc_6b_16b_3c_8c.zip

## Environment
* Python 3.7.3
* Pytorch 1.1.0
* Transformers 2.5.1

## Dataset


## Preprocess
* Preprocess clinical dataset
```bash
python preprocess/prepro.py \
  --data_dir=path/to/clinical_sts_dataset \
  --output_dir=dir/to//output/clinical_sts_dataset
```
* Generate datasets for five fold cross validation
```bash
python preprocess/cross_valid_generate.py \
  --data_dir=path/to/processed_dataset \
  --output_dir=dir/to/output
```

## Training
We provided options of three models: Bert, XLNet, Roberta

Training and prediction processes are provided in the following scripts:  
[single.sh] Using a single model  
[ensemble&lowbar;two.sh] Using two-model ensembling  
[ensemble&lowbar;all.sh] Using all-model ensembling  
