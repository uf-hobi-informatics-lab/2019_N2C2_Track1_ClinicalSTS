# 2019 N2C2 Track-1 Clinical Semantic Textual Similarity


## best RoBERTa model (0.9065)
https://transformer-models.s3.amazonaws.com/2019n2c2_tack1_roberta_pt_stsc_6b_16b_3c_8c.zip

## Environment
* Python 3.7.3
* Pytorch 1.1.0
* Transformers 2.5.1

## Dataset
General corpus: Semantic Textual Similarity Benchmark dataset from GLUE Benchmark [download](https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FSTS-B.zip?alt=media&token=bddb94a7-8706-4e0d-a694-1109e12273b5)  
Clinical corpus: 2019 N2C2 Challenge Track 1 [website](https://n2c2.dbmi.hms.harvard.edu/track1)
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
[single.sh](https://github.com/uf-hobi-informatics-lab/2019_N2C2_Track1_ClinicalSTS/blob/master/single.sh) Using a single model  
[ensemble.sh](https://github.com/uf-hobi-informatics-lab/2019_N2C2_Track1_ClinicalSTS/blob/master/ensemble.sh) Using multi-model ensemble

## Models
Theoretically support all models in https://huggingface.co/transformers/pretrained_models.html. However, we only used Bert, Roberta and XLNet in this task.
