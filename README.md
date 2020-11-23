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
Training and prediction processes are provided in the following scripts:  
[single.sh](https://github.com/uf-hobi-informatics-lab/2019_N2C2_Track1_ClinicalSTS/blob/master/single.sh) Using a single model  
[ensemble.sh](https://github.com/uf-hobi-informatics-lab/2019_N2C2_Track1_ClinicalSTS/blob/master/ensemble.sh) Using multi-model ensemble

## Evaluating 5 fold cross validation results
Use the script [cv_eval.sh](https://github.com/uf-hobi-informatics-lab/2019_N2C2_Track1_ClinicalSTS/blob/master/cv_eval.sh) to get the best hyperparameters (batch size and epoch number) based on the results of 5 fold cross validation.  
  ### Args
```
  --input_dir path    directory containing the results of 5 fold cross validation
  --output_dir path   directory to output the evaluation result
```

## Models
Theoretically support all models in https://huggingface.co/transformers/pretrained_models.html. However, we only used Bert, Roberta and XLNet in this task.

## Citation
- please cite our paper:
>https://medinform.jmir.org/2020/11/e19735/
```
Yang X, He X, Zhang H, Ma Y, Bian J, Wu Y
Measurement of Semantic Textual Similarity in Clinical Texts: Comparison of Transformer-Based Models
JMIR Med Inform 2020;8(11):e19735
DOI: 10.2196/19735
```
