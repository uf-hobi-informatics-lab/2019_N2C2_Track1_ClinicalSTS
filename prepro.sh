# preprocess raw dataset
python preprocess/prepro.py \
  --data_dir=path/to/clinical_sts_dataset \
  --output_dir=dir/to//output/clinical_sts_dataset

# generate datasets for cross validation
python preprocess/cross_valid_generate.py \
  --data_dir=path/to/processed_dataset \
  --output_dir=dir/to/output