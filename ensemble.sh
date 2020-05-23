export CUDA_VISIBLE_DEVICES=0

MODEL1_TYPE=bert
MODEL1_NAME=bert-base-cased
MODEL2_TYPE=roberta
MODEL2_NAME=roberta-large

CLASSIFIER=src/ensemble.py
STS_B_M=/home/ma.yingha/workspace/py3/2019challenge/2019-N2C2-challenge/Track1/train_data/STS-B-M
GEN_OUTPUT=output/ensemble_1


# step 1
python $CLASSIFIER \
    --data_dir $STS_B_M \
    --task_name sts-b \
    --model1_type $MODEL1_TYPE \
    --model1_name_or_path $MODEL1_NAME \
    --model2_type $MODEL2_TYPE \
    --model2_name_or_path $MODEL2_NAME \
    --output_dir $GEN_OUTPUT \
    --per_gpu_train_batch_size 2 \
    --max_seq_length 128 \
    --do_pred \
    --overwrite_cache \
    --num_train_epochs 1
