# This script is used to train on a single model
export CUDA_VISIBLE_DEVICES=0
PATH_NAME=path_name
MODEL_TYPE=model_type
MODEL_NAME=model_name
CLASSIFIER=src/single_task.py
STS_B_M=path/to/sts_b_merged_dataset
GEN_OUTPUT=output/$PATH_NAME/gen
STS_MODEL=$GEN_OUTPUT

CLINICAL_DIR=/path/to/clinical_dataset_for_5_fold_cross_validation
CV_OUTPUT=output/$PATH_NAME/tmp

STS_C=/path/to/clinical_sts_trainset
REFIT_OUTPUT=output/$PATH_NAME/clinical_refit

STS_C_TEST=/path/to/clinical_sts_testset
FINAL_OUTPUT=output/$PATH_NAME/clinical_pred
REFIT_MODEL=output/$PATH_NAME/clinical_refit/


# step 1: pretrain on general corpus

python $CLASSIFIER \
    --data_dir $STS_B_M \
    --model_type $MODEL_TYPE \
    --model_name_or_path $MODEL_NAME \
    --task_name sts-b \
    --output_dir $GEN_OUTPUT \
    --max_seq_length 160 \
    --do_train \
    --overwrite_cache \


# step 2: conduct 5 fold cross validation on clinical dataset

for b in 4 8 16
do
    for ep in 3 4 5
    do
        for i in 0 1 2 3 4
        do
            echo "current hp: ${b}, ${ep}"
            python $CLASSIFIER \
                --data_dir $CLINICAL_DIR/sample${i} \
                --model_type $MODEL_TYPE \
                --model_name_or_path $STS_MODEL \
                --task_name sts-clinical \
                --output_dir $CV_OUTPUT/${b}_${ep}/sample${i} \
                --max_seq_length 160 \
                --do_train \
                --per_gpu_train_batch_size ${b} \
                --num_train_epochs ${ep} \
                --do_eval \
                --overwrite_cache \
                # --overwrite_output_dir \
        done
    done
done

# step 3: refit on clinical dataset using 5f cv best hyperparameter
python $CLASSIFIER \
    --data_dir $STS_C \
    --model_type $MODEL_TYPE \
    --model_name_or_path $STS_MODEL \
    --task_name sts-clinical \
    --output_dir $REFIT_OUTPUT \
    --max_seq_length 160 \
    --do_train \
    --per_gpu_train_batch_size 4 \
    --num_train_epochs 4 \
    --overwrite_cache \


# step 4: prediction
python $CLASSIFIER \
    --data_dir $STS_C_TEST \
    --model_type $MODEL_TYPE \
    --model_name_or_path $REFIT_MODEL \
    --task_name sts-clinical \
    --output_dir $FINAL_OUTPUT \
    --max_seq_length 160 \
    --do_pred \
    --per_gpu_train_batch_size 4 \
    --num_train_epochs 3 \
    --overwrite_cache \
