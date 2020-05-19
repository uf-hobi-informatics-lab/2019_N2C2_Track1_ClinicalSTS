# This script is used to train on two-model ensemble model (two from bert, xlnet and roberta)
export CUDA_VISIBLE_DEVICES=1
MODEL1_TYPE=model1_type
MODEL1_NAME=model1_name
MODEL2_TYPE=model2_type
MODEL2_NAME=model2_name

PATH_NAME=${MODEL1_NAME}_${MODEL2_NAME}
CLASSIFIER=src/ensemble_task.py
STS_B_M=path/to/sts_b_merged_dataset
GEN_OUTPUT=output/ensemble_${PATH_NAME}_general
CLINICAL_DIR=/path/to/clinical_dataset_for_5_fold_cross_validation
STS_MODEL=output/ensemble_${PATH_NAME}_general/pytorch_model.bin
CV_OUTPUT=output/ensemble_${PATH_NAME}/clinical/tmp
STS_C=/path/to/clinical_sts_dataset
REFIT_OUTPUT=output/ensemble_${PATH_NAME}_refit/
REFIT_MODEL=output/ensemble_${PATH_NAME}_refit/pytorch_model.bin
PRED_OUTPUT=output/ensemble_${PATH_NAME}_pred/

###### Step 1: pretrain on general corpus

python $CLASSIFIER \
    --data_dir $STS_B_M \
    --model2_type $MODEL1_TYPE \
    --model2_name_or_path $MODEL1_NAME \
    --model1_type $MODEL2_TYPE \
    --model1_name_or_path $MODEL2_NAME \
    --task_name sts-b \
    --output_dir $GEN_OUTPUT \
    --max_seq_length 160 \
    --do_train \
    --overwrite_cache \
    --num_train_epochs 3 \

######## Step 2 : 5 fold cv

for b in 4 8
do
    for ep in 3 4 5
    do
        for i in 0 1 2 3 4
        do
            echo "current hp: ${b}, ${ep}"
            python $CLASSIFIER \
                --data_dir $CLINICAL_DIR/sample${i} \
                --model2_type $MODEL1_TYPE \
                --model2_name_or_path $MODEL1_NAME \
                --model1_type $MODEL2_TYPE \
                --model1_name_or_path $MODEL2_NAME \
                --ensemble_pretrained_model $STS_MODEL \
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

######### Step 3 : refit

python $CLASSIFIER \
    --data_dir $STS_C \
     --model2_type $MODEL1_TYPE \
    --model2_name_or_path $MODEL1_NAME \
    --model1_type $MODEL2_TYPE \
    --model1_name_or_path $MODEL2_NAME \
    --ensemble_pretrained_model $STS_MODEL \
    --task_name sts-clinical \
    --output_dir $REFIT_OUTPUT \
    --max_seq_length 160 \
    --do_train \
    --per_gpu_train_batch_size 4 \
    --num_train_epochs 3 \
    --overwrite_cache \

####### prediction

python $CLASSIFIER \
    --data_dir $STS_C \
    --model2_type $MODEL1_TYPE \
    --model2_name_or_path $MODEL1_NAME \
    --model1_type $MODEL2_TYPE \
    --model1_name_or_path $MODEL2_NAME \
    --ensemble_pretrained_model $REFIT_MODEL \
    --task_name sts-clinical \
    --output_dir $PRED_OUTPUT \
    --max_seq_length 160 \
    --do_pred \
    --per_gpu_train_batch_size 4 \
    --overwrite_cache \
