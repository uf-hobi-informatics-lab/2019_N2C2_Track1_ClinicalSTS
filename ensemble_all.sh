# This script is used to train on bert-xlnet-roberta ensemble model
export CUDA_VISIBLE_DEVICES=0

CLASSIFIER=src/ensemble_all.py
STS_B_M=path/to/sts_b_merged_dataset
GEN_OUTPUT=output/ensemble_all
STS_MODEL=output/ensemble_all/pytorch_model.bin
CLINICAL_DIR=/path/to/clinical_dataset_for_5_fold_cross_validation
CV_OUTPUT=output/ensemble_all/clinical/tmp
STS_C=/path/to/clinical_sts_dataset
REFIT_OUTPUT=output/ensemble_all_refit_clinical
REFIT_MODEL=output/ensemble_all_refit_clinical/pytorch_model.bin
RRED_OUTPUT=output/ensemble_all_prediction/

###### Step 1: pretrain on general corpus
python $CLASSIFIER \
    --data_dir $STS_B_M \
    --task_name sts-b \
    --output_dir $GEN_OUTPUT \
    --max_seq_length 160 \
    --do_train \
    --per_gpu_train_batch_size 1 \
    --per_gpu_eval_batch_size 1 \
    --overwrite_cache \
    --num_train_epochs 3 \

######## Step 2 : 5 fold cv

for ep in 3 4 5
do
    for i in 0 1 2 3 4
    do
        echo "current hp: ${b}, ${ep}"
        python $CLASSIFIER \
            --data_dir $CLINICAL_DIR/sample${i} \
            --ensemble_pretrained_model $STS_MODEL \
            --task_name sts-clinical \
            --output_dir $CV_OUTPUT/${b}_${ep}/sample${i} \
            --max_seq_length 160 \
            --do_train \
            --per_gpu_train_batch_size 2 \
            --num_train_epochs ${ep} \
            --do_eval \
            --overwrite_cache \
            # --overwrite_output_dir \
    done
done

######## Step 3: refit

python $CLASSIFIER \
    --data_dir $STS_C \
    --ensemble_pretrained_model $STS_MODEL \
    --task_name sts-clinical \
    --output_dir $REFIT_OUTPUT \
    --max_seq_length 160 \
    --do_train \
    --per_gpu_train_batch_size 2 \
    --per_gpu_eval_batch_size 2 \
    --overwrite_cache \
    --num_train_epochs 3 \


# ###### Step 4 prediction

python $CLASSIFIER \
    --data_dir $STS_C \
    --ensemble_pretrained_model $REFIT_MODEL \
    --task_name sts-clinical \
    --output_dir $PRED_OUTPUT \
    --max_seq_length 160 \
    --do_pred \
    --per_gpu_eval_batch_size 2 \
    --overwrite_cache \
    --num_train_epochs 3 \