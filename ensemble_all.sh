export CUDA_VISIBLE_DEVICES=0

###### Step 1: pretrain on general corpus
CLASSIFIER=src/ensemble_all.py
STS_B_M=path/to/sts_b_merged_dataset
OUTPUT=output/ensemble_all

python $CLASSIFIER \
    --data_dir $STS_B_M \
    --task_name sts-b \
    --output_dir $OUTPUT \
    --max_seq_length 160 \
    --do_train \
    --per_gpu_train_batch_size 1 \
    --per_gpu_eval_batch_size 1 \
    --overwrite_cache \
    --num_train_epochs 3 \

######## Step 2 : 5 fold cv
# CLASSIFIER=src/ensemble_task.py
# # sample0-4
# CLINICAL_DIR=/path/to/clinical_dataset_for_5_fold_cross_validation
# # general corpus pretrained model
# STS_MODEL=output/ensemble_all/pytorch_model.bin
# OUTPUT=output/ensemble_all/clinical/tmp

# for ep in 3 4 5
# do
#     for i in 0 1 2 3 4
#     do
#         echo "current hp: ${b}, ${ep}"
#         python $CLASSIFIER \
#             --data_dir $CLINICAL_DIR/sample${i} \
#             --ensemble_pretrained_model $STS_MODEL \
#             --task_name sts-clinical \
#             --output_dir $OUTPUT/${b}_${ep}/sample${i} \
#             --max_seq_length 160 \
#             --do_train \
#             --per_gpu_train_batch_size 2 \
#             --num_train_epochs ${ep} \
#             --do_eval \
#             --overwrite_cache \
#             # --overwrite_output_dir \
#     done
# done

######## Step 3: refit
# STS_C=/path/to/clinical_sts_trainset
# OUTPUT=output/ensemble_all_refit_clinical
# python $CLASSIFIER \
#     --data_dir $STS_C \
#     --ensemble_pretrained_model $STS_MODEL \
#     --task_name sts-clinical \
#     --output_dir $OUTPUT \
#     --max_seq_length 160 \
#     --do_train \
#     --per_gpu_train_batch_size 2 \
#     --per_gpu_eval_batch_size 2 \
#     --overwrite_cache \
#     --num_train_epochs 3 \


# ###### STep 4 prediction
# STS_C=/path/to/clinical_sts_trainset
# OUTPUT=output/ensemble_all/clinical_pred
# STS_MODEL=output/ensemble_all_refit_clinical/pytorch_model.bin

# python $CLASSIFIER \
#     --data_dir $STS_C \
#     --ensemble_pretrained_model $STS_MODEL \
#     --task_name sts-clinical \
#     --output_dir $OUTPUT \
#     --max_seq_length 160 \
#     --do_pred \
#     --per_gpu_eval_batch_size 2 \
#     --overwrite_cache \
#     --num_train_epochs 3 \