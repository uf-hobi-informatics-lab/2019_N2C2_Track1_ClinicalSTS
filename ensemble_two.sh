export CUDA_VISIBLE_DEVICES=1
MODEL1_TYPE=model1_type
MODEL1_NAME=model1_name
MODEL2_TYPE=model2_type
MODEL2_NAME=model2_name

PATH_NAME=path_name
###### Step 1: pretrain on general corpus
CLASSIFIER=src/ensemble_task.py
STS_B_M=path/to/sts_b_merged_dataset
OUTPUT=output/ensemble_${PATH_NAME}_general

python $CLASSIFIER \
    --data_dir $STS_B_M \
    --model2_type $MODEL1_TYPE \
    --model2_name_or_path $MODEL1_NAME \
    --model1_type $MODEL2_TYPE \
    --model1_name_or_path $MODEL2_NAME \
    --task_name sts-b \
    --output_dir $OUTPUT \
    --max_seq_length 160 \
    --do_train \
    --overwrite_cache \
    --num_train_epochs 3 \

######## Step 2 : 5 fold cv
# CLASSIFIER=src/ensemble_task.py
# # sample0-4
# CLINICAL_DIR=/path/to/clinical_dataset_for_5_fold_cross_validation
# # general corpus pretrained model
# STS_MODEL=output/ensemble_${PATH_NAME}_general/pytorch_model.bin
# OUTPUT=output/ensemble_${PATH_NAME}/clinical/tmp
# for b in 4 8
# do
#     for ep in 3 4 5
#     do
#         for i in 0 1 2 3 4
#         do
#             echo "current hp: ${b}, ${ep}"
#             python $CLASSIFIER \
#                 --data_dir $CLINICAL_DIR/sample${i} \
#                 --model2_type $MODEL1_TYPE \
#                 --model2_name_or_path $MODEL1_NAME \
#                 --model1_type $MODEL2_TYPE \
#                 --model1_name_or_path $MODEL2_NAME \
#                 --ensemble_pretrained_model $STS_MODEL \
#                 --task_name sts-clinical \
#                 --output_dir $OUTPUT/${b}_${ep}/sample${i} \
#                 --max_seq_length 160 \
#                 --do_train \
#                 --per_gpu_train_batch_size ${b} \
#                 --num_train_epochs ${ep} \
#                 --do_eval \
#                 --overwrite_cache \
#                 # --overwrite_output_dir \
#         done
#     done
# done

######### Step 3 : refit
# CLASSIFIER=src/ensemble_task.py
# STS_C=/path/to/clinical_sts_trainset
# OUTPUT=output/ensemble_${PATH_NAME}_new/clinical_refit
# STS_MODEL=output/ensemble_${PATH_NAME}_general/pytorch_model.bin

# python $CLASSIFIER \
#     --data_dir $STS_C \
#      --model2_type $MODEL1_TYPE \
#     --model2_name_or_path $MODEL1_NAME \
#     --model1_type $MODEL2_TYPE \
#     --model1_name_or_path $MODEL2_NAME \
#     --ensemble_pretrained_model $STS_MODEL \
#     --task_name sts-clinical \
#     --output_dir $OUTPUT \
#     --max_seq_length 160 \
#     --do_train \
#     --per_gpu_train_batch_size 4 \
#     --num_train_epochs 3 \
#     --overwrite_cache \

####### prediction
# CLASSIFIER=src/ensemble_task.py
# STS_C=/path/to/clinical_sts_trainset
# OUTPUT=output/ensemble_${PATH_NAME}_clinical/clinical_pred
# STS_MODEL=output/ensemble_${PATH_NAME}_general/pytorch_model.bin

# python $CLASSIFIER \
#     --data_dir $STS_C \
#     --model2_type $MODEL1_TYPE \
#     --model2_name_or_path $MODEL1_NAME \
#     --model1_type $MODEL2_TYPE \
#     --model1_name_or_path $MODEL2_NAME \
#     --ensemble_pretrained_model $STS_MODEL \
#     --task_name sts-clinical \
#     --output_dir $OUTPUT \
#     --max_seq_length 160 \
#     --do_pred \
#     --per_gpu_train_batch_size 4 \
#     --overwrite_cache \
