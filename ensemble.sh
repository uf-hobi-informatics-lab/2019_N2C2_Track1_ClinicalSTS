export CUDA_VISIBLE_DEVICES=0

PATH_NAME=path_name # the folder name containing all the outputs
MODEL1_TYPE=model1_type # the type of the first model
MODEL1_NAME=model1_name_or_path # the pretrained model name or path of the first model
MODEL2_TYPE=model2_type
MODEL2_NAME=model2_name_or_path
MODEL3_TYPE=model3_type  # optional, set to "None" if you only want to ensemble two models
MODEL3_NAME=model3_name_or_path # optional, set to "None" if you only want to ensemble two models

CLASSIFIER=src/ensemble.py
STS_B_M=/path/to/sts_b_train_dev_merged  
GEN_OUTPUT=output/${PATH_NAME}/general
STS_MODEL=${GEN_OUTPUT}/pytorch_model.bin
CV_DIR=/path/to/clinical_dataset_for_4_fold_cross_validation
CV_OUTPUT=output/${PATH_NAME}/tmp
STS_C=/path/to/clinical_dataset
REFIT_OUTPUT=output/${PATH_NAME}/refit
REFIT_MODEL=${REFIT_OUTPUT}/pytorch_model.bin
PRED_OUTPUT=output/${PATH_NAME}/prediction

# step 1 pretrain on general corpus
python $CLASSIFIER \
    --data_dir $STS_B_M \
    --task_name sts-b \
    --model1_type $MODEL1_TYPE \
    --model1_name_or_path $MODEL1_NAME \
    --model2_type $MODEL2_TYPE \
    --model2_name_or_path $MODEL2_NAME \
    --model3_type $MODEL3_TYPE \
    --model3_type $MODEL3_NAME \
    --output_dir $GEN_OUTPUT \
    --per_gpu_train_batch_size 8 \
    --max_seq_length 160 \
    --do_train \
    --overwrite_cache \
    --overwrite_output_dir \
    --num_train_epochs 3

# step 2 5 fold cv on clinical training set
for b in 4 8 16
do
    for ep in 3 4 5
    do
        for i in 0 1 2 3 4
        do
            echo "current hp: ${b}, ${ep}"
            python $CLASSIFIER \
                --data_dir $CV_DIR/sample${i} \
                --model1_type $MODEL1_TYPE \
                --model1_name_or_path $MODEL1_NAME \
                --model2_type $MODEL2_TYPE \
                --model2_name_or_path $MODEL2_NAME \
                --model3_type $MODEL3_TYPE \
                --model3_type $MODEL3_NAME \
                --ensemble_pretrained_model $STS_MODEL \
                --task_name sts-clinical \
                --output_dir $CV_OUTPUT/${b}_${ep}/sample${i} \
                --max_seq_length 160 \
                --do_train \
                --per_gpu_train_batch_size ${b} \
                --num_train_epochs ${ep} \
                --do_eval \
                --overwrite_cache \
                --overwrite_output_dir \
        done
    done
done

# step 3 refit on clinical training set with the best hyperparameter got from 5-fold cross validation
python $CLASSIFIER \
    --data_dir $STS_C \
    --model1_type $MODEL1_TYPE \
    --model1_name_or_path $MODEL1_NAME \
    --model2_type $MODEL2_TYPE \
    --model2_name_or_path $MODEL2_NAME \
    --model3_type $MODEL3_TYPE \
    --model3_type $MODEL3_NAME \
    --ensemble_pretrained_model $STS_MODEL \
    --task_name sts-clinical \
    --output_dir $REFIT_OUTPUT \
    --max_seq_length 160 \
    --do_train \
    --per_gpu_train_batch_size best_batch_size_from_5fold_cv \
    --num_train_epochs best_train_epoch_num_from_5fold_cv \
    --overwrite_cache \
    --overwrite_output_dir \

# step 4 prediction

python $CLASSIFIER \
    --data_dir $STS_C \
    --model1_type $MODEL1_TYPE \
    --model1_name_or_path $MODEL1_NAME \
    --model2_type $MODEL2_TYPE \
    --model2_name_or_path $MODEL2_NAME \
    --model3_type $MODEL3_TYPE \
    --model3_type $MODEL3_NAME \
    --ensemble_pretrained_model $REFIT_MODEL \
    --task_name sts-clinical \
    --output_dir $PRED_OUTPUT \
    --max_seq_length 160 \
    --do_pred \
    --per_gpu_train_batch_size 4 \
    --overwrite_cache \
