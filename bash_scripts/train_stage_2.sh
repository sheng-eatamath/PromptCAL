### fine-grained [scars, cub, aircraft]
### ==============================================================================================

export OMP_NUM_THREADS=4

HOME=/home/sheng/PromptCAL
save_dir=scars
dataset_name=scars
EXP_NAME=log-${save_dir}-surveillance
EXP_ID='stage2-gpu-test'

echo ${EXP_NAME}/${EXP_ID} >> ${HOME}/cache/exp_track.log

SAVE_DIR=$HOME/cache/$EXP_NAME/
mkdir -p $SAVE_DIR
LOG_DIR=$HOME/cache/$EXP_NAME/$EXP_ID/
mkdir -p $LOG_DIR


checkpoint_exp=stage1-gpu-test_1
checkpoint_model=model_best
checkpoint_head=model_proj_head_best

nohup \
python -m methods.contrastive_training.contrastive_training_2 \
            --seed 0 \
            --device 'cuda:2' \
            --epochs 100 \
            --lr 0.1 \
            --runner_name $EXP_NAME \
            --exp_id $EXP_ID \
            --eval_interval 1 \
            --eval_interval_t 10 \
            --kmeans_interval -1 \
            --use_fast_kmeans 'True' \
            --use_val 'False' \
            --val_split 0.1 \
            --dataset_name ${dataset_name} \
            --prop_train_labels 0.5 \
            --use_vpt 'True' \
            --num_prompts 5 \
            --predict_token 'cop' \
            --num_dpr 2 \
            --w_prompt_clu 0.35 \
            --knn 5 \
            --w_knn_loss 0.6 \
            --diffusion_q 0.8 \
            --load_from_model ${HOME}/cache/log-${save_dir}-surveillance/log/${checkpoint_exp}/checkpoints/${checkpoint_model}.pt \
            --load_from_head ${HOME}/cache/log-${save_dir}-surveillance/log/${checkpoint_exp}/checkpoints/${checkpoint_head}.pt \
>> ${LOG_DIR}logfile_${EXP_NUM}.out 2>&1 &


### cifar100, cifar10
### ==============================================================================================

export OMP_NUM_THREADS=4

HOME=/home/sheng/PromptCAL
save_dir=cifar100
dataset_name=cifar100
EXP_NAME=log-${save_dir}-surveillance
EXP_ID='stage2-gpu-test'

echo ${EXP_NAME}/${EXP_ID} >> ${HOME}/cache/exp_track.log

SAVE_DIR=$HOME/cache/$EXP_NAME/
mkdir -p $SAVE_DIR
LOG_DIR=$HOME/cache/$EXP_NAME/$EXP_ID/
mkdir -p $LOG_DIR


checkpoint_exp=stage1-gpu-test_1
checkpoint_model=model_best
checkpoint_head=model_proj_head_best

nohup \
python -m methods.contrastive_training.contrastive_training_2 \
            --device 'cuda:2' \
            --epochs 70 \
            --lr 0.1 \
            --runner_name $EXP_NAME \
            --exp_id $EXP_ID \
            --eval_interval 1 \
            --eval_interval_t 10 \
            --kmeans_interval -1 \
            --use_fast_kmeans 'True' \
            --use_val 'False' \
            --val_split 0.1 \
            --dataset_name ${dataset_name} \
            --prop_train_labels 0.5 \
            --use_vpt 'True' \
            --num_prompts 5 \
            --predict_token 'cop' \
            --num_dpr 2 \
            --w_prompt_clu 0.35 \
            --knn 10 \
            --w_knn_loss 0.6 \
            --diffusion_q 0.5 \
            --load_from_model ${HOME}/cache/log-${save_dir}-surveillance/log/${checkpoint_exp}/checkpoints/${checkpoint_model}.pt \
            --load_from_head ${HOME}/cache/log-${save_dir}-surveillance/log/${checkpoint_exp}/checkpoints/${checkpoint_head}.pt \
>> ${LOG_DIR}logfile_${EXP_NUM}.out 2>&1 &


### imagenet_100_gcd
### ==============================================================================================

export OMP_NUM_THREADS=4

HOME=/home/sheng/PromptCAL
save_dir=imagenet_100_gcd
dataset_name=imagenet_100_gcd
EXP_NAME=log-${save_dir}-surveillance
EXP_ID='stage2-gpu-test'

echo ${EXP_NAME}/${EXP_ID} >> ${HOME}/cache/exp_track.log

SAVE_DIR=$HOME/cache/$EXP_NAME/
mkdir -p $SAVE_DIR
LOG_DIR=$HOME/cache/$EXP_NAME/$EXP_ID/
mkdir -p $LOG_DIR


checkpoint_exp=stage1-gpu-test_1
checkpoint_model=model_best
checkpoint_head=model_proj_head_best

nohup \
python -m methods.contrastive_training.contrastive_training_2 \
            --device 'cuda:2' \
            --epochs 70 \
            --lr 0.1 \
            --runner_name $EXP_NAME \
            --exp_id $EXP_ID \
            --eval_interval 2 \
            --eval_interval_t 10 \
            --kmeans_interval 1 \
            --use_fast_kmeans 'False' \
            --use_val 'False' \
            --val_split 0.1 \
            --dataset_name ${dataset_name} \
            --prop_train_labels 0.5 \
            --use_vpt 'True' \
            --num_prompts 5 \
            --predict_token 'cop' \
            --num_dpr 2 \
            --w_prompt_clu 0.35 \
            --knn 10 \
            --w_knn_loss 0.6 \
            --diffusion_q 0.5 \
            --load_from_model ${HOME}/cache/log-${save_dir}-surveillance/log/${checkpoint_exp}/checkpoints/${checkpoint_model}.pt \
            --load_from_head ${HOME}/cache/log-${save_dir}-surveillance/log/${checkpoint_exp}/checkpoints/${checkpoint_head}.pt \
>> ${LOG_DIR}logfile_${EXP_NUM}.out 2>&1 &


### herbarium_19
### ==============================================================================================

export OMP_NUM_THREADS=4

HOME=/home/sheng/PromptCAL
save_dir=herbarium_19
dataset_name=herbarium_19
EXP_NAME=log-${save_dir}-surveillance
EXP_ID='stage2-gpu-test'

echo ${EXP_NAME}/${EXP_ID} >> ${HOME}/cache/exp_track.log

SAVE_DIR=$HOME/cache/$EXP_NAME/
mkdir -p $SAVE_DIR
LOG_DIR=$HOME/cache/$EXP_NAME/$EXP_ID/
mkdir -p $LOG_DIR


checkpoint_exp=stage1-gpu-test_1
checkpoint_model=model_best
checkpoint_head=model_proj_head_best

nohup \
python -m methods.contrastive_training.contrastive_training_2 \
            --device 'cuda:2' \
            --epochs 100 \
            --lr 0.1 \
            --runner_name $EXP_NAME \
            --exp_id $EXP_ID \
            --eval_interval 1 \
            --eval_interval_t 10 \
            --kmeans_interval 1 \
            --use_fast_kmeans 'False' \
            --use_val 'False' \
            --val_split 0.1 \
            --dataset_name ${dataset_name} \
            --prop_train_labels 0.5 \
            --use_vpt 'True' \
            --num_prompts 5 \
            --predict_token 'cop' \
            --num_dpr 2 \
            --w_prompt_clu 0.35 \
            --knn 7 \
            --w_knn_loss 0.6 \
            --membank_size 20000 \
            --neg_sample 5000 \
            --diffusion_q 0.8 \
            --load_from_model ${HOME}/cache/log-${save_dir}-surveillance/log/${checkpoint_exp}/checkpoints/${checkpoint_model}.pt \
            --load_from_head ${HOME}/cache/log-${save_dir}-surveillance/log/${checkpoint_exp}/checkpoints/${checkpoint_head}.pt \
>> ${LOG_DIR}logfile_${EXP_NUM}.out 2>&1 &

### Inductive GNCD
### ==============================================================================================

export OMP_NUM_THREADS=4

HOME=/home/sheng/PromptCAL
save_dir=cub
dataset_name=cub
EXP_NAME=log-${save_dir}-surveillance
EXP_ID='val-stage2-gpu-test'

echo ${EXP_NAME}/${EXP_ID} >> ${HOME}/cache/exp_track.log

SAVE_DIR=$HOME/cache/$EXP_NAME/
mkdir -p $SAVE_DIR
LOG_DIR=$HOME/cache/$EXP_NAME/$EXP_ID/
mkdir -p $LOG_DIR


checkpoint_exp=val-stage1-gpu-test_1
checkpoint_model=model_best
checkpoint_head=model_proj_head_best

nohup \
python -m methods.contrastive_training.contrastive_training_2 \
            --device 'cuda:2' \
            --epochs 100 \
            --lr 0.1 \
            --runner_name $EXP_NAME \
            --exp_id $EXP_ID \
            --eval_interval 1 \
            --eval_interval_t 10 \
            --kmeans_interval -1 \
            --use_fast_kmeans 'True' \
            --use_val 'True' \
            --val_split 0.1 \
            --dataset_name ${dataset_name} \
            --prop_train_labels 0.5 \
            --use_vpt 'True' \
            --num_prompts 5 \
            --predict_token 'cop' \
            --num_dpr 2 \
            --w_prompt_clu 0.35 \
            --knn 5 \
            --w_knn_loss 0.6 \
            --diffusion_q 0.8 \
            --load_from_model ${HOME}/cache/log-${save_dir}-surveillance/log/${checkpoint_exp}/checkpoints/${checkpoint_model}.pt \
            --load_from_head ${HOME}/cache/log-${save_dir}-surveillance/log/${checkpoint_exp}/checkpoints/${checkpoint_head}.pt \
>> ${LOG_DIR}logfile_${EXP_NUM}.out 2>&1 &