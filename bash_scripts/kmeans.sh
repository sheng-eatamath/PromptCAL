
DATASET=cub
FOLDER=cub
TRANSFORM='imagenet'
MODEL_NAME='vpt-model'
NUM_PROMPTS=5
N_SHALLOW_PROMPTS=0
PARALLEL='False'
EVAL_TEST='False'

EXP_NAME=vpt-${FOLDER}-surveillance
EXP_ID='stage2-gpu-test'
DEVICE='cuda:2'

SAVE_DIR=/home/sheng/PromptCAL/cache/$EXP_NAME/
MODEL_PATH=${SAVE_DIR}log/${EXP_ID}/checkpoints/model.pt

mkdir -p ${SAVE_DIR}kmeans

### ==============================================================================================

postfix_1='_best_'
postfix_2="_best_score"


python -m methods.clustering.extract_features --dataset ${DATASET} --use_best_model ${postfix_2} \
 --warmup_model_dir ${MODEL_PATH} --model_name ${MODEL_NAME} --transform ${TRANSFORM} \
 --num_prompts ${NUM_PROMPTS} --device ${DEVICE} \
 --n_shallow_prompts ${N_SHALLOW_PROMPTS} --with_parallel ${PARALLEL}

python -m methods.clustering.k_means --dataset ${DATASET} --semi_sup 'True' --use_ssb_splits 'True' \
 --use_best_model ${postfix_2} --max_kmeans_iter 200 --k_means_init 100 --warmup_model_exp_id ${EXP_ID} \
 --model_name ${MODEL_NAME} --device ${DEVICE} --eval_test ${EVAL_TEST} \
 >> ${SAVE_DIR}kmeans/${EXP_ID}_logfile_.out
