ROOT=r'/home/sheng/PromptCAL'
DATA_ROOT=r'/home/sheng/dataset'

# -----------------
# DATASET PATHS
# -----------------
cifar_10_root = f'{DATA_ROOT}/CIFAR10'
cifar_100_root = f'{DATA_ROOT}/CIFAR100'
cub_root = f'{DATA_ROOT}/CUB'
aircraft_root = f'{DATA_ROOT}/FGVCAircraft/fgvc-aircraft-2013b'
herbarium_dataroot = f'{DATA_ROOT}/herbarium_2019/'
imagenet_root = f'{DATA_ROOT}/imagenet-img'
imagenet_gcd_root = f'{DATA_ROOT}/imagenet_100_gcd'

# -----------------
# OTHER PATHS
# -----------------
osr_split_dir = f'{ROOT}/data/ssb_splits' # OSR Split dir
feature_extract_dir = f'{ROOT}/tmp'     # Extract features to this directory
exp_root = f'{ROOT}/cache'          # All logs and checkpoints will be saved here

# -----------------
# OTHER PATHS
# -----------------
dino_pretrain_path = r'/home/sheng/dino/checkpoint/dino_vitbase16_pretrain.pth'
ibot_pretrain_path = r'/home/sheng/dino/checkpoint/ibot-b16t.pth'


