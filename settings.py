import os

base_architecture = os.environ.get('BASE_ARCHITECTURE', 'resnet34')
img_size = 224
prototype_shape = (2000, 128, 1, 1)
prototype_activation_function = 'log'
add_on_layers_type = 'regular'
use_last_two_blocks = False

num_workers = 8

# on GMUM
data_path = os.environ['DATA_PATH']
train_dir = os.path.join(data_path, os.environ['TRAIN_DIR'])
test_dir = os.path.join(data_path, os.environ['TEST_DIR'])
train_push_dir = os.path.join(data_path, os.environ['TRAIN_PUSH_DIR'])
results_dir = os.path.join(os.environ['RESULTS_DIR'])

# local
# data_path = '/media/mikolaj/HDD/ml_data/CUB/'
# train_dir = data_path + 'train_birds/train_birds/train_birds/'
# test_dir = data_path + 'test_birds/test_birds/test_birds/'
# train_push_dir = data_path + 'train_birds/train_birds/train_birds/'
# results_dir = '/media/mikolaj/HDD/local_prototypes'

if 'cars' in data_path:
    prototype_shape = (1960, 128, 1, 1)
    num_classes = 196
else:
    prototype_shape = (2000, 128, 1, 1)
    num_classes = 200

train_batch_size = 80
test_batch_size = 100
train_push_batch_size = 75

joint_optimizer_lrs = {'features': 1e-4,
                       'add_on_layers': 3e-3,
                       'prototype_vectors': 3e-3}
joint_lr_step_size = 5

warm_optimizer_lrs = {'add_on_layers': 3e-3,
                      'prototype_vectors': 3e-3}

last_layer_optimizer_lr = 1e-4

coefs = {
    'crs_ent': 1,
    'clst': 0.8,
    'sep': -0.08,
    'l1': 1e-4,
    'sim_diff_random': float(os.environ.get('SIM_DIFF', 1)),
    'sim_diff_high_act': float(os.environ.get('SIM_DIFF', 10))
}

num_train_epochs = 1000
num_warm_epochs = 5
max_num_cycles = 5

push_start = 10
push_epochs = [i for i in range(num_train_epochs) if i % 10 == 0]

NEPTUNE_API_TOKEN = os.environ.get('NEPTUNE_API_TOKEN', '')

masking_random_prob = 0.8
