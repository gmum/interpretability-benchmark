import os
import shutil

import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import argparse
import re

from helpers import makedir
import model
import push
import train_and_test as tnt
import save
from log import create_logger
from preprocess import mean, std, preprocess_input_function
from settings import NEPTUNE_API_TOKEN, max_num_cycles, masking_random_prob
import neptune.new as neptune

parser = argparse.ArgumentParser()
parser.add_argument('--experiment_run', type=str, default='001')
parser.add_argument('-gpuid', nargs=1, type=str, default='0')  # python3 main.py -gpuid=0,1,2,3
parser.add_argument('--last_layer_num', type=int, default=-1)
parser.add_argument('--masking_type', type=str, default='none')
parser.add_argument('--sim_diff_weight_annealing', type=bool, default=False)
parser.add_argument('--sim_diff_function', type=str, default='l1')
parser.add_argument("--quantized_mask", type=bool, action=argparse.BooleanOptionalAction)
parser.set_defaults(quantized_mask=True)
parser.add_argument("--mixup", type=bool, action=argparse.BooleanOptionalAction)
parser.set_defaults(mixup=False)
parser.add_argument("--focal_sim", type=bool, action=argparse.BooleanOptionalAction)
parser.set_defaults(focal_sim=False)

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]

# book keeping namings and code
from settings import base_architecture, img_size, prototype_shape, num_classes, \
    prototype_activation_function, add_on_layers_type, results_dir, num_workers

base_architecture_type = re.match('^[a-z]*', base_architecture).group(0)
model_dir = os.path.join(results_dir, base_architecture, args.experiment_run) + '/'

makedir(model_dir)
shutil.copy(src=os.path.join(os.getcwd(), __file__), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'settings.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), base_architecture_type + '_features.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'model.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'train_and_test.py'), dst=model_dir)

with open(os.path.join(model_dir, 'last_layer_num.txt'), 'w') as f:
    f.write(str(args.last_layer_num))

log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log'))
img_dir = os.path.join(model_dir, 'img')
makedir(img_dir)
weight_matrix_filename = 'outputL_weights'
prototype_img_filename_prefix = 'prototype-img'
prototype_self_act_filename_prefix = 'prototype-self-act'
proto_bound_boxes_filename_prefix = 'bb'

# load the data
from settings import train_dir, test_dir, train_push_dir, \
    train_batch_size, test_batch_size, train_push_batch_size

normalize = transforms.Normalize(mean=mean,
                                 std=std)

# all datasets
# train set
train_dataset = datasets.ImageFolder(
    train_dir,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ]))
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=train_batch_size, shuffle=True,
    num_workers=num_workers, pin_memory=False)
# push set
train_push_dataset = datasets.ImageFolder(
    train_push_dir,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
    ]))
train_push_loader = torch.utils.data.DataLoader(
    train_push_dataset, batch_size=train_push_batch_size, shuffle=False,
    num_workers=4, pin_memory=False)
# test set
test_dataset = datasets.ImageFolder(
    test_dir,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ]))
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=test_batch_size, shuffle=False,
    num_workers=4, pin_memory=False)

# we should look into distributed sampler more carefully at torch.utils.data.distributed.DistributedSampler(train_dataset)
# construct the model
ppnet = model.construct_PPNet(base_architecture=base_architecture,
                              pretrained=True, img_size=img_size,
                              prototype_shape=prototype_shape,
                              num_classes=num_classes,
                              prototype_activation_function=prototype_activation_function,
                              add_on_layers_type=add_on_layers_type,
                              last_layer_num=args.last_layer_num,
                              mixup=args.mixup,
                              focal_sim=args.focal_sim)
# if prototype_activation_function == 'linear':
#    ppnet.set_last_layer_incorrect_connection(incorrect_strength=0)
ppnet = ppnet.cuda()
ppnet_multi = torch.nn.DataParallel(ppnet)
class_specific = True

# define optimizer
from settings import joint_optimizer_lrs, joint_lr_step_size

joint_optimizer_specs = \
    [{'params': ppnet.features.parameters(), 'lr': joint_optimizer_lrs['features'], 'weight_decay': 1e-3},
     # bias are now also being regularized
     {'params': ppnet.add_on_layers.parameters(), 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
     {'params': ppnet.prototype_vectors, 'lr': joint_optimizer_lrs['prototype_vectors']},
     ]
joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=joint_lr_step_size, gamma=0.1)

from settings import warm_optimizer_lrs

warm_optimizer_specs = \
    [{'params': ppnet.add_on_layers.parameters(), 'lr': warm_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
     {'params': ppnet.prototype_vectors, 'lr': warm_optimizer_lrs['prototype_vectors']},
     ]
warm_optimizer = torch.optim.Adam(warm_optimizer_specs)

from settings import last_layer_optimizer_lr

last_layer_optimizer_specs = [{'params': ppnet.last_layer.parameters(), 'lr': last_layer_optimizer_lr}]
last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)

# weighting of different training losses
from settings import coefs

# number of training epochs, number of warm epochs, push start epoch, push epochs
from settings import num_train_epochs, num_warm_epochs, push_start, push_epochs

if isinstance(NEPTUNE_API_TOKEN, str) and len(NEPTUNE_API_TOKEN) > 0:
    log('initializing neptune')
    neptune_run = neptune.init_run(
        project='mikolajsacha/protobased-research',
        name=args.experiment_run,
        api_token=NEPTUNE_API_TOKEN,
        tags=['local_prototypes']
    )
    params = {
        "masking_type": args.masking_type,
        "num_train_epochs": num_train_epochs,
        "num_warm_epochs": num_warm_epochs,
        "max_num_cycles": max_num_cycles,
        "coefs": coefs,
        "joint_optimizer_lrs": joint_optimizer_lrs,
        "joint_optimizer_step_size": joint_lr_step_size,
        "last_layer_optimizer_lr": last_layer_optimizer_lr,
        "warm_optimizer_lrs": warm_optimizer_lrs,
        "base_architecture": base_architecture,
        "img_size": img_size,
        "prototype_shape": prototype_shape,
        "num_classes": num_classes,
        "prototype_activation_function": prototype_activation_function,
        "add_on_layers_type": add_on_layers_type,
        "num_workers": num_workers,
        "train_batch_size": train_batch_size,
        "test_batch_size": test_batch_size,
        "train_push_batch_size": train_push_batch_size,
        "push_start": push_start,
        "push_epochs": push_epochs,
        'masking_random_prob': masking_random_prob,
        'quantized_mask': args.quantized_mask,
        'sim_diff_weight_annealing': args.sim_diff_weight_annealing,
        'sim_diff_function': args.sim_diff_function,
    }
    neptune_run["parameters"] = params
else:
    neptune_run = None

log('training set size: {0}'.format(len(train_loader.dataset)))
log('push set size: {0}'.format(len(train_push_loader.dataset)))
log('test set size: {0}'.format(len(test_loader.dataset)))
log('batch size: {0}'.format(train_batch_size))

log('start training')

max_accu_no_push = 0.0
max_accu_push = 0.0
max_accu_finetune = 0.0
n_cycle = 0
min_num_epochs = 20

if args.masking_type == 'random':
    max_sim_diff_weight = coefs['sim_diff_random']
elif args.masking_type == 'high_act' or args.masking_type == 'high_act_aug':
    max_sim_diff_weight = coefs['sim_diff_high_act']
else:
    max_sim_diff_weight = 0.0

for epoch in range(num_train_epochs):
    if args.sim_diff_weight_annealing:
        sim_diff_weight = min(max_sim_diff_weight / min_num_epochs * epoch, max_sim_diff_weight)
    else:
        sim_diff_weight = max_sim_diff_weight
    if neptune_run is not None:
        neptune_run["train/sim_diff_weight"].append(sim_diff_weight)

    if epoch < num_warm_epochs:
        tnt.warm_only(model=ppnet_multi, log=log)
        train_accu, converged, metrics = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=warm_optimizer,
                                                   class_specific=class_specific, coefs=coefs, log=log,
                                                   masking_type=args.masking_type, neptune_run=neptune_run,
                                                   quantized_mask=args.quantized_mask, sim_diff_weight=sim_diff_weight,
                                                   sim_diff_function=args.sim_diff_function, mixup=args.mixup)
    else:
        tnt.joint(model=ppnet_multi, log=log)
        joint_lr_scheduler.step()
        train_accu, converged, metrics = tnt.train(model=ppnet_multi, dataloader=train_loader,
                                                   optimizer=joint_optimizer,
                                                   class_specific=class_specific, coefs=coefs, log=log,
                                                   masking_type=args.masking_type, neptune_run=neptune_run,
                                                   quantized_mask=args.quantized_mask, sim_diff_weight=sim_diff_weight,
                                                   sim_diff_function=args.sim_diff_function, mixup=args.mixup)
    if neptune_run is not None:
        neptune_run["train/epoch/accuracy"].append(train_accu)
        neptune_run["train/epoch/stage"].append(0.0 if epoch < num_warm_epochs else 1.0)
        neptune_run["train/epoch/converged"].append(float(int(converged)))
        for key, val in metrics.items():
            neptune_run[f"train/epoch/{key}"].append(float(val))

    accu, _, metrics = tnt.test(model=ppnet_multi, dataloader=test_loader,
                                class_specific=class_specific, log=log, masking_type=args.masking_type,
                                neptune_run=neptune_run, quantized_mask=args.quantized_mask,
                                sim_diff_weight=sim_diff_weight, sim_diff_function=args.sim_diff_function)

    if neptune_run is not None:
        neptune_run["test/epoch/accuracy"].append(accu)
        for key, val in metrics.items():
            neptune_run[f"test/epoch/{key}"].append(float(val))

    if accu > max_accu_no_push:
        log(f"Cycle {n_cycle} - new best test accuracy: {accu:.2f}")
        save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name='nopush_best', accu=accu,
                                    target_accu=0.10, log=log, cycle=n_cycle)
        max_accu_no_push = accu

    save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name='nopush_last', accu=accu,
                                target_accu=0.10, log=log, cycle=n_cycle)

    if epoch >= push_start and epoch in push_epochs:
        push.push_prototypes(
            train_push_loader,  # pytorch dataloader (must be unnormalized in [0,1])
            prototype_network_parallel=ppnet_multi,  # pytorch network with prototype_vectors
            class_specific=class_specific,
            preprocess_input_function=preprocess_input_function,  # normalize if needed
            prototype_layer_stride=1,
            root_dir_for_saving_prototypes=img_dir,  # if not None, prototypes will be saved here
            epoch_number=None,  # if not provided, prototypes saved previously will be overwritten
            prototype_img_filename_prefix=prototype_img_filename_prefix,
            prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
            proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
            save_prototype_class_identity=True,
            log=log)
        accu, _, metrics = tnt.test(model=ppnet_multi, dataloader=test_loader,
                                    class_specific=class_specific, log=log, masking_type=args.masking_type,
                                    neptune_run=neptune_run, quantized_mask=args.quantized_mask,
                                    sim_diff_weight=sim_diff_weight, sim_diff_function=args.sim_diff_function)

        if accu > max_accu_push:
            save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name='push_best', accu=accu,
                                        target_accu=0.10, log=log, cycle=n_cycle)
            max_accu_push = accu

        save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name='push_last', accu=accu,
                                    target_accu=0.10, log=log, cycle=n_cycle)

        if prototype_activation_function != 'linear':
            tnt.last_only(model=ppnet_multi, log=log)
            for i in range(20):
                # log('iteration: \t{0}'.format(i))
                train_accu, converged, metrics = tnt.train(model=ppnet_multi, dataloader=train_loader,
                                                           optimizer=last_layer_optimizer,
                                                           class_specific=class_specific,
                                                           coefs=coefs, log=log, masking_type=args.masking_type,
                                                           neptune_run=neptune_run, quantized_mask=args.quantized_mask,
                                                           sim_diff_weight=sim_diff_weight,
                                                           sim_diff_function=args.sim_diff_function,
                                                           mixup=args.mixup)

                if neptune_run is not None:
                    neptune_run["train/epoch/accuracy"].append(train_accu)
                    neptune_run["train/epoch/stage"].append(2.0)
                    neptune_run["train/epoch/converged"].append(float(int(converged)))

                    for key, val in metrics.items():
                        neptune_run[f"train/epoch/{key}"].append(float(val))

                accu, _, metrics = tnt.test(model=ppnet_multi, dataloader=test_loader,
                                            class_specific=class_specific, log=log, masking_type=args.masking_type,
                                            neptune_run=neptune_run, quantized_mask=args.quantized_mask,
                                            sim_diff_weight=sim_diff_weight, sim_diff_function=args.sim_diff_function)

                if neptune_run is not None:
                    neptune_run["test/epoch/accuracy"].append(accu)
                    for key, val in metrics.items():
                        neptune_run[f"test/epoch/{key}"].append(float(val))

            if accu > max_accu_finetune:
                save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name='push_finetune_best',
                                            accu=accu, target_accu=0.10, log=log, cycle=n_cycle)
                max_accu_finetune = accu
            save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name='push_finetune_last',
                                        accu=accu, target_accu=0.10, log=log, cycle=n_cycle)

        if train_accu > 0.99 and converged and epoch > min_num_epochs:
            print("EARLY STOPPING")
            break

        # reset metrics after each cycle
        max_accu_no_push = 0.0
        max_accu_push = 0.0
        max_accu_finetune = 0.0
        n_cycle += 1

        if n_cycle >= max_num_cycles:
            print("REACHED MAXIMUM NUMBER OF CYCLES ({})".format(max_num_cycles))
            break

print()
print(f'{args.experiment_run} ACCURACIES: ')
print("nopush: {:.4f}".format(max_accu_no_push))
print("push: {:.4f}".format(max_accu_push))
print("push_finetune: {:.4f}".format(max_accu_finetune))
print()

logclose()

if neptune_run is not None:
    neptune_run.stop()
