"""
An experiment that attacks activation of prototypes adversarially.
Its goal is to determine whether the prototypes are local.
If they are local, the adversarial attack should not be successful.

Example usage:

python run_adversarial_test.py results/2023_01_23_resnet_34_mask_high_act_1/push_best.pth --model_keys baseline --output_dir experiment1
"""
import json
import argparse
import os
from collections import defaultdict
from typing import List
import pandas as pd

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from spatial_alignment_test.adversarial_modification import attack_images_target_class_prototypes
from spatial_alignment_test.run_on_dataset import run_model_on_dataset, run_model_on_batch, normalize
from protopool.model import PrototypeChooser
from settings import results_dir, test_dir, img_size


def get_heatmaps_with_same_normalization(*patch_activations) -> List[np.ndarray]:
    upsampled_activations = []
    for patch_activation in patch_activations:
        upsampled_activations.append(cv2.resize(patch_activation,
                                                dsize=(img_size, img_size),
                                                interpolation=cv2.INTER_CUBIC))
    amin, amax = np.inf, -np.inf
    for act in upsampled_activations:
        amin = min(amin, np.amin(act))
    for act in upsampled_activations:
        amax = max(amax, np.max(act - amin))

    heatmaps = []
    for act in upsampled_activations:
        rescaled_act = act - amin
        rescaled_act = rescaled_act / amax

        heatmap = cv2.applyColorMap(np.uint8(255 * rescaled_act), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        heatmaps.append(heatmap[..., ::-1])
    return heatmaps


def get_activation_change_metrics(act_before, act_after, proto_nums, cls_proto_nums):
    metrics = {}

    protos_act_before = act_before[proto_nums]
    protos_act_after = act_after[proto_nums]

    # for each prototype, get its maximum activation over all patches
    max_activations_before = np.max(protos_act_before.reshape(protos_act_before.shape[0], -1), axis=-1)
    max_activations_after = np.max(protos_act_after.reshape(protos_act_after.shape[0], -1), axis=-1)

    # as a metric, calculate activation change of the top activated prototype
    argmax_act = np.argmax(max_activations_before)
    top_proto_act_before, top_proto_act_after = float(max_activations_before[argmax_act]), \
        float(max_activations_after[argmax_act])

    # metrics['top_proto_act_before'] = top_proto_act_before
    # metrics['top_proto_act_after'] = top_proto_act_after
    # metrics['top_proto_act_diff'] = top_proto_act_after - top_proto_act_before
    metrics['PAC'] = (1 - top_proto_act_after / top_proto_act_before) * 100

    # as a metric, calculate relative change of "place" in argsort over all prototypes, of the top activated prototype
    max_activations_before = np.max(act_before.reshape(act_before.shape[0], -1), axis=-1)
    max_activations_after = np.max(act_after.reshape(act_after.shape[0], -1), axis=-1)

    # argmax_place_before = float(np.sum(max_activations_before > max_activations_before[proto_nums[argmax_act]]))
    # argmax_place_after = float(np.sum(max_activations_after > max_activations_after[proto_nums[argmax_act]]))

    # metrics['top_proto_place_before'] = argmax_place_before
    # metrics['top_proto_place_after'] = argmax_place_after
    # metrics['top_proto_place_diff'] = argmax_place_after - argmax_place_before

    cls_proto_nums = set(cls_proto_nums)
    non_cls_proto_nums = np.asarray([i for i in range(act_before.shape[0]) if i not in cls_proto_nums])
    argmax_place_before = float(np.sum(max_activations_before[non_cls_proto_nums] >
                                       max_activations_before[proto_nums[argmax_act]]))
    argmax_place_after = float(np.sum(max_activations_after[non_cls_proto_nums] >
                                      max_activations_after[proto_nums[argmax_act]]))
    # metrics['non_cls_protos_higher_than_top_proto_before'] = argmax_place_before
    # metrics['non_cls_protos_higher_than_top_proto_after'] = argmax_place_after
    # metrics['non_cls_protos_higher_than_top_proto_diff'] = argmax_place_after - argmax_place_before
    metrics['PRC'] = argmax_place_after - argmax_place_before

    # get mIOU of high activated region before anda after the modification
    protos_act_before = act_before[proto_nums[argmax_act]]
    protos_act_after = act_after[proto_nums[argmax_act]]

    proto_act_before = torch.nn.functional.interpolate(torch.tensor(protos_act_before).unsqueeze(0).unsqueeze(0),
                                                       size=(img_size, img_size),
                                                       mode='bilinear')
    proto_act_before = proto_act_before.squeeze(0).squeeze(0)
    proto_act_after = torch.nn.functional.interpolate(torch.tensor(protos_act_after).unsqueeze(0).unsqueeze(0),
                                                      size=(img_size, img_size),
                                                      mode='bilinear')
    proto_act_after = proto_act_after.squeeze(0).squeeze(0)

    quantile_before = torch.quantile(proto_act_before.flatten(), q=0.9, dim=-1)
    high_act_before = proto_act_before > quantile_before
    quantile_after = torch.quantile(proto_act_after.flatten(), q=0.9, dim=-1)
    high_act_after = proto_act_after > quantile_after

    if torch.sum(high_act_after | high_act_before).item() != 0:
        iou = torch.sum(high_act_after & high_act_before) / torch.sum(high_act_after | high_act_before)
        iou = float(iou.item())
        metrics['PLC'] = 100*(1-iou)
    else:
        metrics['PLC'] = 0.0

    # if len(proto_nums) > 1:
        # same metric as above but for all the prototypes of the target class
        # places_before, places_after, acts_before, acts_after = [], [], [], []
        # for proto_num in proto_nums:
            # places_before.append(float(np.sum(max_activations_before > max_activations_before[proto_num])))
            # places_after.append(float(np.sum(max_activations_after > max_activations_after[proto_num])))

            # acts_before.append(float(max_activations_before[proto_num]))
            # acts_after.append(float(max_activations_after[proto_num]))

        # metrics['proto_place_before'] = places_before
        # metrics['proto_place_after'] = places_after
        # metrics['proto_place_diff'] = [p1 - p2 for p1, p2 in zip(places_after, places_before)]

        # metrics['proto_act_before'] = acts_before
        # metrics['proto_act_after'] = acts_after
        # metrics['proto_act_diff'] = [a1 - a2 for a1, a2 in zip(acts_after, acts_before)]

    return metrics


def run_adversarial_attack_on_prototypes(args):
    experiment_output_dir = os.path.join(results_dir, args.output_dir)
    os.makedirs(experiment_output_dir, exist_ok=True)

    test_dataset = datasets.ImageFolder(
        test_dir,
        transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ]))

    np.random.seed(20230527)

    if args.n_samples != -1:
        random_idx = np.random.choice(np.arange(len(test_dataset)), replace=False, size=args.n_samples)
    else:
        random_idx = np.random.choice(np.arange(len(test_dataset)), replace=False, size=len(test_dataset))

    subset_test_dataset = Subset(test_dataset, random_idx)
    setattr(subset_test_dataset, 'samples', [test_dataset.samples[i] for i in random_idx])
    test_dataset = subset_test_dataset

    metrics_mean, metrics_all = {}, {}
    for ch_path, model_key, is_proto_pool in zip(args.model_checkpoints, args.model_keys, args.proto_pool):
        proto_pool_arch = 'resnet50' if is_proto_pool == '2' else 'resnet34'
        is_proto_pool = is_proto_pool == '1' or is_proto_pool == '2'

        print(f'Loading model {model_key} from {ch_path}...')
        if is_proto_pool:
            # TODO use model config instead of hard-coding
            model = PrototypeChooser(
                num_prototypes=202,
                num_descriptive=10,
                num_classes=200,
                use_thresh=True,
                arch=proto_pool_arch,
                inat=proto_pool_arch == 'resnet50',
                pretrained=True,
                add_on_layers_type='log',
                prototype_activation_function='log',
                proto_depth=256,
                use_last_layer=True,
            )
            if torch.cuda.is_available():
                model.load_state_dict(torch.load(ch_path)['model_state_dict'])
                model = model.cuda()
            else:
                model.load_state_dict(torch.load(ch_path, map_location=torch.device('cpu'))['model_state_dict'])
        else:
            if torch.cuda.is_available():
                model = torch.load(ch_path).cuda()
            else:
                model = torch.load(ch_path, map_location=torch.device('cpu'))
            if args.focal_sim:
                setattr(model, 'focal_sim', True)

        model_output_dir = os.path.join(experiment_output_dir, model_key)
        # output_adv_img_dir_summaries = os.path.join(model_output_dir, 'adversarial_images_summaries')
        # os.makedirs(output_adv_img_dir_summaries, exist_ok=True)

        output_top_k_dir = os.path.join(model_output_dir, 'adversarial_images_summaries_cherrypicked')
        os.makedirs(output_top_k_dir, exist_ok=True)

        pbar = tqdm(total=len(test_dataset))

        n_samples, n_correct_before, n_correct_after = 0, 0, 0
        metrics = defaultdict(list)

        top_k_examples, top_k_examples_diffs = [], []

        for batch_result in run_model_on_dataset(
                model=model,
                dataset=test_dataset,
                num_workers=args.n_jobs,
                batch_size=args.batch_size,
                proto_pool=is_proto_pool,
        ):
            adversarial_result = attack_images_target_class_prototypes(
                model=model,
                img=batch_result['img_tensor'],
                activations=batch_result['patch_activations'],
                attack_type=args.attack_type,
                cls=batch_result['target'],
                epsilon=args.epsilon,
                epsilon_iter=args.epsilon_iter,
                nb_iter=args.nb_iter,
                focal_sim=args.focal_sim or is_proto_pool
            )

            n_samples += len(batch_result['filenames'])
            n_correct_before += np.sum(batch_result['predicted_cls'] == batch_result['target'])

            with torch.no_grad():
                predicted_cls_adv, patch_activations_adv = run_model_on_batch(
                    model=model, batch=adversarial_result['img_modified_tensor'], proto_pool=is_proto_pool
                )

            n_correct_after += np.sum(predicted_cls_adv == batch_result['target'])

            for sample_i in range(len(batch_result['filenames'])):
                filename = batch_result['filenames'][sample_i]
                img_original = batch_result['img_original_numpy'][sample_i]
                img_modified = adversarial_result['img_modified_numpy'][sample_i]
                sample_mask = adversarial_result['mask'][sample_i]
                proto_nums = adversarial_result['proto_nums'][sample_i]
                cls_proto_nums = adversarial_result['cls_proto_nums'][sample_i]

                img_original = img_original.transpose(1, 2, 0)
                img_modified = img_modified.transpose(1, 2, 0)
                sample_mask = sample_mask.transpose(1, 2, 0)

                alpha = 0.7
                modified_masked = img_modified * sample_mask + \
                                  (1 - sample_mask) * (alpha * sample_mask + (1 - alpha) * img_modified)

                activation_before = batch_result['patch_activations'][sample_i, proto_nums]
                activation_after = patch_activations_adv[sample_i, proto_nums]
                total_activation_before = np.sum(activation_before, axis=0)
                total_activation_after = np.sum(activation_after, axis=0)

                heatmap, heatmap_modified = get_heatmaps_with_same_normalization(
                    total_activation_before, total_activation_after
                )

                for metric_key, val in get_activation_change_metrics(batch_result['patch_activations'][sample_i],
                                                                     patch_activations_adv[sample_i],
                                                                     proto_nums, cls_proto_nums).items():
                    if isinstance(val, list):
                        metrics[metric_key].extend(val)
                    else:
                        metrics[metric_key].append(val)

                overlayed_img_original = 0.5 * img_original + 0.3 * heatmap
                overlayed_img_modified = 0.5 * img_modified + 0.3 * heatmap_modified

                all_img = [img_original, overlayed_img_original, img_modified, modified_masked, overlayed_img_modified]
                all_img_desc = ['original', 'original_heatmap', 'modified', 'attack_mask', 'modified_heatmap']

                # uncomment to save all individual images
                # output_adv_img_dir_all = os.path.join(model_output_dir, 'adversarial_images_all')
                # os.makedirs(output_adv_img_dir_all, exist_ok=True)
                # extension = 'jpg'
                # for im, desc in zip(all_img, all_img_desc):
                # plt.figure(figsize=(5, 5))
                # plt.imshow(im, vmin=0, vmax=1)
                # plt.axis('off')
                # plt.savefig(os.path.join(output_adv_img_dir_all,
                # filename.replace(f'.{extension}', f'_{desc}.{extension}')),
                # bbox_inches='tight', pad_inches=0)
                # plt.close()

                # plt.figure(figsize=(25, 5))
                # for img_i, (im, desc) in enumerate(zip(all_img, all_img_desc)):
                # plt.subplot(1, len(all_img), img_i + 1)
                # plt.imshow(im, vmin=0, vmax=1)
                # plt.title(desc)
                # plt.axis('off')
                # plt.tight_layout()
                # plt.savefig(os.path.join(output_adv_img_dir_summaries, filename),
                # bbox_inches='tight', pad_inches=0.2)
                # plt.close()

                # save some cherry-picked samples where activation change is the biggest
                top_proto_act_diff = metrics['PAC'][-1]
                if len(top_k_examples) < args.top_k_save or any(k > top_proto_act_diff for k in top_k_examples_diffs):
                    if len(top_k_examples) >= args.top_k_save:
                        argmax = np.argmax(top_k_examples_diffs)
                        os.remove(os.path.join(output_top_k_dir, top_k_examples[argmax]))
                        top_k_examples.pop(argmax)
                        top_k_examples_diffs.pop(argmax)

                    top_k_examples.append(filename)
                    top_k_examples_diffs.append(top_proto_act_diff)

                    plt.figure(figsize=(25, 5))
                    for img_i, (im, desc) in enumerate(zip(all_img, all_img_desc)):
                        plt.subplot(1, len(all_img), img_i + 1)
                        plt.imshow(im, vmin=0, vmax=1)
                        plt.title(desc)
                        plt.axis('off')
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_top_k_dir, filename),
                                bbox_inches='tight', pad_inches=0.2)

                plt.close()
                pbar.update()

            acc1 = n_correct_before / n_samples * 100
            acc2 = n_correct_after / n_samples * 100
            pbar.set_description('{:s}. Acc before: {:.2f}%, after: {:.2f}%)'.format(model_key, acc1, acc2))

        pbar.close()

        with open(os.path.join(model_output_dir, 'metrics_all.json'), 'w') as f:
            json.dump(metrics, f)

        mean_metrics = {k: float(np.mean(v)) for k, v in metrics.items()}
        mean_metrics['Acc_before'] = float(n_correct_before / n_samples * 100)
        mean_metrics['Acc_after'] = float(n_correct_after / n_samples * 100)
        mean_metrics['AC'] = mean_metrics['Acc_before'] - mean_metrics['Acc_after']
        with open(os.path.join(model_output_dir, 'metrics_mean.json'), 'w') as f:
            json.dump(mean_metrics, f, indent=2)

        metrics_mean[model_key] = mean_metrics
        metrics_all[model_key] = metrics

    metrics_df = defaultdict(list)
    for model_key, metrics in metrics_mean.items():
        metrics_df['model'].append(model_key)
        for metric_key, val in metrics.items():
            metrics_df[metric_key].append(float(np.round(val, 2)))
    pd.DataFrame(metrics_df).to_csv(os.path.join(experiment_output_dir, 'metrics.csv'), index=False)

    histograms_dir = os.path.join(experiment_output_dir, 'histograms')
    os.makedirs(histograms_dir, exist_ok=True)
    for metric_key in metrics_all[args.model_keys[0]].keys():
        model_values = [metrics_all[model][metric_key] for model in args.model_keys]

        plt.figure(figsize=(10, 5))
        for model_key, values in zip(args.model_keys, model_values):
            plt.hist(values, alpha=0.5, label=model_key)
        plt.title(metric_key)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(histograms_dir, f'{metric_key}.png'))
        plt.close()


if __name__ == '__main__':
    np.random.seed(1234)

    parser = argparse.ArgumentParser(description='Adversarially attack prototypes')
    parser.add_argument('model_checkpoints', nargs='+', type=str,
                        help='Paths to the checkpoints (.pth files) of the evaluated models')
    parser.add_argument('--model_keys', nargs='+', type=str, help='Names for the models to display in plot titles')
    parser.add_argument('--proto_pool', nargs='+', type=str,
                        help='Whether the models are ProtoPool. '
                             '"1" for ProtoPool with resnet 34, "2" for ProtoPool '
                             'with resnet50 and iNaturalist pretraining, otherwise ProtoPNet')

    parser.add_argument('--output_dir', type=str, help='Name of the output directory in RESULTS_PATH')

    parser.add_argument('--n_samples', type=int, default=-1, help='Number of samples (-1 == all test set)')
    parser.add_argument('--n_jobs', type=int, default=8, help='Number of parallel jobs (for DataLoader)')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for using the model')
    parser.add_argument('--top_k_save', type=int, default=50, help='Number of Top K most changed images to save')

    parser.add_argument('--attack_type', type=str, default='top_proto',
                        help='Attack type: '
                             '"top_proto" - attack top predicted prototype, '
                             '"gt_protos" - attack all prototypes of the ground truth class'
                        )

    # parameters for the adversarial attack
    parser.add_argument('--epsilon', type=float, default=0.4,
                        help='Maximum perturbation of the adversarial attack')
    parser.add_argument('--epsilon_iter', type=float, default=0.01,
                        help='Maximum perturbation of the adversarial attack within one iteration')
    parser.add_argument('--nb_iter', type=iter, default=40,
                        help='Number of iterations of the adversarial attack')

    parser.add_argument("--focal_sim", type=bool, action=argparse.BooleanOptionalAction)
    parser.set_defaults(focal_sim=False)

    run_adversarial_attack_on_prototypes(parser.parse_args())
