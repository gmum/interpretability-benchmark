from typing import List, Dict

import cv2
import torch
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent
from torch import nn
import numpy as np

from spatial_alignment_test.ppnet_wrapper import PPNetAdversarialWrapper
from helpers import find_high_activation_crop
from preprocess import mean, std
from settings import img_size


def get_all_class_proto_low_activation_bbox_mask(
        proto_nums: List[np.ndarray],
        activations: np.ndarray,
        epsilon_pixels: int = 10,
) -> np.ndarray:
    """
    Get a mask that has 0 on pixels within high activation bounding box of any of the ground truth prototypes.
    :param proto_nums: list of prototype numbers to attack for each image
    :param activations: a tensor of prototype activations over image patches of shape [B, P, Wp, Hp]
    :param epsilon_pixels: number of border pixels to add to the high activation bounding box
    """
    assert len(proto_nums) == activations.shape[0]
    img_mask = np.ones((activations.shape[0], 1, img_size, img_size), dtype=np.float32)

    for sample_i in range(activations.shape[0]):
        for proto_num in proto_nums[sample_i]:
            proto_patch_activation = activations[sample_i, proto_num, :, :]
            proto_pixel_activation = cv2.resize(proto_patch_activation,
                                                dsize=(img_size, img_size),
                                                interpolation=cv2.INTER_CUBIC)
            high_act_original_proto = find_high_activation_crop(proto_pixel_activation)

            img_mask[sample_i, :,
            max(high_act_original_proto[0] - epsilon_pixels, 0):
            min(high_act_original_proto[1] + epsilon_pixels, img_size - 1),
            max(high_act_original_proto[2] - epsilon_pixels, 0):
            min(high_act_original_proto[3] + epsilon_pixels, img_size - 1)
            ] = 0
    return img_mask


def attack_images_target_class_prototypes(
        model: nn.Module,
        img: torch.tensor,
        activations: np.ndarray,
        attack_type: str,
        cls: np.ndarray,
        focal_sim: bool,
        epsilon: float = 0.1,
        epsilon_iter: float = 0.01,
        nb_iter: int = 20,
) -> Dict:
    """
    Adversarially attack activations of prototypes of the ground truth class for a given image.
    We will exclude from the region of the attack the high activation bounding box of the ground truth class prototypes.
    :param model: PPNet model
    :param img: a batch of images to attack [B, C, W, H]
    :param activations: a numpy array with prototype activations over image patches of shape [B, P, Wp, Hp]
    :param attack_type: type of attack, in terms of the attacked prototypes
    :param cls: a vector ground truth classes of the images
    :param epsilon: maximum perturbation of the adversarial attack
    :param epsilon_iter: maximum perturbation of the adversarial attack within one iteration
    :param nb_iter: number of iterations of the adversarial attack
    :return: a dictionary contained the modified images and the mask of the region of the attack
    """
    proto_cls_identity = model.prototype_class_identity.cpu().detach().numpy()
    cls_proto_nums = [np.argwhere(proto_cls_identity[:, c] == 1).flatten() for c in cls]
    if attack_type == 'gt_protos':
        proto_nums = cls_proto_nums
    elif attack_type == 'top_proto':
        proto_nums = []
        for sample_act in activations:
            proto_max_act = np.max(sample_act.reshape(sample_act.shape[0], -1), axis=-1)
            proto_max_act = np.argmax(proto_max_act)
            proto_nums.append(np.asarray([int(proto_max_act)]))
    else:
        raise ValueError(attack_type)

    mask = get_all_class_proto_low_activation_bbox_mask(
        proto_nums=proto_nums,
        activations=activations
    )
    mask = torch.tensor(mask, device=img.device)

    img_modified, activations_before, activations_after = [], [], []
    for sample_i in range(img.shape[0]):
        sample_img = img[sample_i].unsqueeze(0)
        sample_proto_nums = proto_nums[sample_i]
        sample_mask = mask[sample_i].unsqueeze(0)
        wrapper = PPNetAdversarialWrapper(model=model, img=sample_img, proto_nums=sample_proto_nums, mask=sample_mask,
                                          focal_sim=focal_sim)

        sample_modified = projected_gradient_descent(
            model_fn=wrapper,
            x=sample_img,
            eps=epsilon,
            eps_iter=epsilon_iter,
            nb_iter=nb_iter,
            norm=np.inf,
        )
        img_modified.append(sample_modified)
        activations_before.append(np.clip(wrapper.initial_activation, a_min=0.0, a_max=None))
        activations_after.append(np.clip(wrapper.final_activation, a_min=0.0, a_max=None))

    img_modified = torch.cat(img_modified, dim=0)
    img_modified = img_modified * mask + img * (1 - mask)

    activations_before = np.concatenate(activations_before, axis=0)
    activations_after = np.concatenate(activations_after, axis=0)

    img_modified_numpy = img_modified.clone().cpu().detach().numpy()
    for d in range(3):
        img_modified_numpy[:, d] = (img_modified_numpy[:, d] * std[d] + mean[d])
    # the returned image batch has RGB values between 0 and 1.0
    img_modified_numpy = img_modified_numpy.clip(0, 1)

    return {
        'img_modified_numpy': img_modified_numpy,
        'img_modified_tensor': img_modified.detach(),
        'mask': mask.cpu().detach().numpy(),
        'proto_nums': proto_nums,
        'activations_before': activations_before,
        'activations_after': activations_after,
        'cls_proto_nums': cls_proto_nums,
    }
