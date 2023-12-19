import time
from typing import Tuple

import torch
import numpy as np

from helpers import list_of_distances
from settings import masking_random_prob, img_size


def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.
    index = torch.randperm(x.shape[0], dtype=x.dtype, device=x.device).to(torch.long)
    mixed_x = lam * x + (1 - lam) * x[index, ...]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def _train_or_test(model, dataloader, optimizer=None, class_specific=True, use_l1_mask=True,
                   coefs=None, log=print, masking_type='none', neptune_run=None,
                   quantized_mask=False, sim_diff_weight=0.0, sim_diff_function='l1',
                   mixup: bool = False):
    '''
    model: the multi-gpu model
    dataloader:
    optimizer: if None, will be test evaluation
    '''
    is_train = optimizer is not None
    n_examples = 0
    n_correct = 0
    n_batches = 0
    total_loss = 0.0
    total_cross_entropy = 0
    total_cluster_cost = 0
    # separation cost is meaningful only for class_specific
    total_separation_cost = 0
    total_avg_separation_cost = 0
    total_sim_diff_loss = 0.0

    for i, (image, label) in enumerate(dataloader):
        input = image.cuda()
        target = label.cuda()

        if mixup:
            input, targets_a, targets_b, lam = mixup_data(input, target, 0.5)

        # torch.enable_grad() has no effect outside of no_grad()
        grad_req = torch.enable_grad() if is_train else torch.no_grad()

        with grad_req:
            # nn.Module has implemented __call__() function
            # so no need to call .forward
            if masking_type == 'random_no_loss' or masking_type == 'high_act_aug' and is_train:
                min_box_size = img_size // 8
                max_box_size = img_size // 2
                masking_prob = 0.5
                max_num_boxes = 5

                with torch.no_grad():
                    # TODO move this to the Dataset
                    for sample_i in range(input.shape[0]):
                        if np.random.random() < masking_prob:
                            continue

                        possible_modifications = [
                            torch.zeros_like(input[sample_i]),
                            torch.rand(input.shape[1:]),
                            input[sample_i] + torch.rand(input[sample_i].shape, device=input.device)
                        ]

                        num_boxes = np.random.randint(1, max_num_boxes + 1)

                        for _ in range(num_boxes):
                            width = np.random.randint(min_box_size, max_box_size)
                            height = np.random.randint(min_box_size, max_box_size)
                            left = np.random.randint(0, img_size - width)
                            top = np.random.randint(0, img_size - height)

                            input[sample_i, top:top + height, left:left + width] = \
                                possible_modifications[np.random.randint(3)][top:top + height, left:left + width]

            output, min_distances, all_similarities = model(input, return_all_similarities=True)

            sim_diff_loss = 0.0

            if class_specific:
                # input.shape, output.shape,
                # min_distances.shape, all_similarities.shape,
                # model.module.prototype_class_identity.shape
                #
                # torch.Size([40, 3, 224, 224]) torch.Size([40, 200])
                # torch.Size([40, 2000]) torch.Size([40, 2000, 7, 7])
                # torch.Size([2000, 200])

                if masking_type == 'random':
                    random_mask = (torch.cuda.FloatTensor(all_similarities.shape[0], 1, all_similarities.shape[-1],
                                                          all_similarities.shape[
                                                              -1]).uniform_() > masking_random_prob).float()
                    random_mask_img = torch.nn.functional.interpolate(random_mask,
                                                                      size=(input.shape[-1], input.shape[-1])).long()
                    new_input = input * random_mask_img

                    output2, min_distances2, all_similarities2 = model(new_input, return_all_similarities=True)

                    sim_diff = (all_similarities - all_similarities2) ** 2
                    sim_diff_loss = torch.sum(sim_diff * random_mask) / torch.sum(random_mask)

                elif masking_type == 'high_act' or masking_type == 'high_act_aug':
                    with torch.no_grad():
                        proto_sim = []
                        proto_nums = []
                        for sample_i, sample_label in enumerate(label):
                            label_protos = model.module.prototype_class_identity[:, sample_label].nonzero()[:, 0]
                            proto_num = np.random.choice(label_protos)
                            proto_nums.append(proto_num)
                            proto_sim.append(all_similarities[sample_i, proto_num])
                        proto_sim = torch.stack(proto_sim, dim=0).unsqueeze(1)

                        if quantized_mask:
                            all_sim_scaled = torch.nn.functional.interpolate(proto_sim,
                                                                             size=(input.shape[-1], input.shape[-1]),
                                                                             mode='bilinear')
                            q = np.random.uniform(0.5, 0.98)
                            quantile_mask = torch.quantile(all_sim_scaled.flatten(start_dim=-2), q=q, dim=-1)
                            quantile_mask = quantile_mask.unsqueeze(-1).unsqueeze(-1)

                            high_act_mask_img = (all_sim_scaled > quantile_mask).float()
                            high_act_mask_act = torch.nn.functional.interpolate(high_act_mask_img,
                                                                                size=(all_similarities.shape[-1],
                                                                                      all_similarities.shape[-1]),
                                                                                mode='bilinear')
                        else:
                            proto_sim_min = proto_sim.flatten(start_dim=1).min(-1)[0] \
                                .unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                            proto_sim_norm = proto_sim - proto_sim_min
                            proto_sim_max = proto_sim_norm.flatten(start_dim=1).max(-1)[0] \
                                .unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                            proto_sim_norm /= proto_sim_max
                            high_act_mask_act = proto_sim_norm
                            high_act_mask_img = torch.nn.functional.interpolate(high_act_mask_act,
                                                                                size=(input.shape[-1], input.shape[-1]),
                                                                                mode='bilinear')

                        new_input = input * high_act_mask_img

                    output2, min_distances2, all_similarities2 = model(new_input.detach(), return_all_similarities=True)
                    proto_sim2 = []
                    for sample_i, sample_label in enumerate(label):
                        proto_sim2.append(all_similarities2[sample_i, proto_nums[sample_i]])
                    proto_sim2 = torch.stack(proto_sim2, dim=0).unsqueeze(1)

                    if sim_diff_function == 'l2':
                        sim_diff = (proto_sim - proto_sim2) ** 2
                    elif sim_diff_function == 'l1':
                        sim_diff = torch.abs(proto_sim - proto_sim2)
                    else:
                        raise ValueError(f'Unknown sim_diff_function: ', sim_diff_function)

                    if quantized_mask:
                        sim_diff_loss = torch.sum(sim_diff * high_act_mask_act) / torch.sum(high_act_mask_act)
                    else:
                        sim_diff_loss = torch.mean(sim_diff)

                max_dist = (model.module.prototype_shape[1]
                            * model.module.prototype_shape[2]
                            * model.module.prototype_shape[3])

                # prototypes_of_correct_class is a tensor of shape batch_size * num_prototypes
                # calculate cluster cost
                prototypes_of_correct_class = torch.t(model.module.prototype_class_identity[:, label]).cuda()
                inverted_distances, _ = torch.max((max_dist - min_distances) * prototypes_of_correct_class, dim=1)
                cluster_cost = torch.mean(max_dist - inverted_distances)

                # calculate separation cost
                prototypes_of_wrong_class = 1 - prototypes_of_correct_class
                inverted_distances_to_nontarget_prototypes, _ = \
                    torch.max((max_dist - min_distances) * prototypes_of_wrong_class, dim=1)
                separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)

                # calculate avg cluster cost
                avg_separation_cost = \
                    torch.sum(min_distances * prototypes_of_wrong_class, dim=1) / torch.sum(prototypes_of_wrong_class,
                                                                                            dim=1)
                avg_separation_cost = torch.mean(avg_separation_cost)

                if use_l1_mask:
                    l1_mask = 1 - torch.t(model.module.prototype_class_identity).cuda()
                    l1 = (model.module.last_layer.weight * l1_mask).norm(p=1)
                else:
                    l1 = model.module.last_layer.weight.norm(p=1)

            else:
                min_distance, _ = torch.min(min_distances, dim=1)
                cluster_cost = torch.mean(min_distance)
                l1 = model.module.last_layer.weight.norm(p=1)

            # compute loss
            if mixup:
                cross_entropy = lam * \
                                torch.nn.functional.cross_entropy(output, targets_a) + (1 - lam) * \
                                torch.nn.functional.cross_entropy(output, targets_b)
            else:
                cross_entropy = torch.nn.functional.cross_entropy(output, target)

            # evaluation statistics
            _, predicted = torch.max(output.data, 1)
            n_examples += target.size(0)
            n_correct += (predicted == target).sum().item()

            n_batches += 1
            total_cross_entropy += cross_entropy.item()
            total_cluster_cost += cluster_cost.item()
            total_separation_cost += separation_cost.item()
            total_avg_separation_cost += avg_separation_cost.item()
            total_sim_diff_loss += sim_diff_loss.item() if torch.is_tensor(sim_diff_loss) else 0

        # compute gradient and do SGD step
        if is_train:
            if class_specific:
                if coefs is not None:
                    loss = (coefs['crs_ent'] * cross_entropy
                            + coefs['clst'] * cluster_cost
                            + coefs['sep'] * separation_cost
                            + sim_diff_weight * sim_diff_loss
                            + coefs['l1'] * l1)
                else:
                    loss = cross_entropy + 0.8 * cluster_cost - 0.08 * separation_cost + 1e-4 * l1 + 0.1 * sim_diff_loss
            else:
                if coefs is not None:
                    loss = (coefs['crs_ent'] * cross_entropy
                            + coefs['clst'] * cluster_cost
                            + coefs['l1'] * l1)
                else:
                    loss = cross_entropy + 0.8 * cluster_cost + 1e-4 * l1
            if neptune_run is not None:
                neptune_run['train/batch/loss'].append(loss.item())

            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        del input
        del target
        del output
        del predicted
        del min_distances

    # log('\ttime: \t{0}'.format(end - start))
    # if is_train:
    # log('t\loss: \t{0}'.format(total_loss / n_batches))
    # log('\tcross ent: \t{0}'.format(total_cross_entropy / n_batches))
    # log('\tcluster: \t{0}'.format(total_cluster_cost / n_batches))
    # if class_specific:
    # log('\tseparation:\t{0}'.format(total_separation_cost / n_batches))
    # log('\tavg separation:\t{0}'.format(total_avg_separation_cost / n_batches))
    # log('\taccu: \t\t{0}%'.format(n_correct / n_examples * 100))
    # l1 = model.module.last_layer.weight.norm(p=1).item()
    # log('\tl1: \t\t{0}'.format(l1))
    p = model.module.prototype_vectors.view(model.module.num_prototypes, -1).cpu()
    with torch.no_grad():
        p_avg_pair_dist = torch.mean(list_of_distances(p, p)).item()
    # log('\tp dist pair: \t{0}'.format(p_avg_pair_dist))
    # if isinstance(sim_diff_loss, torch.Tensor):
    # sim_diff_loss = sim_diff_loss.item()
    # log('\tsim diff:  \t{0}'.format(sim_diff_loss))

    converged = total_cluster_cost < total_separation_cost

    metrics = {
        'loss_cross_entropy': total_cross_entropy / n_batches,
        'loss_cluster': total_cluster_cost / n_batches,
        'loss_separation': total_separation_cost,
        'avg_separation': total_avg_separation_cost,
        'l1': l1,
        'p_avg_pair_dist': p_avg_pair_dist,
        'sim_diff_loss': total_sim_diff_loss / n_batches
    }

    if is_train:
        metrics['loss'] = total_loss / n_batches

    return n_correct / n_examples, converged, metrics


def train(model, dataloader, optimizer, class_specific=False, coefs=None, log=print, masking_type='none',
          neptune_run=None, quantized_mask=False, sim_diff_weight=0.0, sim_diff_function='l1',
          mixup: bool = True):
    assert (optimizer is not None)
    model.train()
    return _train_or_test(model=model, dataloader=dataloader, optimizer=optimizer,
                          class_specific=class_specific, coefs=coefs, log=log, masking_type=masking_type,
                          neptune_run=neptune_run, quantized_mask=quantized_mask,
                          sim_diff_function=sim_diff_function, sim_diff_weight=sim_diff_weight, mixup=mixup)


def test(model, dataloader, class_specific=False, log=print, masking_type='none', neptune_run=None,
         quantized_mask=False, sim_diff_weight=0.0, sim_diff_function='l1'):
    model.eval()
    return _train_or_test(model=model, dataloader=dataloader, optimizer=None,
                          class_specific=class_specific, log=log, masking_type=masking_type, neptune_run=neptune_run,
                          quantized_mask=quantized_mask, sim_diff_weight=sim_diff_weight,
                          sim_diff_function=sim_diff_function)


def last_only(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = False
    model.module.prototype_vectors.requires_grad = False
    for p in model.module.last_layer.parameters():
        p.requires_grad = True

    log('\tlast layer')


def warm_only(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = True

    log('\twarm')


def joint(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = True
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = True

    log('\tjoint')
