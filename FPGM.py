import numpy as np
import torch
from scipy.spatial import distance
from torch import nn


def get_conv_layer_weights(model, layer_name):
    try:
        layer = dict(model.named_modules())[layer_name]
        if isinstance(layer, (nn.Conv1d, nn.Linear)):
            return layer
        else:
            raise ValueError("The layer named '{layer_name}' is not support.")
    except KeyError:
        raise ValueError("No layer named '{layer_name}' found in the model.")

def prune(model, layer_name, norm_rate, prune_rate):
    layer = get_conv_layer_weights(model, layer_name)
    weights = get_conv_layer_weights(model, layer_name).weight
    weights_vec = weights.view(weights.size()[0], -1)
    norm_num = int(weights_vec.size()[0] * norm_rate)
    pruned_num = int(weights_vec.size()[1] * prune_rate)

    norm = torch.norm(weights_vec, p=1, dim=1)
    norm_small_index = norm.argsort()[:norm_num]
    weight_vec_after_norm = torch.index_select(weights_vec, 0, norm_small_index).cpu().detach().numpy()
    medians = np.median(weight_vec_after_norm, axis=1)

    distances = np.abs(weight_vec_after_norm - medians[:, np.newaxis])
    prune_indices = np.argsort(distances)[:, :pruned_num]

    mask = torch.ones_like(weights)
    if len(weights.shape) == 3:
        with torch.no_grad():
            for i, idx in enumerate(prune_indices):
                weights[norm_small_index[i], :, idx] = 0
                mask[norm_small_index[i], :, idx] = 0
    else:
        with torch.no_grad():
            for i, idx in enumerate(prune_indices):
                weights[norm_small_index[i], idx] = 0
                mask[norm_small_index[i], idx] = 0

    def mask_hook(grad):
        return grad * mask

    layer.weight.register_hook(mask_hook)