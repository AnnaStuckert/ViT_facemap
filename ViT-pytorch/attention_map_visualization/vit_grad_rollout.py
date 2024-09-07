import sys

import cv2
import numpy
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


def grad_rollout(attentions, gradients, discard_ratio):
    result = torch.eye(attentions[0].size(-1))
    with torch.no_grad():
        for attention, grad in zip(attentions, gradients):
            weights = grad
            attention_heads_fused = (attention * weights).max(axis=1)[
                0
            ]  # changed from attention_heads_fused = (attention*weights).mean(axis=1)
            # attention_heads_fused = (attention*weights).mean(axis=1)
            attention_heads_fused[attention_heads_fused < 0] = 0

            # Drop the lowest attentions, but
            # don't drop the class token
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1) * discard_ratio), -1, False)
            # indices = indices[indices != 0]
            flat[0, indices] = 0

            I = torch.eye(attention_heads_fused.size(-1))
            a = (attention_heads_fused + 1.0 * I) / 2
            a = a / a.sum(dim=-1)
            result = torch.matmul(a, result)

    # Look at the total attention between the class token,
    # and the image patches
    mask = result[0, 0, 1:]
    # In case of 224x224 image, this brings us from 196 to 14 - but it essentially has 14 x 14 squares to cover the 224 x 224 picture, so it looks a bit like the attention map is very 'square-y' but blurred out like as if a gaussian fliter is applied.
    width = int(mask.size(-1) ** 0.5)
    mask = mask.reshape(width, width).numpy()
    mask = mask / np.max(mask)
    return mask


class VITAttentionGradRollout:
    def __init__(
        self, model, attention_layer_name="attn_drop", discard_ratio=0.9
    ):  # automatically applies a discard ratior of 0.9
        self.model = model
        self.discard_ratio = discard_ratio
        for name, module in self.model.named_modules():
            if attention_layer_name in name:
                module.register_forward_hook(
                    self.get_attention
                )  # I think this does a forward pass through the model to get KP predictions
                module.register_backward_hook(
                    self.get_attention_gradient
                )  # I thinkt this does a backwards pass to get gradients associated with KP predictions

        self.attentions = []
        self.attention_gradients = []

    def get_attention(self, module, input, output):
        self.attentions.append(output.cpu())

    def get_attention_gradient(self, module, grad_input, grad_output):
        self.attention_gradients.append(grad_input[0].cpu())

    def __call__(self, input_tensor, category_index):
        self.model.zero_grad()  # resets the gradients of all model parameters to zero
        output = self.model(input_tensor)
        category_mask = torch.zeros(
            output[0].size()
        )  # we make a vector of 0s, in the next line we then keep all the zeros, except in the position for the category index we choose the main script, so if we choose keypoint 0, the subsequent vector would look like [1,0,0,0,0.....]
        category_mask[:, category_index] = (
            1  # does this set the category index to 1? wouldn't that just visualize the y coordinate of the first KP?
        )

        loss = (output[0] * category_mask).sum()
        loss.backward()

        return grad_rollout(
            self.attentions, self.attention_gradients, self.discard_ratio
        )
