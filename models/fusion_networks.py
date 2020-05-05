import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

####################################################################
####################################################################

# takes in concatenated list of logits (bx36xn)  where n is number of modalities used
# outputs list of logits (Bx36x1)
# first option: each network multiplied by the same weight for each class
# second option: multiplied by per-class weight

class fusion_class_layer(nn.Module):
    def __init__(self, num_modalities, num_classes):
        super(fusion_class_layer, self).__init__()
        self.num_classes = num_classes
        self.layers = {}
        for i in range(num_classes):
            class_num = str(i)
            self.layers[class_num] = nn.Linear(num_modalities, 1, bias=False)

    def forward(self, logit_concat):
        output_list = []
        j = 0
        for layer_name, layer in self.layers.items():
            output_list.append(layer(logit_concat[:,j,:]))
            j += 1
        return output_list

####################################################################
####################################################################
class fusion_uniform_layer(nn.Module):
    def __init__(self, num_modalities):
        super(fusion_uniform_layer, self).__init__()
        self.layer1 = nn.Linear(num_modalities, 1, bias=False)

    def forward(self, logit_concat):
        output = self.layer1(logit_concat)
        return output
