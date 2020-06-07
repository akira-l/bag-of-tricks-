
import torch.nn as nn
from . import EfficientNet


class EfficientNet_Dense(nn.Module):
    def __init__(self, spec, num_classes):
        super().__init__()
        self.backbone = EfficientNet.from_pretrained(spec, num_classes=num_classes)
        self.conv_1x1 = nn.Conv2d(self.backbone._fc.in_features, num_classes, 1, 1)

    def forward(self, inputs):
        bs = inputs.size(0)
        x = self.backbone.extract_features(inputs)
        x = self.backbone._dropout(x)
        x = self.conv_1x1(x)
        x = self.backbone._avg_pooling(x)
        x = x.view(bs, -1)
        return x

def resume_model(resume, model, dense=False):
    logging.info("resuming model from {}".format(resume))
    state_dict = torch.load(resume, map_location=torch.device('cpu'))
    model_state_dict = state_dict['state_dict']

    if not dense:
        model.load_state_dict(model_state_dict)
    else:
        target_state_dict = {}
        for key, value in model_state_dict.items():
            target_state_dict['backbone.' + key] = value
        target_state_dict['conv_1x1.weight'] = model_state_dict['_fc.weight'].unsqueeze(-1).unsqueeze(-1)
        target_state_dict['conv_1x1.bias'] = model_state_dict['_fc.bias']
        model.load_state_dict(target_state_dict)
    return state_dict['epoch'] - 1


