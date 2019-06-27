import numpy as np
import os, sys
from PIL import Image
import json
import torch
from torchvision import transforms
from torch.autograd import Variable
from .net import get_model

#import imgaug as ia
#import imgaug.augmenters as iaa

BASE_PATH = os.path.abspath(os.path.join('.', '.'))
MODEL_PATH = BASE_PATH + "/agents/CAL_agent/perception/model_data/models/"

# classes of the categorical affordances
CAT_DICT = {
    'red_light': [False, True],
    'hazard_stop': [False, True],
    'speed_sign': [-1, 30, 60, 90],
}

# normalizing constants of the continuous affordances
REG_DICT = {
    'center_distance': 1.6511945645500001,
    'veh_distance': 50.0,
    'relative_angle': 0.759452569632
}


def get_augmentations():
    # applies the given augmenter in 50% of all cases,
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    # Define our sequence of augmentation steps that will be applied to every image
    seq = iaa.Sequential([
            # execute 0 to 5 of the following (less important) augmenters per image
            iaa.SomeOf((0, 5),
                [
                    iaa.OneOf([
                        iaa.GaussianBlur((0, 3.0)),
                        iaa.AverageBlur(k=(2, 7)), 
                        iaa.MedianBlur(k=(3, 11)),
                    ]),
                    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
                    iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), 
                    # search either for all edges or for directed edges,
                    # blend the result with the original image using a blobby mask
                    iaa.SimplexNoiseAlpha(iaa.OneOf([
                        iaa.EdgeDetect(alpha=(0.5, 1.0)),
                        iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                    ])),
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
                    iaa.OneOf([
                        iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                        iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                    ]),
                    iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                    iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
                    # either change the brightness of the whole image (sometimes
                    # per channel) or change the brightness of subareas
                    iaa.OneOf([
                        iaa.Multiply((0.5, 1.5), per_channel=0.5),
                        iaa.FrequencyNoiseAlpha(
                            exponent=(-4, 0),
                            first=iaa.Multiply((0.5, 1.5), per_channel=True),
                            second=iaa.ContrastNormalization((0.5, 2.0))
                        )
                    ]),
                    iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                    sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                ],
                random_order=True
            )
        ],
        random_order=True
    )
    return seq









### helper functions

def load_json(path):
    with open(path + '.json', 'r') as json_file:
        f = json.load(json_file)
    return f

def to_np(t):
    return np.array(t.data.cpu())

def softmax(x):
    #return np.exp(x)/sum(np.exp(x))
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

### data transforms

class Rescale(object):
    def __init__(self, scalar):
        self.scalar = scalar

    def __call__(self, im):
        w, h = [int(s*self.scalar) for s in im.size]
        return transforms.Resize((h, w))(im)

class Crop(object):
    def __init__(self, box):
        assert len(box) == 4
        self.box = box

    def __call__(self, im):
        return im.crop(self.box)


def get_transform():
    return transforms.Compose([
        Crop((0, 120, 800, 480)),
        Rescale(0.4),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])

### network

class CAL_network(object):
    def __init__(self, name='gru'):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._transform = get_transform()

        # get the model
        params = load_json(MODEL_PATH + "params")
        self.model, _ = get_model(params)
        self.model.load_state_dict(torch.load(MODEL_PATH + "test.pth"))
        self.model.eval().to(self.device);

    def predict(self, sequence, direction):
        print('cal_network sequence',len(sequence))
        #print(torch.cat(sequence).size())
        inputs = {
            'sequence': torch.cat(sequence).unsqueeze(0).to(self.device),
            'direction': torch.Tensor([direction]).to(dtype=torch.int),
        }

        preds = self.model(inputs)
        preds = {k: to_np(v) for k,v in preds.items()}
        #print("before cat",preds)
        out = {}
        out.update({k: self.cat_process(k, preds[k]) for k in CAT_DICT})
        out.update({k: self.reg_process(k, preds[k]) for k in REG_DICT})
        return out

    def preprocess(self, arr):
        im = self._transform(Image.fromarray(arr))
        return im.unsqueeze(0)

    @staticmethod
    def cat_process(cl, arr):
        arr=softmax(arr)
        max_idx = np.argmax(arr)
        pred_class = CAT_DICT[cl][max_idx]
        pred_prob = np.max(arr)
        print("probability:",pred_prob)
        return (pred_class, pred_prob)

    @staticmethod
    def reg_process(cl, arr):
        arr = np.clip(arr, -1, 1)
        return arr*REG_DICT[cl]
