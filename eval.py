import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import trange
from utils.func import *
from SGR import DepthNet as SGRnet
from MiDaS.midas_net import MidasNet
from utils.model import Gradient_FusionModel
from torch.optim import lr_scheduler, AdamW
import torchvision.transforms as transforms
from LeRes.multi_depth_model_woauxi import strip_prefix_if_present, RelDepthModel
from utils.middleburry2021 import middleburry
from utils.multiscopic import multiscopic
from utils.hypersim import hypersim
from torchvision.transforms import Compose
from dpt.models import DPTDepthModel
from dpt.midas_net import MidasNet_large
from dpt.transforms import Resize, NormalizeImage, PrepareForNet
from newcrfs.networks.NewCRFDepth import NewCRFDepth
from torch.autograd import Variable


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    size = 224
    if args.pred_model == 'LeRes50':
        Depth_model = RelDepthModel(backbone='resnet50')
        depth_dict = './LeRes/res50.pth'
        depth_dict = torch.load(depth_dict)
        Depth_model.load_state_dict(strip_prefix_if_present(depth_dict['depth_model'], "module."), strict=True)
        model_flag = 1

    elif args.pred_model == 'SGR':
        Depth_model = SGRnet.DepthNet()
        if device == torch.device("cuda"):
            Depth_model = torch.nn.DataParallel(Depth_model, device_ids=[0]).cuda()
        else:
            print('sgr model can not run correctly without cpu')
            exit()
        depth_dict = torch.load('./SGR/model.pth.tar')
        Depth_model.load_state_dict(depth_dict['state_dict'])
        model_flag = 2
    
    elif args.pred_model == 'MiDaS':
        Depth_model = MidasNet('./MiDaS/model.pt', non_negative=True)
        model_flag = 3
        size = 192
    
    elif args.pred_model == 'dpt':
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        Depth_model = DPTDepthModel(
        path="dpt/weights/dpt_hybrid-midas-501f0c75.pt",
        backbone="vitb_rn50_384",
        non_negative=True,
        enable_attention_hooks=False,
        )
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        transform_low = Compose(
            [Resize(
                384,
                384,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),])
        
        transform_high = Compose(
            [Resize(
                384*3,
                384*3,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),])
        if device == torch.device("cuda"):
            Depth_model = Depth_model.to(memory_format=torch.channels_last)
            Depth_model = Depth_model.half()
        model_flag = 4
    
    elif args.pred_model == 'newcrfs':
        max_depth = 1000
        checkpoint_path = './newcrfs/model_nyu.ckpt'
        Depth_model = NewCRFDepth(version='large07', inv_depth=True, max_depth=max_depth)
        Depth_model = torch.nn.DataParallel(Depth_model)
        checkpoint = torch.load(checkpoint_path)
        Depth_model.load_state_dict(checkpoint['model'])
        model_flag = 5

    else:
        print('no such model')
        exit()
        
    Fuse_model = Gradient_FusionModel(dict_path=args.model_weights)

    Fuse_model.to(device)
    Depth_model.to(device)
    Fuse_model = Fuse_model.eval()
    Depth_model = Depth_model.eval()

    if args.eval_dataset == 'middleburry2021':
        dataset = middleburry()
    elif args.eval_dataset == 'multiscopic':
        dataset = multiscopic()
    elif args.eval_dataset == 'hypersim':
        dataset = hypersim()
    else:
        print('no such dataset')
        exit()
    
    # while dataset.index != dataset.num-1:
    for i in trange(dataset.num):
        img, depth, val_mask = dataset.getitem()
        if model_flag == 4:
            img = img.astype('float32')/255.0
            low_img = transform_low({"image": img})["image"]
            high_img = transform_high({"image": img})["image"]
        elif model_flag == 5:
            img = img.astype('float32')/255.0
            low_img = cv2.resize(img, (640, 480))
            high_img = cv2.resize(img, (640*3, 480*3))
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            low_img = np.expand_dims(low_img, axis=0)
            low_img = np.transpose(low_img, (0, 3, 1, 2))
            low_img = Variable(normalize(torch.from_numpy(low_img)).float()).cuda()
            high_img = np.expand_dims(high_img, axis=0)
            high_img = np.transpose(high_img, (0, 3, 1, 2))
            high_img = Variable(normalize(torch.from_numpy(high_img)).float()).cuda()
        else:
            low_img, high_img = scale_image(img, size, device)
        
        with torch.no_grad():
            if model_flag == 1:
                low_dep = Depth_model.inference(low_img)
                high_dep = Depth_model.inference(high_img)

            elif model_flag == 2:
                low_dep = Depth_model.forward(low_img)
                high_dep = Depth_model.forward(high_img)
                low_dep = low_dep.max() - low_dep
                high_dep = high_dep.max() - high_dep

            elif model_flag == 3:
                low_dep = Depth_model.forward(low_img).unsqueeze(0)
                high_dep = Depth_model.forward(high_img).unsqueeze(0)
                low_dep = low_dep.max() - low_dep
                high_dep = high_dep.max() - high_dep

            elif model_flag == 4:
                sample = torch.from_numpy(low_img).to(device).unsqueeze(0)
                sample = sample.to(memory_format=torch.channels_last)
                sample = sample.half()
                low_dep = Depth_model.forward(sample)
                low_dep = (torch.nn.functional.interpolate(
                        low_dep.unsqueeze(1),
                        size=img.shape[:2],
                        mode="bicubic",
                        align_corners=False,)).float()
                sample = torch.from_numpy(high_img).to(device).unsqueeze(0)
                sample = sample.to(memory_format=torch.channels_last)
                sample = sample.half()
                high_dep = Depth_model.forward(sample)
                high_dep = (torch.nn.functional.interpolate(
                        high_dep.unsqueeze(1),
                        size=img.shape[:2],
                        mode="bicubic",
                        align_corners=False,)).float()
                low_dep = low_dep.max() - low_dep
                high_dep = high_dep.max() - high_dep

            elif model_flag == 5:
                low_dep = Depth_model(low_img)
                high_dep = Depth_model(high_img)

            low_dep, high_dep, fusion = Fuse_model.inference(low_dep, high_dep)
        dataset.compute_error(fusion, depth, val_mask)

    print('Results:')
    print('sq_rel = ',  np.nanmean(dataset.sq_rel))
    print('rms    = ',  np.nanmean(dataset.rms))
    print('log10  = ',  np.nanmean(dataset.log10))
    print('thr1   = ',  np.nanmean(dataset.thr1))
    print('thr2   = ',  np.nanmean(dataset.thr2))



if __name__ == '__main__':  
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model_weights', 
        default='./models/model_dict.pt',
        help='path to the trained weights of model'
    )
    
    parser.add_argument('-p', '--pred_model', 
        default='LeRes50',
        help='model type: LeRes50, SGR ,MiDaS, dpt or newcrfs'
    )
    
    parser.add_argument('-d', '--eval_dataset', 
        default='middleburry2021',
        help='dataset: multiscopic, middleburry2021 or hypersim'
    )
    
    args = parser.parse_args()
    run(args)