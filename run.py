import os
import cv2
import glob
import torch
import argparse
import numpy as np
from tqdm import trange
from SGR import DepthNet as SGRnet
from MiDaS.midas_net import MidasNet
from utils.model import Gradient_FusionModel
from utils.func import scale_image, save_orig, visual_crfs
from LeRes.multi_depth_model_woauxi import strip_prefix_if_present, RelDepthModel
from dpt.models import DPTDepthModel
from dpt.transforms import Resize, NormalizeImage, PrepareForNet
from newcrfs.networks.NewCRFDepth import NewCRFDepth
from torchvision.transforms import Compose
import torchvision.transforms as transforms
from torch.autograd import Variable


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def run(args):
    size = 224
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.pred_model == 'LeRes50':
        Depth_model = RelDepthModel(backbone='resnet50')
        depth_dict = './LeRes/res50.pth'
        depth_dict = torch.load(depth_dict)
        Depth_model.load_state_dict(strip_prefix_if_present(depth_dict['depth_model'], "module."), strict=True)
        model_flag = 1

    elif args.pred_model == 'LeRes101':
        Depth_model = RelDepthModel(backbone='resnext101')
        depth_dict='./LeRes/res101.pth'
        depth_dict = torch.load(depth_dict)
        Depth_model.load_state_dict(strip_prefix_if_present(depth_dict['depth_model'], "module."), strict=True)
        model_flag = 2

    elif args.pred_model == 'SGR':
        Depth_model = SGRnet.DepthNet()
        if device == torch.device("cuda"):
            Depth_model = torch.nn.DataParallel(Depth_model, device_ids=[0]).cuda()
        else:
            print('sgr model can not run correctly without gpu')
            exit()
        depth_dict = torch.load('./SGR/model.pth.tar')
        Depth_model.load_state_dict(depth_dict['state_dict'])
        model_flag = 3
    
    elif args.pred_model == 'MiDaS':
        Depth_model = MidasNet('./MiDaS/model.pt', non_negative=True)
        model_flag = 4
        size = 192

    elif args.pred_model == 'DPT':
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        Depth_model = DPTDepthModel(
        path="./dpt/weights/dpt_hybrid-midas-501f0c75.pt",
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
        model_flag = 5
    
    elif args.pred_model == 'NeWCRFs':
        checkpoint_path = './newcrfs/model_nyu.ckpt'
        Depth_model = NewCRFDepth(version='large07', inv_depth=True, max_depth=1000)
        Depth_model = torch.nn.DataParallel(Depth_model)
        checkpoint = torch.load(checkpoint_path)
        Depth_model.load_state_dict(checkpoint['model'])
        model_flag = 6
    
    else:
        print('no such model')
        exit()

    Fusion_model = Gradient_FusionModel(dict_path=args.model_weights)
    
    Depth_model.to(device)
    Depth_model.eval()
    Fusion_model.to(device)
    Fusion_model.eval()

    img_names = glob.glob(os.path.join(args.input_path, "*"))
    img_names.sort()
    for index in trange(len(img_names)):
        img_loc = img_names[index]
        img = cv2.imread(img_loc)
        
        if model_flag == 5:
            img = img.astype('float32')/255.0
            low_img = transform_low({"image": img})["image"]
            high_img = transform_high({"image": img})["image"]
        elif model_flag == 6:
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
            if model_flag == 1 or model_flag == 2:
                low_dep = Depth_model.inference(low_img)
                high_dep = Depth_model.inference(high_img)

            elif model_flag == 3:
                low_dep = Depth_model.forward(low_img)
                high_dep = Depth_model.forward(high_img)
                low_dep = low_dep.max() - low_dep
                high_dep = high_dep.max() - high_dep

            elif model_flag == 4:
                low_dep = Depth_model.forward(low_img).unsqueeze(0)
                high_dep = Depth_model.forward(high_img).unsqueeze(0)
                low_dep = low_dep.max() - low_dep
                high_dep = high_dep.max() - high_dep

            elif model_flag == 5:
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

            elif model_flag == 6:
                low_dep = Depth_model(low_img)
                high_dep = Depth_model(high_img)
                low_dep, high_dep = visual_crfs(low_dep, high_dep)

            low_dep, high_dep, pred = Fusion_model.inference(low_dep, high_dep)
        save_orig(img, f'{args.output_path}/{args.pred_model}_{index}.jpg', low_dep, pred, model_flag)



if __name__ == '__main__':  
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-i', '--input_path', 
        default='./input',
        help='folder with input images'
    )

    parser.add_argument('-o', '--output_path', 
        default='./output',
        help='folder for output images'
    )

    parser.add_argument('-m', '--model_weights', 
        default='./models/model_dict.pt',
        help='path to the trained weights of model'
    )
    
    parser.add_argument('-p', '--pred_model', 
        default='LeRes50',
        help='model type: LeRes50, LeRes101, SGR ,MiDaS, DPT or NeWCRFs'
    )
    
    args = parser.parse_args()
    run(args)
