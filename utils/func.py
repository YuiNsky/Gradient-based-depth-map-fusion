import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from utils.guided_f import guided_filter


def shift_scale(pred, gt, mask_=None):
    gt_flat = gt.flatten()
    pred_flat = pred.flatten()
    mask_valid = np.ones_like(gt_flat)
    mask_valid[gt_flat == 0] = 0
    if mask_ is not None:
        mask_ = mask_.flatten()
        mask_valid[mask_ == 0] = 0

    gt_valid = np.array([gt_flat[i] for i in range(len(mask_valid)) if mask_valid[i]])
    pred_valid = np.array([pred_flat[i] for i in range(len(mask_valid)) if mask_valid[i]])

    para_s = np.polyfit(pred_valid, gt_valid, deg=1)
    pred = np.polyval(para_s, pred)
    return pred


def generate_gf(low_dep, high_dep):
    r = int(high_dep.shape[0] / 12) - 1
    enhanced = guided_filter(high_dep, low_dep, r, 1e-12)
    return enhanced


def visual_crfs(low_dep, high_dep):
    low_dep[:, :, :, :5] = low_dep.min()
    low_dep[:, :, :, -5:] = low_dep.min()
    low_dep[:, :, :5, :] = low_dep.min()
    low_dep[:, :, -5:, :] = low_dep.min()
    high_dep[:, :, :, :15] = high_dep.min()
    high_dep[:, :, :, -15:] = high_dep.min()
    high_dep[:, :, :15, :] = high_dep.min()
    high_dep[:, :, -15:, :] = high_dep.min()
    return low_dep, high_dep


def img2Tensor(img, scale, model_input_size=224):
    transformer = transforms.Compose([transforms.ToTensor(), transforms.Resize(size=(model_input_size*scale, model_input_size*scale)),\
        transforms.Normalize((0.485, 0.456, 0.406) , (0.229, 0.224, 0.225))])
    tens = transformer(img.copy())
    return tens[None, :, :, :]


def scale_image(img, size, device):
    scale_list = [2, 6]
    tensor_list = []
    
    for scale in scale_list:
        tensor = img2Tensor(img, scale, size)
        if device == torch.device("cuda"):
            tensor_list.append(tensor.cuda())
        else:
            tensor_list.append(tensor)
    return tensor_list


def save_orig(input_rgb, img_loc, low_dep, pred, model_flag):
    if model_flag >= 5:
        input_rgb = (input_rgb * 255).astype('uint8')
    input_rgb  = cv2.resize(input_rgb, None, fx=0.5, fy=0.5)
    h, w, _ = input_rgb.shape
    low_dep = cv2.resize(low_dep.cpu().detach().numpy().squeeze(), (w, h))
    pred = cv2.resize(pred.cpu().detach().numpy().squeeze(), (w, h))
    low_dep = 255 - (low_dep - low_dep.min()) / (low_dep.max() - low_dep.min()) * 255
    pred = 255 - (pred - pred.min()) / (pred.max() - pred.min()) * 255
    result = np.hstack((low_dep, low_dep, pred))
    plt.imsave(img_loc, result, cmap='inferno')
    result = cv2.imread(img_loc)
    result[:h, :w, :] = input_rgb
    cv2.imwrite(img_loc, result)
