import os
import cv2
import glob
import h5py
from pylab import *
import numpy as np
from utils.func import *


def convetDep(file):
    intWidth, intHeight, fltFocal = 1024, 768, 886.81
    npyDistance = file
    npyImageplaneX = np.linspace((-0.5 * intWidth) + 0.5, (0.5 * intWidth) - 0.5, intWidth).reshape(1, intWidth).repeat(intHeight, 0).astype(np.float32)[:, :, None]
    npyImageplaneY = np.linspace((-0.5 * intHeight) + 0.5, (0.5 * intHeight) - 0.5, intHeight).reshape(intHeight, 1).repeat(intWidth, 1).astype(np.float32)[:, :, None]
    npyImageplaneZ = np.full([intHeight, intWidth, 1], fltFocal, np.float32)
    npyImageplane = np.concatenate([npyImageplaneX, npyImageplaneY, npyImageplaneZ], 2)

    npyDepth = npyDistance / np.linalg.norm(npyImageplane, 2, 2) * fltFocal
    return npyDepth


def compute_global_errors(gt, pred):
    gt=gt[gt!=0]
    pred=pred[pred!=0]
    
    mask2 = gt > 1e-8
    mask3 = pred > 1e-8
    mask2 = mask2 & mask3
    
    gt = gt[mask2]
    pred = pred[mask2]
    
    #compute global relative errors
    thresh = np.maximum((gt / pred), (pred / gt))
    thr1 = (thresh < 1.25   ).mean()
    thr2 = (thresh < 1.25 ** 2).mean()
    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())
    log10 = np.mean(np.abs(np.log10(gt) - np.log10(pred)))
    sq_rel = np.mean(((gt - pred)**2) / gt)

    return sq_rel, rmse, log10, thr1, thr2


class hypersim():
    def __init__(self, path= './datasets/hypersim'):
        self.input_path = path
        self.num = 286
        self.rms     = np.zeros(self.num, np.float32)
        self.log10   = np.zeros(self.num, np.float32)
        self.sq_rel  = np.zeros(self.num, np.float32)
        self.thr1    = np.zeros(self.num, np.float32)
        self.thr2    = np.zeros(self.num, np.float32)
        self.d3r_rel    = np.zeros(self.num, np.float32)
        self.ord_rel    = np.zeros(self.num, np.float32)
        self.img_names = glob.glob(os.path.join(self.input_path, "*"))
        self.index = -1
        self.name_list = []


    def getitem(self):
        if not len(self.name_list):
            self.sub_set = self.img_names.pop()
            img_loc = self.sub_set + '/images/scene_cam_00_final_hdf5/'
            name_list = glob.glob(os.path.join(img_loc, "*"))
            self.name_list = [name for name in name_list if 'color' in name]
        file_loc = self.name_list.pop()

        dep_loc = file_loc.replace('scene_cam_00_final_hdf5', 'scene_cam_00_geometry_hdf5')
        dep_loc = dep_loc.replace('color', 'depth_meters')
        entity_loc = dep_loc.replace('depth_meters', 'render_entity_id')
        
        with h5py.File(file_loc, "r") as f: rgb_color = f["dataset"][:].astype('float32')
        with h5py.File(entity_loc, "r") as f: render_entity_id = f["dataset"][:].astype('int32')
        # assert all(render_entity_id != 0)
        
        gamma                             = .5/2.2   # standard gamma correction exponent
        inv_gamma                         = 1.0/gamma
        percentile                        = 90        # we want this percentile brightness value in the unmodified image...
        brightness_nth_percentile_desired = 0.8       # ...to be this bright after scaling
        valid_mask = render_entity_id != -1
        if count_nonzero(valid_mask) == 0:
            scale = 1.0 # if there are no valid pixels, then set scale to 1.0
        else:
            brightness       = 0.3*rgb_color[:,:,0] + 0.59*rgb_color[:,:,1] + 0.11*rgb_color[:,:,2] # "CCIR601 YIQ" method for computing brightness
            brightness_valid = brightness[valid_mask]
            eps                               = 0.0001 # if the kth percentile brightness value in the unmodified image is less than this, set the scale to 0.0 to avoid divide-by-zero
            brightness_nth_percentile_current = np.percentile(brightness_valid, percentile)
            if brightness_nth_percentile_current < eps:
                scale = 0.0
            else:
                scale = np.power(brightness_nth_percentile_desired, inv_gamma) / brightness_nth_percentile_current
        rgb_color_tm = np.power(np.maximum(scale*rgb_color,0), gamma)
        
        img = rgb_color_tm
        img = img/img.max() * 255
        img_bgr = img[:, :, ::-1]
        img_bgr = img_bgr.astype('uint8')
        
        with h5py.File(dep_loc, "r") as f: 
            dep = f["dataset"][:]
            
        nan = float('nan')
        dep[np.isnan(dep)] = 0
        dep = dep.astype('float32')
        val_mask = np.ones_like(dep)
        val_mask[dep==0]=0
        self.index += 1
        return img_bgr, dep, val_mask

    def compute_error(self, target, depth, val_mask):
        target = target.cpu().numpy().squeeze()
        h, w = depth.shape
        val_mask = cv2.resize(val_mask, (w, h))
        target = cv2.resize(target, (w, h))
        target = shift_scale(target.astype('float64'), depth.astype('float64'), val_mask)
        
        pred = target.copy()
        pred_org = pred.copy()
        pred_invalid = pred.copy()
        pred_invalid[pred_invalid!=0]=1
        mask_missing = depth.copy() # Mask for further missing depth values in depth map
        mask_missing[mask_missing!=0]=1
        mask_valid = mask_missing*pred_invalid # Combine masks
        depth_valid = depth*mask_valid 
        gt = depth_valid
        gt_vec = gt.flatten()
        pred = pred*mask_valid 
        pred_vec = pred.flatten()
        self.sq_rel[self.index], self.rms[self.index], self.log10[self.index], self.thr1[self.index], self.thr2[self.index] = compute_global_errors(gt_vec,pred_vec)
