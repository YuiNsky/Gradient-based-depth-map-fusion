import copy
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


class FuseNet(nn.Module):
    def __init__(self):
        super(FuseNet, self).__init__()
        self.upsize = 10
        channel_list = [2**(index+3) for index in range(self.upsize)]
        self.grad = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5, stride=1, padding=2, padding_mode='replicate', bias=True)
        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()
        for index in range(len(channel_list)):
            if index == 0:
                layer = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=channel_list[0], kernel_size=3, padding=1, 
                                                stride=2, padding_mode='replicate', bias=True), nn.GroupNorm(1,channel_list[0]))
                self.encoder.add_module('layer_0', layer)
                layer = nn.Sequential(nn.LeakyReLU(), \
                                         nn.Conv2d(in_channels=channel_list[0], out_channels=channel_list[1], kernel_size=3,
                                                   padding=1, stride=2, padding_mode='replicate', bias=True), nn.GroupNorm(1,channel_list[1]))
                self.encoder.add_module('layer_1', layer)

            elif index != len(channel_list)-1:
                layer = nn.Sequential(nn.LeakyReLU(), \
                                         nn.Conv2d(in_channels=channel_list[index], out_channels=channel_list[index+1], kernel_size=3,
                                                   padding=1, stride=2, padding_mode='replicate', bias=True), nn.GroupNorm(channel_list[index+1]//8,channel_list[index+1]))
                self.encoder.add_module('layer_{}'.format(index+1), layer)

        for index in range(len(channel_list)):
            vise_ind = len(channel_list) - index - 1
            if index < len(channel_list)-2:
                layer = nn.Sequential(nn.GroupNorm(channel_list[vise_ind]//8,channel_list[vise_ind]), nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True),  \
                        nn.Conv2d(in_channels=channel_list[vise_ind], out_channels=channel_list[vise_ind-1], kernel_size=3, padding=1, stride=1, padding_mode='replicate', bias=True), nn.LeakyReLU())
                self.decoder.add_module('layer_{}'.format(index), layer)
            else:
                layer = nn.Sequential(nn.GroupNorm(1,channel_list[1]), nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True), \
                        nn.Conv2d(in_channels=channel_list[1], out_channels=channel_list[0], kernel_size=3, padding=1, stride=1, padding_mode='replicate', bias=True), nn.LeakyReLU())
                self.decoder.add_module('layer_{}'.format(index), layer)
                
                layer = nn.Sequential(nn.GroupNorm(1, channel_list[0]), nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True), \
                        nn.Conv2d(in_channels=channel_list[0], out_channels=1, kernel_size=5, padding=2, stride=1, padding_mode='replicate', bias=True))
                self.decoder.add_module('layer_{}'.format(index+1), layer)
                break

        self.mid_out_half = nn.Sequential(nn.GroupNorm(1,channel_list[1]), nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True), \
                        nn.Conv2d(in_channels=channel_list[1], out_channels=1, kernel_size=3, padding=1, stride=1, padding_mode='replicate', bias=True))
        self.mid_out_quart = nn.Sequential(nn.GroupNorm(2,channel_list[2]), nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True), \
                        nn.Conv2d(in_channels=channel_list[2], out_channels=1, kernel_size=3, padding=1, stride=1, padding_mode='replicate', bias=True))
        self.init_weight()


    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.constant_(m.bias.data, 0)
                nn.init.normal_(m.weight, std=0.01)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x, y):
        y0 = self.grad(y)
        feature_low = []
        feature_high = []
        
        for layer in self.encoder.named_children():
            if layer[0] == 'layer_0':
                feature_low.append(layer[1](x))
                feature_high.append(layer[1](y0))
            else:
                feature_low.append(layer[1](feature_low[-1]))
                feature_high.append(layer[1](feature_high[-1]))
        
        feature_low.reverse()
        feature_high.reverse()
        index = 0
        for layer in self.decoder.named_children():
            if layer[0] == 'layer_0':
                result = layer[1](feature_low[index] + feature_high[index])
            else:
                if layer[0] == f'layer_{self.upsize-2}':
                    out_half = self.mid_out_half((feature_low[index] + feature_high[index] + result) / 2)
                elif layer[0] == f'layer_{self.upsize-3}':
                    out_quart = self.mid_out_quart((feature_low[index] + feature_high[index] + result) / 2)
                result = layer[1]((feature_low[index] + feature_high[index] + result) / 2)
            index += 1
        return result, out_half, out_quart


class Gradient_FusionModel(nn.Module):
    def __init__(self, log_path=None, dict_path=None):
        super(Gradient_FusionModel, self).__init__()
        self.Fuse = FuseNet()
        self.pool = nn.AdaptiveAvgPool2d((2*2**self.Fuse.upsize, 2*2**self.Fuse.upsize))
        self.down_sample = nn.AvgPool2d(2, 2)
        if log_path is not None:
            self.log_path = log_path
            self.record_dep = np.array([])
            self.record_rank = np.array([])
            self.total_step = 0

        if dict_path is not None:
            depth_dict = torch.load(dict_path)
            self.Fuse.load_state_dict(depth_dict['net'])


    def ILNR(self, src, target):
        loss = torch.mean(torch.abs(src - target) + torch.abs(torch.tanh(src) - torch.tanh(target)))
        return loss


    def sample_pairs(self, target, alpha=0.15, beta=100, margin=0):
        x_kernel = [[-1, 0., 1], [-2., 0., 2.], [-1, 0., 1]]
        x_kernel = torch.FloatTensor(x_kernel).unsqueeze(0).unsqueeze(0)
        x_weight = nn.Parameter(data=x_kernel, requires_grad=False).cuda()
        
        y_kernel = [[-1, -2., -1], [0., 0., 0.], [1, 2., 1]]
        y_kernel = torch.FloatTensor(y_kernel).unsqueeze(0).unsqueeze(0)
        y_weight = nn.Parameter(data=y_kernel, requires_grad=False).cuda()
        
        gx_img = torch.abs(F.conv2d(target, x_weight, padding=1))
        gy_img = torch.abs(F.conv2d(target, y_weight, padding=1))
        G_img = torch.sqrt(gx_img ** 2 + gy_img ** 2)
        
        edge = int(beta * 0.2)
        G_img[:,:,:edge,:] = 0
        G_img[:,:,:,:edge] = 0
        G_img[:,:,-edge:,:] = 0
        G_img[:,:,:,-edge:] = 0

        GxDivG = torch.mul(gx_img, 1/G_img)
        GyDivG = torch.mul(gy_img, 1/G_img)
        for dex in range(G_img.shape[0]):
            G_img[dex][G_img[dex] <= torch.quantile(G_img[dex], 1-alpha)] = 0
        pix_pos = (G_img>0).nonzero()
        
        sigma_pos = np.random.uniform(margin, beta, (2, pix_pos.shape[0]))
        sigma_neg = np.random.uniform(-beta, -margin, (2, pix_pos.shape[0]))
        sigma = np.vstack((sigma_neg, sigma_pos))
        sigma.sort(axis=0)
        
        pix_loc = pix_pos.detach().long()
        pix_loc = list(pix_loc.T)
        GxDivG_loc = GxDivG[pix_loc]
        GyDivG_loc = GyDivG[pix_loc]

        pair_list = []
        for index in range(4):
            sig = sigma[index]
            sig = torch.Tensor(sig).to(GxDivG_loc.device)
            new_loc = copy.deepcopy(pix_loc)
            if index <= 1:
                new_loc[2] += torch.ceil(sig.mul(GyDivG_loc)).long()
                new_loc[3] += torch.ceil(sig.mul(GxDivG_loc)).long() #gradient direction!
            else:
                new_loc[2] += torch.floor(sig.mul(GyDivG_loc)).long()
                new_loc[3] += torch.floor(sig.mul(GxDivG_loc)).long() #gradient direction!
                
            new_loc[2] = torch.clamp(new_loc[2], 0, target.shape[2]-1)
            new_loc[3] = torch.clamp(new_loc[3], 0, target.shape[3]-1)
            pair_list.append(new_loc)
        return pair_list


    def cal_loss(self, pred_list, guided, low_dep, high_dep, thres = 0.001, beta_list=[60, 30, 15]):
        loss_r = 0
        loss_l = 0
        valid_pix = guided.clone().detach()
        low_dcopy = low_dep.clone().detach()
        target = guided.clone().detach()
        s_target = high_dep.clone().detach()
        
        target[:, :, :, 0] *= 0
        target[:, :, :, -1] *= 0
        target[:, :, 0, :] *= 0
        target[:, :, -1, :] *= 0
        top_thres =  1+thres
        bot_thres = 1./(top_thres)
        for index in range(3):
            beta = beta_list[index]
            pred = pred_list[index].clone()
            pair_list_h = self.sample_pairs(s_target, beta=beta)
            pair_list_g = self.sample_pairs(target, beta=beta)

            loss_rank = 0
            gt = low_dcopy.clone().detach()
            for dex in range(3):
                pix_pos = pair_list_h[dex]
                pair_part = pair_list_h[dex + 1]
                
                if index == 0:
                    valid_pix[pix_pos] = 0
                    valid_pix[pair_part] = 0

                l_part = torch.abs(target[pix_pos] / (target[pair_part]))
                diff_pred = pred[pix_pos] - pred[pair_part]
                diff_guided = target[pix_pos] - target[pair_part]
                zero_part = target[pix_pos] * target[pair_part]

                l_part[zero_part==0] = 0
                l_part[l_part>=top_thres] = 1
                l_part[l_part<=bot_thres] = -1
                l_part[torch.abs(l_part)!=1] = 0
                loss_rank += torch.mean(torch.mul(1-torch.abs(l_part), diff_pred**2) + torch.mul(torch.abs(l_part), torch.log(1 + torch.exp(-1/(torch.abs(diff_pred - diff_guided) + 0.1))))) * 8
            
            for dex in range(3):
                pix_pos = pair_list_g[dex]
                pair_part = pair_list_g[dex + 1]
                
                if index == 0:
                    valid_pix[pix_pos] = 0
                    valid_pix[pair_part] = 0

                l_part = torch.abs(target[pix_pos] / (target[pair_part]))
                diff_pred = pred[pix_pos] - pred[pair_part]
                diff_guided = target[pix_pos] - target[pair_part]
                zero_part = target[pix_pos] * target[pair_part]

                l_part[zero_part==0] = 0
                l_part[l_part>=top_thres] = 1
                l_part[l_part<=bot_thres] = -1
                l_part[torch.abs(l_part)!=1] = 0
                loss_rank += torch.mean(torch.mul(1-torch.abs(l_part), diff_pred**2) + torch.mul(torch.abs(l_part), torch.log(1 + torch.exp(-1/(torch.abs(diff_pred - diff_guided) + 0.1))))) * 12

            loss_l += torch.mean(torch.abs(pred - gt) + torch.abs(torch.tanh(pred) - torch.tanh(gt)))
            loss_r += loss_rank
            target = self.down_sample(target)
            s_target = self.down_sample(s_target)
            low_dcopy = self.down_sample(low_dcopy)
        return loss_l, loss_r, valid_pix


    def predict(self, low_dep, high_dep, axis=0, v_axis=0, guided=None):
        if axis == 1:
            low_dep = torch.flip(low_dep, dims=[2])
            high_dep = torch.flip(high_dep, dims=[2])
            guided = torch.flip(high_dep, dims=[2])
        elif axis == 2:
            low_dep = torch.flip(low_dep, dims=[3])
            high_dep = torch.flip(high_dep, dims=[3])
            guided = torch.flip(guided, dims=[3])
        elif axis == 3:
            low_dep = torch.flip(low_dep, dims=[2,3])
            high_dep = torch.flip(high_dep, dims=[2,3])
            guided = torch.flip(guided, dims=[2,3])
            
        if v_axis == 1:
            low_dep *= -1
            high_dep *= -1
            guided *= -1

        Fusion, out_half, out_quart = self.Fuse(low_dep, high_dep)
        return low_dep, high_dep, Fusion, out_half, out_quart, guided


    def inference(self, low_dep, high_dep):
        low_dep = self.pool(low_dep)
        high_dep = self.pool(high_dep)
        low_dep = (low_dep - low_dep.min()) / (low_dep.max() - low_dep.min())
        high_dep = (high_dep - high_dep.min()) / (high_dep.max() - high_dep.min())
        low_dep = low_dep * 2 -1
        high_dep = high_dep * 2 -1
        low_dep, high_dep, Fusion, _, _, _ = self.predict(low_dep, high_dep)
        return low_dep, high_dep, Fusion


    def evaluate(self, low_dep, high_dep, guided, photo):
        self.Fuse.eval()
        low_dep, high_dep, Fusion, out_half, out_quart, guided = self.predict(low_dep, high_dep, guided=guided)
        loss_low, loss_rank, valid_mask = self.cal_loss([Fusion, out_half, out_quart], guided, low_dep, high_dep)
        dep_val = loss_low.data.cpu().numpy()
        rank_val = loss_rank.data.cpu().numpy()
        self.Fuse.train()
        with SummaryWriter(self.log_path+'_eval') as w:
            w.add_scalar(tag='dep', scalar_value=dep_val, global_step=self.total_step)
            w.add_scalar(tag='rank', scalar_value=rank_val, global_step=self.total_step)
            w.add_scalar(tag='total', scalar_value=(dep_val + rank_val), global_step=self.total_step)


    def cret(self, low_dep, high_dep, guided):
        axis = np.random.randint(0, 4)
        v_axis = np.random.randint(0, 3)
        low_dep, high_dep, Fusion, out_half, out_quart, guided = self.predict(low_dep, high_dep, axis, v_axis, guided)
        loss_low, loss_rank, _ = self.cal_loss([Fusion, out_half, out_quart], guided, low_dep, high_dep)
        self.record_dep = np.append(self.record_dep, loss_low.data.cpu().numpy())
        self.record_rank = np.append(self.record_rank, loss_rank.data.cpu().numpy())
        self.total_step += 1
        return loss_low + loss_rank


    def record(self):
        dep_val = self.record_dep.mean()
        rank_val = self.record_rank.mean()
        self.record_dep = np.array([])
        self.record_rank = np.array([])
        with SummaryWriter(self.log_path) as w:
            w.add_scalar(tag='dep', scalar_value=dep_val, global_step=self.total_step)
            w.add_scalar(tag='rank', scalar_value=rank_val, global_step=self.total_step)
            w.add_scalar(tag='total', scalar_value=(dep_val + rank_val), global_step=self.total_step)

