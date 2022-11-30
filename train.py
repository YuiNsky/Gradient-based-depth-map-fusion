import os
import h5py
import torch
import numpy as np
import torch.utils.data as dataloader
from utils.model import Gradient_FusionModel
from torch.optim import lr_scheduler, AdamW
import torchvision.transforms as transforms


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DataFromH5File(dataloader.Dataset):
    def __init__(self, filepath):
        h5File = h5py.File(filepath, 'r', swmr=True)
        self.photo = h5File['photo'][:]
        self.depth = h5File['depth'][:]

    def __getitem__(self, idx):
        global device
        transform_= transforms.Compose([transforms.ToTensor(), transforms.Resize(size=(2048, 2048))])
        inputs = self.depth[idx]
        photo = self.photo[idx]
        with torch.no_grad():
            low = transform_(inputs[0].astype('float32')).to(device)
            high = transform_(inputs[1].astype('float32')).to(device)
            guided = transform_(inputs[2].astype('float32')).to(device)
        return low, high, guided, photo

    def __len__(self):
        return self.photo.shape[0]


def Train_HR(Fuse_model, optimizer, scheduler):
    index_list = [i+1 for i in range(36)]
    for index in index_list:
        data_loc = f'./datasets/HR/hq_HR_{index}.hdf5'
        data_set = DataFromH5File(data_loc)
        train_loader = dataloader.DataLoader(dataset=data_set, batch_size=2, shuffle=False, num_workers=0, pin_memory=False)
        for step, (low, high, guided, photo) in enumerate(train_loader):
            optimizer.zero_grad()
            loss = Fuse_model.cret(low.clone().detach(), high.clone().detach(), guided.clone().detach())
            loss.backward()
            optimizer.step()
            if Fuse_model.total_step % 100 == 0:
                Fuse_model.record()
                Fuse_model.evaluate(low, high, guided, photo)
                scheduler.step()
    return Fuse_model, optimizer, scheduler


def run():
    global device
    log_path = 'logs/test'
    Fuse_model = Gradient_FusionModel(log_path=log_path)
    Fuse_model.to(device)

    optimizer = AdamW(filter(lambda p: p.requires_grad, Fuse_model.Fuse.parameters()), lr=(1e-4) * 1)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)
    for i in range(2):
        Fuse_model, optimizer, scheduler = Train_HR(Fuse_model, optimizer, scheduler)
        state = {'net': Fuse_model.Fuse.state_dict()}
        torch.save(state, log_path +f'/model_dict_{i}.pt')
    print(f'finish')


if __name__ == '__main__':  
    run()
