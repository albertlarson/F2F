import mat73
import torch
import time
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
import matplotlib.pyplot as plt
from datetime import datetime
from pytz import timezone
import xarray as xr
import scipy.ndimage
east = timezone('US/Eastern')



class dp_patches(Dataset):
    def __init__(self,x):
        self.x = torch.load(x)[:,:,:,:54].to('cuda');
        self.y = torch.load(x)[:,:,:,54].unsqueeze(1).to('cuda');
#         rand_pts = torch.from_numpy(np.random.randint(0,self.x.shape[0],nn_ds_size))
        self.x = self.x
        self.y = self.y
#         self.x = self.x[rand_pts]
#         self.y = self.y[rand_pts]
    def __getitem__(self,idx):
        x = self.x[idx]
        y = self.y[idx]
        return x, y
    def __len__(self):
        return self.x.shape[0]

class MLP(torch.nn.Module):
    def __init__(self,kernel_size):
        super(MLP,self).__init__()
        self.fci = torch.nn.Linear(kernel_size ** 2 * 6,10)
        self.fco = torch.nn.Linear(10,1)
        self.relu = torch.nn.ReLU()
#         self.dropout = torch.nn.Dropout(0.5)

    def forward(self,x):
        x = self.relu(self.fci(x))
        x = self.fco(x)
        return x

    
# nn_ds_size = 1000
dset = dp_patches('nn50000polyreg_matlabmade.pt')
kernel_size=3
epochs=1000
batch_size=100
train_dset_size = int(0.85*len(dset))
valid_dset_size = int(len(dset) - train_dset_size)
train_dset, valid_dset = random_split(dset,[train_dset_size,valid_dset_size])
train_dataloader = DataLoader(dataset=train_dset, batch_size=batch_size, shuffle=False)
valid_dataloader = DataLoader(dataset=valid_dset, batch_size=batch_size, shuffle=False)
model = MLP(kernel_size)
model.cuda()
optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4)
criterion = torch.nn.MSELoss(reduction='mean')
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,step_size=150,verbose=False)


print('training started at {}'.format(datetime.now(east).strftime('%Y-%m-%d %H:%M:%S')))
t_loss = []
v_loss = []
e_time = []
t0 = time.time()
for i in range(epochs):
    t_e_loss = 0
    v_e_loss = 0
    t00 = time.time()
    model.train()
    for idx,(x,y) in enumerate(train_dataloader):
        optimizer.zero_grad()
        predicted = model(x)
        loss = criterion(predicted,y)
        loss.backward()
        optimizer.step()
        t_e_loss += loss.item()
    for xx,yy in valid_dataloader:
        v_pred = model(xx)
        loss2 = criterion(v_pred,yy)
        v_e_loss += loss2.item()
    t_loss.append(t_e_loss/len(train_dataloader))
    v_loss.append(v_e_loss/len(valid_dataloader))
    t11 = time.time()
    e_time.append(t11-t00)
t1 = time.time()
print('training time {} minutes'.format(np.format_float_positional((t1-t0)/60,precision=5)))

fig = plt.figure()
fig.add_subplot(211)
plt.title('mean squared error between predicted target and actual target vs. epoch\n1 hidden layer, 10 neurons, 50,000 samples\n 85/15 split, 1000 epochs, 100 batchsize, 1e-4 lr',fontsize=9)
plt.plot(t_loss)
plt.plot(v_loss)
plt.legend(['t','v'])
plt.ylabel('mse')
fig.add_subplot(212)
plt.title('time',fontsize=9)
plt.tight_layout()
plt.plot(e_time,color='red')
# plt.plot(np.cumsum(e_time),color='green')
plt.legend(['/epoch (sec)','cum (min)'])

plt.savefig('bench_50000_random.png')

t_loss = np.array(t_loss)
v_loss = np.array(v_loss)
e_time = np.array(e_time)

np.savez('benchvars_50000_random',t_loss=t_loss,v_loss=v_loss,e_time=e_time)

torch.save(model,'mlp_model_50000_random.pth')


###infer

am_12221 = xr.open_dataset('../runs_gulf/12-2-21.nc')
kernel_size=3

model = torch.load('mlp_model_50000_random.pth')

am__sst__for_nn = torch.Tensor(np.array(am_12221.amsre)).unsqueeze(0).unsqueeze(0).float()

g1 = torch.from_numpy(np.array([[1,1],[-1,-1]])).unsqueeze(0).unsqueeze(0).float()
g2 = torch.from_numpy(np.array([[1,-1],[1,-1]])).unsqueeze(0).unsqueeze(0).float()
g3 = torch.from_numpy(np.array([[-1,1],[1,-1]])).unsqueeze(0).unsqueeze(0).float()
amsremedfilt = torch.from_numpy(scipy.ndimage.median_filter(am__sst__for_nn.squeeze().squeeze(),3)).unsqueeze(0).unsqueeze(0).float()
unfold = torch.nn.Unfold(kernel_size=(kernel_size,kernel_size),stride=1)
amsre_g1 = torch.nn.functional.conv2d(am__sst__for_nn,g1);
amsre_g2 = torch.nn.functional.conv2d(am__sst__for_nn,g2)
amsre_g3 = torch.nn.functional.conv2d(am__sst__for_nn,g3)
amsre_g4 = torch.nn.functional.conv2d(amsremedfilt,g1)
amsre_g5 = torch.nn.functional.conv2d(amsremedfilt,g2)
amsre_g6 = torch.nn.functional.conv2d(amsremedfilt,g3)

i1patch = unfold(amsre_g1).permute(2,0,1).reshape(-1,1,kernel_size,kernel_size)
i2patch = unfold(amsre_g2).permute(2,0,1).reshape(-1,1,kernel_size,kernel_size)
i3patch = unfold(amsre_g3).permute(2,0,1).reshape(-1,1,kernel_size,kernel_size)
i4patch = unfold(amsre_g4).permute(2,0,1).reshape(-1,1,kernel_size,kernel_size)
i5patch = unfold(amsre_g5).permute(2,0,1).reshape(-1,1,kernel_size,kernel_size)
i6patch = unfold(amsre_g6).permute(2,0,1).reshape(-1,1,kernel_size,kernel_size)
i1patch = i1patch.reshape(i1patch.shape[0],1,1,-1)
i2patch = i2patch.reshape(i2patch.shape[0],1,1,-1)
i3patch = i3patch.reshape(i3patch.shape[0],1,1,-1)
i4patch = i4patch.reshape(i4patch.shape[0],1,1,-1)
i5patch = i5patch.reshape(i5patch.shape[0],1,1,-1)
i6patch = i6patch.reshape(i6patch.shape[0],1,1,-1)
input_patches = torch.cat((i1patch,i2patch,i3patch,i4patch,i5patch,i6patch),dim=3)

class infer_ds(Dataset):
    def __init__(self,inp):
        self.inp = inp.to('cuda')
    def __getitem__(self,idx):
        x = self.inp[idx]
        return x
    def __len__(self):
        return self.inp.shape[0]

ds = infer_ds(input_patches[:,:,:,:kernel_size**2*6])
dl = DataLoader(ds,shuffle=False)

predicted = []

with torch.no_grad():
    model.eval()
    for i in dl:
        prediction = model(i)
        predicted.append(prediction.item())
predicted = np.asarray(predicted)

predicted = predicted.reshape(46,91)
predicted = np.pad(predicted,(0,3),'edge')
np.save('mlp_inferout_50000_random_polyreg_matlabmade.npy',am_12221.amsre + predicted)