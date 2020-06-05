import torch
from torch.utils import data
import h5py
import segProposalNet
import torch.optim as optim
from dataset import PartSpaceData
from torch.autograd import Variable
import torch.nn.functional as F

from tensorboard_logger import configure, log_value

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

configure("logs/run-chair-offline-1")

batchSize = 16
data = PartSpaceData('chair/chairSinglePartsTraining.hdf5')
dataloader = torch.utils.data.DataLoader(data, batch_size=batchSize, shuffle=True)

model = segProposalNet.PartProposalMSGGlobal()
model.cuda()

optimizer = optim.Adam(model.parameters(), lr=1e-5)

step = 0
for epoch in range(1000):
    if epoch is not 0 and epoch%3 is 0:
        torch.save(model, 'chair/PartSpace_Training.pkl')
    if epoch is not 0 and epoch%50 is 0:
        filename = 'chair/PartSpace_Training_'+str(epoch)+'.pkl'
        torch.save(model, filename)
    for i, batch in enumerate(dataloader):
        if batch[0].shape[0] < 2:
            continue
        step = step + 1
        shape = torch.transpose(batch[0],2,1)
        gt = Variable(batch[2]).cuda()
        shape = Variable(shape)
        shape = shape.cuda()

        weight = gt.float()
        weight = weight.sum(1)
        if batchSize > weight.nonzero().size(0):
            continue

        w = torch.tensor([0.3,0.7]).cuda()
        pointLabel= model(shape, gt.float(), 0)
        pointLabel = pointLabel.view(-1, 2)
        gtForLoss = gt.view(-1,1)[:,0]
        claLoss = F.nll_loss(pointLabel, gtForLoss,w)
        loss = claLoss
        loss.backward()
        optimizer.step()

        if i%5 == 0:
            pred_choice = pointLabel.data.max(1)[1]
            correct = float(pred_choice.eq(gtForLoss.data).cpu().sum())/(batchSize*2048)
            log_value('claLoss', claLoss.data.item(), step)
            log_value('correct', correct, step)

        if i%20 == 0:
            pred_choice = pointLabel.data.max(1)[1]
            correct = float(pred_choice.eq(gtForLoss.data).cpu().sum())/(batchSize*2048)
            print(str(claLoss.data.item())+","+str(correct))
            
            