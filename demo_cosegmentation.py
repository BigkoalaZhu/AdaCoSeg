import torch
from torch.utils import data
import h5py
import segProposalNet
import torch.optim as optim
from dataset import ShapeSegDataTest
from torch.autograd import Variable
import torch.nn.functional as F
import random
import draw3DP as draw3DPoints
import numpy as np

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

shapeCategory = "chair"
pretraining = 'chair/PartSpace_Training.pkl'
flag = "Testing"
segmentationK = 3

data = ShapeSegDataTest(shapeCategory+"/"+shapeCategory+"SingleParts"+flag+".hdf5")
batchSize = 6
dataloader = torch.utils.data.DataLoader(
    data, batch_size=batchSize, shuffle=True)

model = segProposalNet.SegProposalNet(num_cls=segmentationK)
partProposal = torch.load(pretraining)
model.partProposal.load_state_dict(partProposal.state_dict(), strict=False)
model.partProposal.eval()
model.cuda()

optimizer = optim.Adam([
    {'params': model.shapeEncoder.parameters()},
    {'params': model.pointClassifier.parameters()}
    ], lr=1e-3)
step = 0

print("Start training!")

torch.set_grad_enabled(True)

for epoch in range(100*segmentationK):
    print(epoch)
    for i, batch in enumerate(dataloader):
        if batch[0].shape[0] < 3:
            continue
        step = step + 1
        shape = torch.transpose(batch[0], 2, 1)
        shape = Variable(shape)
        shape = shape.cuda()

        lowRankLoss, maxminProb = model(shape)
        loss = lowRankLoss + maxminProb*0.3
        model.zero_grad()
        loss.backward()
        model.partProposal.zero_grad()
        optimizer.step()
print(loss)
print("End training!")

batchSize = 1
dataTestLoad= torch.utils.data.DataLoader(data, batch_size=batchSize, shuffle=False)
torch.set_grad_enabled(False)

print("Start testing!")

for i, batch in enumerate(dataTestLoad):

    model.eval()
    shape = torch.transpose(batch[0],2,1)
    with torch.no_grad():
        shape = Variable(shape)
        shape = shape.cuda()

    shapeNp = batch[0].numpy()
    shapeNp = np.squeeze(shapeNp, axis=0)

    seg = model.testSeg(shape)
    segLabel = seg.squeeze().data.max(1)[1]
    segMask = segLabel.cpu().numpy().astype(int)

    draw3DPoints.savePointCloudColorSeg(shapeNp, segMask, segmentationK, "coseg/test-" + str(i) + ".ply")
    
print("End testing!")
