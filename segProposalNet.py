import torch
import torch.nn as nn
from torch.autograd import Variable
import pointnet2
import torch.nn.functional as F
import random

class PointClassifier(nn.Module):
    def __init__(self, dropout = 0.1, input_size = 256, first_size = 64, point_num = 2048, cls_num = 10):
        super(PointClassifier, self).__init__()
        self.point_num = point_num
        self.cls_num = cls_num
        self.conv1 = torch.nn.Conv1d(input_size, first_size, 1)
        self.conv2 = torch.nn.Conv1d(first_size, int(first_size/4), 1)
        self.conv3 = torch.nn.Conv1d(int(first_size/4), cls_num, 1)
        self.bn1 = nn.BatchNorm1d(first_size)
        self.bn2 = nn.BatchNorm1d(int(first_size/4))
        self.drop1 = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x, flag):
        batchsize = x.size()[0]
        x = self.drop1(F.relu(self.bn1(self.conv1(x))))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        x = x.transpose(2,1).contiguous()
        if flag:
            x = self.softmax(x.view(-1, self.cls_num))
        else:
            x = F.log_softmax(x.view(-1, self.cls_num), dim=1)
        x = x.view(batchsize, self.point_num, self.cls_num)
        return x

class PartProposalMSGGlobal(nn.Module):
    def __init__(self, num_shape_points = 2048, hidden_size = 256):
        super(PartProposalMSGGlobal, self).__init__()

        self.num_shape_points = num_shape_points
        self.shapeEncoder = pointnet2.PointNet2PointFeature()

        self.condEncoderMSG = pointnet2.PointNetSetAbstractionMsg(512, [0.05,0.1,0.2], [8,16,32], 0, [[16,16,32], [32,32,64], [64,64,128]])
        self.condEncoderProp = pointnet2.PointNetFeaturePropagation(224, [196, 128])
        self.pointClassifier = PointClassifier(input_size = 256, cls_num = 2)


    def testForward(self, shape, gt):
        shapeFeat = self.shapeEncoder(shape)
        condFeat_xyz, condFeat_sampled = self.condEncoderMSG(shape, None)
        condFeat = self.condEncoderProp(shape, condFeat_xyz, None, condFeat_sampled)

        gtForFeat = gt.unsqueeze(1)
        gtForFeat = gtForFeat.repeat(1, 128, 1)
        
        condFeat = condFeat.mul(gtForFeat)
        condFeatWeight = condFeat.sum(2)

        weight = gt.sum(1)
        weight = 1/weight
        weight = weight.unsqueeze(1)
        weight = weight.repeat(1, 128)
        condFeatWeight = condFeatWeight.mul(weight)

        cond = condFeatWeight.view(-1, 128, 1).repeat(1, 1, self.num_shape_points)
        pointFeat = torch.cat([shapeFeat, cond], 1)
        pointLabel = self.pointClassifier(pointFeat, 1)
        return pointLabel

    def forward(self, shape, gt, flag):
        shapeFeat = self.shapeEncoder(shape)
        condFeat_xyz, condFeat_sampled = self.condEncoderMSG(shape, None)
        condFeat = self.condEncoderProp(shape, condFeat_xyz, None, condFeat_sampled)

        if flag:
            gtNoiseA = torch.mul(Variable(torch.rand(gt.size()).cuda()*0.2), gt)
            gtNoiseB = torch.mul(Variable(torch.rand(gt.size()).cuda()*0.2), gt - 1)
            gtNoise = gt-gtNoiseA-gtNoiseB
        else:
            gtNoiseA = torch.mul(Variable(torch.rand(gt.size()).cuda()*0.05), gt)
            gtNoiseB = torch.mul(Variable(torch.rand(gt.size()).cuda()*0.05), gt - 1)
            gtNoise = gt-gtNoiseA-gtNoiseB

        gtForFeat = gtNoise.unsqueeze(1)
        gtForFeat = gtForFeat.repeat(1, 128, 1)
        
        condFeat = condFeat.mul(gtForFeat)
        condFeatWeight = condFeat.sum(2)

        weight = gtNoise.sum(1)
        weight = 1/weight
        weight = weight.unsqueeze(1)
        weight = weight.repeat(1, 128)
        condFeatWeight = condFeatWeight.mul(weight)

        cond = condFeatWeight.view(-1, 128, 1).repeat(1, 1, self.num_shape_points)
        pointFeat = torch.cat([shapeFeat, cond], 1)
        pointLabel = self.pointClassifier(pointFeat, 0)
        return pointLabel


class SegProposalNet(nn.Module):
    def __init__(self, num_shape_points = 2048, num_cls = 4):
        super(SegProposalNet, self).__init__()

        self.num_shape_points = num_shape_points
        self.num_cls = num_cls

        self.shapeEncoder = pointnet2.PointNet2PointFeatureStable()
        self.pointClassifier = PointClassifier(input_size = 128, cls_num = self.num_cls)
        self.partProposal = PartProposalMSGGlobal()
        self.threshold = nn.Threshold(0.26, 0)
        print(num_cls)

    def testSeg(self, shape):
        shapeFeat = self.shapeEncoder(shape)
        segMatrix = self.pointClassifier(shapeFeat, 1)
        segMatrix = self.threshold(segMatrix)
        segMatrix = segMatrix.pow(8)
        segMatrix = segMatrix + 1e-6
        segMatrix = F.normalize(segMatrix, p=1, dim=2)
        ######################### Apply Part Proposal #############################
        for i in range(1):
            segMatrixTmp = Variable(torch.zeros([shape.size()[0], self.num_shape_points, self.num_cls]).cuda())
            for i in range(self.num_cls):
                segLabel = segMatrix[:,:,i]
                seg = self.partProposal.testForward(shape, segLabel)[:,:,1]
                segMatrixTmp[:,:,i] = seg
            segMatrix = self.threshold(segMatrix)
            segMatrix = F.normalize(segMatrixTmp, p=1, dim=2)
        ###########################################################################
        return segMatrix

    def forward(self, shape):
        batchsize = shape.size()[0]
        shapeFeat = self.shapeEncoder(shape)
        segMatrix = self.pointClassifier(shapeFeat, 0)
        segMatrix = torch.exp(segMatrix)

        segMatrix = self.threshold(segMatrix)
        segMatrix = segMatrix.pow(8)
        segMatrix = segMatrix + 1e-6
        segMatrix = F.normalize(segMatrix, p=1, dim=2)

        ######################### Apply Part Proposal #############################
        segMatrixTmp = Variable(torch.zeros([shape.size()[0], self.num_shape_points, self.num_cls]).cuda())
        for i in range(self.num_cls):
            segLabel = segMatrix[:,:,i]
            seg = self.partProposal.testForward(shape, segLabel)[:,:,1]
            segMatrixTmp[:,:,i] = seg
        segMatrix = segMatrix + 1e-6
        segMatrix = F.normalize(segMatrixTmp, p=1, dim=2)
        maxminProb = 1/torch.exp(torch.mean(segMatrix.max(2)[0]))
        ###########################################################################

        feat_xyz, feat_sampled = self.partProposal.condEncoderMSG(shape, None)
        pointFeat = self.partProposal.condEncoderProp(shape, feat_xyz, None, feat_sampled)

        tensorA = torch.transpose(segMatrix,2,1).unsqueeze(1).repeat(1, 128, 1, 1)
        tensorB =pointFeat.unsqueeze(2).repeat(1, 1, self.num_cls, 1)
        weightedFeat = torch.mul(tensorA, tensorB)
        weightedFeat = weightedFeat.max(3)[0]
    
        weightedFeat = weightedFeat.transpose(0,2).contiguous()

        lowRankLoss = Variable(torch.zeros([self.num_cls], dtype=torch.float).cuda())
        for i in range(self.num_cls):
            _, s, _ = torch.svd(weightedFeat[i,:,:], some=False)
            lowRankLoss[i] = s[1]/s[0]
            
        
        highRankLoss = Variable(torch.zeros([int((self.num_cls-1)*self.num_cls/2)], dtype=torch.float).cuda())
        idx = 0
        for i in range(self.num_cls):
            for j in range(self.num_cls):
                if j <= i:
                    continue
                _, s, _ = torch.svd(torch.cat([weightedFeat[i,:,:], weightedFeat[j,:,:]], 1), some=False)
                highRankLoss[idx] = s[1]/s[0]
                idx = idx + 1
        
        return 1 + torch.max(lowRankLoss) - torch.min(highRankLoss), maxminProb



    