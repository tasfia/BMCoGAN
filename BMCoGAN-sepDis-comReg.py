# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 21:24:59 2020

@author: tshermin
"""

from __future__ import print_function
from easydict import EasyDict
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import util
import classifier
from center_loss import TripCenterLoss_min_margin,TripCenterLoss_margin
import classifier_latent
import sys
import model
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
#import torch
#import h5py
import numpy as np
import scipy.io as sio
import torch
from sklearn import preprocessing
import sys

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label==classes[i]] = i    

    return mapped_label

class Logger(object):
    def __init__(self, filename):
        self.filename = filename
        f = open(self.filename+'.log', "a")
        f.close()

    def write(self, message):
        f = open(self.filename+'.log', "a")
        f.write(message)  
        f.close()

class DATA_LOADER(object):
    def __init__(self, opt):
        if opt.matdataset:
            if opt.dataset == 'imageNet1K':
                self.read_matimagenet(opt)
            else:
                self.read_matdataset(opt)
        self.index_in_epoch = 0
        self.epochs_completed = 0

    def read_matdataset(self, opt):
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat")
        feature = matcontent['features'].T
        label = matcontent['labels'].astype(int).squeeze() - 1
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + "_splits.mat")
        # numpy array index starts from 0, matlab starts from 1
        trainval_loc = matcontent['trainval_loc'].squeeze() - 1
        train_loc = matcontent['train_loc'].squeeze() - 1
        val_unseen_loc = matcontent['val_loc'].squeeze() - 1
        test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
        test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1
    
        self.attribute = torch.from_numpy(matcontent['att'].T).float() 
        
        if not opt.validation:
            if opt.preprocessing:
                if opt.standardization:
                    print('standardization...')
                    scaler = preprocessing.StandardScaler()
                else:
                    scaler = preprocessing.MinMaxScaler()
                
                _train_feature = scaler.fit_transform(feature[trainval_loc])
                _test_seen_feature = scaler.transform(feature[test_seen_loc])
                _test_unseen_feature = scaler.transform(feature[test_unseen_loc])
                self.train_feature = torch.from_numpy(_train_feature).float()
                mx = self.train_feature.max()
                self.train_feature.mul_(1/mx)
                self.train_label = torch.from_numpy(label[trainval_loc]).long() 
                self.test_unseen_feature = torch.from_numpy(_test_unseen_feature).float()
                self.test_unseen_feature.mul_(1/mx)
                self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long() 
                self.test_seen_feature = torch.from_numpy(_test_seen_feature).float() 
                self.test_seen_feature.mul_(1/mx)
                self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()
            else:
                self.train_feature = torch.from_numpy(feature[trainval_loc]).float()
                self.train_label = torch.from_numpy(label[trainval_loc]).long() 
                self.test_unseen_feature = torch.from_numpy(feature[test_unseen_loc]).float()
                self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long() 
                self.test_seen_feature = torch.from_numpy(feature[test_seen_loc]).float() 
                self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()
        else:
            self.train_feature = torch.from_numpy(feature[train_loc]).float()
            self.train_label = torch.from_numpy(label[train_loc]).long()
            self.test_unseen_feature = torch.from_numpy(feature[val_unseen_loc]).float()
            self.test_unseen_label = torch.from_numpy(label[val_unseen_loc]).long() 
            
    
        self.seenclasses = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))
        self.seenTestclasses = torch.from_numpy(np.unique(self.test_seen_label.numpy()))
        self.attribute_unseen=self.attribute[self.unseenclasses]
        self.attribute_seen = self.attribute[self.seenclasses]
        self.attribute_seen_test = self.attribute[self.seenTestclasses]
        self.ntrain = self.train_feature.size()[0]
        self.ntestSeen = self.test_seen_feature.size()[0]
        self.ntrain_u = self.test_unseen_feature.size()[0]
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)
        self.ntest_Seenclass = self.seenTestclasses.size(0)
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.ntrain_class+self.ntest_class).long()
        #print(self.ntrain)
        self.train_mapped_label = map_label(self.train_label, self.seenclasses)
        

        self.all_feature = torch.cat((self.train_feature, self.test_unseen_feature,self.test_seen_feature))
        self.both_feature = torch.cat((self.train_feature, self.test_unseen_feature))
 
    def next_batch(self, batch_size):
        idx = torch.randperm(self.ntrain)[0:batch_size]
        batch_feature = self.train_feature[idx]
        batch_label = self.train_label[idx]
        batch_att = self.attribute[batch_label]       
        batch_unseen_att = torch.zeros(batch_size, 312)#.cuda()
       
        batch_unseen_label = torch.zeros(batch_size, dtype=torch.long)#.cuda()
        j=0
        for i in range(batch_size):
            if j == 50:
                j=0
            batch_unseen_att[i] = self.attribute[self.unseenclasses[j]]
            batch_unseen_label[i] = self.unseenclasses[j]
            j=j+1  
        return batch_feature, batch_label, batch_att,batch_unseen_att,batch_unseen_label#,batch_feature_unseen2
    
    def next_batch_trainM(self, batch_size):
        idx = torch.randperm(self.ntrain)[0:batch_size]
        #batch_feature = self.train_feature[idx]
        batch_label = self.train_label[idx]
        batch_att = self.attribute[batch_label]
        return batch_label, batch_att
    
    def next_batch_TestM(self, batch_size):
        idx = torch.randperm(self.ntestSeen)[0:batch_size]
        #batch_feature = self.train_feature[idx]
        batch_label = self.test_seen_label[idx]
        batch_att = self.attribute[batch_label]
        return batch_label, batch_att
    
    def next_batch_M(self, batch_size):
        print("ok")
        idx_seen = torch.randperm(self.ntrain_class)[0:batch_size]
        idx_unseen = torch.zeros(batch_size, 1, dtype=torch.int32).cuda()
        batch_unseen_att = torch.zeros(batch_size, 312).cuda()
        batch_unseen_label = torch.zeros(batch_size, dtype=torch.int32).cuda()
        j=0
        for i in range(batch_size):
            if j == 50:
                j=0
            idx_unseen[i] = self.unseenclasses[j]
            batch_unseen_att[i] = self.attribute[self.unseenclasses[j]]
            batch_unseen_label[i] = self.unseenclasses[j]
            j=j+1  
                #print(idx_unseen)
                
        print(batch_unseen_label.size())
        #print(batch_unseen_label)
        print(batch_unseen_att.size())
        
        batch_seen_att = self.attribute_seen[idx_seen]
        
        batch_seen_label = self.seenclasses[idx_seen]
       
        return idx_seen, idx_unseen, batch_seen_att, batch_unseen_att, batch_seen_label, batch_unseen_label

    def next_batch_one_class(self, batch_size):
        if self.index_in_epoch == self.ntrain_class:
            self.index_in_epoch = 0 
            perm = torch.randperm(self.ntrain_class)
            self.train_class[perm] = self.train_class[perm]

        iclass = self.train_class[self.index_in_epoch]
        idx = self.train_label.eq(iclass).nonzero().squeeze()
        perm = torch.randperm(idx.size(0))
        idx = idx[perm]
        iclass_feature = self.train_feature[idx]
        iclass_label = self.train_label[idx]
        self.index_in_epoch += 1
        return iclass_feature[0:batch_size], iclass_label[0:batch_size], self.attribute[iclass_label[0:batch_size]] 

    def next_batch_transductive(self, batch_size):
        idx = torch.randperm(self.ntrain)[0:batch_size]
        batch_seen_feature = self.train_feature[idx]
        batch_seen_label = self.train_label[idx]
        batch_seen_att = self.attribute[batch_seen_label]

        idx = torch.randperm(self.all_feature.shape[0])[0:batch_size]
        batch_both_feature = self.all_feature[idx]
        idx_both_att = torch.randint(0, self.attribute.shape[0], (batch_size,))
        batch_both_att = self.attribute[idx_both_att]

        return batch_seen_feature, batch_seen_label, batch_seen_att, batch_both_feature, batch_both_att

    def next_batch_transductive_both(self, batch_size):
        idx = torch.randperm(self.ntrain)[0:batch_size]
        batch_seen_feature = self.train_feature[idx]
        batch_seen_label = self.train_label[idx]
        batch_seen_att = self.attribute[batch_seen_label]

        idx = torch.randperm(self.both_feature.shape[0])[0:batch_size]
        batch_both_feature = self.both_feature[idx]
        idx_both_att = torch.randint(0, self.attribute.shape[0], (batch_size,))
        batch_both_att = self.attribute[idx_both_att]

        return batch_seen_feature, batch_seen_label, batch_seen_att, batch_both_feature, batch_both_att

    def next_batch_MMD(self, batch_size):
        # idx = torch.randperm(self.ntrain)[0:batch_size]
        index = torch.randint(self.seenclasses.shape[0], (2,))
        while index[0]==index[1]:
            index = torch.randint(self.seenclasses.shape[0], (2,))
        select_labels=self.seenclasses[index]
        X_features=self.train_feature[self.train_label==select_labels[0]]
        Y_features = self.train_feature[self.train_label == select_labels[1]]

        idx_X = torch.randperm(X_features.shape[0])[0:batch_size]
        X_features = X_features[idx_X]

        idx_Y = torch.randperm(Y_features.shape[0])[0:batch_size]
        Y_features = Y_features[idx_Y]

        return X_features,Y_features

    def next_batch_MMD_all(self):
        # idx = torch.randperm(self.ntrain)[0:batch_size]
        index = torch.randint(self.seenclasses.shape[0], (2,))
        while index[0]==index[1]:
            index = torch.randint(self.seenclasses.shape[0], (2,))
        select_labels=self.seenclasses[index]
        X_features=self.train_feature[self.train_label==select_labels[0]]
        Y_features = self.train_feature[self.train_label == select_labels[1]]

        return X_features,Y_features

    def next_batch_unseenatt(self, batch_size,unseen_batch_size):
        idx = torch.randperm(self.ntrain)[0:batch_size]
        batch_feature = self.train_feature[idx]
        batch_label = self.train_label[idx]
        batch_att = self.attribute[batch_label]

        # idx = torch.randperm(data)[0:batch_size]
        idx_unseen =torch.randint(0, self.unseenclasses.shape[0], (unseen_batch_size,))
        unseen_label=self.unseenclasses[idx_unseen]
        unseen_att=self.attribute[unseen_label]

        return batch_feature, batch_label, batch_att,unseen_label,unseen_att

    # select batch samples by randomly drawing batch_size classes    
    def next_batch_uniform_class(self, batch_size):
        batch_class = torch.LongTensor(batch_size)
        for i in range(batch_size):
            idx = torch.randperm(self.ntrain_class)[0]
            batch_class[i] = self.train_class[idx]
            
        batch_feature = torch.FloatTensor(batch_size, self.train_feature.size(1))       
        batch_label = torch.LongTensor(batch_size)
        batch_att = torch.FloatTensor(batch_size, self.attribute.size(1))
        for i in range(batch_size):
            iclass = batch_class[i]
            idx_iclass = self.train_label.eq(iclass).nonzero().squeeze()
            idx_in_iclass = torch.randperm(idx_iclass.size(0))[0]
            idx_file = idx_iclass[idx_in_iclass]
            batch_feature[i] = self.train_feature[idx_file]
            batch_label[i] = self.train_label[idx_file]
            batch_att[i] = self.attribute[batch_label[i]] 
        return batch_feature, batch_label, batch_att

parser = argparse.ArgumentParser(description='ADA')
args = parser.parse_known_args()[0]
args = EasyDict({
    "dataset": 'CUB',
    "dataroot": 'C:/PHD/data/RRF-GZSL/data',
    "matdataset": True,
    "image_embedding":'res101',
    "class_embedding":'att',
    "syn_num":400,
    "gzsl":True,
    "preprocessing": False,
    "standardization":False,
    "validation":False,
    "workers":2,
    "batch_size":512,
    "batch_size_M":512,
    "resSize":2048,
    "attSize":312,
    "nz":312,
    "ngh":4096,
    "latenSize":1024,
    "nepoch":5000,
    "critic_iter":5,
    "i_c":0.1,
    "lambda1":10,
    "cls_weight":0.2,
    "lr":0.0001,
    "classifier_lr":0.001,
    "beta1":0.5,
    "cuda":True,
    "ngpu":1,
    "manualSeed":3483,
    "nclass_all":200,
    "nclass_seen":150,
    "lr_dec":False,
    "lr_dec_ep":1,
    "lr_dec_rate":0.95,
    "final_classifier":'softmax',
    "k":1,
    "n_power":1,
    "radius":3.5,
    "center_margin":190,
    "center_marginu":200,
    "center_weight":0.1,
    "save_path":'C:/PHD/data/RRF-GZSL/'
})
args.cuda =  torch.cuda.is_available()

opt = args
#print(opt)

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# load data
data = DATA_LOADER(opt)
#########################################################################################################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        
class MLP_g(nn.Module):
    def __init__(self, opt):
        super(MLP_g, self).__init__()
        self.SharedFC = nn.Linear(opt.attSize + opt.nz, opt.ngh)
        #self.fc1 = nn.Linear(opt.attSize, opt.ngh)
        self.fc1 = nn.Linear(opt.ngh, opt.resSize)
        self.fc2 = nn.Linear(opt.ngh, opt.resSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.apply(weights_init)

    def forward(self, noise, atts, attu):
        hs = torch.cat((noise, atts), 1)
        hu = torch.cat((noise, attu), 1)
        #h = noise * att
        hs = self.lrelu(self.SharedFC(hs))
        hu = self.lrelu(self.SharedFC(hu))
        
        hs = self.lrelu(self.fc1(hs))
        hu = self.lrelu(self.fc2(hu))
        return hs, hu

class Regressor(nn.Module):
    def __init__(self,opt):
        super(Regressor, self).__init__()
        self.Sharedfc = nn.Linear(opt.resSize, 1024)
        self.fc1 = nn.Linear(1024, opt.attSize)
        self.fc2 = nn.Linear(1024, opt.attSize)
        self.relu = nn.ReLU(True)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.discriminatorS = nn.Linear(opt.attSize, 1)
        self.discriminatorU = nn.Linear(opt.attSize, 1)
        self.apply(weights_init)
        
    def forward(self, s, u):
        ConstructedAtts = self.lrelu(self.Sharedfc(s))
        ConstructedAttu = self.lrelu(self.Sharedfc(u))
        
        ConstructedAtts = self.relu(self.fc1(ConstructedAtts))
        ConstructedAttu = self.relu(self.fc2(ConstructedAttu))
        
        #dis_s = self.discriminatorS(ConstructedAtts)
        #dis_u = self.discriminatorU(ConstructedAttu)
        #return dis_s, dis_u
        return ConstructedAtts, ConstructedAttu
        
class Discriminators(nn.Module):
    def __init__(self, opt):
        super(Discriminators, self).__init__()
        #self.latensize = opt.latenSize
        #self.encoder_linear = nn.Linear(opt.resSize, opt.latenSize)
        self.discriminatorS = nn.Linear(opt.attSize, 1)
        self.discriminatorU = nn.Linear(opt.attSize, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.sigmoid = nn.Sigmoid()
        self.logic = nn.LogSoftmax(dim=1)
        self.apply(weights_init)
        

    #def forward(self, s, u):   
        #dis_s = self.discriminatorS(s)
        #dis_u = self.discriminatorU(u)

        #return dis_s, dis_u
        
    def forward(self, x):   
        dis = self.discriminatorS(x)
        #dis_u = self.discriminatorU(u)

        return dis
    
class Mapping(nn.Module):
    def __init__(self, opt):
        super(Mapping, self).__init__()
        #self.latensize = opt.latenSize
        self.encoder_linear = nn.Linear(opt.resSize, opt.latenSize)
        self.discriminator = nn.Linear(opt.latenSize, 1)
        self.classifier = nn.Linear(opt.latenSize, opt.nclass_seen)
        self.classifier2 = nn.Linear(opt.latenSize, 50)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.sigmoid = nn.Sigmoid()
        self.logic = nn.LogSoftmax(dim=1)
        self.apply(weights_init)
        

    def forward(self, x):
        laten=self.lrelu(self.encoder_linear(x))
        #seen,unseen = laten[:,:512],laten[:,512:]
        #stds=self.sigmoid(stds)
        #encoder_out = reparameter(mus, stds)
        #if not se:
        dis_out = self.discriminator(laten)
        #else:
            #dis_out = self.discriminator(laten)
        pred=self.logic(self.classifier(laten))
        #predu=self.logic(self.classifier2(laten))

        return dis_out,pred,laten
    
##################################################################################################

class CLASSIFIER:
    def __init__(self, map,latenSize, _train_X, _train_Y, data_loader, _nclass, _cuda, _lr=0.001, _beta1=0.5,
                 _nepoch=20, _batch_size=100, val=True):
        self.train_X = _train_X
        self.train_Y = _train_Y
        self.test_seen_feature = data_loader.test_seen_feature
        self.test_seen_label = data_loader.test_seen_label
        self.test_unseen_feature = data_loader.test_unseen_feature
        self.test_unseen_label = data_loader.test_unseen_label
        self.seenclasses = data_loader.seenclasses
        self.unseenclasses = data_loader.unseenclasses
        self.batch_size = _batch_size
        self.nepoch = _nepoch
        self.nclass = _nclass
        self.input_dim = _train_X.size(1)
        self.latent_dim = latenSize
        self.cuda = _cuda
        self.model = LINEAR_LOGSOFTMAX(self.latent_dim, self.nclass)
        self.model.apply(util.weights_init)
        self.criterion = nn.NLLLoss()
        self.map = map
        for p in self.map.parameters():  # reset requires_grad
            p.requires_grad = False

        self.input = torch.FloatTensor(_batch_size, self.input_dim)
        self.label = torch.LongTensor(_batch_size)

        self.lr = _lr
        self.beta1 = _beta1
        # setup optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=_lr, betas=(_beta1, 0.999))
        #self.optimizer = _optimizer

        if self.cuda:
            self.model.cuda()
            self.criterion.cuda()
            self.input = self.input.cuda()
            self.label = self.label.cuda()

        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.ntrain = self.train_X.size()[0]

        #if generalized:
        self.acc_seen, self.acc_unseen, self.H = self.fit()
        #else:
            #self.acc = self.fit_zsl()
    
    def fit(self):
        best_H = 0
        best_seen = 0
        best_unseen = 0

        for epoch in range(self.nepoch):
            for i in range(0, self.ntrain, self.batch_size):
                self.model.zero_grad()
                batch_input, batch_label = self.next_batch(self.batch_size)
                self.input.copy_(batch_input)
                self.label.copy_(batch_label)

                inputv = Variable(self.input)
                labelv = Variable(self.label)

                #seen,unseen,dis_out,preds,predu,laten = self.map(inputv)
                #output = self.model(laten)
                output = self.model(inputv)

                loss = self.criterion(output, labelv)
                loss.backward()
                self.optimizer.step()

            acc_seen = self.val_gzsl(self.test_seen_feature, self.test_seen_label, self.seenclasses)
            acc_unseen = self.val_gzsl(self.test_unseen_feature, self.test_unseen_label, self.unseenclasses)
            H = 2 * acc_seen * acc_unseen / (acc_seen + acc_unseen)
            if H > best_H:
                best_seen = acc_seen
                best_unseen = acc_unseen
                best_H = H
        return best_seen, best_unseen, best_H

    def next_batch(self, batch_size):
        start = self.index_in_epoch
        # shuffle the data at the first epoch
        if self.epochs_completed == 0 and start == 0:
            perm = torch.randperm(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
        # the last batch
        if start + batch_size > self.ntrain:
            self.epochs_completed += 1
            rest_num_examples = self.ntrain - start
            if rest_num_examples > 0:
                X_rest_part = self.train_X[start:self.ntrain]
                Y_rest_part = self.train_Y[start:self.ntrain]
            # shuffle the data
            perm = torch.randperm(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
            # start next epoch
            start = 0
            self.index_in_epoch = batch_size - rest_num_examples
            end = self.index_in_epoch
            X_new_part = self.train_X[start:end]
            Y_new_part = self.train_Y[start:end]
            # print(start, end)
            if rest_num_examples > 0:
                return torch.cat((X_rest_part, X_new_part), 0), torch.cat((Y_rest_part, Y_new_part), 0)
            else:
                return X_new_part, Y_new_part
        else:
            self.index_in_epoch += batch_size
            end = self.index_in_epoch
            return self.train_X[start:end], self.train_Y[start:end]

    def val_gzsl(self, test_X, test_label, target_classes):
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start + self.batch_size)
            if self.cuda:
                #mus,stds,dis_out,pred,encoder_out,laten = self.map(test_X[start:end].cuda())
                #output = self.model(laten)
                output = self.model(test_X[start:end].cuda())
            else:

                #mus,stds,dis_out,pred,encoder_out,laten = self.map(test_X[start:end])
                output = self.model(test_X[start:end])
            _, predicted_label[start:end] = torch.max(output.data, 1)

            start = end
        acc = self.compute_per_class_acc_gzsl(test_label, predicted_label, target_classes)
        return acc


    def compute_per_class_acc_gzsl(self, test_label, predicted_label, target_classes):
        acc_per_class = 0
        for i in target_classes:
            idx = (test_label == i)
            acc_per_class += float(torch.sum(test_label[idx] == predicted_label[idx])) / float(torch.sum(idx))
        acc_per_class /= float(target_classes.size(0))
        return acc_per_class

    def val(self, test_X, test_label, target_classes):
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start + self.batch_size)
            if self.cuda:
                #mus,stds,dis_out,pred,encoder_out,laten = self.map(test_X[start:end].cuda())
                output = self.model(test_X[start:end].cuda())
            else:
                #mus,stds,dis_out,pred,encoder_out,laten = self.map(test_X[start:end])
                output = self.model(test_X[start:end])
            _, predicted_label[start:end] = torch.max(output.data, 1)
            start = end

        acc = self.compute_per_class_acc(util.map_label(test_label, target_classes), predicted_label,
                                         target_classes.size(0))
        return acc

    def compute_per_class_acc(self, test_label, predicted_label, nclass):
        acc_per_class = torch.FloatTensor(nclass).fill_(0)
        for i in range(nclass):
            idx = (test_label == i)
            acc_per_class[i] = float(torch.sum(test_label[idx] == predicted_label[idx])) / float(torch.sum(idx))
        return acc_per_class.mean()


class LINEAR_LOGSOFTMAX(nn.Module):
    def __init__(self, input_dim, nclass):
        super(LINEAR_LOGSOFTMAX, self).__init__()
        #self.fc = nn.Linear(input_dim, nclass)
        self.fc = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, nclass)
        self.logic = nn.LogSoftmax(dim=1)
        self.relu = nn.ReLU(True)
        self.lrelu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        o = self.logic(self.fc2(self.lrelu(self.fc(x))))
        #o = self.logic(self.fc(x))
        return o

##########################################################################################################
        

#print("# of training samples: ", data.ntrain)

#print(data.attribute_unseen.size())
#print(data.attribute_seen.size())



# initialize generator and discriminator
netG = MLP_g(opt)
Reg = Regressor(opt)
Dis = Discriminators(opt)
DisS = Discriminators(opt)
DisU = Discriminators(opt)
mapping= Mapping(opt)
#Cls_a = Classifier_A(opt).cuda()
#cls = MLPClassifier(2048, opt.nclass_all)
#cls_criterion = nn.NLLLoss()
criterion = nn.CrossEntropyLoss()
mse = nn.MSELoss()
#vat_loss = VirtualAversarialTraining(cls).cuda()
#cent = ConditionalEntropyLoss(cls).cuda()
#criterion1 = nn.NLLLoss()
#Cls = Classifier(opt).cuda()
cls_criterion = nn.NLLLoss()
if opt.dataset in ['CUB','SUN']:
    center_criterion = TripCenterLoss_margin(num_classes=opt.nclass_seen, feat_dim=opt.latenSize, use_gpu=opt.cuda)
elif opt.dataset in ['AWA1','FLO']:
    center_criterion = TripCenterLoss_min_margin(num_classes=opt.nclass_seen, feat_dim=opt.latenSize, use_gpu=opt.cuda)
else:
    raise ValueError('Dataset %s is not supported'%(opt.dataset))

input_res = torch.FloatTensor(opt.batch_size, opt.resSize)
input_att_s = torch.FloatTensor(opt.batch_size, opt.attSize)
noise_s = torch.FloatTensor(opt.batch_size, opt.nz)
noise_u = torch.FloatTensor(opt.batch_size, opt.nz)
input_label_s = torch.LongTensor(opt.batch_size)
input_att_u = torch.FloatTensor(opt.batch_size, opt.attSize)
input_label_u = torch.LongTensor(opt.batch_size)
input_res_u = torch.FloatTensor(opt.batch_size, opt.resSize)
input_res_u2 = torch.FloatTensor(opt.batch_size, opt.resSize)
beta=0

# i_c=0.2

if opt.cuda:
    mapping.cuda()
    netG.cuda()
    Reg.cuda()
    Dis.cuda()
    DisS.cuda()
    DisU.cuda()
    #cls.cuda()
    criterion.cuda()
    mse.cuda()
    input_res = input_res.cuda()
    input_res_u = input_res_u.cuda()
    input_res_u2 = input_res_u2.cuda()
    noise_s, input_att_s = noise_s.cuda(), input_att_s.cuda()
    noise_u, input_att_u = noise_u.cuda(), input_att_u.cuda()
    cls_criterion.cuda()
    input_label_s = input_label_s.cuda()
    input_label_u = input_label_u.cuda()
    
def sample():
    batch_feature, batch_label, batch_att,batch_unseen_att,batch_unseen_label = data.next_batch(opt.batch_size)
    #batch_feature, batch_label, batch_att,batch_unseen_att,batch_unseen_label,batch_feature_u,batch_feature_u2 = data.next_batch(opt.batch_size)
    input_res.copy_(batch_feature)
    input_att_s.copy_(batch_att)
    input_label_s.copy_(util.map_label(batch_label, data.seenclasses))
    input_att_u.copy_(batch_unseen_att)
    input_label_u.copy_(util.map_label(batch_unseen_label, data.unseenclasses))
    #input_res_u.copy_(batch_feature_u)
    #input_res_u2.copy_(batch_feature_u2)

def generate_syn_feature(netG, seenclasses, unseenclasses, attribute, num):
    Sclass = seenclasses.size(0)
    Uclass = unseenclasses.size(0)
    syn_featureS = torch.FloatTensor(Sclass*num, opt.resSize)
    syn_labelS = torch.LongTensor(Sclass*num)
    syn_featureU = torch.FloatTensor(Uclass*num, opt.resSize)
    syn_labelU = torch.LongTensor(Uclass*num)
    syn_attS = torch.FloatTensor(num, opt.attSize)
    syn_attU = torch.FloatTensor(num, opt.attSize)
    syn_noise = torch.FloatTensor(num, opt.nz)
    if opt.cuda:
        syn_attS = syn_attS.cuda()
        syn_attU = syn_attU.cuda()
        syn_noise = syn_noise.cuda()
    with torch.no_grad():
        for i in range(Sclass):
            iclass = seenclasses[i]
            iclass_att = attribute[iclass]
            syn_attS.copy_(iclass_att.repeat(num, 1))
            syn_noise.normal_(0, 1)
            output,_ = netG(syn_noise,syn_attS,syn_attS)
            #output = output
            #print(output.size())
            syn_featureS.narrow(0, i*num, num).copy_(output.data.cpu())
            syn_labelS.narrow(0, i*num, num).fill_(iclass)
            #syn_feature=output.data.cpu()
            #syn_label.fill_(iclass)
        for i in range(Uclass):
            iclass = unseenclasses[i]
            iclass_att = attribute[iclass]
            syn_attU.copy_(iclass_att.repeat(num, 1))
            syn_noise.normal_(0, 1)
            output,_ = netG(syn_noise,syn_attU, syn_attU)
            #output = output
            #print(output.size())
            syn_featureU.narrow(0, i*num, num).copy_(output.data.cpu())
            syn_labelU.narrow(0, i*num, num).fill_(iclass)
    return syn_featureS, syn_labelS, syn_featureU, syn_labelU

# setup optimizer
optimizerD = optim.Adam(mapping.parameters(), lr=opt.lr,betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr,betas=(opt.beta1, 0.999))
optimizerR = optim.Adam(Reg.parameters(), lr=opt.lr,betas=(opt.beta1, 0.999))
optimizerDis = optim.Adam(Dis.parameters(), lr=opt.lr,betas=(opt.beta1, 0.999))
optimizerDisS = optim.Adam(DisS.parameters(), lr=opt.lr,betas=(opt.beta1, 0.999))
optimizerDisU = optim.Adam(DisU.parameters(), lr=opt.lr,betas=(opt.beta1, 0.999))
optimizer_center=optim.Adam(center_criterion.parameters(), lr=opt.lr,betas=(opt.beta1, 0.999))
#optimizerCls = optim.Adam(cls.parameters(), lr=opt.lr,betas=(opt.beta1, 0.999))


def compute_per_class_acc_gzsl( test_label, predicted_label, target_classes):
    acc_per_class = 0
    for i in target_classes:
        idx = (test_label == i)
        if torch.sum(idx)==0:
            acc_per_class +=0
        else:
            acc_per_class += float(torch.sum(test_label[idx] == predicted_label[idx])) / float(torch.sum(idx))
    acc_per_class /= float(target_classes.size(0))
    return acc_per_class

def calc_gradient_penalty(netD, real_data, fake_data):
    #print real_data.size()
    alpha = torch.rand(opt.batch_size, 1)
    alpha = alpha.expand(real_data.size())
    if opt.cuda:
        alpha = alpha.cuda()
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    if opt.cuda:
        interpolates = interpolates.cuda()

    interpolates = Variable(interpolates, requires_grad=True)
    disc_interpolates,_ ,_= netD(interpolates)
    ones = torch.ones(disc_interpolates.size())
    if opt.cuda:
        ones = ones.cuda()
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda1
    return gradient_penalty

def MI_loss(mus, sigmas, i_c, alpha=1e-8):
    kl_divergence = (0.5 * torch.sum((mus ** 2) + (sigmas ** 2)
                                  - torch.log((sigmas ** 2) + alpha) - 1, dim=1))

    MI_loss = (torch.mean(kl_divergence) - i_c)

    return MI_loss

def optimize_beta(beta, MI_loss,alpha2=1e-6):
    beta_new = max(0, beta + (alpha2 * MI_loss))

    # return the updated beta value:
    return beta_new

# train a classifier on seen classes, obtain \theta of Equation (4)
pretrain_cls = classifier.CLASSIFIER(data.train_feature, map_label(data.train_label, data.seenclasses),
                                     data.seenclasses.size(0), opt.resSize, opt.cuda, 0.001, 0.5, 50, 100)
#cls = CLASSIFIER.(opt.latenSize, train_X, train_Y, data, nclass, opt.cuda,
                                           #opt.classifier_lr, 0.5, 25, opt.syn_num, True)

for p in pretrain_cls.model.parameters(): # set requires_grad to False
    p.requires_grad = False
    
#G_losses = []
#D_losses = []
d = 0
for epoch in range(2000):
    FP = 0
    mean_lossD = 0
    mean_lossG = 0
    for i in range(0, data.ntrain, opt.batch_size):
       #if i % 2 == 0:
         #   d = 0
        #else:
         #   d = 1

        #for p in mapping.parameters(): # reset requires_grad
           # p.requires_grad = True # they are set to False below in netG update

        for iter_d in range(1):
            sample()
            #print(input_label)
            #y_onehot = torch.zeros((opt.batch_size, opt.nclass_all)).cuda()
            #y_onehot.scatter_(1, input_label.unsqueeze(1), 1)
            #y_onehot.requires_grad_(False)
            input_resv = Variable(input_res)
            input_attv = Variable(input_att_s)
            #input_resv_u = Variable(input_res_u)
            input_attv_u = Variable(input_att_u)
            
            noise_s.normal_(0, 1)
            noisev = Variable(noise_s)
            fakeS, fakeU = netG(noisev, input_attv, input_attv_u)
            
            Reg.zero_grad()
            #Dis.zero_grad()
            DisS.zero_grad()
           
            
            
            R_att_S, R_att_U = Reg(fakeS.detach(), fakeU.detach())
            #dis_fake_S, dis_fake_U = Reg(fakeS.detach(), fakeU.detach())
            #dis_loss = (mse(R_att_S, input_att_s) + mse(R_att_U, input_att_u))#/2
            
            #dis_fake_S, dis_fake_U = Dis(R_att_S, R_att_U)            
            #dis_real_S, dis_real_U = Dis(input_attv, input_attv_u)
            Reg_loss = (mse(R_att_S, input_attv) + mse(R_att_U, input_attv_u))#/2
        
            dis_fake_S = DisS(R_att_S)   
            dis_real_S = DisS(input_attv)
            dis_S = (mse(dis_real_S, torch.ones_like(dis_real_S))
            + mse(dis_fake_S, torch.zeros_like(dis_fake_S)))
            com_loss = dis_S + Reg_loss
            com_loss.backward()
            optimizerR.step()
            #optimizerDis.step()
            optimizerDisS.step()
            
            DisU.zero_grad()
            Reg.zero_grad()
            R_att_S, R_att_U = Reg(fakeS.detach(), fakeU.detach())
            Reg_loss = (mse(R_att_S, input_attv) + mse(R_att_U, input_attv_u))#/2
        
            dis_fake_U = DisU(R_att_U)
            dis_real_U = DisU(input_attv_u)
            dis_U = (mse(dis_real_U, torch.ones_like(dis_real_U))            
            + mse(dis_fake_U, torch.zeros_like(dis_fake_U)))#/4
            com_loss = dis_U + Reg_loss  
           ##LOSS
            #valid = 
                
            #dis_loss = (mse(dis_real_S, torch.ones_like(dis_real_S))
            #+ mse(dis_fake_S, torch.zeros_like(dis_fake_S))
            #+ mse(dis_real_U, torch.ones_like(dis_real_U))            
            #+ mse(dis_fake_U, torch.zeros_like(dis_fake_U)))#/4
            
            #dis_loss = (nn.BCELoss()(dis_real_S, torch.ones_like(dis_real_S))
            #+ nn.BCELoss()(dis_real_U, torch.ones_like(dis_real_U))
            #+ nn.BCELoss()(dis_fake_S, torch.zeros_like(dis_fake_S))
            #+ nn.BCELoss()(dis_fake_U, torch.zeros_like(dis_fake_U)))#/4
            
            #com_loss = dis_loss + Reg_loss            
            com_loss.backward()
            
            optimizerR.step()
            #optimizerDis.step()
            #optimizerDisS.step()
            optimizerDisU.step()
                       
            #mapping.zero_grad()

            #criticD_real,_,latens = mapping(input_resv)
            #criticD_real = criticD_real.mean()
            #criticD_fake,_,_ = mapping(fakeS.detach())
            #criticD_fake = criticD_fake.mean()
            #f = fakeS.detach()            
            ## gradient penalty
            
            #gradient_penalty = calc_gradient_penalty(mapping, input_resv, f.data)
            
            ##gradient_penalty_u = calc_gradient_penalty(mapping, input_resv_u, fake_u.data)
            ##mi_loss=MI_loss(torch.cat((muR, muF), dim=0),torch.cat((varR, varF), dim=0), opt.i_c)

            #center_loss = center_criterion(latens, input_label_s, margin=opt.center_margin)
            
            ##center_loss_u=center_criterion(latenu, input_label_u,margin=opt.center_marginu)
            ##Wasserstein_D = criticD_real - criticD_fake + criticD_real_u - criticD_fake_u
            ##D_cost = criticD_fake - criticD_real + 1 * gradient_penalty + 0.001*criticD_real**2 + .01 * (criticD_fake_u - criticD_real_u + gradient_penalty_u + 0.001*criticD_real_u**2)
            ##D_cost = criticD_fake - criticD_real + 1 * gradient_penalty + 0.001*criticD_real**2+center_loss*opt.center_weight \
            ##+ .001 * (criticD_fake_u - criticD_real_u + gradient_penalty_u + 0.001*criticD_real_u**2)#\+.1*latents_loss + .01*latentu_loss
            ##center_loss_u*0.1 + 
            ##adv_loss = (nn.BCELoss()(criticD_real, torch.ones_like(criticD_real))
             ##+ nn.BCELoss()(criticD_fake, torch.zeros_like(criticD_fake)))/2
             
            #adv_loss = Variable((mse(criticD_real, torch.ones_like(criticD_real)) + mse(criticD_fake, torch.zeros_like(criticD_fake))),requires_grad=True)#/2 #+ center_loss*opt.center_weight #+ 0.001*criticD_real**2
            
            #adv_loss = criticD_fake - criticD_real + 1 * gradient_penalty# \
            #+ 0.001*criticD_real**2 + center_loss*opt.center_weight #+ 0.001*criticD_real**2
            
            ##D_cost = mse() + center_loss*opt.center_weight
            #adv_loss = Variable(1*adv_loss, requires_grad = True)
            #adv_loss.backward()
            #optimizerD.step()

            #optimizer_center.step()

        ############################
        # (2) Update G network: optimize WGAN-GP objective, Equation (2)
        ###########################
        for p in mapping.parameters(): # reset requires_grad
            p.requires_grad = False # avoid computation

        netG.zero_grad()
       
        input_attv = Variable(input_att_s)
        #print(input_attv)
        input_attv_u = Variable(input_att_u)
        
        noise_s.normal_(0, 1)
        noisev = Variable(noise_s)
        fakeS, fakeU = netG(noisev, input_attv, input_attv_u)
        
        diff_loss = mse(fakeU, fakeS)
        #print(diff_loss)
        
        
        with torch.no_grad():
            R_att_S, R_att_U = Reg(fakeS.detach(), fakeU.detach())
            #dis_fake_S, dis_fake_U = Dis(R_att_S, R_att_U)
            dis_fake_S = DisS(R_att_S)
            dis_fake_U = DisU(R_att_U)
        
        criticG_fakeS,_,_= mapping(fakeS)
        #criticG_fakeS = criticG_fakeS.mean()
        #criticG_fakeU = criticG_fakeU.mean()
        #G_cost = Variable(xs_h.data, requires_grad=True)
        G_cost = mse(dis_fake_S, torch.ones_like(dis_fake_S)) \
        + mse(dis_fake_U, torch.ones_like(dis_fake_U))#\
        #+ .1*mse(criticG_fakeS, torch.ones_like(criticG_fakeS))
        #)#/3
        
        #G_cost = 1*(- criticG_fakeS) + (#mse(criticG_fakeS, torch.ones_like(criticG_fakeS))
        #mse(dis_fake_S, torch.ones_like(dis_fake_S))
        #+ mse(dis_fake_U, torch.ones_like(dis_fake_U)))#/2
                
        c_errG_fake = cls_criterion(pretrain_cls.model(fakeS), input_label_s)
        c_errG_fakeu = cls_criterion(pretrain_cls.model(fakeU), input_label_u)
                
        errG = G_cost + .5*(c_errG_fake) + .001*(c_errG_fakeu)# + 0.01*diff_loss
                                                                        #+ 0.001 * vatloss_src + 0.001 * centloss_s + 0.001 * vatloss_trg +loss_cls_u) #+center_loss_f
        #errG = G_cost + .5*(c_errG_fake) + .01 * loss_cls_s + .001 * (G_cost_u + 0.001 * vatloss_src + 0.001 * centloss_s + 0.001 * vatloss_trg +loss_cls_u) #+center_loss_f
        #errG = G_cost + .8*(c_errG_fake) + .01 * loss_cls_s + 0.001 * (0.001 * vatloss_src + 0.001 * centloss_s )+ .001 * (G_cost_u + 0.001 * vatloss_trg +loss_cls_u) #+center_loss_f
        
        errG.backward()
        optimizerG.step()
     # Save Losses for plotting later
        #G_losses.append(errG.item())
        #D_losses.append(D_cost.item())

    if opt.lr_dec:
        if (epoch + 1) % opt.lr_dec_ep == 0:
            for param_group in optimizerD.param_groups:
                param_group['lr'] = param_group['lr'] * opt.lr_dec_rate
            for param_group in optimizerG.param_groups:
                param_group['lr'] = param_group['lr'] * opt.lr_dec_rate
            for param_group in optimizerDis.param_groups:
                param_group['lr'] = param_group['lr'] * opt.lr_dec_rate
            for param_group in optimizerR.param_groups:
                param_group['lr'] = param_group['lr'] * opt.lr_dec_rate
            for param_group in optimizer_center.param_groups:
                param_group['lr'] = param_group['lr'] * opt.lr_dec_rate
            #for param_group in optimizerCls.param_groups:
                #param_group['lr'] = param_group['lr'] * opt.lr_dec_rate

    mean_lossG /=  data.ntrain / opt.batch_size
    mean_lossD /=  data.ntrain / opt.batch_size
    #print('[%d/%d] Loss_D: %.4f Loss_G: %.4f,Loss_Gu: %.4f,Loss_errG: %.4f,Loss_cls_s: %.4f,Loss_cls_u: %.4f, Wasserstein_dist: %.4f, c_errG_fake:%.4f'
           #   % (epoch, opt.nepoch, D_cost.item(), G_cost.item(),G_cost_u.item(),errG.item(),loss_cls_s.item(),loss_cls_u.item(),Wasserstein_D.item(),c_errG_fake.item()))
    print('[%d/%d] com_loss: %.3f, reg: %.3f,disS: %.3f,disU: %.3f,map: %.3f, ergG: %.3f, G_cost: %.3f, c_errG_fake:%.3f,c_errG_fakeu:%.3f'
              % (epoch, opt.nepoch,com_loss.item(), Reg_loss.item(), dis_S.item(),dis_U.item(),adv_loss.item(), errG.item(),G_cost.item(),c_errG_fake.item(),c_errG_fakeu.item()))

    # evaluate the model, set G to evaluation mode  ergG2: %.3f,,errG2.item()
    netG.eval()
    mapping.eval()
    Reg.eval()
    Dis.eval()
    DisS.eval()
    DisU.eval()
    #cls.eval()

    # Generalized zero-shot learning
    # Generalized zero-shot learning
    syn_featureS, syn_labelS, syn_featureU, syn_labelU  = generate_syn_feature(netG, data.seenclasses, data.unseenclasses, data.attribute, opt.syn_num)
    #syn_featureS, syn_labelS = generate_syn_feature(netG, data.seenclasses, data.attribute, opt.syn_num)
    train_X = torch.cat((data.train_feature, syn_featureU), 0)
    #train_X = torch.cat((syn_featureS, syn_featureU), 0)
    #train_Y = torch.cat((syn_labelS, syn_labelU), 0)
    train_Y = torch.cat((data.train_label, syn_labelU), 0)
    
    if opt.final_classifier == 'softmax':
        nclass = opt.nclass_all
        cls1 = CLASSIFIER(mapping, opt.resSize, train_X, train_Y, data, nclass, opt.cuda,
                                           opt.classifier_lr, 0.5, 25, opt.syn_num, True)
        print('--------------------------------------------------------------')
        print('unseen=%.4f, seen=%.4f, h=%.4f' % (cls1.acc_unseen, cls1.acc_seen, cls1.H))

    elif opt.final_classifier == 'knn':
        if epoch % 25 == 0:  ## training a knn classifier takes too much time
            clf = KNeighborsClassifier(n_neighbors=opt.k)
            train_z, _, _, _, _ = mapping(train_X.cuda())
            clf.fit(X=train_z.cpu(), y=train_Y)

            test_z_seen, _, _, _, _ = mapping(data.test_seen_feature.cuda())
            pred_Y_s = torch.from_numpy(clf.predict(test_z_seen.cpu()))
            test_z_unseen, _, _, _, _ = mapping(data.test_unseen_feature.cuda())
            pred_Y_u = torch.from_numpy(clf.predict(test_z_unseen.cpu()))
            acc_seen = compute_per_class_acc_gzsl(pred_Y_s, data.test_seen_label, data.seenclasses)
            acc_unseen = compute_per_class_acc_gzsl(pred_Y_u, data.test_unseen_label, data.unseenclasses)
            H = 2 * acc_seen * acc_unseen / (acc_seen + acc_unseen)
            print('unseen=%.4f, seen=%.4f, h=%.4f' % (acc_unseen, acc_seen, H))
    else:
        raise ValueError('Classifier %s is not supported' % (opt.final_classifier))
        
    netG.train()
    mapping.train()
    Reg.train()
    #Dis.train()
    DisS.train()
    DisU.train()
    #cls.train()
    
#plt.figure(figsize=(10,5))
#plt.title("Generator and Discriminator Loss During Training")
#plt.plot(G_losses,label="G")
#plt.plot(D_losses,label="D")
#plt.xlabel("iterations")
#plt.ylabel("Loss")
#plt.legend()
#plt.show()


