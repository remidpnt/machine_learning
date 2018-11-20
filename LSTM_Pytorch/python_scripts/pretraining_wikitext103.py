# coding: utf-8
import torch
import torch.nn.functional as F
import warnings
import numpy as np
import tqdm
import pandas as pd
import time
from torch import nn
from torch.autograd import Variable
from AWD_LSTM_model import *

seed = 3636
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

###################   Parameters   ################
PATH_save           = r'F:\data\text_classification\model_sav\model_wiki103_10-3.pth'
learning_rate       = 0.001
bs                  = 64
weight_decay_L2     = 0
drops               = np.array([0.25, 0.1, 0.2, 0.02, 0.15]) * 0.7
model               = get_language_model(n_tok=30002, emb_sz=400, nhid=1150, nlayers=3, pad_token=0, decode_train=True, dropouts=drops).cuda()
optimizer           = torch.optim.Adam(model.parameters(), learning_rate, weight_decay=weight_decay_L2, betas=(0.7, 0.999))
criterion           = nn.CrossEntropyLoss()
learning_rate_index = 0
t                   = 0
best_val            = np.inf
df                  = pd.DataFrame(columns = ['train_loss', 'val_loss', 'val_acc', 'test_loss'])

T                   = 22036 * 10
cut_fra             = 0.15
cut                 = T * cut_fra
ratio               = 20

##################### CODE ####################

data_train = LanguageModelLoader(np.load(r"F:\data\text_classification\preprocessed_wiki\array_wikitext_train.npy"), bs, 70)
data_val   = LanguageModelLoader(np.load(r"F:\data\text_classification\preprocessed_wiki\array_wikitext_val.npy"),   bs, 70)
data_test  = LanguageModelLoader(np.load(r"F:\data\text_classification\preprocessed_wiki\array_wikitext_test.npy"),  bs, 70)
# model.load_state_dict(torch.load(PATH_model))
# # model = model.load('F:\data\mytrainingALL.pth')

for ep in range(20):
    train_loss      = 0
    val_loss        = 0
    test_loss       = 0
    val_acc         = 0
    num_batch_train = 0
    num_batch_val   = 0
    num_batch_test  = 0

    model.train()
    for a, b in data_train.__iter__():
        t += 1
        prediction = model(Variable(a))
        loss = criterion(prediction[0], Variable(b))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        num_batch_train += 1
        train_loss      += loss.data[0]
        del loss
        del a, b

        p = t / cut if t < cut else 1 - ((t - cut) / (cut * ((1 / cut_fra) - 1)))
        new_lr = learning_rate * max(p, 0.05)
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr

    model.eval()
    for a, b in data_val.__iter__():
        prediction       =  model(Variable(a))
        loss             =  criterion(prediction[0], Variable(b))
        val_loss        +=  loss.data[0]
        num_batch_val   += 1
        val_acc         += (prediction[0].max(1)[1]==Variable(b)).sum().data[0] / b.shape[0]
        del loss
        del a, b

    for a, b in data_test.__iter__():
        prediction       =  model(Variable(a))
        loss             =  criterion(prediction[0], Variable(b))
        test_loss        += loss.data[0]
        num_batch_test   += 1
        del loss
        del a, b

    df_epoch    = pd.DataFrame([[train_loss/num_batch_train, val_loss/num_batch_val, val_acc/num_batch_val, test_loss/num_batch_test]], columns=['train_loss', 'val_loss', 'val_acc', 'test_loss'])
    df          = pd.concat([df, df_epoch], axis=0, ignore_index=True, sort=False)
    print("\n")
    time.sleep(0.3)
    print(df)

    val_loss = val_loss / num_batch_val
    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), PATH_save)
        print(f'\n saved for best_val: {best_val}, learning_rate:{new_lr} \n')
        time.sleep(0.3)
    else:
        print("\n")
        time.sleep(0.3)