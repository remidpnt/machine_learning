# coding: utf-8
import torch
import torch.nn.functional as F
import warnings
import numpy as np
import tqdm
import pandas as pd
import cProfile
import time
from torch import nn
from torch.autograd import Variable
from AWD_LSTM_model import *

profile     = cProfile.Profile()
profile.enable()
seed        = 3636
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
def get_specific_parameter_set(string_in_name):
    for name, param in model.named_parameters():
        if string_in_name in name:
            yield param

###################   Parameters   ################
PATH_model          = r'F:\data\text_classification\model_sav\model_imdb_all_unfrezed_dimanche_fin_aprem.pth'
PATH_save           = r'F:\data\text_classification\model_sav\model_imdb_all_unfrezed_dimanche_fin_aprem_plus15.pth'

batch_size          = 30
learning_rate       = 0.001
weight_decay_L2     = 0
# drops               = np.array([0.25, 0.1, 0.2, 0.02, 0.15]) * 1
drops               = np.array([0.1, 0.1, 0.2, 0.02, 0.15]) * 1
model               = get_language_model(n_tok=60002, emb_sz=400, nhid=1150, nlayers=3, pad_token=0, decode_train=True, dropouts=drops).cuda()
optimizer           = torch.optim.Adam([{'params': get_specific_parameter_set("rnns.2"), 'lr': learning_rate/2.6**0},
                                        {'params': get_specific_parameter_set("rnns.1"), 'lr': learning_rate/2.6**1},
                                        {'params': get_specific_parameter_set("rnns.0"), 'lr': learning_rate/2.6**2},
                                        {'params': get_specific_parameter_set("coder"), 'lr': learning_rate/2.6**0}], weight_decay=weight_decay_L2, betas=(0.7, 0.999))

##################### CODE ####################

data_train = LanguageModelLoader(np.load(r"F:\data\aclImdb1\pretrain\array_imdb_train.npy")[:-100000], batch_size, 70)
data_val   = LanguageModelLoader(np.load(r"F:\data\aclImdb1\pretrain\array_imdb_train.npy")[-100000:],  batch_size, 70)
criterion           = nn.CrossEntropyLoss()
learning_rate_index = 0
best_val            = np.inf
embedding_array     = np.load(r'F:\data\text_classification\model_sav\imdb_embedding_60002.npy')

dicto                                           = torch.load(PATH_model, map_location=lambda storage, loc: storage)
# dicto["1.decoder.weight"]                       = torch.FloatTensor(embedding_array).cuda()
# dicto["0.encoder_with_dropout.embed.weight"]    = torch.FloatTensor(embedding_array).cuda()
# dicto['0.encoder.weight']                       = torch.FloatTensor(embedding_array).cuda()
model.load_state_dict(dicto)

T                   = 22036 * 10
cut_fra             = 0.15
cut                 = T * cut_fra
ratio               = 20
lr_ratio            =  [1/2.6**0, 1/2.6**1, 1/2.6**2, 1/2.6**0]
t                   =  0
best_val            =  np.inf
df                  = pd.DataFrame(columns = ['train_loss', 'val_loss', 'val_acc'])

# for name, param in model.named_parameters():
#     if "rnns" in name:
#         param.requires_grad = False
#         print(f'no gradient for {name} ')
#     else:
#         param.requires_grad = True

# for name, param in model.named_parameters():
#     param.requires_grad = True

for ep in range(20):

    # if ep == 2:
    #     for name, param in model.named_parameters():
    #         if "rnns.0.mod" in name:
    #             param.requires_grad = False
    #             print(f'no gradient for {name} ')
    #         elif "rnns.1.mod" in name:
    #             param.requires_grad = False
    #             print(f'no gradient for {name} ')
    #         else:
    #             param.requires_grad = True
    #     t = 0
    #
    # if ep == 3:
    #     for name, param in model.named_parameters():
    #         if "rnns.0.mod" in name:
    #             param.requires_grad = False
    #             print(f'no gradient for {name} ')
    #         else:
    #             param.requires_grad = True
    #     t = 0
    #
    # if ep == 4:
    #     for name, param in model.named_parameters():
    #         param.requires_grad = True
    #     t = 0

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
        for indice, param_group in enumerate(optimizer.param_groups): #dicriminative learning rate
            param_group["lr"] = lr_ratio[indice]*new_lr


    model.eval()
    for a, b in data_val.__iter__():
        prediction       =  model(Variable(a))
        loss             =  criterion(prediction[0], Variable(b))
        val_loss        +=  loss.data[0]
        num_batch_val   += 1
        val_acc         += (prediction[0].max(1)[1]==Variable(b)).sum().data[0] / b.shape[0]
        del loss
        del a, b

    # for a, b in data_test.__iter__():
    #     prediction       =  model(Variable(a))
    #     loss             =  criterion(prediction[0], Variable(b))
    #     test_loss        += loss.data[0]
    #     num_batch_test   += 1
    #     del loss
    #     del a, b

    # df_epoch    = pd.DataFrame([[train_loss/num_batch_train, val_loss/num_batch_val, val_acc/num_batch_val, test_loss/num_batch_test]], columns=['train_loss', 'val_loss', 'val_acc', 'test_loss'])
    df_epoch    = pd.DataFrame([[train_loss/num_batch_train, val_loss/num_batch_val, val_acc/num_batch_val]], columns=['train_loss', 'val_loss', 'val_acc'])
    df          = pd.concat([df, df_epoch], axis=0, ignore_index=True, sort=False)
    print("\n")
    time.sleep(0.3)
    print(df)

    val_loss = val_loss / num_batch_val
    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), PATH_save)
        print(f'\n saved for best_val: {best_val} \n')
        time.sleep(0.3)
    else:
        print("\n")
        time.sleep(0.3)

profile.disable()
profile.print_stats('time')