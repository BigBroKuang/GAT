#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import utils
import math
import os
import gatlayer
import ray
import argparse
from collections import Counter
import csv
import os

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='', help='A .h5 file to be processed')
parser.add_argument('--outfile',  type=str, default='', help='save classification results to the path')
parser.add_argument('--load_model', type=str, default='', help='load pre-trained model')
parser.add_argument('--mode', type=int, default=0, help='0: Attention-NN (use mean of neighbors), 1: NN')
parser.add_argument('--batch_size', type=int, default=5000, help='the features in a tile are divided into batches')

os.mkdir()

def process_h5s(read_h5, batch_size, mode =0, agg='mean'):
    
    data =utils.LoadData(read_h5= read_h5, normalize=True)
    #print(data.r1_feats.shape)
    training = list(range(data.r1_feats.shape[0]))
    batch_size =batch_size
    batches = math.ceil(len(training) / batch_size)
    if mode==0:
        lay1 =220
    else:
        lay1 = 220
    self_hid =128            #dimension of hidden layer of the center well
    nei_hid = 128            #dimension of the neighbors
    nheads = 1               #the number of heads
    ndim=32                  #dimension of 3rd layer
    classes =3
    dropout = 0.2
    model = gatlayer.model(in_feats = lay1,#data.r1_feats.shape[1],
                        self_hid = self_hid, 
                        nei_hid = nei_hid, 
                        nheads = nheads,
                        classes = classes, 
                        dropout = dropout)    

    model.load_state_dict(torch.load(args.load_model))
    model.eval()
    pred_tags =[]
    for index in range(batches):
        feat_batch = training[index*batch_size:(index+1)*batch_size]
        if mode==0:
            r1_batch,nei_batch = data.GenerateNeighbors(feat_batch, agg_func = agg, neighbor= True)
            #nei_batch = r1_batch[:,220:]
            #r1_batch = r1_batch[:,:220]
            output = model(r1_batch,nei_batch)
        else:
            r1_batch = data.GenerateNeighbors(feat_batch, agg_func = agg, neighbor= False)
            output = model(r1_batch)
        y_pred_softmax = torch.log_softmax(output, dim = 1)
        _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)
        pred_tags += y_pred_tags.tolist()

    savefile = '_'.join(read_h5.split('.')[:-1])
    savefile = savefile.split('/')[-1]

    results =[['0: empty','1: monoclonal', '2: polyclonal']]
    results.append(list(Counter(pred_tags).items()))
    results.append(pred_tags)
    
    out = csv.writer(open(savefile+'.csv',"w"), delimiter=',')
    out.writerows(results)
    #return 
        
args = parser.parse_args()

h5_input = args.input      # '../data/h5_train' 
# dir_list = os.listdir(h5_input)
# h5_files = [f'{h5_input}/{x}' for x in dir_list if '.h5' in x]
process_h5s(h5_input, args.batch_size, args.mode)



# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# import argparse
# import utils
# import ray
# import torch
# import math
# import os
# import gatlayer
# from collections import Counter
# import csv

# parser = argparse.ArgumentParser()
# parser.add_argument('--input', type=str, default='', help='directory of the files to be classified')
# parser.add_argument('--savepath',  type=str, default='', help='save classification results to the path')
# parser.add_argument('--load_model', type=str, default='', help='load pre-trained model')
# parser.add_argument('--mode', type=int, default=0, help='0: Attention-NN (use mean of neighbors), 1: NN')
# parser.add_argument('--num_cpus', type=int, default=5, help='the number of CPUs')
# parser.add_argument('--batch_size', type=int, default=5000, help='the features in a tile are divided into batches')

# os.makedirs("results/", exist_ok=True)

# @ray.remote(num_cpus=1)#
# def process_h5s(read_h5, batch_size, savepath, mode =0, agg='mean'):
    
#     data =utils.LoadData(read_h5= read_h5, normalize=True)
#     #print(data.r1_feats.shape)
#     training = list(range(data.r1_feats.shape[0]))
#     batch_size =batch_size
#     batches = math.ceil(len(training) / batch_size)
#     if mode==0:
#         lay1 =220
#     else:
#         lay1 = 220
#     self_hid =128            #dimension of hidden layer of the center well
#     nei_hid = 128            #dimension of the neighbors
#     nheads = 1               #the number of heads
#     ndim=32                  #dimension of 3rd layer
#     classes =3
#     dropout = 0.2
#     model = gatlayer.model(in_feats = lay1,#data.r1_feats.shape[1],
#                         self_hid = self_hid, 
#                         nei_hid = nei_hid, 
#                         nheads = nheads,
#                         classes = classes, 
#                         dropout = dropout)    

#     model.load_state_dict(torch.load(args.load_model))
#     model.eval()
#     pred_tags =[]
#     for index in range(batches):
#         feat_batch = training[index*batch_size:(index+1)*batch_size]
#         if mode==0:
#             r1_batch,nei_batch = data.GenerateNeighbors(feat_batch, agg_func = agg, neighbor= True)
#             #nei_batch = r1_batch[:,220:]
#             #r1_batch = r1_batch[:,:220]
#             output = model(r1_batch,nei_batch)
#         else:
#             r1_batch = data.GenerateNeighbors(feat_batch, agg_func = agg, neighbor= False)
#             output = model(r1_batch)
#         y_pred_softmax = torch.log_softmax(output, dim = 1)
#         _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)
#         pred_tags += y_pred_tags.tolist()

#     savefile = '_'.join(read_h5.split('.')[:-1])
#     savefile = savefile.split('/')[-1]

#     results =[['0: empty','1: monoclonal', '2: polyclonal']]
#     results.append(list(Counter(pred_tags).items()))
#     results.append(pred_tags)
    
#     out = csv.writer(open(savepath+savefile+'.csv',"w"), delimiter=',')
#     out.writerows(results)
#     #return 
        
# args = parser.parse_args()

# h5_input = args.input      # '../data/h5_train' 
# dir_list = os.listdir(h5_input)
# h5_files = [f'{h5_input}/{x}' for x in dir_list if '.h5' in x]

# ray.shutdown()
# ray.init(ignore_reinit_error=False, num_cpus=args.num_cpus) #
# all_data = ray.get([process_h5s.remote(h5, args.batch_size, 'results/', args.mode) for h5 in h5_files])
# ray.shutdown()







