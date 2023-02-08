import torch 
import torch.nn as nn
import torch.nn.functional as F


class neigh_atten(nn.Module):
    def __init__(self,in_feats, out_feats, dropout=0.3):
        super(neigh_atten, self).__init__()
        self.dropout = dropout
        self.concat = True
        self.out_feats = out_feats
        
        self.W1 = nn.Parameter(torch.empty(size = (in_feats, out_feats)))
        nn.init.xavier_uniform_(self.W1.data, gain=1.414)
        
        self.W2 = nn.Parameter(torch.empty(size = (in_feats, out_feats)))
        nn.init.xavier_uniform_(self.W2.data, gain=1.414)
        
        self.a = nn.Parameter(torch.empty(size = (2*out_feats, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.leaky =nn.LeakyReLU(0.1)#nn.LeakyRelu(0.1)  nn.Sigmoid()# 
        self.relu =nn.ReLU()#nn.ELU
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, self_feat, nei_feat):
        Wh1 = torch.mm(self_feat, self.W1) # (batch, in_feats) * (in_feats, out)-> (batch, out_feats)
        
        Wh2 = torch.mm(nei_feat,self.W2)#(batch, in_feats) * (in_feats, out)-> (batch, out_feats)
        #change this part to re-design the attention mechanism
        e = self.attention_mechanism(Wh1, Wh2) # weight matrix: (batch, N)
        Wh2 =Wh1 - Wh2# torch.mul(e, Wh2)
        #torch.cat([Wh1, Wh2], dim=1)
        return Wh2 
    def attention_mechenism(self,Wh1, Wh2):
	pass
	#this part is deleted intentionally
        
    
class model(nn.Module):
    def __init__(self, in_feats, self_hid=128, nei_hid=8, nheads =8, hid2=32, classes=3, dropout=0.3, agg_func='max'):
        super(model, self).__init__()
        
        self.dropout = dropout
        self.nheads = nheads
		
        self.neigh_atten =[neigh_atten(in_feats, nei_hid, dropout=dropout) for _ in range(self.nheads)]
        for i in range(len(self.neigh_atten)):
            self.add_module('neigh_atten'+str(i), self.neigh_atten[i])      
            
        gain1 = 1.414
        gain2 = 1
        self.W0 = nn.Parameter(torch.empty(size=(in_feats, self_hid)))
        nn.init.xavier_uniform_(self.W0.data, gain = gain2)
        
        self.W1 = nn.Parameter(torch.empty(size=(self_hid+nei_hid*self.nheads, hid2)))
        nn.init.xavier_uniform_(self.W1.data, gain = gain1)

        self.leaky = nn.LeakyReLU(0.1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        self.W2 = nn.Parameter(torch.empty(size=(hid2, classes))) #int((self_hid+nei_hid)/2)
        nn.init.xavier_uniform_(self.W2.data, gain = gain2)
        
    def forward(self, train_feats, nei_feats):
        '''
			train_feats: (batch, feats)
			nei_feats: (batch, feats)
        '''
        self_att = torch.mm(train_feats, self.W0)
		        
        x = torch.cat([att(train_feats, nei_feats) for att in self.neigh_atten], dim=1)
        x = torch.cat([self_att, x], dim =1)
        x = self.relu(x)
        
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.mm(x, self.W1)#[nbatch, nhidden] 
        x = self.sigmoid(x)
        
        #x = torch.sigmoid(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.mm(x, self.W2)
        #x = F.elu(x)
        return x







