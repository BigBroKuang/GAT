import torch 
import numpy as np
import h5py
from scipy.spatial import KDTree

Tile = {7:'007', 13:'013', 19:'019', 25:'025', 31:'031'}
Lane = {1:'1', 2:'2', 3:'3', 4:'4'}

def load_h5(filepath=None,normalize=True):
    
    print('reading '+filepath)
    file = h5py.File(filepath, 'r')
    df_coord = np.array(file['seqdata']['locations'])#pd.DataFrame(np.array(file['seqdata']['locations']),columns=['x_coord','y_coord'])
    df_feats = np.array(file['seqdata']['raws'])
    
    if normalize:
        #XTmat = np.array(file['metadata']['crosstalk_mat'])
        #print(XTmat.shape)
        XTmat = np.array([[ 1.04654975, -0.0840663 ,  0.02533226, -0.14064648],
            [-0.57732318,  1.04713325, -0.04130868,  0.02028682],
            [ 0.01037201, -0.02026268,  1.09569202, -0.31071656],
            [-0.00245064,  0.00192372, -0.33485673,  1.09514074]])

        df_feats = np.dot(df_feats, XTmat.T)
    # ## dye normalization
        scale = np.percentile(df_feats[:, :10], 75, axis =(0,1))
        df_feats = df_feats/scale

    return df_coord, df_feats


class LoadData():
    def __init__(self,read_h5='', rseed=1,normalize=False, well_pitch_pixels=2.256):
        
        self.r1_coord, r1_feats = load_h5(read_h5, normalize=normalize)
        self.rseed =rseed
        #read r1_h5 data
        #self.Labels = torch.LongTensor(Labels.T)
        self.r1_feats = torch.FloatTensor(r1_feats)
        
        self.coord2index = {(ele[0],ele[1]):idx for idx,ele in enumerate(self.r1_coord)}

        self.tree = KDTree(self.r1_coord, compact_nodes=False, balanced_tree=False, copy_data=False)
        self.neighbor_bound = well_pitch_pixels * 1.4 * 100  # well_pitch_pixels varies for F2 or F3 flowcells

        self.monoclonals = []#save the indices of monoclonals
        self.polyclonals = []
        self.emptywells = []
        
                        
    def TrainSize(self, train_size=1000, val_size =300, test_size=4000):
        assert (train_size + val_size + test_size) <=30000
        
        if val_size > 0:
            self.val_indices = torch.LongTensor(range(train_size, train_size+val_size))
        
        self.train_indices = torch.LongTensor(range(0, train_size))
        self.test_indices = torch.LongTensor(range(train_size+val_size, train_size+val_size+test_size))
        
    def GenerateNeighbors(self, batch, agg_func ='mean', neighbor= True):
        #batch: list (order of wells)
        
        #batch = batch.tolist()
        distance, adj = self.tree.query(self.r1_coord[batch,:], k=7, distance_upper_bound=self.neighbor_bound)
        index_mask = np.logical_and(distance != np.inf, distance > 1)
        
        adj_neighs = adj[index_mask]
        all_wells = list(set(adj_neighs)-set(batch))
        all_wells = batch + all_wells
        idx_map = list(range(len(all_wells)))
        unique_dic = dict(list(zip(all_wells, idx_map)))   
        col_indices = list(map(unique_dic.get, adj_neighs))# [unique_dic[i] for i in adj_neighs]
        
        batch_order = [unique_dic[e] for e in batch]
        adj_batch = torch.zeros(len(batch), len(all_wells))    
        row_col = np.nonzero(index_mask)
        adj_batch[row_col[0], col_indices] = 1 
          
        r1_feat = self.r1_feats[all_wells,:,:]
        batch_feats = r1_feat[batch_order,:,:]
        
        if not neighbor:
            batch_feats = torch.sort(batch_feats, dim=2).Value
            return batch_feats.reshape(batch_feats.shape[0], -1)
        
        if agg_func=='mean':
            adj_batch = adj_batch/torch.sum(adj_batch, dim=1, keepdim=True)    
            
        nei_feats = torch.einsum('ij,jkl->ikl',adj_batch, r1_feat) #(batch, N) * (N, 55, 4) -> (batch, 55, 4)

        argsort = torch.argsort(batch_feats, dim =2)
        batch_feats = torch.gather(batch_feats, dim=2, index =argsort)
        nei_feats = torch.gather(nei_feats, dim=2, index =argsort)
        batch_feats = torch.reshape(batch_feats,(batch_feats.shape[0],-1))
        nei_feats = torch.reshape(nei_feats, (nei_feats.shape[0],-1))
        
        return batch_feats,nei_feats#torch.cat([batch_feats, nei_feats], dim =1)

def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    

    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    
    acc = torch.round(acc * 100, decimals=3)

    return acc


