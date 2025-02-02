#%%
import os, logging
import datetime
import torch as t
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from emetrics import get_cindex
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): validation loss가 개선된 후 기다리는 기간
                            Default: 7
            verbose (bool): True일 경우 각 validation loss의 개선 사항 메세지 출력
                            Default: False
            delta (float): 개선되었다고 인정되는 monitered quantity의 최소 변화
                            Default: 0
            path (str): checkpoint저장 경로
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''validation loss가 감소하면 모델을 저장한다.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        t.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


DRUG_FP_SIZE=2564
PROT_FP_SIZE=9920
class DTA(nn.Module):
    def __init__(self, CHEM_EMBEDDING_SIZE, PROT_EMBEDDING_SIZE,FILTER_LEN1, FILTER_LEN2, NUM_FILTERS = 32):
        super(DTA,self).__init__()
        self.chem_head = nn.ModuleList()
        self.prot_head = nn.ModuleList()
        self.chem_fp = nn.ModuleList()
        self.prot_fp = nn.ModuleList()
        self.body = nn.ModuleList()
        self.pool = nn.AdaptiveMaxPool1d(1)

        self.chem_embedding = nn.Embedding(num_embeddings = CHEM_EMBEDDING_SIZE+1,embedding_dim=128,padding_idx=0)
        self.chem_head.append(nn.Conv1d(in_channels=128,out_channels=NUM_FILTERS,kernel_size=FILTER_LEN1,padding=0,stride=1))
        self.chem_head.append(nn.Conv1d(in_channels=NUM_FILTERS,out_channels=NUM_FILTERS*2,kernel_size=FILTER_LEN1,padding=0,stride=1))
        self.chem_head.append(nn.Conv1d(in_channels=NUM_FILTERS*2,out_channels=NUM_FILTERS*3,kernel_size=FILTER_LEN1,padding=0,stride=1))

        self.prot_embedding = nn.Embedding(num_embeddings = PROT_EMBEDDING_SIZE+1,embedding_dim=128,padding_idx=0)
        self.prot_head.append(nn.Conv1d(in_channels=128,out_channels=NUM_FILTERS,kernel_size=FILTER_LEN2,padding=0,stride=1))
        self.prot_head.append(nn.Conv1d(in_channels=NUM_FILTERS,out_channels=NUM_FILTERS*2,kernel_size=FILTER_LEN2,padding=0,stride=1))
        self.prot_head.append(nn.Conv1d(in_channels=NUM_FILTERS*2,out_channels=NUM_FILTERS*3,kernel_size=FILTER_LEN2,padding=0,stride=1))
        
        self.chem_fp.append(nn.Sequential(nn.Linear(DRUG_FP_SIZE,1024),nn.ReLU(),nn.Dropout(p=0.1)))
        self.chem_fp.append(nn.Sequential(nn.Linear(1024,512),nn.Softmax(dim=1),nn.Dropout(p=0.1)))

        self.prot_fp.append(nn.Sequential(nn.Linear(PROT_FP_SIZE,1024),nn.ReLU(),nn.Dropout(p=0.1)))
        self.prot_fp.append(nn.Sequential(nn.Linear(1024,512),nn.Softmax(dim=1),nn.Dropout(p=0.1)))

        self.body.append(nn.Sequential(nn.Linear(NUM_FILTERS*6+1024,1024),nn.ReLU(),nn.Dropout(p=0.1)))
        self.body.append(nn.Sequential(nn.Linear(1024,512),nn.ReLU(),nn.Dropout(p=0.1)))

        self.predictions = nn.Linear(512, 1)
        nn.init.normal_(self.predictions.weight)
    
    def forward(self,x_chem,x_prot, x_cfp, x_pfp):
        x_chem = self.chem_embedding(x_chem)
        x_chem = x_chem.permute(0,2,1)
        for l in self.chem_head:
            x_chem=t.relu(l(x_chem))
        x_chem = t.squeeze(self.pool(x_chem))
    

        x_prot = self.prot_embedding(x_prot)
        x_prot = x_prot.permute(0,2,1)
        for l in self.prot_head:
            x_prot = t.relu(l(x_prot))
        x_prot = t.squeeze(self.pool(x_prot))

        for l in self.chem_fp:
            x_cfp = l(x_cfp)
        for l in self.prot_fp:
            x_pfp = l(x_pfp)

        x = t.cat([x_chem,x_prot,x_cfp,x_pfp],dim=1)
  
        for l in self.body:
            x = l(x)
        x = self.predictions(x).squeeze()
        return x


def training_loop(FLAGS, p1,p2,p3,
                  train_drugs, train_prots, train_cfp,train_pfp, train_Y, 
                  val_drugs, val_prots, val_cfp,val_pfp,val_Y, 
                  test_drugs, test_prots, test_cfp,test_pfp,test_Y,
                  use_gpu=True,lossfn=nn.MSELoss(), verbose = 1, path = 'checkpoint.pt'
    ):
    model = DTA(FLAGS.charsmiset_size+1,FLAGS.charseqset_size+1,FLAGS.smi_window_lengths[p1],FLAGS.seq_window_lengths[p2], NUM_FILTERS=FLAGS.num_windows[p3])
    optimizer = optim.AdamW(model.parameters(), lr=FLAGS.learning_rate, betas=(0.9,0.999), eps=1e-8, weight_decay=0.01, amsgrad=False)
    earlystopping = EarlyStopping(patience = 15, verbose=verbose>=2, path=path)
    
    
    xchtr=t.from_numpy(np.array(train_drugs)).int()
    xprtr=t.from_numpy(np.array(train_prots)).int()
    xcftr=t.from_numpy(train_cfp).float()
    xpftr=t.from_numpy(train_pfp).float()
    ytr=t.from_numpy(np.array(train_Y)).float()

    xchval = t.from_numpy(np.array(val_drugs)).int()
    xprval = t.from_numpy(np.array(val_prots)).int()
    xcfval=t.from_numpy(val_cfp).float()
    xpfval=t.from_numpy(val_pfp).float()
    yval = t.from_numpy(np.array(val_Y)).float()
    
    xchte =t.from_numpy(np.array(test_drugs)).int()
    xprte=t.from_numpy(np.array(test_prots)).int()
    xcfte=t.from_numpy(test_cfp).float()
    xpfte=t.from_numpy(test_pfp).float()
    yte=t.from_numpy(np.array(test_Y)).float()
 
    dataset=TensorDataset(xchtr,xprtr,xcftr,xpftr,ytr)
    dataloader = DataLoader(dataset, batch_size=FLAGS.batch_size, shuffle=False, pin_memory=False)
    res = pd.DataFrame(columns=['epoch','MSE','train_CI','validation_CI','test_CI'])
    print("%s: training start"%datetime.datetime.now())
    for epoch in range(FLAGS.num_epoch):
        model.train()
        if use_gpu:
            model.cuda()

        mse=[]
        for i, (xch,xpr,xcf,xpf,y) in enumerate(dataloader):
            if use_gpu:
                xch = xch.cuda()
                xpr=xpr.cuda()
                xcf=xcf.cuda()
                xpf=xpf.cuda()
                y=y.cuda()
            optimizer.zero_grad()
            ypred = model(xch,xpr,xcf,xpf)
            err = lossfn(ypred,y)
            err.backward()
            optimizer.step()
            mse.append(err.cpu().detach().numpy())
                
            if verbose>=1 and (i%100==0 or i==len(dataloader)-1):
                print("%s: epoch %d\ttraining batch %d/%d err = %f"%(datetime.datetime.now(),epoch+1, i+1,len(dataloader),err))
        mse_avg = np.mean(mse)
        model.eval()
        with t.no_grad():
            window=5000
            if use_gpu:
                model.cpu()

            trpred = []
            for i in range(0,len(xchtr),window):
                trpred.append(model(xchtr[i:min(len(xchtr),i+window)],xprtr[i:min(len(xchtr),i+window)],xcftr[i:min(len(xchtr),i+window)],xpftr[i:min(len(xchtr),i+window)]))
            trpred = t.concat(trpred)
            Y=ytr.detach().numpy()
            P=trpred.detach().numpy()
            tloss=get_cindex(Y,P)
            if verbose>=2:
                print("%s:tr loss"%datetime.datetime.now())
    
            valpred = []
            for i in range(0,len(xchval),window):
                valpred.append(model(xchval[i:min(len(xchval),i+window)],xprval[i:min(len(xchval),i+window)],xcfval[i:min(len(xchval),i+window)],xpfval[i:min(len(xchval),i+window)]))
            valpred = t.concat(valpred)
            vloss=get_cindex(yval.detach().numpy(),valpred.detach().numpy())
            if verbose>=2:    
                print("%s: val loss"%datetime.datetime.now())
            
            tepred = []
            for i in range(0,len(xchte),window):
                tepred.append(model(xchte[i:min(len(xchte),i+window)],xprte[i:min(len(xchte),i+window)],xcfte[i:min(len(xchte),i+window)],xpfte[i:min(len(xchte),i+window)]))
            tepred = t.concat(tepred)
            teloss=get_cindex(yte.detach().numpy(),tepred.detach().numpy())
            if verbose>=2:
                print("%s: te loss"%datetime.datetime.now())
            
            res.loc[len(res)] = [epoch, mse_avg,tloss,vloss,teloss]
            text='%s: epoch:%d\tMSE=%f\ttraining loss:%f\tvalloss: %f'%(datetime.datetime.now(),epoch+1,mse_avg,tloss,vloss)
            print(text)

            earlystopping(-1*vloss,model)
            if earlystopping.early_stop:
                print("epoch:%d early stop triggered"%(epoch+1))
                model.load_state_dict(t.load(path))
                break
    return model, res

def training_loop_tf(FLAGS, model,
                  train_drugs, train_prots, train_cfp,train_pfp, train_Y, 
                 
                  test_drugs, test_prots, test_cfp,test_pfp,test_Y,
                  use_gpu=True,lossfn=nn.MSELoss(), verbose = 1, path='tf.pt'
    ):
    optimizer = optim.AdamW(model.parameters(), lr=FLAGS.learning_rate, betas=(0.9,0.999), eps=1e-8, weight_decay=0.01, amsgrad=False)
   
    
    xchtr=t.from_numpy(np.array(train_drugs)).int()
    xprtr=t.from_numpy(np.array(train_prots)).int()
    xcftr=t.from_numpy(train_cfp).float()
    xpftr=t.from_numpy(train_pfp).float()
    ytr=t.from_numpy(np.array(train_Y)).float()

    
    xchte =t.from_numpy(np.array(test_drugs)).int()
    xprte=t.from_numpy(np.array(test_prots)).int()
    xcfte=t.from_numpy(test_cfp).float()
    xpfte=t.from_numpy(test_pfp).float()
    yte=t.from_numpy(np.array(test_Y)).float()
 
    dataset=TensorDataset(xchtr,xprtr,xcftr,xpftr,ytr)
    dataloader = DataLoader(dataset, batch_size=FLAGS.batch_size, shuffle=False, pin_memory=False)
 
    print("%s: training start"%datetime.datetime.now())
    res = pd.DataFrame(columns=['epoch','MSE','train_CI','test_CI'])
    for epoch in range(FLAGS.num_epoch):
        model.train()
        if use_gpu:
            model.cuda()

        mse=[]
        for i, (xch,xpr,xcf,xpf,y) in enumerate(dataloader):
            if use_gpu:
                xch = xch.cuda()
                xpr=xpr.cuda()
                xcf=xcf.cuda()
                xpf=xpf.cuda()
                y=y.cuda()
            optimizer.zero_grad()
            ypred = model(xch,xpr,xcf,xpf)
            err = lossfn(ypred,y)
            err.backward()
            optimizer.step()
            mse.append(err.cpu().detach().numpy())
                
            if verbose>=1 and (i%100==0 or i==len(dataloader)-1):
                print("%s: epoch %d\ttraining batch %d/%d err = %f"%(datetime.datetime.now(),epoch+1, i+1,len(dataloader),err))
        mse_avg = np.mean(mse)
        model.eval()
        with t.no_grad():
            window=5000
            if use_gpu:
                model.cpu()

            trpred = []
            for i in range(0,len(xchtr),window):
                trpred.append(model(xchtr[i:min(len(xchtr),i+window)],xprtr[i:min(len(xchtr),i+window)],xcftr[i:min(len(xchtr),i+window)],xpftr[i:min(len(xchtr),i+window)]))
            trpred = t.concat(trpred)
            Y=ytr.detach().numpy()
            P=trpred.detach().numpy()
            tloss=get_cindex(Y,P)
            if verbose>=2:
                print("%s:tr loss"%datetime.datetime.now())
    
            tepred = []
            for i in range(0,len(xchte),window):
                tepred.append(model(xchte[i:min(len(xchte),i+window)],xprte[i:min(len(xchte),i+window)],xcfte[i:min(len(xchte),i+window)],xpfte[i:min(len(xchte),i+window)]))
            tepred = t.concat(tepred)
            teloss=get_cindex(yte.detach().numpy(),tepred.detach().numpy())
            if verbose>=2:
                print("%s: te loss"%datetime.datetime.now())
            
            res.loc[len(res)] = [epoch, mse_avg,tloss,teloss]
            text='%s: epoch:%d\tMSE=%f\ttraining loss:%f\ttest loss: %f'%(datetime.datetime.now(),epoch+1,mse_avg,tloss,teloss)
            print(text)

       
    return model, res


if __name__=="__main__":
    from argparse import Namespace

    dataset_path='data/kiba/'
    use_gpu = True
    max_seq_len=1000
    max_smi_len=100
    FLAGS = Namespace
    FLAGS.dataset_path=dataset_path
    FLAGS.is_log=0
    FLAGS.problem_type=1
    FLAGS.smi_window_lengths = [4,6,8]
    FLAGS.seq_window_lengths = [4,8,12]
    FLAGS.num_windows=[32]
    FLAGS.batch_size=256
    FLAGS.learning_rate=0.001
    FLAGS.num_epoch=50
    res_path="bin/"
    if not os.path.exists(res_path):
        os.mkdir(res_path)


    fi=0
    p1=1
    p2=2
    p3=0
    PATH = res_path+"model_tf_f%d.pt"%fi
    
    model=DTA(FLAGS.charsmiset_size+1,FLAGS.charseqset_size+1,FLAGS.smi_window_lengths[p1],FLAGS.seq_window_lengths[p2], NUM_FILTERS=FLAGS.num_windows[p3])
    model.load_state_dict(t.load(PATH))




# %%
