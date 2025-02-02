#%%
import os
import numpy as np
import pandas as pd
import torch as t
from torchmodel import DTA, training_loop, training_loop_tf
from DTAfunctions import prepare_interaction_pairs, prepare_interaction_real
from datahelper import DataSet
from argparse import Namespace
import json
from collections import OrderedDict
from copy import deepcopy
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def baseline_task(FLAGS,res_path):
    dataset = DataSet( fpath = dataset_path,
                    setting_no = 1,
                    seqlen = max_seq_len,
                    smilen = max_smi_len,
                    need_shuffle = False )
    FLAGS.charseqset_size = dataset.charseqset_size 
    FLAGS.charsmiset_size = dataset.charsmiset_size 
    XD, XT, Y = dataset.parse_data(FLAGS)
    XD = np.asarray(XD)
    XT = np.asarray(XT)
    Y = np.asarray(Y)

    label_row_inds, label_col_inds = np.where(np.isnan(Y)==False)

    test_set, outer_train_sets = dataset.read_sets(FLAGS) 
    foldinds = len(outer_train_sets)

    val_sets = []
    train_sets = []
    for val_foldind in range(foldinds):
        val_fold = outer_train_sets[val_foldind]
        val_sets.append(val_fold)
        otherfolds = deepcopy(outer_train_sets)
        otherfolds.pop(val_foldind)
        otherfoldsinds = [item for sublist in otherfolds for item in sublist]
        train_sets.append(otherfoldsinds)
        print("val set", str(len(val_fold)))
        print("train set", str(len(otherfoldsinds)))
    
    
    ## excluded chemicals
    ligands = json.load(open(FLAGS.dataset_path+"ligands_can.txt"), object_pairs_hook=OrderedDict)
    chems = pd.DataFrame(list(ligands.values()))[0]    
    proteins = json.load(open(FLAGS.dataset_path+"proteins.txt"), object_pairs_hook=OrderedDict)
    prots = pd.DataFrame(list(proteins.values()))[0]

    chemblacklist = pd.read_csv(FLAGS.dataset_path+"remove_chemicals.txt",header=None)[0]
    exclude_index = chemblacklist.apply(lambda x: chems.index[chems.tolist().index(x)])
    

    # ## Chem and prot fp data
    cscalar = MinMaxScaler()
    chem_fp = pd.read_csv(FLAGS.dataset_path+'fp_chemical.csv')
    chemcols = list(chem_fp.columns)[1:]
    chem_fp['idx'] = chem_fp.name.apply(lambda x: chems.index[chems.tolist().index(x)])
    cfp = cscalar.fit_transform(chem_fp[chemcols])
    ci = pd.Index(chem_fp['idx'])


    pscalar = MinMaxScaler()
    prot_fp = pd.read_csv(FLAGS.dataset_path+'fp_protein.csv')
    protcols=list(prot_fp.columns)[1:]
    prot_fp['idx'] = prot_fp.name.apply(lambda x: prots.index[prots.tolist().index(x)])
    pfp = pscalar.fit_transform(prot_fp[protcols])
    pi=pd.Index(prot_fp['idx'])

    
    # # prepare fold data
    for fi in range(len(train_sets)):
        tinds = train_sets[fi]
        trrows = label_row_inds[tinds]
        trcols = label_col_inds[tinds]
        bi=np.argwhere(~np.isin(trrows, exclude_index)).ravel()
        trrows = trrows[bi]
        trcols = trcols[bi]
        trcf = cfp[ci[trrows]]
        trpf = pfp[pi[trcols]]
        train_drugs, train_prots,  train_Y = prepare_interaction_pairs(XD, XT, Y, trrows, trcols)

        vinds = val_sets[fi]
        valrows = label_row_inds[vinds]
        valcols = label_col_inds[vinds]
        bi=np.argwhere(~np.isin(valrows, exclude_index)).ravel()
        valrows=valrows[bi]
        valcols=valcols[bi]
        valcf = cfp[ci[valrows]]
        valpf = pfp[pi[valcols]]    
        val_drugs, val_prots,  val_Y = prepare_interaction_pairs(XD, XT,  Y, valrows, valcols)

        terows = label_row_inds[test_set]
        tecols=label_col_inds[test_set]
        bi=np.argwhere(~np.isin(terows, exclude_index)).ravel()
        terows=terows[bi]
        tecols=tecols[bi]
        tecf = cfp[ci[terows]]
        tepf = pfp[pi[tecols]]    
        test_drugs, test_prots, test_Y = prepare_interaction_pairs(XD,XT,Y,terows,tecols)

        
        ## Training
        for p1 in range(len(FLAGS.smi_window_lengths)):
            for p2 in range(len(FLAGS.seq_window_lengths)):
                for p3 in range(len(FLAGS.num_windows)):
                    PATH = res_path+"model_f%dp1%dp2%dp3%d.pt"%(fi,p1,p2,p3)
                    model, res = training_loop(FLAGS,p1,p2,p3,
                                train_drugs,train_prots,trcf,trpf,train_Y,
                                val_drugs,val_prots,valcf,valpf,val_Y,
                                test_drugs,test_prots,tecf,tepf,test_Y,
                                use_gpu=use_gpu,path=PATH,verbose=2)
                    res.to_csv(res_path+"res_f%dp1%dp2%dp3%d.csv"%(fi,p1,p2,p3),index=False)


def transfer_task(FLAGS, res_path, p1=1, p2=2, p3=0):
    dataset = DataSet( fpath = dataset_path,
                    setting_no = 1,
                    seqlen = max_seq_len,
                    smilen = max_smi_len,
                    need_shuffle = False )
    FLAGS.charseqset_size = dataset.charseqset_size 
    FLAGS.charsmiset_size = dataset.charsmiset_size 
    XD, XT, Y = dataset.parse_data(FLAGS)
    XD = np.asarray(XD)
    XT = np.asarray(XT)
    Y = np.asarray(Y)


    test_set, outer_train_sets = dataset.read_sets(FLAGS) 
    foldinds = len(outer_train_sets)

    val_sets = []
    train_sets = []
    for val_foldind in range(foldinds):
        val_fold = outer_train_sets[val_foldind]
        val_sets.append(val_fold)
        otherfolds = deepcopy(outer_train_sets)
        otherfolds.pop(val_foldind)
        otherfoldsinds = [item for sublist in otherfolds for item in sublist]
        train_sets.append(otherfoldsinds)
        print("val set", str(len(val_fold)))
        print("train set", str(len(otherfoldsinds)))
    
    
    ## excluded chemicals
    ligands = json.load(open(FLAGS.dataset_path+"ligands_can.txt"), object_pairs_hook=OrderedDict)
    chems = pd.DataFrame(list(ligands.values()))[0]    
    proteins = json.load(open(FLAGS.dataset_path+"proteins.txt"), object_pairs_hook=OrderedDict)
    prots = pd.DataFrame(list(proteins.values()))[0]
    

    # ## Chem and prot fp data
    cscalar = MinMaxScaler()
    chem_fp = pd.read_csv(FLAGS.dataset_path+'fp_chemical.csv')
    chemcols = list(chem_fp.columns)[1:]
    chem_fp['idx'] = chem_fp.name.apply(lambda x: chems.index[chems.tolist().index(x)])
    cfp = cscalar.fit_transform(chem_fp[chemcols])



    pscalar = MinMaxScaler()
    prot_fp = pd.read_csv(FLAGS.dataset_path+'fp_protein.csv')
    protcols=list(prot_fp.columns)[1:]
    prot_fp['idx'] = prot_fp.name.apply(lambda x: prots.index[prots.tolist().index(x)])
    pfp = pscalar.fit_transform(prot_fp[protcols])



    # # transfer data
    tfD, tfT, tfY = dataset.parse_data(FLAGS,path='data/transferset/',islog=False)
    tfrows=list(range(len(tfD)))
    tfcols=[0 for n in range(len(tfD))]
    tfdrugs, tfprots, tfY = prepare_interaction_pairs(tfD,tfT,tfY,tfrows,tfcols)

    chem_fp = pd.read_csv('data/transferset/'+'fp_chemical.csv')
    chemcols = list(chem_fp.columns)[1:]
    tfcfp = cscalar.transform(chem_fp[chemcols])
    prot_fp = pd.read_csv('data/transferset/'+'fp_protein.csv')
    protcols=list(prot_fp.columns)[1:]
    tfpfp = pscalar.transform(prot_fp[protcols])
    tfc=tfcfp[tfrows]
    tfp = tfpfp[tfcols]
    
    train_drugs, test_drugs, train_prots, test_prots, train_cf, test_cf, train_pf, test_pf, train_Y, test_Y = train_test_split(tfdrugs,tfprots,tfc,tfp,tfY, test_size=0.1, random_state=42)
    # prepare fold data
    for fi in range(len(train_sets)):
        PATH = res_path+"model_tf_f%d.pt"%fi

        # Transfer learning
        model=DTA(FLAGS.charsmiset_size+1,FLAGS.charseqset_size+1,FLAGS.smi_window_lengths[p1],FLAGS.seq_window_lengths[p2], NUM_FILTERS=FLAGS.num_windows[p3])
        model.load_state_dict(t.load(PATH))
        model,res = training_loop_tf(FLAGS,model,train_drugs,train_prots,train_cf,train_pf,train_Y, 
                    test_drugs,test_prots,test_cf,test_pf,test_Y,
                    use_gpu=use_gpu,verbose=2 , path="model_tf_f%d.pt"%fi)
        t.save(model.state_dict(), res_path+"model_tf_f%d.pt"%fi)
        res.to_csv(res_path+"transfer_res_f%d.csv"%fi,index=False)


def real_task(FLAGS, real_path = "real/cmnpd/", p1=1, p2=2, p3=0):
    dataset = DataSet( fpath = dataset_path,
                    setting_no = 1,
                    seqlen = max_seq_len,
                    smilen = max_smi_len,
                    need_shuffle = False )
    FLAGS.charseqset_size = dataset.charseqset_size 
    FLAGS.charsmiset_size = dataset.charsmiset_size 
    XD, XT, Y = dataset.parse_data(FLAGS)
    XD = np.asarray(XD)
    XT = np.asarray(XT)
    Y = np.asarray(Y)

    test_set, outer_train_sets = dataset.read_sets(FLAGS) 
    foldinds = len(outer_train_sets)

    val_sets = []
    train_sets = []
    for val_foldind in range(foldinds):
        val_fold = outer_train_sets[val_foldind]
        val_sets.append(val_fold)
        otherfolds = deepcopy(outer_train_sets)
        otherfolds.pop(val_foldind)
        otherfoldsinds = [item for sublist in otherfolds for item in sublist]
        train_sets.append(otherfoldsinds)
        print("val set", str(len(val_fold)))
        print("train set", str(len(otherfoldsinds)))

    ligands = json.load(open(FLAGS.dataset_path+"ligands_can.txt"), object_pairs_hook=OrderedDict)
    chems = pd.DataFrame(list(ligands.values()))[0]    
    proteins = json.load(open(FLAGS.dataset_path+"proteins.txt"), object_pairs_hook=OrderedDict)
    prots = pd.DataFrame(list(proteins.values()))[0]
    

    # ## Chem and prot fp data
    cscalar = MinMaxScaler()
    chem_fp = pd.read_csv(FLAGS.dataset_path+'fp_chemical.csv')
    chemcols = list(chem_fp.columns)[1:]
    chem_fp['idx'] = chem_fp.name.apply(lambda x: chems.index[chems.tolist().index(x)])
    cfp = cscalar.fit_transform(chem_fp[chemcols])


    pscalar = MinMaxScaler()
    prot_fp = pd.read_csv(FLAGS.dataset_path+'fp_protein.csv')
    protcols=list(prot_fp.columns)[1:]
    prot_fp['idx'] = prot_fp.name.apply(lambda x: prots.index[prots.tolist().index(x)])
    pfp = pscalar.fit_transform(prot_fp[protcols])



    
    # real data 
    realD,realT = dataset.parse_real(fpath=real_path)
    realrows=list(range(len(realD)))
    realcols=[0 for n in range(len(realD))]
    realdrugs, realprots = prepare_interaction_real(realD,realT,realrows,realcols)


    chem_fp = pd.read_csv(real_path+'fp_chemical.csv')
    chemcols = list(chem_fp.columns)[1:]
    tfcfp = cscalar.transform(chem_fp[chemcols])
    prot_fp = pd.read_csv(real_path+'fp_protein.csv')
    protcols=list(prot_fp.columns)[1:]
    tfpfp = pscalar.transform(prot_fp[protcols])
    realcf = tfcfp[realrows]
    realpf = tfpfp[realcols]
    

    # # prepare fold data
    for fi in range(len(train_sets)):
        # # Real data
        model=DTA(FLAGS.charsmiset_size+1,FLAGS.charseqset_size+1,FLAGS.smi_window_lengths[p1],FLAGS.seq_window_lengths[p2], NUM_FILTERS=FLAGS.num_windows[p3])
        model.load_state_dict(t.load("bin/model_tf_f%d.pt"%fi))
        model.eval()
        xc =t.from_numpy(np.array(realdrugs)).int()
        xp=t.from_numpy(np.array(realprots)).int()
        xcf=t.from_numpy(realcf).float()
        xpf=t.from_numpy(realpf).float()
        
        window=2000
        pred = []
        for i in range(0,len(xc),window):
            pred.append(model(xc[i:min(len(xc),i+window)],xp[i:min(len(xc),i+window)],xcf[i:min(len(xc),i+window)],xpf[i:min(len(xc),i+window)]))
        pred = t.concat(pred)
        p=pred.detach().numpy()
        
        np.savetxt(real_path+'prediction_f%d.csv'%fi,p,delimiter=',')


        
if __name__=="__main__":
    skiplist=[]
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

    # baseline_task(FLAGS,res_path)    
    # transfer_task(FLAGS,res_path)
    # real_task(FLAGS,real_path='real/cmnpd/')
    real_task(FLAGS,real_path='real/bfkorea/')

   
# %%
