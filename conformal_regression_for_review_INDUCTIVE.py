# Author: Isidro Cortes Ciriano (isidrolauscher@gmail.com) 
from __future__ import print_function
import os,sys, os.path
target_now = sys.argv[1]
bio_type = sys.argv[2]
#from validation_metrics import *
#import gzip
#from collections import defaultdict
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import rdkit.rdBase
#from rdkit.DataStructs import BitVectToText
from rdkit import DataStructs
#from rdkit.Chem import Descriptors as Descriptors
#from rdkit.ML.Descriptors import MoleculeDescriptors
#import pylab as plt
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn import ensemble
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn import cross_validation

def find_index(l,element):
    return ([k for k, j in enumerate(l) if j == element])


extract_dataset=False
if extract_dataset:
    from chembl_webresource_client.new_client import new_client
    target = new_client.target
    activity = new_client.activity
    target_now_info = target.filter(target_synonym__icontains=target_now)
    def index_homo_sapiens(target_now_info,organism="Homo sapiens"):
        for idx,pp in enumerate(target_now_info):
            if pp['organism'] == organism:
                return idx
    idx_now = index_homo_sapiens(target_now_info)
    target_now_info = target_now_info[idx_now+1]
    dataset_info = target_now_info['target_components'][0]['accession'] + "\t" + target_now_info['target_chembl_id'] + "\t" + target_now_info['organism'] + "\t" +target_now_info['target_type'] +"\t"+target_now_info['pref_name'] + "\t"+bio_type +"\t"
    
    target_now_activities = activity.filter(target_chembl_id=target_now_info['target_chembl_id']).filter(standard_type=bio_type)
    #print "Number of data points", len(target_now_activities)
    smiles = [ x['canonical_smiles'] for x in target_now_activities if x['standard_units'] == "nM" and x['published_relation'] == '=']
    activities = [ x['pchembl_value'] for x in target_now_activities if x['standard_units'] == "nM" and x['published_relation'] == '=']
    chembl_ids = [ x['molecule_chembl_id'] for x in target_now_activities if x['standard_units'] == "nM" and x['published_relation'] == '=']
    
    activities_float = []
    torm = []
    for i,x in enumerate(activities):
        try:
            activities_float.append(float(x))
        except:
            torm.append(i)
    
    activities = activities_float
    smiles = [x for i,x in enumerate(smiles) if i not in torm]
    chembl_ids = [x for i,x in enumerate(chembl_ids) if i not in torm]
    chembl_ids = [x.encode("ascii") for x in chembl_ids]
    if len(activities) != len(smiles):
        raise "lengths differ"
    # standardize molecules
    from chembl_webresource_client.utils import utils
    incorrect_mols=[]
    mols=[]
    for i,x in enumerate(smiles):
        if i in torm:
            pass
        try:
            a = Chem.MolFromSmiles(x)
            if Chem.MolFromSmiles(x) is None:
                incorrect_mols.append(i)
            if Chem.MolFromSmiles(x) is not None:
                mols.append(Chem.MolFromSmiles(x))
        except:
            incorrect_mols.append(i)
    
    activities = [x for i,x in enumerate(activities) if i not in incorrect_mols]
    chembl_ids = [x for i,x in enumerate(chembl_ids) if i not in incorrect_mols]
    #print "# mols processed", len(mols), " Bioactivity points: ", len(activities), len(chembl_ids)
    
    if len(mols) != len(activities) or len(mols) != len(chembl_ids):
        raise 
    activities = np.asarray(activities)
    visited=[]
    mols_now=[]
    ys=[]
    mol_ids=[]
    for i,m in enumerate(mols):
        if chembl_ids[i] not in visited:
            visited.append(chembl_ids[i])
            idx = find_index(chembl_ids, chembl_ids[i] )
            if len(idx) > 1:
                y_now = np.mean(np.asarray( activities[idx]))
            else:
                y_now = activities[i]
            mols_now.append(m)
            ys.append(y_now)
            mol_ids.append(chembl_ids[i])
    activities = ys
    chembl_ids = visited
    
    from standardiser import standardise
    import logging
    incorrect_mols=[] # to remove those that cannot be standardised
    mols=[]
    #standardise.logger.setLevel('DEBUG')
    for i,m in enumerate(mols_now):
        #print "Standardizing molecule: ", i
        parent = None
        try:
            parent = standardise.run(m)
            mols.append(parent)
        except standardise.StandardiseException as e:
            logging.warning(e.message)
            incorrect_mols.append(i)
    
    activities = [x for i,x in enumerate(activities) if i not in incorrect_mols]
    chembl_ids = [x for i,x in enumerate(chembl_ids) if i not in incorrect_mols]
    
    #--------------------------------------------------------
    # writing in .sdf format:
    #--------------------------------------------------------
    for i,m in enumerate(mols):
        m.SetProp("p"+str(sys.argv[2]),str(activities[i]))
        m.SetProp("ChEMBL_ID",chembl_ids[i])
    outname="./datasets/"+target_now+"/"+target_now+".sdf"
    if not os.path.isfile(outname):
        os.system("mkdir -p ./datasets/"+target_now)
        w = Chem.SDWriter(outname)
        for m in mols: w.write(m)
    # save data set info to file
    dataset_info  = dataset_info + "\t"+str(len(activities))+"\n"
    #print dataset_info
    with open('results_regression_inductive/datasets_info', 'a') as file:
        file.write(dataset_info)
    #print dataset_info



suppl = Chem.SDMolSupplier('datasets/{}/{}.sdf'.format(target_now, target_now)) 
mols = [x for x in suppl if x is not None]
activities =[float(m.GetProp("pIC50")) for m in mols]
if len(activities) != len(mols):
    raise "Dimensions for mols and bioactivities differ"

#--------------------------------------------------------
# calculate Morgan fingerprints
#--------------------------------------------------------
Morgan_fps = []; torm=[]
for i,sample in enumerate(mols):
    use = True
    try:
            fp = AllChem.GetMorganFingerprintAsBitVect(sample,2,nBits=1024)
    except:
            use = False
            #print 'Error computing descriptor %s for %s'%(D[0],sample.GetProp('Name'))
            torm.append(i)

    if use:
        Morgan_fps.append(fp)

Morgan_fps = np.array(Morgan_fps,dtype="float")
activities = [x for i,x in enumerate(activities) if i not in torm]

if Morgan_fps.shape[0] != len(activities):
    raise "Dimensions do not match"
activities = np.asarray(activities,dtype="float")

#--------------------------------------------------------
# Divide the original dataset into training and tesst set
#--------------------------------------------------------
for repetition in [2]:#,2,3,4,5,6,7,8,9,10]:
    for test_size in [0.15]: #,0.2,0.3,0.4,0.5]:
        #print target_now, test_size
        if not os.path.isfile("results_regression_inductive/test_"+target_now+"_"+str(repetition)+"_"+str(test_size)):
             X_train, X_test, y_train, y_test = cross_validation.train_test_split(Morgan_fps, activities, test_size=test_size, random_state=int(repetition))
             X_test = X_test.astype(float)
             X_train = X_train.astype(float)

             
             X_calibration, X_test, y_calibration, y_test = cross_validation.train_test_split(X_test, y_test, test_size=0.5, random_state=int(repetition))
             X_test = X_test.astype(float)
             X_calibration = X_calibration.astype(float)
             #--------------------------------------------------------
             # Train a Random Forest Model (no need to optimize parameters; see my papers)
             # Do k-fold CV, each time calculating the alpha values
             #--------------------------------------------------------
             alphas = []; diffs=[]; ys=[]; zs=[]; stds=[]; pred_errors=[]
             indexes = np.arange(len(y_train))
             np.random.shuffle(indexes)
             np.random.shuffle(indexes)
             np.random.shuffle(indexes)
             stride= len(indexes)#/10
             idx_used = []; stds_tr=[]
             folds = []
             
             for i in [0]:#range(0,10):
                 #print "Training model: ",i
                 test_index = indexes[stride*(i):stride*(i+1)]
                 train_index = np.concatenate((indexes[0:stride*(i)],indexes[stride*(i+1):stride*10]))
                 idx_used.append(indexes[stride*(i):stride*(i+1)])
                 folds.append([i]*len(test_index))
             
                 X_train_now = X_train#[train_index,]
                 X_test_now = X_calibration#[test_index,]
                 y_train_now = y_train#[train_index]
                 y_test_now = y_calibration#[test_index]

                 RF_model = RandomForestRegressor(n_estimators=500,random_state=23,n_jobs=2)
                 RF_model.fit(X_train_now,y_train_now)
                 ztest = RF_model.predict(X_test_now); print(y_train_now)
                 diff = np.abs(y_test_now - ztest)
                 # get as error the variability across the ensemble:
                 a=RF_model.estimators_
                 pep = True
                 for tree in range(500):
                     if pep:
                         a = RF_model.estimators_[tree].predict(X_test_now)
                         pep=False
                     else:
                         a = np.vstack((a,RF_model.estimators_[tree].predict(X_test_now)))
                 #calculate std across the ensemble and calculate alphas
                 std_ensemble = a.std(axis=0); stds_tr.append(std_ensemble)
                 if len(std_ensemble) != len(diff):
                     raise "Lengths differ"
                 alphas_now = diff / std_ensemble; alphas.append(alphas_now)

                 ys.append(y_test_now)
                 zs.append(ztest)
                 if i == 0:
                     atr=a
                 else:
                     atr=np.hstack((atr,a))
             
             def flatten(l):
                 return [item for sublist in l for item in sublist]
             
             ys = flatten(ys)
             zs =  flatten(zs)
             alphas_now = flatten(alphas)
             idx_used = flatten(idx_used)
             folds = flatten(folds); stds_tr=flatten(stds_tr)
             
             
             #--------------------------------------------------------
             # Train a model on full training data
             #--------------------------------------------------------
             #print "Training model on entire training set..\n"
             #RF_model = RandomForestClassifier(n_estimators=500,random_state=23,n_jobs=2)
             RF_model = RandomForestRegressor(n_estimators=500,random_state=23,n_jobs=2)
             RF_model.fit(X_train,y_train)
             ztest = RF_model.predict(X_test)
             
             # get as error the variability across the ensemble:
             atest=RF_model.estimators_
             flag = True
             for tree in range(500):
                 if flag:
                     atest = RF_model.estimators_[tree].predict(X_test)
                     flag=False
                 else:
                     atest = np.vstack((atest,RF_model.estimators_[tree].predict(X_test)))
             
             
             std_ensemble_test = atest.std(axis=0)

             #-----------------------
             # Save training 
             #-----------------------
             print(len(ys))
             f = open('results_regression_inductive/tr_'+target_now+'_'+str(repetition)+"_"+str(test_size), 'w')
             f.write("obs\tpred\tfolds\tcomps_used\talpha\tstd_forest\n")
             for i in np.arange(0,len(ys)):
                 f.write(str(ys[i])+"\t"+str(zs[i])+"\t"+str(folds[i])+"\t"+str(idx_used[i])+"\t"+str(alphas_now[i])+"\t"+str(stds_tr[i])+"\n")
             f.close() 
             np.savetxt('results_regression_inductive/tr_forest_preds_{}_{}_{}'.format(target_now,repetition,test_size), atr, delimiter="\t")
             #-----------------------
             # Save test
             #-----------------------
             f = open('results_regression_inductive/test_'+target_now+'_'+str(repetition)+"_"+str(test_size), 'w')
             f.write("obs\tpred\tstd_forest\n")
             for i in np.arange(0,len(y_test)):
                 f.write(str(y_test[i])+"\t"+str(ztest[i])+"\t"+str(std_ensemble_test[i])+"\n")
             f.close() 
             # save predictions across the forest
             np.savetxt('results_regression_inductive/test_forest_preds_{}_{}_{}'.format(target_now,repetition,test_size), atest, delimiter="\t")
             
             
             
