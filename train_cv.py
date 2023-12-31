import gc
import logging
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Env
from options import parse_args
from train_test import train, test
from utils import hazard2grade

np.set_printoptions(suppress=True)  

##################################
# 1. Initializes parser and device
##################################

opt = parse_args()
device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) \
    if opt.gpu_ids and torch.cuda.is_available() else torch.device('cpu')
print("Using device:", device)

if not os.path.exists(os.path.join(opt.model_save, opt.exp_name, opt.model_name)):
    os.makedirs(os.path.join(opt.model_save, opt.exp_name, opt.model_name))

#####################
# 2. Initializes Data
#####################

data_cv_path = '%s/%s' % (opt.dataroot, opt.datatype)
print("Loading training data from: %s\n" % data_cv_path)
data_cv = pickle.load(open(data_cv_path, 'rb'))
data_cv_splits = data_cv['cv_splits']         
average_results = []
os_time = []
os_status = []
risk_pred = []
code_pred = []
label_pred = []

###########################################
# 3. Sets-Up Main Loop of k-fold validation
###########################################

for k, data in data_cv_splits.items():  # the k th fold / fold loop
    print("************** SPLIT (%d/%d) **************" % (k+1, len(data_cv_splits.items())))

    # create folder to save results
    if not os.path.exists(os.path.join(opt.results, opt.exp_name, opt.model_name)):
        os.makedirs(os.path.join(opt.results, opt.exp_name, opt.model_name))

    ##################
    # 3.1 Trains Model
    ##################
    
    model, optimizer, metric_logger = train(opt, data, device, k)  
    print("Training is Over!")

    ##################################################
    # 3.2 Evaluate Valid + Test Error, and Saves Model
    ##################################################
    
    print("Evaluate Valid Dataset...")
    loss_train, cindex_train, pvalue_train, surv_acc_train, pred_train, code_train = test(opt, model, data, 'validation',
                                                                                          device)
    print("[Final] Apply MIF to valid set: C-Index: %.4f, epoch: %d"
          % (np.max(np.array(metric_logger['valid']['cindex'])),
             np.argmax(np.array(metric_logger['valid']['cindex']))))

    print("Evaluate Test Dataset...")
    loss_test, cindex_test, pvalue_test, surv_acc_test, pred_test, code_test = test(opt, model, data, 'test', device)

    print("[Final] Apply model_MIF to test set: C-Index:  %.4f" % cindex_test)
    print()
    average_results.append(cindex_test)

    #################
    # 3.3 Saves Model
    #################
    
    model_state_dict = model.state_dict()
    torch.save({
        'split': k,
        'opt': opt,
        'epoch': opt.niter + opt.niter_decay,
        'data': data,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metric_logger},
        os.path.join(opt.model_save, opt.exp_name, opt.model_name, '%s_%d.pt' % (opt.model_name, k))
    )

    pickle.dump(pred_test, open(
        os.path.join(opt.results, opt.exp_name, opt.model_name, '%s_%d_pred_test.pkl' % (opt.model_name, k)), 'wb'))

    df = pd.DataFrame({'os_time': pred_test[1], 'os_status': pred_test[2], 'risk_pred': pred_test[0]})
    df.to_csv(opt.results + "%d-fold_pred.csv" % k,
              index=False,                                   # if write row names
              header=True)                                   # if write out column names

    PI = hazard2grade(pred_test[1], 60)
    np.savetxt(opt.results + "%d-fold_code_test.csv" % k, code_test,
               delimiter=","                                             # String or character separating columns
               )
    np.savetxt(opt.results + "%d-fold_label_test.csv" % k, PI, delimiter=",")

    risk_pred.extend(pred_test[0])
    os_time.extend(pred_test[1])
    os_status.extend(pred_test[2])
    code_pred.extend(code_test)
    label_pred.extend(PI)

df1 = pd.DataFrame({'os_time': os_time, 'os_status': os_status, 'risk_pred': risk_pred})
df1.to_csv(opt.results + "out_pred_5fold.csv", index=False, header=True)

df2 = pd.DataFrame({'os_time': os_time, 'os_status': os_status, 'risk_pred': label_pred})
df2.to_csv(opt.results + "risk_pred_5fold.csv", index=False, header=True)

np.savetxt(opt.results + "code_test.csv", code_pred, delimiter=",")

np.savetxt(opt.results + "label_test.csv", label_pred, fmt="%d", delimiter=",")
print(f"Split Average Results: {[float('{:.4f}'.format(i)) for i in average_results]}")
print(f"Average_results: %.4f, std: %.4f"
      % (np.array(average_results).mean(), np.array(average_results).std()))
pickle.dump(average_results,
            open(os.path.join(opt.results, opt.exp_name, opt.model_name, '%s_results.pkl' % opt.model_name), 'wb'))
