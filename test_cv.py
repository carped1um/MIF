import os
import logging
import numpy as np
import pickle
import torch

# Env
from model import MIF
from options import parse_args
from train_test import train, test


opt = parse_args()
device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
print("Using device:", device)

if not os.path.exists(os.path.join(opt.model_save, opt.exp_name, opt.model_name)):
		os.makedirs(os.path.join(opt.model_save, opt.exp_name, opt.model_name))

data_cv_path = '%s/%s' % (opt.dataroot, opt.datatype)
print("Loading %s" % data_cv_path)
data_cv = pickle.load(open(data_cv_path, 'rb'))
data_cv_splits = data_cv['cv_splits']
results = []

for k, data in data_cv_splits.items():
	print("*******************************************")
	print("************** SPLIT (%d/%d) **************" % (k, len(data_cv_splits.items())))
	print("*******************************************")
	load_path = os.path.join(opt.model_save, opt.exp_name, opt.model_name, '%s_%d.pt' % (opt.model_name, k))
	model_ckpt = torch.load(load_path, map_location=device)

	model_state_dict = model_ckpt['model_state_dict']
	if hasattr(model_state_dict, '_metadata'): del model_state_dict._metadata

	model = MIF(
        input_dims = (80, 80, 80),
        hidden_dims = (80, 80, 80, 256),
        output_dims = (20, 20, 1),
        dropout = 0.3).to(device)

	if isinstance(model, torch.nn.DataParallel): model = model.module
	print('Loading the model_MIF from %s' % load_path)
	model.load_state_dict(model_state_dict)


	### 3.2 Evalutes Train + Test Error, and Saves Model
	loss_test, cindex_test, pvalue_test, surv_acc_test, pred_test, code_pred = test(opt, model, data, 'test', device)

	print("[Final] Apply model_MIF to testing set: C-Index: %.10f" % (cindex_test))
	logging.info("[Final] Apply model_MIF to testing set: cC-Index: %.10f" % (cindex_test))
	results.append(cindex_test)
	### 3.3 Saves Model
	# pickle.dump(pred_test, open(os.path.join(opt.results, opt.exp_name, opt.model_name, '%s_%dpred_test.pkl' % (opt.model_name, k)), 'wb'))
	pickle.dump(pred_test, open(os.path.join(opt.results, '%s_%dpred_test.pkl' % (opt.model_name, k)), 'wb'))

np.savetxt(opt.results + "CV_cindex.csv", results, delimiter=",")
print('Split Results:', results)
print("Average:", np.array(results).mean())
# pickle.dump(results, open(os.path.join(opt.results, opt.exp_name, opt.model_name, '%s_results.pkl' % opt.model_name), 'wb'))
pickle.dump(results, open(os.path.join(opt.results, '%s_results.pkl' % opt.model_name), 'wb'))
