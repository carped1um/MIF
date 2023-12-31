import gc
import random
from copy import deepcopy

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import MIF
from data_loaders import Dataset_loader
from utils import CoxLoss, regularize_weights, CIndex_lifeline, cox_log_rank, accuracy_cox, count_parameters

batch_size = 1024


def train(opt, data, device, k):
    """
    To train the MIF
    Return: trained MIF, optimizer, metric_logger
    """

    # Make certain seeds
    cudnn.deterministic = True
    torch.cuda.manual_seed_all(666)
    torch.manual_seed(666)
    random.seed(666)

    # Initialize model_MIF and optimizer
    model = MIF(
        input_dims = (80, 80, 80),
        hidden_dims = (80, 80, 80, 256),
        output_dims = (20, 20, 1),
        dropout = 0.3).to(device)
        
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=opt.lr,
                                 betas=(opt.beta1, opt.beta2),
                                 weight_decay=opt.weight_decay)

    
    print("Number of Trainable Parameters: %d" % count_parameters(model))

    # Batch data iterator
    custom_data_loader = Dataset_loader(data, split='train')
    """Dataset_loader.__init__(data = dict, split = strings)"""
    train_loader = DataLoader(dataset=custom_data_loader,
                              batch_size=len(custom_data_loader),
                              shuffle=False)

    metric_logger = {'train': {'loss': [], 'pvalue': [], 'cindex': [], 'surv_acc': []},
                     'valid': {'loss': [], 'pvalue': [], 'cindex': [], 'surv_acc': []}}        # dicts in dict

    c_index_best = 0
    model_best = None
    epoch_best = 0

    # Epoch loop
    for epoch in tqdm(range(opt.epoch_count, (opt.niter + opt.niter_decay) + 1)):
        """
        opt.epoch_count: start of epoch, default = 1;
        opt.niter: of iter at starting learning rate, default = 0;
        opt.niter_decay: of iter to linearly decay learning rate to zero, default = 100
        """

        model.train()
        loss_epoch = 0

        # Batch loop
        for batch_idx, (x_gene, x_path, x_cna, censor, survtime) in enumerate(train_loader):

            # Model predicting
            censor = censor.to(device)
            pred, _ = model(x_gene.to(device), x_path.to(device), x_cna.to(device))  
            
            # Loss calculation
            loss_cox = CoxLoss(survtime, censor, pred, device)
            loss_reg = regularize_weights(model=model)
            loss = loss_cox + opt.lambda_reg * loss_reg 
            loss_epoch += loss_cox.data.item()                  # accumulate batch cox_losses as epoch loss
            '''.data: tensor → tensor; .items(): tensor with one element → value'''
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache() 

        # Evaluate
        if opt.measure or epoch == (opt.niter + opt.niter_decay):
            '''
            opt.measure = 1(default): enable measure while training (make program faster),
            in the end of training.
            '''

            # Valid
            loss_test, cindex_test, pvalue_test, surv_acc_test, pred_test, code_test = \
                test(opt, model, data, split = 'validation', device = device)

            metric_logger['valid']['loss'].append(loss_test)
            metric_logger['valid']['cindex'].append(cindex_test)
            metric_logger['valid']['pvalue'].append(pvalue_test)
            metric_logger['valid']['surv_acc'].append(surv_acc_test)

            if cindex_test > c_index_best:
                c_index_best = cindex_test
                model_best, epoch_best = deepcopy(model), epoch

    return model_best, optimizer, metric_logger


def test(opt, model, data, split, device):
    model.eval()
    custom_data_loader = Dataset_loader(data, split)
    test_loader = DataLoader(dataset=custom_data_loader,
                             batch_size=len(custom_data_loader),
                             shuffle=False)
    risk_pred_all, censor_all, survtime_all = np.array([]), np.array([]), np.array([])
    loss_test = 0
    code_final = None

    # Batch loop
    for batch_idx, (x_gene, x_path, x_cna, censor, survtime) in enumerate(test_loader):
        censor = censor.to(device)
        with torch.no_grad():
            pred, code = model(x_gene.to(device), x_path.to(device), x_cna.to(device))

        loss_cox = CoxLoss(survtime, censor, pred, device)
        loss_test += loss_cox.data.item()                   # .data.item(): tensor → value

        risk_pred_all = np.concatenate((risk_pred_all, pred.detach().cpu().numpy().reshape(-1)))
        censor_all = np.concatenate((censor_all, censor.detach().cpu().numpy().reshape(-1)))
        survtime_all = np.concatenate((survtime_all, survtime.detach().cpu().numpy().reshape(-1)))

        if batch_idx == 0:
            code_final = code
        else:
            code_final = torch.cat([code_final, code])

    ###################################################
    # ==== Measuring Test Loss, C-Index, P-Value ==== #
    ###################################################
    loss_test /= len(test_loader.dataset)
    cindex_test = CIndex_lifeline(risk_pred_all, censor_all, survtime_all)
    pvalue_test = cox_log_rank(risk_pred_all, censor_all, survtime_all)
    surv_acc_test = accuracy_cox(risk_pred_all, censor_all)
    pred_test = [risk_pred_all, survtime_all, censor_all]
    code_final_data = code_final.data.cpu().numpy()

    return loss_test, cindex_test, pvalue_test, surv_acc_test, pred_test, code_final_data
