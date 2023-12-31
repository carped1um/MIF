import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class EbdLayer(nn.Module):

    def __init__(self, in_size, hidden_size):
        super(EbdLayer, self).__init__()
        encoder1 = nn.Sequential(nn.Linear(in_size, hidden_size), nn.LeakyReLU())
        encoder2 = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.LeakyReLU())
        self.encoder = nn.Sequential(encoder1, encoder2)

    def forward(self, x):
        y = self.encoder(x)
        return y


class AGI(nn.Module):

    def __init__(self, ebdDim, nLayers):
    
        super(AGI, self).__init__()
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        
        self.ebdDim = ebdDim
        self.nLayers = nLayers

        self.theta_linear_list = nn.Sequential()
        self.phi_linear_list = nn.Sequential()
        self.rho_linear_list = nn.Sequential()
        for i in range(nLayers):
            self.theta_linear_list.add_module(f'theta_linear_{i}', nn.Linear(self.ebdDim, self.ebdDim))
            self.phi_linear_list.add_module(f'phi_linear_{i}', nn.Linear(self.ebdDim, self.ebdDim))
            self.rho_linear_list.add_module(f'rho_linear_{i}', nn.Linear(self.ebdDim, self.ebdDim))

        self.interacted_linear = nn.Linear(self.ebdDim, self.ebdDim) 
        self.feed_fc = nn.Sequential(
            nn.Linear(self.ebdDim, self.ebdDim),
            nn.Linear(self.ebdDim, self.ebdDim),
            nn.LeakyReLU())

    def forward(self, input_tensor):
    
        interactedOut = torch.zeros_like(input_tensor)

        for i in range(self.nLayers):
            theta = self.theta_linear_list[i](input_tensor)
            phi = self.phi_linear_list[i](input_tensor)
            rho = self.rho_linear_list[i](input_tensor)
            affMat = F.softmax(
                (torch.bmm(theta, torch.transpose(phi, -1, -2)) * (float(self.ebdDim) ** -0.5)),
                dim=-1)
            interactedOut += torch.bmm(affMat, rho)

        resOut = F.normalize(
            torch.add(input_tensor, self.interacted_linear(interactedOut / self.nLayers)))
        output = F.normalize(
            torch.add(resOut, self.feed_fc(resOut)))

        return output


class ACGI(nn.Module):

    def __init__(self, ebdDim, nLayers):
    
        super(ACGI, self).__init__()
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        
        self.ebdDim = ebdDim
        self.nLayers = nLayers

        self.theta_linear_list = nn.Sequential()
        self.phi_linear_list = nn.Sequential()
        self.rho_linear_list = nn.Sequential()
        self.psi_linear_list = nn.Sequential()
        for i in range(nLayers):
            self.theta_linear_list.add_module(f'theta_linear_{i}', nn.Linear(self.ebdDim, self.ebdDim))
            self.phi_linear_list.add_module(f'phi_linear_{i}', nn.Linear(self.ebdDim, self.ebdDim))
            self.rho_linear_list.add_module(f'rho_linear_{i}', nn.Linear(self.ebdDim, self.ebdDim))
            self.psi_linear_list.add_module(f'psi_linear_{i}', nn.Linear(self.ebdDim, self.ebdDim))

        self.interactedLinear_1 = nn.Linear(ebdDim, ebdDim)
        self.feedFC = nn.Sequential(nn.Linear(ebdDim, ebdDim), nn.Linear(ebdDim, ebdDim), nn.LeakyReLU())
        self.AGI_1 = AGI(ebdDim=ebdDim, nLayers=nLayers)
        self.AGI_2 = AGI(ebdDim=ebdDim, nLayers=nLayers)

    def forward(self, input_1, input_2):
    
        interactedOut_1 = torch.zeros_like(input_1)
        interactedOut_2 = torch.zeros_like(input_2)
        
        for i in range(self.nLayers):
        
            theta = self.theta_linear_list[i](input_1)
            rho = self.rho_linear_list[i](input_1)
            
            phi = self.phi_linear_list[i](input_2)
            psi = self.psi_linear_list[i](input_2)
            
            affMat = torch.bmm(theta, torch.transpose(phi, -1, -2)) * (float(self.ebdDim) ** -0.5)
            
            interactedOut_1 += torch.bmm(affMat, rho)
            interactedOut_2 += torch.bmm(affMat, psi)
        
        resOut1 = F.normalize(input_1 + self.interactedLinear_1(interactedOut_1 / self.nLayers))
        resOut2 = F.normalize(input_2 + self.interactedLinear_1(interactedOut_2 / self.nLayers))
        
        feedOut1 = self.feedFC(resOut1)
        feedOut2 = self.feedFC(resOut2)
        
        selfOut_1 = self.AGI_1(feedOut1)
        selfOut_2 = self.AGI_1(feedOut2)
        
        crossOut = self.AGI_2(selfOut_1 + selfOut_2)

        return crossOut

class MIF(nn.Module):

    def __init__(self, input_dims, hidden_dims, output_dims, dropout):
    
        super(MIF, self).__init__()

        # Params
        self.nLayers = 5
        self.ebdDim = 4
        self.featDim = 80

        self.gene_in = input_dims[0]  
        self.path_in = input_dims[1]  
        self.cona_in = input_dims[2]  

        self.gene_hidden = hidden_dims[0]  
        self.path_hidden = hidden_dims[1]  
        self.cona_hidden = hidden_dims[2]  
        self.cox_hidden = hidden_dims[3]  

        self.label_dim = output_dims[2]
        
        self.cox_prob = dropout  

        self.norm = nn.BatchNorm1d(3*self.featDim)
 
        # Framework
        self.encoder_gene = EbdLayer(self.gene_in, self.gene_hidden)
        self.encoder_path = EbdLayer(self.path_in, self.path_hidden)
        self.encoder_cona = EbdLayer(self.cona_in, self.cona_hidden)

        self.g_uAGI = AGI(ebdDim=self.ebdDim, nLayers=self.nLayers)
        self.p_uAGI = AGI(ebdDim=self.ebdDim, nLayers=self.nLayers)
        self.c_uAGI = AGI(ebdDim=self.ebdDim, nLayers=self.nLayers)

        self.gp_bACGI = ACGI(ebdDim=self.ebdDim, nLayers=self.nLayers)
        self.pc_bACGI = ACGI(ebdDim=self.ebdDim, nLayers=self.nLayers)
        self.cg_bACGI = ACGI(ebdDim=self.ebdDim, nLayers=self.nLayers)

        self.gpc_tACGI = ACGI(ebdDim=self.ebdDim, nLayers=self.nLayers)
        self.pcg_tACGI = ACGI(ebdDim=self.ebdDim, nLayers=self.nLayers)
        self.cgp_tACGI = ACGI(ebdDim=self.ebdDim, nLayers=self.nLayers)


        # Predictor

        # For LGG
        encoder1 = nn.Sequential(nn.Linear(3*self.featDim, 384), nn.Tanh(), nn.Dropout(p=self.cox_prob))
        encoder2 = nn.Sequential(nn.Linear(384, 128), nn.Tanh(), nn.Dropout(p=self.cox_prob))
        encoder3 = nn.Sequential(nn.Linear(128, 64), nn.Tanh(), nn.Dropout(p=self.cox_prob))
        self.encoder = nn.Sequential(encoder1, encoder2,
                                     encoder3)
        # For BRCA
        # encoder1 = nn.Sequential(nn.Linear(3*self.featDim, self.cox_hidden), nn.Tanh(), nn.Dropout(p=self.cox_prob))
        # encoder2 = nn.Sequential(nn.Linear(self.cox_hidden, 64), nn.Tanh(), nn.Dropout(p=self.cox_prob))
        # self.encoder = nn.Sequential(encoder1, encoder2)

        self.classifier = nn.Sequential(nn.Linear(64, self.label_dim), nn.Sigmoid())
        self.output_range = Parameter(torch.FloatTensor([4]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-2]), requires_grad=False)


    def cycMat(self, input_tensor):
        """
        input_tensor is of shape (bsz, feat_dim)
        output_tensor is expected of shape (bsz, feat_dim, feat_dim)"""
        
        output_tensor = input_tensor
        
        for i in range(self.ebdDim-1):
            new_tensor = torch.cat((input_tensor[:, :, -i-1:], input_tensor[:, :, : -i-1]), dim=-1)
            output_tensor = torch.cat((output_tensor, new_tensor), dim=-2)
            
        return torch.transpose(output_tensor, -1, -2)


    def forward(self, x1, x2, x3):
    
        gene_feature = self.cycMat(self.encoder_gene(x1))
        path_feature = self.cycMat(self.encoder_path(x2))
        cona_feature = self.cycMat(self.encoder_cona(x3))

        g = self.g_uAGI(gene_feature)
        p = self.p_uAGI(path_feature)
        c = self.c_uAGI(cona_feature)
        uniModal = g + p + c
        
        gp = self.gp_bACGI(g, p)
        pc = self.pc_bACGI(p, c)
        cg = self.cg_bACGI(c, g)
        biModal = gp + pc + cg
        
        gpc = self.gpc_tACGI(gp, c)
        pcg = self.pcg_tACGI(pc, g)
        cgp = self.cgp_tACGI(cg, p)
        triModal = gpc + pcg + cgp
        
        fusion = torch.cat((uniModal,
                            biModal,
                            triModal
                            ), -2)

        fusion = self.norm(torch.sum(fusion, dim=-1))
        code = self.encoder(fusion)
        out = self.classifier(code)
        out = out * self.output_range + self.output_shift

        return out, code
        
