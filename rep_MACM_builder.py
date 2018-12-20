import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from lib_torch.data_utils import list_shuffle, load
from lib_torch.torch_utils import Dot, Gaussian_ker, GenDotM, cossim, MaxM_fromBatch
from lib_torch.torch_utils import MaxM_fromBatchTopk


class MultiMatch(nn.Module):
    """MultiMatch Model"""
    def __init__(self, BS, q_len, d_len, q_filt1, q_filt2, q_filt3,
                 q_stride1, q_stride2, q_stride3,
                 d_filt1, d_filt2, d_filt3, d_stride1, d_stride2, d_stride3,
                 intermat_topk, vocab_size,
                 emb_size, hidden_size,  dropout,
                 sim_type="Gaussian", preemb=False, preemb_path=''):
        """q_filt1: q conv1 window size
           q_filt2: q conv2 window size
           q_filt3: q conv3 window size
           d_filt1: d conv1 window size
           d_filt2: d conv2 window size
           d_filt3: d conv3 window size
           emb_size: input emb_size
           dropout: dropout rate
           sim_type: interaction function, chosen from "Gaussian, Dot, Gen_Dot(generalized dot)"
        """
        super(MultiMatch, self).__init__()
        self.BS = BS
        self.q_len = q_len
        self.d_len = d_len
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.q_filt1 = q_filt1
        self.q_filt2 = q_filt2
        self.q_filt3 = q_filt3
        self.q_stride1 = q_stride1
        self.q_stride2 = q_stride2
        self.q_stride3 = q_stride3
        self.d_filt1 = d_filt1
        self.d_filt2 = d_filt2
        self.d_filt3 = d_filt3
        self.d_stride1 = d_stride1
        self.d_stride2 = d_stride2
        self.d_stride3 = d_stride3
        self.intermat_topk = intermat_topk
        self.preemb = preemb
        self.sim_type = sim_type
        if sim_type == "Gaussian":
            self.sim = Gaussian_ker
        elif sim_type == "Dot":
            self.sim = Dot
        elif sim_type == "Cos":
            self.sim = cossim
        elif sim_type == "Gen_Dot":
            self.sim = GenDotM
        else:
            raise AttributeError("sim_type {} is invalid".format(sim_type))
        self.q_conv1 = nn.Conv1d(in_channels=emb_size, out_channels=hidden_size,
                                 kernel_size=q_filt1, stride=q_stride1, padding=0,
                                 dilation=1, bias=True)
        self.q_conv2 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size,
                                 kernel_size=q_filt2, stride=q_stride2, padding=0,
                                 dilation=1, bias=True)
        self.q_conv3 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size,
                                 kernel_size=q_filt3, stride=q_stride3, padding=0,
                                 dilation=1, bias=True)
        self.d_conv1 = nn.Conv1d(in_channels=emb_size, out_channels=hidden_size,
                                 kernel_size=d_filt1, stride=d_stride1, padding=0,
                                 dilation=1, bias=True)
        self.d_conv2 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size,
                                 kernel_size=d_filt2, stride=d_stride2, padding=0,
                                 dilation=1, bias=True)
        self.d_conv3 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size,
                                 kernel_size=d_filt3, stride=d_stride3, padding=0,
                                 dilation=1, bias=True)
        self.emb_mod = nn.Embedding(vocab_size, emb_size, padding_idx=0)
        if preemb is True:
            emb_data = load(preemb_path)
            self.emb_mod.weight = nn.Parameter(torch.from_numpy(emb_data))
        else:
            init_tensor = torch.cat((torch.zeros(1, emb_size),
                                    torch.randn(vocab_size - 1, emb_size).normal_(0, 5e-3)),
                                    dim=0)
            self.emb_mod.weight = nn.Parameter(init_tensor)
        self.alpha = nn.Parameter(torch.randn(4).normal_(0, 1e-2))

    def forward(self, q, d_pos, d_neg):
        """ apply rel
            q: LongTensor (BS, qlen)
            d: LongTensor (BS, dlen)
            returns R1, R2, R3: relevance of 3 level (BS,)
        """
        # mask out padding's variable embeddings
        q_mask = torch.ne(q.data, 0).unsqueeze(2).float()  # (BS, qlen, 1)
        q_mask = Variable(q_mask, requires_grad=False)
        d_pos_mask = torch.ne(d_pos.data, 0).unsqueeze(2).float()  # (BS, dlen, 1)
        d_pos_mask = Variable(d_pos_mask, requires_grad=False)
        d_neg_mask = torch.ne(d_neg.data, 0).unsqueeze(2).float()  # (BSm dlen, 1)
        d_neg_mask = Variable(d_neg_mask, requires_grad=False)
        q_emb = self.emb_mod(q) * q_mask  # (BS, qlen, emb_size)
        d_pos_emb = self.emb_mod(d_pos) * d_pos_mask  # (BS, dlen, emb_size)
        d_neg_emb = self.emb_mod(d_neg) * d_neg_mask  # (BS, dlen, emb_size)
        # do convs
        q_conved1 = F.relu(self.q_conv1(q_emb.transpose(1, 2)))  # (BS, hs, qLen1)
        q_conved2 = F.relu(self.q_conv2(q_conved1))  # (BS, hs, qLen2)
        q_conved3 = F.relu(self.q_conv3(q_conved2))  # (BS, hs, 1)
        d_pos_conved1 = F.relu(self.d_conv1(d_pos_emb.transpose(1, 2)))  # (BS, hs, dLen1)
        d_pos_conved2 = F.relu(self.d_conv2(d_pos_conved1))  # (BS, hs, dLen2)
        d_pos_conved3 = F.relu(self.d_conv3(d_pos_conved2))  # (BS, hs, 1)
        d_neg_conved1 = F.relu(self.d_conv1(d_neg_emb.transpose(1, 2)))  # (BS, hs, dLen1)
        d_neg_conved2 = F.relu(self.d_conv2(d_neg_conved1))  # (BS, hs, dLen2)
        d_neg_conved3 = F.relu(self.d_conv3(d_neg_conved2))  # (BS, hs, 1)
        # interactions matrices
        if self.sim_type == "Dot" or self.sim_type == "Cos":
            interact_pos_mat1 = self.sim(q_emb, d_pos_emb)  # (BS, qlen, dlen)
            interact_pos_mat2 = self.sim(q_conved1.transpose(1, 2),
                                         d_pos_conved1.transpose(1, 2))  # (BS, qLen1, dLen1)
            interact_pos_mat3 = self.sim(q_conved2.transpose(1, 2),
                                         d_pos_conved2.transpose(1, 2))  # (BS, qLen1, dLen2)
            interact_neg_mat1 = self.sim(q_emb, d_neg_emb)  # (BS, qlen, dlen)
            interact_neg_mat2 = self.sim(q_conved1.transpose(1, 2),
                                         d_neg_conved1.transpose(1, 2))  # (BS, qLen1, dLen1)
            interact_neg_mat3 = self.sim(q_conved2.transpose(1, 2),
                                         d_neg_conved2.transpose(1, 2))  # (BS, qLen1, dLen2)
        elif self.sim_type == "Gaussian":
            interact_pos_mat1 = self.sim(q_emb, d_pos_emb, 1.0)  # (BS, qlen, dlen)
            interact_pos_mat2 = self.sim(q_conved1.transpose(1, 2),
                                         d_pos_conved1.transpose(1, 2), 1.0)  # (BS, qLen1, dLen1)
            interact_pos_mat3 = self.sim(q_conved2.transpose(1, 2),
                                         d_pos_conved2.transpose(1, 2), 1.0)  # (BS, qLen1, dLen2)
            interact_neg_mat1 = self.sim(q_emb, d_neg_emb, 1.0)  # (BS, qlen, dlen)
            interact_neg_mat2 = self.sim(q_conved1.transpose(1, 2),
                                         d_neg_conved1.transpose(1, 2), 1.0)  # (BS, qLen1, dLen1)
            interact_neg_mat3 = self.sim(q_conved2.transpose(1, 2),
                                         d_neg_conved2.transpose(1, 2), 1.0)  # (BS, qLen1, dLen2)
        else:  # using GenDotM
            gendotM1 = self.sim(hidden_size=self.emb_size)
            gendotM2 = self.sim(hidden_size=self.hidden_size)
            gendotM3 = self.sim(hidden_size=self.hidden_size)
            gendotM4 = self.sim(hidden_size=self.hidden_size)
            interact_pos_mat1 = gendotM1(q_emb, d_pos_emb)  # (BS, qlen, dlen)
            interact_pos_mat2 = gendotM2(q_conved1.transpose(1, 2), d_pos_conved1.transpose(1, 2)) # (BS, qLen1, qLen2)
            interact_pos_mat3 = gendotM2(q_conved2.transpose(1, 2), d_pos_conved2.transpose(1, 2)) # (BS, qLen1, qLen2)
            interact_neg_mat1 = gendotM1(q_emb, d_neg_emb)  # (BS, qlen, dlen)
            interact_neg_mat2 = gendotM2(q_conved1.transpose(1, 2), d_neg_conved1.transpose(1, 2)) # (BS, qLen1, qLen2)
            interact_neg_mat3 = gendotM2(q_conved2.transpose(1, 2), d_neg_conved2.transpose(1, 2)) # (BS, qLen1, qLen2)
        # calculate intermediate level scores
        M1_pos = MaxM_fromBatchTopk(interact_pos_mat1, self.intermat_topk)  # (BS,)
        M1_neg = MaxM_fromBatchTopk(interact_neg_mat1, self.intermat_topk)  # (BS,)
        M2_pos = MaxM_fromBatchTopk(interact_pos_mat2, self.intermat_topk)  # (BS,)
        M2_neg = MaxM_fromBatchTopk(interact_neg_mat2, self.intermat_topk)  # (BS,)
        M3_pos = MaxM_fromBatchTopk(interact_pos_mat3, self.intermat_topk)  # (BS,)
        M3_neg = MaxM_fromBatchTopk(interact_neg_mat3, self.intermat_topk)  # (BS,)
        if self.sim_type == "Gaussian":
            M4_pos = self.sim(q_conved3.transpose(1, 2), d_pos_conved3.transpose(1, 2), 1.0)[:, 0, 0] # (BS, 1, 1) -> (BS,)
            M4_neg = self.sim(q_conved3.transpose(1, 2), d_neg_conved3.transpose(1, 2), 1.0)[:, 0, 0] # (BS, 1, 1) -> (BS,)
        elif self.sim_type == "Dot" or self.sim_type == "Cos":
            M4_pos = self.sim(q_conved3.transpose(1, 2), d_pos_conved3.transpose(1, 2))[:, 0, 0] # (BS, 1, 1) -> (BS,)
            M4_neg = self.sim(q_conved3.transpose(1, 2), d_neg_conved3.transpose(1, 2))[:, 0, 0] # (BS, 1, 1) -> (BS,)
        else:
            M4_pos = gendotM4(q_conved3.transpose(1, 2), d_pos_conved3.transpose(1, 2))[:, 0, 0] # (BS, 1, 1) -> (BS,)
            M4_neg = gendotM4(q_conved3.transpose(1, 2), d_neg_conved3.transpose(1, 2))[:, 0, 0] # (BS, 1, 1) -> (BS,)
        M_pos = torch.stack((M1_pos, M2_pos, M3_pos, M4_pos), dim=1)  # (BS, 4)
        beta_pos = F.softmax(self.alpha * M_pos, dim=1)  # (BS, 4)
        S_pos = torch.sum(beta_pos * M_pos, dim=1)  # (BS,)
        M_neg = torch.stack((M1_neg, M2_neg, M3_neg, M4_neg), dim=1)  # (BS, 3)
        beta_neg = F.softmax(self.alpha * M_neg, dim=1)  # (BS, 4)
        S_neg = torch.sum(beta_neg * M_neg, dim=1)  # (BS,)
        return S_pos, S_neg

if __name__ == '__main__':
    model = MultiMatch(BS=2, q_len=5, d_len=10, q_filt1=3, q_filt2=2, q_filt3=2,
                       q_stride1=1, q_stride2=1, q_stride3=1,
                       d_filt1=3, d_filt2=2, d_filt3=2,
                       d_stride1=1, d_stride2=2, d_stride3=1,
                       intermat_topk=2, vocab_size=12, emb_size=300,
                       hidden_size=100, dropout=0.1, sim_type="Cos",
                       preemb=False)
    q = Variable(torch.LongTensor([[1,2,3,4,0],[6,7,8,9,0]]), requires_grad=False)
    d_pos = Variable(torch.LongTensor([[1,2,3,4,0,0,0,0,0,0], [9,8,9,7,0,0,0,0,0,0]]), requires_grad=False)
    d_neg = Variable(torch.LongTensor([[1,2,3,4,0,0,0,0,0,0], [1,8,7,7,1,9,8,9,7,0]]), requires_grad=False)
    spos, sneg = model(q, d_pos, d_neg)
    print(spos)
    print(sneg)
