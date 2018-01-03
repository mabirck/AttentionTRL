import torch
import torch.nn as nn
import math
from torch.autograd import Variable
import torch.nn.functional as F

# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias
class att(nn.Module):
    def __init__(self, hidden_in, hidden_out):
        super(att, self).__init__()
        self.hidden_in = hidden_in
        self.hidden_out = hidden_out
        self.context_att = nn.Linear(self.hidden_in, self.hidden_out)
        self.hidden_att = nn.Linear(self.hidden_out, self.hidden_out, bias=False) # NO BIAS
        self.joint_att = nn.Linear(self.hidden_out, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, contexts, hidden):
        #print(contexts.size(), hidden.size(), "IN THE FORWARD GUY")
        ##################### -- CONTEXT ENCODED-- ########################
        #print(contexts.size())
        c = self.context_att(contexts)
        #print(c.size(), "THIS IS THE SIZE OF THE CONTEXT GUY")

        ###################################################################
        #----------------------------------------------------------------#
        ##################### -- HIDDEN ENCODED -- ######################
        #print(hidden.size(), "THE HIDDEN INSIDE ATTENTION")
        h = self.hidden_att(hidden)
        h = h.unsqueeze(1)
        #print("BEFORE REPEAT",h.size())
        h = h.repeat(1, 49, 1)
        #print(h.size(), "THIS IS THE SIZE OF THE HIDDEN GUY")
        #h = h.expand(49, 512)
        ###############################################################
        #print(c.size(), h.size())
        final = c + h
        final = F.tanh(final)

        alpha = self.joint_att(final)
        #print(alpha.size())
        alpha = alpha.squeeze(2)
        #print(alpha)
        alpha = self.softmax(alpha)
        #print("THIS IS FINAL", final.size(), "THIS IS alpha", alpha.size(), "AND THIS IS THE CONTEXT", contexts.size())
        alpha = alpha.unsqueeze(2)
        weighted_context = torch.sum((alpha * contexts), 1)
        #print("SHIIIITI I FINISHED ?????????????????",weighted_context.size())
        return weighted_context


class temporal_att(nn.Module):
    def __init__(self, hidden_in, hidden_out):
        super(temporal_att, self).__init__()
        self.hidden_out = hidden_out
        self.hidden_att = nn.Linear(self.hidden_out, 1, bias=False) # NO BIAS
        self.softmax = nn.Softmax(dim=1)

    def forward(self, hidden):
        #print(hidden.size(), "h before")
        hidden = hidden.view(-1, 8, 256)
        sz = hidden.size(0)
        hidden = hidden.permute(1, 0, 2)

        h = self.hidden_att(hidden)
        #print(h.size(), "h after")
        alpha = self.softmax(h)
        #print("THIS IS alpha", alpha.size(), "AND THIS IS THE HIDDEN", hidden.size())
        #print(alpha)
        weighted_context = alpha * hidden
        #print("THIS IS WC", weighted_context)
        weighted_context = torch.sum(weighted_context, dim=1)
        #print(weighted_context.size(), "HERE WE GO")
        weighted_context.unsqueeze_(1)
        weighted_context = weighted_context.repeat(1,sz,1)
        weighted_context = weighted_context.permute(0, 1, 2)
        weighted_context = weighted_context.view(-1, 256)
        return weighted_context



"""class att(nn.Module):
    def __init__(self, method, hidden_size):
        super(att, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        # update: initalizing with torch.rand is not a good idea
        # Better practice is to initialize with zero mean and 1/sqrt(n) standard deviation
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)
        # end of update
        self.softmax = nn.Softmax()
        self.USE_CUDA = False


    def forward(self, hidden, encoder_outputs):
        '''
        :param hidden:
            previous hidden state of the decoder, in shape (layers*directions,B,H)
        :param encoder_outputs:
            encoder outputs from Encoder, in shape (T,B,H)
        :return
            attention energies in shape (B,T)
        '''
        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)
        # For storing attention energies
        attn_energies = Variable(torch.zeros(this_batch_size, max_len))
        if self.USE_CUDA:
            attn_energies = attn_energies.cuda()
        H = hidden.repeat(max_len,1,1).transpose(0,1)
        encoder_outputs = encoder_outputs.transpose(0,1) # [B*T*H]
        attn_energies = self.score(H,encoder_outputs) # compute attention score
        return self.softmax(attn_energies).unsqueeze(1) # normalize with softmax

    def score(self, hidden, encoder_outputs):
        energy = self.attn(torch.cat([hidden, encoder_outputs], 2)) # [B*T*2H]->[B*T*H]
        energy = energy.transpose(2,1) # [B*H*T]
        v = self.v.repeat(encoder_outputs.data.shape[0],1).unsqueeze(1) #[B*1*H]
        energy = torch.bmm(v,energy) # [B*1*T]
        return energy.squeeze(1) #[B*T]"""

# A temporary solution from the master branch.
# https://github.com/pytorch/pytorch/blob/7752fe5d4e50052b3b0bbc9109e599f8157febc0/torch/nn/init.py#L312
# Remove after the next version of PyTorch gets release.
def orthogonal(tensor, gain=1):
    if tensor.ndimension() < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")

    rows = tensor.size(0)
    cols = tensor[0].numel()
    flattened = torch.Tensor(rows, cols).normal_(0, 1)

    if rows < cols:
        flattened.t_()

    # Compute the qr factorization
    q, r = torch.qr(flattened)
    # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
    d = torch.diag(r, 0)
    ph = d.sign()
    q *= ph.expand_as(q)

    if rows < cols:
        q.t_()

    tensor.view_as(q).copy_(q)
    tensor.mul_(gain)
    return tensor

def where(cond, x_1, x_2):
    return (cond * x_1) + ((1-cond) * x_2)
