import torch.optim
import torch.nn as nn
import numpy as np

from FrEIA.framework import *
from FrEIA.modules import *
from extra_modules import *
import data
import config as c
import cond_net

import functools

partial = functools.partial

c.colorize

if c.colorize:
    nodes = [InputNode(3, *c.img_dims, name='inp')]
else:
    nodes = [InputNode(*c.img_dims, name='inp')]

if c.colorize:
    cond_size = 1 + c.cond_width
    cond_node = ConditionNode(cond_size, *c.img_dims)
else:
    cond_size = 10
    cond_node = ConditionNode(cond_size)

if c.colorize:
    for i in range(c.col_n_blocks_conv):
        nodes.append(Node([nodes[-1].out0], PermuteRandom, {'seed':i}, name=F'permute_{i}'))
        nodes.append(Node([nodes[-1].out0], GLOWCouplingBlock, {'clamp':c.col_clamping, 'subnet_constructor':partial(F_fully_conv,
                                                              kernel_size=1, channels_hidden=c.col_internal_width_conv)
                                                              },
                                                              conditions=cond_node, name=F'conv_{i}'))


    nodes.append(Node([nodes[-1].out0], Flatten, {}, name='flatten'))

    for i in range(c.col_n_blocks):
        nodes.append(Node([nodes[-1].out0], PermuteRandom, {'seed':i}, name=F'permute_{i}'))
        nodes.append(Node([nodes[-1].out0], GLOWCouplingBlock, {'clamp':c.col_clamping,
                                                                  'subnet_constructor':partial(F_fully_shallow,
                                                                  dropout=c.fc_dropout, internal_size=c.col_internal_width)
                                                                  },
                                                              name=F'fc_{i}'))

else:
    nodes.append(Node([nodes[-1].out0], Flatten, {}, name='flatten'))
    for i in range(c.cond_n_blocks):
        nodes.append(Node([nodes[-1].out0], PermuteRandom, {'seed':i}, name=F'permute_{i}'))
        nodes.append(Node([nodes[-1].out0], GLOWCouplingBlock, {'clamp':c.cond_clamping,'subnet_constructor':partial(F_fully_connected,
                                                              dropout=c.fc_dropout, internal_size=c.cond_internal_width)
                                                              },
                                                              conditions=cond_node,
                                                              name=F'fc_{i}'))


nodes.append(OutputNode([nodes[-1].out0], name='out'))
nodes.append(cond_node)

def init_model(mod):
    for key, param in mod.named_parameters():
        split = key.split('.')
        if param.requires_grad:
            param.data = c.init_scale * torch.randn(param.data.shape).cuda()
            if split[3][-1] == '3': # last convolution in the coeff func
                param.data.fill_(0.)


def optim_step():
    optim.step()
    optim.zero_grad()

def save(name):
    save_dict = {'opt':optim.state_dict(),
                 'net':model.state_dict()}
    if c.colorize:
        save_dict['cond'] = cond_net.model.state_dict()

    torch.save(save_dict, name)

def load(name):
    state_dicts = torch.load(name)
    model.load_state_dict(state_dicts['net'])
    if c.colorize:
        cond_net.model.load_state_dict(state_dicts['cond'])
    try:
        optim.load_state_dict(state_dicts['opt'])
    except ValueError:
        print('Cannot load optimizer for some reason or other')

model = ReversibleGraphNet(nodes, verbose=False)
model.cuda()
init_model(model)
model.to('cpu')

params_trainable = list(filter(lambda p: p.requires_grad, model.parameters()))

gamma = (c.decay_by)**(1./c.n_epochs)
optim = torch.optim.Adam(params_trainable, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)
weight_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=gamma)
