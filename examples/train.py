import numpy as np
import torch
from chester import logger
import os
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
import json

def MLP(channels):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU())
        for i in range(1, len(channels))
    ])

def run_task(vv, log_dir=None, exp_name=None):
    # create a logging dir for chester to write logs 
    if log_dir or logger.get_dir() is None:
        logger.configure(dir=log_dir, exp_name=exp_name, format_strs=['csv'])
    logdir = logger.get_dir()
    assert logdir is not None
    os.makedirs(logdir, exist_ok=True)

    # dump all variant parameters
    with open(os.path.join(logger.get_dir(), 'variant.json'), 'w') as f:
        json.dump(vv, f, indent=2, sort_keys=True)

    # you can also use wandb for logging
    # if vv['use_wandb']:
    #     import wandb
    #     vv['wandb_group'] = None
    #     vv.init(project="xxx", name=exp_name, entity="xxx", config=vv, settings=wandb.Settings(start_method='fork'))

    # do the actual task
    device = torch.device("cuda:{}".format(vv['cuda_id']))
    target = torch.from_numpy(np.random.rand(100, 3)).float()
    input = torch.from_numpy(np.random.rand(100, 7)).float()
    mlp = MLP([7, 128, 128, 3]) 
    optimizer = torch.optim.Adam(mlp.parameters(), lr=vv['lr'])

    for epoch in range(vv['epoch']):
        predict = mlp(input)
        loss = torch.mean((target - predict)**2)

        logger.record_tabular("loss", loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        logger.dump_tabular()
        print("{}/{}, Loss: {}".format(epoch, vv['epoch'], loss.item()))

    

