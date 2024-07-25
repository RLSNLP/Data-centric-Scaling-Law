import torch
import numpy as np
from tqdm import tqdm
import math
import pandas as pd
import csv
import random

random.seed(0)

def loss(inp, params):

    a, b, c, e, f, g = params[0], params[1], params[2], params[3], params[4], params[5]

    pre_lse = torch.stack([1 - a*torch.log((c*inp[:, 0] + e*inp[:, 2] + f*inp[:, 3]) * inp[:, 4] + g*10e12), b.expand((inp.shape[0]))])

    post_lse = torch.logsumexp(pre_lse, dim=0)
    huber_loss = torch.nn.functional.huber_loss(post_lse, torch.log(inp[:, 5]), delta=1e-3, reduction='none')
    return huber_loss.sum()

def minimize_loss(inp, init_params=[0.28, 6, 0.28, 0.28, 0.28], steps=50):
    params = torch.nn.Parameter(data=torch.Tensor(init_params))

    lbfgs = torch.optim.LBFGS([params],
                    lr=1e-1,
                    history_size=10,
                    max_iter=20,
                    line_search_fn="strong_wolfe")

    def closure():
        lbfgs.zero_grad()
        l = loss(inp, params)

        l.backward()
        return l

    for i in range(steps):
        l = lbfgs.step(closure)
    return l, params

min_loss = 1e9

file1 = open(r'results.csv', 'r')
csv_reader = csv.reader(file1)

store = []

for idx, line in enumerate(csv_reader):
    if idx == 0:
        continue
    tmp_line = []
    for id, element in enumerate(line):
        if id == 4:
            tmp_line.append(eval(element)*10e9)
            continue
        tmp_line.append(eval(element))
    store.append(tmp_line)

print(len(store))
random.shuffle(store)
inp = torch.tensor(store)
print(inp.shape[0])

inp.require_grad = True

addition = []

for c in tqdm(np.linspace(0, 2, 8)):
    for e in np.linspace(0, 2, 8):
        for f in np.linspace(0, 2, 8):
            for b in np.linspace(0, 2, 4):
                for a in np.linspace(0, 2, 4):
                    for g in np.linspace(1, 3, 4):
                        # print(a, b, c, e, f)
                        l, params = minimize_loss(inp, [a, b, c, e, f, g])
                        tmp = params.detach().numpy()[2]
                        if l < min_loss:
                            min_loss = l
                            best_params = params.detach().numpy()

print(addition)
print("Samples: ", len(inp))
print("Min Loss: ", min_loss)
print("a, b, c, e, f, g: ", list(best_params))

file1.close()