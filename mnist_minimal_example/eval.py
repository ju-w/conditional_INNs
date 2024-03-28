import torch
import numpy as np
import matplotlib.pyplot as plt

import model
import data

cinn = model.MNIST_cINN(0)
cinn.cuda()
state_dict = {k:v for k,v in torch.load('output/mnist_cinn.pt').items() if 'tmp_var' not in k}
cinn.load_state_dict(state_dict)

cinn.eval()

def show_samples_(label):
    '''produces and shows cINN samples for a given label (0-9)'''

    N_samples = 100
    l = torch.cuda.LongTensor(N_samples)
    l[:] = label

    z = 1.0 * torch.randn(N_samples, model.ndim_total).cuda()

    with torch.no_grad():
        samples = cinn.reverse_sample(z, l)[0].cpu().numpy()
        samples = data.unnormalize(samples)

    full_image = np.zeros((28*10, 28*10))

    for k in range(N_samples):
        i, j = k // 10, k % 10
        full_image[28 * i : 28 * (i + 1),
                   28 * j : 28 * (j + 1)] = samples[k, 0]

    # full_image = np.clip(full_image, 0, 1)
    # plt.imshow(full_image, vmin=0, vmax=1, cmap='gray')

    r, c = 10, 10
    plt.close(label)
    fig = plt.figure(label, figsize=(10, 10))
    fig.suptitle(F'Generated digits for c={label}')

    axs = fig.subplots(r,c, squeeze=False)
    fig.set_constrained_layout(True)

    plt_c = {'cmap':'seismic_r', "vmin":-3, "vmax":3}

    for ix, (rr, cc) in enumerate(it_2d(r, c)):
        ax = axs[rr][cc]
        ax.imshow(samples[ix, 0], **plt_c)
        ax.axis('off')

    # plt.imshow(full_image)

    plt.savefig(f'images/label{label}a.png')

def show_samples(label):

    normal = torch.distributions.normal.Normal(torch.tensor([0.]*28*28),torch.tensor([1.,]*28*28))
    N_samples = 100
    l = torch.cuda.LongTensor(N_samples)
    l[:] = label

    z_ = torch.zeros(28*28).cuda()
    z = 1/5.0 * torch.randn(N_samples-1, model.ndim_total).cuda()
    z = torch.cat([z_[None], z])


    with torch.no_grad():
        samples, jac = cinn.reverse_sample(z, l)
        samples = samples.cpu().numpy()
        jac = jac.cpu().numpy()
        samples = data.unnormalize(samples)

    samples[0,0,0:5,0:5] = -1

    z = z.cpu()
    log_p = torch.sum(normal.log_prob(z),1)

    log_pj = log_p+jac
    # arg = torch.argsort(-log_p)
    arg = torch.argsort(-log_pj)


    z = z[arg]
    jac = jac[arg]
    samples = samples[arg]
    log_p = log_p[arg]
    log_pj = log_pj[arg]


    r, c = 10, 10
    plt.close('label')
    fig = plt.figure('label', figsize=(15, 15))
    fig.suptitle(F'Generated digits for c={label}')

    axs = fig.subplots(r,c, squeeze=False)
    fig.set_constrained_layout(True)

    plt_c = {'cmap':'seismic_r', "vmin":-3, "vmax":3}

    for ix, (rr, cc) in enumerate(it_2d(r, c)):
        ax = axs[rr][cc]
        ax.imshow(samples[ix, 0], **plt_c)
        axoff(ax)
        ax.set_xlabel(f'{log_pj[ix]:.1f}, {log_p[ix].item():.1f}')
    plt.savefig(f'images/_slabel{label}a.png')


def axoff(ax):
    ax.set_xticks([])
    ax.set_yticks([])

def it_2d(snd, fst):
    from itertools import product
    return product(range(snd), range(fst))


def val_loss():
    '''prints the final validiation loss of the model'''

    with torch.no_grad():
        z, log_j = cinn(data.val_x, data.val_l)
        nll_val = torch.mean(z**2) / 2 - torch.mean(log_j) / model.ndim_total

    print('Validation loss:')
    print(nll_val.item())

val_loss()

for i in range(10):
    show_samples(i)

plt.show()
