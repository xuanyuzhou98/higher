import argparse
import time
import typing

import pandas as pd
import numpy as np
import scipy.stats
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import higher

from examples.support.miniimagenet_loaders import MiniImagenet
from examples.irevnet.models.iRevNet import iRevNet



def mean_confidence_interval(accs, confidence=0.95):
    n = accs.shape[0]
    m, se = np.mean(accs), scipy.stats.sem(accs)
    h = se * scipy.stats.t._ppf((1 + confidence) / 2, n - 1)
    return m, h


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument(
        '--k_spt', type=int, help='k shot for support set', default=5)
    argparser.add_argument(
        '--k_qry', type=int, help='k shot for query set', default=15)
    argparser.add_argument(
        '--task_num',
        type=int,
        help='meta batch size, namely task num',
        default=4)
    argparser.add_argument('--seed', type=int, help='random seed', default=1)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=84)
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    argparser.add_argument('--path', type=str, help='imagepath', default='/home/jiangshanli/higher/miniimagenet')

    args = argparser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    print(args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # config = [
    #     ('conv2d', [32, 3, 3, 3, 1, 0]),
    #     ('relu', [True]),
    #     ('bn', [32]),
    #     ('max_pool2d', [2, 2, 0]),
    #     ('conv2d', [32, 32, 3, 3, 1, 0]),
    #     ('relu', [True]),
    #     ('bn', [32]),
    #     ('max_pool2d', [2, 2, 0]),
    #     ('conv2d', [32, 32, 3, 3, 1, 0]),
    #     ('relu', [True]),
    #     ('bn', [32]),
    #     ('max_pool2d', [2, 2, 0]),
    #     ('conv2d', [32, 32, 3, 3, 1, 0]),
    #     ('relu', [True]),
    #     ('bn', [32]),
    #     ('max_pool2d', [2, 1, 0]),
    #     ('flatten', []),
    #     ('linear', [args.n_way, 32 * 5 * 5])
    # ]
    #
    # device = torch.device('cuda')
    # maml = Meta(args, config).to(device)
    # TODO
    net = iRevNet([18, 18, 18], [1, 2, 2], args.n_way, nChannels=[32, 128, 512], init_ds=0,
                  dropout_rate=0.1, affineBN=True, in_shape=[args.imgc, args.imgsz, args.imgsz], mult=4).to(device)

    meta_opt = optim.Adam(net.parameters(), lr=1e-3)

    log = []

    tmp = filter(lambda x: x.requires_grad, net.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(net)
    print('Total trainable tensors:', num)

    # batchsz here means total episode number
    mini = MiniImagenet(args.path, mode='train', n_way=args.n_way, k_shot=args.k_spt,
                        k_query=args.k_qry,
                        batchsz=500, resize=args.imgsz)
    mini_test = MiniImagenet(args.path, mode='test', n_way=args.n_way, k_shot=args.k_spt,
                             k_query=args.k_qry,
                             batchsz=100, resize=args.imgsz)

    for epoch in range(100):
        train(mini, net, device, meta_opt, epoch, log, args.task_num)
        test(mini_test, net, device, epoch, log, args.task_num)
        plot(log)

def train(mini, net, device, meta_opt, epoch, log, task_num):
    net.train()
    db = DataLoader(mini, task_num, shuffle=True, num_workers=1, pin_memory=True)

    for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db):
        start_time = time.time()
        x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)

        task_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)

        # Initialize the inner optimizer to adapt the parameters to
        # the support set.
        n_inner_iter = 5
        inner_opt = torch.optim.SGD(net.parameters(), lr=1e-1)

        qry_losses = []
        qry_accs = []
        meta_opt.zero_grad()
        for i in range(task_num):
            with higher.innerloop_ctx(
                    net, inner_opt, copy_initial_weights=False
            ) as (fnet, diffopt):
                # Optimize the likelihood of the support set by taking
                # gradient steps w.r.t. the model's parameters.
                # This adapts the model's meta-parameters to the task.
                # higher is able to automatically keep copies of
                # your network's parameters as they are being updated.
                for _ in range(n_inner_iter):
                    spt_logits = fnet(x_spt[i])[0]
                    spt_loss = F.cross_entropy(spt_logits, y_spt[i])
                    diffopt.step(spt_loss)

                # The final set of adapted parameters will induce some
                # final loss and accuracy on the query dataset.
                # These will be used to update the model's meta-parameters.
                qry_logits = fnet(x_qry[i])[0]
                qry_loss = F.cross_entropy(qry_logits, y_qry[i])
                qry_losses.append(qry_loss.detach())
                qry_acc = (qry_logits.argmax(
                    dim=1) == y_qry[i]).sum().item() / querysz
                qry_accs.append(qry_acc)

                # Update the model's meta-parameters to optimize the query
                # losses across all of the tasks sampled in this batch.
                # This unrolls through the gradient steps.
                qry_loss.backward()

        meta_opt.step()
        qry_losses = sum(qry_losses) / task_num
        qry_accs = 100. * sum(qry_accs) / task_num
        i = epoch + float(step) / 500
        iter_time = time.time() - start_time
        if step % 100 == 0:
            print(
                f'[Epoch {i:.2f}] Train Loss: {qry_losses:.2f} | Acc: {qry_accs:.2f} | Time: {iter_time:.2f}'
            )

        log.append({
            'epoch': i,
            'loss': qry_losses,
            'acc': qry_accs,
            'mode': 'train',
            'time': time.time(),
        })

def test(mini_test, net, device, epoch, log, task_num):
    # Crucially in our testing procedure here, we do *not* fine-tune
    # the model during testing for simplicity.
    # Most research papers using MAML for this task do an extra
    # stage of fine-tuning here that should be added if you are
    # adapting this code for research.
    net.train()

    qry_losses = []
    qry_accs = []

    db = DataLoader(mini_test, task_num, shuffle=True, num_workers=1, pin_memory=True)

    for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db):
        x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)
        task_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)

        # TODO: Maybe pull this out into a separate module so it
        # doesn't have to be duplicated between `train` and `test`?
        n_inner_iter = 5
        inner_opt = torch.optim.SGD(net.parameters(), lr=1e-1)

        for i in range(task_num):
            with higher.innerloop_ctx(net, inner_opt, track_higher_grads=False) as (fnet, diffopt):
                # Optimize the likelihood of the support set by taking
                # gradient steps w.r.t. the model's parameters.
                # This adapts the model's meta-parameters to the task.
                for _ in range(n_inner_iter):
                    spt_logits = fnet(x_spt[i])[0]
                    spt_loss = F.cross_entropy(spt_logits, y_spt[i])
                    diffopt.step(spt_loss)

                # The query loss and acc induced by these parameters.
                qry_logits = fnet(x_qry[i])[0].detach()
                qry_loss = F.cross_entropy(
                    qry_logits, y_qry[i], reduction='none')
                qry_losses.append(qry_loss.detach())
                qry_accs.append(
                    (qry_logits.argmax(dim=1) == y_qry[i]).detach())

    qry_losses = torch.cat(qry_losses).mean().item()
    qry_accs = 100. * torch.cat(qry_accs).float().mean().item()
    print(
        f'[Epoch {epoch + 1:.2f}] Test Loss: {qry_losses:.2f} | Acc: {qry_accs:.2f}'
    )
    log.append({
        'epoch': epoch + 1,
        'loss': qry_losses,
        'acc': qry_accs,
        'mode': 'test',
        'time': time.time(),
    })

def plot(log):
    # Generally you should pull your plotting code out of your training
    # script but we are doing it here for brevity.
    df = pd.DataFrame(log)

    fig, ax = plt.subplots(figsize=(6, 4))
    train_df = df[df['mode'] == 'train']
    test_df = df[df['mode'] == 'test']
    ax.plot(train_df['epoch'], train_df['acc'], label='Train')
    ax.plot(test_df['epoch'], test_df['acc'], label='Test')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(70, 100)
    fig.legend(ncol=2, loc='lower right')
    fig.tight_layout()
    fname = 'maml-accs.png'
    print(f'--- Plotting accuracy to {fname}')
    fig.savefig(fname)
    plt.close(fig)


if __name__ == '__main__':
    main()