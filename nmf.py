from pathlib import Path
import torch
import numpy as np
import math
from tqdm import tqdm


def itakura_saito(V, Vhat):
    return torch.sum(V / (Vhat) - torch.log(V / (Vhat)) - 1)


def kullback_leibler(V, Vhat):
    return torch.sum(V * (torch.log(V / Vhat)) + (Vhat - V))


def beta_divergence(b, V, Vhat):
    if b == 0:
        return itakura_saito(V, Vhat)
    elif b == 1:
        return kullback_leibler(V, Vhat)

    return torch.sum( (1 / (b * (b - 1))) * (V ** b - Vhat ** b - b * Vhat ** (b - 1) * (V - Vhat)) )


def nndsvd(A, k):
    F, N = A.shape
    W = torch.zeros((F, k), device=A.device, dtype=A.dtype)
    H = torch.zeros((k, N), device=A.device, dtype=A.dtype)
    U, S, V = torch.svd(A)
    for j in range(k):
        x = U[:, j]
        y = V[:, j]
        xp = x * (x >= 0).type(A.dtype)
        xn = -x * (x < 0).type(A.dtype)
        yp = y * (y >= 0).type(A.dtype)
        yn = -y * (y < 0).type(A.dtype)
        xpnrm = torch.norm(xp)
        ypnrm = torch.norm(yp)
        mp = xpnrm * ypnrm
        xnnrm = torch.norm(xn)
        ynnrm = torch.norm(yn)
        mn = xnnrm * ynnrm
        if mp > mn:
            u = xp / xpnrm
            v = yp / ypnrm
            s = mp
        else:
            u = xn / xnnrm
            v = yn / ynnrm
            s = mn
        W[:, j] = torch.sqrt(S[j] * s) * u
        H[j, :] = torch.sqrt(S[j] * s) * v.T

    return W, H


class NMF:

    def __init__(self, F, K, b=0, m=0, robust_normalization=True, tol=1e-4, initialization='random',
                 dtype=torch.float32, device='cpu', keep_history=True):
        """
        This class implements beta-NMF with L1 sparsity constraint as presented in
        Le Roux, Jonathan, Felix J. Weninger, and John R. Hershey. "Sparse NMF–half-baked or well done?."
        Mitsubishi Electric Research Labs (MERL), Cambridge, MA, USA, Tech. Rep., no. TR2015-023 11 (2015): 13-15.

        A classic NMF algorithm along with the proposed using robust normalization are implemented and can be selected
        via the robust_normalization flag.
        The initialization can be either random or nndsvd, using the method described in
        Boutsidis, Christos, and Efstratios Gallopoulos.
        "SVD based initialization: A head start for nonnegative matrix factorization."
        Pattern recognition 41.4 (2008): 1350-1362.

        :param F: is the number of frequency bins used (first dim of W).
        :param K: is the number of dictionary bases used (second dim of W and first dim of H).
        :param b: is the beta-divergence to use. Special cases are 0: IS, 1: KL, 2: EUC but the continuous range can be
            used.
        :param m: is the L1 sparsity penalty weight.
        :param robust_normalization: choose between the classic algorithm and the "well-done" method.
        :param tol: is the relative cost difference threshold used for the stopping criterion.
        :param initialization: choose from 'random' or 'nndsvd' methods.
        :param dtype: select the torch datatype to use for calculations.
        :param device: select the device to be used for calculations ('cpu' or 'cuda').
        :param keep_history: When set to True, training stats are logged in self.history and can be accessed.
            self.history keeps track of the b-div cost, the sparsity cost and the total cost for each train iteration.
        """
        self.b = b  # beta divergence
        self.m = m  # sparsity penalty.
        self.robust_normalization = robust_normalization
        self.tol = tol
        self.dtype = dtype
        self.device = device
        if initialization in ['random', 'nndsvd']:
            self.initialization = initialization
        else:
            ValueError('self.initialization should be either random or nndsvd.')

        # self.W = torch.rand((F, K), dtype=self.dtype, device=self.device) + 1e-12
        self.W = torch.abs(torch.randn((F, K), dtype=self.dtype, device=self.device) + 1e-12)
        self.W /= torch.sum(self.W, dim=0, keepdim=True)

        self.keep_history = keep_history
        self.history = {
            'divergence': [],
            'sparsity': [],
            'total': []
        }

    def train(self, V):
        """
        Update W and H using V until convergence.

        :param V: The matrix to be factorized, with dims (F x N).
        :return: The (K x N) matrix H that occurs from the factorization.
        """
        with torch.no_grad():
            if self.initialization == 'random':
                H = torch.abs(1e-3 * torch.randn((self.K, V.shape[1]), dtype=self.dtype, device=self.device) + 1e-12)
            elif self.initialization == 'nndsvd':
                W, H = nndsvd(V, self.K)
                m = torch.mean(V) * 1e-2
                ix = W == 0
                W += ix.type(self.dtype) * torch.rand(ix.shape, device=self.device, dtype=self.dtype) * m + 1e-12
                ix = H == 0
                H += ix.type(self.dtype) * torch.rand(ix.shape, device=self.device, dtype=self.dtype) * m + 1e-12

                self.W = W.detach().clone()
                S = torch.norm(self.W, p=2, dim=0, keepdim=True)
                self.W /= S
                H *= S.T
            else:
                ValueError('self.initialization should be either random or nndsvd.')

            Vhat = self.W @ H
            cost_prev, diverg, sparsity = self.cost(V, Vhat, H)
            if self.keep_history:
                self.save_costs(diverg, sparsity, cost_prev)

            H *= (self.W.T @ (V * Vhat ** (self.b - 2))) / ((self.W.T @ (Vhat ** (self.b - 1))) + self.m)
            Vhat = self.W @ H
            if self.robust_normalization:
                A = (V * Vhat ** (self.b - 2)) @ H.T
                B = (Vhat ** (self.b - 1)) @ H.T
                self.W *= (A + self.W * torch.sum(self.W * B, dim=0, keepdim=True)) / \
                          (B + self.W * torch.sum(self.W * A, dim=0, keepdim=True))
                S = torch.norm(self.W, p=2, dim=0, keepdim=True)
                self.W /= S
                H *= S.T
            else:
                self.W *= ((V * Vhat ** (self.b - 2)) @ H.T) / ((Vhat ** (self.b - 1)) @ H.T)
                S = torch.sum(self.W, dim=0, keepdim=True)
                self.W /= S
                H *= S.T
            Vhat = self.W @ H
            cost_curr, diverg, sparsity = self.cost(V, Vhat, H)
            if self.keep_history:
                self.save_costs(diverg, sparsity, cost_curr)

            while np.abs(cost_prev - cost_curr) / cost_prev > self.tol:
                cost_prev = cost_curr
                H *= (self.W.T @ (V * Vhat ** (self.b - 2))) / ((self.W.T @ (Vhat ** (self.b - 1))) + self.m)
                Vhat = self.W @ H
                if self.robust_normalization:
                    A = (V * Vhat ** (self.b - 2)) @ H.T
                    B = (Vhat ** (self.b - 1)) @ H.T
                    self.W *= (A + self.W * torch.sum(self.W * B, dim=0, keepdim=True)) / \
                              (B + self.W * torch.sum(self.W * A, dim=0, keepdim=True))
                    S = torch.norm(self.W, p=2, dim=0, keepdim=True)
                    self.W /= S
                    H *= S.T
                else:
                    self.W *= ((V * Vhat ** (self.b - 2)) @ H.T) / ((Vhat ** (self.b - 1)) @ H.T)
                    S = torch.sum(self.W, dim=0, keepdim=True)
                    self.W /= S
                    H *= S.T

                Vhat = self.W @ H
                cost_curr, diverg, sparsity = self.cost(V, Vhat, H)
                if self.keep_history:
                    self.save_costs(diverg, sparsity, cost_curr)

        return H

    def infer(self, V):
        """
        Infer the values of H, given an (F x N) input V.

        :param V: An (F x N) matrix with non-negative values.
        :return: The (K x N) matrix H that optimizes the factorization WH ~= V based on the selected b-div.
        """
        with torch.no_grad():
            H = torch.rand((self.K, V.shape[1]), dtype=self.dtype, device=self.device)
            Vhat = self.W @ H
            cost_prev, diverg, sparsity = self.cost(V, Vhat, H)
            H *= (self.W.T @ (V * Vhat ** (self.b - 2))) / (self.W.T @ (Vhat ** (self.b - 1)) + self.m)
            Vhat = self.W @ H
            cost_curr, diverg, sparsity = self.cost(V, Vhat, H)
            while np.abs(cost_prev - cost_curr) / cost_prev > self.tol:
                cost_prev = cost_curr
                H *= (self.W.T @ (V * Vhat ** (self.b - 2))) / (self.W.T @ (Vhat ** (self.b - 1)) + self.m)
                Vhat = self.W @ H
                cost_curr, diverg, sparsity = self.cost(V, Vhat, H)

            return H

    def cost(self, V, Vhat, H):
        diverg = beta_divergence(self.b, V, Vhat).item()
        sparsity = 0
        if self.m > 0:
            sparsity = self.m * torch.sum(H).item()
        total = diverg + sparsity
        return total, diverg, sparsity

    def save_costs(self, divergence, sparsity, total):
        self.history['divergence'].append(divergence)
        self.history['sparsity'].append(sparsity)
        self.history['total'].append(total)

    def reset_history(self):
        self.history = {
            'divergence': [],
            'sparsity': [],
            'total': []
        }

    @property
    def F(self):
        return self.W.shape[0]

    @property
    def K(self):
        return self.W.shape[1]

    def to(self, device):
        self.device = device
        self.W = self.W.to(device)


def stack_frames(X, n_frames):
    """
    Stack consequent frames of X together to form "super-vectors".
    If X can't be split perfectly in its second dimension, the final frame
    will be repeated for the final supervector.
    :param X: The input matrix with dims F x N.
    :param n_frames: The number of frames contained in a "super-vector".
    :return: The matrix with the stacked frames.
    """
    F, N = X.shape
    rem = (n_frames - (N % n_frames)) % n_frames
    div = math.ceil(N / n_frames)
    A = np.append(X, X[:, -rem:], axis=1)
    B = np.zeros((n_frames * F, div), dtype=X.dtype)
    for i in range(div):
        for j in range(n_frames):
            B[j * F:(j + 1) * F, i] = A[:, i * n_frames + j]

    return B


def unstack_frames(X, n_frames):
    """
    Un-stack supervectors in X to the original size.
    :param X: The input matrix with dims n_frames * F x N / n_frames.
    :param n_frames: The number of frames contained in a "super-vector".
    :return: The matrix with un-stacked frames.
    """
    f, n = X.shape
    F = f // n_frames
    N = n * n_frames
    B = np.zeros((F, N), dtype=X.dtype)
    for i in range(n):
        for j in range(n_frames):
            B[:, i * n_frames + j] = X[j * F:(j + 1) * F, i]

    return B


def concat_dictionaries(dict_path, model_name='concat_dict'):
    dict_path = Path(dict_path).expanduser()
    model_paths = list(dict_path.glob('*.pt'))
    model = torch.load(model_paths[0])
    pbar = tqdm(model_paths[1:])
    pbar.set_description('Concatenating dictionaries')
    for model_path in pbar:
        new_W = torch.load(model_path).W
        model.W = torch.cat([model.W, new_W], dim=1)

    model.reset_history()
    torch.save(model, dict_path / f"{model_name}.pt")
    return model
