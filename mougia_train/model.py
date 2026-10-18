"""A language model whose hidden state is a fruit-fly brain.

Every hidden unit is a real neuron, and the recurrent weight matrix has exactly the fly's connections:
  w_ij = sign(j) · syn_ij / Σ_k syn_ik · exp(θ_ij)
- sign(j) comes from the presynaptic neuron's predicted neurotransmitter and never changes;
- syn_ij is the synapse count (inputs normalised per neuron, so every neuron sums to unit |input|);
- θ_ij is a learnable log-scale per connection (optional), so learning can strengthen or weaken an existing synapse
  but can never create a new one or flip excitation/inhibition.
Dynamics (rate model, Euler step α, `ticks` steps per syllable):
  h ← (1-α)·h + α·(ρ·W·tanh(h) + b + I_in)
Syllables enter as currents into the olfactory receptor neurons (one learned channel per ORN type, shared across
sexes) and are read out from the descending neurons, pooled per DN type (also shared), through a small MLP.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class _ConnectomeMM(torch.autograd.Function):
    """y = W·x for the fixed fly wiring, W given by per-edge strengths w (CSR order, rows = postsynaptic).
    Hand-written backward: PyTorch's built-in sparse backward materialises an N×N mask, which doesn't fit."""

    @staticmethod
    def forward(ctx, w, x, s):
        ctx.save_for_backward(w, x)
        ctx.s = s
        W = torch.sparse_csr_tensor(s.crow, s.col, w, (s.n, s.n))
        return torch.sparse.mm(W, x)

    @staticmethod
    def backward(ctx, gy):
        w, x = ctx.saved_tensors
        s = ctx.s
        gx = gw = None
        if ctx.needs_input_grad[1]:
            WT = torch.sparse_csr_tensor(s.crow_t, s.col_t, w[s.perm_t], (s.n, s.n))
            gx = torch.sparse.mm(WT, gy.contiguous())
        if ctx.needs_input_grad[0]:
            gw = torch.empty_like(w)
            step = max(1, (64 << 20) // max(1, gy.shape[1]))  # ~256 MB of float32 per chunk
            for a in range(0, len(w), step):
                b = min(len(w), a + step)
                gw[a:b] = (gy[s.row[a:b]] * x[s.col[a:b]]).sum(1)
        return gw, gx, None


class _Structure:
    """Holds the CSR structure (and its transpose) as plain attributes for the autograd function."""

    def __init__(self, module):
        self.m = module

    def __getattr__(self, k):
        return getattr(self.m, k)


def load_graph(path, wiring="connectome", max_edges=None, seed=0):
    """Load a processed connectome. wiring='random' keeps each edge's source, weight and sign but rewires its target."""
    g = torch.load(path)
    pre, post, syn = g["pre"], g["post"], g["syn"]
    if max_edges and len(syn) > max_edges:  # density matching: keep the strongest connections
        keep = torch.topk(syn, max_edges).indices.sort().values
        pre, post, syn = pre[keep], post[keep], syn[keep]
    if wiring == "random":
        gen = torch.Generator().manual_seed(seed)
        post = torch.randint(0, g["n"], (len(pre),), generator=gen)
        same = post == pre
        post[same] = (post[same] + 1) % g["n"]
    g.update(pre=pre, post=post, syn=syn, wiring=wiring)
    return g


class FlyBrainLM(nn.Module):
    def __init__(self, graph, vocab_size, d_model=256, ticks=3, alpha=0.5, learn_edges=True, readout_hidden=512):
        super().__init__()
        n = graph["n"]
        self.n, self.ticks, self.alpha = n, ticks, alpha
        pre, post, syn, sign = graph["pre"], graph["post"], graph["syn"], graph["sign"]

        # Fixed structure in CSR order (rows = postsynaptic neuron).
        order = torch.argsort(post * n + pre)
        pre, post, syn = pre[order], post[order], syn[order]
        in_total = torch.zeros(n).index_add_(0, post, syn).clamp_min(1.0)
        w0 = sign[pre] * syn / in_total[post]
        crow = torch.zeros(n + 1, dtype=torch.int64)
        crow[1:] = torch.cumsum(torch.bincount(post, minlength=n), 0)
        self.register_buffer("crow", crow)
        self.register_buffer("col", pre.contiguous())
        self.register_buffer("w0", w0)
        self.register_buffer("row", post.contiguous())
        perm_t = torch.argsort(pre * n + post)  # transpose: rows = presynaptic
        crow_t = torch.zeros(n + 1, dtype=torch.int64)
        crow_t[1:] = torch.cumsum(torch.bincount(pre, minlength=n), 0)
        self.register_buffer("perm_t", perm_t)
        self.register_buffer("crow_t", crow_t)
        self.register_buffer("col_t", post[perm_t].contiguous())
        self.edge_logscale = nn.Parameter(torch.zeros(len(w0))) if learn_edges else None
        self.rho = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.zeros(n, 1))

        # Interface: ORN types in, DN types out (same type lists for both sexes).
        in_groups = [v for v in graph["input_groups"].values()]
        out_groups = [v for v in graph["output_groups"].values()]
        self.register_buffer("in_idx", torch.tensor([i for g in in_groups for i in g], dtype=torch.int64))
        self.register_buffer("in_type", torch.tensor([t for t, g in enumerate(in_groups) for _ in g], dtype=torch.int64))
        self.register_buffer("out_idx", torch.tensor([i for g in out_groups for i in g], dtype=torch.int64))
        self.register_buffer("out_type", torch.tensor([t for t, g in enumerate(out_groups) for _ in g], dtype=torch.int64))
        self.register_buffer("out_count", torch.bincount(self.out_type, minlength=len(out_groups)).clamp_min(1).float())
        self.n_in, self.n_out = len(in_groups), len(out_groups)

        self.embed = nn.Embedding(vocab_size, d_model)
        self.to_orn = nn.Linear(d_model, self.n_in)
        self.in_gain = nn.Parameter(torch.tensor(2.0))
        self.readout = nn.Sequential(nn.LayerNorm(self.n_out), nn.Linear(self.n_out, readout_hidden), nn.GELU(),
                                     nn.Linear(readout_hidden, vocab_size))
        nn.init.normal_(self.embed.weight, std=0.5)

    # ---- structure ------------------------------------------------------------------------------
    def weights(self):
        """Dense vector of connection strengths in CSR order (the sparse matrix is rebuilt from it inside each step,
        so gradients accumulate densely instead of as ever-growing sparse tensors)."""
        return self.w0 if self.edge_logscale is None else self.w0 * torch.exp(self.edge_logscale.clamp(-6, 6))

    def init_state(self, batch):
        return torch.zeros(self.n, batch, device=self.crow.device)

    # ---- one syllable ---------------------------------------------------------------------------
    def _step(self, h, x, w):
        """h: (N, B) state; x: (B, d) syllable embedding; w: connection strengths. Returns new state and DN readout."""
        S = _Structure(self)
        drive = (self.to_orn(x) * self.in_gain).T  # (n_in, B)
        inp = torch.zeros_like(h).index_add(0, self.in_idx, drive[self.in_type])
        inp = inp + self.bias
        for _ in range(self.ticks):
            rec = _ConnectomeMM.apply(w, torch.tanh(h), S)
            h = (1 - self.alpha) * h + self.alpha * (self.rho * rec + inp)
        act = torch.tanh(h[self.out_idx])  # (n_dn, B)
        pooled = torch.zeros(self.n_out, h.shape[1], device=h.device).index_add(0, self.out_type, act) / self.out_count[:, None]
        return h, pooled.T

    def forward(self, tokens, h=None, grad_ckpt=True):
        """tokens: (B, L) → logits (B, L, V), final state."""
        B, L = tokens.shape
        h = self.init_state(B) if h is None else h
        W = self.weights()
        x = self.embed(tokens)
        outs = []
        for t in range(L):
            if grad_ckpt and self.training:
                h, o = checkpoint(self._step, h, x[:, t], W, use_reentrant=False)
            else:
                h, o = self._step(h, x[:, t], W)
            outs.append(o)
        return self.readout(torch.stack(outs, 1)), h

    def n_params(self):
        return {name: p.numel() for name, p in self.named_parameters()}

    @torch.no_grad()
    def generate(self, prompt, max_new=60, temperature=0.8, top_k=40, eos_id=3, ban_ids=()):
        """prompt: list[int]. Returns generated ids (without prompt)."""
        self.eval()
        dev = self.crow.device
        W = self.weights()
        h = self.init_state(1)
        o = None
        for t in prompt:
            h, o = self._step(h, self.embed(torch.tensor([t], device=dev)), W)
        out = []
        for _ in range(max_new):
            logits = self.readout(o)[0] / max(temperature, 1e-5)
            for b in ban_ids:
                logits[b] = -1e9
            if top_k:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[-1]] = -1e9
            nxt = int(torch.multinomial(F.softmax(logits, -1), 1))
            if nxt == eos_id:
                break
            out.append(nxt)
            h, o = self._step(h, self.embed(torch.tensor([nxt], device=dev)), W)
        return out
