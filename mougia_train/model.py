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
    def __init__(self, graph, vocab_size, d_model=256, ticks=3, alpha=0.5, learn_edges=True, readout_hidden=512,
                 learn_tau=False, ctx_drive=False, dropout=0.0, interface="type", sep_id=5):
        """learn_tau: every neuron learns its own integration rate α_i ∈ (0.05, 1) — slow neurons can hold memory.
        ctx_drive: the input neurons also receive a running summary (mean embedding) of everything heard so far in
        the sequence, like a lingering smell, so the call stays present while the reply is produced.
        interface="neuron" (v4): every olfactory receptor neuron and every descending neuron is its own channel
        (fair per-type counts, identical in both sexes), and the fly gets EARS: tokens before <sep> (the rival's
        call) arrive through Johnston's-organ auditory neurons, its own words through the olfactory neurons, and
        while it answers the auditory neurons keep receiving an echo (mean) of the call."""
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
        self.alpha_logit = nn.Parameter(torch.full((n, 1), math.log((alpha - 0.05) / (1 - alpha)))) if learn_tau else None
        self.bias = nn.Parameter(torch.zeros(n, 1))

        self.interface, self.sep_id = interface, sep_id
        if interface == "neuron":
            self.register_buffer("in_idx", torch.tensor(graph["input_neurons"], dtype=torch.int64))
            self.register_buffer("au_idx", torch.tensor(graph["audio_neurons"], dtype=torch.int64))
            self.register_buffer("out_idx", torch.tensor(graph["output_neurons"], dtype=torch.int64))
            self.n_in, self.n_au, self.n_out = len(self.in_idx), len(self.au_idx), len(self.out_idx)
            self.embed = nn.Embedding(vocab_size, d_model)
            self.to_orn = nn.Linear(d_model, self.n_in)      # own syllable -> smell
            self.to_jo = nn.Linear(d_model, self.n_au)       # heard syllable -> hearing
            self.to_echo = nn.Linear(d_model, self.n_au, bias=False)  # echo of the call while answering
            self.ctx_to_orn = None
            self.in_gain = nn.Parameter(torch.tensor(2.0))
            self._build_readout(readout_hidden, vocab_size, dropout)
            nn.init.normal_(self.embed.weight, std=0.5)
            return
        # Interface (v1–v3): ORN types in, DN types out (same type lists for both sexes).
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
        self.ctx_to_orn = nn.Linear(d_model, self.n_in) if ctx_drive else None
        self.in_gain = nn.Parameter(torch.tensor(2.0))
        self._build_readout(readout_hidden, vocab_size, dropout)
        nn.init.normal_(self.embed.weight, std=0.5)

    def _build_readout(self, readout_hidden, vocab_size, dropout):
        if dropout:
            self.readout = nn.Sequential(nn.LayerNorm(self.n_out), nn.Dropout(dropout), nn.Linear(self.n_out, readout_hidden),
                                         nn.GELU(), nn.Dropout(dropout), nn.Linear(readout_hidden, vocab_size))
        else:
            self.readout = nn.Sequential(nn.LayerNorm(self.n_out), nn.Linear(self.n_out, readout_hidden), nn.GELU(),
                                         nn.Linear(readout_hidden, vocab_size))

    # ---- structure ------------------------------------------------------------------------------
    def weights(self):
        """Dense vector of connection strengths in CSR order (the sparse matrix is rebuilt from it inside each step,
        so gradients accumulate densely instead of as ever-growing sparse tensors)."""
        return self.w0 if self.edge_logscale is None else self.w0 * torch.exp(self.edge_logscale.clamp(-6, 6))

    def init_state(self, batch):
        return torch.zeros(self.n, batch, device=self.crow.device)

    # ---- one syllable ---------------------------------------------------------------------------
    def rates(self):
        return 0.05 + 0.95 * torch.sigmoid(self.alpha_logit) if self.alpha_logit is not None else self.alpha

    def _step(self, h, x, w, ctx=None, heard=None, echo=None):
        """h: (N, B) state; x: (B, d) syllable embedding; w: connection strengths.
        neuron interface: heard (B,) 1 = this syllable is the rival's (→ hearing), 0 = own (→ smell);
        echo (B, d) = mean embedding of the call heard so far. Returns new state and the readout features."""
        S = _Structure(self)
        a = self.rates()
        if self.interface == "neuron":
            own = 1.0 - heard
            orn = self.to_orn(x) * own[:, None]
            jo = self.to_jo(x) * heard[:, None] + self.to_echo(echo) * own[:, None]
            inp = torch.zeros_like(h).index_add(0, self.in_idx, (orn * self.in_gain).T)
            inp = inp.index_add(0, self.au_idx, (jo * self.in_gain).T)
        else:
            d = self.to_orn(x)
            if self.ctx_to_orn is not None and ctx is not None:
                d = d + self.ctx_to_orn(ctx)
            drive = (d * self.in_gain).T  # (n_in, B)
            inp = torch.zeros_like(h).index_add(0, self.in_idx, drive[self.in_type])
        inp = inp + self.bias
        for _ in range(self.ticks):
            rec = _ConnectomeMM.apply(w, torch.tanh(h), S)
            h = (1 - a) * h + a * (self.rho * rec + inp)
        act = torch.tanh(h[self.out_idx])  # (n_dn, B)
        if self.interface == "neuron":
            return h, act.T
        pooled = torch.zeros(self.n_out, h.shape[1], device=h.device).index_add(0, self.out_type, act) / self.out_count[:, None]
        return h, pooled.T

    def forward(self, tokens, h=None, grad_ckpt=True):
        """tokens: (B, L) → logits (B, L, V), final state."""
        B, L = tokens.shape
        h = self.init_state(B) if h is None else h
        W = self.weights()
        x = self.embed(tokens)
        ctx = torch.cumsum(x, 1) / torch.arange(1, L + 1, device=x.device)[None, :, None] if self.ctx_to_orn is not None else None
        heard = echo = None
        if self.interface == "neuron":
            # Before the first <sep> = the rival's call (heard); rows without <sep> are all the fly's own words.
            is_sep = tokens == self.sep_id
            heard = ((torch.cumsum(is_sep.long(), 1) == 0) & is_sep.any(1, keepdim=True)).float()
            cnt = torch.cumsum(heard, 1).clamp_min(1)[..., None]
            echo = torch.cumsum(x * heard[..., None], 1) / cnt
        outs = []
        for t in range(L):
            c = ctx[:, t] if ctx is not None else None
            hd = heard[:, t] if heard is not None else None
            ec = echo[:, t] if echo is not None else None
            if grad_ckpt and self.training:
                h, o = checkpoint(self._step, h, x[:, t], W, c, hd, ec, use_reentrant=False)
            else:
                h, o = self._step(h, x[:, t], W, c, hd, ec)
            outs.append(o)
        return self.readout(torch.stack(outs, 1)), h

    def n_params(self):
        return {name: p.numel() for name, p in self.named_parameters()}

    @torch.no_grad()
    def generate_batch(self, prompt, n=16, max_new=70, temperature=0.8, top_k=40, ban_ids=(), verse=None, eos_id=3, nl_id=4):
        """Sample n continuations of one prompt in parallel. Returns (list of id lists, mean log-prob per sample).

        verse: optional dict(syllable=BoolTensor[V], word_start=BoolTensor[V], min_syl=13, max_syl=16) — enforces a
        two-line couplet: no line break before min_syl syllables, after max_syl only the current word may finish,
        line 1 must end with <nl> and line 2 with <eos>. The model still chooses every word."""
        self.eval()
        dev = self.crow.device
        w = self.weights()
        h = self.init_state(1)
        o = None
        csum, cnt = 0, 0
        hsum, hcnt = torch.zeros(1, self.embed.embedding_dim, device=dev), 0
        listening = self.sep_id in prompt  # the call is heard until <sep>
        for t in prompt:
            e = self.embed(torch.tensor([t], device=dev))
            csum, cnt = csum + e, cnt + 1
            if listening and t == self.sep_id:
                listening = False
            hd = torch.tensor([1.0 if listening else 0.0], device=dev)
            if listening:
                hsum, hcnt = hsum + e, hcnt + 1
            h, o = self._step(h, e, w, csum / cnt, hd, hsum / max(hcnt, 1))
        h, o = h.repeat(1, n), o.repeat(n, 1)
        csum = csum.repeat(n, 1)
        echo = (hsum / max(hcnt, 1)).repeat(n, 1)
        own = torch.zeros(n, device=dev)
        seqs = [[] for _ in range(n)]
        logp = torch.zeros(n, device=dev)
        count = torch.zeros(n, device=dev)
        line = torch.zeros(n, dtype=torch.long, device=dev)
        syl_n = torch.zeros(n, dtype=torch.long, device=dev)
        done = torch.zeros(n, dtype=torch.bool, device=dev)
        prev_ns = torch.ones(n, dtype=torch.bool, device=dev)  # previous token carried no syllable (start counts as one)
        V = self.readout[-1].out_features
        if verse:
            is_syl = verse["syllable"].to(dev)
            is_ws = verse["word_start"].to(dev)
            lo, hi = verse.get("min_syl", 13), verse.get("max_syl", 16)
        for _ in range(max_new):
            logits = self.readout(o) / max(temperature, 1e-5)
            if ban_ids:
                logits[:, list(ban_ids)] = -1e9
            if verse:
                short = (syl_n < lo)[:, None]
                ban = torch.zeros(n, V, dtype=torch.bool, device=dev)
                ban[:, nl_id] |= short[:, 0] | (line == 1)
                ban[:, eos_id] |= short[:, 0] | (line == 0)
                ban |= (syl_n >= hi)[:, None] & (is_ws & is_syl)[None, :]       # finish the current word only
                ban |= (syl_n >= hi + 2)[:, None] & is_syl[None, :]           # then end the line
                no_syl = ~is_syl
                no_syl[nl_id] = no_syl[eos_id] = False
                ban |= prev_ns[:, None] & no_syl[None, :]                     # no runs of consonant/punctuation tokens
                logits = logits.masked_fill(ban, -1e9)
            if top_k:
                v, _ = torch.topk(logits, top_k, dim=-1)
                logits = logits.masked_fill(logits < v[:, -1:], -1e9)
            probs = F.softmax(logits, -1)
            nxt = torch.multinomial(probs, 1)[:, 0]
            lp = torch.log(probs.gather(1, nxt[:, None])[:, 0].clamp_min(1e-12))
            live = ~done
            logp += lp * live
            count += live.float()
            for i in torch.nonzero(live & (nxt != eos_id))[:, 0].tolist():
                seqs[i].append(int(nxt[i]))
            if verse:
                syl_n = torch.where(nxt == nl_id, torch.zeros_like(syl_n), syl_n + is_syl[nxt].long())
                prev_ns = ~is_syl[nxt]
                line = line + (nxt == nl_id).long()
            done |= nxt == eos_id
            if verse:
                done |= line >= 2
            if bool(done.all()):
                break
            e = self.embed(nxt)
            csum, cnt = csum + e, cnt + 1
            h, o = self._step(h, e, w, csum / cnt, own, echo)
        return seqs, (logp / count.clamp_min(1)).tolist()

    @torch.no_grad()
    def generate(self, prompt, max_new=60, temperature=0.8, top_k=40, eos_id=3, ban_ids=()):
        """Single unconstrained sample (kept for compatibility)."""
        seqs, _ = self.generate_batch(prompt, n=1, max_new=max_new, temperature=temperature, top_k=top_k, ban_ids=ban_ids, eos_id=eos_id)
        return seqs[0]
