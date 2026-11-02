'''Stage 2's other two sub-stages, alongside Collapse (`PrototypeGenerators.
py`): G (support self-context) and F (query-reads-support context) --
`training/PrototypicalRoutingHead/spec.md`'s pipeline is `pool -> G -> F ->
Collapse -> Stage 3`, each stage independently switchable (`identity` = off)
and independently instanced. Both G and F live HERE, in one file, the same
"one file per axis" organisation `PrototypeGenerators.py`/
`PrototypeRoutingHeads.py` already use -- they are two halves of ONE
architectural idea (Full Context Embeddings, Vinyals et al. 2016), always
discussed and built together, even though their call shapes differ (G
touches only support, F touches both support and query).

    ctx_support = support_context(support)               # G: [K,D] -> [K,D]
    ctx_query   = query_context(query, support_by_rung)    # F: [N,D],{rung:[K_r,D]} -> [N,D]

F reads whichever `support_by_rung` it is handed -- G's own output if G
ran, the raw pooled support if G is `identity` -- `episode_forward` decides
that by calling G before F, never by F importing or calling G itself. F
has NO OPINION about whether Collapse (the stage AFTER it) will later
run: it always sees the PRE-collapse, per-rung support, because Collapse
happening first would leave F with one member per rung, the meaningless
degenerate case its own docstring warns about.
'''
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Union

import torch
import torch.nn as nn


# ══════════════════════════════════════════════════════════════════════════
#  off switch for F -- G's own off switch is PLAIN `nn.Identity()`
#  (`PrototypeChoices._build_context_identity_support`), not a class of
#  its own here: `forward(support) -> support` unchanged is exactly what
#  `nn.Identity` already is, so writing it a second time would be a second
#  definition of it. F cannot reuse `nn.Identity` the same way:
#  its own call takes TWO positional args (`query`, `support_by_rung`),
#  `nn.Identity.forward` takes exactly one.
# ══════════════════════════════════════════════════════════════════════════

class IdentityQueryContext(nn.Module):
    '''F, switched off. `forward(query, support_by_rung) -> query`,
    unchanged -- `support_by_rung` is accepted and ignored, so `episode_
    forward` can call this exactly like the real thing.'''

    def forward(self, query: torch.Tensor, support_by_rung) -> torch.Tensor:
        return query


# ══════════════════════════════════════════════════════════════════════════
#  G -- support members see each other before anything else happens to them
# ══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class BiLstmContextConfig:
    in_dim: int = 1536


class BiLstmSupportContext(nn.Module):
    '''`g(x_i, S) = h-forward_i + h-backward_i + x_i` (Vinyals et al.
    2016's own FCE formula for support embeddings): a bidirectional LSTM
    runs once over the K support members as ONE sequence (order is
    arbitrary -- a biLSTM sees both directions, so no member is
    structurally privileged by where it happens to land in the sequence),
    forward and backward hidden states are SUMMED (not concatenated --
    `nn.LSTM(bidirectional=True)`'s own output is the concatenation,
    `[K, 2*D]`; this splits it back into two `[K,D]` halves and adds them,
    to match the paper's own `+` rather than inventing a projection layer
    the paper does not have), then added back to the ORIGINAL pooled
    vector as a residual, same reasoning every other residual in this
    project carries: if the LSTM has learned nothing useful yet, this
    starts out as the identity rather than an arbitrary transform.

    `forward(support) -> [K,D]`, SAME shape in and out -- this is G, not
    Collapse; nothing here reduces the K members to one vector.
    '''

    def __init__(self, cfg: BiLstmContextConfig):
        super().__init__()
        self.cfg = cfg
        self.lstm = nn.LSTM(cfg.in_dim, cfg.in_dim, batch_first=True,
                            bidirectional=True)

    def forward(self, support: torch.Tensor) -> torch.Tensor:
        x = support.unsqueeze(0)                          # [1, K, D]
        out, _ = self.lstm(x)                                # [1, K, 2D]
        fwd, bwd = out.split(self.cfg.in_dim, dim=-1)          # each [1, K, D]
        h = (fwd + bwd).squeeze(0)                              # [K, D]
        return h + support


# ══════════════════════════════════════════════════════════════════════════
#  F -- query reads the (G-contextualised) support set K times before
#  Stage 3 ever compares the two
# ══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class AttnLstmContextConfig:
    in_dim: int = 1536
    k_steps: int = 3


class AttnLstmQueryContext(nn.Module):
    '''Vinyals et al. 2016's own "attLSTM" / Full Context Embedding for
    the query, adapted from the Read-Process-Write set-reading process
    (Vinyals, Bengio, Kudlur, "Order Matters", 2015) their own FCE
    section points to. `cfg.k_steps` read steps: at each step, an
    `LSTMCell` takes the query's OWN base embedding as input (the SAME
    vector every step -- only the hidden/cell/read state changes) combined
    with the running hidden state PLUS the previous step's attention-read
    vector, then attends (dot-product, softmax) over EVERY support member
    across ALL rungs at once (flattened -- this is exactly the same
    "compare against everything together" premise `CosineLogSumExpHead`'s own
    Stage 3 kernel keeps) to produce this step's own read. After `k_steps`
    steps, the final hidden state plus the ORIGINAL query embedding
    (residual, same reasoning as G's own) is the contextualised query.

    `forward(query, support_by_rung) -> [N,D]`, SAME shape as `query` in
    -- this is F, not a classifier; no rung/class decision happens here,
    only a refinement of what the query vector itself looks like before
    Stage 3 ever sees it.

    EXACT wiring (how `h` and the read vector `r` combine as the LSTMCell's
    hidden state input) is this file's own adaptation, not something the
    paper pins down to one specific tensor operation -- the paper's own
    text and the "Order Matters" companion describe the mechanism, not a
    reference implementation. Revisit this specific composition if it
    turns out to matter empirically; the part that is NOT a judgement
    call is the premise itself (query reads support K times, attending
    over the full un-collapsed set each time).
    '''

    def __init__(self, cfg: AttnLstmContextConfig):
        super().__init__()
        self.cfg = cfg
        self.cell = nn.LSTMCell(cfg.in_dim, cfg.in_dim)

    def forward(self, query: torch.Tensor,
               support_by_rung: Union[torch.Tensor, Dict[float, torch.Tensor]]
               ) -> torch.Tensor:
        members = (support_by_rung if isinstance(support_by_rung, torch.Tensor)
                  else torch.cat(list(support_by_rung.values()), dim=0))  # [K_total, D]
        n = query.shape[0]
        h = torch.zeros(n, self.cfg.in_dim, dtype=query.dtype, device=query.device)
        c = torch.zeros(n, self.cfg.in_dim, dtype=query.dtype, device=query.device)
        r = torch.zeros(n, self.cfg.in_dim, dtype=query.dtype, device=query.device)
        kv = members.unsqueeze(0).expand(n, -1, -1)          # [N, K_total, D]
        for _ in range(self.cfg.k_steps):
            h, c = self.cell(query, (h + r, c))
            scores = torch.einsum('nd,nkd->nk', h, kv)          # [N, K_total]
            a = torch.softmax(scores, dim=-1)                     # [N, K_total]
            r = torch.einsum('nk,nkd->nd', a, kv)                 # [N, D]
        return h + query
