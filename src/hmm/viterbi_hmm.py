"""HMM + Viterbi route reconstruction on top of the classifier probabilities.

Idea in one paragraph
  The classifier scores each (GPS point, candidate arc) pair independently. A real vehicle cannot jump between
  unconnected roads from one second to the next, so we add that knowledge with a Hidden Markov Model:
    hidden state at time t = which candidate arc the vehicle is really on
    emission   = classifier probability that this candidate is correct   (how well the point fits the arc)
    transition = how plausible it is to go from arc i (at t-1) to arc j (at t), from road connectivity
  The Viterbi algorithm finds the single best sequence of arcs (highest total log-probability).

Transition model (all in log space, so products become sums and nothing underflows)
    same arc                      hops = 0 -> log T = 0
    reachable in h arcs (h<=H)    log T = -lam * h
    not reachable within H hops   log T = -disc_penalty      (a "disconnected jump" is allowed but expensive)
    time gap between the two GPS points > gap_s -> transitions carry no information (reset): any arc may follow.
  Observations with NO candidate arc are absent from the chain; the next observation just follows the previous one
  (and a long time gap triggers the reset above).

Parameters (see DEFAULT_PARAMS): lam, disc_penalty, max_hops, gap_s, emission_power.
  emission_power > 1 trusts the classifier more, < 1 trusts the road connectivity more.
"""
import numpy as np
import pandas as pd

DEFAULT_PARAMS = dict(lam=1.0, disc_penalty=8.0, max_hops=4, gap_s=30.0, emission_power=1.0)
EPS = 1e-6


class TripHMM:
    """Prepares one trip once (road graph + candidate groups) so that decode() can be called many times."""

    def __init__(self, arcs, cand):
        """arcs: (n_arcs, 2) array of (from_node, to_node). cand: DataFrame with columns obs, arc, prob, gps_t."""
        self.to_node = arcs[:, 1]
        self.out = {}                                   # node -> list of arcs that START at that node
        for arc_id, f in enumerate(arcs[:, 0]):
            self.out.setdefault(int(f), []).append(arc_id)
        self._reach_cache = {}

        cand = cand.sort_values(["obs", "arc"]).reset_index(drop=True)
        obs = cand["obs"].values
        first = np.r_[0, np.flatnonzero(np.diff(obs)) + 1]            # start row of each observation's block
        last = np.r_[first[1:], len(cand)]
        self.obs_ids = obs[first]
        self.groups = [(cand["arc"].values[a:b], cand["prob"].values[a:b], float(cand["gps_t"].values[a]))
                       for a, b in zip(first, last)]

    def _reach(self, arc, max_hops):
        """{arc: hops} for every arc reachable from `arc` by following the road direction in <= max_hops steps."""
        key = (arc, max_hops)
        if key not in self._reach_cache:
            dist = {arc: 0}
            frontier = [arc]
            for h in range(1, max_hops + 1):
                nxt = []
                for a in frontier:
                    for b in self.out.get(int(self.to_node[a]), ()):
                        if b not in dist:
                            dist[b] = h
                            nxt.append(b)
                frontier = nxt
            self._reach_cache[key] = dist
        return self._reach_cache[key]

    def decode(self, params=None):
        """Viterbi. Returns DataFrame(obs, arc) = most likely arc for every observation that has candidates."""
        p = {**DEFAULT_PARAMS, **(params or {})}
        n = len(self.groups)
        if n == 0:
            return pd.DataFrame(dict(obs=[], arc=[]))
        arcs0, prob0, _ = self.groups[0]
        delta = p["emission_power"] * np.log(np.clip(prob0, EPS, 1))   # best log-score of any path ending in each candidate
        back = [None]                                                 # back[k][j] = best previous candidate for candidate j
        for k in range(1, n):
            prev_arcs, _, prev_t = self.groups[k - 1]
            cur_arcs, cur_prob, cur_t = self.groups[k]
            emit = p["emission_power"] * np.log(np.clip(cur_prob, EPS, 1))
            if cur_t - prev_t > p["gap_s"]:
                # long GPS gap: we cannot say how the vehicle moved, so every previous arc is equally good
                b = np.full(len(cur_arcs), int(np.argmax(delta)))
                new = delta.max() + emit
            else:
                hops = np.empty((len(prev_arcs), len(cur_arcs)))
                for i, a in enumerate(prev_arcs):
                    r = self._reach(int(a), p["max_hops"])
                    hops[i] = [r.get(int(c), -1) for c in cur_arcs]    # -1 = not reachable
                log_t = np.where(hops >= 0, -p["lam"] * hops, -p["disc_penalty"])
                total = delta[:, None] + log_t                         # (previous candidate, current candidate)
                b = total.argmax(axis=0)
                new = total[b, np.arange(len(cur_arcs))] + emit
            back.append(b)
            delta = new
        # backtrack from the best final state
        idx = int(np.argmax(delta))
        chosen = np.empty(n, dtype=np.int64)
        for k in range(n - 1, -1, -1):
            chosen[k] = self.groups[k][0][idx]
            if k > 0:
                idx = int(back[k][idx])
        return pd.DataFrame(dict(obs=self.obs_ids, arc=chosen))


# ----------------------------------------------------------------------------- baselines and scoring
def classifier_only(cand):
    """Pick the highest-probability candidate per observation (no HMM)."""
    cand = cand.reset_index(drop=True)
    i = cand.groupby("obs")["prob"].idxmax()
    return cand.loc[i, ["obs", "arc"]].reset_index(drop=True)


def score_prediction(pred, ref_by_obs, n_obs, arcs, route=None):
    """Metrics for one trip.

    pred       DataFrame(obs, arc)
    ref_by_obs array of length n_obs: reference arc of each observation (-1 if unknown)
    n_obs      number of GPS observations (observations without any prediction count as wrong)
    arcs       (n_arcs, 2) from/to nodes, for route continuity
    route      reference route (list of arc ids) -> sequence-level precision / recall / F1 (optional)
    """
    pred = pred.sort_values("obs")
    a = pred.arc.values
    out = dict(n_obs=n_obs, correct=int((ref_by_obs[pred.obs.values] == a).sum()))
    out["top1"] = out["correct"] / n_obs
    if len(a) > 1:
        same = a[1:] == a[:-1]
        conn = arcs[a[:-1], 1] == arcs[a[1:], 0]
        out["cont_ok"], out["cont_n"] = int((same | conn).sum()), len(a) - 1
    else:
        out["cont_ok"], out["cont_n"] = 0, 0
    if route is not None:
        ps, rs = set(a.tolist()), set(np.asarray(route).tolist())
        inter = len(ps & rs)
        prec = inter / len(ps) if ps else 0.0
        rec = inter / len(rs) if rs else 0.0
        out["route_precision"], out["route_recall"] = prec, rec
        out["route_f1"] = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return out


def route_from_prediction(pred):
    """Predicted route = arcs in time order with consecutive repeats removed."""
    a = pred.sort_values("obs").arc.values
    if len(a) == 0:
        return []
    keep = np.r_[True, a[1:] != a[:-1]]
    return a[keep].tolist()
