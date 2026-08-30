"""Score perturbation deltas into calibrated enhancer-gene link statistics.

The p-value uses an empirical background null (the same idea Signac LinkPeaks and FigR
use), so the raw delta need not follow any distribution: for each silenced feature, the cis targets'
delta is compared against a background set of distal response features. `zscore` is a
descriptive z relative to that background (`(mean(delta_bg) - delta_cis)/sd(delta_bg)`, so an
activating link (delta < 0) gives a large positive z), but `pvalue` is **not** read off a normal
table from that z: the backgrounds are measurably leptokurtic (real runs show excess kurtosis
up to ~3.5, not 0), so a normal-tail p-value understates the true tail probability by 1-2 orders
of magnitude once z is large. Instead `pvalue` comes from the background's own empirical rank,
extrapolated past the empirical resolution with a Generalized Pareto tail fit (peaks-over-
threshold; Hosking & Wallis 1987 method-of-moments estimator) rather than an assumed Gaussian
tail. `twosided_pvalue` is the same treatment on both tails. This step is pure CPU and
null-agnostic vs the cached deltas -- changing the null (random vs accessibility-matched), n_bg,
bins, or sidedness only reruns this.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12


def _gpd_sf(y, xi, sigma):
    """Survival function of a Generalized Pareto distribution at y >= 0."""
    if sigma <= 0:
        return 0.0
    if abs(xi) < 1e-8:
        return float(np.exp(-y / sigma))
    val = 1.0 + xi * y / sigma
    if val <= 0:
        return 0.0
    return float(val ** (-1.0 / xi))


def _lmom_gpd(exceed):
    """L-moment estimator (Hosking & Wallis 1987) for a zero-threshold GPD fit to ``exceed``.

    Preferred over ordinary moment-matching for the GPD: substantially less variance in the
    small samples (tens of points) this function is called on. Returns ``None`` if the sample
    cannot support a stable estimate.
    """
    x = np.sort(exceed)
    n = x.size
    i = np.arange(1, n + 1, dtype=np.float64)
    b0 = x.mean()
    b1 = np.sum((i - 1) / (n - 1) * x) / n
    l1, l2 = b0, 2.0 * b1 - b0
    if l2 <= 0:
        return None
    xi = l1 / l2 - 2.0
    sigma = l1 * (l1 / l2 - 1.0)
    if sigma <= 0:
        return None
    return xi, sigma


def _tail_pvalue(bg, x, tail_frac=0.1, min_exceed=30):
    """Upper-tail p-value of ``x`` within background sample ``bg``.

    Uses the empirical rank when ``x`` sits below the tail threshold, when too few exceedances
    are available to fit a tail, or when the fitted tail is bounded (shape < 0) and ``x`` falls
    beyond its finite endpoint -- a small exceedance sample cannot support extrapolating past a
    fitted hard boundary, so that case backs off to the empirical rank rather than overclaiming
    "impossible". Otherwise uses a Generalized Pareto peaks-over-threshold extrapolation
    (L-moment estimator, Hosking & Wallis 1987) beyond the threshold -- avoiding the normal-tail
    assumption, since the observed backgrounds are measurably heavier-tailed than Gaussian.

    Parameters
    ----------
    bg : numpy.ndarray
        Background sample.
    x : float
        Value whose upper-tail probability within ``bg`` is estimated.
    tail_frac : float, optional
        Fraction of ``bg`` treated as the tail region for the GPD fit (default 0.1).
    min_exceed : int, optional
        Minimum exceedances required to fit the GPD tail; falls back to the empirical rank
        below this (default 30).

    Returns
    -------
    float
        One-sided upper-tail p-value: the empirical rank (floored at ``1/(len(bg)+1)``), or the
        fitted Generalized Pareto tail probability beyond the threshold -- which can and should
        go below the empirical floor for a genuinely extreme ``x``.
    """
    n = bg.size

    def emp():
        return (np.count_nonzero(bg >= x) + 1) / (n + 1)

    thresh = np.quantile(bg, 1.0 - tail_frac)
    if x <= thresh:
        return emp()
    exceed = bg[bg > thresh] - thresh
    if exceed.size < min_exceed:
        return emp()
    fit = _lmom_gpd(exceed)
    if fit is None:
        return emp()
    xi, sigma = fit
    y = x - thresh
    if xi < 0 and y >= -sigma / xi:
        return emp()          # beyond the fitted bounded tail's support -- can't trust that far
    tail_prob = exceed.size / n
    p = tail_prob * _gpd_sf(y, xi, sigma)
    return float(np.clip(p, 1e-300, 1.0))


def score_from_deltas(D, kept, p0, groups, resp_index, direction,
                      n_bg=500, seed=0, match_bg=True, n_bins=25, log=print):
    """Turn per-silenced-feature deltas into per-pair z-scores and p-values under a null.

    Each candidate (cis) target's delta is z-scored against a background set of response
    features: ``z = (mean(delta_bg) - delta) / sd(delta_bg)``, so an activating link
    (delta < 0) yields a large positive z. The p-value is the one-sided activating tail, with
    a two-sided variant also returned.

    Parameters
    ----------
    D : numpy.ndarray
        Float32 array (n_kept x n_resp) of mean deltas, one row per silenced feature.
    kept : list
        Silenced-feature names in ``D`` row order.
    p0 : numpy.ndarray
        Baseline predicted softmax share per response feature (used only when ``match_bg``).
    groups : dict
        ``{silenced_feature -> [response_feature, ...]}`` candidate targets to score.
    resp_index : dict
        Response-feature name -> column index in ``D``.
    direction : str
        ``"gene2peak"`` or ``"peak2gene"`` (sets which name becomes peak vs gene).
    n_bg : int, optional
        Background response features drawn per candidate (default 500).
    seed : int, optional
        Seed for background sampling (default 0).
    match_bg : bool, optional
        If True (default), draw the background from the candidate's own baseline-accessibility
        bin (log ``p0``), controlling for peak accessibility like Signac/FigR, so a
        high-accessibility promoter peak is not called significant merely for being accessible.
        If False, use a random distal-feature background per silenced feature -- more sensitive,
        but its rankings are inflated by promoter/high-accessibility peaks (kept for benchmark
        parity).
    n_bins : int, optional
        Number of log-``p0`` bins for the matched null (default 25; capped by feature count).
    log : callable, optional
        Logging function (default ``print``).

    Returns
    -------
    list of dict
        One record per scored pair with keys: ``peak``, ``gene``, ``delta``, ``zscore``,
        ``pvalue`` (one-sided), ``twosided_pvalue``.
    """
    n_resp = D.shape[1]
    rng = np.random.default_rng(seed)
    all_resp = np.arange(n_resp)

    if match_bg:
        lp0 = np.log(np.maximum(p0, 1e-30))
        n_bins = int(min(n_bins, max(2, n_resp // (2 * n_bg))))
        qs = np.quantile(lp0, np.linspace(0, 1, n_bins + 1))
        qs[0] -= 1e-9; qs[-1] += 1e-9
        peak_bin = np.clip(np.digitize(lp0, qs[1:-1]), 0, n_bins - 1)
        bin_members = [np.where(peak_bin == b)[0] for b in range(n_bins)]
        log(f"  accessibility-matched null: {n_bins} log-p0 bins")

    records = []
    for row, silenced in enumerate(kept):
        d = D[row]
        cis_names = groups[silenced]
        cis_idx = np.array([resp_index[t] for t in cis_names], dtype=int)

        if not match_bg:
            pool = np.setdiff1d(all_resp, cis_idx, assume_unique=False)
            k = min(n_bg, pool.size)
            bg_idx = rng.choice(pool, size=k, replace=False) if k > 0 else all_resp
            bgv_g = d[bg_idx]
            mu_g, sd_g = float(bgv_g.mean()), float(bgv_g.std(ddof=1))
            if sd_g < EPS:
                sd_g = EPS

        for t, j in zip(cis_names, cis_idx):
            delta = float(d[j])
            if match_bg:
                b = int(peak_bin[j]); w = 0
                cand = bin_members[b]
                while cand.size - cis_idx.size < n_bg and w < n_bins:
                    w += 1; lo = max(0, b - w); hi = min(n_bins, b + w + 1)
                    cand = np.concatenate([bin_members[bb] for bb in range(lo, hi)])
                cand = cand[~np.isin(cand, cis_idx)]
                if cand.size > n_bg:
                    cand = rng.choice(cand, size=n_bg, replace=False)
                bgv = d[cand]
                mu, sd = float(bgv.mean()), float(bgv.std(ddof=1))
                if sd < EPS:
                    sd = EPS
            else:
                bgv, mu, sd = bgv_g, mu_g, sd_g
            z = (mu - delta) / sd                 # activating (delta<0) -> z>0, descriptive only
            p1 = _tail_pvalue(-bgv, -delta)        # 1-sided activating tail (delta unusually low)
            p_hi = _tail_pvalue(bgv, delta)        # opposite tail (delta unusually high)
            p2 = min(1.0, 2.0 * min(p1, p_hi))
            peak, gene = (t, silenced) if direction == "gene2peak" else (silenced, t)
            records.append(dict(peak=peak, gene=gene, delta=delta,
                                zscore=z, pvalue=p1, twosided_pvalue=p2))
    return records


def bh_fdr(pvals):
    """Benjamini-Hochberg adjusted p-values (q-values).

    Matches R's ``p.adjust(method="BH")``: the rank factor is ``m / i`` with ``i`` the
    ascending rank (1..m), and a reverse cumulative minimum enforces monotonicity.

    Parameters
    ----------
    pvals : array_like
        Raw p-values.

    Returns
    -------
    numpy.ndarray
        BH q-values in the input order, clipped to [0, 1].
    """
    p = np.asarray(pvals, float)
    m = len(p)
    order = p.argsort()
    ranked = p[order]
    adj = np.minimum.accumulate((ranked * m / np.arange(1, m + 1))[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.clip(adj, 0, 1)
    return out


def records_to_p2g(records):
    """Format scored records into the standard p2g DataFrame.

    Adds BH FDR (one- and two-sided), the ``score`` (-log10 p) ranking column, the ``link_dir``
    sign, and the peak coordinates parsed from the ``chr-start-end`` id.

    Parameters
    ----------
    records : list of dict
        Output of :func:`score_from_deltas`.

    Returns
    -------
    pandas.DataFrame
        Columns: ``peak``, ``peak_chr``, ``peak_start``, ``peak_end``, ``gene``, ``score``,
        ``delta``, ``zscore``, ``pvalue``, ``fdr``, ``twosided_pvalue``, ``twosided_fdr``,
        ``link_dir`` (+1 activating / -1 repressive).
    """
    df = pd.DataFrame.from_records(records)
    df["fdr"] = bh_fdr(df["pvalue"].values)
    df["twosided_fdr"] = bh_fdr(df["twosided_pvalue"].values)
    df["score"] = -np.log10(df["pvalue"].clip(lower=1e-300))     # higher = more significant
    df["link_dir"] = -np.sign(df["delta"]).astype(int)          # +1 activating, -1 repressive
    parts = df["peak"].str.split("-", n=2, expand=True)
    df["peak_chr"], df["peak_start"], df["peak_end"] = parts[0], parts[1].astype(int), parts[2].astype(int)
    if not df["peak_chr"].str.startswith("chr").all():
        df["peak_chr"] = "chr" + df["peak_chr"].str.replace("^chr", "", regex=True)
        df["peak"] = df["peak_chr"] + "-" + df["peak_start"].astype(str) + "-" + df["peak_end"].astype(str)
    cols = ["peak", "peak_chr", "peak_start", "peak_end", "gene", "score", "delta",
            "zscore", "pvalue", "fdr", "twosided_pvalue", "twosided_fdr", "link_dir"]
    return df[cols]
