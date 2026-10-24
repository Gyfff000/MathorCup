"""AI-assisted implementation: OpenAI Codex / GPT-6, 2026-10-24.
Descriptive graph-regularized attribution, not causal identification.
"""
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.linalg import solve
from scipy.stats import rankdata

from .common import preprocessing, progress, save_csv, write_json
from .data import NON_MOLECULES


def signed_laplacian(corr, threshold):
    W = np.asarray(corr, float).copy()
    W[~np.isfinite(W)] = 0
    W[np.abs(W) < threshold] = 0
    np.fill_diagonal(W, 0)
    W = (W + W.T) / 2
    return np.diag(np.abs(W).sum(axis=1)) - W


def graph_coefficients(Z, Y, corr, cfg, graph=True):
    L = signed_laplacian(corr, cfg["graph_threshold"]) if graph else np.zeros((Z.shape[1], Z.shape[1]))
    yc = Y - Y.mean(axis=0)
    return solve(Z.T @ Z + cfg["graph_ridge_alpha"] * np.eye(Z.shape[1]) + cfg["graph_ridge_lambda"] * L, Z.T @ yc, assume_a="pos")


def cross_spearman(X, Y):
    a, b = rankdata(X, axis=0), rankdata(Y, axis=0)
    a -= a.mean(axis=0)
    b -= b.mean(axis=0)
    denom = np.sqrt((a * a).sum(axis=0)[:, None] * (b * b).sum(axis=0)[None, :])
    return np.divide(a.T @ b, denom, out=np.zeros((a.shape[1], b.shape[1])), where=denom > 0)


def top_indices(values, eligible):
    return eligible[np.argsort(-np.abs(values[eligible]), axis=0, kind="stable")[:3]]


def run_q2(data, out, cfg):
    dest = Path(out) / "q2"
    dest.mkdir(parents=True, exist_ok=True)
    X, Y = data.X.to_numpy(), data.Y.to_numpy()
    features, targets = data.X.columns.tolist(), data.Y.columns.tolist()
    eligible = np.array([j for j, c in enumerate(features) if c not in NON_MOLECULES])
    given = data.given_corr.to_numpy()
    graph = nx.Graph()
    graph.add_nodes_from(features)
    edge_rows = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            r = float(given[i, j])
            if abs(r) >= cfg["graph_threshold"]:
                graph.add_edge(features[i], features[j], weight=abs(r), correlation=r)
                edge_rows.append({"source": features[i], "target": features[j], "spearman_r": r, "absolute_weight": abs(r)})
    save_csv(pd.DataFrame(edge_rows), dest / "coexpression_edges.csv")
    communities = list(nx.community.greedy_modularity_communities(graph, weight="weight"))
    modules = {node: i for i, group in enumerate(communities) for node in group}
    save_csv(pd.DataFrame([{"field": c, "module": modules[c], "degree": graph.degree(c), "weighted_degree": graph.degree(c, weight="weight")} for c in features]), dest / "network_nodes.csv")
    graph_stats = []
    for threshold in [.6, .7, .8, .9]:
        gg = nx.Graph()
        gg.add_nodes_from(features)
        gg.add_edges_from((features[i], features[j]) for i in range(len(features)) for j in range(i + 1, len(features)) if abs(given[i, j]) >= threshold)
        graph_stats.append({"threshold": threshold, "edges": gg.number_of_edges(), "components": nx.number_connected_components(gg), "isolated_nodes": len(list(nx.isolates(gg)))})
    save_csv(pd.DataFrame(graph_stats), dest / "network_threshold_sensitivity.csv")
    prep = preprocessing()
    Z = prep.fit_transform(X)
    beta = graph_coefficients(Z, Y, given, cfg)
    ridge = graph_coefficients(Z, Y, given, cfg, graph=False)
    marginal = cross_spearman(Z, Y)
    frequencies = np.zeros_like(beta)
    rng = np.random.default_rng(cfg["seed"])
    for b in range(cfg["bootstrap_repeats"]):
        if b % 20 == 0:
            progress(f"Q2: bootstrap {b}/{cfg['bootstrap_repeats']}")
        ix = rng.integers(0, len(X), len(X))
        zb = preprocessing().fit_transform(X[ix])
        # Refit the network too. Do not merely resample previously computed ranks.
        corrb = np.corrcoef(rankdata(zb, axis=0), rowvar=False)
        bb = graph_coefficients(zb, Y[ix], corrb, cfg)
        for k in range(Y.shape[1]):
            frequencies[top_indices(bb[:, k], eligible), k] += 1
    frequencies /= cfg["bootstrap_repeats"]
    meta = data.chemical_meta.set_index("field")
    top_rows, agreement = [], []
    for k, target in enumerate(targets):
        chosen = top_indices(beta[:, k], eligible)
        other = top_indices(ridge[:, k], eligible)
        univariate = top_indices(marginal[:, k], eligible)
        def jac(a, b):
            return len(set(a) & set(b)) / len(set(a) | set(b))
        agreement.append({"target": target, "graph_vs_ridge_jaccard": jac(chosen, other), "graph_vs_spearman_jaccard": jac(chosen, univariate), "ridge_vs_spearman_jaccard": jac(other, univariate)})
        for rank, j in enumerate(chosen, 1):
            c = features[j]
            origin = meta.at[c, "origin_paper"]
            top_rows.append({"target": target, "rank": rank, "field": c, "chemical_name": meta.at[c, "matrix_alias"],
                             "graph_coefficient": beta[j, k], "association_spearman": marginal[j, k],
                             "bootstrap_top3_frequency": frequencies[j, k], "module": modules[c],
                             "origin": origin, "unit": meta.at[c, "unit"], "known_off_flavors": meta.at[c, "off_flavors"],
                             "stability_flag": "low_selection_frequency" if frequencies[j, k] < .5 else "frequently_selected_in_this_resampling",
                             "chemical_argument": f"Reference origin: {origin}. Verify target-specific mechanism and units; coefficient is association, not causation.",
                             "team_mechanism_review": "required"})
    save_csv(pd.DataFrame(top_rows), dest / "top3_by_sensory_dimension.csv")
    save_csv(pd.DataFrame(agreement), dest / "method_jaccard.csv")
    save_csv(pd.DataFrame(frequencies, index=features, columns=targets).reset_index(names="field"), dest / "bootstrap_selection_frequencies.csv")
    write_json(dest / "status.json", {
        "scope": "descriptive_exploratory_full_dataset_association",
        "nodes": graph.number_of_nodes(), "edges": graph.number_of_edges(), "modules": len(communities),
        "bootstrap_repeats": cfg["bootstrap_repeats"], "bootstrap_rebuilds_preprocessing_and_graph": True,
        "causal_status": "not_identified",
        "chemical_mechanism_status": "reference metadata attached; target-specific human literature review required",
        "regularization_parameters": "fixed research defaults, not optimized for causal recovery",
        "eligible_compounds": len(eligible),
    })
