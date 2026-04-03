"""Stage 04 — Causal Discovery (CD-NOTS via causal-learn)"""
import time
import numpy as np
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, field
from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CausalOutput:
    adjacency_matrix: np.ndarray
    causal_parents: Dict[str, List[Dict]]
    nonstationary_vars: List[str]
    graph_json: Dict
    discovery_metadata: Dict


def _prefilter_skeleton(data: np.ndarray, col_names: List[str], alpha_liberal: float = 0.15):
    """Pre-filter using partial correlation to reduce edges before expensive kernel tests."""
    try:
        from causallearn.utils.cit import CIT
        n_vars = data.shape[1]
        keep_pairs = set()

        # Simple correlation-based pre-filter
        corr = np.corrcoef(data.T)
        threshold = 0.05  # keep pairs with abs correlation > threshold
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                if abs(corr[i, j]) > threshold or i == n_vars - 1 or j == n_vars - 1:
                    keep_pairs.add((i, j))
                    keep_pairs.add((j, i))

        logger.info(f"Pre-filter: kept {len(keep_pairs)//2} of {n_vars*(n_vars-1)//2} pairs")
        return keep_pairs
    except Exception as e:
        logger.warning(f"Pre-filter failed: {e}")
        return None


def run(
    lag_matrix: np.ndarray,
    c_indx: np.ndarray,
    column_names: List[str],
    target_col_index: int,
    alpha: float = 0.05,
    indep_test: str = "rcot",
    progress_cb: Optional[Callable] = None,
) -> CausalOutput:
    def emit(msg: str, pct: float = 0.0):
        logger.info(f"[Stage 04] {msg}")
        if progress_cb:
            progress_cb(pct, msg)

    emit("Starting causal discovery (CD-NOTS)", 0.0)
    t_start = time.time()

    # Limit data size for tractability
    max_samples = 2000
    max_vars = 60

    data = lag_matrix.copy()
    used_col_names = column_names[:]
    used_target_idx = target_col_index

    if data.shape[0] > max_samples:
        data = data[-max_samples:]
        c_indx_used = np.arange(len(data), dtype=np.int64)
        emit(f"Truncated to {max_samples} samples", 0.05)
    else:
        c_indx_used = c_indx[-len(data):] if len(c_indx) > len(data) else c_indx

    # If too many features, keep most correlated with target
    if data.shape[1] > max_vars:
        emit(f"Reducing from {data.shape[1]} to {max_vars} features (correlation filter)", 0.08)
        target_series = data[:, used_target_idx]
        corrs = []
        for i in range(data.shape[1]):
            c = np.corrcoef(data[:, i], target_series)[0, 1]
            corrs.append(abs(c) if not np.isnan(c) else 0.0)
        corrs_arr = np.array(corrs)
        # Always keep target
        top_idx = np.argsort(corrs_arr)[::-1][:max_vars - 1]
        if used_target_idx not in top_idx:
            top_idx = np.append(top_idx[:max_vars - 2], used_target_idx)
        top_idx = np.sort(top_idx)
        data = data[:, top_idx]
        used_col_names = [column_names[i] for i in top_idx]
        used_target_idx = list(top_idx).index(target_col_index)

    # Normalize data for numerical stability
    data_mean = np.nanmean(data, axis=0)
    data_std = np.nanstd(data, axis=0)
    data_std[data_std < 1e-8] = 1.0
    data_norm = (data - data_mean) / data_std
    data_norm = np.nan_to_num(data_norm, nan=0.0)

    emit(f"Running cdnod with {data_norm.shape[1]} features, {data_norm.shape[0]} samples, test={indep_test}", 0.1)

    try:
        from causallearn.search.ConstraintBased.CDNOD import cdnod

        # Use faster test for large datasets
        actual_test = indep_test
        if data_norm.shape[1] > 30 or data_norm.shape[0] > 1000:
            actual_test = "fisherz"
            emit(f"Using fisherz (faster) due to dataset size", 0.12)

        cg = cdnod(
            data_norm,
            c_indx_used,
            alpha=alpha,
            indep_test=actual_test,
            stable=True,
            uc_rule=0,
            uc_priority=2,
            mvcdnod=False,
            correction_name="MV_Crtn_Fisher_Z",
            background_knowledge=None,
            verbose=False,
            show_progress=False,
        )
        emit("CD-NOTS algorithm complete, parsing graph", 0.8)
    except Exception as e:
        logger.error(f"cdnod failed: {e}, falling back to correlation-based causal graph")
        return _fallback_causal(data_norm, used_col_names, used_target_idx, alpha, time.time() - t_start)

    # Parse adjacency matrix
    try:
        adj = np.array(cg.G.graph)
    except Exception:
        adj = np.zeros((data_norm.shape[1] + 1, data_norm.shape[1] + 1), dtype=int)

    n_vars_with_time = len(used_col_names) + 1  # +1 for time node
    if adj.shape[0] < n_vars_with_time:
        # Pad if needed
        new_adj = np.zeros((n_vars_with_time, n_vars_with_time), dtype=int)
        new_adj[:adj.shape[0], :adj.shape[1]] = adj
        adj = new_adj

    time_node_idx = len(used_col_names)  # last column is time node

    # Parse causal parents of target
    causal_parents = {}
    target_name = used_col_names[used_target_idx]
    parents = []

    emit("Extracting causal parents", 0.88)

    for i, col in enumerate(used_col_names):
        if i == used_target_idx:
            continue
        # Check for directed edge i → target (adj[i, target] == 1 and adj[target, i] == -1)
        directed = False
        try:
            if adj[i, used_target_idx] == 1 and adj[used_target_idx, i] == -1:
                directed = True
            elif adj[i, used_target_idx] != 0:
                directed = True  # undirected, include as potential parent
        except Exception:
            pass

        if directed:
            # Compute lag from column name
            lag = 0
            if "_t-" in col:
                try:
                    lag = int(col.split("_t-")[-1])
                except Exception:
                    lag = 0

            # Estimate causal strength via correlation
            corr_val = abs(np.corrcoef(data_norm[:, i], data_norm[:, used_target_idx])[0, 1])
            strength = float(corr_val) if not np.isnan(corr_val) else 0.0

            parents.append({
                "name": col,
                "strength": round(strength, 4),
                "p_value": alpha,  # placeholder
                "lag": lag,
                "directed": True,
            })

    # Identify nonstationary variables (connected to time node)
    nonstationary_vars = []
    for i, col in enumerate(used_col_names):
        try:
            if adj[time_node_idx, i] != 0 or adj[i, time_node_idx] != 0:
                nonstationary_vars.append(col)
        except Exception:
            pass

    causal_parents[target_name] = parents

    # Build graph JSON for frontend
    emit("Building graph JSON", 0.93)
    nodes = []
    edges = []

    # Add nodes
    for i, col in enumerate(used_col_names):
        ntype = "target" if i == used_target_idx else "feature"
        nodes.append({
            "id": col,
            "type": ntype,
            "nonstationary": col in nonstationary_vars,
        })
    nodes.append({"id": "_time_", "type": "time_node", "nonstationary": False})

    # Add edges
    n = adj.shape[0]
    for i in range(min(n - 1, len(used_col_names))):
        for j in range(min(n - 1, len(used_col_names))):
            if i >= j:
                continue
            try:
                if adj[i, j] == 1 and adj[j, i] == -1:
                    strength = abs(float(np.corrcoef(data_norm[:, i], data_norm[:, j])[0, 1]))
                    edges.append({
                        "source": used_col_names[i],
                        "target": used_col_names[j],
                        "directed": True,
                        "strength": round(strength, 4),
                        "p_value": alpha,
                    })
                elif adj[j, i] == 1 and adj[i, j] == -1:
                    strength = abs(float(np.corrcoef(data_norm[:, i], data_norm[:, j])[0, 1]))
                    edges.append({
                        "source": used_col_names[j],
                        "target": used_col_names[i],
                        "directed": True,
                        "strength": round(strength, 4),
                        "p_value": alpha,
                    })
                elif adj[i, j] != 0:
                    strength = abs(float(np.corrcoef(data_norm[:, i], data_norm[:, j])[0, 1]))
                    edges.append({
                        "source": used_col_names[i],
                        "target": used_col_names[j],
                        "directed": False,
                        "strength": round(strength, 4),
                        "p_value": alpha,
                    })
            except Exception:
                pass

    # Time node edges
    for i, col in enumerate(used_col_names):
        try:
            if adj[time_node_idx, i] != 0:
                edges.append({
                    "source": "_time_",
                    "target": col,
                    "directed": True,
                    "strength": 0.2,
                    "p_value": alpha,
                })
        except Exception:
            pass

    runtime = time.time() - t_start
    n_directed = sum(1 for e in edges if e["directed"])
    n_undirected = sum(1 for e in edges if not e["directed"])

    emit(f"Causal discovery done: {len(parents)} parents, {len(edges)} edges ({runtime:.1f}s)", 1.0)
    logger.info(f"[Stage 04] Runtime: {runtime:.1f}s, directed edges: {n_directed}, parents: {len(parents)}")

    return CausalOutput(
        adjacency_matrix=adj,
        causal_parents=causal_parents,
        nonstationary_vars=nonstationary_vars,
        graph_json={"nodes": nodes, "edges": edges},
        discovery_metadata={
            "n_edges": len(edges),
            "n_directed": n_directed,
            "n_undirected": n_undirected,
            "n_parents": len(parents),
            "runtime_sec": round(runtime, 2),
            "indep_test": indep_test,
            "alpha": alpha,
            "n_features": data_norm.shape[1],
            "n_samples": data_norm.shape[0],
        },
    )


def _fallback_causal(data: np.ndarray, col_names: List[str], target_idx: int,
                     alpha: float, runtime: float) -> CausalOutput:
    """Fallback: use top correlated features as 'causal' parents."""
    logger.warning("Using correlation-based fallback for causal discovery")
    target_series = data[:, target_idx]
    parents = []

    for i, col in enumerate(col_names):
        if i == target_idx:
            continue
        corr = np.corrcoef(data[:, i], target_series)[0, 1]
        if np.isnan(corr):
            continue
        if abs(corr) > 0.1:
            lag = 0
            if "_t-" in col:
                try:
                    lag = int(col.split("_t-")[-1])
                except Exception:
                    lag = 0
            parents.append({
                "name": col,
                "strength": round(abs(float(corr)), 4),
                "p_value": alpha,
                "lag": lag,
                "directed": True,
            })

    parents.sort(key=lambda x: x["strength"], reverse=True)
    parents = parents[:10]

    target_name = col_names[target_idx]
    nodes = [{"id": c, "type": "target" if i == target_idx else "feature", "nonstationary": False}
             for i, c in enumerate(col_names)]
    nodes.append({"id": "_time_", "type": "time_node", "nonstationary": False})
    edges = [{"source": p["name"], "target": target_name, "directed": True,
               "strength": p["strength"], "p_value": p["p_value"]} for p in parents]

    n = len(col_names) + 1
    adj = np.zeros((n, n), dtype=int)

    return CausalOutput(
        adjacency_matrix=adj,
        causal_parents={target_name: parents},
        nonstationary_vars=[],
        graph_json={"nodes": nodes, "edges": edges},
        discovery_metadata={
            "n_edges": len(edges), "n_directed": len(edges), "n_undirected": 0,
            "n_parents": len(parents), "runtime_sec": round(runtime, 2),
            "indep_test": "fallback_correlation", "alpha": alpha,
            "n_features": data.shape[1], "n_samples": data.shape[0],
        },
    )
