import os
import json
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import streamlit as st

from simulator import IndustrialProcessSimulator


# -------------------------
# Page + folders
# -------------------------
st.set_page_config(
    page_title="AI Capacity Optimization Engine",
    layout="wide",
)

os.makedirs("data", exist_ok=True)
os.makedirs("reports", exist_ok=True)

MODEL_PATH = "data/ai_model.npz"
TRAIN_DATA_PATH = "data/ai_training_data.csv"
MODEL_METRICS_PATH = "data/ai_model_metrics.json"
DECISIONS_LOG_PATH = "data/decision_log.csv"


# -------------------------
# Helpers
# -------------------------
@dataclass(frozen=True)
class SimInputs:
    incoming_rate: int
    conveyor_capacity: int
    rejection_probability: float
    simulation_time: int
    mc_runs: int


def safe_int(x: float) -> int:
    return int(np.round(float(x)))


def clamp_int(x: int, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, x)))


def histogram_df(values: pd.Series, bins: int = 20) -> pd.DataFrame:
    v = values.to_numpy()
    if len(v) == 0:
        return pd.DataFrame({"bin_left": [], "count": []})
    counts, edges = np.histogram(v, bins=bins)
    left = edges[:-1]
    return pd.DataFrame({"bin_left": left, "count": counts})


def _now_iso() -> str:
    return pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _append_row_csv(path: str, row: Dict) -> None:
    df_new = pd.DataFrame([row])
    if os.path.exists(path):
        df_old = pd.read_csv(path)
        df_out = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_out = df_new
    df_out.to_csv(path, index=False)


def df_to_html_table(df: pd.DataFrame, max_rows: int = 25) -> str:
    if df is None:
        return "<p><i>No data</i></p>"
    df_show = df.head(max_rows).copy()
    return df_show.to_html(index=False, border=0)


def build_html_report(payload: Dict) -> str:
    title = payload.get("title", "Industrial Control Decision Suite Report")
    created = payload.get("created_utc", _now_iso())

    sections_html = []
    for sec in payload.get("sections", []):
        h = f"<h2>{sec.get('heading', 'Section')}</h2>"
        p = sec.get("paragraphs", [])
        p_html = "".join([f"<p>{x}</p>" for x in p])
        table_html = sec.get("table_html", "")
        sections_html.append(h + p_html + table_html)

    style = """
    <style>
      body { font-family: Arial, sans-serif; margin: 32px; color: #111; }
      h1 { margin-bottom: 6px; }
      .meta { color: #444; margin-bottom: 18px; }
      h2 { margin-top: 26px; border-bottom: 1px solid #eee; padding-bottom: 6px; }
      table { border-collapse: collapse; width: 100%; margin-top: 10px; }
      th, td { border: 1px solid #ddd; padding: 8px; font-size: 13px; }
      th { background: #f6f6f6; text-align: left; }
      code { background: #f6f6f6; padding: 2px 5px; border-radius: 4px; }
      .badge { display: inline-block; padding: 4px 8px; border-radius: 10px; background: #f1f1f1; font-size: 12px; }
    </style>
    """

    html = f"""
    <html>
      <head>
        <meta charset="utf-8" />
        <title>{title}</title>
        {style}
      </head>
      <body>
        <h1>{title}</h1>
        <div class="meta">
          <span class="badge">Created (UTC): {created}</span>
        </div>
        {''.join(sections_html)}
      </body>
    </html>
    """
    return html


# -------------------------
# Monte Carlo core
# -------------------------
@st.cache_data(show_spinner=False)
def run_monte_carlo_kpis(
    incoming_rate: int,
    conveyor_capacity: int,
    rejection_probability: float,
    simulation_time: int,
    mc_runs: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    last_df = None
    total_incoming = incoming_rate * simulation_time

    for seed in range(mc_runs):
        sim = IndustrialProcessSimulator(
            incoming_rate=incoming_rate,
            conveyor_capacity=conveyor_capacity,
            rejection_probability=rejection_probability,
            simulation_time=simulation_time,
            seed=seed,
        )
        df = sim.run()
        last_df = df

        total_completed = int(df["completed_total"].iloc[-1])
        final_queue = int(df["queue_length"].iloc[-1])
        problem_solve = int(df["problem_solve_total"].iloc[-1])

        throughput_rate = total_completed / simulation_time if simulation_time > 0 else 0.0
        efficiency = (total_completed / total_incoming) * 100 if total_incoming > 0 else 0.0

        rows.append(
            {
                "seed": seed,
                "total_incoming": total_incoming,
                "total_completed": total_completed,
                "final_queue": final_queue,
                "problem_solve": problem_solve,
                "throughput_rate": throughput_rate,
                "efficiency": efficiency,
            }
        )

    mc_df = pd.DataFrame(rows)
    return mc_df, last_df


def stability_check(
    mc_df: pd.DataFrame,
    backlog_tolerance: int,
    efficiency_target: float,
    percentile: int,
) -> Tuple[bool, float, float]:
    q_pct = float(np.percentile(mc_df["final_queue"], percentile))
    e_pct = float(np.percentile(mc_df["efficiency"], percentile))

    stable_backlog = q_pct <= backlog_tolerance
    stable_eff = e_pct >= efficiency_target

    return (stable_backlog and stable_eff), q_pct, e_pct


def find_min_stable_capacity(
    incoming_rate: int,
    rejection_probability: float,
    simulation_time: int,
    mc_runs: int,
    backlog_tolerance: int,
    efficiency_target: float,
    confidence_pct: int,
    low: int = 1,
    high: int = 200,
) -> Tuple[Optional[int], Optional[float], Optional[float]]:
    best_cap = None
    best_q = None
    best_e = None

    while low <= high:
        mid = (low + high) // 2

        mc_df, _ = run_monte_carlo_kpis(
            incoming_rate=incoming_rate,
            conveyor_capacity=mid,
            rejection_probability=rejection_probability,
            simulation_time=simulation_time,
            mc_runs=mc_runs,
        )

        stable, q_pct, e_pct = stability_check(
            mc_df=mc_df,
            backlog_tolerance=backlog_tolerance,
            efficiency_target=efficiency_target,
            percentile=confidence_pct,
        )

        if stable:
            best_cap, best_q, best_e = mid, q_pct, e_pct
            high = mid - 1
        else:
            low = mid + 1

    return best_cap, best_q, best_e


def quick_engineering_anchor(
    incoming_rate: int,
    rejection_probability: float,
    simulation_time: int,
    backlog_tolerance: int,
    efficiency_target: float,
    confidence_pct: int,
    mc_runs_anchor: int = 25,
    high: int = 200,
) -> Tuple[Optional[int], Optional[int], Optional[float], Optional[float]]:
    best_cap, q_pct, e_pct = find_min_stable_capacity(
        incoming_rate=incoming_rate,
        rejection_probability=rejection_probability,
        simulation_time=simulation_time,
        mc_runs=mc_runs_anchor,
        backlog_tolerance=backlog_tolerance,
        efficiency_target=efficiency_target,
        confidence_pct=confidence_pct,
        low=1,
        high=high,
    )

    if best_cap is None:
        return None, None, None, None

    buffered = int(np.ceil(best_cap * 1.05))
    return best_cap, buffered, q_pct, e_pct


# -------------------------
# Lightweight AI model
# -------------------------
def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z))


def _standardize_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    return mu, sigma


def _standardize_apply(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return (X - mu) / sigma


def train_logreg_numpy(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.10,
    steps: int = 1000,
    l2: float = 0.20,
) -> Dict[str, np.ndarray]:
    mu, sigma = _standardize_fit(X)
    Xs = _standardize_apply(X, mu, sigma)

    n, d = Xs.shape
    w = np.zeros(d, dtype=float)
    b = 0.0

    for _ in range(steps):
        p = _sigmoid(Xs @ w + b)
        grad_w = (Xs.T @ (p - y)) / n + l2 * w
        grad_b = float(np.mean(p - y))
        w -= lr * grad_w
        b -= lr * grad_b

    return {"w": w, "b": np.array([b]), "mu": mu, "sigma": sigma}


def predict_proba(model: Dict[str, np.ndarray], X: np.ndarray) -> np.ndarray:
    Xs = _standardize_apply(X, model["mu"], model["sigma"])
    b = float(model["b"][0])
    return _sigmoid(Xs @ model["w"] + b)


def save_model_npz(path: str, model: Dict[str, np.ndarray], feature_names: List[str]) -> None:
    np.savez(
        path,
        w=model["w"],
        b=model["b"],
        mu=model["mu"],
        sigma=model["sigma"],
        feature_names=np.array(feature_names, dtype=object),
    )


def load_model_npz(path: str) -> Optional[Dict[str, np.ndarray]]:
    if not os.path.exists(path):
        return None
    data = np.load(path, allow_pickle=True)
    return {
        "w": data["w"],
        "b": data["b"],
        "mu": data["mu"],
        "sigma": data["sigma"],
        "feature_names": data["feature_names"].tolist(),
    }


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    y_true_i = y_true.astype(int)

    tp = int(np.sum((y_pred == 1) & (y_true_i == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true_i == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true_i == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true_i == 1)))

    acc = (tp + tn) / max(1, (tp + tn + fp + fn))
    prec = tp / max(1, (tp + fp))
    rec = tp / max(1, (tp + fn))
    f1 = (2 * prec * rec) / max(1e-12, (prec + rec))

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def roc_curve_points(y_true: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
    y_true_i = y_true.astype(int)
    thresholds = np.unique(y_prob)
    thresholds = np.sort(thresholds)[::-1]

    rows = []
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        tp = int(np.sum((y_pred == 1) & (y_true_i == 1)))
        tn = int(np.sum((y_pred == 0) & (y_true_i == 0)))
        fp = int(np.sum((y_pred == 1) & (y_true_i == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true_i == 1)))

        tpr = tp / max(1, (tp + fn))
        fpr = fp / max(1, (fp + tn))
        rows.append({"threshold": float(thr), "tpr": float(tpr), "fpr": float(fpr)})

    rows.append({"threshold": 1.01, "tpr": 0.0, "fpr": 0.0})
    rows.append({"threshold": -0.01, "tpr": 1.0, "fpr": 1.0})

    df = pd.DataFrame(rows).sort_values("fpr")
    df = df.drop_duplicates(subset=["fpr", "tpr"]).reset_index(drop=True)
    return df


def auc_trapezoid(fpr: np.ndarray, tpr: np.ndarray) -> float:
    order = np.argsort(fpr)
    f = fpr[order]
    t = tpr[order]
    return float(np.trapezoid(t, f))


@st.cache_data(show_spinner=False)
def generate_ai_training_data(
    n_samples: int,
    mc_runs_train: int,
    incoming_min: int,
    incoming_max: int,
    cap_min: int,
    cap_max: int,
    rej_min: float,
    rej_max: float,
    sim_time: int,
    backlog_tolerance: int,
    efficiency_target: float,
    confidence_pct: int,
    boundary_focus_pct: int = 50,
    boundary_band: int = 8,
    anchor_mc_runs: int = 12,
    rng_seed: int = 123,
) -> pd.DataFrame:
    """
    Generates a mixed training set:
    - broad random scenarios
    - boundary-focused scenarios around the minimum stable capacity
    """
    rng = np.random.default_rng(rng_seed)
    rows = []

    n_boundary = int(n_samples * (boundary_focus_pct / 100.0))
    n_random = n_samples - n_boundary

    def evaluate_sample(incoming: int, cap: int, rej: float) -> Dict:
        mc_df, _ = run_monte_carlo_kpis(
            incoming_rate=incoming,
            conveyor_capacity=cap,
            rejection_probability=rej,
            simulation_time=sim_time,
            mc_runs=mc_runs_train,
        )

        stable, q_pct, e_pct = stability_check(
            mc_df=mc_df,
            backlog_tolerance=backlog_tolerance,
            efficiency_target=efficiency_target,
            percentile=confidence_pct,
        )

        return {
            "incoming_rate": incoming,
            "conveyor_capacity": cap,
            "rejection_probability": rej,
            "simulation_time": sim_time,
            "backlog_tolerance": backlog_tolerance,
            "efficiency_target": efficiency_target,
            "confidence_pct": confidence_pct,
            "backlog_pctl": q_pct,
            "efficiency_pctl": e_pct,
            "stable_label": 1 if stable else 0,
        }

    # Broad random coverage
    for _ in range(n_random):
        incoming = int(rng.integers(incoming_min, incoming_max + 1))
        cap = int(rng.integers(cap_min, cap_max + 1))
        rej = float(rng.uniform(rej_min, rej_max))
        rows.append(evaluate_sample(incoming, cap, rej))

    # Boundary-focused coverage near engineering transition
    tries = 0
    while len(rows) < n_samples and tries < n_samples * 6:
        tries += 1

        incoming = int(rng.integers(incoming_min, incoming_max + 1))
        rej = float(rng.uniform(rej_min, rej_max))

        best_cap, _, _ = find_min_stable_capacity(
            incoming_rate=incoming,
            rejection_probability=rej,
            simulation_time=sim_time,
            mc_runs=anchor_mc_runs,
            backlog_tolerance=backlog_tolerance,
            efficiency_target=efficiency_target,
            confidence_pct=confidence_pct,
            low=max(1, cap_min),
            high=cap_max,
        )

        if best_cap is None:
            cap = int(rng.integers(cap_min, cap_max + 1))
        else:
            lo = clamp_int(best_cap - boundary_band, cap_min, cap_max)
            hi = clamp_int(best_cap + boundary_band, cap_min, cap_max)
            cap = int(rng.integers(lo, hi + 1))

        rows.append(evaluate_sample(incoming, cap, rej))

    return pd.DataFrame(rows)


def forecast_time_to_tolerance(last_df: pd.DataFrame, backlog_tolerance: int) -> Optional[float]:
    if last_df is None or len(last_df) < 10:
        return None

    t = last_df["time"].to_numpy(dtype=float)
    q = last_df["queue_length"].to_numpy(dtype=float)

    if np.all(q == q[0]):
        return None

    a, b = np.polyfit(t, q, deg=1)
    if a <= 1e-9:
        return None

    if q[-1] >= backlog_tolerance:
        return 0.0

    t_hit = (backlog_tolerance - b) / a
    minutes_to_hit = float(max(0.0, t_hit - t[-1]))
    return minutes_to_hit


# -------------------------
# App title
# -------------------------
st.title("AI Capacity Optimization Engine")
st.write("A hybrid capacity planning system combining Monte Carlo simulation, optimisation, and AI calibration for conveyor-based industrial processes.")


# -------------------------
# Sidebar inputs
# -------------------------
with st.sidebar:
    st.subheader("Inputs")

    incoming_rate = st.slider("Incoming parcels per minute", 5, 60, 25)
    conveyor_capacity = st.slider("Conveyor capacity per minute", 5, 200, 28)
    rejection_probability = st.slider("Rejection probability", 0.0, 0.2, 0.03)
    simulation_time = st.slider("Simulation time (minutes)", 10, 300, 120)

    st.divider()

    st.subheader("Monte Carlo")
    mc_runs = st.slider("Monte Carlo runs", 10, 500, 100, step=10)

    st.divider()

    st.subheader("Stability rule")
    backlog_tolerance = st.slider("Backlog tolerance (parcels)", 0, 500, 5, step=1)
    efficiency_target = st.slider("Efficiency target (%)", 50, 100, 95, step=1)
    confidence_pct = st.slider("Confidence percentile (%)", 50, 99, 95, step=1)

    st.divider()

    st.subheader("Safety buffer")
    buffer_percent = st.slider("Safety buffer (%)", 0, 30, 5, step=1)


inputs = SimInputs(
    incoming_rate=incoming_rate,
    conveyor_capacity=conveyor_capacity,
    rejection_probability=rejection_probability,
    simulation_time=simulation_time,
    mc_runs=mc_runs,
)

st.divider()


# -------------------------
# Tabs
# -------------------------
tab_sim, tab_opt, tab_dec, tab_ai = st.tabs(
    [
        "Digital Twin Simulator",
        "Capacity Optimiser",
        "Decision Engine",
        "AI / Control Optimisation",
    ]
)


# -------------------------
# 1) Digital Twin Simulator
# -------------------------
with tab_sim:
    st.subheader("Digital Twin Simulator")
    run_btn = st.button("Run Simulation", key="run_sim_btn")

    if run_btn:
        with st.spinner("Running Monte Carlo simulation..."):
            mc_df, last_df = run_monte_carlo_kpis(
                incoming_rate=inputs.incoming_rate,
                conveyor_capacity=inputs.conveyor_capacity,
                rejection_probability=inputs.rejection_probability,
                simulation_time=inputs.simulation_time,
                mc_runs=inputs.mc_runs,
            )

            stable, q_pct, e_pct = stability_check(
                mc_df=mc_df,
                backlog_tolerance=backlog_tolerance,
                efficiency_target=efficiency_target,
                percentile=confidence_pct,
            )

            total_incoming_avg = int(mc_df["total_incoming"].mean())
            total_completed_avg = safe_int(mc_df["total_completed"].mean())
            final_queue_avg = safe_int(mc_df["final_queue"].mean())
            problem_solve_avg = safe_int(mc_df["problem_solve"].mean())
            throughput_avg = float(mc_df["throughput_rate"].mean())
            efficiency_avg = float(mc_df["efficiency"].mean())

            last_df.to_csv("data/simulation_results.csv", index=False)
            mc_df.to_csv("data/monte_carlo_kpis.csv", index=False)

            with open("reports/summary.txt", "w", encoding="utf-8") as f:
                f.write("Industrial Process Simulation Report (Monte Carlo Averaged)\n")
                f.write("---------------------------------------------------------\n")
                f.write(f"Monte Carlo runs: {inputs.mc_runs}\n")
                f.write(f"Incoming rate: {inputs.incoming_rate}\n")
                f.write(f"Conveyor capacity: {inputs.conveyor_capacity}\n")
                f.write(f"Rejection probability: {inputs.rejection_probability}\n")
                f.write(f"Simulation time: {inputs.simulation_time}\n")
                f.write(f"Backlog tolerance: {backlog_tolerance}\n")
                f.write(f"Efficiency target: {efficiency_target}\n")
                f.write(f"Confidence percentile: {confidence_pct}\n\n")

                f.write(f"Avg Total Incoming: {total_incoming_avg}\n")
                f.write(f"Avg Total Completed: {total_completed_avg}\n")
                f.write(f"Avg Final Backlog: {final_queue_avg}\n")
                f.write(f"Avg Problem Solve: {problem_solve_avg}\n")
                f.write(f"Avg Throughput (per min): {throughput_avg:.2f}\n")
                f.write(f"Avg Efficiency (%): {efficiency_avg:.2f}\n\n")

                f.write(f"Backlog p{confidence_pct}: {q_pct:.2f}\n")
                f.write(f"Efficiency p{confidence_pct}: {e_pct:.2f}\n")
                f.write(f"Stable (rule): {stable}\n")

            st.session_state["mc_df"] = mc_df
            st.session_state["last_df"] = last_df
            st.session_state["stable"] = stable
            st.session_state["q_pct"] = q_pct
            st.session_state["e_pct"] = e_pct
            st.session_state["sim_payload"] = {
                "incoming_rate": inputs.incoming_rate,
                "conveyor_capacity": inputs.conveyor_capacity,
                "rejection_probability": inputs.rejection_probability,
                "simulation_time": inputs.simulation_time,
                "mc_runs": inputs.mc_runs,
                "backlog_tolerance": backlog_tolerance,
                "efficiency_target": efficiency_target,
                "confidence_pct": confidence_pct,
                "stable": bool(stable),
                "backlog_pctl": float(q_pct),
                "efficiency_pctl": float(e_pct),
                "avg_total_incoming": int(total_incoming_avg),
                "avg_total_completed": int(total_completed_avg),
                "avg_final_backlog": int(final_queue_avg),
                "avg_problem_solve": int(problem_solve_avg),
                "avg_throughput_per_min": float(throughput_avg),
                "avg_efficiency_pct": float(efficiency_avg),
                "created_utc": _now_iso(),
            }

    if "mc_df" in st.session_state and "last_df" in st.session_state:
        mc_df = st.session_state["mc_df"]
        last_df = st.session_state["last_df"]
        stable = bool(st.session_state["stable"])
        q_pct = float(st.session_state["q_pct"])
        e_pct = float(st.session_state["e_pct"])

        st.subheader("System Status")

        final_queue_avg = safe_int(mc_df["final_queue"].mean())
        backlog_growth_rate = final_queue_avg / inputs.simulation_time if inputs.simulation_time > 0 else 0.0

        if stable:
            st.success("System stable under the confidence rule.")
        else:
            st.error("System NOT stable under the confidence rule (risk of backlog growth).")

        st.write(f"Avg backlog growth rate: {backlog_growth_rate:.2f} parcels/min")
        st.write(f"Backlog p{confidence_pct}: {q_pct:.1f} (tolerance: {backlog_tolerance})")
        st.write(f"Efficiency p{confidence_pct}: {e_pct:.2f}% (target: {efficiency_target}%)")

        minutes_to_hit = forecast_time_to_tolerance(last_df, backlog_tolerance)
        if minutes_to_hit is not None and not stable:
            st.warning(f"Predictive congestion: estimated time to hit backlog tolerance is ~{minutes_to_hit:.1f} minutes.")

        st.subheader("Final KPIs (Monte Carlo averages)")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Incoming", int(mc_df["total_incoming"].mean()))
        c2.metric("Total Completed", safe_int(mc_df["total_completed"].mean()))
        c3.metric("Final Backlog", safe_int(mc_df["final_queue"].mean()))
        c4.metric("Problem Solve", safe_int(mc_df["problem_solve"].mean()))
        c5.metric("Throughput/min", f"{float(mc_df['throughput_rate'].mean()):.2f}")
        st.metric("Efficiency (%)", f"{float(mc_df['efficiency'].mean()):.2f}")

        st.subheader("Recommendation")
        rule_of_thumb_capacity = int(np.ceil(inputs.incoming_rate * 1.05))
        if (not stable) or (inputs.conveyor_capacity < rule_of_thumb_capacity):
            st.info(
                f"Suggested action: increase conveyor capacity to about {rule_of_thumb_capacity} parcels/min "
                f"(current: {inputs.conveyor_capacity}), or reduce incoming rate."
            )
        else:
            st.write("No capacity increase recommended under the current settings.")

        st.subheader("Representative run (time-series)")
        st.dataframe(last_df, use_container_width=True)

        st.subheader("Queue Length (Backlog)")
        st.line_chart(last_df.set_index("time")[["queue_length"]])

        st.subheader("Completed vs Problem Solve")
        st.line_chart(last_df.set_index("time")[["completed_total", "problem_solve_total"]])

        st.subheader("Monte Carlo KPI table (all runs)")
        st.dataframe(mc_df, use_container_width=True)

        st.subheader("Monte Carlo distribution (final backlog)")
        if mc_df["final_queue"].nunique() <= 1 and int(mc_df["final_queue"].iloc[0]) == 0:
            st.write("All Monte Carlo runs ended with zero final backlog.")
        else:
            hdf = histogram_df(mc_df["final_queue"], bins=25)
            st.bar_chart(hdf.set_index("bin_left")["count"])

        st.subheader("Monte Carlo distribution (efficiency)")
        if mc_df["efficiency"].nunique() <= 1:
            st.write(f"Efficiency is constant across runs: {float(mc_df['efficiency'].iloc[0]):.2f}%.")
        else:
            hdf_e = histogram_df(mc_df["efficiency"], bins=25)
            st.bar_chart(hdf_e.set_index("bin_left")["count"])

        st.divider()
        st.subheader("Export report")

        if os.path.exists("reports/summary.txt"):
            with open("reports/summary.txt", "rb") as f:
                st.download_button(
                    "Download summary.txt",
                    data=f,
                    file_name="summary.txt",
                    mime="text/plain",
                )

        sim_payload = st.session_state.get("sim_payload", {})
        report_payload = {
            "title": "Industrial Process Simulation Report",
            "created_utc": sim_payload.get("created_utc", _now_iso()),
            "sections": [
                {
                    "heading": "Inputs",
                    "paragraphs": [
                        f"Incoming rate: <b>{incoming_rate}</b> parcels/min",
                        f"Conveyor capacity: <b>{conveyor_capacity}</b> parcels/min",
                        f"Rejection probability: <b>{rejection_probability}</b>",
                        f"Simulation time: <b>{simulation_time}</b> minutes",
                        f"Monte Carlo runs: <b>{mc_runs}</b>",
                        f"Stability rule: backlog p{confidence_pct} <= {backlog_tolerance} and efficiency p{confidence_pct} >= {efficiency_target}%",
                    ],
                },
                {
                    "heading": "Decision",
                    "paragraphs": [
                        f"Stable under confidence rule: <b>{bool(stable)}</b>",
                        f"Backlog p{confidence_pct}: <b>{q_pct:.2f}</b>",
                        f"Efficiency p{confidence_pct}: <b>{e_pct:.2f}%</b>",
                    ],
                },
                {
                    "heading": "Monte Carlo KPIs (first 25 rows)",
                    "table_html": df_to_html_table(mc_df, max_rows=25),
                    "paragraphs": ["Full CSV saved as <code>data/monte_carlo_kpis.csv</code>."],
                },
                {
                    "heading": "Representative run time-series (first 25 rows)",
                    "table_html": df_to_html_table(last_df, max_rows=25),
                    "paragraphs": ["Full CSV saved as <code>data/simulation_results.csv</code>."],
                },
            ],
        }
        html_report = build_html_report(report_payload)
        st.download_button(
            "Download HTML report",
            data=html_report.encode("utf-8"),
            file_name="simulation_report.html",
            mime="text/html",
        )
    else:
        st.caption("Click Run Simulation to generate results.")


# -------------------------
# 2) Capacity Optimiser
# -------------------------
with tab_opt:
    st.subheader("Capacity Optimiser")

    left, right = st.columns([1, 1])

    with left:
        st.write("Find the minimum conveyor capacity satisfying the confidence-based stability rule.")

        opt_high = st.number_input(
            "Optimiser upper bound (search max capacity)",
            min_value=10,
            max_value=2000,
            value=max(200, incoming_rate * 5),
            step=10,
        )

        optimise_btn = st.button("Optimise Conveyor Capacity", key="optimise_btn")

    if optimise_btn:
        with st.spinner("Optimising with Monte Carlo tests..."):
            best_cap, q_pct, e_pct = find_min_stable_capacity(
                incoming_rate=incoming_rate,
                rejection_probability=rejection_probability,
                simulation_time=simulation_time,
                mc_runs=mc_runs,
                backlog_tolerance=backlog_tolerance,
                efficiency_target=efficiency_target,
                confidence_pct=confidence_pct,
                low=1,
                high=int(opt_high),
            )

            if best_cap is None:
                st.error("No stable capacity found within the chosen bounds. Increase the upper bound.")
            else:
                buffered = int(np.ceil(best_cap * (1 + buffer_percent / 100)))

                st.success(f"Minimum stable conveyor capacity: {best_cap} parcels/min")
                st.info(f"Buffered capacity with {buffer_percent}% safety buffer: {buffered} parcels/min")

                st.write(
                    f"Stability rule: backlog p{confidence_pct} <= {backlog_tolerance} "
                    f"and efficiency p{confidence_pct} >= {efficiency_target}%"
                )
                st.write(
                    f"At best capacity: backlog p{confidence_pct} = {q_pct:.1f}, "
                    f"efficiency p{confidence_pct} = {e_pct:.2f}%"
                )

                opt_out = pd.DataFrame(
                    [
                        {
                            "incoming_rate": incoming_rate,
                            "rejection_probability": rejection_probability,
                            "simulation_time": simulation_time,
                            "mc_runs": mc_runs,
                            "confidence_pct": confidence_pct,
                            "backlog_tolerance": backlog_tolerance,
                            "efficiency_target": efficiency_target,
                            "best_capacity": best_cap,
                            "buffer_percent": buffer_percent,
                            "buffered_capacity": buffered,
                            "backlog_pctl": q_pct,
                            "efficiency_pctl": e_pct,
                        }
                    ]
                )
                opt_out.to_csv("data/optimiser_result.csv", index=False)

                st.session_state["optimiser_best_cap"] = best_cap
                st.session_state["optimiser_buffered_cap"] = buffered
                st.session_state["optimiser_q_pct"] = q_pct
                st.session_state["optimiser_e_pct"] = e_pct

    with right:
        st.write("Latest optimiser result (if available):")
        try:
            opt_df = pd.read_csv("data/optimiser_result.csv")
            st.dataframe(opt_df, use_container_width=True)
        except FileNotFoundError:
            st.caption("Run the optimiser to generate optimiser_result.csv.")


# -------------------------
# 3) Decision Engine
# -------------------------
with tab_dec:
    st.subheader("Decision Engine")

    st.write(
        "Turns simulation outputs into a clear operating recommendation. "
        "Use current slider capacity, or the optimiser buffered capacity if available."
    )

    mode = st.radio(
        "Capacity source",
        options=["Use current slider capacity", "Use optimiser buffered capacity (if available)"],
        horizontal=True,
    )

    if mode == "Use optimiser buffered capacity (if available)" and "optimiser_buffered_cap" in st.session_state:
        cap_for_decision = int(st.session_state["optimiser_buffered_cap"])
        st.write(f"Using buffered capacity from optimiser: {cap_for_decision} parcels/min")
    else:
        cap_for_decision = conveyor_capacity
        st.write(f"Using current slider capacity: {cap_for_decision} parcels/min")

    st.divider()

    assess_btn = st.button("Assess decision", key="assess_btn")

    if assess_btn:
        with st.spinner("Assessing decision with Monte Carlo..."):
            mc_df, _ = run_monte_carlo_kpis(
                incoming_rate=incoming_rate,
                conveyor_capacity=cap_for_decision,
                rejection_probability=rejection_probability,
                simulation_time=simulation_time,
                mc_runs=mc_runs,
            )

            stable, q_pct, e_pct = stability_check(
                mc_df=mc_df,
                backlog_tolerance=backlog_tolerance,
                efficiency_target=efficiency_target,
                percentile=confidence_pct,
            )

            _append_row_csv(
                DECISIONS_LOG_PATH,
                {
                    "created_utc": _now_iso(),
                    "incoming_rate": incoming_rate,
                    "capacity_used": cap_for_decision,
                    "rejection_probability": rejection_probability,
                    "simulation_time": simulation_time,
                    "mc_runs": mc_runs,
                    "backlog_tolerance": backlog_tolerance,
                    "efficiency_target": efficiency_target,
                    "confidence_pct": confidence_pct,
                    "stable": bool(stable),
                    "backlog_pctl": float(q_pct),
                    "efficiency_pctl": float(e_pct),
                    "mode": mode,
                },
            )

            st.subheader("Decision summary")

            if stable:
                st.success("Recommended operating point is stable under the confidence rule.")
                st.write(f"Backlog p{confidence_pct}: {q_pct:.1f} (tolerance: {backlog_tolerance})")
                st.write(f"Efficiency p{confidence_pct}: {e_pct:.2f}% (target: {efficiency_target}%)")
                st.write("Action: maintain settings.")
            else:
                st.error("Operating point is not stable under the confidence rule.")
                st.write(f"Backlog p{confidence_pct}: {q_pct:.1f} (tolerance: {backlog_tolerance})")
                st.write(f"Efficiency p{confidence_pct}: {e_pct:.2f}% (target: {efficiency_target}%)")

                st.write("Action options:")
                st.write("- Increase conveyor capacity")
                st.write("- Reduce incoming rate")
                st.write("- Reduce rejection probability (process quality improvement)")

                st.write("Suggested fix (auto-search):")
                high = max(200, incoming_rate * 5)
                best_cap, best_q, best_e = find_min_stable_capacity(
                    incoming_rate=incoming_rate,
                    rejection_probability=rejection_probability,
                    simulation_time=simulation_time,
                    mc_runs=mc_runs,
                    backlog_tolerance=backlog_tolerance,
                    efficiency_target=efficiency_target,
                    confidence_pct=confidence_pct,
                    low=max(1, cap_for_decision),
                    high=int(high),
                )

                if best_cap is None:
                    st.write("No stable capacity found within the auto-search bounds. Increase bounds or reduce incoming rate.")
                else:
                    buffered = int(np.ceil(best_cap * (1 + buffer_percent / 100)))
                    st.write(f"Minimum stable capacity: {best_cap} parcels/min")
                    st.write(f"Buffered recommendation: {buffered} parcels/min")
                    st.write(f"At that capacity: backlog p{confidence_pct} = {best_q:.1f}, efficiency p{confidence_pct} = {best_e:.2f}%")

            st.divider()
            st.subheader("Decision log")
            try:
                log_df = pd.read_csv(DECISIONS_LOG_PATH)
                st.dataframe(log_df.tail(50), use_container_width=True)
                st.caption(f"Saved to {DECISIONS_LOG_PATH}")
            except FileNotFoundError:
                st.caption("No decision log yet.")
    else:
        st.caption("Click Assess decision to generate a recommendation.")


# -------------------------
# 4) AI / Control Optimisation
# -------------------------
with tab_ai:
    st.subheader("AI / Control Optimisation")
    st.write(
        "Train a lightweight AI model on simulator-generated scenarios to predict stability risk and recommend capacity fast."
    )

    st.divider()

    model = load_model_npz(MODEL_PATH)

    ai_left, ai_right = st.columns([1, 1])

    with ai_left:
        st.subheader("Train / Retrain model")

        n_samples = st.number_input("Training scenarios (samples)", min_value=60, max_value=1500, value=300, step=20)
        mc_runs_train = st.number_input("Monte Carlo runs per training sample", min_value=5, max_value=80, value=12, step=1)
        test_split = st.slider("Train/test split (%)", 10, 40, 20, step=5)

        st.write("Scenario sampling ranges:")
        incoming_min = st.slider("Incoming min", 5, 60, 10)
        incoming_max = st.slider("Incoming max", 5, 60, 45)
        cap_min = st.slider("Capacity min", 5, 200, 10)
        cap_max = st.slider("Capacity max", 5, 200, 90)
        rej_min = st.slider("Rejection min", 0.0, 0.2, 0.01)
        rej_max = st.slider("Rejection max", 0.0, 0.2, 0.08)

        st.write("Calibration settings:")
        boundary_focus_pct = st.slider("Boundary-focused samples (%)", 20, 90, 55, step=5)
        boundary_band = st.slider("Boundary sampling band (+/- capacity units)", 2, 20, 8, step=1)
        anchor_mc_runs = st.slider("Anchor Monte Carlo runs", 5, 30, 12, step=1)

        train_btn = st.button("Generate data + Train AI", key="train_ai_btn")

    with ai_right:
        st.subheader("Model status")
        if model is None:
            st.warning("No model saved yet.")
            if os.path.exists(TRAIN_DATA_PATH):
                st.caption("Saved training data found.")
        else:
            st.success("Model loaded from disk.")
            st.write("Features:")
            st.write(model["feature_names"])
            if os.path.exists(MODEL_METRICS_PATH):
                try:
                    with open(MODEL_METRICS_PATH, "r", encoding="utf-8") as f:
                        m = json.load(f)
                    st.caption(f"Last trained (UTC): {m.get('trained_utc', 'unknown')}")
                except Exception:
                    pass

    if train_btn:
        with st.spinner("Generating training data and training AI model..."):
            df_train = generate_ai_training_data(
                n_samples=int(n_samples),
                mc_runs_train=int(mc_runs_train),
                incoming_min=int(incoming_min),
                incoming_max=int(incoming_max),
                cap_min=int(cap_min),
                cap_max=int(cap_max),
                rej_min=float(rej_min),
                rej_max=float(rej_max),
                sim_time=int(simulation_time),
                backlog_tolerance=int(backlog_tolerance),
                efficiency_target=float(efficiency_target),
                confidence_pct=int(confidence_pct),
                boundary_focus_pct=int(boundary_focus_pct),
                boundary_band=int(boundary_band),
                anchor_mc_runs=int(anchor_mc_runs),
                rng_seed=123,
            )

            feature_names = [
                "incoming_rate",
                "conveyor_capacity",
                "rejection_probability",
                "simulation_time",
                "backlog_tolerance",
                "efficiency_target",
                "confidence_pct",
            ]

            X = df_train[feature_names].to_numpy(dtype=float)
            y = df_train["stable_label"].to_numpy(dtype=float)

            rng = np.random.default_rng(123)
            idx = np.arange(len(df_train))
            rng.shuffle(idx)
            X = X[idx]
            y = y[idx]

            n_test = int(len(df_train) * (test_split / 100.0))
            n_test = clamp_int(n_test, 1, max(1, len(df_train) - 1))

            X_test = X[:n_test]
            y_test = y[:n_test]
            X_tr = X[n_test:]
            y_tr = y[n_test:]

            model_fit = train_logreg_numpy(X_tr, y_tr, lr=0.10, steps=1000, l2=0.20)
            save_model_npz(MODEL_PATH, model_fit, feature_names)

            y_prob = predict_proba(model_fit, X_test)
            metrics = classification_metrics(y_test, y_prob, threshold=0.5)

            roc_df = roc_curve_points(y_test, y_prob)
            auc = auc_trapezoid(roc_df["fpr"].to_numpy(dtype=float), roc_df["tpr"].to_numpy(dtype=float))

            df_train.to_csv(TRAIN_DATA_PATH, index=False)

            metrics_payload = {
                "trained_utc": _now_iso(),
                "n_samples": int(n_samples),
                "mc_runs_train": int(mc_runs_train),
                "test_split_pct": float(test_split),
                "boundary_focus_pct": int(boundary_focus_pct),
                "boundary_band": int(boundary_band),
                "anchor_mc_runs": int(anchor_mc_runs),
                "accuracy": float(metrics["accuracy"]),
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "f1": float(metrics["f1"]),
                "auc": float(auc),
                "tp": int(metrics["tp"]),
                "tn": int(metrics["tn"]),
                "fp": int(metrics["fp"]),
                "fn": int(metrics["fn"]),
                "feature_names": feature_names,
                "class_balance_stable_pct": float(df_train["stable_label"].mean() * 100.0),
            }
            with open(MODEL_METRICS_PATH, "w", encoding="utf-8") as f:
                json.dump(metrics_payload, f, indent=2)

            st.success("AI model trained and saved.")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Accuracy", f"{metrics['accuracy']*100:.1f}%")
            m2.metric("Precision", f"{metrics['precision']:.2f}")
            m3.metric("Recall", f"{metrics['recall']:.2f}")
            m4.metric("F1", f"{metrics['f1']:.2f}")
            m5.metric("AUC", f"{auc:.3f}")

            cm = pd.DataFrame(
                [[int(metrics["tn"]), int(metrics["fp"])], [int(metrics["fn"]), int(metrics["tp"])]],
                index=["Actual 0 (unstable)", "Actual 1 (stable)"],
                columns=["Pred 0", "Pred 1"],
            )
            st.write("Confusion matrix (test set):")
            st.dataframe(cm, use_container_width=True)

            st.subheader("ROC curve (test set)")
            roc_plot_df = roc_df[["fpr", "tpr"]].copy().sort_values("fpr")
            st.line_chart(roc_plot_df.set_index("fpr")[["tpr"]])

            st.subheader("Feature influence (model weights)")
            w = model_fit["w"]
            fi = pd.DataFrame({"feature": feature_names, "weight": w})
            fi["abs_weight"] = np.abs(fi["weight"])
            fi = fi.sort_values("abs_weight", ascending=False).drop(columns=["abs_weight"])
            st.dataframe(fi, use_container_width=True)

            st.subheader("Training set snapshot")
            st.dataframe(df_train.head(15), use_container_width=True)

            st.divider()
            st.subheader("Export training artifacts")
            st.download_button(
                "Download training data (CSV)",
                data=df_train.to_csv(index=False).encode("utf-8"),
                file_name="ai_training_data.csv",
                mime="text/csv",
            )
            st.download_button(
                "Download model metrics (JSON)",
                data=json.dumps(metrics_payload, indent=2).encode("utf-8"),
                file_name="ai_model_metrics.json",
                mime="application/json",
            )

            model = load_model_npz(MODEL_PATH)

    st.divider()

    st.subheader("Live AI prediction (current sliders)")

    model = load_model_npz(MODEL_PATH)
    if model is None:
        st.caption("Train the AI model first to enable live predictions and AI auto-tuning.")
    else:
        def build_feature_row(capacity: int) -> np.ndarray:
            row = np.array(
                [
                    float(incoming_rate),
                    float(capacity),
                    float(rejection_probability),
                    float(simulation_time),
                    float(backlog_tolerance),
                    float(efficiency_target),
                    float(confidence_pct),
                ],
                dtype=float,
            )
            return row.reshape(1, -1)

        p_stable = float(predict_proba(model, build_feature_row(conveyor_capacity))[0])
        p_unstable = 1.0 - p_stable

        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted stability probability", f"{p_stable*100:.1f}%")
        c2.metric("Predicted congestion risk", f"{p_unstable*100:.1f}%")
        c3.metric("Current capacity", f"{conveyor_capacity} parcels/min")

        if p_stable >= 0.80:
            st.success("AI view: system likely stable (fast estimate).")
        elif p_stable >= 0.60:
            st.warning("AI view: borderline stability. Consider a small capacity increase.")
        else:
            st.error("AI view: high instability risk. Increase capacity or reduce incoming rate.")

        st.caption("This is a fast AI estimate. Treat Monte Carlo / optimiser results as engineering ground truth.")

        st.divider()
        st.subheader("AI vs engineering anchor")

        anchor_best, anchor_buffered, anchor_q, anchor_e = quick_engineering_anchor(
            incoming_rate=incoming_rate,
            rejection_probability=rejection_probability,
            simulation_time=simulation_time,
            backlog_tolerance=backlog_tolerance,
            efficiency_target=efficiency_target,
            confidence_pct=confidence_pct,
            mc_runs_anchor=min(30, max(10, mc_runs // 4)),
            high=max(200, incoming_rate * 5),
        )

        if anchor_best is None:
            st.warning("Could not compute engineering anchor for the current sliders.")
        else:
            a1, a2, a3, a4 = st.columns(4)
            a1.metric("Anchor minimum stable", f"{anchor_best} parcels/min")
            a2.metric("Anchor buffered", f"{anchor_buffered} parcels/min")
            a3.metric(f"Backlog p{confidence_pct}", f"{anchor_q:.1f}")
            a4.metric(f"Efficiency p{confidence_pct}", f"{anchor_e:.2f}%")

        st.divider()
        st.subheader("AI auto-tune capacity (target stability probability)")

        target_prob = st.slider("Target stability probability (%)", 50, 99, 95, step=1) / 100.0
        search_low = st.number_input(
            "Auto-tune min capacity",
            min_value=1,
            max_value=2000,
            value=max(1, incoming_rate),
            step=1,
        )
        search_high = st.number_input(
            "Auto-tune max capacity",
            min_value=5,
            max_value=2000,
            value=max(200, incoming_rate * 5),
            step=5,
        )

        verify_mc_runs = st.number_input(
            "Verification Monte Carlo runs (optional)",
            min_value=0,
            max_value=200,
            value=30,
            step=10,
            help="Set to 0 to skip verification. Otherwise, the app will verify the AI recommendation via Monte Carlo.",
        )

        drift_warn_abs = st.slider("Calibration warning threshold (capacity difference)", 5, 80, 15, step=1)
        drift_warn_ratio = st.slider("Calibration warning threshold (AI / anchor ratio)", 1.1, 4.0, 1.6, step=0.1)

        auto_btn = st.button("AI Auto-tune capacity", key="ai_autotune_btn")

        if auto_btn:
            with st.spinner("Auto-tuning using AI predictions..."):
                best = None
                lo = clamp_int(int(search_low), 1, 2000)
                hi = clamp_int(int(search_high), lo, 2000)

                for cap in range(lo, hi + 1):
                    p = float(predict_proba(model, build_feature_row(cap))[0])
                    if p >= target_prob:
                        best = (cap, p)
                        break

                if best is None:
                    st.error("AI could not find a capacity meeting the target probability in the search range.")
                else:
                    ai_cap, ai_prob = best
                    ai_buffered = int(np.ceil(ai_cap * (1 + buffer_percent / 100)))

                    st.success(
                        f"AI recommended minimum capacity: {ai_cap} parcels/min (predicted stable: {ai_prob*100:.1f}%)"
                    )
                    st.info(f"Buffered AI recommendation ({buffer_percent}%): {ai_buffered} parcels/min")

                    final_cap = ai_cap
                    final_buffered = ai_buffered
                    calibration_flag = False
                    calibration_reason = "AI and engineering anchor are reasonably aligned."

                    if anchor_best is not None:
                        abs_diff = abs(ai_cap - anchor_best)
                        ratio = (ai_cap / anchor_best) if anchor_best > 0 else np.inf

                        st.subheader("Calibration check")
                        d1, d2, d3 = st.columns(3)
                        d1.metric("AI vs anchor difference", f"{abs_diff} capacity units")
                        d2.metric("AI / anchor ratio", f"{ratio:.2f}")
                        d3.metric("Anchor buffered", f"{anchor_buffered} parcels/min")

                        over_conservative = (ai_cap > anchor_best and abs_diff >= drift_warn_abs) or (ratio >= drift_warn_ratio)
                        under_conservative = (ai_cap < anchor_best and abs_diff >= drift_warn_abs) or (ratio <= (1 / drift_warn_ratio))

                        if over_conservative:
                            calibration_flag = True
                            calibration_reason = (
                                "AI recommendation is materially above the engineering anchor. "
                                "This suggests the AI is being over-conservative on this scenario."
                            )
                            st.warning(calibration_reason)

                            # Ground-truth correction: when AI drifts high, use the engineering anchor as the primary recommendation.
                            final_cap = anchor_best
                            final_buffered = anchor_buffered if anchor_buffered is not None else int(
                                np.ceil(anchor_best * (1 + buffer_percent / 100))
                            )
                            st.info(
                                f"Hybrid recommendation: use engineering buffered capacity {final_buffered} parcels/min "
                                f"as the primary operating recommendation, and treat the AI value ({ai_cap}) as a conservative upper estimate."
                            )

                        elif under_conservative:
                            calibration_flag = True
                            calibration_reason = (
                                "AI recommendation is materially below the engineering anchor. "
                                "This suggests the AI is under-estimating required capacity on this scenario."
                            )
                            st.warning(calibration_reason)

                            # Safety correction: when AI drifts low, still use the engineering anchor.
                            final_cap = anchor_best
                            final_buffered = anchor_buffered if anchor_buffered is not None else int(
                                np.ceil(anchor_best * (1 + buffer_percent / 100))
                            )
                            st.info(
                                f"Hybrid recommendation: use engineering buffered capacity {final_buffered} parcels/min "
                                f"as the primary operating recommendation because Monte Carlo/optimiser logic is the engineering ground truth."
                            )

                        else:
                            st.success("AI recommendation is well aligned with the engineering anchor.")

                    verified_payload = None
                    if int(verify_mc_runs) > 0:
                        st.divider()
                        st.subheader("Verification (Monte Carlo check)")

                        # Verify the actual primary operating capacity shown to the user.
                        verification_capacity = int(final_buffered)

                        with st.spinner("Verifying primary operating capacity with Monte Carlo..."):
                            mc_df_v, _ = run_monte_carlo_kpis(
                                incoming_rate=incoming_rate,
                                conveyor_capacity=verification_capacity,
                                rejection_probability=rejection_probability,
                                simulation_time=simulation_time,
                                mc_runs=int(verify_mc_runs),
                            )
                            stable_v, q_v, e_v = stability_check(
                                mc_df=mc_df_v,
                                backlog_tolerance=backlog_tolerance,
                                efficiency_target=efficiency_target,
                                percentile=confidence_pct,
                            )

                        verified_payload = {
                            "stable_verified": bool(stable_v),
                            "backlog_pctl": float(q_v),
                            "efficiency_pctl": float(e_v),
                            "verify_mc_runs": int(verify_mc_runs),
                            "verified_capacity": int(verification_capacity),
                        }

                        if stable_v:
                            st.success(
                                f"Verified stable at primary operating capacity {verification_capacity} parcels/min. "
                                f"Backlog p{confidence_pct}={q_v:.1f}, efficiency p{confidence_pct}={e_v:.2f}%."
                            )
                        else:
                            st.warning(
                                f"Verification failed at primary operating capacity {verification_capacity} parcels/min. "
                                f"Backlog p{confidence_pct}={q_v:.1f}, efficiency p{confidence_pct}={e_v:.2f}%."
                            )
                            st.write("Try increasing the AI target probability or retraining with stronger boundary-focused data.")
                    else:
                        st.caption("Verification skipped. Recommendation remains an AI fast estimate.")

                    st.divider()
                    st.subheader("Final recommendation summary")

                    r1, r2, r3 = st.columns(3)
                    r1.metric("Primary operating capacity", f"{final_buffered} parcels/min")

                    if calibration_flag:
                        r2.metric("AI conservative upper bound", f"{ai_cap} parcels/min")
                        r3.metric("Recommendation type", "Hybrid calibrated")
                    else:
                        r2.metric("AI aligned estimate", f"{ai_cap} parcels/min")
                        r3.metric("Recommendation type", "Verified" if verified_payload is not None else "AI-aligned")

                    st.info(f"Calibration note: {calibration_reason}")

                    st.divider()
                    st.subheader("Export AI recommendation report")

                    report_payload = {
                        "title": "AI Capacity Recommendation Report",
                        "created_utc": _now_iso(),
                        "sections": [
                            {
                                "heading": "Current inputs",
                                "paragraphs": [
                                    f"Incoming rate: <b>{incoming_rate}</b> parcels/min",
                                    f"Rejection probability: <b>{rejection_probability}</b>",
                                    f"Simulation time: <b>{simulation_time}</b> minutes",
                                    f"Stability rule: backlog p{confidence_pct} <= {backlog_tolerance} and efficiency p{confidence_pct} >= {efficiency_target}%",
                                ],
                            },
                            {
                                "heading": "AI outputs",
                                "paragraphs": [
                                    f"Target stability probability: <b>{target_prob*100:.1f}%</b>",
                                    f"AI minimum capacity estimate: <b>{ai_cap}</b> parcels/min",
                                    f"AI predicted stability at estimate: <b>{ai_prob*100:.1f}%</b>",
                                    f"AI buffered recommendation ({buffer_percent}%): <b>{ai_buffered}</b> parcels/min",
                                ],
                            },
                        ],
                    }

                    if anchor_best is not None:
                        report_payload["sections"].append(
                            {
                                "heading": "Engineering anchor comparison",
                                "paragraphs": [
                                    f"Anchor minimum stable capacity: <b>{anchor_best}</b> parcels/min",
                                    f"Anchor buffered capacity: <b>{anchor_buffered}</b> parcels/min",
                                    f"AI-anchor difference: <b>{abs(ai_cap - anchor_best)}</b>",
                                    f"AI-anchor ratio: <b>{(ai_cap / anchor_best) if anchor_best > 0 else 0:.2f}</b>",
                                    f"Calibration flag raised: <b>{calibration_flag}</b>",
                                    f"Calibration note: <b>{calibration_reason}</b>",
                                ],
                            }
                        )

                    report_payload["sections"].append(
                        {
                            "heading": "Final recommendation",
                            "paragraphs": [
                                f"Primary operating capacity: <b>{final_buffered}</b> parcels/min",
                                f"Engineering-selected minimum capacity: <b>{final_cap}</b> parcels/min",
                                f"Primary operating capacity after safety buffer: <b>{final_buffered}</b> parcels/min",
                                f"AI conservative upper-bound estimate: <b>{ai_cap}</b> parcels/min",
                                f"Recommendation type: <b>{'Hybrid calibrated' if calibration_flag else ('Verified' if verified_payload is not None else 'AI-aligned')}</b>",
                            ],
                        }
                    )

                    if verified_payload is not None:
                        report_payload["sections"].append(
                            {
                                "heading": "Verification (Monte Carlo)",
                                "paragraphs": [
                                    f"Monte Carlo runs: <b>{verified_payload['verify_mc_runs']}</b>",
                                    f"Verified stable: <b>{verified_payload['stable_verified']}</b>",
                                    f"Backlog p{confidence_pct}: <b>{verified_payload['backlog_pctl']:.2f}</b>",
                                    f"Efficiency p{confidence_pct}: <b>{verified_payload['efficiency_pctl']:.2f}%</b>",
                                ],
                            }
                        )

                    html_report = build_html_report(report_payload)
                    st.download_button(
                        "Download AI recommendation (HTML)",
                        data=html_report.encode("utf-8"),
                        file_name="ai_recommendation_report.html",
                        mime="text/html",
                    )

                    json_payload = {
                        "created_utc": report_payload["created_utc"],
                        "incoming_rate": incoming_rate,
                        "rejection_probability": rejection_probability,
                        "simulation_time": simulation_time,
                        "backlog_tolerance": backlog_tolerance,
                        "efficiency_target": efficiency_target,
                        "confidence_pct": confidence_pct,
                        "target_prob": float(target_prob),
                        "ai_recommended_capacity": int(ai_cap),
                        "ai_predicted_stability": float(ai_prob),
                        "primary_operating_capacity": int(final_buffered),
                        "engineering_selected_minimum_capacity": int(final_cap),
                        "verified_capacity": None if verified_payload is None else int(verified_payload["verified_capacity"]),
                        "final_recommended_capacity": int(final_buffered),
                        "buffer_percent": int(buffer_percent),
                        "final_buffered_capacity": int(final_buffered),
                        "recommendation_type": "Hybrid calibrated" if calibration_flag else ("Verified" if verified_payload is not None else "AI-aligned"),
                        "engineering_anchor_capacity": None if anchor_best is None else int(anchor_best),
                        "engineering_anchor_buffered": None if anchor_buffered is None else int(anchor_buffered),
                        "calibration_flag": bool(calibration_flag),
                        "calibration_note": calibration_reason,
                        "verification": verified_payload,
                    }
                    st.download_button(
                        
                        "Download AI recommendation (JSON)",
                        data=json.dumps(json_payload, indent=2).encode("utf-8"),
                        file_name="ai_recommendation.json",
                        mime="application/json",
                    )