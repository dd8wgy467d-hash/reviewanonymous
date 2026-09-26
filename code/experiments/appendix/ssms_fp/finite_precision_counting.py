"""Finite-precision phase drift of a 2x2 rotation SSM counting modulo N.

Two quantization regimes:
    matrix_only      the rotation matrix is quantized once and the recurrence is exact
                     (solved in closed form);
    full_recurrence  every multiply / add of the state update is also rounded.

    python -m experiments.appendix.ssms_fp.finite_precision_counting [--plot-only]

Writes figures to results/ssms_fp/figures/ and CSVs to results/ssms_fp/data/ (relative to the working
directory).
"""

import os
import csv
import math
import time
import struct
import shutil
import argparse

import numpy as np
import ml_dtypes

# ============================================================================
# Matplotlib setup
# ============================================================================

HAS_LATEX = shutil.which("latex") is not None


def plt_setup(rc=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base = {
        "text.usetex": HAS_LATEX, "font.family": "serif", "font.size": 10,
        "axes.labelsize": 11, "axes.titlesize": 12, "legend.fontsize": 8,
    }
    if rc:
        base.update(rc)
    plt.rcParams.update(base)
    return plt


def save_fig(fig, plt, out_path, dpi=300, tight=True):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if tight:
        try:
            fig.tight_layout()
        except Exception:
            pass
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_path}", flush=True)


# ============================================================================
# Floating-point formats and per-format scalar quantization
#
# fp64/fp32/fp16 round-trip through the platform's native IEEE-754 conversion
# (struct, or a no-op for fp64 since Python floats already are binary64).
# bf16/fp8-e4m3/fp8-e5m2 round-trip through ml_dtypes, which implements the
# same round-to-nearest-even IEEE-like formats. Both are exact and orders of
# magnitude faster than a hand-rolled log2/floor quantizer.
# ============================================================================

FORMATS = {
    "fp64": "fp64", "fp32": "fp32", "fp16": "fp16",
    "bf16": "bf16", "fp8-e4m3": "fp8 E4M3", "fp8-e5m2": "fp8 E5M2",
}

FMT_MARKERS = {
    "fp64": "o", "fp32": "s", "fp16": "^",
    "bf16": "D", "fp8-e4m3": "v", "fp8-e5m2": "P",
}

_ML_DTYPE = {"bf16": ml_dtypes.bfloat16, "fp8-e4m3": ml_dtypes.float8_e4m3, "fp8-e5m2": ml_dtypes.float8_e5m2}


def quantize_scalar(x, fmt):
    """Round x to the nearest representable value in `fmt` (round-to-nearest-even)."""
    if fmt == "fp64":
        return float(x)
    if fmt == "fp32":
        return struct.unpack("f", struct.pack("f", x))[0]
    if fmt == "fp16":
        try:
            return struct.unpack("e", struct.pack("e", x))[0]
        except OverflowError:
            return math.copysign(math.inf, x)
    return float(_ML_DTYPE[fmt](x))


def _make_step_fn(fmt):
    """Build a fast closure implementing one full-recurrence-quantized rotation step."""
    if fmt == "fp64":
        return lambda x0, x1, c, s: (c * x0 - s * x1, s * x0 + c * x1)

    q = lambda x: quantize_scalar(x, fmt)

    def step(x0, x1, c, s):
        p0, p1 = q(c * x0), q(s * x1)
        y0 = q(p0 - p1)
        p2, p3 = q(s * x0), q(c * x1)
        y1 = q(p2 + p3)
        return y0, y1

    return step


STEP_FUNCTIONS = {fmt: _make_step_fn(fmt) for fmt in FORMATS}


# ============================================================================
# Rotation matrix and readout
# ============================================================================

def ideal_rotation(N):
    """Ideal rotation A = [[cos, -sin], [sin, cos]] with theta = 2*pi/N."""
    theta = 2.0 * math.pi / N
    return math.cos(theta), math.sin(theta), theta


def quantized_rotation(N, fmt):
    """Quantize the matrix entries once."""
    c, s, theta = ideal_rotation(N)
    return quantize_scalar(c, fmt), quantize_scalar(s, fmt), theta


def phase_error(angle, ideal_angle):
    """Principal phase error in [-pi, pi]."""
    return (angle - ideal_angle + math.pi) % (2.0 * math.pi) - math.pi


def matrix_only_prediction(N, fmt):
    """Analytical matrix-only diagnostic: the quantized matrix is exactly r * R(phi)."""
    c, s, theta = quantized_rotation(N, fmt)
    radius = math.hypot(c, s)
    effective_theta = math.atan2(s, c)
    delta_theta = phase_error(effective_theta, theta)
    threshold = math.pi / N
    lifetime = threshold / abs(delta_theta) if delta_theta != 0.0 else math.inf
    return {
        "matrix_radius": radius, "matrix_effective_theta": effective_theta,
        "matrix_phase_error": delta_theta, "matrix_predicted_failure": lifetime,
    }


# ============================================================================
# Sampling
# ============================================================================

def make_sample_steps(max_steps, sample_points):
    """Logarithmically spaced sample points, safe up to ~10^12."""
    if max_steps <= 1:
        return np.asarray([1], dtype=np.int64)
    points = np.geomspace(1, max_steps, sample_points)
    return np.unique(np.maximum(1, points.astype(np.int64)))


# ============================================================================
# Matrix-only simulation (closed form)
#
# A_q = r * R(phi) exactly, so x_t = r^t * R(t*phi) x_0. With x_0 = (1, 0):
#   log|x_t| = t * log(r)                          (exact)
#   phase_error(t) = wrap(t * delta)                (exact, delta = phi - theta)
# No per-step float64 accumulation is needed, and the result is *more* exact
# than iterating the map t times (which accumulates O(sqrt(t)) rounding noise).
# ============================================================================

def simulate_matrix_only(N, fmt, max_steps, sample_points=10000, **_):
    c, s, theta = quantized_rotation(N, fmt)
    prediction = matrix_only_prediction(N, fmt)
    delta, radius = prediction["matrix_phase_error"], prediction["matrix_radius"]

    lifetime = prediction["matrix_predicted_failure"]
    failure = int(math.ceil(lifetime)) if lifetime <= max_steps else None

    if failure is not None:
        head = make_sample_steps(failure - 1, sample_points) if failure > 1 else np.array([], dtype=np.int64)
        ts = np.unique(np.concatenate([head, [failure]]))
    else:
        ts = make_sample_steps(max_steps, sample_points)

    log_r = math.log(radius) if radius > 0.0 else -math.inf
    tf = ts.astype(np.float64)
    radial_err = tf * log_r
    phase_err = (tf * delta + math.pi) % (2.0 * math.pi) - math.pi

    print(f"[matrix_only] [N={N:>6} {fmt:>8}] closed-form: "
          f"delta={delta:+.6e} radius={radius:.12g} "
          f"failure={'none' if failure is None else f'{failure:,}'}", flush=True)

    return {
        "N": N, "format": fmt, "mode": "matrix_only", "failure": failure,
        "steps_simulated": int(ts[-1]) if len(ts) else 0, "theta": theta,
        "c_quantized": float(c), "s_quantized": float(s),
        "ts": ts, "phase_err": phase_err, "radial_err": radial_err,
    }


# ============================================================================
# Full-recurrence simulation
#
# Every multiply/add is rounded to `fmt` using the fast step function above.
# Only the sampled and failure points pay for a log/hypot call; every other
# step pays only for the two flops plus the atan2/mod needed for the failure
# check (which cannot be skipped without breaking early-stopping).
# ============================================================================

def _log_norm(x0, x1):
    norm = math.hypot(x0, x1)
    if norm == 0.0:
        return -math.inf
    if math.isfinite(norm):
        return math.log(norm)
    return math.inf


def simulate_full_recurrence(N, fmt, max_steps, sample_points=10000, progress=True):
    c, s, theta = quantized_rotation(N, fmt)
    step_fn = STEP_FUNCTIONS[fmt]
    threshold = math.pi / N
    sample_steps = make_sample_steps(max_steps, sample_points)
    sidx = 0

    x0, x1 = 1.0, 0.0
    ts, phase_errs, radial_errs = [], [], []
    failure = None
    start_time = time.time()
    next_progress = max(1, max_steps // 20)

    for t in range(1, max_steps + 1):
        x0, x1 = step_fn(x0, x1, c, s)
        error = phase_error(math.atan2(x1, x0), t * theta)

        if abs(error) >= threshold:
            failure = t
            ts.append(t); phase_errs.append(error); radial_errs.append(_log_norm(x0, x1))
            print(f"[full_recurrence] [N={N:>6} {fmt:>8}] FAIL at step={t:,}  "
                  f"phase drift={error:+.6e}  threshold={threshold:.6e}", flush=True)
            break

        if sidx < len(sample_steps) and t >= sample_steps[sidx]:
            ts.append(t); phase_errs.append(error); radial_errs.append(_log_norm(x0, x1))
            sidx += 1

        if progress and t >= next_progress:
            rate = t / max(time.time() - start_time, 1e-12)
            print(f"[full_recurrence] [N={N:>6} {fmt:>8}] step={t:>14,d}  "
                  f"phase={error:+.4e}  rate={rate:,.0f}/s", flush=True)
            next_progress = min(max_steps, max(next_progress + 1, int(next_progress * 1.7)))

    return {
        "N": N, "format": fmt, "mode": "full_recurrence", "failure": failure,
        "steps_simulated": t, "theta": theta, "c_quantized": float(c), "s_quantized": float(s),
        "ts": np.asarray(ts, dtype=np.int64), "phase_err": np.asarray(phase_errs, dtype=float),
        "radial_err": np.asarray(radial_errs, dtype=float),
    }


def simulate(N, fmt, max_steps, mode, sample_points=10000, progress=True):
    if mode == "matrix_only":
        return simulate_matrix_only(N, fmt, max_steps, sample_points=sample_points)
    if mode == "full_recurrence":
        return simulate_full_recurrence(N, fmt, max_steps, sample_points=sample_points, progress=progress)
    raise ValueError(f"Unknown mode: {mode}")


# ============================================================================
# Result lookup
# ============================================================================

def find_result(all_results, N, fmt, mode):
    for result in all_results:
        if result["N"] == N and result["format"] == fmt and result["mode"] == mode:
            return result
    raise KeyError(f"No result: N={N}, fmt={fmt}, mode={mode}")


# ============================================================================
# CSV I/O
# ============================================================================

def write_summary_csv(all_results, data_dir):
    """One row per experiment: lifetime, analytical prediction, and final values."""
    path = os.path.join(data_dir, "drift_summary.csv")
    os.makedirs(data_dir, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "N", "format", "mode", "failure_observed", "failure_step", "steps_simulated",
            "matrix_predicted_failure", "matrix_phase_error_per_step", "matrix_radius",
            "final_phase_error", "final_log_norm",
        ])
        for r in all_results:
            p = matrix_only_prediction(r["N"], r["format"])
            phase = r["phase_err"][-1] if len(r["phase_err"]) else ""
            radial = r["radial_err"][-1] if len(r["radial_err"]) else ""
            writer.writerow([
                r["N"], r["format"], r["mode"], r["failure"] is not None,
                "" if r["failure"] is None else r["failure"], r["steps_simulated"],
                p["matrix_predicted_failure"], p["matrix_phase_error"], p["matrix_radius"],
                phase, radial,
            ])
    print(f"[data] wrote {path}", flush=True)


def write_curves_csv(all_results, data_dir):
    """Every sampled curve point; this is what --plot-only reloads."""
    path = os.path.join(data_dir, "drift_curves.csv")
    os.makedirs(data_dir, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["N", "format", "mode", "t", "phase_error", "radial_error", "failure_step"])
        for r in all_results:
            failure = "" if r["failure"] is None else r["failure"]
            for t, phase, radial in zip(r["ts"], r["phase_err"], r["radial_err"]):
                writer.writerow([r["N"], r["format"], r["mode"], int(t), float(phase), float(radial), failure])
    print(f"[data] wrote {path}", flush=True)


def load_results_from_csv(data_dir):
    """Reconstruct result dicts from drift_curves.csv. No simulation is performed."""
    curves_path = os.path.join(data_dir, "drift_curves.csv")
    summary_path = os.path.join(data_dir, "drift_summary.csv")
    if not os.path.exists(curves_path):
        raise FileNotFoundError(f"\nCould not find:\n    {curves_path}\n\nRun the experiment once without --plot-only first.")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"\nCould not find:\n    {summary_path}")

    grouped = {}
    with open(curves_path, "r", newline="") as f:
        for row in csv.DictReader(f):
            key = (int(row["N"]), row["format"], row["mode"])
            g = grouped.setdefault(key, {
                "N": key[0], "format": key[1], "mode": key[2],
                "ts": [], "phase_err": [], "radial_err": [], "failure": None,
            })
            g["ts"].append(int(row["t"]))
            g["phase_err"].append(float(row["phase_error"]))
            g["radial_err"].append(float(row["radial_error"]))
            if row["failure_step"]:
                g["failure"] = int(row["failure_step"])

    all_results = []
    for r in grouped.values():
        r["ts"] = np.asarray(r["ts"], dtype=np.int64)
        r["phase_err"] = np.asarray(r["phase_err"], dtype=float)
        r["radial_err"] = np.asarray(r["radial_err"], dtype=float)
        c, s, theta = quantized_rotation(r["N"], r["format"])
        r["theta"], r["c_quantized"], r["s_quantized"] = theta, float(c), float(s)
        r["steps_simulated"] = int(r["ts"][-1]) if len(r["ts"]) else 0
        all_results.append(r)

    print(f"[data] loaded {len(all_results)} experiment curves from {curves_path}", flush=True)
    return all_results


# ============================================================================
# Plot helpers
# ============================================================================

def finite_for_plot(values):
    values = np.asarray(values, dtype=float)
    return np.where(np.isfinite(values), values, np.nan)


def _shared_legend(fig, ax, ncol, y=1.005):
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=ncol, bbox_to_anchor=(0.5, y))


# ============================================================================
# MAIN PHASE-DRIFT FIGURE (matrix-only | full-recurrence, one row per N)
# ============================================================================

def plot_drift_curves(all_results, Ns, formats, figures_dir):
    plt = plt_setup()
    fig, axes = plt.subplots(len(Ns), 2, figsize=(11.0, 3.0 * len(Ns)), squeeze=False)

    for row, N in enumerate(Ns):
        threshold = math.pi / N
        for col, mode in enumerate(("matrix_only", "full_recurrence")):
            ax = axes[row, col]
            for fmt in formats:
                r = find_result(all_results, N, fmt, mode)
                if not len(r["ts"]):
                    continue
                y = np.clip(r["phase_err"], -threshold, threshold)
                ax.plot(r["ts"], y, marker=FMT_MARKERS.get(fmt, "o"), markevery=max(1, len(r["ts"]) // 8),
                         linewidth=1.1, markersize=3.2, label=FORMATS[fmt])
            ax.axhline(threshold, ls="--", lw=0.8, alpha=0.7, color="k")
            ax.axhline(-threshold, ls="--", lw=0.8, alpha=0.7, color="k")
            ax.set_xscale("log")
            ax.set_ylim(-threshold * 1.15, threshold * 1.15)
            ax.set_xlabel(r"Sequence length $t$")
            ax.set_ylabel(r"Phase error $\Delta\theta_t$ [rad]")
            ax.grid(True, which="both", alpha=0.2)
        axes[row, 0].set_title(fr"$N={N}$ — matrix quantized")
        axes[row, 1].set_title(fr"$N={N}$ — recurrence quantized")

    axes[0, 0].text(0.5, 1.18, "Matrix-only quantization", transform=axes[0, 0].transAxes,
                     ha="center", va="bottom", fontsize=13, fontweight="bold")
    axes[0, 1].text(0.5, 1.18, "Full-recurrence quantization", transform=axes[0, 1].transAxes,
                     ha="center", va="bottom", fontsize=13, fontweight="bold")

    _shared_legend(fig, axes[0, 0], ncol=min(len(formats), 3))
    fig.suptitle("Finite-precision phase drift", y=1.045, fontsize=15)
    save_fig(fig, plt, os.path.join(figures_dir, "drift_curves_phase.pdf"))


# ============================================================================
# RADIAL DRIFT FIGURE
# ============================================================================

def plot_radial_curves(all_results, Ns, formats, figures_dir):
    plt = plt_setup()
    fig, axes = plt.subplots(len(Ns), 2, figsize=(11.0, 3.0 * len(Ns)), squeeze=False)

    for row, N in enumerate(Ns):
        for col, mode in enumerate(("matrix_only", "full_recurrence")):
            ax = axes[row, col]
            for fmt in formats:
                r = find_result(all_results, N, fmt, mode)
                if not len(r["ts"]):
                    continue
                ax.plot(r["ts"], finite_for_plot(r["radial_err"]), marker=FMT_MARKERS.get(fmt, "o"),
                         markevery=max(1, len(r["ts"]) // 8), linewidth=1.1, markersize=3.2, label=FORMATS[fmt])
            ax.set_xscale("log")
            ax.set_xlabel(r"Sequence length $t$")
            ax.set_ylabel(r"$\log\|x_t\|_2$")
            ax.grid(True, which="both", alpha=0.2)
            ax.set_title(fr"$N={N}$ — " + ("matrix quantized" if mode == "matrix_only" else "recurrence quantized"))

    _shared_legend(fig, axes[0, 0], ncol=min(len(formats), 3))
    fig.suptitle("Finite-precision radial drift", y=1.045, fontsize=15)
    save_fig(fig, plt, os.path.join(figures_dir, "drift_curves_radial.pdf"))


# ============================================================================
# MATRIX VS FULL-RECURRENCE COMPARISON
# ============================================================================

def plot_comparison(all_results, Ns, formats, figures_dir):
    plt = plt_setup()
    fig, axes = plt.subplots(len(Ns), 1, figsize=(9.0, 3.0 * len(Ns)), squeeze=False)
    axes = axes[:, 0]

    for ax, N in zip(axes, Ns):
        threshold = math.pi / N
        for fmt in formats:
            marker = FMT_MARKERS.get(fmt, "o")
            for mode, ls, suffix in (("matrix_only", "-", " — matrix"), ("full_recurrence", "--", " — recurrence")):
                r = find_result(all_results, N, fmt, mode)
                if not len(r["ts"]):
                    continue
                y = np.clip(r["phase_err"], -threshold, threshold)
                ax.plot(r["ts"], y, linestyle=ls, marker=marker, markevery=max(1, len(r["ts"]) // 8),
                         linewidth=1.0, markersize=3, label=FORMATS[fmt] + suffix)
        ax.axhline(threshold, ls=":", lw=0.8, color="k")
        ax.axhline(-threshold, ls=":", lw=0.8, color="k")
        ax.set_xscale("log")
        ax.set_ylim(-threshold * 1.15, threshold * 1.15)
        ax.set_xlabel(r"Sequence length $t$")
        ax.set_ylabel(r"$\Delta\theta_t$ [rad]")
        ax.set_title(fr"$N={N}$")
        ax.grid(True, which="both", alpha=0.2)

    _shared_legend(fig, axes[0], ncol=2)
    fig.suptitle("Matrix-only vs full-recurrence quantization", y=1.01, fontsize=14)
    save_fig(fig, plt, os.path.join(figures_dir, "drift_curves_comparison.pdf"))


# ============================================================================
# STATE-SPACE TRAJECTORIES
#
# One column per precision, one row per N; markers only (no connecting lines),
# colored by normalized time t/t_end so a single colorbar is meaningful across
# panels that span wildly different lifetimes; the theoretical fixed points
# (the N ideal readout targets on the unit circle) are overlaid as stars.
# ============================================================================

def _draw_readout_boundaries(ax, N, theta, r_max):
    """Thin guides at the N ideal decision boundaries, halfway between fixed points."""
    for k in range(N):
        b = k * theta - theta / 2.0
        ax.plot([0.0, r_max * math.cos(b)], [0.0, r_max * math.sin(b)],
                 color="0.6", linewidth=0.5, linestyle=":", alpha=0.6, zorder=0)


def plot_state_trajectories(all_results, Ns, formats, mode, figures_dir):
    plt = plt_setup()
    n_rows, n_cols = len(Ns), len(formats)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols, 2.6 * n_rows), squeeze=False)

    sc = None
    for i, N in enumerate(Ns):
        theta = 2.0 * math.pi / N
        fixed_x = np.cos(np.arange(N) * theta)
        fixed_y = np.sin(np.arange(N) * theta)

        for j, fmt in enumerate(formats):
            ax = axes[i, j]
            r = find_result(all_results, N, fmt, mode)

            r_max = 1.15
            if len(r["ts"]):
                angles = r["ts"] * theta + r["phase_err"]
                radii = np.exp(np.clip(r["radial_err"], -50.0, 50.0))
                r_max = max(1.15, 1.1 * float(radii.max()))
                t_end = r["ts"][-1] if r["ts"][-1] > 0 else 1
                sc = ax.scatter(radii * np.cos(angles), radii * np.sin(angles), c=r["ts"] / t_end,
                                 cmap="viridis_r", vmin=0.0, vmax=1.0, s=8, linewidths=0, zorder=3)

            _draw_readout_boundaries(ax, N, theta, r_max)
            ax.scatter(fixed_x, fixed_y, marker="*", s=70, c="black", zorder=1,
                       label="theoretical fixed points" if (i, j) == (0, 0) else None)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(FORMATS[fmt], fontsize=10)
            if j == 0:
                ax.set_ylabel(fr"$N={N}$", fontsize=10)

    if sc is not None:
        cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.6, pad=0.015)
        cbar.set_label(r"$t / t_{\mathrm{end}}$")

    _shared_legend(fig, axes[0, 0], ncol=1, y=1.02)
    fig.suptitle(f"State-space trajectories ({mode.replace('_', ' ')})", y=1.04, fontsize=14)
    save_fig(fig, plt, os.path.join(figures_dir, f"state_trajectories_{mode}.pdf"), tight=False)


# ============================================================================
# NORMALIZED MATRIX-ONLY TRAJECTORY (fp8-E4M3)
#
# Same style as plot_state_trajectories, but restricted to the matrix-only
# mode and a single format, and with the radius forced to 1 so only the
# angular (phase) drift is visible, undistorted by the radial spiral.
# ============================================================================

def plot_matrix_trajectory_normalized(all_results, Ns, figures_dir, fmt="fp8-e4m3"):
    if not any(r["format"] == fmt and r["mode"] == "matrix_only" for r in all_results):
        return

    plt = plt_setup()
    fig, axes = plt.subplots(1, len(Ns), figsize=(2.6 * len(Ns), 2.9), squeeze=False)
    axes = axes[0]

    sc = None
    for ax, N in zip(axes, Ns):
        theta = 2.0 * math.pi / N
        fixed_x = np.cos(np.arange(N) * theta)
        fixed_y = np.sin(np.arange(N) * theta)

        r = find_result(all_results, N, fmt, "matrix_only")
        if len(r["ts"]):
            angles = r["ts"] * theta + r["phase_err"]
            t_end = r["ts"][-1] if r["ts"][-1] > 0 else 1
            sc = ax.scatter(np.cos(angles), np.sin(angles), c=r["ts"] / t_end,
                             cmap="viridis_r", vmin=0.0, vmax=1.0, s=10, linewidths=0, zorder=3)

        _draw_readout_boundaries(ax, N, theta, r_max=1.15)
        ax.scatter(fixed_x, fixed_y, marker="*", s=70, c="black", zorder=1,
                   label="theoretical fixed points" if N == Ns[0] else None)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(fr"$N={N}$", fontsize=10)

    if sc is not None:
        cbar = fig.colorbar(sc, ax=axes.tolist(), shrink=0.7, pad=0.02)
        cbar.set_label(r"$t / t_{\mathrm{end}}$")

    _shared_legend(fig, axes[0], ncol=1, y=1.14)
    fig.suptitle(f"Matrix-only trajectory, normalized radius ({FORMATS[fmt]})", y=1.22, fontsize=13)
    save_fig(fig, plt, os.path.join(figures_dir, f"state_trajectory_matrix_only_{fmt.replace('-', '_')}_normalized.pdf"),
             tight=False)


# ============================================================================
# FAILURE LIFETIME
# ============================================================================

def plot_failure_lifetime(all_results, Ns, formats, figures_dir):
    plt = plt_setup()
    x = np.arange(len(formats))
    width = 0.36
    fig, axes = plt.subplots(len(Ns), 1, figsize=(8.0, 3.0 * len(Ns)), squeeze=False)
    axes = axes[:, 0]

    for ax, N in zip(axes, Ns):
        matrix_failures, full_failures, predictions = [], [], []
        for fmt in formats:
            rm = find_result(all_results, N, fmt, "matrix_only")
            rf = find_result(all_results, N, fmt, "full_recurrence")
            matrix_failures.append(np.nan if rm["failure"] is None else rm["failure"])
            full_failures.append(np.nan if rf["failure"] is None else rf["failure"])
            predictions.append(matrix_only_prediction(N, fmt)["matrix_predicted_failure"])

        ax.bar(x - width / 2, matrix_failures, width, label="Matrix only")
        ax.bar(x + width / 2, full_failures, width, label="Full recurrence")
        ax.plot(x, predictions, linestyle="none", marker="x", markersize=6, markeredgewidth=1.2,
                 label="Matrix-only analytical estimate")
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([FORMATS[f] for f in formats])
        ax.set_ylabel(r"First failure $T_{\mathrm{fail}}$")
        ax.set_title(fr"$N={N}$")
        ax.grid(True, which="both", axis="y", alpha=0.2)
        ax.legend(frameon=False, fontsize=7.5)

    axes[-1].set_xlabel("Floating-point format")
    save_fig(fig, plt, os.path.join(figures_dir, "failure_lifetime.pdf"))


# ============================================================================
# Generate all figures / run experiment / plot-only / CLI
# ============================================================================

def generate_all_figures(all_results, Ns, formats, figures_dir):
    plot_failure_lifetime(all_results, Ns, formats, figures_dir)
    plot_drift_curves(all_results, Ns, formats, figures_dir)
    plot_radial_curves(all_results, Ns, formats, figures_dir)
    plot_comparison(all_results, Ns, formats, figures_dir)
    plot_state_trajectories(all_results, Ns, formats, "matrix_only", figures_dir)
    plot_state_trajectories(all_results, Ns, formats, "full_recurrence", figures_dir)
    plot_matrix_trajectory_normalized(all_results, Ns, figures_dir, fmt="fp8-e4m3")


def run_experiment(figures_dir, data_dir, Ns, formats, max_steps, sample_points):
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    all_results = []

    for N in Ns:
        for fmt in formats:
            p = matrix_only_prediction(N, fmt)
            print("\n" + "=" * 80, flush=True)
            print(f"N={N}, format={fmt}", flush=True)
            print(f"matrix-only phase error/step = {p['matrix_phase_error']:+.6e}", flush=True)
            print(f"matrix radius = {p['matrix_radius']:.12g}", flush=True)
            print(f"matrix-only predicted failure = {p['matrix_predicted_failure']:.6e}", flush=True)
            print("=" * 80, flush=True)

            for mode in ("matrix_only", "full_recurrence"):
                print(f"\n=== Running N={N}, fmt={fmt}, mode={mode} ===", flush=True)
                all_results.append(simulate(N=N, fmt=fmt, max_steps=max_steps, mode=mode,
                                             sample_points=sample_points, progress=True))

    write_summary_csv(all_results, data_dir)
    write_curves_csv(all_results, data_dir)
    generate_all_figures(all_results, Ns, formats, figures_dir)

    print("\nDone.", flush=True)
    print("Figures written to:", os.path.abspath(figures_dir), flush=True)
    print("Data written to:", os.path.abspath(data_dir), flush=True)


def run_plot_only(figures_dir, data_dir):
    print("\n" + "=" * 80, flush=True)
    print("PLOT-ONLY MODE", flush=True)
    print("Loading previously generated CSV data...", flush=True)

    all_results = load_results_from_csv(data_dir)
    if not all_results:
        raise RuntimeError("No results found in CSV.")

    Ns = sorted({r["N"] for r in all_results})
    present = {r["format"] for r in all_results}
    formats = [f for f in ("fp64", "fp32", "bf16", "fp16", "fp8-e4m3", "fp8-e5m2") if f in present]

    missing = {"matrix_only", "full_recurrence"} - {r["mode"] for r in all_results}
    if missing:
        raise RuntimeError("CSV is missing modes: " + ", ".join(sorted(missing)))

    print(f"[plot-only] N values: {Ns}", flush=True)
    print(f"[plot-only] formats: {formats}", flush=True)
    print("[plot-only] No simulations will be run.", flush=True)

    generate_all_figures(all_results, Ns, formats, figures_dir)
    print("\nPlot-only mode finished.", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Finite-precision rotation SSM drift experiment.")
    parser.add_argument("--plot-only", action="store_true",
                         help="Load CSVs from --data-dir and regenerate figures without simulating.")
    parser.add_argument("--figures-dir", type=str, default="results/ssms_fp/figures",
                         help="Directory for output figures (relative to cwd).")
    parser.add_argument("--data-dir", type=str, default="results/ssms_fp/data",
                         help="Directory for output CSVs (relative to cwd).")
    parser.add_argument("--max-steps", type=int, default=10 ** 8, help="Maximum recurrence steps. Default: 10^8.")
    parser.add_argument("--sample-points", type=int, default=10000, help="Log-spaced curve samples per experiment.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    Ns = (3, 5, 7, 11)
    formats = ("fp64", "fp32", "bf16", "fp16", "fp8-e4m3", "fp8-e5m2")

    if args.plot_only:
        run_plot_only(figures_dir=args.figures_dir, data_dir=args.data_dir)
    else:
        run_experiment(figures_dir=args.figures_dir, data_dir=args.data_dir, Ns=Ns, formats=formats,
                        max_steps=args.max_steps, sample_points=args.sample_points)
