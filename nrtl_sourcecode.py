"""
NRTL ternary fitter for ln(gamma1) with optional ternary (E) terms.

Outputs
-------
- *_params.csv : parameter estimates + SE + 95% CI (SE=NaN if fixed)
- *_predictions.csv : per-sample ln_gamma1_pred and residuals
- *_fit_report.txt : metrics (RAD/RMSD/R^2) and solver info
- (optional) *_alpha_scan.csv and *_alpha_heatmap.png for alpha scan

Example
-------
python nrtl_fit_ln_gamma1.py --csv EmimOAc_NRTL.csv --alpha12 0.225 --alpha13 0.300 --alpha23 0.10243 --fix-b23 -3090.0890064950686 --fix-b32 3480.635073370219  --fix-b13 433.4632 --fix-b31 5620.47 --use-e --lb-b -20000 --ub-b 20000 --lb-e -100 --ub-e 100
  --plot-residuals --verbose
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import matplotlib.pyplot as plt


# -------------------- Data I/O --------------------

def read_data(csv_path: Path):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    for col in ("x1", "x2", "t"):
        if col not in df.columns:
            raise ValueError(f"CSV must include columns: x1, x2, t (Kelvin). Missing: {col}")
    if "x3" not in df.columns:
        df["x3"] = 1.0 - df["x1"] - df["x2"]

    # target
    if "ln_gamma1" in df.columns:
        y = df["ln_gamma1"].astype(float).to_numpy()
    elif "gamma1" in df.columns:
        g = df["gamma1"].astype(float).to_numpy()
        if np.any(g <= 0):
            raise ValueError("gamma1 has non-positive values; cannot take log.")
        y = np.log(g)
    else:
        raise ValueError("CSV must contain 'ln_gamma1' or 'gamma1'.")

    x1 = df["x1"].astype(float).to_numpy()
    x2 = df["x2"].astype(float).to_numpy()
    x3 = df["x3"].astype(float).to_numpy()
    T = df["t"].astype(float).to_numpy()

    # normalize compositions if slightly off
    sumx = x1 + x2 + x3
    if np.max(np.abs(sumx - 1.0)) > 5e-3:
        x1 = x1 / sumx
        x2 = x2 / sumx
        x3 = x3 / sumx
    X = np.stack([x1, x2, x3], axis=1)
    return df, X, T, y


def build_alpha(alpha12, alpha13, alpha23):
    A = np.zeros((3, 3), dtype=float)
    A[0, 1] = A[1, 0] = alpha12
    A[0, 2] = A[2, 0] = alpha13
    A[1, 2] = A[2, 1] = alpha23
    return A


# -------------------- Model --------------------

def ln_gamma1_core(b_all, X, T, alpha):
    """
    Multicomponent NRTL ln(gamma1) (no ternary E terms).
    b_all: [b12,b21,b13,b31,b23,b32]
    """
    n = X.shape[0]
    tau = np.zeros((n, 3, 3), dtype=float)
    tau[:, 0, 1] = b_all[0] / T
    tau[:, 1, 0] = b_all[1] / T
    tau[:, 0, 2] = b_all[2] / T
    tau[:, 2, 0] = b_all[3] / T
    tau[:, 1, 2] = b_all[4] / T
    tau[:, 2, 1] = b_all[5] / T

    G = np.exp(-alpha * tau)
    for i in range(3):
        G[:, i, i] = 1.0

    x = X
    G_T = np.transpose(G, (0, 2, 1))

    S = np.einsum("nj,nji->ni", x, G_T)
    num1 = np.einsum("nj,nji,nji->ni", x, tau.transpose(0, 2, 1), G_T)
    sum1 = num1 / S
    sum1_i = sum1[:, 0]

    denom_j = np.einsum("nk,nkj->nj", x, G)
    termA = (x[:, None, :] * G) / denom_j[:, None, :]
    numBj = np.einsum("nm,nmj,nmj->nj", x, tau, G)
    Bj = numBj / denom_j
    sum2_i = np.sum(termA[:, 0, :] * (tau[:, 0, :] - Bj), axis=1)

    return sum1_i + sum2_i


def ln_gamma1_with_E(b_all, E, X, T, alpha):
    ln_g1 = ln_gamma1_core(b_all, X, T, alpha)
    x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
    E1, E2, E3 = E
    ln_g1 += (
        2 * E1 * x1 * x2 * x3 - 3 * E1 * (x1 ** 2) * x2 * x3
        + E2 * (x2 ** 2) * x3 - 3 * E2 * x1 * (x2 ** 2) * x3
        + E3 * x2 * (x3 ** 2) - 3 * E3 * x1 * x2 * (x3 ** 2)
    )
    return ln_g1


# -------------------- SE & Reporting --------------------

def compute_se(res_obj):
    J = res_obj.jac
    r = res_obj.fun
    n, p = J.shape
    dof = max(n - p, 1)
    sigma2 = float(np.sum(r * r)) / dof
    JTJ = J.T @ J
    try:
        cov = sigma2 * np.linalg.inv(JTJ)
    except np.linalg.LinAlgError:
        cov = sigma2 * np.linalg.pinv(JTJ)
    SE = np.sqrt(np.clip(np.diag(cov), 0, None))
    return SE, cov, sigma2, dof


# -------------------- Fit --------------------

def fit_model(args):
    df, X, T, y = read_data(args.csv)
    alpha = build_alpha(args.alpha12, args.alpha13, args.alpha23)

    # Order: [b12,b21,b13,b31,b23,b32]
    b13_val = args.fix_b13 if args.fix_b13 is not None else args.init_b13
    b31_val = args.fix_b31 if args.fix_b31 is not None else args.init_b31
    b23_val = args.fix_b23 if args.fix_b23 is not None else args.init_b23
    b32_val = args.fix_b32 if args.fix_b32 is not None else args.init_b32

    b_all_init = np.array([
        args.init_b12, args.init_b21, b13_val, b31_val, b23_val, b32_val
    ], dtype=float)

    fixed_mask = np.array([
        False,                       # b12 free
        False,                       # b21 free
        args.fix_b13 is not None,    # b13 fixed?
        args.fix_b31 is not None,    # b31 fixed?
        args.fix_b23 is not None,    # b23 fixed?
        args.fix_b32 is not None,    # b32 fixed?
    ], dtype=bool)

    lb_all = np.array([args.lb_b] * 6, dtype=float)
    ub_all = np.array([args.ub_b] * 6, dtype=float)

    free_idx = np.where(~fixed_mask)[0]
    b_free_init = b_all_init[free_idx]
    lb_free = lb_all[free_idx]
    ub_free = ub_all[free_idx]

    use_E = args.use_e
    E_init = np.array([args.init_e1, args.init_e2, args.init_e3], dtype=float)
    E_lb = np.array([args.lb_e] * 3, dtype=float)
    E_ub = np.array([args.ub_e] * 3, dtype=float)

    if use_E:
        theta0 = np.concatenate([b_free_init, E_init])
        lb = np.concatenate([lb_free, E_lb])
        ub = np.concatenate([ub_free, E_ub])
    else:
        theta0 = b_free_init.copy()
        lb = lb_free.copy()
        ub = ub_free.copy()

    def residuals(theta):
        if use_E:
            b_free = theta[:-3]
            E = theta[-3:]
        else:
            b_free = theta
            E = np.zeros(3, dtype=float)
        b_all = b_all_init.copy()
        b_all[free_idx] = b_free
        ln_pred = ln_gamma1_with_E(b_all, E, X, T, alpha) if use_E else ln_gamma1_core(b_all, X, T, alpha)
        return ln_pred - y

    res = least_squares(
        residuals, theta0, bounds=(lb, ub), method="trf",
        max_nfev=args.max_nfev, ftol=1e-12, xtol=1e-12, gtol=1e-12,
        verbose=2 if args.verbose else 0
    )
    theta_hat = res.x

    # unpack estimates
    if use_E:
        b_free_hat = theta_hat[:-3]
        E_hat = theta_hat[-3:]
    else:
        b_free_hat = theta_hat
        E_hat = np.zeros(3, dtype=float)

    b_all_hat = b_all_init.copy()
    b_all_hat[free_idx] = b_free_hat

    # predictions
    ln_pred = ln_gamma1_with_E(b_all_hat, E_hat, X, T, alpha) if use_E else ln_gamma1_core(b_all_hat, X, T, alpha)
    resid = ln_pred - y
    RAD_ln = float(np.mean(np.abs(resid)))
    RMSD_ln = float(np.sqrt(np.mean(resid ** 2)))
    # R^2 on ln(gamma1)
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # SEs for the optimized theta (free b's and E's)
    SE_theta, cov, sigma2, dof = compute_se(res)

    # Build parameter table with SE mapped back
    all_names = ["b12", "b21", "b13", "b31", "b23", "b32"]
    param_rows = []

    se_cursor = 0
    for k, nm in enumerate(all_names):
        if k in free_idx:
            est = float(b_all_hat[k])
            se = float(SE_theta[se_cursor])
            se_cursor += 1 if se_cursor < len(SE_theta) else 0
            param_rows.append((nm, est, se))
        else:
            est = float(b_all_hat[k])
            param_rows.append((nm + " (fixed)", est, np.nan))

    if use_E:
        # remaining 3 SE correspond to E1,E2,E3
        e_SE = SE_theta[-3:]
        for nm, est, se in zip(["E1", "E2", "E3"], E_hat.tolist(), e_SE.tolist()):
            param_rows.append((nm, float(est), float(se)))

    z = 1.96
    params_df = pd.DataFrame(param_rows, columns=["param", "estimate", "SE"])
    params_df["CI95_lower"] = params_df["estimate"] - z * params_df["SE"]
    params_df["CI95_upper"] = params_df["estimate"] + z * params_df["SE"]

    out_prefix = args.out_prefix if args.out_prefix else Path(args.csv).stem + "_nrtl"

    # save params
    params_csv = Path(out_prefix + "_params.csv")
    params_df.to_csv(params_csv, index=False)

    # save predictions
    pred_df = df.copy()
    pred_df["ln_gamma1_pred"] = ln_pred
    pred_df["residual_ln"] = resid
    preds_csv = Path(out_prefix + "_predictions.csv")
    pred_df.to_csv(preds_csv, index=False)

    # report
    report_path = Path(out_prefix + "_fit_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("NRTL ternary fit for ln(gamma1)\n")
        f.write(f"CSV: {args.csv}\n")
        f.write(f"Use E terms: {use_E}\n")
        f.write(f"Alpha matrix (alpha12={args.alpha12}, alpha13={args.alpha13}, alpha23={args.alpha23}):\n")
        f.write(str(build_alpha(args.alpha12, args.alpha13, args.alpha23)) + "\n\n")
        f.write("Parameters (estimate, SE, 95% CI):\n")
        for _, row in params_df.iterrows():
            if pd.isna(row["SE"]):
                f.write(f"  {row['param']:>10s}: {row['estimate']: .8f} (fixed)\n")
            else:
                f.write(f"  {row['param']:>10s}: {row['estimate']: .8f}, SE={row['SE']: .8f}, "
                        f"95% CI=({row['CI95_lower']: .8f}, {row['CI95_upper']: .8f})\n")
        f.write("\nMetrics:\n")
        f.write(f"  RAD_ln  = {RAD_ln:.6f}\n  RMSD_ln = {RMSD_ln:.6f}\n")
        f.write(f"  R2_ln   = {R2:.6f}\n")
        f.write("\nFit internals:\n")
        f.write(f"  n={len(y)}, p={len(theta_hat)}, dof={max(len(y)-len(theta_hat),1)}\n")
        f.write(f"  sigma2={sigma2:.6e}\n")
        f.write(f"  success={res.success}, message={res.message}\n")

    # optional residual plot
    if args.plot_residuals:
        plt.figure(figsize=(6, 5))
        sc = plt.scatter(X[:, 1], X[:, 2], c=resid, s=28)
        plt.xlabel("x2"); plt.ylabel("x3")
        plt.title("Residual (ln γ1) in x2–x3 plane")
        plt.colorbar(sc, label="residual ln")
        plt.tight_layout()
        plot_path = Path(out_prefix + "_residuals_x2x3.png")
        plt.savefig(plot_path, dpi=200)
        plt.close()

    return {
        "params_csv": str(params_csv),
        "preds_csv": str(preds_csv),
        "report": str(report_path),
        "RAD_ln": RAD_ln,
        "RMSD_ln": RMSD_ln,
        "R2_ln": R2,
    }


# -------------------- Alpha Scan (optional) --------------------

def scan_alphas(args):
    df, X, T, y = read_data(args.csv)

    a12_vals = np.arange(args.a12_min, args.a12_max + 1e-12, args.a12_step)
    a13_vals = np.arange(args.a13_min, args.a13_max + 1e-12, args.a13_step)

    # Fixed b's honored here as well
    b13_val = args.fix_b13 if args.fix_b13 is not None else args.init_b13
    b31_val = args.fix_b31 if args.fix_b31 is not None else args.init_b31
    b23_val = args.fix_b23 if args.fix_b23 is not None else args.init_b23
    b32_val = args.fix_b32 if args.fix_b32 is not None else args.init_b32

    b_all_init = np.array([
        args.init_b12, args.init_b21, b13_val, b31_val, b23_val, b32_val
    ], dtype=float)

    fixed_mask = np.array([
        False,                       # b12 free
        False,                       # b21 free
        args.fix_b13 is not None,    # b13 fixed?
        args.fix_b31 is not None,    # b31 fixed?
        args.fix_b23 is not None,    # b23 fixed?
        args.fix_b32 is not None,    # b32 fixed?
    ], dtype=bool)

    free_idx = np.where(~fixed_mask)[0]

    use_E = args.use_e
    theta0 = b_all_init[free_idx].copy()
    if use_E:
        theta0 = np.concatenate([theta0, np.array([args.init_e1, args.init_e2, args.init_e3], dtype=float)])

    lb_b = np.array([args.lb_b] * len(free_idx), dtype=float)
    ub_b = np.array([args.ub_b] * len(free_idx), dtype=float)
    if use_E:
        lb = np.concatenate([lb_b, np.array([args.lb_e] * 3, dtype=float)])
        ub = np.concatenate([ub_b, np.array([args.ub_e] * 3, dtype=float)])
    else:
        lb, ub = lb_b, ub_b

    results = []
    heat = np.zeros((len(a12_vals), len(a13_vals)), dtype=float)

    def residuals(theta, alpha):
        if use_E:
            b_free = theta[:-3]
            E = theta[-3:]
        else:
            b_free = theta
            E = np.zeros(3, dtype=float)
        b_all = b_all_init.copy()
        b_all[free_idx] = b_free
        ln_pred = ln_gamma1_with_E(b_all, E, X, T, alpha) if use_E else ln_gamma1_core(b_all, X, T, alpha)
        return ln_pred - y

    for i, a12 in enumerate(a12_vals):
        init_row = theta0.copy()
        for j, a13 in enumerate(a13_vals):
            alpha = build_alpha(a12, a13, args.alpha23)
            res = least_squares(lambda th: residuals(th, alpha),
                                init_row, bounds=(lb, ub), method="trf",
                                max_nfev=args.max_nfev, ftol=1e-10, xtol=1e-10, gtol=1e-10, verbose=0)
            th = res.x
            init_row = th.copy()  # warm start next column

            # unpack to compute metrics + store params
            if use_E:
                b_free = th[:-3]
                E = th[-3:]
            else:
                b_free = th
                E = np.zeros(3, dtype=float)

            b_all = b_all_init.copy()
            b_all[free_idx] = b_free

            ln_pred = ln_gamma1_with_E(b_all, E, X, T, alpha) if use_E else ln_gamma1_core(b_all, X, T, alpha)
            resid = ln_pred - y
            RAD_ln = float(np.mean(np.abs(resid)))
            RMSD_ln = float(np.sqrt(np.mean(resid ** 2)))
            ss_res = float(np.sum(resid ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

            heat[i, j] = RMSD_ln
            row = {
                "alpha12": float(a12),
                "alpha13": float(a13),
                "RMSD_ln": RMSD_ln,
                "RAD_ln": RAD_ln,
                "R2_ln": R2,
            }
            names_b = ["b12", "b21", "b13", "b31", "b23", "b32"]
            for idx, nm in enumerate(names_b):
                row[nm] = float(b_all[idx])
            if use_E:
                row["E1"], row["E2"], row["E3"] = float(E[0]), float(E[1]), float(E[2])
            results.append(row)

    out_prefix = args.out_prefix if args.out_prefix else Path(args.csv).stem + "_nrtl"

    grid_csv = Path(out_prefix + "_alpha_scan.csv")
    pd.DataFrame(results).sort_values("RMSD_ln").to_csv(grid_csv, index=False)

    # heatmap
    plt.figure(figsize=(6, 5))
    im = plt.imshow(
        heat.T, origin="lower",
        extent=[a12_vals[0], a12_vals[-1], a13_vals[0], a13_vals[-1]], aspect="auto"
    )
    plt.xlabel("alpha12 (=alpha21)")
    plt.ylabel("alpha13 (=alpha31)")
    plt.title("RMSD (ln gamma1) heatmap")
    plt.colorbar(im, label="RMSD_ln")
    plt.tight_layout()
    heat_png = Path(out_prefix + "_alpha_heatmap.png")
    plt.savefig(heat_png, dpi=200)
    plt.close()

    return {"grid_csv": str(grid_csv), "heat_png": str(heat_png)}


# -------------------- CLI --------------------

def main():
    ap = argparse.ArgumentParser(description="Fit ternary NRTL ln(gamma1) with optional E terms.")
    ap.add_argument("--csv", type=Path, required=True, help="Input CSV with x1,x2,(x3), t, ln_gamma1 or gamma1")

    # alpha (symmetric) and fixed b options
    ap.add_argument("--alpha12", type=float, default=0.2)
    ap.add_argument("--alpha13", type=float, default=0.2)
    ap.add_argument("--alpha23", type=float, default=0.2)

    ap.add_argument("--fix-b13", type=float, default=None, help="Fix b13 (K)")
    ap.add_argument("--fix-b31", type=float, default=None, help="Fix b31 (K)")
    ap.add_argument("--fix-b23", type=float, default=None, help="Fix b23 (K)")
    ap.add_argument("--fix-b32", type=float, default=None, help="Fix b32 (K)")

    # initial guesses
    ap.add_argument("--init-b12", type=float, default=300.0)
    ap.add_argument("--init-b21", type=float, default=300.0)
    ap.add_argument("--init-b13", type=float, default=300.0)
    ap.add_argument("--init-b31", type=float, default=300.0)
    ap.add_argument("--init-b23", type=float, default=300.0)
    ap.add_argument("--init-b32", type=float, default=300.0)

    # bounds
    ap.add_argument("--lb-b", type=float, default=-5000.0)
    ap.add_argument("--ub-b", type=float, default=5000.0)

    # ternary E terms
    ap.add_argument("--use-e", action="store_true")
    ap.add_argument("--init-e1", type=float, default=0.0)
    ap.add_argument("--init-e2", type=float, default=0.0)
    ap.add_argument("--init-e3", type=float, default=0.0)
    ap.add_argument("--lb-e", type=float, default=-20.0)
    ap.add_argument("--ub-e", type=float, default=20.0)

    # fitting/output
    ap.add_argument("--max-nfev", type=int, default=80000)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out-prefix", type=str, default=None)
    ap.add_argument("--plot-residuals", action="store_true")

    # alpha scan
    ap.add_argument("--scan-alphas", action="store_true")
    ap.add_argument("--a12-min", type=float, default=0.05)
    ap.add_argument("--a12-max", type=float, default=0.40)
    ap.add_argument("--a12-step", type=float, default=0.025)
    ap.add_argument("--a13-min", type=float, default=0.05)
    ap.add_argument("--a13-max", type=float, default=0.40)
    ap.add_argument("--a13-step", type=float, default=0.025)

    args = ap.parse_args()

    res = fit_model(args)
    print("Fit done.")
    print(f"  Params CSV : {res['params_csv']}")
    print(f"  Preds  CSV : {res['preds_csv']}")
    print(f"  Report TXT : {res['report']}")
    print(f"  RAD_ln={res['RAD_ln']:.6f}, RMSD_ln={res['RMSD_ln']:.6f}, R2_ln={res['R2_ln']:.6f}")

    if args.scan_alphas:
        scan = scan_alphas(args)
        print("Alpha scan done.")
        print(f"  Grid CSV   : {scan['grid_csv']}")
        print(f"  Heatmap PNG: {scan['heat_png']}")


if __name__ == "__main__":
    main()
