from podio import root_io
import ROOT
import glob
import pickle
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import functions
import math
import random
from collections import defaultdict

ROOT.gROOT.SetBatch(True)

# === Command-line args ===
parser = argparse.ArgumentParser()
parser.add_argument('--run', action='store_true', help='Process and save muon, signal, and background files')
parser.add_argument('--plots', action='store_true', help='Plots')
parser.add_argument('--classify', action='store_true', help='Train and evaluate a classifier')
args = parser.parse_args()

# === Geometry config ===
PITCH = functions.PITCH_MM
RADIUS = functions.RADIUS_MM
LAYER_RADII = [14, 36, 58]
TARGET_LAYER = 0

"""if args.run:
    all_configs = {
        'muons': {
            'files': glob.glob('/ceph/submit/data/user/j/jaeyserm/fccee/beam_backgrounds/CLD_o2_v05/mu_theta_0-180_p_50/*.root'),
            'outfile': 'ABmuons_edep.pkl'
        },
        'signal': {
            'files': glob.glob('/ceph/submit/data/group/fcc/ee/detector/VTXStudiesFullSim/CLD_wz3p6_ee_qq_ecm91p2/*.root'),
            'outfile': 'ABsignal_edep.pkl'
        },
        'background': {
            'files': glob.glob('/ceph/submit/data/group/fcc/ee/detector/VTXStudiesFullSim/CLD_o2_v05/FCCee_Z_4IP_04may23_FCCee_Z/*.root'),
            'outfile': 'ABbkg_edep.pkl'
        }
    }

    seen_cellids = set()
    unknown_cellids = set()

    for label, config in all_configs.items():
        files = config['files']
        outfile = config['outfile']
        cluster_metrics = []
        limit = {
            'muons': 100,
            'signal': 50,
            'background': 500
        }[label]

        for i, filename in enumerate(files):
            if i >= limit:
                break
            print(f"[{label.upper()}] Processing file {i+1}/{limit}: {filename}")
            reader = root_io.Reader(filename)
            events = reader.get('events')

            for event in events:
                module_hits = defaultdict(list)

                for hit in event.get('VertexBarrelCollection'):
                    try:
                        if functions.radius_idx(hit, LAYER_RADII) != TARGET_LAYER:
                            continue
                        if hit.isProducedBySecondary():
                            continue
                        pos = hit.getPosition()
                        mc = hit.getMCParticle()
                        if mc is None:
                            continue
                        trackID = mc.getObjectID().index
                        energy = mc.getEnergy()
                        pid = mc.getPDG()
                        try:
                            edep = hit.getEDep()
                        except AttributeError:
                            edep = 0

                        h = functions.Hit(x=pos.x, y=pos.y, z=pos.z, energy=energy, edep=edep, trackID=trackID)

                        cellID = hit.getCellID()
                        group_id = functions.cellid_to_group(cellID)
                        seen_cellids.add(cellID)

                        if group_id is None:
                            unknown_cellids.add(cellID)
                            continue  # skip cellIDs not in defined 16 groups

                        # Group hits by shared cellID group
                        module_hits[group_id].append((trackID, h, pid))

                    except Exception as e:
                        print(f"Skipping hit due to error: {e}")

                # Cluster hits per merged group
                for group_id, hit_group in module_hits.items():
                    particles = {}
                    for trackID, h, pid in hit_group:
                        key = (trackID, group_id)
                        if key not in particles:
                            particles[key] = functions.Particle(trackID=trackID, cellID=group_id, pid=pid)
                        particles[key].add_hit(h)

                    for p in particles.values():
                        #if len(p.hits) == 2:
                        #    p.hits = functions.merge_cluster_hits(p.hits)
                        multiplicity = len(p.hits)
                        total_edep = p.total_energy()
                        b_x, b_y, b_z = functions.geometric_baricenter(p.hits)
                        cos_theta = functions.cos_theta(b_x, b_y, b_z)
                        mc_energy = p.hits[0].energy
                        z_ext = p.z_extent()
                        nrows = p.n_phi_rows(PITCH, RADIUS)
                        cluster_metrics.append((z_ext, nrows, multiplicity, total_edep, mc_energy, cos_theta, b_x, b_y, p.pid))

        with open(outfile, 'wb') as f:
            pickle.dump(cluster_metrics, f)
        print(f"✅ Saved {label} clusters to {outfile}")

    # Report cellIDs not in your defined mapping
    if unknown_cellids:
        print(f"\n⚠️ WARNING: The following {len(unknown_cellids)} cellIDs were encountered but not mapped to any group:")
        print(sorted(unknown_cellids))
    else:
        print("\n✅ All encountered cellIDs were successfully mapped to groups.")"""
#{'learning_rate': 0.2, 'max_depth': 8, 'scale_pos_weight': 0.25}
if args.run:
    all_configs = {
        'muons': {
            'files': glob.glob('/ceph/submit/data/user/j/jaeyserm/fccee/beam_backgrounds/CLD_o2_v05/mu_theta_0-180_p_50/*.root'),
            'outfile': 'ABmuons_edep.pkl'
        },
        'signal': {
            'files': glob.glob('/ceph/submit/data/group/fcc/ee/detector/VTXStudiesFullSim/CLD_wz3p6_ee_qq_ecm91p2/*.root'),
            'outfile': 'ABsignal_edep.pkl'
        },
        'background': {
            'files': glob.glob('/ceph/submit/data/group/fcc/ee/detector/VTXStudiesFullSim/CLD_o2_v05/FCCee_Z_4IP_04may23_FCCee_Z/*.root'),
            'outfile': 'ABbkg_edep.pkl'
        }
    }

    for label, config in all_configs.items():
        files = config['files']
        outfile = config['outfile']
        cluster_metrics = []
        limit = {
            'muons': 978,
            'signal': 100,
            'background': 1247
        }[label]

        for i, filename in enumerate(files):
            if i >= limit:
                break
            print(f"[{label.upper()}] Processing file {i+1}/{limit}: {filename}")
            reader = root_io.Reader(filename)
            events = reader.get('events')

            for event in events:
                particle_hits = defaultdict(list)

                for hit in event.get('VertexBarrelCollection'):
                    try:
                        if functions.radius_idx(hit, LAYER_RADII) != TARGET_LAYER:
                            continue
                        if hit.isProducedBySecondary():
                            continue
                        pos = hit.getPosition()
                        mc = hit.getMCParticle()
                        if mc is None:
                            continue
                        trackID = mc.getObjectID().index
                        energy = mc.getEnergy()
                        pid = mc.getPDG()
                        try:
                            edep = hit.getEDep()
                        except AttributeError:
                            edep = 0
                        h = functions.Hit(x=pos.x, y=pos.y, z=pos.z, energy=energy, edep=edep, trackID=trackID)
                        particle_hits[trackID].append((trackID, h, pid))
                    except Exception as e:
                        print(f"Skipping hit due to error: {e}")

                for trackID, hit_group in particle_hits.items():
                    if not hit_group:
                        continue
                    _, _, pid = hit_group[0]
                    p = functions.Particle(trackID=trackID)
                    p.pid = pid
                    for _, h, _ in hit_group:
                        p.add_hit(h)
                        
                    multiplicity = len(p.hits)
                    if multiplicity == 2:
                        p.hits = functions.merge_cluster_hits(p.hits)
                    total_edep = p.total_energy()
                    b_x, b_y, b_z = functions.geometric_baricenter(p.hits)
                    cos_theta = functions.cos_theta(b_x, b_y, b_z)
                    mc_energy = p.hits[0].energy
                    z_ext = p.z_extent()
                    nrows = p.n_phi_rows(PITCH, RADIUS)

                    cluster_metrics.append((z_ext, nrows, multiplicity, total_edep, mc_energy, cos_theta, b_x, b_y, pid))

        with open(outfile, 'wb') as f:
            pickle.dump(cluster_metrics, f)
        print(f"✅ Saved {label} clusters to {outfile}")

"""if args.classify:
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        roc_auc_score,
        roc_curve
    )
    import matplotlib.pyplot as plt
    import numpy as np
    import pickle
    import os
    from functions import get_features_and_labels

    outdir = 'Classification_AB'
    os.makedirs(outdir, exist_ok=True)

    # === Load data ===
    with open('ABsignal_edep.pkl', 'rb') as f:
        signal_data = pickle.load(f)
    with open('ABbkg_edep.pkl', 'rb') as f:
        background_data = pickle.load(f)

    # === Use transformed features ===
    X, y = get_features_and_labels(signal_data, background_data)

    # === Train-test split ===
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # === Train XGBoost Classifier with fixed params ===
    clf = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        n_estimators=100,
        max_depth=8,
        learning_rate=0.1,
        scale_pos_weight=0.5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    clf.fit(X_train, y_train)
    y_proba = clf.predict_proba(X_test)[:, 1]

    # === Sweep thresholds to find one that preserves ≥99% signal ===
    thresholds = np.linspace(0.0, 1.0, 500)
    tpr_list, fpr_list = [], []
    best_thresh, best_fpr = None, 1.0
    target_tpr = 0.99

    for thresh in thresholds:
        y_pred_temp = (y_proba >= thresh).astype(int)
        TP = np.sum((y_pred_temp == 1) & (y_test == 1))
        FP = np.sum((y_pred_temp == 1) & (y_test == 0))
        FN = np.sum((y_pred_temp == 0) & (y_test == 1))
        TN = np.sum((y_pred_temp == 0) & (y_test == 0))

        tpr = TP / (TP + FN) if TP + FN > 0 else 0
        fpr = FP / (FP + TN) if FP + TN > 0 else 0

        tpr_list.append(tpr)
        fpr_list.append(fpr)

        if tpr >= target_tpr and fpr < best_fpr:
            best_thresh, best_fpr = thresh, fpr

    if best_thresh is not None:
        print(f"Threshold for ≥{target_tpr*100:.1f}% signal retention: {best_thresh:.4f}")
        print(f"Background rejection at that threshold: {1 - best_fpr:.4f}")
    else:
        print(f"No threshold found that satisfies TPR ≥ {target_tpr*100:.1f}%")
        best_thresh = 0.5

    # === Final prediction and metrics ===
    y_pred = (y_proba >= best_thresh).astype(int)
    print("\n=== Final Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=["Background", "Signal"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("ROC AUC score: %.4f" % roc_auc_score(y_test, y_proba))

    # === Plot ROC Curve ===
    fpr_vals, tpr_vals, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    plt.plot(fpr_vals, tpr_vals, label=f'ROC (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — AB Dataset Classifier")
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(os.path.join(outdir, "AB_ROC_curve.png"))
    plt.close()

    # === Plot threshold sweep ===
    plt.plot(thresholds, tpr_list, label='TPR (Signal Retention)')
    plt.plot(thresholds, fpr_list, label='FPR (Background Acceptance)')
    if best_thresh is not None:
        plt.axvline(best_thresh, color='g', linestyle='--', label=f'TPR ≥ {target_tpr*100:.0f}% @ {best_thresh:.3f}')
    plt.xlabel("Threshold")
    plt.ylabel("Metric Value")
    plt.title("Threshold Sweep — TPR vs FPR")
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig(os.path.join(outdir, "AB_threshold_sweep.png"))
    plt.close()"""
    
if args.plots:
    from functions import extract, plot_overlay
    import os, pickle, random

    outdir = 'cos_vs_z_extent'
    os.makedirs(outdir, exist_ok=True)
    random.seed(42)

    with open('ABsignal_edep.pkl', 'rb') as f:
        signal = pickle.load(f)
    with open('ABbkg_edep.pkl', 'rb') as f:
        background = pickle.load(f)

    bkg_all = background
    sig_all = random.sample(signal, len(bkg_all))

    sig_z_all, _, _, _, _, sig_cos_all, _, _ = extract(sig_all, 0, 1, 2, 3, 4, 5, 6, 7)
    bkg_z_all, _, _, _, _, bkg_cos_all, _, _ = extract(bkg_all, 0, 1, 2, 3, 4, 5, 6, 7)

    plot_overlay(
        sig_cos_all, sig_z_all,
        bkg_cos_all, bkg_z_all,
        name="cos_theta_vs_z_extent",
        xlabel="cos(θ)",
        ylabel=r"$\Delta z$ [mm]",
        logy=True,
        outdir=outdir,
        label_sig="muons",
        label_bkg="background"
    )


if args.classify:
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        roc_auc_score,
        precision_recall_fscore_support,
        roc_curve,
        ConfusionMatrixDisplay
    )
    import matplotlib.pyplot as plt
    import numpy as np
    import pickle
    import os
    import random
    from functions import relabel_noise_clusters, get_features_and_labels

    outdir = 'Classification_AB'
    os.makedirs(outdir, exist_ok=True)
    random.seed(42)

    # === Load data ===
    with open('ABmuons_edep.pkl', 'rb') as f:
        muons = pickle.load(f)
    with open('ABsignal_edep.pkl', 'rb') as f:
        signal = pickle.load(f)
    with open('ABbkg_edep.pkl', 'rb') as f:
        background = pickle.load(f)

    # === Reassign noise-like clusters to background ===
    noise_pids = {11, -11, 13, -211, 22, 211, 2212, -2212}
    energy_cut = 0.01
    clean_muons, reassigned_muons = relabel_noise_clusters(muons, noise_pids, energy_cut)
    clean_signal, reassigned_signal = relabel_noise_clusters(signal, noise_pids, energy_cut)

    all_background = background + reassigned_muons + reassigned_signal
    all_signal = clean_muons + clean_signal
    sampled_signal = random.sample(all_signal, len(all_background))

    print(f"Clean muons: {len(clean_muons)}")
    print(f"Clean signal: {len(clean_signal)}")
    print(f"Reassigned to background: {len(reassigned_muons) + len(reassigned_signal)}")

    # === Feature extraction and split ===
    X, y = get_features_and_labels(sampled_signal, all_background)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # ---- add CLI args near the top (once) ----
    # parser.add_argument("--n_seeds", type=int, default=10)
    # parser.add_argument("--seed0", type=int, default=0)

    # ---- inside if args.classify: after you build X, y ----
    import numpy as np
    import random
    import pickle
    import os
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

    def set_all_seeds(seed: int):
        random.seed(seed)
        np.random.seed(seed)

    def find_best_thresholds(y_test, y_proba, thresholds, targets):
        # returns dict: target -> dict(thresh, fpr, tpr)
        out = {}
        # Precompute TP/FP/FN/TN for each threshold cheaply
        y_test = np.asarray(y_test).astype(int)
        pos = (y_test == 1)
        neg = (y_test == 0)
        n_pos = pos.sum()
        n_neg = neg.sum()

        best_for_target = {t: {"thresh": None, "fpr": 1.0, "tpr": 0.0} for t in targets}

        for thresh in thresholds:
            y_pred = (y_proba >= thresh).astype(int)

            TP = np.sum((y_pred == 1) & pos)
            FP = np.sum((y_pred == 1) & neg)
            FN = n_pos - TP
            TN = n_neg - FP

            tpr = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0

            for target in targets:
                cur = best_for_target[target]
                if tpr >= target and fpr < cur["fpr"]:
                    best_for_target[target] = {"thresh": float(thresh), "fpr": float(fpr), "tpr": float(tpr)}

        return best_for_target

    targets = [0.90, 0.99, 0.999]
    thresholds = np.linspace(0.0, 1.0, 500)

    results = {
        "seed": [],
        "auc": [],
        # store per target
        "ops": {t: {"thresh": [], "fpr": [], "tpr": [], "bkg_rej": []} for t in targets},
    }

    n_seeds = getattr(args, "n_seeds", 10)
    seed0 = getattr(args, "seed0", 0)

    for k in range(n_seeds):
        seed = seed0 + k
        set_all_seeds(seed)

        # split changes with seed
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )

        clf = xgb.XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            n_estimators=100,
            max_depth=8,
            learning_rate=0.1,
            scale_pos_weight=0.5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=4
        )
        clf.fit(X_train, y_train)

        y_proba = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)

        best = find_best_thresholds(y_test, y_proba, thresholds, targets)

        results["seed"].append(seed)
        results["auc"].append(float(auc))

        print(f"\n[seed {seed}] AUC = {auc:.6f}")
        for t in targets:
            op = best[t]
            if op["thresh"] is None:
                print(f"  TPR ≥ {t*100:.1f}%: no threshold found")
                # push NaNs so summary still works
                results["ops"][t]["thresh"].append(np.nan)
                results["ops"][t]["fpr"].append(np.nan)
                results["ops"][t]["tpr"].append(np.nan)
                results["ops"][t]["bkg_rej"].append(np.nan)
            else:
                bkg_rej = 1.0 - op["fpr"]
                print(f"  TPR ≥ {t*100:.1f}%: thr={op['thresh']:.4f} | "
                    f"TPR={op['tpr']:.4f} | FPR={op['fpr']:.4f} | bkg rej={bkg_rej:.4f}")
                results["ops"][t]["thresh"].append(op["thresh"])
                results["ops"][t]["fpr"].append(op["fpr"])
                results["ops"][t]["tpr"].append(op["tpr"])
                results["ops"][t]["bkg_rej"].append(bkg_rej)

    # ---- summary ----
    def mean_std(x):
        x = np.asarray(x, dtype=float)
        x = x[~np.isnan(x)]
        if len(x) == 0:
            return np.nan, np.nan
        return float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else 0.0

    auc_mean, auc_std = mean_std(results["auc"])
    print("\n================ SUMMARY (mean ± std over seeds) ================")
    print(f"AUC: {auc_mean:.6f} ± {auc_std:.6f}")

    for t in targets:
        thr_m, thr_s = mean_std(results["ops"][t]["thresh"])
        fpr_m, fpr_s = mean_std(results["ops"][t]["fpr"])
        rej_m, rej_s = mean_std(results["ops"][t]["bkg_rej"])
        print(f"TPR ≥ {t*100:.1f}%:")
        print(f"  thr:     {thr_m:.4f} ± {thr_s:.4f}")
        print(f"  FPR:     {fpr_m:.4f} ± {fpr_s:.4f}")
        print(f"  bkg rej: {rej_m:.4f} ± {rej_s:.4f}")

    # ---- save per-seed operating points + AUC ----
    with open(os.path.join(outdir, "seed_sweep_results.pkl"), "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved per-seed results to {os.path.join(outdir, 'seed_sweep_results.pkl')}")
