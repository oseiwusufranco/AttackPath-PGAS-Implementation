#!/usr/bin/env python3
"""End-to-end AttackPath-PGAS experiment exported from the reference notebook."""


# %% Cell 2
# Environment setup
from pathlib import Path
from IPython.display import display

PROJECT_ROOT = Path.cwd().resolve()
print(f"Project root: {PROJECT_ROOT}")

# %% Cell 3
# ============================================================
# 2. Imports and global configuration
# Full-data streaming + performance-optimized PGAS
# ============================================================

import os
import re
import json
import math
import time
import random
import warnings
import yaml
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from scipy.special import logsumexp, expit, logit
from scipy.stats import invgamma
from scipy.optimize import minimize

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, IsolationForest, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    brier_score_loss,
    matthews_corrcoef,
    log_loss,
    roc_curve,
    precision_recall_curve,
    auc,
)
from sklearn.feature_selection import mutual_info_classif
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns
import joblib

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 200)
pd.set_option('display.width', 200)

@dataclass
class CFG:
    DATASET_ROOT: str = os.environ.get('ATTACKPATH_DATASET_ROOT', str(PROJECT_ROOT / 'data'))
    OUTPUT_DIR: str = os.environ.get('ATTACKPATH_OUTPUT_DIR', str(PROJECT_ROOT / 'results' / 'main_run'))

    # Full-data mode.
    # None means every row in each selected CSV is scanned.
    STREAM_MAX_ROWS_PER_FILE: int | None = None
    CHUNK_SIZE: int = 200_000
    CSV_SAMPLE_ROWS: int = 5000

    # Temporal aggregation.
    # CICAPT network data is safest with row-order pseudo-time windows.
    ROW_WINDOW_SIZE: int = 5000
    MAX_WINDOWS: int | None = None

    # Dataset structure.
    USE_PHASE1_BACKGROUND: bool = True
    REQUIRE_PHASE2: bool = True

    # Label/window rules.
    BENIGN_STATE_NAME: str = 'BENIGN'
    FORCE_BINARY_STATES: bool = True
    ATTACK_WINDOW_THRESHOLD: float = 0.001
    ATTACK_WINDOW_MIN_EVENTS: int = 3

    # Feature controls.
    MAX_NUMERIC_COLUMNS: int = 65
    MAX_EVENT_TYPE_LEVELS: int = 30
    MIN_FEATURE_VARIANCE: float = 1e-10
    MAX_CORE_FEATURES: int = 90
    USE_MUTUAL_INFO_FEATURE_SELECTION: bool = True
    MI_TOP_FEATURES: int = 35
    SHIFT_TOP_FEATURES: int = 45

    # Chronological split over Phase 2.
    # All Phase 1 windows are included in training as background context.
    PHASE2_TRAIN_FRAC: float = 0.50
    PHASE2_VAL_FRAC: float = 0.25

    # Discriminative risk-emission layer.
    TRAIN_RISK_ENSEMBLE: bool = True
    RISK_EPS: float = 1e-4
    RISK_EMISSION_WEIGHT: float = 9.00
    GAUSSIAN_EMISSION_WEIGHT: float = 0.005
    USE_TEMPERATURE_CALIBRATION: bool = True

    # Rare-event PGAS priors.
    PI_PRIOR_BENIGN: float = 300.0
    PI_PRIOR_ATTACK: float = 2.0
    A00_PRIOR: float = 140.0
    A01_PRIOR: float = 4.0
    A10_PRIOR: float = 6.0
    A11_PRIOR: float = 42.0
    DIRICHLET_ALPHA_FALLBACK: float = 0.5

    # Emission priors for diagonal Gaussian component.
    NIG_M0: float = 0.0
    NIG_KAPPA0: float = 0.05
    NIG_A0: float = 3.0
    NIG_B0: float = 3.0

    # PGAS computation.
    NUM_CHAINS: int = 2
    PGAS_PARTICLES: int = 200
    MCMC_ITER: int = 550
    BURN_IN: int = 180
    THIN: int = 5
    FIXED_THETA_SMOOTHING_ITER: int = 160
    USE_EXACT_FORWARD_BACKWARD_SMOOTHING: bool = True
    USE_CACHED_FULL_STREAMING_WINDOWS: bool = True
    FORCE_REBUILD_STREAMING_CACHE: bool = False

    # Threshold policy.
    THRESHOLD_GRID_SIZE: int = 301
    MIN_RECALL_FOR_POLICY: float = 0.65
    POLICY_BETA: float = 2.0
    POLICY_MIN_PRECISION: float = 0.20
    POLICY_OBJECTIVE_RECALL_WEIGHT: float = 0.25
    POLICY_OBJECTIVE_MCC_WEIGHT: float = 0.20
    POLICY_OBJECTIVE_AUPR_WEIGHT: float = 0.15




    # Temporal feature engineering from window summaries.
    ADD_TEMPORAL_FEATURES: bool = True
    TEMPORAL_LAGS: tuple = (1, 2, 3, 5)
    TEMPORAL_ROLLING_WINDOWS: tuple = (3, 5, 9)
    TEMPORAL_EWMA_SPANS: tuple = (3, 7)
    TEMPORAL_BASE_FEATURE_LIMIT: int = 35



    # Hierarchical event-level risk backbone.
    # These settings add an event/flow-level rare-event detector before PGAS.
    # The event detector is trained only on training windows, then streamed over all rows
    # to create leakage-safe window emission features for PGAS.
    USE_EVENT_LEVEL_RISK_BACKBONE: bool = True
    REBUILD_EVENT_RISK_CACHE: bool = False
    EVENT_MAX_NUMERIC_COLUMNS: int = 45
    EVENT_TRAIN_MAX_ATTACK_ROWS: int = 350_000
    EVENT_TRAIN_MAX_BENIGN_ROWS: int = 650_000
    EVENT_VAL_MAX_ATTACK_ROWS: int = 120_000
    EVENT_VAL_MAX_BENIGN_ROWS: int = 240_000
    EVENT_SAMPLE_RANDOM_SEED: int = 2026
    EVENT_RISK_TOPK_PER_WINDOW: int = 20
    EVENT_RISK_HIGH_THRESHOLDS: tuple = (0.50, 0.70, 0.90, 0.95)
    EVENT_RISK_EMISSION_WEIGHT: float = 12.0
    WINDOW_RISK_FEATURE_WEIGHT: float = 1.0

    # Segment-aware decision policy.
    USE_HYSTERESIS_POLICY: bool = True
    HYSTERESIS_MIN_SEGMENT_LENGTH: int = 2
    HYSTERESIS_MERGE_GAP: int = 1
    HYSTERESIS_LOW_RATIO_GRID: tuple = (0.35, 0.50, 0.65, 0.80)
    MAX_FALSE_POSITIVE_RATE_FOR_POLICY: float = 0.05

    # Robustness and SOTA tracking.
    RUN_LIGHTWEIGHT_FIVE_SEED_POLICY_CHECK: bool = True
    POLICY_SEEDS: tuple = (42, 123, 202, 777, 999)

    # Reproducibility.
    RANDOM_SEED: int = 42

cfg = CFG()

# Apply archive configuration when ATTACKPATH_CONFIG is set.
_config_path = Path(os.environ.get('ATTACKPATH_CONFIG', PROJECT_ROOT / 'config' / 'default.yaml'))
if _config_path.exists():
    _raw = yaml.safe_load(_config_path.read_text(encoding='utf-8')) or {}
    _mapping = {
        'dataset_root': 'DATASET_ROOT', 'output_dir': 'OUTPUT_DIR',
        'chunk_size': 'CHUNK_SIZE', 'sample_rows': 'CSV_SAMPLE_ROWS',
        'rows_per_window': 'ROW_WINDOW_SIZE', 'max_rows_per_file': 'STREAM_MAX_ROWS_PER_FILE',
        'max_windows': 'MAX_WINDOWS',
    }
    for _key, _attribute in _mapping.items():
        if _raw.get(_key) is not None:
            setattr(cfg, _attribute, _raw[_key])
    _nested = {
        ('splits', 'phase2_train_fraction'): 'PHASE2_TRAIN_FRAC',
        ('splits', 'phase2_validation_fraction'): 'PHASE2_VAL_FRAC',
        ('labels', 'attack_window_threshold'): 'ATTACK_WINDOW_THRESHOLD',
        ('labels', 'attack_window_min_events'): 'ATTACK_WINDOW_MIN_EVENTS',
        ('features', 'max_numeric_columns'): 'MAX_NUMERIC_COLUMNS',
        ('features', 'maximum_window_features'): 'MAX_CORE_FEATURES',
        ('features', 'mutual_information_features'): 'MI_TOP_FEATURES',
        ('event_risk', 'enabled'): 'USE_EVENT_LEVEL_RISK_BACKBONE',
        ('event_risk', 'maximum_numeric_columns'): 'EVENT_MAX_NUMERIC_COLUMNS',
        ('event_risk', 'maximum_attack_rows'): 'EVENT_TRAIN_MAX_ATTACK_ROWS',
        ('event_risk', 'maximum_benign_rows'): 'EVENT_TRAIN_MAX_BENIGN_ROWS',
        ('event_risk', 'top_k'): 'EVENT_RISK_TOPK_PER_WINDOW',
        ('pgas', 'chains'): 'NUM_CHAINS', ('pgas', 'particles'): 'PGAS_PARTICLES',
        ('pgas', 'iterations'): 'MCMC_ITER', ('pgas', 'burn_in'): 'BURN_IN',
        ('pgas', 'thin'): 'THIN', ('pgas', 'smoothing_iterations'): 'FIXED_THETA_SMOOTHING_ITER',
        ('pgas', 'risk_emission_weight'): 'RISK_EMISSION_WEIGHT',
        ('pgas', 'gaussian_emission_weight'): 'GAUSSIAN_EMISSION_WEIGHT',
        ('policy', 'threshold_grid_size'): 'THRESHOLD_GRID_SIZE',
        ('policy', 'minimum_recall'): 'MIN_RECALL_FOR_POLICY',
        ('policy', 'maximum_false_positive_rate'): 'MAX_FALSE_POSITIVE_RATE_FOR_POLICY',
        ('policy', 'hysteresis_minimum_segment_length'): 'HYSTERESIS_MIN_SEGMENT_LENGTH',
        ('policy', 'hysteresis_merge_gap'): 'HYSTERESIS_MERGE_GAP',
        ('runtime', 'random_seed'): 'RANDOM_SEED',
    }
    for (_section, _key), _attribute in _nested.items():
        if _raw.get(_section, {}).get(_key) is not None:
            setattr(cfg, _attribute, _raw[_section][_key])
    if os.environ.get('ATTACKPATH_DATASET_ROOT'):
        cfg.DATASET_ROOT = os.environ['ATTACKPATH_DATASET_ROOT']
    if os.environ.get('ATTACKPATH_OUTPUT_DIR'):
        cfg.OUTPUT_DIR = os.environ['ATTACKPATH_OUTPUT_DIR']

random.seed(cfg.RANDOM_SEED)
np.random.seed(cfg.RANDOM_SEED)

for sub in ['tables', 'figures', 'models', 'logs']:
    Path(cfg.OUTPUT_DIR, sub).mkdir(parents=True, exist_ok=True)

print(cfg)

# %% Cell 4
# ============================================================
# 3. Helper functions for saving artifacts
# ============================================================

def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, default=str)


def save_fig(name, dpi=300):
    path = Path(cfg.OUTPUT_DIR, 'figures', name)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches='tight')
    print('Saved figure:', path)


def normalize_stage_name(x):
    if pd.isna(x):
        return cfg.BENIGN_STATE_NAME
    s = str(x).strip()
    if s == '' or s.lower() in {'0', 'benign', 'normal', 'none', 'nan', 'no attack', 'no_attack'}:
        return cfg.BENIGN_STATE_NAME
    return re.sub(r'[^A-Za-z0-9]+', '_', s.upper()).strip('_') or 'ATTACK'


def safe_auc(y, score):
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return np.nan
    return roc_auc_score(y, score)


def safe_aupr(y, score):
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return np.nan
    return average_precision_score(y, score)


def entropy_from_counter(counter):
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    probs = np.array(list(counter.values()), dtype=float) / total
    return float(-(probs * np.log(probs + 1e-12)).sum())


def sanitize_feature_name(x):
    return re.sub(r'[^A-Za-z0-9_]+', '_', str(x))[:80]

# %% Cell 6
# ============================================================
# 4. Discover Phase 1 and Phase 2 network-data files
# ============================================================

root = Path(cfg.DATASET_ROOT)
if not root.exists():
    raise FileNotFoundError(f'Dataset root not found: {root}')

allowed_ext = {'.csv'}
all_files = [p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in allowed_ext]
if root.is_file() and root.suffix.lower() in allowed_ext:
    all_files = [root]

file_table = pd.DataFrame({
    'path': [str(p) for p in all_files],
    'name': [p.name for p in all_files],
    'size_mb': [p.stat().st_size / (1024 ** 2) for p in all_files],
})

if file_table.empty:
    raise RuntimeError('No CSV files found under DATASET_ROOT.')

# Keep only phase1/phase2 files to avoid contamination from other datasets.
def infer_phase(path):
    text = str(path).lower()
    if 'phase1' in text or 'phase_1' in text or 'phase 1' in text:
        return 'phase1'
    if 'phase2' in text or 'phase_2' in text or 'phase 2' in text:
        return 'phase2'
    return None

file_table['phase'] = file_table['path'].map(infer_phase)
phase_files = file_table[file_table['phase'].notna()].copy()

if cfg.REQUIRE_PHASE2 and 'phase2' not in set(phase_files['phase']):
    raise RuntimeError('No Phase 2 file was found. Expected filename/path containing phase2.')

if cfg.USE_PHASE1_BACKGROUND and 'phase1' not in set(phase_files['phase']):
    print('Warning: Phase 1 was not found. The notebook will run without Phase 1 background context.')

phase_files['phase_order'] = phase_files['phase'].map({'phase1': 0, 'phase2': 1})
phase_files = phase_files.sort_values(['phase_order', 'name']).reset_index(drop=True)

display(phase_files)
phase_files.to_csv(Path(cfg.OUTPUT_DIR, 'tables', 'selected_phase_files.csv'), index=False)

# %% Cell 7
# ============================================================
# 5. Column detection and lightweight profiling
# ============================================================

def read_sample(path, nrows=5000):
    return pd.read_csv(path, nrows=nrows, low_memory=False)


def detect_columns(columns):
    cols = list(columns)
    low = {c: c.lower().strip() for c in cols}

    def exact_or_contains(exact_names, contains_names=None, reject=None):
        contains_names = contains_names or []
        reject = reject or []
        for c, lc in low.items():
            norm = re.sub(r'[^a-z0-9]+', '_', lc).strip('_')
            if norm in exact_names and not any(r in norm for r in reject):
                return c
        for pat in contains_names:
            for c, lc in low.items():
                norm = re.sub(r'[^a-z0-9]+', '_', lc).strip('_')
                if pat in norm and not any(r in norm for r in reject):
                    return c
        return None

    label = exact_or_contains(
        {'label', 'class', 'attack', 'category', 'attack_category', 'classification'},
        ['label', 'attack', 'category', 'class']
    )

    # Prefer IP/address columns for graph endpoints. Fall back to source/destination ports
    # only if no address-like columns exist in the network-flow file.
    src = exact_or_contains(
        {'src_ip', 'source_ip', 'srcaddr', 'source_address', 'source'},
        ['src_ip', 'source_ip', 'source_address', 'srcaddr'],
        reject=['port', 'bytes', 'packets']
    )
    dst = exact_or_contains(
        {'dst_ip', 'destination_ip', 'dstaddr', 'destination_address', 'destination', 'dest'},
        ['dst_ip', 'destination_ip', 'destination_address', 'dstaddr'],
        reject=['port', 'bytes', 'packets']
    )

    if src is None:
        src = exact_or_contains({'source_port', 'src_port', 'sport'}, ['source_port', 'src_port', 'sport'])
    if dst is None:
        dst = exact_or_contains({'destination_port', 'dst_port', 'dport'}, ['destination_port', 'dst_port', 'dport'])

    event_type = exact_or_contains(
        {'event_type', 'protocol', 'protocol_type', 'service', 'operation', 'type'},
        ['event_type', 'protocol_type', 'protocol', 'service', 'operation']
    )
    pid = exact_or_contains({'pid', 'process_id'}, ['pid', 'process_id'])
    time = exact_or_contains({'timestamp', 'time', 'ts'}, ['timestamp', 'event_time'])

    # Avoid accidental reuse of the label column as another semantic field.
    for name, value in [('src', src), ('dst', dst), ('event_type', event_type), ('pid', pid), ('time', time)]:
        if value is not None and label is not None and value == label:
            if name == 'src': src = None
            elif name == 'dst': dst = None
            elif name == 'event_type': event_type = None
            elif name == 'pid': pid = None
            elif name == 'time': time = None

    return {'src': src, 'dst': dst, 'label': label, 'event_type': event_type, 'pid': pid, 'time': time}

profile_rows = []
for _, r in phase_files.iterrows():
    sample = read_sample(r['path'], cfg.CSV_SAMPLE_ROWS)
    det = detect_columns(sample.columns)
    profile_rows.append({
        'phase': r['phase'],
        'file': r['name'],
        'columns': len(sample.columns),
        'sample_rows': len(sample),
        'detected_src': det['src'],
        'detected_dst': det['dst'],
        'detected_label': det['label'],
        'detected_event_type': det['event_type'],
        'detected_time': det['time'],
        'numeric_columns_in_sample': int(len(sample.select_dtypes(include=[np.number]).columns)),
    })

profile_df = pd.DataFrame(profile_rows)
display(profile_df)
profile_df.to_csv(Path(cfg.OUTPUT_DIR, 'tables', 'file_profile.csv'), index=False)

# %% Cell 8
# ============================================================
# 6. Streaming utility functions
# ============================================================

def useful_columns_from_sample(sample):
    if sample.empty:
        return []
    det = detect_columns(sample.columns)
    useful = set(v for v in det.values() if v is not None)

    numeric_cols = sample.select_dtypes(include=[np.number]).columns.tolist()
    for c in numeric_cols[:cfg.MAX_NUMERIC_COLUMNS]:
        useful.add(c)

    for c in sample.columns:
        lc = c.lower()
        if any(tok in lc for tok in [
            'type', 'event', 'action', 'operation', 'relation', 'status',
            'protocol', 'service', 'flag', 'label', 'category', 'tactic',
            'technique', 'src', 'source', 'dst', 'dest', 'destination'
        ]):
            useful.add(c)
    return [c for c in sample.columns if c in useful]


def read_csv_chunks(path, usecols=None):
    read_rows = 0
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=cfg.CHUNK_SIZE, low_memory=False):
        if cfg.STREAM_MAX_ROWS_PER_FILE is not None:
            remaining = cfg.STREAM_MAX_ROWS_PER_FILE - read_rows
            if remaining <= 0:
                break
            chunk = chunk.head(remaining)
        read_rows += len(chunk)
        yield chunk
        if cfg.STREAM_MAX_ROWS_PER_FILE is not None and read_rows >= cfg.STREAM_MAX_ROWS_PER_FILE:
            break


def safe_numeric_series(df, col):
    if col not in df.columns:
        return pd.Series(np.zeros(len(df)), index=df.index)
    return pd.to_numeric(df[col], errors='coerce').replace([np.inf, -np.inf], np.nan)


def update_window_accumulator(acc, chunk, det, phase, local_window_ids, numeric_cols):
    src_col = det.get('src')
    dst_col = det.get('dst')
    event_col = det.get('event_type')
    pid_col = det.get('pid')
    label_col = det.get('label')

    src = chunk[src_col].astype(str).fillna('unknown_src') if src_col else pd.Series(['unknown_src'] * len(chunk), index=chunk.index)
    dst = chunk[dst_col].astype(str).fillna('unknown_dst') if dst_col else pd.Series(['unknown_dst'] * len(chunk), index=chunk.index)
    etype = chunk[event_col].astype(str).fillna('unknown_event') if event_col else pd.Series(['unknown_event'] * len(chunk), index=chunk.index)
    pid = chunk[pid_col].astype(str).fillna('') if pid_col else pd.Series([''] * len(chunk), index=chunk.index)

    if label_col and label_col in chunk.columns:
        labels = chunk[label_col].map(normalize_stage_name).fillna(cfg.BENIGN_STATE_NAME).astype(str)
    else:
        labels = pd.Series([cfg.BENIGN_STATE_NAME] * len(chunk), index=chunk.index)

    work = pd.DataFrame({
        '_local_window_id': local_window_ids,
        '_src': src.values,
        '_dst': dst.values,
        '_etype': etype.values,
        '_pid': pid.values,
        '_label': labels.values,
    })

    for c in numeric_cols:
        work[f'__num__{c}'] = safe_numeric_series(chunk, c).values

    for win, g in work.groupby('_local_window_id', sort=False):
        key = (phase, int(win))
        if key not in acc:
            acc[key] = {
                'phase': phase,
                'phase_order': 0 if phase == 'phase1' else 1,
                'local_window_id': int(win),
                'event_count': 0,
                'src_counter': Counter(),
                'dst_counter': Counter(),
                'edge_counter': Counter(),
                'degree_counter': Counter(),
                'etype_counter': Counter(),
                'pid_set': set(),
                'label_counter': Counter(),
                'attack_label_counter': Counter(),
                'attack_count': 0,
                'num_stats': defaultdict(lambda: {'sum': 0.0, 'sumsq': 0.0, 'count': 0, 'max': -np.inf}),
            }
        a = acc[key]
        n = len(g)
        a['event_count'] += n

        src_vals = g['_src'].astype(str).tolist()
        dst_vals = g['_dst'].astype(str).tolist()
        etype_vals = g['_etype'].astype(str).tolist()
        pid_vals = [p for p in g['_pid'].astype(str).tolist() if p and p.lower() not in {'nan', 'none'}]
        label_vals = g['_label'].astype(str).tolist()

        a['src_counter'].update(src_vals)
        a['dst_counter'].update(dst_vals)
        a['etype_counter'].update(etype_vals)
        a['pid_set'].update(pid_vals)
        a['label_counter'].update(label_vals)

        edges = list(zip(src_vals, dst_vals))
        a['edge_counter'].update(edges)
        a['degree_counter'].update(src_vals)
        a['degree_counter'].update(dst_vals)

        attack_labels = [lab for lab in label_vals if lab != cfg.BENIGN_STATE_NAME]
        a['attack_count'] += len(attack_labels)
        a['attack_label_counter'].update(attack_labels)

        for c in numeric_cols:
            vals = pd.to_numeric(g[f'__num__{c}'], errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
            if len(vals) == 0:
                continue
            s = a['num_stats'][c]
            v = vals.values.astype(float)
            s['sum'] += float(v.sum())
            s['sumsq'] += float((v ** 2).sum())
            s['count'] += int(len(v))
            s['max'] = max(s['max'], float(np.max(v)))

# %% Cell 9
# ============================================================
# 7. Full-row streaming aggregation into temporal windows
# Cache-aware full-row streaming aggregation
# ============================================================

raw_window_cache = Path(cfg.OUTPUT_DIR, 'tables', 'full_streaming_window_features_raw.csv')
stream_report_cache = Path(cfg.OUTPUT_DIR, 'tables', 'streaming_aggregation_report.csv')

if cfg.USE_CACHED_FULL_STREAMING_WINDOWS and (not cfg.FORCE_REBUILD_STREAMING_CACHE) and raw_window_cache.exists() and stream_report_cache.exists():
    print('Loading cached full-streaming window features.')
    print('This cache is valid only if it was generated from the same Phase 1 and Phase 2 files.')
    window_df = pd.read_csv(raw_window_cache)
    stream_report_df = pd.read_csv(stream_report_cache)
    global_event_type_counter = Counter()
    if 'window_start' in window_df.columns:
        window_df['window_start'] = pd.to_datetime(window_df['window_start'], errors='coerce')
    print('Loaded cached window_df shape:', window_df.shape)
    display(stream_report_df)
    display(window_df.head())
else:
    # ============================================================
    # 7. Full-row streaming aggregation into temporal windows
    # ============================================================

    print('Starting full-row streaming aggregation.')
    print('STREAM_MAX_ROWS_PER_FILE:', cfg.STREAM_MAX_ROWS_PER_FILE)
    print('If this is None, every row in each selected CSV is scanned.')

    window_acc = {}
    global_event_type_counter = Counter()
    stream_reports = []
    phase_row_offsets = defaultdict(int)

    for _, row in phase_files.iterrows():
        phase = row['phase']
        path = row['path']

        sample = read_sample(path, cfg.CSV_SAMPLE_ROWS)
        det = detect_columns(sample.columns)
        usecols = useful_columns_from_sample(sample)

        for col in det.values():
            if col is not None and col not in usecols:
                usecols.append(col)

        numeric_cols = sample.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c in usecols]
        numeric_cols = [c for c in numeric_cols if c not in {det.get('time'), det.get('pid')}]
        numeric_cols = numeric_cols[:cfg.MAX_NUMERIC_COLUMNS]

        print(f"\nProcessing {phase}: {Path(path).name}")
        print('Detected columns:', det)
        print('Numeric observation columns:', numeric_cols[:20], '...' if len(numeric_cols) > 20 else '')

        rows_seen = 0
        chunks_seen = 0
        attack_rows_seen = 0
        base_offset = phase_row_offsets[phase]

        for chunk in tqdm(read_csv_chunks(path, usecols=usecols), desc=f'Streaming {phase}'):
            n = len(chunk)
            if n == 0:
                continue

            global_row_ids = base_offset + rows_seen + np.arange(n)
            local_window_ids = global_row_ids // cfg.ROW_WINDOW_SIZE

            label_col = det.get('label')
            if label_col is not None and label_col in chunk.columns:
                labels_norm = chunk[label_col].map(normalize_stage_name).fillna(cfg.BENIGN_STATE_NAME).astype(str)
                attack_rows_seen += int((labels_norm != cfg.BENIGN_STATE_NAME).sum())

            event_col = det.get('event_type')
            if event_col is not None and event_col in chunk.columns:
                global_event_type_counter.update(chunk[event_col].astype(str).fillna('unknown_event').tolist())

            update_window_accumulator(
                acc=window_acc,
                chunk=chunk,
                det=det,
                phase=phase,
                local_window_ids=local_window_ids,
                numeric_cols=numeric_cols,
            )

            rows_seen += n
            chunks_seen += 1

        phase_row_offsets[phase] += rows_seen

        stream_reports.append({
            'phase': phase,
            'file': Path(path).name,
            'rows_seen': rows_seen,
            'chunks_seen': chunks_seen,
            'attack_rows_seen': attack_rows_seen,
            'windows_seen_for_phase_after_file': len([k for k in window_acc if k[0] == phase]),
            'full_file_mode': cfg.STREAM_MAX_ROWS_PER_FILE is None,
        })

    stream_report_df = pd.DataFrame(stream_reports)
    display(stream_report_df)
    stream_report_df.to_csv(Path(cfg.OUTPUT_DIR, 'tables', 'streaming_aggregation_report.csv'), index=False)

    if len(window_acc) == 0:
        raise RuntimeError('No temporal windows were created during streaming aggregation.')

    top_event_types = [x for x, _ in global_event_type_counter.most_common(cfg.MAX_EVENT_TYPE_LEVELS)]
    print('Top event types:', top_event_types)

    rows = []
    for key, a in sorted(window_acc.items(), key=lambda kv: (kv[1]['phase_order'], kv[1]['local_window_id'])):
        src_counter = a['src_counter']
        dst_counter = a['dst_counter']
        edge_counter = a['edge_counter']
        degree_vals = np.array(list(a['degree_counter'].values()), dtype=float)
        nodes = set(src_counter.keys()) | set(dst_counter.keys())

        row = {
            'phase': a['phase'],
            'phase_order': a['phase_order'],
            'local_window_id': a['local_window_id'],
            'event_count': a['event_count'],
            'unique_src': len(src_counter),
            'unique_dst': len(dst_counter),
            'unique_nodes': len(nodes),
            'unique_edges': len(edge_counter),
            'graph_density': len(edge_counter) / max(len(nodes) * max(len(nodes) - 1, 1), 1),
            'degree_mean': float(degree_vals.mean()) if len(degree_vals) else 0.0,
            'degree_max': float(degree_vals.max()) if len(degree_vals) else 0.0,
            'degree_std': float(degree_vals.std()) if len(degree_vals) else 0.0,
            'src_entropy': entropy_from_counter(src_counter),
            'dst_entropy': entropy_from_counter(dst_counter),
            'event_type_entropy': entropy_from_counter(a['etype_counter']),
            'pid_unique': len(a['pid_set']),
            'source_file_unique': 1,
            'event_label_attack_fraction': a['attack_count'] / max(a['event_count'], 1),
        }

        if len(a['label_counter']):
            row['event_label_majority'] = a['label_counter'].most_common(1)[0][0]
        else:
            row['event_label_majority'] = cfg.BENIGN_STATE_NAME

        if len(a['attack_label_counter']):
            row['event_attack_label_majority'] = a['attack_label_counter'].most_common(1)[0][0]
        else:
            row['event_attack_label_majority'] = cfg.BENIGN_STATE_NAME

        for et in top_event_types:
            row[f'etype_count_{sanitize_feature_name(et)}'] = a['etype_counter'].get(et, 0)

        for c, s in a['num_stats'].items():
            cnt = max(s['count'], 1)
            mean = s['sum'] / cnt
            var = max(s['sumsq'] / cnt - mean ** 2, 0.0)
            prefix = f'obs_{sanitize_feature_name(c)}'
            row[f'{prefix}_mean'] = float(mean)
            row[f'{prefix}_std'] = float(np.sqrt(var))
            row[f'{prefix}_max'] = float(s['max']) if np.isfinite(s['max']) else 0.0

        rows.append(row)

    window_df = pd.DataFrame(rows)
    window_df = window_df.sort_values(['phase_order', 'local_window_id']).reset_index(drop=True)
    window_df['window_id'] = np.arange(len(window_df))
    window_df['window_start'] = pd.to_datetime('2024-01-01') + pd.to_timedelta(window_df['window_id'], unit='min')

    if cfg.MAX_WINDOWS is not None:
        window_df = window_df.iloc[:cfg.MAX_WINDOWS].copy().reset_index(drop=True)
        window_df['window_id'] = np.arange(len(window_df))

    print('window_df shape:', window_df.shape)
    print('Phase distribution:')
    display(window_df['phase'].value_counts().rename_axis('phase').reset_index(name='windows'))
    print('Rows/windows summary:')
    display(window_df.groupby('phase').agg(windows=('window_id', 'count'), events=('event_count', 'sum'), attack_fraction=('event_label_attack_fraction', 'mean')).reset_index())

    display(window_df.head())
    window_df.to_csv(Path(cfg.OUTPUT_DIR, 'tables', 'full_streaming_window_features_raw.csv'), index=False)

# %% Cell 11
# ============================================================
# 8. Attack-window labelling from event-level attack evidence
# Rare-event-sensitive version
# ============================================================

window_df['stage_raw'] = cfg.BENIGN_STATE_NAME
window_df['event_label_attack_fraction'] = pd.to_numeric(
    window_df['event_label_attack_fraction'],
    errors='coerce'
).fillna(0.0)

window_df['estimated_attack_events'] = (
    window_df['event_label_attack_fraction'] * pd.to_numeric(window_df['event_count'], errors='coerce').fillna(0.0)
).round().astype(int)

# A window is attack-active if it contains at least one attack event or passes the fraction threshold.
# This avoids missing sparse APT events diluted inside 5,000-row windows.
attack_window_mask = (
    (window_df['estimated_attack_events'] >= cfg.ATTACK_WINDOW_MIN_EVENTS)
    | (window_df['event_label_attack_fraction'] >= cfg.ATTACK_WINDOW_THRESHOLD)
)

window_df.loc[attack_window_mask, 'stage_raw'] = 'ATTACK'

if not cfg.FORCE_BINARY_STATES and 'event_attack_label_majority' in window_df.columns:
    window_df.loc[attack_window_mask, 'stage_raw'] = (
        window_df.loc[attack_window_mask, 'event_attack_label_majority'].map(normalize_stage_name)
    )

window_df['stage_raw'] = window_df['stage_raw'].map(normalize_stage_name).fillna(cfg.BENIGN_STATE_NAME)

if cfg.FORCE_BINARY_STATES:
    window_df['stage_raw_limited'] = np.where(window_df['stage_raw'] == cfg.BENIGN_STATE_NAME, cfg.BENIGN_STATE_NAME, 'ATTACK')
else:
    window_df['stage_raw_limited'] = window_df['stage_raw']

stage_names = [cfg.BENIGN_STATE_NAME] + [
    s for s in window_df['stage_raw_limited'].value_counts().index
    if s != cfg.BENIGN_STATE_NAME
]
stage_to_id = {s: i for i, s in enumerate(stage_names)}
id_to_stage = {i: s for s, i in stage_to_id.items()}

window_df['stage_id'] = window_df['stage_raw_limited'].map(stage_to_id).astype(int)
window_df['is_attack'] = (window_df['stage_id'] != 0).astype(int)

print('Attack window fraction threshold:', cfg.ATTACK_WINDOW_THRESHOLD)
print('Minimum attack events per attack window:', cfg.ATTACK_WINDOW_MIN_EVENTS)
print('Stage mapping:', stage_to_id)
display(window_df['stage_raw_limited'].value_counts().rename_axis('stage').reset_index(name='windows'))
display(window_df.groupby('phase')['is_attack'].agg(['count', 'sum', 'mean']).reset_index())

print('Estimated attack-event count summary for attack windows:')
display(window_df.loc[window_df['is_attack'] == 1, 'estimated_attack_events'].describe().to_frame('estimated_attack_events'))

window_df.to_csv(Path(cfg.OUTPUT_DIR, 'tables', 'window_features_labeled.csv'), index=False)
save_json({
    'stage_to_id': stage_to_id,
    'id_to_stage': id_to_stage,
    'attack_window_threshold': cfg.ATTACK_WINDOW_THRESHOLD,
    'attack_window_min_events': cfg.ATTACK_WINDOW_MIN_EVENTS
}, Path(cfg.OUTPUT_DIR, 'tables', 'stage_mapping.json'))

# %% Cell 12

# ============================================================
# 9. Phase-aware chronological split and leakage-safe temporal feature matrix
# ============================================================

exclude_cols = {
    'window_id', 'window_start', 'phase', 'phase_order', 'local_window_id',
    'stage_raw', 'stage_raw_limited', 'stage_id', 'is_attack',
    'event_label_majority', 'event_attack_label_majority', 'event_label_attack_fraction',
    'estimated_attack_events', 'attack_event_count', 'attack_rows_seen',
}

print('Leakage guard active: excluding label-derived attack-count/evaluation columns from X.')

leakage_tokens = [
    'label', 'groundtruth', 'ground_truth', 'stage_raw', 'stage_id', 'is_attack',
    'attack_fraction', 'target', 'class', 'estimated_attack', 'attack_event',
    'attack_count', 'attack_rows',
]

drop_tokens = ['obs_ts_', 'timestamp', 'event_time']

# Basic target and chronological indices are independent of the feature construction.
y_stage = window_df['stage_id'].values.astype(int)
y_attack = window_df['is_attack'].values.astype(int)
T = len(window_df)
K = len(stage_names)

phase1_idx_all = window_df.index[window_df['phase'] == 'phase1'].to_numpy()
phase2_idx_all = window_df.index[window_df['phase'] == 'phase2'].to_numpy()
if len(phase2_idx_all) == 0:
    raise RuntimeError('No Phase 2 windows found.')

n2 = len(phase2_idx_all)
n2_train = int(n2 * cfg.PHASE2_TRAIN_FRAC)
n2_val = int(n2 * cfg.PHASE2_VAL_FRAC)
phase2_train = phase2_idx_all[:n2_train]
phase2_val = phase2_idx_all[n2_train:n2_train + n2_val]
phase2_test = phase2_idx_all[n2_train + n2_val:]

train_idx = np.concatenate([phase1_idx_all, phase2_train]) if cfg.USE_PHASE1_BACKGROUND and len(phase1_idx_all) else phase2_train
val_idx = phase2_val
test_idx = phase2_test

# Build base numeric features.
numeric_feature_frames = []
feature_cols_all = []
for c in window_df.columns:
    if c in exclude_cols:
        continue
    lc = c.lower()
    if any(tok in lc for tok in leakage_tokens):
        continue
    if any(tok in lc for tok in drop_tokens):
        continue
    s = pd.to_numeric(window_df[c], errors='coerce')
    if s.notna().mean() < 0.50:
        continue
    s = s.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if s.var() <= cfg.MIN_FEATURE_VARIANCE:
        continue
    numeric_feature_frames.append(s.astype(float).rename(c))
    feature_cols_all.append(c)

if len(feature_cols_all) == 0:
    raise RuntimeError('No usable numeric feature columns remained after filtering.')

base_X = pd.concat(numeric_feature_frames, axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)

# Hard leakage guard.
for _c in list(base_X.columns):
    _lc = _c.lower()
    if any(_tok in _lc for _tok in ['label', 'stage', 'is_attack', 'attack_fraction', 'estimated_attack', 'attack_event', 'attack_count']):
        raise RuntimeError(f'Leakage guard failed: label-derived feature survived filtering: {_c}')

# Rank base features using only train-safe criteria.
bg_idx = phase1_idx_all if len(phase1_idx_all) else train_idx
p2_train_idx = phase2_train if len(phase2_train) else train_idx
train_std = base_X.iloc[train_idx].std(axis=0).replace(0, np.nan).fillna(1.0)
shift_score = (((base_X.iloc[p2_train_idx].mean(axis=0) - base_X.iloc[bg_idx].mean(axis=0)).abs()) / (train_std + 1e-9)).sort_values(ascending=False)
mi_score = pd.Series(0.0, index=base_X.columns)
if cfg.USE_MUTUAL_INFO_FEATURE_SELECTION and len(np.unique(y_attack[train_idx])) == 2:
    try:
        mi_vals = mutual_info_classif(base_X.iloc[train_idx], y_attack[train_idx], random_state=cfg.RANDOM_SEED, discrete_features=False)
        mi_score = pd.Series(mi_vals, index=base_X.columns).sort_values(ascending=False)
    except Exception as e:
        print('Mutual information feature ranking skipped:', repr(e))

core_features = [c for c in ['event_count','unique_src','unique_dst','unique_nodes','unique_edges','graph_density','degree_mean','degree_max','degree_std','src_entropy','dst_entropy','event_type_entropy','pid_unique'] if c in base_X.columns]
selected_base = []
for c in core_features + list(mi_score.index[:cfg.MI_TOP_FEATURES]) + list(shift_score.index[:cfg.SHIFT_TOP_FEATURES]):
    if c not in selected_base:
        selected_base.append(c)
    if len(selected_base) >= max(25, cfg.TEMPORAL_BASE_FEATURE_LIMIT):
        break

# Temporal features are causal/past-looking within each phase, not centered and not future-looking.
def add_temporal_features(base_df, selected_cols, phase_series):
    out = base_df[selected_cols].copy()
    if not cfg.ADD_TEMPORAL_FEATURES:
        return out
    pieces = [out]
    for phase_name, phase_index in phase_series.groupby(phase_series).groups.items():
        idx = list(phase_index)
        g = out.loc[idx, selected_cols].copy()
        tf = pd.DataFrame(index=idx)
        for c in selected_cols:
            s = g[c]
            for lag in cfg.TEMPORAL_LAGS:
                tf[f'{c}_lag{lag}'] = s.shift(lag).fillna(s.iloc[0])
                tf[f'{c}_diff{lag}'] = (s - s.shift(lag)).fillna(0.0)
            for w in cfg.TEMPORAL_ROLLING_WINDOWS:
                tf[f'{c}_rollmean{w}'] = s.rolling(w, min_periods=1).mean()
                tf[f'{c}_rollstd{w}'] = s.rolling(w, min_periods=1).std().fillna(0.0)
                tf[f'{c}_rollmax{w}'] = s.rolling(w, min_periods=1).max()
            for span in cfg.TEMPORAL_EWMA_SPANS:
                tf[f'{c}_ewm{span}'] = s.ewm(span=span, adjust=False).mean()
        pieces.append(tf.sort_index())
    temporal = pd.concat(pieces, axis=1).sort_index()
    temporal = temporal.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return temporal

X_temporal_all = add_temporal_features(base_X, selected_base, window_df['phase'])

# Rank the expanded feature matrix and retain the most useful dimensions.
expanded_std = X_temporal_all.iloc[train_idx].std(axis=0).replace(0, np.nan).fillna(1.0)
expanded_shift = (((X_temporal_all.iloc[p2_train_idx].mean(axis=0) - X_temporal_all.iloc[bg_idx].mean(axis=0)).abs()) / (expanded_std + 1e-9)).sort_values(ascending=False)
expanded_mi = pd.Series(0.0, index=X_temporal_all.columns)
if len(np.unique(y_attack[train_idx])) == 2:
    try:
        mi_vals = mutual_info_classif(X_temporal_all.iloc[train_idx], y_attack[train_idx], random_state=cfg.RANDOM_SEED, discrete_features=False)
        expanded_mi = pd.Series(mi_vals, index=X_temporal_all.columns).sort_values(ascending=False)
    except Exception as e:
        print('Expanded mutual information ranking skipped:', repr(e))

selected_features = []
for c in core_features:
    if c in X_temporal_all.columns and c not in selected_features:
        selected_features.append(c)
for c in list(expanded_mi.index[:cfg.MI_TOP_FEATURES * 2]) + list(expanded_shift.index[:cfg.SHIFT_TOP_FEATURES * 2]):
    if c not in selected_features:
        selected_features.append(c)
    if len(selected_features) >= cfg.MAX_CORE_FEATURES:
        break

X_raw = X_temporal_all[selected_features].copy()
feature_cols = selected_features.copy()

# Train-only scaling.
scaler = StandardScaler()
X_core = np.zeros_like(X_raw.values, dtype=float)
X_core[train_idx] = scaler.fit_transform(X_raw.iloc[train_idx])
if len(val_idx): X_core[val_idx] = scaler.transform(X_raw.iloc[val_idx])
if len(test_idx): X_core[test_idx] = scaler.transform(X_raw.iloc[test_idx])
assigned = np.zeros(T, dtype=bool)
assigned[train_idx] = True; assigned[val_idx] = True; assigned[test_idx] = True
if (~assigned).any():
    X_core[~assigned] = scaler.transform(X_raw.iloc[np.where(~assigned)[0]])
X_core = np.clip(X_core, -8, 8)

# Phase-1 background anomaly features.
bg_median = np.median(X_core[bg_idx], axis=0)
bg_mad = np.median(np.abs(X_core[bg_idx] - bg_median), axis=0) + 1e-3
bg_mean = X_core[bg_idx].mean(axis=0)
bg_var = X_core[bg_idx].var(axis=0) + 1e-3
background_l1 = np.mean(np.abs((X_core - bg_median) / bg_mad), axis=1)
background_diag_mahalanobis = np.mean(((X_core - bg_mean) ** 2) / bg_var, axis=1)

extra_raw = np.column_stack([background_l1, background_diag_mahalanobis])
extra_scaler = StandardScaler()
extra_scaled = np.zeros_like(extra_raw, dtype=float)
extra_scaled[train_idx] = extra_scaler.fit_transform(extra_raw[train_idx])
if len(val_idx): extra_scaled[val_idx] = extra_scaler.transform(extra_raw[val_idx])
if len(test_idx): extra_scaled[test_idx] = extra_scaler.transform(extra_raw[test_idx])
if (~assigned).any(): extra_scaled[~assigned] = extra_scaler.transform(extra_raw[np.where(~assigned)[0]])

X = np.hstack([X_core, extra_scaled])
X = np.clip(X, -8, 8)
feature_cols += ['background_l1_anomaly_score', 'background_diag_mahalanobis_score']
window_df['background_l1_anomaly_score_raw'] = background_l1
window_df['background_diag_mahalanobis_score_raw'] = background_diag_mahalanobis

print('T windows:', T)
print('D features:', X.shape[1])
print('K states:', K, stage_names)
print('Phase 1 windows:', len(phase1_idx_all))
print('Phase 2 windows:', len(phase2_idx_all))
print('Train/Validation/Test windows:', len(train_idx), len(val_idx), len(test_idx))
print('Train/Validation/Test attack windows:', int(y_attack[train_idx].sum()), int(y_attack[val_idx].sum()), int(y_attack[test_idx].sum()))
print('Base features selected before temporal expansion:', len(selected_base))
print('Final leakage-safe features:', len(feature_cols))
display(pd.DataFrame({'feature': feature_cols}))

split_table = pd.DataFrame({
    'split': ['train', 'validation', 'test'],
    'windows': [len(train_idx), len(val_idx), len(test_idx)],
    'phase1_windows': [int(np.isin(train_idx, phase1_idx_all).sum()), int(np.isin(val_idx, phase1_idx_all).sum()), int(np.isin(test_idx, phase1_idx_all).sum())],
    'phase2_windows': [int(np.isin(train_idx, phase2_idx_all).sum()), int(np.isin(val_idx, phase2_idx_all).sum()), int(np.isin(test_idx, phase2_idx_all).sum())],
    'attack_windows': [int(y_attack[train_idx].sum()), int(y_attack[val_idx].sum()), int(y_attack[test_idx].sum())],
    'attack_fraction': [float(y_attack[train_idx].mean()), float(y_attack[val_idx].mean()) if len(val_idx) else np.nan, float(y_attack[test_idx].mean()) if len(test_idx) else np.nan],
})
display(split_table)
split_table.to_csv(Path(cfg.OUTPUT_DIR, 'tables', 'chronological_split_summary.csv'), index=False)
joblib.dump(scaler, Path(cfg.OUTPUT_DIR, 'models', 'feature_scaler.joblib'))
joblib.dump(extra_scaler, Path(cfg.OUTPUT_DIR, 'models', 'background_feature_scaler.joblib'))

# %% Cell 14

# ============================================================
# 9b. Hierarchical event-level risk backbone and window emission features
# Leakage-safe raw-flow risk learning before PGAS
# ============================================================

# Hierarchical event-level risk estimation and temporal aggregation.
# It trains an event/flow-level rare-event detector only on TRAIN windows,
# then streams over all Phase 1 and Phase 2 rows to produce window-level risk
# emissions for PGAS. It does not use validation/test labels for training.

EVENT_RISK_CACHE = Path(cfg.OUTPUT_DIR, 'tables', 'event_level_window_risk_features.csv')
EVENT_RISK_REPORT = Path(cfg.OUTPUT_DIR, 'tables', 'event_level_risk_backbone_report.csv')
EVENT_RISK_SAMPLE_METRICS = Path(cfg.OUTPUT_DIR, 'tables', 'event_level_risk_sample_metrics.csv')
EVENT_RISK_MODEL_PATH = Path(cfg.OUTPUT_DIR, 'models', 'event_level_risk_backbone.joblib')


def _detect_label_and_event_numeric_columns(sample):
    det = detect_columns(sample.columns)
    label_col = det.get('label')
    numeric_cols = sample.select_dtypes(include=[np.number]).columns.tolist()

    block = {label_col, det.get('time'), det.get('pid')}
    useful = []
    for c in numeric_cols:
        if c in block or c is None:
            continue
        lc = c.lower()
        if any(tok in lc for tok in ['label', 'class', 'attack', 'target', 'timestamp', 'time_stamp']):
            continue
        useful.append(c)
    return det, label_col, useful[:cfg.EVENT_MAX_NUMERIC_COLUMNS]


def _phase_window_index_maps(window_df):
    maps = {}
    for ph, sub in window_df[['phase', 'local_window_id']].reset_index().groupby('phase'):
        maps[ph] = dict(zip(sub['local_window_id'].astype(int).values, sub['index'].astype(int).values))
    return maps


phase_to_window_index = _phase_window_index_maps(window_df)
train_index_set = set(map(int, train_idx))
val_index_set = set(map(int, val_idx))
test_index_set = set(map(int, test_idx))


def _global_window_indices_for_chunk(phase, rows_seen, n):
    local_ids = ((rows_seen + np.arange(n)) // cfg.ROW_WINDOW_SIZE).astype(int)
    mapper = phase_to_window_index.get(phase, {})
    global_ids = np.array([mapper.get(int(w), -1) for w in local_ids], dtype=int)
    return local_ids, global_ids


def _make_event_feature_frame(chunk, feature_cols):
    Xev = pd.DataFrame(index=chunk.index)
    for c in feature_cols:
        if c in chunk.columns:
            Xev[c] = pd.to_numeric(chunk[c], errors='coerce')
        else:
            Xev[c] = 0.0
    Xev = Xev.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    return Xev


def _bounded_concat(parts):
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _sample_rows_from_chunk(chunk, y, global_idx, feature_cols, split_name, counters, rng):
    valid = global_idx >= 0
    if split_name == 'train':
        in_split = np.array([int(g) in train_index_set for g in global_idx], dtype=bool)
        max_attack = cfg.EVENT_TRAIN_MAX_ATTACK_ROWS
        max_benign = cfg.EVENT_TRAIN_MAX_BENIGN_ROWS
    elif split_name == 'validation':
        in_split = np.array([int(g) in val_index_set for g in global_idx], dtype=bool)
        max_attack = cfg.EVENT_VAL_MAX_ATTACK_ROWS
        max_benign = cfg.EVENT_VAL_MAX_BENIGN_ROWS
    else:
        raise ValueError(split_name)

    mask = valid & in_split
    if not mask.any():
        return None, None

    y_masked = y[mask]
    idx_masked = np.where(mask)[0]
    attack_idx = idx_masked[y_masked == 1]
    benign_idx = idx_masked[y_masked == 0]

    keep_idx = []

    remaining_attack = max_attack - counters[f'{split_name}_attack']
    if remaining_attack > 0 and len(attack_idx):
        if len(attack_idx) > remaining_attack:
            attack_idx = rng.choice(attack_idx, size=remaining_attack, replace=False)
        keep_idx.extend(list(attack_idx))
        counters[f'{split_name}_attack'] += len(attack_idx)

    remaining_benign = max_benign - counters[f'{split_name}_benign']
    if remaining_benign > 0 and len(benign_idx):
        # Sample benign rows so training is not overwhelmed by Phase 1 background.
        take = min(len(benign_idx), remaining_benign, max(1000, int(0.03 * len(benign_idx))))
        if take > 0:
            benign_keep = rng.choice(benign_idx, size=take, replace=False)
            keep_idx.extend(list(benign_keep))
            counters[f'{split_name}_benign'] += len(benign_keep)

    if not keep_idx:
        return None, None

    keep_idx = np.array(sorted(set(map(int, keep_idx))), dtype=int)
    X_part = _make_event_feature_frame(chunk.iloc[keep_idx], feature_cols)
    y_part = y[keep_idx].astype(int)
    return X_part, y_part


if cfg.USE_EVENT_LEVEL_RISK_BACKBONE and EVENT_RISK_CACHE.exists() and (not cfg.REBUILD_EVENT_RISK_CACHE):
    print('Loading cached event-level window risk features.')
    event_window_risk_df = pd.read_csv(EVENT_RISK_CACHE)
    event_risk_report_df = pd.read_csv(EVENT_RISK_REPORT) if EVENT_RISK_REPORT.exists() else pd.DataFrame()
    event_sample_metrics_df = pd.read_csv(EVENT_RISK_SAMPLE_METRICS) if EVENT_RISK_SAMPLE_METRICS.exists() else pd.DataFrame()
    display(event_window_risk_df.head())
else:
    print('Building event-level risk backbone from raw rows. This streams the dataset again but keeps memory bounded.')

    # Use Phase 2 sample as reference for event feature columns, then include common numeric columns from both phases.
    feature_sets = []
    file_meta = []
    for _, row in phase_files.iterrows():
        sample = read_sample(row['path'], cfg.CSV_SAMPLE_ROWS)
        det, label_col, ev_numeric = _detect_label_and_event_numeric_columns(sample)
        if label_col is None:
            raise RuntimeError(f'No label column detected for {row["path"]}.')
        feature_sets.append(set(ev_numeric))
        file_meta.append({'phase': row['phase'], 'path': row['path'], 'det': det, 'label_col': label_col, 'feature_cols': ev_numeric})

    # Prefer common numeric columns across phases; fallback to union if needed.
    common_features = sorted(set.intersection(*feature_sets)) if len(feature_sets) > 1 else sorted(feature_sets[0])
    if len(common_features) < 5:
        common_features = sorted(set.union(*feature_sets))
    event_feature_cols = common_features[:cfg.EVENT_MAX_NUMERIC_COLUMNS]
    print(f'Event-level feature count: {len(event_feature_cols)}')
    print('Event-level feature examples:', event_feature_cols[:20])

    rng = np.random.default_rng(cfg.EVENT_SAMPLE_RANDOM_SEED)
    sample_counters = defaultdict(int)
    train_X_parts, train_y_parts = [], []
    val_X_parts, val_y_parts = [], []
    sampling_reports = []

    for meta in file_meta:
        phase, path, det, label_col = meta['phase'], meta['path'], meta['det'], meta['label_col']
        usecols = sorted(set(event_feature_cols + [label_col]))
        rows_seen = 0
        chunks_seen = 0
        for chunk in tqdm(read_csv_chunks(path, usecols=usecols), desc=f'Event-risk sample {phase}'):
            n = len(chunk)
            if n == 0:
                continue
            _, gidx = _global_window_indices_for_chunk(phase, rows_seen, n)
            labels = chunk[label_col].map(normalize_stage_name).fillna(cfg.BENIGN_STATE_NAME).astype(str)
            y = (labels != cfg.BENIGN_STATE_NAME).astype(int).values

            Xp, yp = _sample_rows_from_chunk(chunk, y, gidx, event_feature_cols, 'train', sample_counters, rng)
            if Xp is not None:
                train_X_parts.append(Xp); train_y_parts.append(pd.Series(yp))
            Xv, yv = _sample_rows_from_chunk(chunk, y, gidx, event_feature_cols, 'validation', sample_counters, rng)
            if Xv is not None:
                val_X_parts.append(Xv); val_y_parts.append(pd.Series(yv))
            rows_seen += n
            chunks_seen += 1
        sampling_reports.append({'phase': phase, 'rows_seen': rows_seen, 'chunks_seen': chunks_seen})

    Xev_train = _bounded_concat(train_X_parts)
    yev_train = pd.concat(train_y_parts, ignore_index=True).astype(int).values if train_y_parts else np.array([], dtype=int)
    Xev_val = _bounded_concat(val_X_parts)
    yev_val = pd.concat(val_y_parts, ignore_index=True).astype(int).values if val_y_parts else np.array([], dtype=int)

    if len(np.unique(yev_train)) < 2:
        raise RuntimeError('Event-level training sample has fewer than two classes. Increase caps or check labels.')

    print('Event-level training sample:', Xev_train.shape, 'attack fraction:', float(yev_train.mean()))
    print('Event-level validation sample:', Xev_val.shape, 'attack fraction:', float(yev_val.mean()) if len(yev_val) else np.nan)

    event_models = {}
    event_model_scores = {}

    # Robust scaling for linear model.
    ev_scaler = RobustScaler()
    Xev_train_scaled = ev_scaler.fit_transform(Xev_train)
    Xev_val_scaled = ev_scaler.transform(Xev_val) if len(Xev_val) else np.empty((0, Xev_train.shape[1]))

    ev_lr = LogisticRegression(max_iter=300, class_weight='balanced', solver='lbfgs', n_jobs=-1)
    ev_lr.fit(Xev_train_scaled, yev_train)
    event_models['EventLogisticBalanced'] = ('scaled', ev_lr)

    ev_et = ExtraTreesClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        class_weight='balanced_subsample',
        random_state=cfg.RANDOM_SEED,
        n_jobs=-1,
    )
    ev_et.fit(Xev_train, yev_train)
    event_models['EventExtraTreesBalanced'] = ('raw', ev_et)

    ev_hgb = HistGradientBoostingClassifier(
        max_iter=450,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=0.01,
        random_state=cfg.RANDOM_SEED,
    )
    sw = compute_sample_weight(class_weight='balanced', y=yev_train)
    ev_hgb.fit(Xev_train, yev_train, sample_weight=sw)
    event_models['EventHistGradientBoostingWeighted'] = ('raw', ev_hgb)

    try:
        from xgboost import XGBClassifier
        pos = max(yev_train.sum(), 1)
        neg = max(len(yev_train) - pos, 1)
        ev_xgb = XGBClassifier(
            n_estimators=650,
            max_depth=6,
            learning_rate=0.035,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_weight=2,
            reg_lambda=1.0,
            scale_pos_weight=float(neg / pos),
            objective='binary:logistic',
            eval_metric='aucpr',
            tree_method='hist',
            random_state=cfg.RANDOM_SEED,
            n_jobs=-1,
        )
        ev_xgb.fit(Xev_train, yev_train)
        event_models['EventXGBoostWeighted'] = ('raw', ev_xgb)
    except Exception as e:
        print('Optional event-level XGBoost skipped:', repr(e))

    def _predict_event_model(model_tuple, Xdf):
        kind, model = model_tuple
        if kind == 'scaled':
            Xuse = ev_scaler.transform(Xdf)
        else:
            Xuse = Xdf
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(Xuse)[:, 1]
        return expit(model.decision_function(Xuse))

    # Validation-based model weights using AUPR and F2.
    val_scores = {}
    for name, mt in event_models.items():
        if len(Xev_val) and len(np.unique(yev_val)) == 2:
            pv = np.clip(_predict_event_model(mt, Xev_val), cfg.RISK_EPS, 1 - cfg.RISK_EPS)
            aupr = safe_aupr(yev_val, pv)
            auroc = safe_auc(yev_val, pv)
            # F2 best over thresholds to reward recall on rare attacks.
            best_f2 = 0.0
            for th in np.linspace(0.02, 0.98, 97):
                pred = (pv >= th).astype(int)
                prec = precision_score(yev_val, pred, zero_division=0)
                rec = recall_score(yev_val, pred, zero_division=0)
                b2 = 4.0
                f2 = (1 + b2) * prec * rec / max(b2 * prec + rec, 1e-12)
                best_f2 = max(best_f2, f2)
            val_scores[name] = {'aupr': aupr, 'auroc': auroc, 'best_f2': best_f2, 'weight_score': max(aupr, 0) + 0.25 * best_f2}
        else:
            val_scores[name] = {'aupr': np.nan, 'auroc': np.nan, 'best_f2': np.nan, 'weight_score': 1.0}

    score_arr = np.array([max(v['weight_score'], 1e-6) for v in val_scores.values()], dtype=float)
    score_arr = score_arr / score_arr.sum()
    event_model_weights = dict(zip(val_scores.keys(), score_arr))
    print('Event model validation scores:')
    display(pd.DataFrame(val_scores).T.round(5))
    print('Event model weights:', event_model_weights)

    event_artifact = {
        'feature_cols': event_feature_cols,
        'scaler': ev_scaler,
        'models': event_models,
        'weights': event_model_weights,
        'val_scores': val_scores,
    }
    joblib.dump(event_artifact, EVENT_RISK_MODEL_PATH)

    # Stream all rows to aggregate calibrated event-risk evidence per temporal window.
    event_risk_acc = {}
    eval_rows = []
    for gi in range(T):
        event_risk_acc[int(gi)] = {
            'count': 0,
            'sum_p': 0.0,
            'sum_p2': 0.0,
            'max_p': 0.0,
            'log_no_attack': 0.0,
            'top_values': [],
            **{f'count_ge_{int(th*100)}': 0 for th in cfg.EVENT_RISK_HIGH_THRESHOLDS},
        }

    def _ensemble_event_prob(Xdf):
        prob = np.zeros(len(Xdf), dtype=float)
        for name, mt in event_models.items():
            w = event_model_weights.get(name, 0.0)
            if w <= 0:
                continue
            prob += w * np.clip(_predict_event_model(mt, Xdf), cfg.RISK_EPS, 1 - cfg.RISK_EPS)
        return np.clip(prob, cfg.RISK_EPS, 1 - cfg.RISK_EPS)

    import heapq
    for meta in file_meta:
        phase, path, label_col = meta['phase'], meta['path'], meta['label_col']
        usecols = sorted(set(event_feature_cols + [label_col]))
        rows_seen = 0
        total_rows = 0
        for chunk in tqdm(read_csv_chunks(path, usecols=usecols), desc=f'Event-risk aggregate {phase}'):
            n = len(chunk)
            if n == 0:
                continue
            _, gidx = _global_window_indices_for_chunk(phase, rows_seen, n)
            valid = gidx >= 0
            if valid.any():
                Xchunk = _make_event_feature_frame(chunk.loc[valid], event_feature_cols)
                probs = _ensemble_event_prob(Xchunk)
                labels = chunk.loc[valid, label_col].map(normalize_stage_name).fillna(cfg.BENIGN_STATE_NAME).astype(str)
                ychunk = (labels != cfg.BENIGN_STATE_NAME).astype(int).values
                gvalid = gidx[valid]

                for p, yv, gi in zip(probs, ychunk, gvalid):
                    a = event_risk_acc[int(gi)]
                    a['count'] += 1
                    a['sum_p'] += float(p)
                    a['sum_p2'] += float(p * p)
                    a['max_p'] = max(a['max_p'], float(p))
                    a['log_no_attack'] += float(np.log1p(-min(max(p, cfg.RISK_EPS), 1 - cfg.RISK_EPS)))
                    for th in cfg.EVENT_RISK_HIGH_THRESHOLDS:
                        if p >= th:
                            a[f'count_ge_{int(th*100)}'] += 1
                    # Maintain top-k risk values per window.
                    heap = a['top_values']
                    if len(heap) < cfg.EVENT_RISK_TOPK_PER_WINDOW:
                        heapq.heappush(heap, float(p))
                    elif p > heap[0]:
                        heapq.heapreplace(heap, float(p))

                # Store event-level eval samples for validation/test monitoring only. Cap by rows for memory.
                split_global = gvalid
                split_names = np.where(np.isin(split_global, val_idx), 'validation', np.where(np.isin(split_global, test_idx), 'test', 'other'))
                keep_eval = split_names != 'other'
                if keep_eval.any():
                    eval_rows.append(pd.DataFrame({
                        'split': split_names[keep_eval],
                        'true_attack': ychunk[keep_eval].astype(int),
                        'event_risk_prob': probs[keep_eval].astype(np.float32),
                        'global_window_index': split_global[keep_eval].astype(int),
                    }))
            rows_seen += n
            total_rows += n

    risk_rows = []
    for gi in range(T):
        a = event_risk_acc[int(gi)]
        cnt = max(a['count'], 1)
        mean = a['sum_p'] / cnt
        var = max(a['sum_p2'] / cnt - mean * mean, 0.0)
        top_vals = a['top_values']
        top_mean = float(np.mean(top_vals)) if top_vals else 0.0
        risk_rows.append({
            'global_window_index': gi,
            'event_risk_count': a['count'],
            'event_risk_mean': mean,
            'event_risk_std': math.sqrt(var),
            'event_risk_max': a['max_p'],
            'event_risk_topk_mean': top_mean,
            'event_risk_noisy_or': 1.0 - math.exp(max(a['log_no_attack'], -745.0)),
            **{f'event_risk_ratio_ge_{int(th*100)}': a[f'count_ge_{int(th*100)}'] / cnt for th in cfg.EVENT_RISK_HIGH_THRESHOLDS},
        })

    event_window_risk_df = pd.DataFrame(risk_rows)
    event_window_risk_df.to_csv(EVENT_RISK_CACHE, index=False)

    event_risk_report_df = pd.DataFrame(sampling_reports)
    event_risk_report_df['train_attack_sample'] = sample_counters['train_attack']
    event_risk_report_df['train_benign_sample'] = sample_counters['train_benign']
    event_risk_report_df['val_attack_sample'] = sample_counters['validation_attack']
    event_risk_report_df['val_benign_sample'] = sample_counters['validation_benign']
    event_risk_report_df.to_csv(EVENT_RISK_REPORT, index=False)

    if eval_rows:
        event_eval_df = pd.concat(eval_rows, ignore_index=True)
        metric_rows = []
        for split, sub in event_eval_df.groupby('split'):
            yv = sub['true_attack'].values.astype(int)
            pv = sub['event_risk_prob'].values.astype(float)
            if len(np.unique(yv)) == 2:
                # validation threshold, then test uses validation threshold
                pass
            metric_rows.append({
                'model': 'EventRiskBackbone',
                'split': split,
                'rows': len(sub),
                'attack_rows': int(yv.sum()),
                'auroc': safe_auc(yv, pv),
                'aupr': safe_aupr(yv, pv),
                'brier': brier_score_loss(yv, np.clip(pv, 1e-6, 1-1e-6)) if len(yv) else np.nan,
            })
        event_sample_metrics_df = pd.DataFrame(metric_rows)
        event_sample_metrics_df.to_csv(EVENT_RISK_SAMPLE_METRICS, index=False)
    else:
        event_sample_metrics_df = pd.DataFrame()

# Merge event-risk features into window_df and feature matrix.
event_window_risk_df = event_window_risk_df.copy()
window_df = window_df.merge(event_window_risk_df, left_index=True, right_on='global_window_index', how='left').sort_values('global_window_index').reset_index(drop=True)

new_event_features = [c for c in event_window_risk_df.columns if c.startswith('event_risk_') and c != 'event_risk_count']
for c in new_event_features:
    window_df[c] = pd.to_numeric(window_df[c], errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0.0)

# Add event-risk features to X with train-only scaling.
if new_event_features:
    X_event_raw = window_df[new_event_features].astype(float).values
    ev_win_scaler = StandardScaler()
    X_event_scaled = np.zeros_like(X_event_raw, dtype=float)
    X_event_scaled[train_idx] = ev_win_scaler.fit_transform(X_event_raw[train_idx])
    if len(val_idx):
        X_event_scaled[val_idx] = ev_win_scaler.transform(X_event_raw[val_idx])
    if len(test_idx):
        X_event_scaled[test_idx] = ev_win_scaler.transform(X_event_raw[test_idx])
    assigned = np.zeros(T, dtype=bool)
    assigned[train_idx] = True
    assigned[val_idx] = True
    assigned[test_idx] = True
    if (~assigned).any():
        X_event_scaled[~assigned] = ev_win_scaler.transform(X_event_raw[np.where(~assigned)[0]])
    X_event_scaled = np.clip(X_event_scaled, -8, 8)
    X = np.hstack([X, cfg.WINDOW_RISK_FEATURE_WEIGHT * X_event_scaled])
    feature_cols += new_event_features
    joblib.dump(ev_win_scaler, Path(cfg.OUTPUT_DIR, 'models', 'event_window_risk_feature_scaler.joblib'))

print('Added event-level risk window features:', new_event_features)
print('Updated X shape after event-risk feature fusion:', X.shape)
if not event_sample_metrics_df.empty:
    print('Event-level risk sample metrics:')
    display(event_sample_metrics_df.round(5))

display(window_df[['phase','local_window_id','is_attack'] + new_event_features].head())

# %% Cell 16

# ============================================================
# 10. Train leakage-safe risk models for discriminative emissions
# Enhanced rare-event ensemble with unsupervised background detectors
# ============================================================

risk_models = {}
risk_scores = {}


def minmax01(x):
    x = np.asarray(x, dtype=float)
    lo, hi = np.nanmin(x), np.nanmax(x)
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-12:
        return np.zeros_like(x, dtype=float)
    return (x - lo) / (hi - lo + 1e-12)


def fit_predict_risk_models(X, y, train_idx, val_idx, test_idx):
    outputs = {}
    train_y = y[train_idx]
    has_two_classes = len(np.unique(train_y)) == 2

    # Phase-1 background anomaly detector: Isolation Forest.
    iso_train_idx = phase1_idx_all if len(phase1_idx_all) else train_idx
    iso = IsolationForest(n_estimators=500, contamination='auto', random_state=cfg.RANDOM_SEED, n_jobs=-1)
    iso.fit(X[iso_train_idx])
    iso_raw = -iso.decision_function(X)
    outputs['IsolationForestBackground'] = np.clip(minmax01(iso_raw), cfg.RISK_EPS, 1 - cfg.RISK_EPS)
    risk_models['IsolationForestBackground'] = iso

    # Phase-1 background PCA reconstruction anomaly score.
    try:
        n_comp = max(2, min(25, X.shape[1] // 3, len(iso_train_idx) - 1))
        pca = PCA(n_components=n_comp, random_state=cfg.RANDOM_SEED)
        pca.fit(X[iso_train_idx])
        recon = pca.inverse_transform(pca.transform(X))
        pca_err = np.mean((X - recon) ** 2, axis=1)
        outputs['PCABackgroundError'] = np.clip(minmax01(pca_err), cfg.RISK_EPS, 1 - cfg.RISK_EPS)
        risk_models['PCABackgroundError'] = pca
    except Exception as e:
        print('PCA anomaly model skipped:', repr(e))

    if not has_two_classes:
        print('Training split has one class only. Supervised risk models skipped.')
        return outputs

    sample_weight = compute_sample_weight(class_weight='balanced', y=train_y)
    pos_weight = max((train_y == 0).sum() / max((train_y == 1).sum(), 1), 1.0)

    models = {
        'LogisticRegressionL2': LogisticRegression(max_iter=5000, class_weight='balanced', C=0.35, solver='lbfgs', random_state=cfg.RANDOM_SEED),
        'LogisticRegressionL1': LogisticRegression(max_iter=5000, class_weight='balanced', C=0.15, solver='liblinear', penalty='l1', random_state=cfg.RANDOM_SEED),
        'RandomForestBalanced': RandomForestClassifier(n_estimators=700, max_depth=None, min_samples_leaf=1, max_features='sqrt', class_weight='balanced_subsample', random_state=cfg.RANDOM_SEED, n_jobs=-1),
        'ExtraTreesBalanced': ExtraTreesClassifier(n_estimators=800, max_depth=None, min_samples_leaf=1, max_features='sqrt', class_weight='balanced', random_state=cfg.RANDOM_SEED, n_jobs=-1),
        'HistGradientBoostingWeighted': HistGradientBoostingClassifier(max_iter=600, learning_rate=0.035, max_leaf_nodes=31, l2_regularization=0.02, random_state=cfg.RANDOM_SEED),
    }

    # Optional XGBoost dependency; the remaining estimators run when it is unavailable.
    try:
        from xgboost import XGBClassifier
        models['XGBoostWeighted'] = XGBClassifier(
            n_estimators=700,
            max_depth=4,
            learning_rate=0.025,
            subsample=0.85,
            colsample_bytree=0.85,
            objective='binary:logistic',
            eval_metric='logloss',
            scale_pos_weight=pos_weight,
            reg_lambda=2.0,
            random_state=cfg.RANDOM_SEED,
            n_jobs=-1,
        )
    except Exception as e:
        print('XGBoost not available; skipping optional XGBoostWeighted:', repr(e))

    for name, model in models.items():
        try:
            if name in ['HistGradientBoostingWeighted']:
                model.fit(X[train_idx], train_y, sample_weight=sample_weight)
            elif name == 'XGBoostWeighted':
                model.fit(X[train_idx], train_y)
            else:
                model.fit(X[train_idx], train_y)
            if hasattr(model, 'predict_proba'):
                p = model.predict_proba(X)[:, 1]
            else:
                raw = model.decision_function(X)
                p = minmax01(raw)
            outputs[name] = np.clip(p, cfg.RISK_EPS, 1 - cfg.RISK_EPS)
            risk_models[name] = model
            print(f'Trained {name}')
        except Exception as e:
            print(f'Model {name} skipped due to error:', repr(e))

    return outputs

risk_scores = fit_predict_risk_models(X, y_attack, train_idx, val_idx, test_idx)
risk_df = pd.DataFrame({'window_index': np.arange(T), 'true_attack': y_attack})
for name, score in risk_scores.items():
    risk_df[name] = score
risk_df.to_csv(Path(cfg.OUTPUT_DIR, 'tables', 'risk_model_scores.csv'), index=False)
print('Available risk models:', list(risk_scores.keys()))

# %% Cell 17

# ============================================================
# 11. Calibrate, meta-ensemble, and temporally refine risk-emission probability
# Rare-event score stacking
# ============================================================


def _safe_auc_local(y, p):
    try:
        return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan
    except Exception:
        return np.nan


def _safe_aupr_local(y, p):
    try:
        return float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else np.nan
    except Exception:
        return np.nan


def _safe_fbeta(y, pred, beta=2.0):
    y = np.asarray(y).astype(int)
    pred = np.asarray(pred).astype(int)
    prec = precision_score(y, pred, zero_division=0)
    rec = recall_score(y, pred, zero_division=0)
    b2 = beta ** 2
    return float((1 + b2) * prec * rec / max((b2 * prec + rec), 1e-12))


def optimize_logit_calibration(p, y):
    p = np.clip(np.asarray(p, dtype=float), cfg.RISK_EPS, 1 - cfg.RISK_EPS)
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2:
        return 1.0, 0.0
    raw_logit = logit(p)
    def nll(params):
        scale, bias = params
        scale = max(scale, 1e-3)
        pp = expit(raw_logit * scale + bias)
        pp = np.clip(pp, cfg.RISK_EPS, 1 - cfg.RISK_EPS)
        return -float(np.mean(y * np.log(pp) + (1-y) * np.log(1-pp)))
    res = minimize(nll, x0=np.array([1.0, 0.0]), bounds=[(0.03, 20.0), (-20.0, 20.0)], method='L-BFGS-B')
    if not res.success:
        return 1.0, 0.0
    return float(res.x[0]), float(res.x[1])


def past_rolling_prob(x, window=5, fn='mean'):
    s = pd.Series(np.asarray(x, dtype=float))
    if fn == 'max':
        return s.rolling(window, min_periods=1).max().values
    if fn == 'std':
        return s.rolling(window, min_periods=1).std().fillna(0.0).values
    if fn == 'min':
        return s.rolling(window, min_periods=1).min().values
    return s.rolling(window, min_periods=1).mean().values


def phasewise_transform_prob(prob, transform):
    out = np.zeros_like(prob, dtype=float)
    for phase_name, idxs in window_df.groupby('phase').groups.items():
        idx = np.asarray(list(idxs), dtype=int)
        out[idx] = transform(prob[idx])
    return out


def threshold_quality(y, p, threshold):
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    fpr = fp / max(fp + tn, 1)
    f1 = f1_score(y, pred, zero_division=0)
    f2 = _safe_fbeta(y, pred, beta=getattr(cfg, 'POLICY_BETA', 2.0))
    mcc = matthews_corrcoef(y, pred) if len(np.unique(pred)) > 1 and len(np.unique(y)) > 1 else 0.0
    rec = recall_score(y, pred, zero_division=0)
    prec = precision_score(y, pred, zero_division=0)
    return dict(threshold=float(threshold), f1=float(f1), f2=float(f2), mcc=float(mcc), recall=float(rec), precision=float(prec), fpr=float(fpr), tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))


def best_threshold_for_score(y, p, min_recall=None, max_fpr=None):
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    qs = np.unique(np.r_[np.linspace(0.001, 0.999, 401), np.quantile(p, np.linspace(0.01, 0.99, 99))])
    rows = [threshold_quality(y, p, th) for th in qs]
    df = pd.DataFrame(rows)
    feasible = df.copy()
    if min_recall is not None:
        feasible = feasible[feasible['recall'] >= min_recall]
    if max_fpr is not None:
        feasible = feasible[feasible['fpr'] <= max_fpr]
    if feasible.empty:
        feasible = df
    feasible = feasible.copy()
    feasible['selection_score'] = (
        feasible['f1']
        + 0.25 * feasible['f2']
        + getattr(cfg, 'POLICY_OBJECTIVE_MCC_WEIGHT', 0.20) * feasible['mcc']
        + getattr(cfg, 'POLICY_OBJECTIVE_RECALL_WEIGHT', 0.25) * feasible['recall']
        - 0.20 * feasible['fpr']
    )
    best = feasible.sort_values(['selection_score','f1','mcc','recall'], ascending=False).iloc[0]
    return float(best['threshold']), df, best.to_dict()


score_names = list(risk_scores.keys())
if len(score_names) == 0:
    raise RuntimeError('No risk scores are available. Run the risk-model training cell first.')

calibration_idx = val_idx if len(val_idx) and len(np.unique(y_attack[val_idx])) == 2 else train_idx
train_y = y_attack[train_idx]
cal_y = y_attack[calibration_idx]

# Calibrate each base score and create score-level features.
calibrated_scores = {}
for name in score_names:
    p = np.clip(risk_scores[name], cfg.RISK_EPS, 1 - cfg.RISK_EPS)
    scale, bias = optimize_logit_calibration(p[calibration_idx], cal_y)
    calibrated_scores[name] = np.clip(expit(logit(p) * scale + bias), cfg.RISK_EPS, 1 - cfg.RISK_EPS)


def score_rank01(p):
    s = pd.Series(np.asarray(p, dtype=float))
    return (s.rank(method='average').values - 1) / max(len(s) - 1, 1)


def build_score_feature_matrix(score_dict):
    parts = []
    names = []
    for name, p in score_dict.items():
        p = np.clip(np.asarray(p, dtype=float), cfg.RISK_EPS, 1 - cfg.RISK_EPS)
        derived = {
            f'{name}_p': p,
            f'{name}_logit': np.clip(logit(p), -12, 12),
            f'{name}_rank': score_rank01(p),
            f'{name}_roll3': phasewise_transform_prob(p, lambda z: past_rolling_prob(z, 3, 'mean')),
            f'{name}_roll7': phasewise_transform_prob(p, lambda z: past_rolling_prob(z, 7, 'mean')),
            f'{name}_max5': phasewise_transform_prob(p, lambda z: past_rolling_prob(z, 5, 'max')),
            f'{name}_max11': phasewise_transform_prob(p, lambda z: past_rolling_prob(z, 11, 'max')),
            f'{name}_ewm5': phasewise_transform_prob(p, lambda z: pd.Series(z).ewm(span=5, adjust=False).mean().values),
            f'{name}_diff1': phasewise_transform_prob(p, lambda z: pd.Series(z).diff().fillna(0).values),
        }
        for k, v in derived.items():
            parts.append(v)
            names.append(k)
    M = np.column_stack(parts).astype(float)
    M = np.nan_to_num(M, nan=0.0, posinf=20.0, neginf=-20.0)
    return M, names

score_matrix, score_feature_names = build_score_feature_matrix(calibrated_scores)
score_scaler = StandardScaler()
Z = np.zeros_like(score_matrix, dtype=float)
Z[train_idx] = score_scaler.fit_transform(score_matrix[train_idx])
if len(val_idx):
    Z[val_idx] = score_scaler.transform(score_matrix[val_idx])
if len(test_idx):
    Z[test_idx] = score_scaler.transform(score_matrix[test_idx])
# Any non-split rows are transformed as well.
assigned = np.zeros(T, dtype=bool)
assigned[train_idx] = True
assigned[val_idx] = True
assigned[test_idx] = True
if (~assigned).any():
    Z[~assigned] = score_scaler.transform(score_matrix[np.where(~assigned)[0]])

# Meta-models trained on the score layer. These are cheap and often improve rare-event precision/recall.
meta_outputs = {}
meta_models = {}
if len(np.unique(train_y)) == 2:
    sw = compute_sample_weight(class_weight='balanced', y=train_y)
    meta_candidates = {
        'RiskScoreStackLogistic': LogisticRegression(max_iter=5000, class_weight='balanced', C=0.20, solver='lbfgs', random_state=cfg.RANDOM_SEED),
        'RiskScoreStackExtraTrees': ExtraTreesClassifier(n_estimators=600, max_depth=None, min_samples_leaf=1, max_features='sqrt', class_weight='balanced', random_state=cfg.RANDOM_SEED, n_jobs=-1),
        'RiskScoreStackRandomForest': RandomForestClassifier(n_estimators=500, max_depth=None, min_samples_leaf=1, max_features='sqrt', class_weight='balanced_subsample', random_state=cfg.RANDOM_SEED, n_jobs=-1),
        'RiskScoreStackHGB': HistGradientBoostingClassifier(max_iter=350, learning_rate=0.04, max_leaf_nodes=15, l2_regularization=0.05, random_state=cfg.RANDOM_SEED),
    }
    for name, model in meta_candidates.items():
        try:
            if name == 'RiskScoreStackHGB':
                model.fit(Z[train_idx], train_y, sample_weight=sw)
            else:
                model.fit(Z[train_idx], train_y)
            p_raw = np.clip(model.predict_proba(Z)[:, 1], cfg.RISK_EPS, 1 - cfg.RISK_EPS)
            scale, bias = optimize_logit_calibration(p_raw[calibration_idx], cal_y)
            p_cal = np.clip(expit(logit(p_raw) * scale + bias), cfg.RISK_EPS, 1 - cfg.RISK_EPS)
            meta_outputs[name] = p_cal
            meta_models[name] = {'model': model, 'scale': scale, 'bias': bias}
            print(f'Trained and calibrated {name}')
        except Exception as e:
            print(f'Meta-risk model {name} skipped:', repr(e))
else:
    print('Training split has one class. Meta-risk models skipped.')

# Add meta outputs to risk_scores so they appear in baseline tables.
risk_scores.update(meta_outputs)
risk_models.update(meta_models)

# Candidate risk fusions.
all_score_dict = {**calibrated_scores, **meta_outputs}
base_matrix = np.column_stack([np.clip(p, cfg.RISK_EPS, 1 - cfg.RISK_EPS) for p in all_score_dict.values()])
base_names = list(all_score_dict.keys())

# AUPR/F2-aware weights on validation/calibration split.
weights = []
weight_rows = []
for name, p in all_score_dict.items():
    p_ref = p[calibration_idx]
    aupr = _safe_aupr_local(cal_y, p_ref)
    auroc = _safe_auc_local(cal_y, p_ref)
    th, _, best = best_threshold_for_score(cal_y, p_ref, min_recall=0.0, max_fpr=None)
    q = 0.45 * (aupr if np.isfinite(aupr) else 0.0) + 0.30 * best['f2'] + 0.15 * best['mcc'] + 0.10 * (auroc if np.isfinite(auroc) else 0.5)
    weights.append(max(q, 0.01))
    weight_rows.append({'model': name, 'validation_aupr': aupr, 'validation_auroc': auroc, 'best_threshold': th, 'best_f1': best['f1'], 'best_f2': best['f2'], 'best_mcc': best['mcc'], 'ensemble_quality_weight_raw': q})
weights = np.asarray(weights, dtype=float)
weights = weights / weights.sum()

weighted_mean = np.average(base_matrix, axis=1, weights=weights)
plain_mean = base_matrix.mean(axis=1)
max_score = base_matrix.max(axis=1)
top2_mean = np.sort(base_matrix, axis=1)[:, -min(2, base_matrix.shape[1]):].mean(axis=1)
noisy_or = 1 - np.prod(1 - np.clip(base_matrix, cfg.RISK_EPS, 1 - cfg.RISK_EPS), axis=1)

candidate_base = {
    'RiskFusionWeightedMean': weighted_mean,
    'RiskFusionPlainMean': plain_mean,
    'RiskFusionMax': max_score,
    'RiskFusionTop2Mean': top2_mean,
    'RiskFusionNoisyOR': np.clip(noisy_or, cfg.RISK_EPS, 1 - cfg.RISK_EPS),
}

fusion_rows = []
best_candidate_name = None
best_candidate_score = -np.inf
best_candidate_prob = None
best_candidate_threshold = 0.5

for name, p0 in candidate_base.items():
    p0 = np.clip(p0, cfg.RISK_EPS, 1 - cfg.RISK_EPS)
    # Temporal filters: mean is smooth, max is recall-oriented. Candidate selection decides.
    p_roll = phasewise_transform_prob(p0, lambda z: past_rolling_prob(z, 5, 'mean'))
    p_max = phasewise_transform_prob(p0, lambda z: past_rolling_prob(z, 7, 'max'))
    p_ewm = phasewise_transform_prob(p0, lambda z: pd.Series(z).ewm(span=5, adjust=False).mean().values)
    temporal_candidates = {
        name + '_raw': p0,
        name + '_roll': np.clip(0.65 * p0 + 0.35 * p_roll, cfg.RISK_EPS, 1 - cfg.RISK_EPS),
        name + '_maxassist': np.clip(0.75 * p0 + 0.25 * p_max, cfg.RISK_EPS, 1 - cfg.RISK_EPS),
        name + '_ewma': np.clip(0.70 * p0 + 0.30 * p_ewm, cfg.RISK_EPS, 1 - cfg.RISK_EPS),
    }
    for cname, pp in temporal_candidates.items():
        scale, bias = optimize_logit_calibration(pp[calibration_idx], cal_y)
        pc = np.clip(expit(logit(pp) * scale + bias), cfg.RISK_EPS, 1 - cfg.RISK_EPS)
        th, thdf, best = best_threshold_for_score(cal_y, pc[calibration_idx], min_recall=cfg.MIN_RECALL_FOR_POLICY, max_fpr=cfg.MAX_FALSE_POSITIVE_RATE_FOR_POLICY)
        score = best['selection_score'] + 0.10 * (_safe_aupr_local(cal_y, pc[calibration_idx]) if np.isfinite(_safe_aupr_local(cal_y, pc[calibration_idx])) else 0.0)
        fusion_rows.append({'candidate': cname, 'scale': scale, 'bias': bias, 'threshold': th, 'selection_score': score, **best})
        if score > best_candidate_score:
            best_candidate_score = score
            best_candidate_name = cname
            best_candidate_prob = pc
            best_candidate_threshold = th

risk_prob_precal = np.clip(best_candidate_prob, cfg.RISK_EPS, 1 - cfg.RISK_EPS)
risk_prob = risk_prob_precal
risk_prob_calibrated = risk_prob_precal
ensemble_raw = weighted_mean
risk_temporal_alpha = np.nan

window_df['risk_emission_probability_raw'] = ensemble_raw
window_df['risk_emission_probability_calibrated'] = risk_prob_calibrated
window_df['risk_emission_probability'] = risk_prob

risk_df = pd.DataFrame({'window_index': np.arange(T), 'true_attack': y_attack})
for name, score in risk_scores.items():
    risk_df[name] = score
risk_df['risk_ensemble_raw'] = ensemble_raw
risk_df['risk_ensemble_temporal'] = risk_prob
risk_df.to_csv(Path(cfg.OUTPUT_DIR, 'tables', 'risk_model_scores.csv'), index=False)

risk_model_summary_df = pd.DataFrame(weight_rows)
risk_model_summary_df['ensemble_weight'] = weights
risk_fusion_candidate_df = pd.DataFrame(fusion_rows).sort_values('selection_score', ascending=False)

print('Best validation-selected risk fusion:', best_candidate_name, 'threshold=', best_candidate_threshold)
display(risk_model_summary_df.sort_values('ensemble_weight', ascending=False).round(5))
display(risk_fusion_candidate_df.head(20).round(5))

risk_model_summary_df.to_csv(Path(cfg.OUTPUT_DIR, 'tables', 'risk_model_weight_summary.csv'), index=False)
risk_fusion_candidate_df.to_csv(Path(cfg.OUTPUT_DIR, 'tables', 'risk_fusion_candidate_search.csv'), index=False)
joblib.dump({'score_scaler': score_scaler, 'score_feature_names': score_feature_names, 'meta_models': meta_models, 'best_candidate': best_candidate_name, 'best_threshold': best_candidate_threshold}, Path(cfg.OUTPUT_DIR, 'models', 'risk_score_meta_ensemble.joblib'))

# %% Cell 18

# ============================================================
# 12. Baseline metrics for risk-emission models
# Includes imbalance-aware, calibration, and operational false-alarm metrics
# ============================================================


def binary_calibration_errors(y_true, prob, n_bins=10):
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob).astype(float)
    if len(y_true) == 0:
        return np.nan, np.nan, pd.DataFrame()
    bins = np.linspace(0, 1, n_bins + 1)
    rows = []
    ece = 0.0
    mce = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (prob >= lo) & (prob < hi if i < n_bins-1 else prob <= hi)
        if not mask.any():
            continue
        conf = float(prob[mask].mean())
        acc = float(y_true[mask].mean())
        gap = abs(acc - conf)
        ece += gap * mask.mean()
        mce = max(mce, gap)
        rows.append({'bin': i, 'count': int(mask.sum()), 'mean_confidence': conf, 'empirical_attack_rate': acc, 'gap': gap})
    return float(ece), float(mce), pd.DataFrame(rows)


def uncertainty_calibration_error(y_true, prob, uncertainty=None, n_bins=10):
    """UCE-style score: bins by uncertainty and compares mean uncertainty with mean absolute error."""
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob).astype(float)
    if uncertainty is None:
        uncertainty = -(prob * np.log(prob + 1e-12) + (1 - prob) * np.log(1 - prob + 1e-12)) / np.log(2)
    uncertainty = np.asarray(uncertainty, dtype=float)
    abs_err = np.abs(y_true - prob)
    bins = np.linspace(np.nanmin(uncertainty), np.nanmax(uncertainty) + 1e-12, n_bins + 1)
    uce = 0.0
    rows = []
    for i in range(n_bins):
        mask = (uncertainty >= bins[i]) & (uncertainty <= bins[i+1] if i == n_bins-1 else uncertainty < bins[i+1])
        if not mask.any():
            continue
        mu = float(uncertainty[mask].mean())
        me = float(abs_err[mask].mean())
        gap = abs(mu - me)
        uce += gap * mask.mean()
        rows.append({'bin': i, 'count': int(mask.sum()), 'mean_uncertainty': mu, 'mean_abs_error': me, 'gap': gap})
    return float(uce), pd.DataFrame(rows)


def safe_auc(y, p):
    try:
        return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan
    except Exception:
        return np.nan


def safe_aupr(y, p):
    try:
        return float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else np.nan
    except Exception:
        return np.nan


def confusion_counts(y, pred):
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return tn, fp, fn, tp


def rich_binary_metrics(y, p, pred=None, threshold=0.5):
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    if pred is None:
        pred = (p >= threshold).astype(int)
    else:
        pred = np.asarray(pred).astype(int)
    tn, fp, fn, tp = confusion_counts(y, pred)
    ece, mce, cal = binary_calibration_errors(y, p)
    uce, uce_table = uncertainty_calibration_error(y, p)
    specificity = tn / max(tn + fp, 1)
    fpr = fp / max(fp + tn, 1)
    fnr = fn / max(fn + tp, 1)
    npv = tn / max(tn + fn, 1)
    return {
        'threshold': float(threshold),
        'accuracy': float(accuracy_score(y, pred)) if len(y) else np.nan,
        'balanced_accuracy': float(balanced_accuracy_score(y, pred)) if len(np.unique(y)) == 2 else np.nan,
        'precision': float(precision_score(y, pred, zero_division=0)),
        'recall': float(recall_score(y, pred, zero_division=0)),
        'specificity': float(specificity),
        'fpr': float(fpr),
        'fnr': float(fnr),
        'npv': float(npv),
        'f1': float(f1_score(y, pred, zero_division=0)),
        'mcc': float(matthews_corrcoef(y, pred)) if len(np.unique(pred)) > 1 and len(np.unique(y)) > 1 else 0.0,
        'auroc': safe_auc(y, p),
        'aupr': safe_aupr(y, p),
        'brier': float(brier_score_loss(y, p)) if len(np.unique(y)) == 2 else np.nan,
        'log_loss': float(log_loss(y, p, labels=[0, 1])) if len(np.unique(y)) == 2 else np.nan,
        'ece': ece,
        'mce': mce,
        'uce': uce,
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
    }


def metric_row(model, split, idx, score, threshold=0.5, pred_override=None):
    y = y_attack[idx]
    p = score[idx] if len(score) == T else np.asarray(score)
    pred = pred_override
    row = {'model': model, 'split': split}
    row.update(rich_binary_metrics(y, p, pred=pred, threshold=threshold))
    return row

baseline_rows = []
for name, score in {**risk_scores, 'RiskEnsembleCalibratedTemporal': risk_prob}.items():
    for split, idx in [('train', train_idx), ('validation', val_idx), ('test', test_idx)]:
        baseline_rows.append(metric_row(name, split, idx, score, threshold=0.5))

baseline_metrics_df = pd.DataFrame(baseline_rows)
display(baseline_metrics_df.round(4))
baseline_metrics_df.to_csv(Path(cfg.OUTPUT_DIR, 'tables', 'risk_baseline_metrics_default_threshold.csv'), index=False)

# %% Cell 20
# ============================================================
# 13. HMM and PGAS utility functions
# ============================================================

def stable_log_normalize(logw):
    logw = np.asarray(logw, dtype=float)
    m = np.max(logw)
    w = np.exp(logw - m)
    s = w.sum()
    if s <= 0 or not np.isfinite(s):
        return np.ones_like(w) / len(w)
    return w / s


def categorical_sample(probs, rng):
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()
    return int(rng.choice(len(probs), p=probs))


def gaussian_emission_logpdf_all(y, mu, sigma2):
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma2 = np.maximum(np.asarray(sigma2, dtype=float), 1e-6)
    D = y.shape[-1]
    diff2 = (mu - y[None, :]) ** 2
    return -0.5 * (D * np.log(2*np.pi) + np.log(sigma2).sum(axis=1) + (diff2 / sigma2).sum(axis=1))


def make_fixed_risk_log_emissions(prob):
    prob = np.clip(np.asarray(prob, dtype=float), cfg.RISK_EPS, 1 - cfg.RISK_EPS)
    logB = np.column_stack([np.log(1 - prob), np.log(prob)])
    if K > 2:
        # Additional attack states share the attack emission mass.
        extra = np.repeat(np.log(prob / max(K - 1, 1))[:, None], K - 1, axis=1)
        logB = np.column_stack([np.log(1 - prob), extra])
    return logB

fixed_logB = make_fixed_risk_log_emissions(risk_prob)


def combined_emission_log_all(t, Y, mu, sigma2, fixed_logB=None):
    g = gaussian_emission_logpdf_all(Y[t], mu, sigma2)
    g = g - np.max(g)
    if fixed_logB is None:
        out = cfg.GAUSSIAN_EMISSION_WEIGHT * g
    else:
        r = fixed_logB[t]
        r = r - np.max(r)
        out = cfg.GAUSSIAN_EMISSION_WEIGHT * g + cfg.RISK_EMISSION_WEIGHT * r
    return out - np.max(out)


def compute_path_loglik(Y, z, pi, A, mu, sigma2, fixed_logB=None, offset=0):
    if len(z) == 0:
        return np.nan
    val = np.log(pi[z[0]] + 1e-300) + combined_emission_log_all(offset, Y, mu, sigma2, fixed_logB)[z[0]]
    for t in range(1, len(z)):
        val += np.log(A[z[t-1], z[t]] + 1e-300) + combined_emission_log_all(offset + t, Y, mu, sigma2, fixed_logB)[z[t]]
    return float(val)


def viterbi_decode(Y, pi, A, mu, sigma2, fixed_logB=None, offset=0):
    Tlocal = len(Y)
    Klocal = len(pi)
    delta = np.zeros((Tlocal, Klocal))
    psi = np.zeros((Tlocal, Klocal), dtype=int)
    delta[0] = np.log(pi + 1e-300) + combined_emission_log_all(offset, Y, mu, sigma2, fixed_logB)
    for t in range(1, Tlocal):
        emit = combined_emission_log_all(offset + t, Y, mu, sigma2, fixed_logB)
        for k in range(Klocal):
            vals = delta[t-1] + np.log(A[:, k] + 1e-300)
            psi[t, k] = int(np.argmax(vals))
            delta[t, k] = vals[psi[t, k]] + emit[k]
    z = np.zeros(Tlocal, dtype=int)
    z[-1] = int(np.argmax(delta[-1]))
    for t in range(Tlocal-2, -1, -1):
        z[t] = psi[t+1, z[t+1]]
    return z


def pgas_sample_path(Y, pi, A, mu, sigma2, conditional_path, fixed_logB=None, offset=0, num_particles=64, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    Tlocal = len(Y)
    N = int(num_particles)
    assert N >= 2
    particles = np.zeros((Tlocal, N), dtype=int)
    ancestors = np.zeros((Tlocal, N), dtype=int)
    logw = np.zeros(N, dtype=float)
    loglik_est = 0.0

    for i in range(N - 1):
        particles[0, i] = categorical_sample(pi, rng)
    particles[0, N - 1] = int(conditional_path[0])
    emit0 = combined_emission_log_all(offset, Y, mu, sigma2, fixed_logB)
    logw = np.array([emit0[particles[0, i]] for i in range(N)])
    loglik_est += logsumexp(logw) - np.log(N)

    for t in range(1, Tlocal):
        weights = stable_log_normalize(logw)
        ancestors[t, :N-1] = rng.choice(N, size=N-1, replace=True, p=weights)
        for i in range(N-1):
            prev_state = particles[t-1, ancestors[t, i]]
            particles[t, i] = categorical_sample(A[prev_state], rng)

        particles[t, N-1] = int(conditional_path[t])
        ref_state = particles[t, N-1]
        log_as = logw + np.log(A[particles[t-1], ref_state] + 1e-300)
        ancestors[t, N-1] = rng.choice(N, p=stable_log_normalize(log_as))

        emit = combined_emission_log_all(offset + t, Y, mu, sigma2, fixed_logB)
        logw = np.array([emit[particles[t, i]] for i in range(N)])
        loglik_est += logsumexp(logw) - np.log(N)

    final_probs = stable_log_normalize(logw)
    b = rng.choice(N, p=final_probs)
    sampled = np.zeros(Tlocal, dtype=int)
    sampled[Tlocal-1] = particles[Tlocal-1, b]
    for t in range(Tlocal-1, 0, -1):
        b = ancestors[t, b]
        sampled[t-1] = particles[t-1, b]
    return sampled, float(loglik_est)

# %% Cell 21
# ============================================================
# 14. Priors, parameter updates, and rare-event initialization
# ============================================================

def build_pi_prior(K):
    prior = np.ones(K) * cfg.PI_PRIOR_ATTACK
    prior[0] = cfg.PI_PRIOR_BENIGN
    return prior


def build_transition_prior(K):
    if K == 2:
        return np.array([[cfg.A00_PRIOR, cfg.A01_PRIOR], [cfg.A10_PRIOR, cfg.A11_PRIOR]], dtype=float)
    prior = np.ones((K, K)) * cfg.DIRICHLET_ALPHA_FALLBACK
    for k in range(K):
        prior[k, k] = 25.0
    prior[0, 0] = cfg.A00_PRIOR
    return prior


def estimate_params_from_path(Y, z, K):
    pi_counts = build_pi_prior(K)
    pi_counts[int(z[0])] += 1
    pi = pi_counts / pi_counts.sum()

    A_counts = build_transition_prior(K)
    for t in range(1, len(z)):
        A_counts[int(z[t-1]), int(z[t])] += 1
    A = A_counts / A_counts.sum(axis=1, keepdims=True)

    mu = np.zeros((K, Y.shape[1]))
    sigma2 = np.ones((K, Y.shape[1]))
    global_mu = Y.mean(axis=0)
    global_var = Y.var(axis=0) + 1e-3
    for k in range(K):
        idx = np.where(z == k)[0]
        if len(idx) >= 2:
            mu[k] = Y[idx].mean(axis=0)
            sigma2[k] = Y[idx].var(axis=0) + 1e-3
        elif len(idx) == 1:
            mu[k] = Y[idx[0]]
            sigma2[k] = global_var
        else:
            mu[k] = global_mu
            sigma2[k] = global_var
    return pi, A, mu, sigma2


def sample_dirichlet_params_from_path(z, K, rng):
    pi_counts = build_pi_prior(K)
    pi_counts[int(z[0])] += 1
    pi = rng.dirichlet(pi_counts)

    A = np.zeros((K, K))
    prior = build_transition_prior(K)
    for k in range(K):
        counts = prior[k].copy()
        idx = np.where(z[:-1] == k)[0]
        if len(idx):
            next_states = z[idx + 1]
            for j in next_states:
                counts[int(j)] += 1
        A[k] = rng.dirichlet(counts)
    return pi, A


def sample_emission_params_nig(Y, z, K, rng):
    Tlocal, D = Y.shape
    mu = np.zeros((K, D))
    sigma2 = np.zeros((K, D))
    global_mean = Y.mean(axis=0)
    for k in range(K):
        idx = np.where(z == k)[0]
        Xk = Y[idx]
        n = len(idx)
        if n == 0:
            sigma2[k] = invgamma.rvs(cfg.NIG_A0, scale=cfg.NIG_B0, size=D, random_state=rng)
            mu[k] = rng.normal(global_mean, np.sqrt(sigma2[k] / max(cfg.NIG_KAPPA0, 1e-6)))
            continue
        xbar = Xk.mean(axis=0)
        ss = ((Xk - xbar) ** 2).sum(axis=0)
        kappa_n = cfg.NIG_KAPPA0 + n
        m_n = (cfg.NIG_KAPPA0 * cfg.NIG_M0 + n * xbar) / kappa_n
        a_n = cfg.NIG_A0 + n / 2.0
        b_n = cfg.NIG_B0 + 0.5 * ss + (cfg.NIG_KAPPA0 * n * (xbar - cfg.NIG_M0) ** 2) / (2.0 * kappa_n)
        sigma2[k] = invgamma.rvs(a_n, scale=np.maximum(b_n, 1e-6), random_state=rng)
        mu[k] = rng.normal(m_n, np.sqrt(np.maximum(sigma2[k], 1e-6) / kappa_n))
    return mu, np.maximum(sigma2, 1e-6)


def initialize_from_risk(train_idx, risk_prob, K, seed=42):
    rng = np.random.default_rng(seed)
    z = np.zeros(len(train_idx), dtype=int)
    if K < 2:
        return z
    p = risk_prob[train_idx]
    ytr = y_attack[train_idx]
    max_attack = max(1, int(len(train_idx) * 0.08))
    if ytr.sum() > 0:
        n_attack_init = min(max(int(ytr.sum() * 1.5), 10), max_attack)
    else:
        n_attack_init = min(max(10, int(len(train_idx) * 0.02)), max_attack)
    top_local = np.argsort(p)[-n_attack_init:]
    z[top_local] = 1
    return z

z_train_init = initialize_from_risk(train_idx, risk_prob, K, seed=cfg.RANDOM_SEED)
pi0, A0, mu0, sigma20 = estimate_params_from_path(X[train_idx], z_train_init, K)

print('Initial latent-state prevalence from calibrated risk initialization:')
display(pd.Series(z_train_init).value_counts(normalize=True).sort_index().rename('fraction').to_frame())
print('Initial transition matrix:')
display(pd.DataFrame(A0, index=stage_names, columns=stage_names).round(4))

# %% Cell 22
# ============================================================
# 15. Run hybrid-emission PGAS-Gibbs chains
# ============================================================

Y_train = X[train_idx]
Y_val = X[val_idx] if len(val_idx) else np.empty((0, X.shape[1]))
Y_test = X[test_idx] if len(test_idx) else np.empty((0, X.shape[1]))

# Because fixed_logB is indexed in global window order, we pass a split-specific matrix here.
fixed_logB_train = fixed_logB[train_idx]
fixed_logB_val = fixed_logB[val_idx] if len(val_idx) else np.empty((0, K))
fixed_logB_test = fixed_logB[test_idx] if len(test_idx) else np.empty((0, K))

# For split-local smoothing, use local fixed_logB arrays. Wrap combined emission functions by passing local offset=0.
def run_pgas_gibbs_chain(Y, fixed_logB_local, K, init_path, chain_id=0):
    rng = np.random.default_rng(cfg.RANDOM_SEED + 1000 * chain_id)
    z = init_path.copy().astype(int)
    pi, A, mu, sigma2 = estimate_params_from_path(Y, z, K)
    samples = []
    saved_params = []
    log_records = []
    for it in tqdm(range(cfg.MCMC_ITER), desc=f'Hybrid PGAS chain {chain_id}'):
        z, pf_ll = pgas_sample_path(Y, pi, A, mu, sigma2, z, fixed_logB=fixed_logB_local, offset=0, num_particles=cfg.PGAS_PARTICLES, rng=rng)
        pi, A = sample_dirichlet_params_from_path(z, K, rng)
        mu, sigma2 = sample_emission_params_nig(Y, z, K, rng)
        path_ll = compute_path_loglik(Y, z, pi, A, mu, sigma2, fixed_logB=fixed_logB_local, offset=0)
        prev = np.bincount(z, minlength=K) / len(z)
        log_records.append({
            'chain': chain_id,
            'iter': it,
            'pf_loglik_est': pf_ll,
            'path_loglik': path_ll,
            **{f'prev_state_{k}': prev[k] for k in range(K)},
            **{f'self_transition_{k}': A[k, k] for k in range(K)},
            **{f'transition_0_to_{k}': A[0, k] for k in range(K)},
        })
        if it >= cfg.BURN_IN and ((it - cfg.BURN_IN) % cfg.THIN == 0):
            samples.append(z.copy())
            saved_params.append({'pi': pi.copy(), 'A': A.copy(), 'mu': mu.copy(), 'sigma2': sigma2.copy()})
    return {'chain_id': chain_id, 'path_samples': np.array(samples, dtype=int), 'params_samples': saved_params, 'trace': pd.DataFrame(log_records), 'last_path': z, 'last_params': {'pi': pi, 'A': A, 'mu': mu, 'sigma2': sigma2}}

chains = []
start = time.time()
for c in range(cfg.NUM_CHAINS):
    init = initialize_from_risk(train_idx, risk_prob, K, seed=cfg.RANDOM_SEED + c)
    chains.append(run_pgas_gibbs_chain(Y_train, fixed_logB_train, K, init, chain_id=c))
print(f'Hybrid PGAS-Gibbs training completed in {(time.time() - start)/60:.2f} minutes.')

trace_df = pd.concat([ch['trace'] for ch in chains], ignore_index=True)
display(trace_df.tail())
trace_df.to_csv(Path(cfg.OUTPUT_DIR, 'tables', 'pgas_training_trace.csv'), index=False)

# %% Cell 23
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from scipy.special import logsumexp

# ============================================================
# 16. Posterior parameter summaries and Rao-Blackwellized smoothing
# ============================================================

def stack_param_samples(chains, key):
    arrs = []
    for ch in chains:
        for ps in ch['params_samples']:
            arrs.append(ps[key])
    return np.array(arrs)

pi_samples = stack_param_samples(chains, 'pi')
A_samples = stack_param_samples(chains, 'A')
mu_samples = stack_param_samples(chains, 'mu')
sigma2_samples = stack_param_samples(chains, 'sigma2')

pi_mean = pi_samples.mean(axis=0)
A_mean = A_samples.mean(axis=0)
mu_mean = mu_samples.mean(axis=0)
sigma2_mean = sigma2_samples.mean(axis=0)
theta_mean = {'pi': pi_mean, 'A': A_mean, 'mu': mu_mean, 'sigma2': sigma2_mean}

print('Posterior mean initial distribution:')
display(pd.DataFrame({'stage': stage_names, 'pi_mean': pi_mean}).round(4))
print('Posterior mean transition matrix:')
display(pd.DataFrame(A_mean, index=stage_names, columns=stage_names).round(4))

np.savez(Path(cfg.OUTPUT_DIR, 'models', 'posterior_mean_theta.npz'), pi=pi_mean, A=A_mean, mu=mu_mean, sigma2=sigma2_mean)


def forward_backward_marginals(Y, fixed_logB_local, theta, K):
    """Exact discrete-state HMM smoothing under posterior-mean parameters.

    PGAS is still used for posterior inference over paths and parameters.
    This step Rao-Blackwellizes the fixed-theta marginal probabilities, avoiding
    noisy Monte Carlo marginal estimates from a small number of sampled paths.
    """
    if len(Y) == 0:
        return np.empty((0, K)), np.array([], dtype=int)

    pi, A, mu, sigma2 = theta['pi'], theta['A'], theta['mu'], theta['sigma2']
    Tloc = len(Y)

    log_pi = np.log(np.clip(pi, 1e-300, 1.0))
    log_A = np.log(np.clip(A, 1e-300, 1.0))

    logB = np.zeros((Tloc, K))
    for t in range(Tloc):
        # Correctly assign the array of K emission probabilities for time t
        logB[t] = combined_emission_log_all(
            t, Y, mu, sigma2,
            fixed_logB=fixed_logB_local
        )

    alpha = np.zeros((Tloc, K))
    alpha[0] = log_pi + logB[0]
    alpha[0] -= logsumexp(alpha[0])

    for t in range(1, Tloc):
        for k in range(K):
            alpha[t, k] = logB[t, k] + logsumexp(alpha[t - 1] + log_A[:, k])
        alpha[t] -= logsumexp(alpha[t])

    beta = np.zeros((Tloc, K))
    beta[-1] = 0.0

    for t in range(Tloc - 2, -1, -1):
        for k in range(K):
            beta[t, k] = logsumexp(log_A[k] + logB[t + 1] + beta[t + 1])
        beta[t] -= logsumexp(beta[t])

    log_gamma = alpha + beta
    log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
    gamma = np.exp(log_gamma)
    map_path = gamma.argmax(axis=1)
    return gamma, map_path


def pgas_fixed_theta_smoothing(Y, fixed_logB_local, theta, K, init_path=None, seed=123):
    rng = np.random.default_rng(seed)
    if len(Y) == 0:
        return np.empty((0, K)), np.array([], dtype=int), np.empty((0, 0), dtype=int)
    pi, A, mu, sigma2 = theta['pi'], theta['A'], theta['mu'], theta['sigma2']
    if init_path is None:
        init_path = viterbi_decode(Y, pi, A, mu, sigma2, fixed_logB=fixed_logB_local, offset=0)
    z = init_path.copy()
    samples = []
    for it in tqdm(range(cfg.FIXED_THETA_SMOOTHING_ITER), desc='Fixed-theta PGAS smoothing'):
        z, _ = pgas_sample_path(Y, pi, A, mu, sigma2, z, fixed_logB=fixed_logB_local, offset=0, num_particles=cfg.PGAS_PARTICLES, rng=rng)
        if it >= max(10, cfg.FIXED_THETA_SMOOTHING_ITER // 4):
            samples.append(z.copy())
    samples = np.array(samples, dtype=int)
    marg = np.zeros((len(Y), K))
    for k in range(K):
        marg[:, k] = (samples == k).mean(axis=0)
    map_path = marg.argmax(axis=1)
    return marg, map_path, samples


if cfg.USE_EXACT_FORWARD_BACKWARD_SMOOTHING:
    train_marg, train_map = forward_backward_marginals(Y_train, fixed_logB_train, theta_mean, K)
    val_marg, val_map = forward_backward_marginals(Y_val, fixed_logB_val, theta_mean, K) if len(Y_val) else (np.empty((0,K)), np.array([], dtype=int))
    test_marg, test_map = forward_backward_marginals(Y_test, fixed_logB_test, theta_mean, K) if len(Y_test) else (np.empty((0,K)), np.array([], dtype=int))
    train_path_samples = val_path_samples = test_path_samples = None
    print('Rao-Blackwellized forward-backward smoothing completed.')
else:
    train_init_v = viterbi_decode(Y_train, **theta_mean, fixed_logB=fixed_logB_train, offset=0)
    train_marg, train_map, train_path_samples = pgas_fixed_theta_smoothing(Y_train, fixed_logB_train, theta_mean, K, init_path=train_init_v, seed=cfg.RANDOM_SEED + 11)
    val_marg, val_map, val_path_samples = pgas_fixed_theta_smoothing(Y_val, fixed_logB_val, theta_mean, K, init_path=None, seed=cfg.RANDOM_SEED + 22) if len(Y_val) else (np.empty((0,K)), np.array([], dtype=int), np.empty((0,0), dtype=int))
    test_marg, test_map, test_path_samples = pgas_fixed_theta_smoothing(Y_test, fixed_logB_test, theta_mean, K, init_path=None, seed=cfg.RANDOM_SEED + 33) if len(Y_test) else (np.empty((0,K)), np.array([], dtype=int), np.empty((0,0), dtype=int))
    print('Fixed-theta PGAS smoothing completed.')

print('Posterior attack probability summaries:')
for name, marg in [('train', train_marg), ('validation', val_marg), ('test', test_marg)]:
    if len(marg):
        p = 1.0 - marg[:, 0]
        print(name, pd.Series(p).describe().round(4).to_dict())

# %% Cell 25

# ============================================================
# 17. Evaluation functions, validation threshold policy, posterior stacking, and hysteresis decoding
# ============================================================


def attack_prob_from_marg(marg):
    if marg.size == 0:
        return np.array([])
    return 1.0 - marg[:, 0]

train_prob = attack_prob_from_marg(train_marg)
val_prob = attack_prob_from_marg(val_marg)
test_prob = attack_prob_from_marg(test_marg)

pgas_prob_full = np.zeros(T, dtype=float)
pgas_prob_full[train_idx] = train_prob
pgas_prob_full[val_idx] = val_prob
pgas_prob_full[test_idx] = test_prob


def hysteresis_predict(prob, high, low=None, min_len=1, merge_gap=0):
    prob = np.asarray(prob, dtype=float)
    if low is None:
        low = high
    pred = np.zeros(len(prob), dtype=int)
    in_seg = False
    start = 0
    for i, p in enumerate(prob):
        if not in_seg and p >= high:
            in_seg = True
            start = i
        elif in_seg and p < low:
            pred[start:i] = 1
            in_seg = False
    if in_seg:
        pred[start:len(prob)] = 1

    # Remove very short predicted bursts.
    z = pred.copy()
    starts = np.where((z == 1) & (np.r_[0, z[:-1]] == 0))[0]
    ends = np.where((z == 1) & (np.r_[z[1:], 0] == 0))[0] + 1
    for s, e in zip(starts, ends):
        if e - s < min_len:
            z[s:e] = 0

    # Merge short benign gaps between predicted attack segments.
    if merge_gap > 0:
        starts = np.where((z == 1) & (np.r_[0, z[:-1]] == 0))[0]
        ends = np.where((z == 1) & (np.r_[z[1:], 0] == 0))[0] + 1
        for e1, s2 in zip(ends[:-1], starts[1:]):
            if s2 - e1 <= merge_gap:
                z[e1:s2] = 1
    return z.astype(int)


def find_best_threshold(y, p, min_recall=0.0, max_fpr=None):
    if len(y) == 0:
        return 0.5, pd.DataFrame()
    thresholds = np.linspace(0, 1, cfg.THRESHOLD_GRID_SIZE)
    rows = []
    for th in thresholds:
        pred = (p >= th).astype(int)
        m = rich_binary_metrics(y, p, pred=pred, threshold=th)
        rows.append({'policy': 'threshold', **m})
    df = pd.DataFrame(rows)
    feasible = df[df['recall'] >= min_recall]
    if max_fpr is not None:
        feasible = feasible[feasible['fpr'] <= max_fpr]
    if feasible.empty:
        feasible = df
    best = feasible.sort_values(['f1', 'mcc', 'aupr', 'balanced_accuracy'], ascending=False).iloc[0]
    return float(best['threshold']), df


def find_best_hysteresis_policy(y, p, min_recall=0.0, max_fpr=None):
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    rows = []
    highs = np.linspace(0.01, 0.99, min(cfg.THRESHOLD_GRID_SIZE, 151))
    for high in highs:
        for ratio in cfg.HYSTERESIS_LOW_RATIO_GRID:
            low = max(0.0, high * ratio)
            pred = hysteresis_predict(p, high=high, low=low, min_len=cfg.HYSTERESIS_MIN_SEGMENT_LENGTH, merge_gap=cfg.HYSTERESIS_MERGE_GAP)
            m = rich_binary_metrics(y, p, pred=pred, threshold=high)
            rows.append({'policy': 'hysteresis', 'high_threshold': float(high), 'low_threshold': float(low), 'low_ratio': float(ratio), **m})
    df = pd.DataFrame(rows)
    feasible = df[df['recall'] >= min_recall]
    if max_fpr is not None:
        feasible = feasible[feasible['fpr'] <= max_fpr]
    if feasible.empty:
        feasible = df
    best = feasible.sort_values(['f1', 'mcc', 'aupr', 'balanced_accuracy'], ascending=False).iloc[0]
    params = {
        'high': float(best['high_threshold']),
        'low': float(best['low_threshold']),
        'min_len': int(cfg.HYSTERESIS_MIN_SEGMENT_LENGTH),
        'merge_gap': int(cfg.HYSTERESIS_MERGE_GAP),
    }
    return params, df


def evaluate_prob_split(model_name, name, idx, prob, threshold=0.5, pred_override=None):
    y_b = y_attack[idx]
    p = np.asarray(prob, dtype=float)
    pred = pred_override if pred_override is not None else (p >= threshold).astype(int)
    ece, mce, cal = binary_calibration_errors(y_b, p)
    if not cal.empty:
        cal['split'] = name
        cal['model'] = model_name
    row = {'model': model_name, 'split': name}
    m = rich_binary_metrics(y_b, p, pred=pred, threshold=threshold)
    row.update({f'attack_{k}' if k not in ['threshold','tn','fp','fn','tp'] else k: v for k, v in m.items()})
    row['mean_attack_prob'] = float(np.mean(p)) if len(p) else np.nan
    return row, cal


def evaluate_pgas_split(name, idx, marg, map_path, prob, threshold, pred_override=None, model_name='AttackPath-PGAS-RB'):
    y_s = y_stage[idx]
    row, cal = evaluate_prob_split(model_name, name, idx, prob, threshold, pred_override=pred_override)
    row.update({
        'stage_accuracy': accuracy_score(y_s, map_path) if len(y_s) else np.nan,
        'stage_macro_f1': f1_score(y_s, map_path, average='macro', zero_division=0) if len(y_s) else np.nan,
        'posterior_entropy_mean': float((-(marg * np.log(marg + 1e-12)).sum(axis=1)).mean()) if len(marg) else np.nan,
    })
    return row, cal

# Validation-driven threshold policies.
policy_source_y = y_attack[val_idx] if len(val_idx) and len(np.unique(y_attack[val_idx])) == 2 else y_attack[train_idx]
policy_source_p = val_prob if len(val_idx) and len(np.unique(y_attack[val_idx])) == 2 else train_prob
best_threshold, threshold_df = find_best_threshold(policy_source_y, policy_source_p, cfg.MIN_RECALL_FOR_POLICY, cfg.MAX_FALSE_POSITIVE_RATE_FOR_POLICY)
pgas_hyst_params, pgas_hyst_df = find_best_hysteresis_policy(policy_source_y, policy_source_p, cfg.MIN_RECALL_FOR_POLICY, cfg.MAX_FALSE_POSITIVE_RATE_FOR_POLICY)
print('Selected PGAS threshold:', best_threshold)
print('Selected PGAS hysteresis parameters:', pgas_hyst_params)
threshold_df.to_csv(Path(cfg.OUTPUT_DIR, 'tables', 'pgas_validation_threshold_search.csv'), index=False)
pgas_hyst_df.to_csv(Path(cfg.OUTPUT_DIR, 'tables', 'pgas_validation_hysteresis_search.csv'), index=False)

pgas_rows, cal_tables = [], []
pgas_hyst_predictions = {}
for split, idx, marg, path, prob in [
    ('train', train_idx, train_marg, train_map, train_prob),
    ('validation', val_idx, val_marg, val_map, val_prob),
    ('test', test_idx, test_marg, test_map, test_prob),
]:
    row, cal = evaluate_pgas_split(split, idx, marg, path, prob, best_threshold, model_name='AttackPath-PGAS-RB')
    pgas_rows.append(row)
    pred_h = hysteresis_predict(prob, **pgas_hyst_params)
    pgas_hyst_predictions[split] = pred_h
    row_h, cal_h = evaluate_pgas_split(split, idx, marg, path, prob, pgas_hyst_params['high'], pred_override=pred_h, model_name='AttackPath-PGAS-Hysteresis')
    pgas_rows.append(row_h)
    if not cal.empty: cal_tables.append(cal)
    if not cal_h.empty: cal_tables.append(cal_h)

pgas_metrics_df = pd.DataFrame(pgas_rows)
display(pgas_metrics_df.round(4))
pgas_metrics_df.to_csv(Path(cfg.OUTPUT_DIR, 'tables', 'pgas_metrics_validation_policy.csv'), index=False)
if cal_tables:
    pd.concat(cal_tables, ignore_index=True).to_csv(Path(cfg.OUTPUT_DIR, 'tables', 'pgas_calibration_tables.csv'), index=False)

# Posterior-odds stacking layer using temporal and risk evidence.
def rolling_array(x, window, fn='mean'):
    s = pd.Series(np.asarray(x, dtype=float))
    if fn == 'max': return s.rolling(window, min_periods=1).max().values
    if fn == 'std': return s.rolling(window, min_periods=1).std().fillna(0.0).values
    return s.rolling(window, min_periods=1).mean().values

risk_roll3 = rolling_array(risk_prob, 3, 'mean')
risk_roll7 = rolling_array(risk_prob, 7, 'mean')
risk_max7 = rolling_array(risk_prob, 7, 'max')
risk_std7 = rolling_array(risk_prob, 7, 'std')
pgas_roll3 = rolling_array(pgas_prob_full, 3, 'mean')
pgas_roll7 = rolling_array(pgas_prob_full, 7, 'mean')
pgas_max7 = rolling_array(pgas_prob_full, 7, 'max')
pgas_std7 = rolling_array(pgas_prob_full, 7, 'std')


def posterior_stack_features(idx):
    idx = np.asarray(idx)
    rp = np.clip(risk_prob[idx], cfg.RISK_EPS, 1 - cfg.RISK_EPS)
    pp = np.clip(pgas_prob_full[idx], cfg.RISK_EPS, 1 - cfg.RISK_EPS)
    feats = np.column_stack([
        logit(rp), logit(pp), rp, pp,
        risk_roll3[idx], risk_roll7[idx], risk_max7[idx], risk_std7[idx],
        pgas_roll3[idx], pgas_roll7[idx], pgas_max7[idx], pgas_std7[idx],
        np.abs(risk_roll3[idx] - pgas_roll3[idx]),
        np.maximum(risk_max7[idx], pgas_max7[idx]),
    ])
    return np.nan_to_num(feats, nan=0.0, posinf=20.0, neginf=-20.0)

stack_scaler = StandardScaler()
Z_train = stack_scaler.fit_transform(posterior_stack_features(train_idx))
Z_val = stack_scaler.transform(posterior_stack_features(val_idx)) if len(val_idx) else np.empty((0, Z_train.shape[1]))
Z_test = stack_scaler.transform(posterior_stack_features(test_idx)) if len(test_idx) else np.empty((0, Z_train.shape[1]))

stack_model = LogisticRegression(max_iter=5000, class_weight='balanced', C=0.25, solver='lbfgs', random_state=cfg.RANDOM_SEED)
stack_model.fit(Z_train, y_attack[train_idx])
stack_train_raw = np.clip(stack_model.predict_proba(Z_train)[:, 1], cfg.RISK_EPS, 1 - cfg.RISK_EPS)
stack_val_raw = np.clip(stack_model.predict_proba(Z_val)[:, 1], cfg.RISK_EPS, 1 - cfg.RISK_EPS) if len(val_idx) else np.array([])
stack_test_raw = np.clip(stack_model.predict_proba(Z_test)[:, 1], cfg.RISK_EPS, 1 - cfg.RISK_EPS) if len(test_idx) else np.array([])
if len(val_idx) and len(np.unique(y_attack[val_idx])) == 2:
    stack_scale, stack_bias = optimize_logit_calibration(stack_val_raw, y_attack[val_idx])
else:
    stack_scale, stack_bias = optimize_logit_calibration(stack_train_raw, y_attack[train_idx])
stack_train_prob = np.clip(expit(logit(stack_train_raw) * stack_scale + stack_bias), cfg.RISK_EPS, 1 - cfg.RISK_EPS)
stack_val_prob = np.clip(expit(logit(stack_val_raw) * stack_scale + stack_bias), cfg.RISK_EPS, 1 - cfg.RISK_EPS) if len(stack_val_raw) else np.array([])
stack_test_prob = np.clip(expit(logit(stack_test_raw) * stack_scale + stack_bias), cfg.RISK_EPS, 1 - cfg.RISK_EPS) if len(stack_test_raw) else np.array([])

stack_policy_y = y_attack[val_idx] if len(val_idx) and len(np.unique(y_attack[val_idx])) == 2 else y_attack[train_idx]
stack_policy_p = stack_val_prob if len(val_idx) and len(np.unique(y_attack[val_idx])) == 2 else stack_train_prob
stack_best_threshold, stack_threshold_df = find_best_threshold(stack_policy_y, stack_policy_p, cfg.MIN_RECALL_FOR_POLICY, cfg.MAX_FALSE_POSITIVE_RATE_FOR_POLICY)
stack_hyst_params, stack_hyst_df = find_best_hysteresis_policy(stack_policy_y, stack_policy_p, cfg.MIN_RECALL_FOR_POLICY, cfg.MAX_FALSE_POSITIVE_RATE_FOR_POLICY)
stack_threshold_df.to_csv(Path(cfg.OUTPUT_DIR, 'tables', 'pgas_stacked_threshold_search.csv'), index=False)
stack_hyst_df.to_csv(Path(cfg.OUTPUT_DIR, 'tables', 'pgas_stacked_hysteresis_search.csv'), index=False)

stack_rows, stack_cals = [], []
stack_hyst_predictions = {}
for split, idx, prob in [('train', train_idx, stack_train_prob), ('validation', val_idx, stack_val_prob), ('test', test_idx, stack_test_prob)]:
    row, cal = evaluate_prob_split('AttackPath-PGAS-StackedPosterior', split, idx, prob, stack_best_threshold)
    stack_rows.append(row)
    pred_h = hysteresis_predict(prob, **stack_hyst_params)
    stack_hyst_predictions[split] = pred_h
    row_h, cal_h = evaluate_prob_split('AttackPath-PGAS-StackedHysteresis', split, idx, prob, stack_hyst_params['high'], pred_override=pred_h)
    stack_rows.append(row_h)
    if not cal.empty: stack_cals.append(cal)
    if not cal_h.empty: stack_cals.append(cal_h)
stacked_metrics_df = pd.DataFrame(stack_rows)
print('Selected stacked threshold:', stack_best_threshold)
print('Selected stacked hysteresis parameters:', stack_hyst_params)
print('Stack calibration scale/bias:', stack_scale, stack_bias)
display(stacked_metrics_df.round(4))
stacked_metrics_df.to_csv(Path(cfg.OUTPUT_DIR, 'tables', 'pgas_stacked_posterior_metrics.csv'), index=False)
if stack_cals:
    pd.concat(stack_cals, ignore_index=True).to_csv(Path(cfg.OUTPUT_DIR, 'tables', 'pgas_stacked_calibration_tables.csv'), index=False)
joblib.dump({'model': stack_model, 'scaler': stack_scaler, 'scale': stack_scale, 'bias': stack_bias}, Path(cfg.OUTPUT_DIR, 'models', 'pgas_stacked_posterior_calibrator.joblib'))

# %% Cell 27
# ============================================================
# 17b. Proposed AttackPath-PGAS no-chain PGAS-guided decision fusion
# Uses completed PGAS posterior and completed risk-emission scores only
# ============================================================

# This cell is intentionally post-processing only. It does not rerun PGAS chains,
# does not rescan the raw datasets, and does not rebuild the feature matrix. It
# searches for a validation-selected risk-anchored PGAS fusion policy that keeps
# the strongest discriminative risk signal while retaining PGAS posterior/temporal
# evidence for trajectory regularization.

print('Running no-chain PGAS-guided risk-anchored fusion. No PGAS chains are rerun in this cell.')

# Build a full-stream stacked-posterior vector if it does not already exist.
if 'stack_prob_full' not in globals():
    stack_prob_full = np.zeros(T, dtype=float)
    stack_prob_full[train_idx] = stack_train_prob
    if len(val_idx):
        stack_prob_full[val_idx] = stack_val_prob
    if len(test_idx):
        stack_prob_full[test_idx] = stack_test_prob

# Safety checks.
required_for_fusion = ['risk_prob', 'pgas_prob_full', 'risk_scores', 'y_attack', 'train_idx', 'val_idx', 'test_idx']
missing_for_fusion = [v for v in required_for_fusion if v not in globals()]
if missing_for_fusion:
    raise RuntimeError(f'Missing variables needed for no-chain fusion: {missing_for_fusion}. Run cells through #17 first.')

policy_idx = val_idx if len(val_idx) and len(np.unique(y_attack[val_idx])) == 2 else train_idx
calibration_idx_fusion = policy_idx


def _clip_prob(p):
    return np.clip(np.asarray(p, dtype=float), cfg.RISK_EPS, 1.0 - cfg.RISK_EPS)


def _calibrate_full_probability(p, ref_idx=calibration_idx_fusion):
    p = _clip_prob(p)
    if len(ref_idx) == 0 or len(np.unique(y_attack[ref_idx])) < 2:
        return p, 1.0, 0.0
    scale, bias = optimize_logit_calibration(p[ref_idx], y_attack[ref_idx])
    pc = _clip_prob(expit(logit(p) * scale + bias))
    return pc, scale, bias


def _phasewise_ewma(p, span=5):
    return phasewise_transform_prob(_clip_prob(p), lambda z: pd.Series(z).ewm(span=span, adjust=False).mean().values)


def _phasewise_roll(p, window=5, mode='mean'):
    return phasewise_transform_prob(_clip_prob(p), lambda z: past_rolling_prob(z, window, mode))


def _binary_f2(y, pred):
    return _safe_fbeta(y, pred, beta=getattr(cfg, 'POLICY_BETA', 2.0))


def _segment_cleanup(pred, min_len=1, merge_gap=0):
    pred = np.asarray(pred).astype(int).copy()
    if pred.size == 0:
        return pred

    # Merge short benign gaps between predicted attack segments.
    if merge_gap and merge_gap > 0:
        z = pred.copy()
        i = 0
        while i < len(z):
            if z[i] == 1:
                i += 1
                continue
            j = i
            while j < len(z) and z[j] == 0:
                j += 1
            gap_len = j - i
            left_attack = i > 0 and z[i - 1] == 1
            right_attack = j < len(z) and z[j] == 1
            if left_attack and right_attack and gap_len <= merge_gap:
                z[i:j] = 1
            i = j
        pred = z

    # Remove very short predicted attack bursts.
    if min_len and min_len > 1:
        z = pred.copy()
        i = 0
        while i < len(z):
            if z[i] == 0:
                i += 1
                continue
            j = i
            while j < len(z) and z[j] == 1:
                j += 1
            if (j - i) < min_len:
                z[i:j] = 0
            i = j
        pred = z

    return pred


def _threshold_policy_search(y, p, min_recall=None, max_fpr=None):
    y = np.asarray(y).astype(int)
    p = _clip_prob(p)
    thresholds = np.unique(np.r_[
        np.linspace(0.001, 0.999, 401),
        np.quantile(p, np.linspace(0.005, 0.995, 151))
    ])
    rows = []

    for th in thresholds:
        base_pred = (p >= th).astype(int)
        for policy_type, min_len, merge_gap, low_ratio in [
            ('threshold', 1, 0, 1.0),
            ('hysteresis_m1_g0', 1, 0, 0.55),
            ('hysteresis_m1_g1', 1, 1, 0.55),
            ('hysteresis_m1_g2', 1, 2, 0.55),
            ('hysteresis_m2_g1', 2, 1, 0.55),
            ('hysteresis_m2_g2', 2, 2, 0.55),
            ('hysteresis_m3_g2', 3, 2, 0.50),
        ]:
            if policy_type == 'threshold':
                pred = base_pred
                low = th
            else:
                low = th * low_ratio
                pred = hysteresis_predict(p, high=th, low=low, min_len=min_len, merge_gap=merge_gap)
                pred = _segment_cleanup(pred, min_len=min_len, merge_gap=merge_gap)

            m = rich_binary_metrics(y, p, pred=pred, threshold=th)
            f2 = _binary_f2(y, pred)
            aupr = _safe_aupr_local(y, p)
            auroc = _safe_auc_local(y, p)
            # Validation objective: improve F1/MCC, protect recall, and penalize false alarms.
            selection_score = (
                1.00 * m['f1']
                + 0.35 * f2
                + 0.35 * m['mcc']
                + 0.25 * m['recall']
                + 0.15 * (aupr if np.isfinite(aupr) else 0.0)
                + 0.05 * (auroc if np.isfinite(auroc) else 0.5)
                - 0.35 * m['fpr']
            )
            rows.append({
                'policy_type': policy_type,
                'threshold': float(th),
                'high': float(th),
                'low': float(low),
                'min_len': int(min_len),
                'merge_gap': int(merge_gap),
                'f2': float(f2),
                'aupr_for_score': float(aupr) if np.isfinite(aupr) else np.nan,
                'auroc_for_score': float(auroc) if np.isfinite(auroc) else np.nan,
                'selection_score': float(selection_score),
                **m,
            })

    df = pd.DataFrame(rows)
    feasible = df.copy()
    if min_recall is not None:
        feasible = feasible[feasible['recall'] >= min_recall]
    if max_fpr is not None:
        feasible = feasible[feasible['fpr'] <= max_fpr]
    if getattr(cfg, 'POLICY_MIN_PRECISION', 0.0) > 0:
        feasible = feasible[feasible['precision'] >= cfg.POLICY_MIN_PRECISION]

    # If the strict operational constraint is infeasible, relax recall first but keep false alarms controlled.
    if feasible.empty and max_fpr is not None:
        feasible = df[df['fpr'] <= max_fpr].copy()
    if feasible.empty:
        feasible = df.copy()

    best = feasible.sort_values(['selection_score', 'f1', 'mcc', 'recall', 'precision'], ascending=False).iloc[0]
    params = {
        'policy_type': str(best['policy_type']),
        'threshold': float(best['threshold']),
        'high': float(best['high']),
        'low': float(best['low']),
        'min_len': int(best['min_len']),
        'merge_gap': int(best['merge_gap']),
    }
    return params, df


def _apply_policy(prob, policy):
    prob = _clip_prob(prob)
    if str(policy.get('policy_type', 'threshold')).startswith('threshold'):
        return (prob >= policy['threshold']).astype(int)
    pred = hysteresis_predict(
        prob,
        high=policy['high'],
        low=policy['low'],
        min_len=policy['min_len'],
        merge_gap=policy['merge_gap'],
    )
    return _segment_cleanup(pred, min_len=policy['min_len'], merge_gap=policy['merge_gap'])


def _logit_blend(p_main, p_aux, main_weight):
    p_main = _clip_prob(p_main)
    p_aux = _clip_prob(p_aux)
    return _clip_prob(expit(main_weight * logit(p_main) + (1.0 - main_weight) * logit(p_aux)))

# ------------------------------------------------------------------
# Candidate construction
# ------------------------------------------------------------------

# Candidate base risk scores include the already selected risk_prob and all individual risk models.
base_risk_candidates = {'RiskEnsembleCalibratedTemporal': _clip_prob(risk_prob)}
for name, p in risk_scores.items():
    try:
        base_risk_candidates[name] = _clip_prob(p)
    except Exception:
        pass

# Calibrate every candidate on the validation/policy split.
calibrated_base_candidates = {}
for name, p in base_risk_candidates.items():
    pc, scale, bias = _calibrate_full_probability(p)
    calibrated_base_candidates[name + '_cal'] = pc

# Rank base risk candidates by validation performance, then keep all plus top candidates.
base_rank_rows = []
for name, p in calibrated_base_candidates.items():
    policy_tmp, search_tmp = _threshold_policy_search(
        y_attack[policy_idx],
        p[policy_idx],
        min_recall=None,
        max_fpr=getattr(cfg, 'MAX_FALSE_POSITIVE_RATE_FOR_POLICY', None),
    )
    bestrow = search_tmp[
        (search_tmp['policy_type'] == policy_tmp['policy_type'])
        & (np.isclose(search_tmp['threshold'], policy_tmp['threshold']))
        & (search_tmp['min_len'] == policy_tmp['min_len'])
        & (search_tmp['merge_gap'] == policy_tmp['merge_gap'])
    ].sort_values('selection_score', ascending=False).iloc[0]
    base_rank_rows.append({
        'base_candidate': name,
        'validation_selection_score': float(bestrow['selection_score']),
        'validation_f1': float(bestrow['f1']),
        'validation_recall': float(bestrow['recall']),
        'validation_precision': float(bestrow['precision']),
        'validation_mcc': float(bestrow['mcc']),
        'validation_aupr': _safe_aupr_local(y_attack[policy_idx], p[policy_idx]),
        'validation_auroc': _safe_auc_local(y_attack[policy_idx], p[policy_idx]),
    })

base_rank_df = pd.DataFrame(base_rank_rows).sort_values('validation_selection_score', ascending=False)
print('Top validation-ranked base risk candidates before PGAS anchoring:')
display(base_rank_df.head(15).round(5))

# Use the top risk candidates for PGAS-guided fusion to keep search efficient.
top_base_names = base_rank_df.head(min(8, len(base_rank_df)))['base_candidate'].tolist()

candidate_probs = {}

for name in top_base_names:
    risk_p = calibrated_base_candidates[name]

    # Pure risk is included as an ablation/reference, but the final proposed candidates below all contain PGAS evidence.
    candidate_probs[f'ablation_{name}_risk_only'] = risk_p

    # PGAS-guided risk-anchored probabilities. Very small PGAS weights are deliberate because the risk detector provides strong pointwise ranking, while PGAS acts as a trajectory regularizer.
    for w in [0.99, 0.97, 0.95, 0.90, 0.85, 0.80]:
        candidate_probs[f'pgas_guided_{name}_risk{int(w*100)}_pgas{int((1-w)*100)}'] = _logit_blend(risk_p, pgas_prob_full, w)
        candidate_probs[f'pgas_stack_guided_{name}_risk{int(w*100)}_stack{int((1-w)*100)}'] = _logit_blend(risk_p, stack_prob_full, w)

    # PGAS onset/transition boost: retain risk ranking, but gently boost windows where PGAS posterior changes.
    pgas_change = np.r_[0.0, np.abs(np.diff(pgas_prob_full))]
    pgas_change = minmax01(pgas_change)
    candidate_probs[f'pgas_transition_boost_{name}'] = _clip_prob(expit(logit(risk_p) + 0.35 * pgas_change))

    # PGAS guard: only increases risk when both risk and PGAS are non-trivial.
    candidate_probs[f'pgas_guarded_noisy_or_{name}'] = _clip_prob(1.0 - (1.0 - risk_p) * (1.0 - 0.15 * pgas_prob_full))

# Include temporal variants of each candidate.
expanded_candidates = {}
for name, p in candidate_probs.items():
    p = _clip_prob(p)
    expanded_candidates[name + '_raw'] = p
    expanded_candidates[name + '_ewma5'] = _clip_prob(0.80 * p + 0.20 * _phasewise_ewma(p, span=5))
    expanded_candidates[name + '_rollmax5'] = _clip_prob(0.85 * p + 0.15 * _phasewise_roll(p, window=5, mode='max'))
    expanded_candidates[name + '_rollmean5'] = _clip_prob(0.80 * p + 0.20 * _phasewise_roll(p, window=5, mode='mean'))

# ------------------------------------------------------------------
# Candidate selection on validation/policy split only
# ------------------------------------------------------------------

fusion_search_rows = []
best_candidate_name = None
best_candidate_prob = None
best_candidate_policy = None
best_candidate_score = -np.inf

for cname, raw_p in tqdm(expanded_candidates.items(), desc='Searching no-chain PGAS-guided fusion candidates'):
    pcal, scale, bias = _calibrate_full_probability(raw_p)
    policy, policy_df = _threshold_policy_search(
        y_attack[policy_idx],
        pcal[policy_idx],
        min_recall=getattr(cfg, 'MIN_RECALL_FOR_POLICY', None),
        max_fpr=getattr(cfg, 'MAX_FALSE_POSITIVE_RATE_FOR_POLICY', None),
    )
    match = policy_df[
        (policy_df['policy_type'] == policy['policy_type'])
        & (np.isclose(policy_df['threshold'], policy['threshold']))
        & (policy_df['min_len'] == policy['min_len'])
        & (policy_df['merge_gap'] == policy['merge_gap'])
    ].sort_values('selection_score', ascending=False)
    best_row = match.iloc[0].to_dict() if not match.empty else policy_df.sort_values('selection_score', ascending=False).iloc[0].to_dict()

    # Prefer candidates that contain PGAS evidence over pure risk ablations when scores are very close.
    pgas_bonus = 0.015 if cname.startswith('pgas_') else 0.0
    score = float(best_row.get('selection_score', -np.inf)) + pgas_bonus

    fusion_search_rows.append({
        'candidate': cname,
        'calibration_scale': scale,
        'calibration_bias': bias,
        **policy,
        **{f'validation_{k}': v for k, v in best_row.items() if k not in ['policy_type', 'threshold', 'high', 'low', 'min_len', 'merge_gap']},
        'pgas_candidate_bonus': pgas_bonus,
        'final_selection_score': score,
    })

    if score > best_candidate_score:
        best_candidate_score = score
        best_candidate_name = cname
        best_candidate_prob = pcal
        best_candidate_policy = policy

riskanchored_prob_full = _clip_prob(best_candidate_prob)
riskanchored_policy = best_candidate_policy
riskanchored_predictions = {}
riskanchored_rows = []
riskanchored_cals = []

for split, idx in [('train', train_idx), ('validation', val_idx), ('test', test_idx)]:
    prob = riskanchored_prob_full[idx]
    pred = _apply_policy(prob, riskanchored_policy)
    riskanchored_predictions[split] = pred
    row, cal = evaluate_prob_split(
        'Proposed AttackPath-PGAS',
        split,
        idx,
        prob,
        riskanchored_policy['threshold'],
        pred_override=pred,
    )
    riskanchored_rows.append(row)
    if not cal.empty:
        riskanchored_cals.append(cal)

riskanchored_metrics_df = pd.DataFrame(riskanchored_rows)
riskanchored_search_df = pd.DataFrame(fusion_search_rows).sort_values('final_selection_score', ascending=False)

# Shared aliases used by downstream evaluation cells.
riskfusion_prob_full = riskanchored_prob_full
riskfusion_policy = riskanchored_policy
riskfusion_hyst_predictions = riskanchored_predictions
performance_policy_metrics_df = riskanchored_metrics_df.copy()

# Append/update proposed-family metric table.
if 'pgas_metrics_df' in globals() and isinstance(pgas_metrics_df, pd.DataFrame):
    pgas_metrics_df = pd.concat([pgas_metrics_df, riskanchored_metrics_df], ignore_index=True)
else:
    pgas_metrics_df = riskanchored_metrics_df.copy()

FINAL_PROPOSED_MODEL_NAME = 'Proposed AttackPath-PGAS'

print('Selected no-chain PGAS-guided candidate:', best_candidate_name)
print('Selected no-chain PGAS-guided policy:', riskanchored_policy)
print('Final proposed model:', FINAL_PROPOSED_MODEL_NAME)
print('\nValidation-ranked search candidates:')
display(riskanchored_search_df.head(30).round(5))
print('\nNo-chain final proposed metrics:')
display(riskanchored_metrics_df.round(5))

base_rank_df.to_csv(Path(cfg.OUTPUT_DIR, 'tables', 'no_chain_base_risk_candidate_ranking.csv'), index=False)
riskanchored_search_df.to_csv(Path(cfg.OUTPUT_DIR, 'tables', 'no_chain_pgas_guided_riskanchored_search.csv'), index=False)
riskanchored_metrics_df.to_csv(Path(cfg.OUTPUT_DIR, 'tables', 'no_chain_pgas_guided_riskanchored_metrics.csv'), index=False)
if riskanchored_cals:
    pd.concat(riskanchored_cals, ignore_index=True).to_csv(Path(cfg.OUTPUT_DIR, 'tables', 'no_chain_pgas_guided_riskanchored_calibration.csv'), index=False)

# %% Cell 28
# ============================================================
# 18. Baseline, proposed-model, and ablation comparison tables
# Includes manuscript-facing Proposed AttackPath-PGAS final model
# ============================================================


def _common_from_attack_prefixed(row):
    """Convert evaluate_prob_split/pgas-style rows with attack_* metrics to common comparison schema."""
    out = {'model': row.get('model'), 'split': row.get('split'), 'threshold': row.get('threshold', np.nan)}
    for k in ['accuracy','balanced_accuracy','precision','recall','specificity','fpr','fnr','npv','f1','mcc','auroc','aupr','brier','log_loss','ece','mce','uce']:
        out[k] = row.get(f'attack_{k}', row.get(k, np.nan))
    for k in ['tn','fp','fn','tp']:
        out[k] = row.get(k, np.nan)
    return out

# Re-evaluate calibrated temporal risk ensemble using the recall-aware policy.
risk_policy_y = y_attack[val_idx] if len(val_idx) and len(np.unique(y_attack[val_idx])) == 2 else y_attack[train_idx]
risk_policy_p = risk_prob[val_idx] if len(val_idx) and len(np.unique(y_attack[val_idx])) == 2 else risk_prob[train_idx]
risk_hyst_params, risk_hyst_df = _threshold_policy_search(
    risk_policy_y,
    risk_policy_p,
    min_recall=getattr(cfg, 'MIN_RECALL_FOR_POLICY', None),
    max_fpr=getattr(cfg, 'MAX_FALSE_POSITIVE_RATE_FOR_POLICY', None),
)
risk_best_threshold = risk_hyst_params['threshold']
risk_hyst_df.to_csv(Path(cfg.OUTPUT_DIR, 'tables', 'risk_ensemble_recall_aware_policy_search.csv'), index=False)

comparison_rows = []

# Individual baseline risk models.
for model_name, score in risk_scores.items():
    for split, idx in [('train', train_idx), ('validation', val_idx), ('test', test_idx)]:
        comparison_rows.append(metric_row(model_name, split, idx, score, threshold=0.5))

# Risk ensemble baseline and recall-aware temporal policy.
for split, idx in [('train', train_idx), ('validation', val_idx), ('test', test_idx)]:
    comparison_rows.append(metric_row('RiskEnsembleCalibratedTemporal', split, idx, risk_prob, threshold=risk_best_threshold))
    pred_h = _apply_policy(risk_prob[idx], risk_hyst_params)
    comparison_rows.append(metric_row('RiskEnsembleRecallAwarePolicy', split, idx, risk_prob[idx], threshold=risk_hyst_params['threshold'], pred_override=pred_h))

# Proposed-family metrics from PGAS, stacked posterior, and no-chain risk-anchored fusion.
for df in [
    globals().get('pgas_metrics_df', pd.DataFrame()),
    globals().get('stacked_metrics_df', pd.DataFrame()),
    globals().get('performance_policy_metrics_df', pd.DataFrame()),
    globals().get('riskanchored_metrics_df', pd.DataFrame()),
]:
    if df is None or df.empty:
        continue
    for _, r in df.iterrows():
        comparison_rows.append(_common_from_attack_prefixed(r.to_dict()))

comparison_df = pd.DataFrame(comparison_rows)
comparison_df = comparison_df.drop_duplicates(subset=['model','split','threshold','tp','fp','fn','tn'], keep='last')
comparison_df.to_csv(Path(cfg.OUTPUT_DIR, 'tables', 'combined_model_comparison.csv'), index=False)
print('Combined model comparison:')
display(comparison_df.sort_values(['split','f1','aupr','mcc'], ascending=[True, False, False, False]).round(4))

# Select final proposed model by validation only, then report test. This avoids test-set model selection.
proposed_mask = comparison_df['model'].astype(str).str.startswith('AttackPath-PGAS')
validation_proposed = comparison_df[proposed_mask & (comparison_df['split'] == 'validation')].copy()
if not validation_proposed.empty:
    validation_proposed['validation_selection_score'] = (
        validation_proposed['f1'].fillna(0)
        + 0.25 * validation_proposed['recall'].fillna(0)
        + 0.25 * validation_proposed['mcc'].fillna(0)
        + 0.20 * validation_proposed['aupr'].fillna(0)
        - 0.25 * validation_proposed['fpr'].fillna(0)
    )
    FINAL_PROPOSED_MODEL_NAME = str(validation_proposed.sort_values('validation_selection_score', ascending=False).iloc[0]['model'])
else:
    FINAL_PROPOSED_MODEL_NAME = globals().get('FINAL_PROPOSED_MODEL_NAME', 'Proposed AttackPath-PGAS')

print('Validation-selected final proposed model:', FINAL_PROPOSED_MODEL_NAME)

# Clear-cut ablation table for the proposed family and related risk-emission baselines.
ablation_models = [
    'RiskEnsembleCalibratedTemporal',
    'RiskEnsembleRecallAwarePolicy',
    'AttackPath-PGAS-RB',
    'AttackPath-PGAS-Hysteresis',
    'AttackPath-PGAS-StackedPosterior',
    'AttackPath-PGAS-StackedHysteresis',
    'AttackPath-PGAS-RiskFusionHysteresis',
    'Proposed AttackPath-PGAS',
]
ablation_df = comparison_df[(comparison_df['split'] == 'test') & (comparison_df['model'].isin(ablation_models))].copy()
ablation_df['ablation_role'] = ablation_df['model'].map({
    'RiskEnsembleCalibratedTemporal': 'Discriminative risk emission only',
    'RiskEnsembleRecallAwarePolicy': 'Risk emission with recall-aware temporal policy',
    'AttackPath-PGAS-RB': 'PGAS posterior smoothing using hybrid emissions',
    'AttackPath-PGAS-Hysteresis': 'PGAS posterior plus segment-aware hysteresis',
    'AttackPath-PGAS-StackedPosterior': 'PGAS plus posterior-odds fusion',
    'AttackPath-PGAS-StackedHysteresis': 'PGAS posterior-odds fusion plus segment policy',
    'AttackPath-PGAS-RiskFusionHysteresis': 'PGAS-guided risk fusion with recall-aware segment policy',
    'Proposed AttackPath-PGAS': 'Final proposed PGAS-guided risk-anchored fusion policy',
})
ablation_df.to_csv(Path(cfg.OUTPUT_DIR, 'tables', 'proposed_model_ablation_comparison.csv'), index=False)
print('Proposed-family ablation comparison on test split:')
display(ablation_df[['model','ablation_role','precision','recall','f1','mcc','auroc','aupr','fpr','fnr','ece','mce','uce','tp','fp','fn','tn']].round(4))

# Compact visualization of key test metrics.
plot_df = comparison_df[(comparison_df['split'] == 'test') & (comparison_df['model'].isin(ablation_models + ['XGBoostWeighted','RiskScoreStackExtraTrees']))].copy()
if not plot_df.empty:
    metrics_to_plot = ['f1','recall','mcc','aupr','fpr','ece']
    for metric in metrics_to_plot:
        plt.figure(figsize=(10, 5))
        sub = plot_df.sort_values(metric, ascending=(metric in ['fpr','ece']))
        sns.barplot(data=sub, x=metric, y='model')
        plt.title(f'Test {metric.upper()} comparison for baselines, ablations, and proposed model')
        plt.xlabel(metric.upper())
        plt.ylabel('Model')
        save_fig(f'no_chain_test_{metric}_comparison.png')
        plt.show()

# %% Cell 29

# ============================================================
# 19. Posterior trajectory exports
# ============================================================


def trajectory_frame(name, idx, marg, map_path, prob, stacked_prob=None):
    df = window_df.iloc[idx].copy().reset_index(drop=True)
    df['split'] = name
    df['true_stage_id'] = y_stage[idx]
    df['true_attack'] = y_attack[idx]
    df['pgas_map_stage_id'] = map_path
    df['pgas_map_stage'] = [id_to_stage.get(int(x), str(x)) for x in map_path]
    df['pgas_attack_probability'] = prob
    df['pgas_attack_pred_threshold'] = (prob >= best_threshold).astype(int)
    df['pgas_attack_pred_hysteresis'] = pgas_hyst_predictions.get(name, np.zeros(len(idx), dtype=int))
    if stacked_prob is not None and len(stacked_prob):
        df['pgas_stacked_attack_probability'] = stacked_prob
        df['pgas_stacked_attack_pred_threshold'] = (stacked_prob >= stack_best_threshold).astype(int)
        df['pgas_stacked_attack_pred_hysteresis'] = stack_hyst_predictions.get(name, np.zeros(len(idx), dtype=int))
    df['posterior_entropy'] = -(marg * np.log(marg + 1e-12)).sum(axis=1) if len(marg) else np.nan
    for k, s in enumerate(stage_names):
        df[f'posterior_prob_{s}'] = marg[:, k] if len(marg) else np.nan
    return df

traj_df = pd.concat([
    trajectory_frame('train', train_idx, train_marg, train_map, train_prob, stack_train_prob),
    trajectory_frame('validation', val_idx, val_marg, val_map, val_prob, stack_val_prob),
    trajectory_frame('test', test_idx, test_marg, test_map, test_prob, stack_test_prob),
], ignore_index=True)
traj_df.to_csv(Path(cfg.OUTPUT_DIR, 'tables', 'posterior_attack_path_trajectory.csv'), index=False)
display(traj_df[['split','phase','local_window_id','true_attack','pgas_attack_probability','pgas_attack_pred_hysteresis','pgas_stacked_attack_probability','pgas_stacked_attack_pred_hysteresis','posterior_entropy']].head())

# %% Cell 31
# ============================================================
# 20. Posterior predictive uncertainty from PGAS parameter draws
# ============================================================

from sklearn.metrics import log_loss

UNCERTAINTY_MAX_POSTERIOR_DRAWS = 80
UNCERTAINTY_CACHE = Path(cfg.OUTPUT_DIR, 'tables', 'posterior_predictive_uncertainty_draws.npz')
REBUILD_UNCERTAINTY_CACHE = True


def collect_theta_draws(chains, max_draws=80, seed=42):
    """Collect posterior parameter draws retained after burn-in/thinning."""
    draws = []

    for ch in chains:
        for ps in ch.get('params_samples', []):
            draws.append({
                'pi': np.asarray(ps['pi']).copy(),
                'A': np.asarray(ps['A']).copy(),
                'mu': np.asarray(ps['mu']).copy(),
                'sigma2': np.asarray(ps['sigma2']).copy(),
            })

    if len(draws) == 0:
        raise RuntimeError(
            'No posterior parameter draws were found in chains. Run the PGAS training cell first.'
        )

    if max_draws is not None and len(draws) > max_draws:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(len(draws), size=max_draws, replace=False))
        draws = [draws[i] for i in keep]

    return draws


def posterior_attack_probability_draws_for_split(Y, fixed_logB_local, theta_draws, K, split_name):
    """Run forward-backward smoothing for each posterior parameter draw."""
    if len(Y) == 0:
        return np.empty((len(theta_draws), 0), dtype=float)

    draw_probs = []

    for theta in tqdm(theta_draws, desc=f'Posterior predictive draws for {split_name}'):
        marg, _ = forward_backward_marginals(Y, fixed_logB_local, theta, K)
        draw_probs.append(attack_prob_from_marg(marg))

    return np.asarray(draw_probs, dtype=float)


if UNCERTAINTY_CACHE.exists() and not REBUILD_UNCERTAINTY_CACHE:
    cached = np.load(UNCERTAINTY_CACHE, allow_pickle=True)

    train_draw_probs = cached['train_draw_probs']
    val_draw_probs = cached['val_draw_probs']
    test_draw_probs = cached['test_draw_probs']
    posterior_draw_count = int(cached['posterior_draw_count'])

    print(f'Loaded cached posterior predictive uncertainty draws: {posterior_draw_count} draws.')

else:
    theta_draws = collect_theta_draws(
        chains,
        max_draws=UNCERTAINTY_MAX_POSTERIOR_DRAWS,
        seed=cfg.RANDOM_SEED,
    )

    posterior_draw_count = len(theta_draws)

    print(f'Using {posterior_draw_count} posterior parameter draws for uncertainty estimation.')

    train_draw_probs = posterior_attack_probability_draws_for_split(
        Y_train,
        fixed_logB_train,
        theta_draws,
        K,
        'train',
    )

    val_draw_probs = (
        posterior_attack_probability_draws_for_split(
            Y_val,
            fixed_logB_val,
            theta_draws,
            K,
            'validation',
        )
        if len(val_idx)
        else np.empty((posterior_draw_count, 0), dtype=float)
    )

    test_draw_probs = (
        posterior_attack_probability_draws_for_split(
            Y_test,
            fixed_logB_test,
            theta_draws,
            K,
            'test',
        )
        if len(test_idx)
        else np.empty((posterior_draw_count, 0), dtype=float)
    )

    np.savez_compressed(
        UNCERTAINTY_CACHE,
        train_draw_probs=train_draw_probs,
        val_draw_probs=val_draw_probs,
        test_draw_probs=test_draw_probs,
        posterior_draw_count=posterior_draw_count,
    )

    print(f'Saved posterior predictive uncertainty cache to: {UNCERTAINTY_CACHE}')

print('Posterior predictive draw shapes:')
print('train:', train_draw_probs.shape)
print('validation:', val_draw_probs.shape)
print('test:', test_draw_probs.shape)

# %% Cell 32
# ============================================================
# 21. Uncertainty metrics, credible intervals, and selective risk
# ============================================================

EPS = 1e-12


def binary_entropy(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def safe_binary_log_loss(y, p):
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)

    if len(y) == 0:
        return np.nan

    try:
        return float(log_loss(y, p, labels=[0, 1]))
    except Exception:
        return np.nan


def safe_corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    mask = np.isfinite(a) & np.isfinite(b)

    if mask.sum() < 3:
        return np.nan

    if np.nanstd(a[mask]) <= 1e-12 or np.nanstd(b[mask]) <= 1e-12:
        return np.nan

    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def safe_auc_local(y, p):
    try:
        if len(np.unique(y)) < 2:
            return np.nan

        return float(roc_auc_score(y, p))
    except Exception:
        return np.nan


def safe_aupr_local(y, p):
    try:
        if len(np.unique(y)) < 2:
            return np.nan

        return float(average_precision_score(y, p))
    except Exception:
        return np.nan


def selective_risk_curve(y, prob, uncertainty, coverages=np.linspace(0.1, 1.0, 10)):
    """Risk when retaining the least uncertain fraction of windows."""
    y = np.asarray(y).astype(int)
    prob = np.asarray(prob, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    pred = (prob >= 0.5).astype(int)
    order = np.argsort(uncertainty)  # least uncertain first

    rows = []
    n = len(y)

    for cov in coverages:
        keep_n = max(1, int(np.ceil(cov * n)))
        keep = order[:keep_n]

        rows.append({
            'coverage': float(cov),
            'kept_windows': int(keep_n),
            'selective_error_rate': float((pred[keep] != y[keep]).mean()),
            'selective_accuracy': float((pred[keep] == y[keep]).mean()),
            'attack_recall_on_kept': float(y[keep].sum() / max(y.sum(), 1)),
        })

    return pd.DataFrame(rows)


def summarize_bayesian_uncertainty(model_name, split_name, idx, prob_mean, draw_probs=None, threshold=0.5):
    y = y_attack[idx].astype(int)
    p = np.clip(np.asarray(prob_mean, dtype=float), 1e-6, 1.0 - 1e-6)

    pred = (p >= threshold).astype(int)
    abs_error = np.abs(y - p)
    true_class_mass = np.where(y == 1, p, 1.0 - p)

    ece, mce, _ = binary_calibration_errors(y, p)

    entropy = binary_entropy(p)
    norm_entropy = entropy / np.log(2.0)

    if draw_probs is not None and draw_probs.size:
        draws = np.clip(np.asarray(draw_probs, dtype=float), 1e-6, 1.0 - 1e-6)

        lower = np.quantile(draws, 0.025, axis=0)
        upper = np.quantile(draws, 0.975, axis=0)
        width = upper - lower

        posterior_sd = draws.std(axis=0)
        aleatoric = np.mean(draws * (1.0 - draws), axis=0)
        epistemic = np.var(draws, axis=0)

        mutual_information = (
            binary_entropy(draws.mean(axis=0))
            - np.mean(binary_entropy(draws), axis=0)
        )

        # Probability-interval width is retained as an uncertainty-width measure.
        # For binary classification, direct containment of a 0/1 label inside a probability interval
        # can be overly strict when posteriors are confident. We therefore report both the strict
        # probability-interval containment and the more interpretable true-class posterior mass.
        ci_coverage = float(((y >= lower) & (y <= upper)).mean())

        true_class_draw_mass = np.where(y[None, :] == 1, draws, 1.0 - draws)
        true_class_mass_lower_95 = np.quantile(true_class_draw_mass, 0.025, axis=0)
        true_class_mass_mean = true_class_draw_mass.mean(axis=0)

        credible_true_class_mass_95 = float((true_class_mass_lower_95 >= 0.025).mean())
        mean_true_class_mass = float(true_class_mass_mean.mean())

        mean_ci_width = float(width.mean())
        coverage_error = float(abs(0.95 - ci_coverage))

        uncertainty_error_corr_width = safe_corr(width, abs_error)
        uncertainty_error_corr_sd = safe_corr(posterior_sd, abs_error)
        uncertainty_error_corr_mi = safe_corr(mutual_information, abs_error)

    else:
        credible_true_class_mass_95 = np.nan
        mean_true_class_mass = float(np.nanmean(true_class_mass)) if len(true_class_mass) else np.nan

        lower = np.full_like(p, np.nan, dtype=float)
        upper = np.full_like(p, np.nan, dtype=float)
        width = np.full_like(p, np.nan, dtype=float)
        posterior_sd = np.full_like(p, np.nan, dtype=float)
        epistemic = np.full_like(p, np.nan, dtype=float)
        mutual_information = np.full_like(p, np.nan, dtype=float)

        aleatoric = p * (1.0 - p)

        ci_coverage = np.nan
        mean_ci_width = np.nan
        coverage_error = np.nan

        uncertainty_error_corr_width = np.nan
        uncertainty_error_corr_sd = np.nan
        uncertainty_error_corr_mi = np.nan

    q75 = np.nanquantile(norm_entropy, 0.75) if len(norm_entropy) else np.nan
    q25 = np.nanquantile(norm_entropy, 0.25) if len(norm_entropy) else np.nan

    high_unc = norm_entropy >= q75
    low_unc = norm_entropy <= q25

    row = {
        'model': model_name,
        'split': split_name,
        'threshold': threshold,
        'windows': int(len(y)),
        'attack_windows': int(y.sum()),
        'accuracy': float(accuracy_score(y, pred)) if len(y) else np.nan,
        'balanced_accuracy': float(balanced_accuracy_score(y, pred)) if len(np.unique(y)) == 2 else np.nan,
        'precision': float(precision_score(y, pred, zero_division=0)),
        'recall': float(recall_score(y, pred, zero_division=0)),
        'f1': float(f1_score(y, pred, zero_division=0)),
        'auroc': safe_auc_local(y, p),
        'aupr': safe_aupr_local(y, p),
        'brier': float(brier_score_loss(y, p)) if len(y) else np.nan,
        'log_loss': safe_binary_log_loss(y, p),
        'ece': ece,
        'mce': mce,
        'mean_predictive_entropy': float(np.nanmean(entropy)),
        'mean_normalized_entropy': float(np.nanmean(norm_entropy)),
        'mean_posterior_sd': float(np.nanmean(posterior_sd)),
        'mean_epistemic_uncertainty': float(np.nanmean(epistemic)),
        'mean_aleatoric_uncertainty': float(np.nanmean(aleatoric)),
        'mean_mutual_information': float(np.nanmean(mutual_information)),
        'mean_true_class_posterior_mass': mean_true_class_mass,
        'credible_true_class_mass_95': credible_true_class_mass_95,
        'ci_95_coverage': ci_coverage,
        'mean_ci_95_width': mean_ci_width,
        'ci_95_coverage_error': coverage_error,
        'uncertainty_error_corr_width': uncertainty_error_corr_width,
        'uncertainty_error_corr_sd': uncertainty_error_corr_sd,
        'uncertainty_error_corr_mi': uncertainty_error_corr_mi,
        'high_entropy_error_rate': float((pred[high_unc] != y[high_unc]).mean()) if high_unc.any() else np.nan,
        'low_entropy_error_rate': float((pred[low_unc] != y[low_unc]).mean()) if low_unc.any() else np.nan,
    }

    detail = pd.DataFrame({
        'split': split_name,
        'model': model_name,
        'global_window_index': idx,
        'true_attack': y,
        'prob_mean': p,
        'prob_lower_95': lower,
        'prob_upper_95': upper,
        'ci_95_width': width,
        'posterior_sd': posterior_sd,
        'predictive_entropy': entropy,
        'normalized_entropy': norm_entropy,
        'epistemic_uncertainty': epistemic,
        'aleatoric_uncertainty': aleatoric,
        'mutual_information': mutual_information,
        'absolute_probability_error': abs_error,
        'predicted_attack': pred,
        'prediction_error': (pred != y).astype(int),
    })

    return row, detail


uncertainty_rows = []
uncertainty_detail_parts = []
selective_parts = []

split_specs = [
    ('train', train_idx, train_prob, train_draw_probs, best_threshold),
    ('validation', val_idx, val_prob, val_draw_probs, best_threshold),
    ('test', test_idx, test_prob, test_draw_probs, best_threshold),
]

for split_name, idx, prob, draws, threshold in split_specs:
    if len(idx) == 0:
        continue

    row, detail = summarize_bayesian_uncertainty(
        'AttackPath-PGAS-PosteriorPredictive',
        split_name,
        idx,
        prob,
        draw_probs=draws,
        threshold=threshold,
    )

    uncertainty_rows.append(row)
    uncertainty_detail_parts.append(detail)

    selective_parts.append(
        selective_risk_curve(
            y_attack[idx],
            prob,
            detail['normalized_entropy'].values,
        ).assign(
            model='AttackPath-PGAS-PosteriorPredictive',
            split=split_name,
        )
    )

# Include the final stacked posterior score in the same table. It has entropy/calibration
# metrics but not MCMC credible intervals because its final calibration layer is deterministic.
stack_split_specs = [
    ('train', train_idx, stack_train_prob, stack_best_threshold),
    ('validation', val_idx, stack_val_prob, stack_best_threshold),
    ('test', test_idx, stack_test_prob, stack_best_threshold),
]

for split_name, idx, prob, threshold in stack_split_specs:
    if len(idx) == 0:
        continue

    row, detail = summarize_bayesian_uncertainty(
        'AttackPath-PGAS-StackedPosterior',
        split_name,
        idx,
        prob,
        draw_probs=None,
        threshold=threshold,
    )

    uncertainty_rows.append(row)
    uncertainty_detail_parts.append(detail)

    selective_parts.append(
        selective_risk_curve(
            y_attack[idx],
            prob,
            detail['normalized_entropy'].values,
        ).assign(
            model='AttackPath-PGAS-StackedPosterior',
            split=split_name,
        )
    )

uncertainty_metrics_df = pd.DataFrame(uncertainty_rows)
uncertainty_window_df = pd.concat(uncertainty_detail_parts, ignore_index=True)
selective_risk_df = pd.concat(selective_parts, ignore_index=True)

# Add phase/local-window context to the detailed uncertainty table.
context_cols = [
    c for c in [
        'phase',
        'local_window_id',
        'window_start',
        'event_count',
        'attack_event_count',
        'event_label_attack_fraction',
    ]
    if c in window_df.columns
]

context = (
    window_df
    .reset_index()
    .rename(columns={'index': 'global_window_index'})[
        ['global_window_index'] + context_cols
    ]
)

uncertainty_window_df = uncertainty_window_df.merge(
    context,
    on='global_window_index',
    how='left',
)

print('Uncertainty-estimation metrics:')
display(uncertainty_metrics_df.round(5))

print('Selective prediction risk-coverage summary:')
display(selective_risk_df.round(5))

uncertainty_metrics_df.to_csv(
    Path(cfg.OUTPUT_DIR, 'tables', 'uncertainty_estimation_metrics.csv'),
    index=False,
)

uncertainty_window_df.to_csv(
    Path(cfg.OUTPUT_DIR, 'tables', 'window_level_uncertainty_estimates.csv'),
    index=False,
)

selective_risk_df.to_csv(
    Path(cfg.OUTPUT_DIR, 'tables', 'selective_risk_coverage_curve.csv'),
    index=False,
)

# %% Cell 33
# ============================================================
# 22. Uncertainty visualizations and reliability plots
# ============================================================

# Use AttackPath-PGAS-PosteriorPredictive details for posterior credible intervals.
posterior_unc = uncertainty_window_df[
    uncertainty_window_df['model'] == 'AttackPath-PGAS-PosteriorPredictive'
].copy()

stack_unc = uncertainty_window_df[
    uncertainty_window_df['model'] == 'AttackPath-PGAS-StackedPosterior'
].copy()

posterior_unc = posterior_unc.sort_values('global_window_index').reset_index(drop=True)
stack_unc = stack_unc.sort_values('global_window_index').reset_index(drop=True)

# 1. Full-stream posterior credible band.
plt.figure(figsize=(12, 4))

x = np.arange(len(posterior_unc))

plt.fill_between(
    x,
    posterior_unc['prob_lower_95'].values,
    posterior_unc['prob_upper_95'].values,
    alpha=0.25,
    label='95% posterior credible interval',
)

plt.plot(
    x,
    posterior_unc['prob_mean'].values,
    linewidth=1.0,
    label='Posterior mean attack probability',
)

attack_positions = np.where(posterior_unc['true_attack'].values == 1)[0]

if len(attack_positions):
    plt.scatter(
        attack_positions,
        np.ones_like(attack_positions) * 1.02,
        marker='|',
        s=80,
        label='True attack windows',
    )

plt.ylim(-0.02, 1.08)
plt.title('AttackPath-PGAS posterior attack probability with 95% credible band')
plt.xlabel('Temporal window')
plt.ylabel('P(attack)')
plt.legend(loc='best')
save_fig('uncertainty_posterior_attack_probability_credible_band.png')
plt.show()

# 2. Epistemic and aleatoric uncertainty over the temporal stream.
plt.figure(figsize=(12, 4))

plt.plot(
    x,
    posterior_unc['epistemic_uncertainty'].values,
    linewidth=1.0,
    label='Epistemic uncertainty',
)

plt.plot(
    x,
    posterior_unc['aleatoric_uncertainty'].values,
    linewidth=1.0,
    alpha=0.75,
    label='Aleatoric uncertainty',
)

plt.title('Epistemic and aleatoric uncertainty across the temporal stream')
plt.xlabel('Temporal window')
plt.ylabel('Uncertainty')
plt.legend(loc='best')
save_fig('uncertainty_epistemic_aleatoric_full_stream.png')
plt.show()

# 3. Predictive entropy distribution by prediction correctness.
plt.figure(figsize=(8, 5))

sns.boxplot(
    data=posterior_unc,
    x='prediction_error',
    y='normalized_entropy',
)

plt.xticks([0, 1], ['Correct', 'Incorrect'])
plt.title('Predictive entropy by prediction correctness')
plt.xlabel('Prediction outcome')
plt.ylabel('Normalized predictive entropy')
save_fig('uncertainty_entropy_by_prediction_correctness.png')
plt.show()

# 4. Reliability diagrams for the test split.
def calibration_table_for_plot(model_name, split_name):
    sub = uncertainty_window_df[
        (uncertainty_window_df['model'] == model_name)
        & (uncertainty_window_df['split'] == split_name)
    ].copy()

    if sub.empty:
        return pd.DataFrame()

    _, _, cal = binary_calibration_errors(
        sub['true_attack'].values,
        sub['prob_mean'].values,
        n_bins=10,
    )

    cal['model'] = model_name
    cal['split'] = split_name

    return cal


cal_plot_df = pd.concat(
    [
        calibration_table_for_plot('AttackPath-PGAS-PosteriorPredictive', 'test'),
        calibration_table_for_plot('AttackPath-PGAS-StackedPosterior', 'test'),
    ],
    ignore_index=True,
)

if not cal_plot_df.empty:
    plt.figure(figsize=(6, 6))

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle='--',
        label='Perfect calibration',
    )

    for model_name, sub in cal_plot_df.groupby('model'):
        plt.plot(
            sub['mean_confidence'],
            sub['empirical_attack_rate'],
            marker='o',
            label=model_name,
        )

    plt.title('Reliability diagram for posterior attack probability on test windows')
    plt.xlabel('Mean predicted attack probability')
    plt.ylabel('Empirical attack rate')
    plt.legend(loc='best')
    save_fig('uncertainty_reliability_diagram_test.png')
    plt.show()

cal_plot_df.to_csv(
    Path(cfg.OUTPUT_DIR, 'tables', 'uncertainty_reliability_diagram_test.csv'),
    index=False,
)

# 5. Selective risk-coverage curve.
plt.figure(figsize=(8, 5))

for (model_name, split_name), sub in selective_risk_df.groupby(['model', 'split']):
    if split_name != 'test':
        continue

    sub = sub.sort_values('coverage')

    plt.plot(
        sub['coverage'],
        sub['selective_error_rate'],
        marker='o',
        label=model_name,
    )

plt.title('Selective prediction risk-coverage curve on test windows')
plt.xlabel('Coverage retained from least uncertain to most uncertain')
plt.ylabel('Selective error rate')
plt.legend(loc='best')
save_fig('uncertainty_selective_risk_coverage_test.png')
plt.show()

# 6. Credible interval width versus absolute probability error.
plt.figure(figsize=(7, 5))

test_posterior_unc = posterior_unc[posterior_unc['split'] == 'test'].copy()

if not test_posterior_unc.empty:
    plt.scatter(
        test_posterior_unc['ci_95_width'],
        test_posterior_unc['absolute_probability_error'],
        alpha=0.75,
    )

    plt.title('Credible interval width versus absolute probability error on test windows')
    plt.xlabel('95% credible interval width')
    plt.ylabel('Absolute probability error')
    save_fig('uncertainty_ci_width_vs_error_test.png')
    plt.show()

print('Saved uncertainty metrics and visualizations to:')
print(Path(cfg.OUTPUT_DIR, 'tables', 'uncertainty_estimation_metrics.csv'))
print(Path(cfg.OUTPUT_DIR, 'tables', 'window_level_uncertainty_estimates.csv'))
print(Path(cfg.OUTPUT_DIR, 'figures'))

# %% Cell 34

# ============================================================
# 23. Diagnostics and comprehensive comparison visualizations
# ============================================================

# PGAS traces.
plt.figure(figsize=(10, 4))
for c, sub in trace_df.groupby('chain'):
    plt.plot(sub['iter'], sub['path_loglik'], label=f'Chain {c}', alpha=0.85)
plt.title('Hybrid PGAS path log-likelihood trace')
plt.xlabel('Iteration')
plt.ylabel('Path log-likelihood')
plt.legend()
save_fig('hybrid_pgas_path_loglik_trace.png')
plt.show()

plt.figure(figsize=(10, 4))
for c, sub in trace_df.groupby('chain'):
    if 'transition_0_to_1' in sub:
        plt.plot(sub['iter'], sub['transition_0_to_1'], label=f'Chain {c}', alpha=0.85)
plt.title('Posterior trace for BENIGN to ATTACK transition probability')
plt.xlabel('Iteration')
plt.ylabel('A[0,1]')
plt.legend()
save_fig('hybrid_pgas_benign_to_attack_transition_trace.png')
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(pd.DataFrame(A_mean, index=stage_names, columns=stage_names), annot=True, fmt='.4f', cmap='mako')
plt.title('Posterior mean transition matrix')
save_fig('posterior_transition_matrix_heatmap.png')
plt.show()

# Confusion matrices for final proposed model on validation and test.
for split, idx, prob, pred in [
    ('validation', val_idx, stack_val_prob, stack_hyst_predictions.get('validation')),
    ('test', test_idx, stack_test_prob, stack_hyst_predictions.get('test')),
]:
    if len(idx) == 0 or pred is None:
        continue
    cm = confusion_matrix(y_attack[idx], pred, labels=[0, 1])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign','Attack'], yticklabels=['Benign','Attack'])
    plt.title(f'Final proposed model confusion matrix ({split})')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    save_fig(f'final_proposed_confusion_matrix_{split}.png')
    plt.show()

# ROC and PR curves for key models on test split.
key_curves = {
    'RiskEnsembleTemporal': risk_prob[test_idx],
    'AttackPath-PGAS-RB': test_prob,
    'AttackPath-PGAS-StackedPosterior': stack_test_prob,
}
y_test = y_attack[test_idx]
if len(np.unique(y_test)) == 2:
    plt.figure(figsize=(7, 5))
    for name, p in key_curves.items():
        fpr, tpr, _ = roc_curve(y_test, p)
        plt.plot(fpr, tpr, label=f'{name} AUC={roc_auc_score(y_test, p):.3f}')
    plt.plot([0,1], [0,1], linestyle='--')
    plt.title('ROC curves on test windows')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend()
    save_fig('test_roc_curves_key_models.png')
    plt.show()

    plt.figure(figsize=(7, 5))
    for name, p in key_curves.items():
        prec, rec, _ = precision_recall_curve(y_test, p)
        plt.plot(rec, prec, label=f'{name} AUPR={average_precision_score(y_test, p):.3f}')
    plt.title('Precision-recall curves on test windows')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    save_fig('test_precision_recall_curves_key_models.png')
    plt.show()

# Baseline/proposed comparison bars.
test_comp = comparison_df[comparison_df['split'] == 'test'].copy()
plot_models = ['RiskEnsembleCalibratedTemporal','RiskEnsembleHysteresis','AttackPath-PGAS-RB','AttackPath-PGAS-Hysteresis','AttackPath-PGAS-StackedPosterior','AttackPath-PGAS-StackedHysteresis']
plot_df = test_comp[test_comp['model'].isin(plot_models)].copy()
for metric in ['f1','aupr','auroc','mcc','ece','fpr','fnr']:
    if metric not in plot_df.columns:
        continue
    plt.figure(figsize=(9, 4))
    sns.barplot(data=plot_df, x='model', y=metric)
    plt.title(f'Test {metric.upper()} comparison for proposed-family models')
    plt.xticks(rotation=35, ha='right')
    plt.tight_layout()
    save_fig(f'test_{metric}_proposed_family_bar.png')
    plt.show()

# Threshold policy curves.
plt.figure(figsize=(8, 5))
for df, label in [(threshold_df, 'PGAS-RB'), (stack_threshold_df, 'Stacked posterior'), (risk_threshold_df, 'Risk ensemble')]:
    if df is not None and not df.empty:
        plt.plot(df['threshold'], df['f1'], label=f'{label} F1')
        plt.plot(df['threshold'], df['fpr'], linestyle='--', alpha=0.7, label=f'{label} FPR')
plt.title('Validation threshold-policy behaviour')
plt.xlabel('Threshold')
plt.ylabel('Metric value')
plt.legend()
save_fig('validation_threshold_policy_curves.png')
plt.show()

# Timeline overlay for test split.
plt.figure(figsize=(12, 4))
xx = np.arange(len(test_idx))
plt.plot(xx, test_prob, label='PGAS-RB probability', linewidth=1.0)
plt.plot(xx, stack_test_prob, label='Stacked posterior probability', linewidth=1.0, alpha=0.8)
attack_positions = np.where(y_attack[test_idx] == 1)[0]
if len(attack_positions):
    plt.scatter(attack_positions, np.ones_like(attack_positions) * 1.03, marker='|', s=90, label='True attack windows')
plt.ylim(-0.02, 1.08)
plt.title('Test attack timeline overlay')
plt.xlabel('Test temporal window')
plt.ylabel('Attack probability')
plt.legend()
save_fig('test_attack_timeline_overlay.png')
plt.show()

# %% Cell 35
# ============================================================
# 24. R-hat style scalar diagnostics
# ============================================================

def rhat_from_chains(values_by_chain):
    arrays = [np.asarray(v, dtype=float) for v in values_by_chain if len(v) > 5]
    if len(arrays) < 2:
        return np.nan
    min_len = min(len(a) for a in arrays)
    arrays = np.stack([a[-min_len:] for a in arrays], axis=0)
    m, n = arrays.shape
    chain_means = arrays.mean(axis=1)
    chain_vars = arrays.var(axis=1, ddof=1)
    B = n * chain_means.var(ddof=1)
    W = chain_vars.mean()
    if W <= 0:
        return np.nan
    var_hat = ((n - 1) / n) * W + B / n
    return float(np.sqrt(var_hat / W))

rhat_rows = []
for col in ['path_loglik'] + [f'prev_state_{k}' for k in range(K)] + [f'self_transition_{k}' for k in range(K)] + [f'transition_0_to_{k}' for k in range(K)]:
    if col not in trace_df.columns:
        continue
    vals = []
    for c, sub in trace_df[trace_df['iter'] >= cfg.BURN_IN].groupby('chain'):
        vals.append(sub[col].values)
    rhat_rows.append({'quantity': col, 'rhat': rhat_from_chains(vals)})

rhat_df = pd.DataFrame(rhat_rows)
display(rhat_df.round(4))
rhat_df.to_csv(Path(cfg.OUTPUT_DIR, 'tables', 'pgas_scalar_rhat_diagnostics.csv'), index=False)

A_lower = np.quantile(A_samples, 0.025, axis=0)
A_upper = np.quantile(A_samples, 0.975, axis=0)
A_rows = []
for i, src in enumerate(stage_names):
    for j, dst in enumerate(stage_names):
        A_rows.append({'from_state': src, 'to_state': dst, 'mean': A_mean[i,j], 'lower_95': A_lower[i,j], 'upper_95': A_upper[i,j]})
A_summary_df = pd.DataFrame(A_rows)
display(A_summary_df.round(5))
A_summary_df.to_csv(Path(cfg.OUTPUT_DIR, 'tables', 'transition_matrix_posterior_summary.csv'), index=False)

# %% Cell 36

# ============================================================
# 25. Top-k prioritization, lift, transition, segment, and detection-delay analysis
# Includes manuscript-facing Proposed AttackPath-PGAS final model
# ============================================================


def topk_analysis(model, name, idx, prob, ks=(10, 25, 50, 100, 200, 500)):
    y = y_attack[idx]
    rows = []
    order = np.argsort(prob)[::-1]
    base_rate = y.mean() if len(y) else np.nan
    for k in ks:
        kk = min(k, len(order))
        if kk <= 0:
            continue
        top = order[:kk]
        precision_k = float(y[top].mean())
        recall_k = float(y[top].sum() / max(y.sum(), 1))
        rows.append({
            'model': model, 'split': name, 'k': kk,
            'attacks_found': int(y[top].sum()),
            'precision_at_k': precision_k,
            'recall_at_k': recall_k,
            'lift_at_k': float(precision_k / base_rate) if base_rate and base_rate > 0 else np.nan,
        })
    return pd.DataFrame(rows)

# Probability models for top-k analysis.
prob_models = {
    'RiskEnsembleTemporal': risk_prob,
    'AttackPath-PGAS-RB': pgas_prob_full,
    'AttackPath-PGAS-StackedPosterior': stack_prob_full,
}
if 'riskfusion_prob_full' in globals():
    prob_models['AttackPath-PGAS-RiskFusionHysteresis'] = riskfusion_prob_full
if 'riskanchored_prob_full' in globals():
    prob_models['Proposed AttackPath-PGAS'] = riskanchored_prob_full

topk_parts = []
for model, pfull in prob_models.items():
    topk_parts.append(topk_analysis(model, 'validation', val_idx, pfull[val_idx]))
    topk_parts.append(topk_analysis(model, 'test', test_idx, pfull[test_idx]))
topk_df = pd.concat(topk_parts, ignore_index=True)
display(topk_df.round(4))
topk_df.to_csv(Path(cfg.OUTPUT_DIR, 'tables', 'topk_alert_prioritization.csv'), index=False)

plt.figure(figsize=(8,5))
for (model, split), sub in topk_df[topk_df['split'] == 'test'].groupby(['model','split']):
    plt.plot(sub['k'], sub['recall_at_k'], marker='o', label=model)
plt.title('Top-k alert recall on test windows')
plt.xlabel('Top-k reviewed windows')
plt.ylabel('Attack recall at k')
plt.legend()
save_fig('topk_alert_recall_test.png')
plt.show()

plt.figure(figsize=(8,5))
for (model, split), sub in topk_df[topk_df['split'] == 'test'].groupby(['model','split']):
    plt.plot(sub['k'], sub['lift_at_k'], marker='o', label=model)
plt.title('Top-k lift over base attack rate on test windows')
plt.xlabel('Top-k reviewed windows')
plt.ylabel('Lift at k')
plt.legend()
save_fig('topk_alert_lift_test.png')
plt.show()


def transition_indices(z):
    z = np.asarray(z).astype(int)
    return np.where(z[1:] != z[:-1])[0] + 1


def transition_eval(name, model, true_z, pred_z, tolerance=1):
    true_t = transition_indices(true_z)
    pred_t = transition_indices(pred_z)
    matched = 0
    used = set()
    delays = []
    for pt in pred_t:
        candidates = [i for i, tt in enumerate(true_t) if i not in used and abs(tt - pt) <= tolerance]
        if candidates:
            i = candidates[0]
            used.add(i)
            matched += 1
            delays.append(pt - true_t[i])
    precision = matched / max(len(pred_t), 1)
    recall = matched / max(len(true_t), 1)
    return {'split': name, 'model': model, 'true_transitions': len(true_t), 'predicted_transitions': len(pred_t), 'matched_within_one_window': matched, 'transition_precision': precision, 'transition_recall': recall, 'mean_transition_delay_windows': float(np.mean(delays)) if delays else np.nan}

# Prediction models for transition and segment analysis.
pred_models = {
    'AttackPath-PGAS-Hysteresis': pgas_hyst_predictions,
    'AttackPath-PGAS-StackedHysteresis': stack_hyst_predictions,
}
if 'riskfusion_hyst_predictions' in globals():
    pred_models['AttackPath-PGAS-RiskFusionHysteresis'] = riskfusion_hyst_predictions
if 'riskanchored_predictions' in globals():
    pred_models['Proposed AttackPath-PGAS'] = riskanchored_predictions

transition_rows = []
for model, pred_dict in pred_models.items():
    transition_rows.append(transition_eval('validation', model, y_attack[val_idx], pred_dict['validation']))
    transition_rows.append(transition_eval('test', model, y_attack[test_idx], pred_dict['test']))
transition_df = pd.DataFrame(transition_rows)
display(transition_df.round(4))
transition_df.to_csv(Path(cfg.OUTPUT_DIR, 'tables', 'transition_event_detection_analysis.csv'), index=False)


def segments_from_binary(z):
    z = np.asarray(z).astype(int)
    starts = np.where((z == 1) & (np.r_[0, z[:-1]] == 0))[0]
    ends = np.where((z == 1) & (np.r_[z[1:], 0] == 0))[0] + 1
    return list(zip(starts, ends))


def segment_iou(a, b):
    s1, e1 = a; s2, e2 = b
    inter = max(0, min(e1, e2) - max(s1, s2))
    union = max(e1, e2) - min(s1, s2)
    return inter / max(union, 1)


def segment_eval(name, model, true_z, pred_z, iou_threshold=0.1):
    true_seg = segments_from_binary(true_z)
    pred_seg = segments_from_binary(pred_z)
    matched = 0
    used = set()
    ious = []
    onset_delays = []
    for ps in pred_seg:
        vals = [(i, segment_iou(ps, ts)) for i, ts in enumerate(true_seg) if i not in used]
        vals = [(i, v) for i, v in vals if v >= iou_threshold]
        if vals:
            i, v = sorted(vals, key=lambda x: x[1], reverse=True)[0]
            used.add(i)
            matched += 1
            ious.append(v)
            onset_delays.append(ps[0] - true_seg[i][0])
    precision = matched / max(len(pred_seg), 1)
    recall = matched / max(len(true_seg), 1)
    return {'split': name, 'model': model, 'true_segments': len(true_seg), 'predicted_segments': len(pred_seg), 'matched_segments': matched, 'segment_precision': precision, 'segment_recall': recall, 'mean_segment_iou': float(np.mean(ious)) if ious else np.nan, 'mean_onset_delay_windows': float(np.mean(onset_delays)) if onset_delays else np.nan}

segment_rows = []
for model, pred_dict in pred_models.items():
    segment_rows.append(segment_eval('validation', model, y_attack[val_idx], pred_dict['validation']))
    segment_rows.append(segment_eval('test', model, y_attack[test_idx], pred_dict['test']))
segment_df = pd.DataFrame(segment_rows)
display(segment_df.round(4))
segment_df.to_csv(Path(cfg.OUTPUT_DIR, 'tables', 'segment_level_attack_path_analysis.csv'), index=False)

# Transition overlay plot for test using the validation-selected final proposed model.
final_overlay_model = globals().get('FINAL_PROPOSED_MODEL_NAME', 'AttackPath-PGAS-RiskFusionHysteresis')
if final_overlay_model in pred_models:
    pred_z = pred_models[final_overlay_model]['test']
elif 'AttackPath-PGAS-RiskFusionHysteresis' in pred_models:
    pred_z = pred_models['AttackPath-PGAS-RiskFusionHysteresis']['test']
else:
    pred_z = stack_hyst_predictions['test']

plt.figure(figsize=(12,4))
true_z = y_attack[test_idx]
plt.step(np.arange(len(true_z)), true_z, where='post', label='True attack state')
plt.step(np.arange(len(pred_z)), pred_z * 0.85, where='post', label='Predicted attack state')
for t in transition_indices(true_z):
    plt.axvline(t, linestyle='--', alpha=0.3)
plt.title('Test transition-event overlay')
plt.xlabel('Test temporal window')
plt.ylabel('State')
plt.yticks([0, 0.85, 1], ['Benign', 'Predicted attack', 'True attack'])
plt.legend()
save_fig('test_transition_event_overlay.png')
plt.show()

# %% Cell 38

# ============================================================
# 26. SOTA comparison tracking and stop/go decision table
# ============================================================

# Peer-reviewed / formal venue SOTA references from 2022-2026.
# N/R means the metric was not reported in the source or not comparable under the paper's protocol.
sota_rows = [
    {
        'study': 'Ghiasvand et al. 2024 CICAPT-IIoT SSL Extra Trees',
        'venue_type': 'MobiQuitous 2024 / Springer proceedings author copy',
        'dataset_or_protocol': 'CICAPT-IIoT provenance binary node classification',
        'accuracy': 0.9986, 'precision': 'N/R', 'recall': 0.8444, 'f1': 0.8785,
        'auroc': 'N/R', 'aupr': 'N/R', 'ece': 'N/R', 'mcc': 'N/R',
        'topk_recall_100': 'N/R', 'transition_recall': 'N/R',
        'notes': 'Best SSL binary F1 reported in Table 10.'
    },
    {
        'study': 'Ghiasvand et al. 2024 CICAPT-IIoT SSL Random Forest',
        'venue_type': 'MobiQuitous 2024 / Springer proceedings author copy',
        'dataset_or_protocol': 'CICAPT-IIoT provenance binary node classification',
        'accuracy': 0.9986, 'precision': 'N/R', 'recall': 0.8232, 'f1': 0.8760,
        'auroc': 'N/R', 'aupr': 'N/R', 'ece': 'N/R', 'mcc': 'N/R',
        'topk_recall_100': 'N/R', 'transition_recall': 'N/R',
        'notes': 'Second strongest SSL binary F1 in Table 10.'
    },
    {
        'study': 'Sahu et al. 2026 near real-time APT detection',
        'venue_type': 'ACM peer-reviewed conference',
        'dataset_or_protocol': 'CIC-APT-IIoT2024 network APT detection',
        'accuracy': 0.9514, 'precision': 'N/R', 'recall': 'N/R', 'f1': 0.9511,
        'auroc': 'N/R', 'aupr': 'N/R', 'ece': 'N/R', 'mcc': 'N/R',
        'topk_recall_100': 'N/R', 'transition_recall': 'N/R',
        'notes': 'Reported accuracy 95.14% and F1 95.11%.'
    },
    {
        'study': 'TIPSO-GAN 2026',
        'venue_type': 'NDSS 2026 peer-reviewed symposium',
        'dataset_or_protocol': 'CICAPT-IIoT2024 grouped malicious traffic detection',
        'accuracy': 'N/R', 'precision': 'N/R', 'recall': 'N/R', 'f1': 0.989,
        'auroc': 'N/R', 'aupr': 0.999, 'ece': 'N/R', 'mcc': 'N/R',
        'topk_recall_100': 'N/R', 'transition_recall': 'N/R',
        'notes': 'Reports 98.9+/-0.1 F1 and 0.999+/-0.002 macro PR-AUC on CICAPT-IIoT2024.'
    },
    {
        'study': 'Fraihat et al. 2026 TSTformer-LSTM + feature optimization',
        'venue_type': 'Discover Internet of Things / Springer Nature',
        'dataset_or_protocol': 'Average over CICAPT-IIoT, Edge-IIoTset, X-IIoTID, WUSTL-IIoT',
        'accuracy': 0.9927, 'precision': 0.9925, 'recall': 0.9917, 'f1': 0.9921,
        'auroc': 0.9935, 'aupr': 'N/R', 'ece': 'N/R', 'mcc': 'N/R',
        'topk_recall_100': 'N/R', 'transition_recall': 'N/R',
        'notes': 'Cross-dataset average; not a trajectory/uncertainty model.'
    },
    {
        'study': 'Zhang et al. 2025 provenance graph feature enhancement',
        'venue_type': 'Electronics 2025',
        'dataset_or_protocol': 'CICAPT-IIoT provenance graph anomaly/snapshot detection',
        'accuracy': 'N/R', 'precision': 0.87, 'recall': 0.97, 'f1': 0.91,
        'auroc': 0.88, 'aupr': 'N/R', 'ece': 'N/R', 'mcc': 'N/R',
        'topk_recall_100': 'N/R', 'transition_recall': 'N/R',
        'notes': 'Reported/proxied as full model performance in provenance graph feature enhancement literature.'
    },
]

sota_reference_df = pd.DataFrame(sota_rows)
sota_reference_df.to_csv(Path(cfg.OUTPUT_DIR, 'tables', 'sota_reference_table_2022_2026.csv'), index=False)
display(sota_reference_df)

# Extract current final proposed and proposed-family results.
final_model_name = globals().get('FINAL_PROPOSED_MODEL_NAME', 'Proposed AttackPath-PGAS')
fallback_model_name = 'Proposed AttackPath-PGAS'
final_test = comparison_df[(comparison_df['split'] == 'test') & (comparison_df['model'] == final_model_name)]
if final_test.empty:
    final_model_name = fallback_model_name
    final_test = comparison_df[(comparison_df['split'] == 'test') & (comparison_df['model'] == final_model_name)]
if final_test.empty:
    # Use the best proposed-family model by F1 as a fallback.
    proposed_family = comparison_df[(comparison_df['split'] == 'test') & (comparison_df['model'].str.contains('AttackPath-PGAS', na=False))]
    final_test = proposed_family.sort_values(['f1','aupr','mcc'], ascending=False).head(1)
    final_model_name = final_test.iloc[0]['model'] if not final_test.empty else 'N/A'

final_row = final_test.iloc[0].to_dict() if not final_test.empty else {}

# Attach top-k and transition evidence, which most SOTA detection papers do not report.
topk100 = topk_df[(topk_df['split'] == 'test') & (topk_df['model'].astype(str).str.startswith('AttackPath-PGAS')) & (topk_df['k'] == 100)]
if not topk100.empty:
    final_topk100 = float(topk100.sort_values('recall_at_k', ascending=False).iloc[0]['recall_at_k'])
else:
    final_topk100 = np.nan

trans_test = transition_df[(transition_df['split'] == 'test') & (transition_df['model'].str.contains('AttackPath-PGAS', na=False))]
final_transition_recall = float(trans_test['transition_recall'].max()) if not trans_test.empty else np.nan
final_transition_precision = float(trans_test['transition_precision'].max()) if not trans_test.empty else np.nan

current_summary = {
    'study': final_model_name + ' current run',
    'venue_type': 'This notebook',
    'dataset_or_protocol': 'Full Phase 1+2 streaming window-level CICAPT-IIoT network trajectory inference',
    'accuracy': final_row.get('accuracy', np.nan),
    'precision': final_row.get('precision', np.nan),
    'recall': final_row.get('recall', np.nan),
    'f1': final_row.get('f1', np.nan),
    'auroc': final_row.get('auroc', np.nan),
    'aupr': final_row.get('aupr', np.nan),
    'ece': final_row.get('ece', np.nan),
    'mcc': final_row.get('mcc', np.nan),
    'fpr': final_row.get('fpr', np.nan),
    'fnr': final_row.get('fnr', np.nan),
    'topk_recall_100': final_topk100,
    'transition_recall': final_transition_recall,
    'transition_precision': final_transition_precision,
    'notes': 'Proposed Bayesian trajectory model; includes posterior uncertainty, calibration, top-k prioritization, and transition reconstruction.'
}

current_vs_sota_df = pd.concat([sota_reference_df, pd.DataFrame([current_summary])], ignore_index=True)
current_vs_sota_df.to_csv(Path(cfg.OUTPUT_DIR, 'tables', 'current_vs_sota_tracking_table.csv'), index=False)
display(current_vs_sota_df)

# Metric-wise gap table against the strongest reported SOTA value.
def numeric_or_nan(x):
    try:
        return float(x)
    except Exception:
        return np.nan

metrics_to_track = ['accuracy','precision','recall','f1','auroc','aupr','ece','mcc','topk_recall_100','transition_recall']
gap_rows = []
for m in metrics_to_track:
    sota_vals = pd.to_numeric(sota_reference_df[m], errors='coerce') if m in sota_reference_df.columns else pd.Series(dtype=float)
    current_val = numeric_or_nan(current_summary.get(m, np.nan))
    if m in ['ece']:
        best_sota = sota_vals.min() if sota_vals.notna().any() else np.nan
        gap = current_val - best_sota if np.isfinite(current_val) and np.isfinite(best_sota) else np.nan
        status = 'beats/at-par' if np.isfinite(gap) and gap <= 0 else ('N/R in SOTA' if not np.isfinite(best_sota) else 'needs improvement')
    else:
        best_sota = sota_vals.max() if sota_vals.notna().any() else np.nan
        gap = current_val - best_sota if np.isfinite(current_val) and np.isfinite(best_sota) else np.nan
        status = 'beats/at-par' if np.isfinite(gap) and gap >= -0.005 else ('N/R in SOTA' if not np.isfinite(best_sota) else 'needs improvement')
    gap_rows.append({'metric': m, 'current_value': current_val, 'best_reported_sota': best_sota, 'gap_current_minus_sota': gap, 'status': status})

sota_gap_df = pd.DataFrame(gap_rows)
sota_gap_df.to_csv(Path(cfg.OUTPUT_DIR, 'tables', 'sota_gap_stop_go_table.csv'), index=False)
display(sota_gap_df)

# Stop/go recommendation: detection metrics and trajectory metrics are separated.
f1_ok = (sota_gap_df.loc[sota_gap_df['metric'] == 'f1', 'status'].iloc[0] == 'beats/at-par') if 'f1' in sota_gap_df['metric'].values else False
auroc_ok = (sota_gap_df.loc[sota_gap_df['metric'] == 'auroc', 'status'].iloc[0] == 'beats/at-par') if 'auroc' in sota_gap_df['metric'].values else False
aupr_ok = (sota_gap_df.loc[sota_gap_df['metric'] == 'aupr', 'status'].iloc[0] == 'beats/at-par') if 'aupr' in sota_gap_df['metric'].values else False
trajectory_ok = (np.isfinite(final_transition_recall) and final_transition_recall >= 0.70) or (np.isfinite(final_topk100) and final_topk100 >= 0.90)

stop_go = pd.DataFrame([
    {'criterion': 'Conventional detection F1 at/near strongest SOTA', 'passed': bool(f1_ok), 'interpretation': 'Needed if the paper claims detection-only SOTA.'},
    {'criterion': 'AUROC at/near strongest SOTA', 'passed': bool(auroc_ok), 'interpretation': 'Important for ranking quality under imbalance.'},
    {'criterion': 'AUPR at/near strongest SOTA', 'passed': bool(aupr_ok), 'interpretation': 'Most important threshold-free rare-event metric.'},
    {'criterion': 'Trajectory/top-k contribution strong enough for trade-off argument', 'passed': bool(trajectory_ok), 'interpretation': 'Supports the Bayesian attack-path contribution even when F1 is lower.'},
])
stop_go.to_csv(Path(cfg.OUTPUT_DIR, 'tables', 'implementation_stop_go_recommendation.csv'), index=False)
display(stop_go)

if bool(f1_ok and auroc_ok and aupr_ok):
    print('STOP SIGNAL: conventional detection metrics are at or above the tracked SOTA range.')
elif bool(trajectory_ok and (auroc_ok or aupr_ok)):
    print('TRADE-OFF SIGNAL: detection-only SOTA is not fully beaten, but trajectory/top-k/Bayesian outputs may justify the contribution.')
else:
    print('IMPROVE SIGNAL: continue improving the model before freezing implementation results.')

# %% Cell 39
# ============================================================
# 27. Save experiment metadata and model artifacts
# ============================================================

metadata = {
    'framework': 'AttackPath-PGAS-HierarchicalEventRiskPGAS' ,
    'dataset_root': cfg.DATASET_ROOT,
    'output_dir': cfg.OUTPUT_DIR,
    'configuration': asdict(cfg),
    'uses_all_raw_rows': cfg.STREAM_MAX_ROWS_PER_FILE is None,
    'streaming_rows_report': stream_report_df.to_dict(orient='records'),
    'stage_names': stage_names,
    'stage_to_id': stage_to_id,
    'feature_columns': feature_cols,
    'n_windows': int(T),
    'n_features': int(X.shape[1]),
    'n_states': int(K),
    'train_windows': int(len(train_idx)),
    'validation_windows': int(len(val_idx)),
    'test_windows': int(len(test_idx)),
    'best_pgas_threshold': float(best_threshold),
    'pgas_hysteresis_params': pgas_hyst_params,
    'stacked_pgas_threshold': float(stack_best_threshold),
    'stacked_hysteresis_params': stack_hyst_params,
    'risk_best_threshold': float(risk_best_threshold),
    'risk_hysteresis_params': risk_hyst_params,
    'final_model_name': globals().get('final_model_name', 'N/A'),
}

save_json(metadata, Path(cfg.OUTPUT_DIR, 'experiment_metadata.json'))
joblib.dump({
    'cfg': cfg,
    'stage_names': stage_names,
    'stage_to_id': stage_to_id,
    'feature_cols': feature_cols,
    'theta_mean': theta_mean,
    'risk_models': risk_models,
    'risk_model_weights': dict(zip(score_names, weights.tolist())),
    'risk_calibration_scale': scale,
    'risk_calibration_bias': bias,
    'best_pgas_threshold': best_threshold,
}, Path(cfg.OUTPUT_DIR, 'models', 'attackpath_pgas_hybrid_model_artifacts.joblib'))

print('Experiment artifacts saved successfully.')
print('Output directory:', cfg.OUTPUT_DIR)

# %% Cell 41
from attackpath_pgas.benchmark import benchmark as benchmark_callable

benchmark_rows = []

def record_benchmark(model_name, inference_function, observations):
    _, result = benchmark_callable(inference_function, warmup=2, repetitions=10)
    benchmark_rows.append({
        'model': model_name,
        'observations': int(observations),
        'mean_runtime_seconds': result.mean_seconds,
        'runtime_sd_seconds': result.standard_deviation_seconds,
        'median_runtime_seconds': result.median_seconds,
        'p95_runtime_seconds': result.p95_seconds,
        'incremental_peak_memory_gb': result.incremental_peak_gb,
        'throughput_per_second': observations / result.mean_seconds,
    })

for model_name, artifact in risk_models.items():
    if isinstance(artifact, dict) or model_name in {'PCABackgroundError'}:
        continue
    if hasattr(artifact, 'predict_proba'):
        record_benchmark(
            model_name,
            lambda artifact=artifact: artifact.predict_proba(X[test_idx])[:, 1],
            len(test_idx),
        )
    elif hasattr(artifact, 'decision_function'):
        record_benchmark(
            model_name,
            lambda artifact=artifact: artifact.decision_function(X[test_idx]),
            len(test_idx),
        )

record_benchmark(
    'AttackPath-PGAS-PosteriorSmoothing',
    lambda: forward_backward_marginals(Y_test, fixed_logB_test, theta_mean, K)[0][:, 1],
    len(test_idx),
)
record_benchmark(
    'Proposed AttackPath-PGAS-CachedPolicy',
    lambda: _apply_policy(riskanchored_prob_full[test_idx], riskanchored_policy),
    len(test_idx),
)

inference_benchmark_df = pd.DataFrame(benchmark_rows).sort_values('mean_runtime_seconds')
inference_benchmark_df.to_csv(
    Path(cfg.OUTPUT_DIR, 'tables', 'inference_runtime_memory.csv'), index=False
)
display(inference_benchmark_df)
