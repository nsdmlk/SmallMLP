"""
Ablation: Fixed Alpha/Beta vs Adaptive Alpha.
Uses 5 datasets.
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from smallmlp import SmallMLPClassifier
from run_full_benchmark import build_classification_datasets

warnings.filterwarnings("ignore")

TARGETS = ["sk_iris", "sk_breast_cancer", "uci_glass", "uci_yeast", "uci_sonar"]

# --- ФОРМУЛЫ ДЛЯ АДАПТИВНОГО ALPHA ---
def get_adaptive_alpha_v1(n, d, K):
    # Чем меньше признаков и больше классов, тем больше alpha
    raw = (K / max(d, 1)) * 10.0
    return float(np.clip(raw, 2.0, 10.0))

def get_adaptive_alpha_v2(n, d, K):
    # Учитываем также соотношение данных на класс
    n_per_class = max(n / K, 1.0)
    raw = 2.0 * np.log2(n_per_class) * (K / max(d, 1))
    return float(np.clip(raw, 2.0, 10.0))

def run_benchmark(n_splits=5, seed=42):
    datasets = build_classification_datasets()
    datasets = [(n, X, y) for n, X, y in datasets if n in TARGETS]
    print(f"Using {len(datasets)} datasets: {[d[0] for d in datasets]}")

    results = []
    
    for ds_name, X, y in datasets:
        n, d = X.shape
        K = len(np.unique(y))
        
        # Считаем адаптивные альфы для этого датасета
        a1 = get_adaptive_alpha_v1(n, d, K)
        a2 = get_adaptive_alpha_v2(n, d, K)
        
        print(f"\n{ds_name} (n={n}, d={d}, K={K})")
        print(f"  Formula 1 Alpha: {a1:.2f} | Formula 2 Alpha: {a2:.2f}")

        configs = {
            "Default (4.0)": 4.0,
            "Best Fixed (6.0)": 6.0,
            "Adaptive V1": a1,
            "Adaptive V2": a2
        }
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        
        for name, alpha_val in configs.items():
            accs = []
            for tr, te in skf.split(X, y):
                clf = SmallMLPClassifier(
                    width_mode="formula", class_weight=None,
                    alpha=alpha_val, beta=0.7, # fix beta for fair test
                    max_epochs=500, patience=30, random_state=seed
                )
                clf.fit(X[tr], y[tr])
                acc = accuracy_score(y[te], clf.predict(X[te]))
                accs.append(acc)
            
            results.append({
                "dataset": ds_name,
                "config": name,
                "alpha": alpha_val,
                "acc": np.mean(accs)
            })

    df = pd.DataFrame(results)
    
    # Вывод результатов
    pivot = df.pivot(index="dataset", columns="config", values="acc")
    ranks = pivot.rank(axis=1, ascending=False, method="average")
    
    print("\n" + "="*80)
    print("ACCURACY PER DATASET")
    print("="*80)
    print(pivot.round(4).to_string())
    
    print("\n" + "="*80)
    print("MEAN RANK (lower = better) & MEAN ACCURACY")
    print("="*80)
    summary = pd.DataFrame({
        "Mean Rank": ranks.mean(),
        "Mean Acc": pivot.mean()
    }).sort_values("Mean Rank")
    
    print(summary.round(4).to_string())

if __name__ == "__main__":
    run_benchmark()