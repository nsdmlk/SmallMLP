import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from smallgbm import SmallGBMClassifier  # Ваша библиотека

# ==========================================
# 1. GENERATE WEIERSTRASS DATA (Classification)
# ==========================================
def weierstrass(x, a=0.5, b=3, num_terms=10):
    total = np.zeros_like(x)
    for n in range(num_terms):
        total += a**n * np.cos(b**n * np.pi * x)
    return total

def generate_classification_data(n_samples, noise_std=0.1, seed=42):
    np.random.seed(seed)
    X = np.random.uniform(-2, 2, n_samples).reshape(-1, 1)
    y_cont = weierstrass(X.flatten()) + np.random.normal(0, noise_std, n_samples)
    # Binary classification: Is the value positive or negative?
    y = (y_cont > 0).astype(int)
    return X, y

# ==========================================
# 2. BENCHMARK
# ==========================================
def run_benchmark():
    n_samples_list = [50, 100, 200, 500]
    results = {'n': [], 'XGBoost_AUC': [], 'SmallGBM_AUC': []}
    
    for n in n_samples_list:
        print(f"Testing n = {n}...")
        X, y = generate_classification_data(n_samples=n)
        
        # Standard split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Model 1: XGBoost (Default hyperparameters, zero-tuning)
        xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        xgb_model.fit(X_train, y_train)
        preds_xgb = xgb_model.predict_proba(X_test)[:, 1]
        auc_xgb = roc_auc_score(y_test, preds_xgb)
        
        # Model 2: SmallGBM (Default hyperparameters, zero-tuning)
        small_model = SmallGBMClassifier(random_state=42)
        small_model.fit(X_train, y_train)
        preds_small = small_model.predict_proba(X_test)[:, 1]
        auc_small = roc_auc_score(y_test, preds_small)
        
        results['n'].append(n)
        results['XGBoost_AUC'].append(auc_xgb)
        results['SmallGBM_AUC'].append(auc_small)
    
    return results

if __name__ == "__main__":
    results = run_benchmark()
    
    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(results['n'], results['XGBoost_AUC'], marker='o', label='XGBoost')
    plt.plot(results['n'], results['SmallGBM_AUC'], marker='s', label='SmallGBM')
    plt.xscale('log')
    plt.xlabel('Sample Size (n)')
    plt.ylabel('ROC-AUC')
    plt.title('Weierstrass Classification (Fractal Boundary)')
    plt.legend()
    plt.grid(True)
    plt.ylim(0.4, 1.0)
    plt.savefig('weierstrass_classification.png')
    plt.show()
    
    print("\nResults:")
    for i, n in enumerate(results['n']):
        print(f"n={n}: XGBoost AUC={results['XGBoost_AUC'][i]:.4f}, SmallGBM AUC={results['SmallGBM_AUC'][i]:.4f}")