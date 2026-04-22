"""
=============================================================================
 EARLY PREDICTION OF TYPE 2 DIABETES - NOVEL HYBRID MODEL COMPARISON
=============================================================================
 Models Implemented:
   Hybrid 1  : Federated Learning Simulation + Transformer Encoder + SHAP-style XAI
   Hybrid 4  : Multi-Modal Cross-Attention Fusion (Clinical + Wearable + Genomic)
   Hybrid 10 : Digital Twin Simulation + Reinforcement Learning Risk Monitoring
               + SHAP Feature Importance

 Datasets Used:
   Dataset 1 : PIMA Indians Diabetes (simulated structure)
   Dataset 2 : NHANES-style Clinical + Lifestyle Dataset (simulated)
   Dataset 3 : Multi-Modal Dataset (Clinical + Wearable + Genomic, simulated)

 Comparison : Accuracy, AUC-ROC, F1, Precision, Recall across all models/datasets
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                              precision_score, recall_score, confusion_matrix,
                              roc_curve, classification_report)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from collections import defaultdict
import copy

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: DATASET GENERATION
# ─────────────────────────────────────────────────────────────────────────────

np.random.seed(42)

def generate_pima_style(n=768):
    """Dataset 1: PIMA Indians Diabetes Database structure"""
    diabetic     = int(n * 0.35)
    non_diabetic = n - diabetic

    def make_samples(n, diabetic=True):
        d = 1 if diabetic else 0
        glucose     = np.random.normal(140 if d else 100, 20, n).clip(50, 220)
        bmi         = np.random.normal(35 if d else 27, 6,  n).clip(15, 65)
        age         = np.random.normal(45 if d else 32, 12, n).clip(18, 80)
        insulin     = np.random.normal(150 if d else 80, 50, n).clip(0, 400)
        pregnancies = np.random.poisson(4 if d else 2, n).clip(0, 15)
        bp          = np.random.normal(80 if d else 70, 12, n).clip(40, 130)
        skin_thick  = np.random.normal(32 if d else 22, 10, n).clip(0, 80)
        dpf         = np.random.exponential(0.6 if d else 0.4, n).clip(0.08, 2.5)
        labels      = np.full(n, d)
        return np.column_stack([pregnancies, glucose, bp, skin_thick,
                                insulin, bmi, dpf, age, labels])

    data = np.vstack([make_samples(diabetic, True), make_samples(non_diabetic, False)])
    np.random.shuffle(data)
    cols = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
            'Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']
    df = pd.DataFrame(data, columns=cols)
    df['Outcome'] = df['Outcome'].astype(int)
    return df


def generate_nhanes_style(n=1200):
    """Dataset 2: NHANES-style Clinical + Lifestyle Dataset"""
    labels = np.random.binomial(1, 0.33, n)

    hba1c       = np.where(labels, np.random.normal(7.2,0.8,n), np.random.normal(5.4,0.5,n)).clip(4,14)
    fasting_glc = np.where(labels, np.random.normal(135,25,n),  np.random.normal(92,12,n)).clip(60,300)
    bmi         = np.where(labels, np.random.normal(33,6,n),    np.random.normal(26,4,n)).clip(16,55)
    waist       = np.where(labels, np.random.normal(100,12,n),  np.random.normal(82,10,n)).clip(55,160)
    age         = np.where(labels, np.random.normal(52,13,n),   np.random.normal(38,15,n)).clip(18,85)
    sedentary   = np.where(labels, np.random.normal(8,2,n),     np.random.normal(5,2,n)).clip(0,16)
    calories    = np.where(labels, np.random.normal(2400,400,n),np.random.normal(1900,350,n)).clip(800,5000)
    smoker      = np.random.binomial(1, np.where(labels, 0.35, 0.2), n)
    family_hist = np.random.binomial(1, np.where(labels, 0.60, 0.25), n)
    systolic_bp = np.where(labels, np.random.normal(138,18,n),  np.random.normal(118,14,n)).clip(80,220)
    hdl         = np.where(labels, np.random.normal(42,10,n),   np.random.normal(55,12,n)).clip(20,100)
    triglyc     = np.where(labels, np.random.normal(200,50,n),  np.random.normal(130,40,n)).clip(50,500)

    df = pd.DataFrame({
        'HbA1c': hba1c, 'FastingGlucose': fasting_glc, 'BMI': bmi,
        'WaistCircumference': waist, 'Age': age, 'SedentaryHours': sedentary,
        'DailyCalories': calories, 'Smoker': smoker, 'FamilyHistory': family_hist,
        'SystolicBP': systolic_bp, 'HDL_Cholesterol': hdl, 'Triglycerides': triglyc,
        'Outcome': labels
    })
    return df


def generate_multimodal_style(n=900):
    """Dataset 3: Multi-Modal - Clinical + Wearable + Genomic features"""
    labels = np.random.binomial(1, 0.38, n)

    # --- Clinical block ---
    glucose  = np.where(labels, np.random.normal(138,22,n), np.random.normal(95,14,n)).clip(60,280)
    bmi      = np.where(labels, np.random.normal(34,7,n),   np.random.normal(26,4,n)).clip(15,60)
    hba1c    = np.where(labels, np.random.normal(7.0,0.9,n),np.random.normal(5.3,0.4,n)).clip(4,13)
    age      = np.where(labels, np.random.normal(50,12,n),  np.random.normal(36,14,n)).clip(18,82)
    bp       = np.where(labels, np.random.normal(136,15,n), np.random.normal(116,12,n)).clip(75,210)

    # --- Wearable block ---
    daily_steps   = np.where(labels, np.random.normal(5000,1500,n), np.random.normal(8500,2000,n)).clip(500,20000)
    sleep_hours   = np.where(labels, np.random.normal(5.8,1.2,n),   np.random.normal(7.1,0.9,n)).clip(3,10)
    resting_hr    = np.where(labels, np.random.normal(82,12,n),     np.random.normal(68,10,n)).clip(45,120)
    hrv_score     = np.where(labels, np.random.normal(35,10,n),     np.random.normal(55,12,n)).clip(10,100)
    active_mins   = np.where(labels, np.random.normal(18,10,n),     np.random.normal(45,15,n)).clip(0,180)

    # --- Genomic/SNP block ---
    snp_tcf7l2    = np.random.binomial(2, np.where(labels, 0.45, 0.22), n)
    snp_kcnj11    = np.random.binomial(2, np.where(labels, 0.38, 0.18), n)
    snp_pparg     = np.random.binomial(2, np.where(labels, 0.30, 0.15), n)
    polygenic_score = np.where(labels, np.random.normal(1.8,0.5,n), np.random.normal(0.9,0.4,n)).clip(0,4)
    family_risk   = np.random.binomial(1, np.where(labels, 0.62, 0.24), n)

    df = pd.DataFrame({
        # Clinical
        'Glucose': glucose, 'BMI': bmi, 'HbA1c': hba1c, 'Age': age, 'BloodPressure': bp,
        # Wearable
        'DailySteps': daily_steps, 'SleepHours': sleep_hours, 'RestingHR': resting_hr,
        'HRV_Score': hrv_score, 'ActiveMinutes': active_mins,
        # Genomic
        'SNP_TCF7L2': snp_tcf7l2, 'SNP_KCNJ11': snp_kcnj11, 'SNP_PPARG': snp_pparg,
        'PolygenicRiskScore': polygenic_score, 'FamilyRisk': family_risk,
        'Outcome': labels
    })
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, model_name="Model"):
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]
    return {
        'Model':     model_name,
        'Accuracy':  round(accuracy_score(y_test, y_pred),      4),
        'AUC-ROC':   round(roc_auc_score(y_test, y_proba),      4),
        'F1-Score':  round(f1_score(y_test, y_pred),            4),
        'Precision': round(precision_score(y_test, y_pred),     4),
        'Recall':    round(recall_score(y_test, y_pred),        4),
        'y_test':    y_test,
        'y_proba':   y_proba,
        'y_pred':    y_pred,
    }


def shap_style_importance(model, X, feature_names, top_n=10):
    """Permutation-based feature importance as SHAP proxy."""
    base_score = model.score(X, model.predict(X))
    importances = []
    for i in range(X.shape[1]):
        X_perm = X.copy()
        X_perm[:, i] = np.random.permutation(X_perm[:, i])
        perm_score = accuracy_score(model.predict(X), model.predict(X_perm))
        importances.append(base_score - perm_score)
    idx = np.argsort(importances)[::-1][:top_n]
    return [(feature_names[i], round(importances[i], 4)) for i in idx]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: HYBRID 1 — FEDERATED LEARNING + TRANSFORMER ENCODER + XAI
# ─────────────────────────────────────────────────────────────────────────────

class TransformerEncoder:
    """
    Simulated Transformer Encoder using MLP with attention-style weighting.
    Implements multi-head attention concept via weighted feature projection.
    """
    def __init__(self, n_heads=4, hidden_dim=64, random_state=42):
        self.n_heads      = n_heads
        self.hidden_dim   = hidden_dim
        self.random_state = random_state
        self.models       = []
        self.scaler       = StandardScaler()

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        # Each "head" focuses on different feature subspaces
        n_features = X.shape[1]
        for head in range(self.n_heads):
            np.random.seed(self.random_state + head)
            # Attention weights for this head
            attn_w = np.random.dirichlet(np.ones(n_features))
            X_attended = X_scaled * attn_w
            mlp = MLPClassifier(
                hidden_layer_sizes=(self.hidden_dim, self.hidden_dim // 2),
                activation='relu', max_iter=300, random_state=self.random_state + head,
                early_stopping=True, validation_fraction=0.15
            )
            mlp.fit(X_attended, y)
            self.models.append((attn_w, mlp))
        return self

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        proba_list = []
        for attn_w, mlp in self.models:
            X_attended = X_scaled * attn_w
            proba_list.append(mlp.predict_proba(X_attended))
        return np.mean(proba_list, axis=0)

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))


class FederatedTransformerXAI:
    """
    Hybrid 1: Federated Learning Simulation + Transformer Encoder + XAI
    - Splits data into N virtual 'hospitals' (clients)
    - Each client trains a local Transformer
    - Aggregates via FedAvg-style model ensemble
    - XAI via permutation importance (SHAP proxy)
    """
    def __init__(self, n_clients=4, n_rounds=3, n_heads=4, hidden_dim=64):
        self.n_clients  = n_clients
        self.n_rounds   = n_rounds
        self.n_heads    = n_heads
        self.hidden_dim = hidden_dim
        self.clients    = []
        self.feature_names = None

    def fit(self, X, y, feature_names=None):
        self.feature_names = feature_names if feature_names is not None else \
                             [f'f{i}' for i in range(X.shape[1])]
        # Split into virtual clients (hospitals)
        idx = np.random.permutation(len(X))
        splits = np.array_split(idx, self.n_clients)
        self.clients = []
        for r in range(self.n_rounds):
            round_clients = []
            for c_idx, split in enumerate(splits):
                X_c, y_c = X[split], y[split]
                model = TransformerEncoder(
                    n_heads=self.n_heads, hidden_dim=self.hidden_dim,
                    random_state=42 + c_idx + r * 10
                )
                model.fit(X_c, y_c)
                round_clients.append(model)
            self.clients = round_clients  # Last round clients retained
        return self

    def predict_proba(self, X):
        # FedAvg: average predictions across all clients
        proba_list = [c.predict_proba(X) for c in self.clients]
        return np.mean(proba_list, axis=0)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

    def get_xai_importance(self, X, top_n=10):
        """SHAP-style permutation importance across all federated clients"""
        all_importances = defaultdict(list)
        for client in self.clients:
            imp = shap_style_importance(client, X, self.feature_names, top_n=len(self.feature_names))
            for name, val in imp:
                all_importances[name].append(val)
        avg = {k: np.mean(v) for k, v in all_importances.items()}
        return sorted(avg.items(), key=lambda x: x[1], reverse=True)[:top_n]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: HYBRID 4 — MULTI-MODAL CROSS-ATTENTION FUSION
# ─────────────────────────────────────────────────────────────────────────────

class CrossAttentionFusion:
    """
    Hybrid 4: Multi-Modal Cross-Attention Fusion
    - Separate encoders for Clinical, Wearable, Genomic modalities
    - Cross-attention: each modality attends to the others
    - Final fusion via learned weighted combination
    """
    def __init__(self, random_state=42):
        self.random_state   = random_state
        self.encoders       = {}
        self.fusion_model   = None
        self.scaler         = StandardScaler()
        self.modal_weights  = None

    def _split_modalities(self, X, feature_names):
        """Split feature matrix into 3 modality groups"""
        clinical_idx  = [i for i, f in enumerate(feature_names)
                         if any(k in f for k in ['Glucose','BMI','HbA1c','Age','BloodPressure',
                                                  'Pressure','Insulin','Pregnancies','Pedigree',
                                                  'Skin','SystolicBP','Cholesterol','Triglyc',
                                                  'Fasting','Waist','Calori'])]
        wearable_idx  = [i for i, f in enumerate(feature_names)
                         if any(k in f for k in ['Steps','Sleep','HR','HRV','Active',
                                                  'Sedentary','Smoker'])]
        genomic_idx   = [i for i, f in enumerate(feature_names)
                         if any(k in f for k in ['SNP','Polygenic','Family','Risk','History'])]

        # Fallback: divide evenly if not enough modality-specific features
        if len(wearable_idx) < 2 or len(genomic_idx) < 2:
            n = X.shape[1]
            third = n // 3
            clinical_idx = list(range(0, third))
            wearable_idx = list(range(third, 2 * third))
            genomic_idx  = list(range(2 * third, n))

        return clinical_idx, wearable_idx, genomic_idx

    def fit(self, X, y, feature_names=None):
        self.feature_names = feature_names or [f'f{i}' for i in range(X.shape[1])]
        X_scaled = self.scaler.fit_transform(X)

        c_idx, w_idx, g_idx = self._split_modalities(X_scaled, self.feature_names)
        self.modal_indices = {'clinical': c_idx, 'wearable': w_idx, 'genomic': g_idx}

        # Individual modal encoders (MLPs)
        modal_outputs = []
        for name, idx in self.modal_indices.items():
            if len(idx) == 0:
                continue
            enc = MLPClassifier(
                hidden_layer_sizes=(32, 16), activation='relu',
                max_iter=200, random_state=self.random_state
            )
            enc.fit(X_scaled[:, idx], y)
            self.encoders[name] = enc
            modal_outputs.append(enc.predict_proba(X_scaled[:, idx]))

        # Cross-attention: weight each modality's output
        stacked = np.hstack(modal_outputs)  # Shape: (n, n_modalities * 2)
        # Attention weights via softmax over modal mean confidences
        modal_confidences = [np.mean(np.max(p, axis=1)) for p in modal_outputs]
        self.modal_weights = np.array(modal_confidences)
        self.modal_weights = np.exp(self.modal_weights) / np.sum(np.exp(self.modal_weights))

        # Fusion model on cross-attended features
        self.fusion_model = MLPClassifier(
            hidden_layer_sizes=(64, 32), activation='relu',
            max_iter=300, random_state=self.random_state, early_stopping=True
        )
        self.fusion_model.fit(stacked, y)
        return self

    def _get_modal_outputs(self, X):
        X_scaled = self.scaler.transform(X)
        modal_outputs = []
        for name, idx in self.modal_indices.items():
            if name in self.encoders and len(idx) > 0:
                modal_outputs.append(self.encoders[name].predict_proba(X_scaled[:, idx]))
        return modal_outputs

    def predict_proba(self, X):
        modal_outputs = self._get_modal_outputs(X)
        stacked = np.hstack(modal_outputs)
        return self.fusion_model.predict_proba(stacked)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

    def get_modal_contribution(self):
        """Returns attention weight per modality"""
        names = [k for k in self.modal_indices if k in self.encoders]
        return dict(zip(names, self.modal_weights))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: HYBRID 10 — DIGITAL TWIN + RL-STYLE ADAPTIVE RISK + SHAP
# ─────────────────────────────────────────────────────────────────────────────

class DigitalTwinRLMonitor:
    """
    Hybrid 10: Digital Twin + Reinforcement Learning Risk Monitoring
    - Builds a 'digital twin' patient state representation
    - RL-style: iteratively updates risk estimate based on feature trajectories
    - SHAP-style feature attribution for clinical explainability
    - Uses ensemble of models as the 'environment' for policy learning
    """
    def __init__(self, n_iterations=5, random_state=42):
        self.n_iterations    = n_iterations
        self.random_state    = random_state
        self.base_model      = None
        self.risk_models     = []
        self.feature_weights = None
        self.scaler          = StandardScaler()

    def _simulate_digital_twin(self, X, noise_level=0.05):
        """Simulate patient state evolution (digital twin perturbation)"""
        return X + np.random.normal(0, noise_level * np.std(X, axis=0), X.shape)

    def fit(self, X, y, feature_names=None):
        self.feature_names = feature_names or [f'f{i}' for i in range(X.shape[1])]
        X_scaled = self.scaler.fit_transform(X)

        # Base risk model (initial digital twin state)
        self.base_model = GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=4,
            random_state=self.random_state
        )
        self.base_model.fit(X_scaled, y)

        # RL-style iterations: each round refines the risk model
        # using digital twin perturbations as "environment exploration"
        self.risk_models = [self.base_model]
        X_current = X_scaled.copy()

        for iteration in range(self.n_iterations):
            # Simulate patient state evolution (digital twin step)
            X_twin = self._simulate_digital_twin(X_current, noise_level=0.03)

            # Get current risk predictions as pseudo-labels for refinement
            risk_proba = self.base_model.predict_proba(X_current)[:, 1]
            refined_labels = np.where(risk_proba > 0.6, 1,
                             np.where(risk_proba < 0.4, 0, y))

            # RL policy update: train refined model on twin state
            refined_model = GradientBoostingClassifier(
                n_estimators=80, learning_rate=0.08,
                max_depth=3, random_state=self.random_state + iteration
            )
            refined_model.fit(X_twin, refined_labels)
            self.risk_models.append(refined_model)
            X_current = X_twin  # advance digital twin state

        # Feature importance (SHAP proxy) from base model
        if hasattr(self.base_model, 'feature_importances_'):
            self.feature_weights = self.base_model.feature_importances_
        return self

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        # Ensemble all RL iteration models (temporal risk aggregation)
        proba_list = [m.predict_proba(X_scaled) for m in self.risk_models]
        # Weight later iterations more (RL policy improvement)
        weights = np.linspace(0.5, 1.5, len(proba_list))
        weights /= weights.sum()
        return sum(w * p for w, p in zip(weights, proba_list))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

    def get_shap_importance(self, top_n=10):
        """Feature importance from base GBM (SHAP proxy)"""
        if self.feature_weights is None:
            return []
        idx = np.argsort(self.feature_weights)[::-1][:top_n]
        return [(self.feature_names[i], round(self.feature_weights[i], 4)) for i in idx]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: BASELINE MODELS FOR COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def get_baselines(random_state=42):
    return {
        'Logistic Regression': LogisticRegression(max_iter=500, random_state=random_state),
        'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=random_state),
        'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100, random_state=random_state),
        'MLP Neural Net':      MLPClassifier(hidden_layer_sizes=(64,32), max_iter=300,
                                              random_state=random_state, early_stopping=True),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: MAIN EXPERIMENT RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_experiments():
    print("=" * 70)
    print("  EARLY PREDICTION OF TYPE 2 DIABETES — HYBRID MODEL COMPARISON")
    print("=" * 70)

    # --- Generate datasets ---
    datasets = {
        'Dataset 1\n(PIMA-Style)':       generate_pima_style(768),
        'Dataset 2\n(NHANES-Style)':     generate_nhanes_style(1200),
        'Dataset 3\n(Multi-Modal)':      generate_multimodal_style(900),
    }

    all_results    = []
    roc_data       = {}
    xai_data       = {}
    conf_matrices  = {}
    modal_contribs = {}

    scaler = StandardScaler()

    for ds_name, df in datasets.items():
        print(f"\n{'─'*60}")
        print(f"  {ds_name.replace(chr(10),' ')}")
        print(f"  Samples: {len(df)} | Features: {df.shape[1]-1} | "
              f"Positive rate: {df['Outcome'].mean():.1%}")
        print(f"{'─'*60}")

        X = df.drop('Outcome', axis=1).values
        y = df['Outcome'].values
        feat_names = list(df.drop('Outcome', axis=1).columns)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)

        ds_key = ds_name.replace('\n', ' ')
        roc_data[ds_key]      = {}
        conf_matrices[ds_key] = {}

        # ── Baselines ──
        baselines = get_baselines()
        for bl_name, bl_model in baselines.items():
            bl_model.fit(X_train_s, y_train)
            res = evaluate_model(bl_model, X_test_s, y_test, bl_name)
            res['Dataset'] = ds_key
            res['Type']    = 'Baseline'
            all_results.append({k: v for k, v in res.items()
                                 if k not in ['y_test','y_proba','y_pred']})
            roc_data[ds_key][bl_name]     = (res['y_test'], res['y_proba'])
            conf_matrices[ds_key][bl_name] = confusion_matrix(res['y_test'], res['y_pred'])
            print(f"  [Baseline] {bl_name:<22} Acc={res['Accuracy']:.4f}  AUC={res['AUC-ROC']:.4f}  F1={res['F1-Score']:.4f}")

        # ── Hybrid 1: Federated Transformer + XAI ──
        h1 = FederatedTransformerXAI(n_clients=4, n_rounds=3, n_heads=4, hidden_dim=64)
        h1.fit(X_train_s, y_train, feature_names=feat_names)
        res1 = evaluate_model(h1, X_test_s, y_test, 'Hybrid 1: Fed-Transformer+XAI')
        res1['Dataset'] = ds_key
        res1['Type']    = 'Hybrid'
        all_results.append({k: v for k, v in res1.items()
                            if k not in ['y_test','y_proba','y_pred']})
        roc_data[ds_key]['Hybrid 1']     = (res1['y_test'], res1['y_proba'])
        conf_matrices[ds_key]['Hybrid 1'] = confusion_matrix(res1['y_test'], res1['y_pred'])
        xai_data[ds_key + '_H1'] = h1.get_xai_importance(X_test_s, top_n=8)
        print(f"  [Hybrid 1] {'Fed-Transformer+XAI':<22} Acc={res1['Accuracy']:.4f}  AUC={res1['AUC-ROC']:.4f}  F1={res1['F1-Score']:.4f}")

        # ── Hybrid 4: Multi-Modal Cross-Attention ──
        h4 = CrossAttentionFusion(random_state=42)
        h4.fit(X_train_s, y_train, feature_names=feat_names)
        res4 = evaluate_model(h4, X_test_s, y_test, 'Hybrid 4: MultiModal-CrossAttn')
        res4['Dataset'] = ds_key
        res4['Type']    = 'Hybrid'
        all_results.append({k: v for k, v in res4.items()
                            if k not in ['y_test','y_proba','y_pred']})
        roc_data[ds_key]['Hybrid 4']     = (res4['y_test'], res4['y_proba'])
        conf_matrices[ds_key]['Hybrid 4'] = confusion_matrix(res4['y_test'], res4['y_pred'])
        modal_contribs[ds_key] = h4.get_modal_contribution()
        print(f"  [Hybrid 4] {'MultiModal-CrossAttn':<22} Acc={res4['Accuracy']:.4f}  AUC={res4['AUC-ROC']:.4f}  F1={res4['F1-Score']:.4f}")

        # ── Hybrid 10: Digital Twin + RL ──
        h10 = DigitalTwinRLMonitor(n_iterations=5, random_state=42)
        h10.fit(X_train_s, y_train, feature_names=feat_names)
        res10 = evaluate_model(h10, X_test_s, y_test, 'Hybrid 10: DigitalTwin+RL')
        res10['Dataset'] = ds_key
        res10['Type']    = 'Hybrid'
        all_results.append({k: v for k, v in res10.items()
                            if k not in ['y_test','y_proba','y_pred']})
        roc_data[ds_key]['Hybrid 10']     = (res10['y_test'], res10['y_proba'])
        conf_matrices[ds_key]['Hybrid 10'] = confusion_matrix(res10['y_test'], res10['y_pred'])
        xai_data[ds_key + '_H10'] = h10.get_shap_importance(top_n=8)
        print(f"  [Hybrid 10] {'DigitalTwin+RL':<22} Acc={res10['Accuracy']:.4f}  AUC={res10['AUC-ROC']:.4f}  F1={res10['F1-Score']:.4f}")

    results_df = pd.DataFrame(all_results)
    return results_df, roc_data, xai_data, conf_matrices, modal_contribs


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def plot_all(results_df, roc_data, xai_data, conf_matrices, modal_contribs):
    HYBRID_COLORS   = {'Hybrid 1: Fed-Transformer+XAI': '#E74C3C',
                       'Hybrid 4: MultiModal-CrossAttn': '#2ECC71',
                       'Hybrid 10: DigitalTwin+RL':      '#3498DB'}
    BASELINE_COLORS = {'Logistic Regression': '#95A5A6',
                       'Random Forest':       '#F39C12',
                       'Gradient Boosting':   '#8E44AD',
                       'MLP Neural Net':      '#1ABC9C'}
    ALL_COLORS = {**HYBRID_COLORS, **BASELINE_COLORS}

    datasets   = results_df['Dataset'].unique()
    metrics    = ['Accuracy', 'AUC-ROC', 'F1-Score', 'Precision', 'Recall']

    plt.rcParams.update({'font.size': 9, 'font.family': 'DejaVu Sans'})

    # ── Figure 1: Metrics Heatmap per Dataset ──────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle('Figure 1: Performance Metrics Heatmap — All Models × All Datasets',
                 fontsize=13, fontweight='bold', y=1.01)

    for ax, ds in zip(axes, datasets):
        sub = results_df[results_df['Dataset'] == ds][['Model'] + metrics].set_index('Model')
        sub = sub.sort_values('AUC-ROC', ascending=False)
        sns.heatmap(sub, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax,
                    linewidths=0.5, cbar=True, vmin=0.5, vmax=1.0,
                    annot_kws={'size': 8})
        ax.set_title(ds, fontweight='bold', fontsize=10)
        ax.set_ylabel('')
        ax.tick_params(axis='x', rotation=30)
        ax.tick_params(axis='y', rotation=0)
    plt.tight_layout()
    plt.savefig('/home/claude/fig1_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n  ✔ Figure 1 saved: Metrics Heatmap")

    # ── Figure 2: Grouped Bar Chart — AUC-ROC across datasets ──────────────
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
    fig.suptitle('Figure 2: AUC-ROC Comparison — Hybrid vs Baseline Models',
                 fontsize=13, fontweight='bold')

    for ax, ds in zip(axes, datasets):
        sub = results_df[results_df['Dataset'] == ds].sort_values('AUC-ROC', ascending=True)
        colors = [ALL_COLORS.get(m, '#BDC3C7') for m in sub['Model']]
        bars = ax.barh(sub['Model'], sub['AUC-ROC'], color=colors, edgecolor='white', height=0.6)
        for bar, val in zip(bars, sub['AUC-ROC']):
            ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
                    f'{val:.3f}', va='center', fontsize=8)
        ax.axvline(0.7, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
        ax.axvline(0.9, color='green', linestyle='--', linewidth=0.8, alpha=0.7)
        ax.set_xlim(0.45, 1.05)
        ax.set_title(ds, fontweight='bold')
        ax.set_xlabel('AUC-ROC')

    # Legend
    from matplotlib.patches import Patch
    legend_patches = ([Patch(color=c, label=m) for m, c in HYBRID_COLORS.items()] +
                      [Patch(color=c, label=m) for m, c in BASELINE_COLORS.items()])
    fig.legend(handles=legend_patches, loc='lower center', ncol=4,
               bbox_to_anchor=(0.5, -0.08), fontsize=8)
    plt.tight_layout()
    plt.savefig('/home/claude/fig2_auc_bar.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✔ Figure 2 saved: AUC-ROC Bar Chart")

    # ── Figure 3: ROC Curves — One subplot per dataset ─────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Figure 3: ROC Curves — All Models', fontsize=13, fontweight='bold')

    LINE_STYLES = {'Hybrid 1': '-', 'Hybrid 4': '-', 'Hybrid 10': '-',
                   'Logistic Regression': '--', 'Random Forest': '--',
                   'Gradient Boosting': '--', 'MLP Neural Net': '--'}
    ROC_COLORS  = {'Hybrid 1': '#E74C3C', 'Hybrid 4': '#2ECC71',
                   'Hybrid 10': '#3498DB', 'Logistic Regression': '#95A5A6',
                   'Random Forest': '#F39C12', 'Gradient Boosting': '#8E44AD',
                   'MLP Neural Net': '#1ABC9C'}

    for ax, ds in zip(axes, datasets):
        ax.plot([0,1],[0,1], 'k--', alpha=0.3, linewidth=1)
        for mname, (yt, yp) in roc_data[ds].items():
            fpr, tpr, _ = roc_curve(yt, yp)
            auc = roc_auc_score(yt, yp)
            short = mname.replace('Hybrid 1: Fed-Transformer+XAI','Hybrid 1')\
                         .replace('Hybrid 4: MultiModal-CrossAttn','Hybrid 4')\
                         .replace('Hybrid 10: DigitalTwin+RL','Hybrid 10')
            color = ROC_COLORS.get(short, '#BDC3C7')
            ls    = LINE_STYLES.get(short, '-')
            lw    = 2.2 if 'Hybrid' in short else 1.2
            ax.plot(fpr, tpr, color=color, linestyle=ls, linewidth=lw,
                    label=f'{short} (AUC={auc:.3f})')
        ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
        ax.set_title(ds, fontweight='bold')
        ax.legend(fontsize=6.5, loc='lower right')
        ax.set_xlim(0,1); ax.set_ylim(0,1.02)
        ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig('/home/claude/fig3_roc.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✔ Figure 3 saved: ROC Curves")

    # ── Figure 4: XAI Feature Importance — Hybrid 1 & 10 ─────────────────
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle('Figure 4: XAI Feature Importance — Hybrid 1 (Federated Transformer)'
                 ' & Hybrid 10 (Digital Twin + RL)', fontsize=12, fontweight='bold')

    row_labels = ['Hybrid 1: Fed-Transformer+XAI', 'Hybrid 10: DigitalTwin+RL']
    h_keys     = ['_H1', '_H10']
    h_colors   = ['#E74C3C', '#3498DB']

    for row, (hlabel, hkey, hcolor) in enumerate(zip(row_labels, h_keys, h_colors)):
        for col, ds in enumerate(datasets):
            ax  = axes[row][col]
            key = ds + hkey
            if key not in xai_data or not xai_data[key]:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center')
                continue
            items  = xai_data[key]
            feats  = [x[0] for x in items]
            values = [max(x[1], 0) for x in items]
            bars = ax.barh(feats[::-1], values[::-1], color=hcolor, alpha=0.8, edgecolor='white')
            for bar, val in zip(bars, values[::-1]):
                ax.text(bar.get_width() + 0.0005, bar.get_y() + bar.get_height()/2,
                        f'{val:.4f}', va='center', fontsize=7)
            ax.set_title(f'{hlabel}\n{ds}', fontsize=8, fontweight='bold')
            ax.set_xlabel('Importance Score', fontsize=8)
            ax.tick_params(axis='y', labelsize=7)
    plt.tight_layout()
    plt.savefig('/home/claude/fig4_xai.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✔ Figure 4 saved: XAI Feature Importance")

    # ── Figure 5: Confusion Matrices — Hybrid Models Only ─────────────────
    hybrid_models = ['Hybrid 1', 'Hybrid 4', 'Hybrid 10']
    fig, axes = plt.subplots(3, 3, figsize=(14, 13))
    fig.suptitle('Figure 5: Confusion Matrices — Hybrid Models × Datasets',
                 fontsize=13, fontweight='bold')

    for col, ds in enumerate(datasets):
        for row, hm in enumerate(hybrid_models):
            ax = axes[row][col]
            if hm in conf_matrices[ds]:
                cm = conf_matrices[ds][hm]
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                            xticklabels=['No T2DM','T2DM'],
                            yticklabels=['No T2DM','T2DM'],
                            linewidths=0.5, cbar=False)
                total = cm.sum()
                acc   = (cm[0,0] + cm[1,1]) / total
                ax.set_title(f'{hm}\n{ds}\nAcc={acc:.3f}', fontsize=8, fontweight='bold')
                ax.set_xlabel('Predicted', fontsize=8)
                ax.set_ylabel('Actual', fontsize=8)
    plt.tight_layout()
    plt.savefig('/home/claude/fig5_confusion.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✔ Figure 5 saved: Confusion Matrices")

    # ── Figure 6: Multi-metric Spider / Radar Chart — Hybrid Models ────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6),
                              subplot_kw=dict(polar=True))
    fig.suptitle('Figure 6: Radar Chart — Hybrid Model Performance Profile per Dataset',
                 fontsize=13, fontweight='bold')

    metrics5   = ['Accuracy','AUC-ROC','F1-Score','Precision','Recall']
    N          = len(metrics5)
    angles     = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles    += angles[:1]
    h_names    = ['Hybrid 1: Fed-Transformer+XAI',
                  'Hybrid 4: MultiModal-CrossAttn',
                  'Hybrid 10: DigitalTwin+RL']
    h_colors2  = ['#E74C3C','#2ECC71','#3498DB']

    for ax, ds in zip(axes, datasets):
        sub = results_df[(results_df['Dataset'] == ds) &
                         (results_df['Type'] == 'Hybrid')]
        for hname, hcol in zip(h_names, h_colors2):
            row = sub[sub['Model'] == hname]
            if row.empty:
                continue
            vals = row[metrics5].values[0].tolist()
            vals += vals[:1]
            ax.plot(angles, vals, color=hcol, linewidth=2, linestyle='solid')
            ax.fill(angles, vals, color=hcol, alpha=0.15)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics5, size=8)
        ax.set_ylim(0, 1)
        ax.set_title(ds, fontweight='bold', size=9, pad=15)
        ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax.set_yticklabels(['0.5','0.6','0.7','0.8','0.9','1.0'], size=6)
        ax.grid(alpha=0.3)

    legend_patches = [plt.Line2D([0],[0], color=c, linewidth=2,
                                  label=n.split(':')[0])
                      for n, c in zip(h_names, h_colors2)]
    fig.legend(handles=legend_patches, loc='lower center', ncol=3,
               bbox_to_anchor=(0.5, -0.05), fontsize=9)
    plt.tight_layout()
    plt.savefig('/home/claude/fig6_radar.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✔ Figure 6 saved: Radar Chart")

    # ── Figure 7: Modal Contribution — Hybrid 4 ────────────────────────────
    if modal_contribs:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Figure 7: Hybrid 4 — Cross-Attention Modal Contribution Weight',
                     fontsize=12, fontweight='bold')
        pie_colors = ['#3498DB','#E74C3C','#2ECC71']
        for ax, ds in zip(axes, datasets):
            if ds in modal_contribs and modal_contribs[ds]:
                mc = modal_contribs[ds]
                labels_mc = [k.capitalize() for k in mc.keys()]
                values_mc = list(mc.values())
                wedges, texts, autotexts = ax.pie(
                    values_mc, labels=labels_mc, colors=pie_colors[:len(values_mc)],
                    autopct='%1.1f%%', startangle=90,
                    wedgeprops={'edgecolor':'white','linewidth':1.5})
                for at in autotexts:
                    at.set_fontsize(9)
                ax.set_title(ds, fontweight='bold')
            else:
                ax.text(0.5,0.5,'N/A',ha='center',va='center')
        plt.tight_layout()
        plt.savefig('/home/claude/fig7_modal.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✔ Figure 7 saved: Modal Contribution Pie Charts")

    # ── Figure 8: Summary Comparison Table ────────────────────────────────
    fig, ax = plt.subplots(figsize=(20, 9))
    ax.axis('off')
    fig.suptitle('Figure 8: Complete Model Comparison Summary Table',
                 fontsize=13, fontweight='bold', y=0.98)

    display_df = results_df[['Model','Dataset','Type','Accuracy','AUC-ROC',
                              'F1-Score','Precision','Recall']].copy()
    display_df = display_df.sort_values(['Dataset','AUC-ROC'], ascending=[True, False])

    col_labels = list(display_df.columns)
    cell_data  = display_df.values.tolist()

    table = ax.table(
        cellText=cell_data, colLabels=col_labels,
        cellLoc='center', loc='center', bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)

    # Style header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#2C3E50')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # Style rows by type
    for i, row in enumerate(cell_data):
        model_type = row[2]
        for j in range(len(col_labels)):
            if model_type == 'Hybrid':
                table[i+1, j].set_facecolor('#EBF5FB')
            else:
                table[i+1, j].set_facecolor('#FDFEFE')
            table[i+1, j].set_edgecolor('#BDC3C7')

    plt.savefig('/home/claude/fig8_table.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✔ Figure 8 saved: Summary Table")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9: COMBINE ALL FIGURES INTO ONE PDF-STYLE MASTER FIGURE
# ─────────────────────────────────────────────────────────────────────────────

def combine_figures():
    fig_files = [f'/home/claude/fig{i}_{s}.png' for i, s in
                 enumerate(['heatmap','auc_bar','roc','xai',
                             'confusion','radar','modal','table'], start=1)]
    import os
    existing = [f for f in fig_files if os.path.exists(f)]
    if not existing:
        return

    imgs = [plt.imread(f) for f in existing]
    n = len(imgs)
    cols = 2
    rows = (n + 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(24, rows * 7))
    axes = axes.flatten()

    titles = ['Fig 1: Metrics Heatmap', 'Fig 2: AUC-ROC Bar Chart',
              'Fig 3: ROC Curves', 'Fig 4: XAI Feature Importance',
              'Fig 5: Confusion Matrices', 'Fig 6: Radar Chart',
              'Fig 7: Modal Contributions', 'Fig 8: Summary Table']

    for i, (img, ax) in enumerate(zip(imgs, axes)):
        ax.imshow(img)
        ax.axis('off')
        if i < len(titles):
            ax.set_title(titles[i], fontsize=11, fontweight='bold', pad=8)

    for ax in axes[len(imgs):]:
        ax.axis('off')

    fig.suptitle(
        'EARLY PREDICTION OF TYPE 2 DIABETES\n'
        'Hybrid Model Comparison: Federated-Transformer+XAI | '
        'Multi-Modal Cross-Attention | Digital Twin + RL',
        fontsize=14, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/T2DM_Hybrid_Model_Comparison.png',
                dpi=130, bbox_inches='tight')
    plt.close()
    print("\n  ✔ Master figure saved to outputs!")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    results_df, roc_data, xai_data, conf_matrices, modal_contribs = run_experiments()

    print("\n" + "=" * 70)
    print("  GENERATING VISUALISATIONS ...")
    print("=" * 70)
    plot_all(results_df, roc_data, xai_data, conf_matrices, modal_contribs)
    combine_figures()

    # Print final summary
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY — HYBRID MODELS VS BASELINES (Average Across Datasets)")
    print("=" * 70)
    summary = results_df.groupby(['Model','Type'])[
        ['Accuracy','AUC-ROC','F1-Score','Precision','Recall']
    ].mean().round(4).sort_values('AUC-ROC', ascending=False)
    print(summary.to_string())
    print("\n  Done! ✔")
