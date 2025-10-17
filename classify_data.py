import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import os

DATASETS_PATH = "datasets"
VISUALIZATION_PATH = "visualizations"

HU_ORIGINAL_DF_PATH = "hu_original_dataset.pkl"
HU_MASK_DF_PATH = "hu_mask_dataset.pkl"
GLCM_ORIGINAL_DF_PATH = "GLCM_original_dataset.pkl"
GLCM_MASK_DF_PATH = "GLCM_mask_dataset.pkl"
LBP_ORIGINAL_DATASET = "LBP_original_dataset.pkl"
LBP_MASK_DATASET = "LBP_mask_dataset.pkl"

def prepare_dataset(df: pd.DataFrame):
    values = df["Values"].tolist()
    max_len = max(len(v) for v in values)
    
    X = np.array([np.pad(v, (0, max_len - len(v))) for v in values])
    y = np.array(df["Classification"].tolist())
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

def visualize_metrics(name: str, y_test: np.ndarray, y_pred: np.ndarray, history: dict = None) -> None:
    os.makedirs(VISUALIZATION_PATH, exist_ok=True)
    
    
    
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    precision = precision_score(y_pred=y_pred, y_true=y_test, average='weighted', zero_division=0)
    metrics = {"Accuracy": acc, "F1-Score": f1, "Recall": rec, "Precision": precision}
    print(f"\n{name} Metrics: {metrics}\n")
    
    
    plt.figure()
    plt.bar(metrics.keys(), metrics.values())
    plt.title(f"{name} Metrics")
    plt.savefig(os.path.join(VISUALIZATION_PATH, f"{name}_metrics.png"))
    plt.close()
    
    
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title(f"{name} Confusion Matrix")
    plt.savefig(os.path.join(VISUALIZATION_PATH, f"{name}_confusion_matrix.png"))
    plt.close()
    
    
    if history is not None:
        plt.figure()
        plt.plot(history["loss"], label="Loss")
        plt.title(f"{name} Loss")
        plt.legend()
        plt.savefig(os.path.join(VISUALIZATION_PATH, f"{name}_loss.png"))
        plt.close()

def knn_classify(df: pd.DataFrame, name: str) -> object:
    X_train, X_test, y_train, y_test = prepare_dataset(df)
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    visualize_metrics(name, y_test, y_pred)
    return model

def svm_classify(df: pd.DataFrame, name: str) -> object:
    X_train, X_test, y_train, y_test = prepare_dataset(df)
    model = SVC(kernel='rbf', probability=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    visualize_metrics(name, y_test, y_pred)
    return model

def decision_tree_classify(df: pd.DataFrame, name: str) -> object:
    X_train, X_test, y_train, y_test = prepare_dataset(df)
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    visualize_metrics(name, y_test, y_pred)
    return model

def random_forest_classify(df: pd.DataFrame, name: str) -> object:
    X_train, X_test, y_train, y_test = prepare_dataset(df)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    visualize_metrics(name, y_test, y_pred)
    return model

def mlp_classify(df: pd.DataFrame, name: str) -> object:
    X_train, X_test, y_train, y_test = prepare_dataset(df)
    model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
    history = {"loss": []}
    
    
    for i in range(model.max_iter):
        
        if i == 0:
            model.partial_fit(X_train, y_train, classes=np.unique(y_train))
        else:
            model.partial_fit(X_train, y_train)
        history["loss"].append(model.loss_)
        
    y_pred = model.predict(X_test)
    visualize_metrics(name, y_test, y_pred, history)
    return model

def classify_datasets() -> None:
    classifier_map = {
        "KNN": knn_classify,
        "SVM": svm_classify,
        "DecisionTree": decision_tree_classify,
        "RandomForest": random_forest_classify,
        "MLP": mlp_classify
    }
    dataset_map = {
        "HU_Mask": HU_MASK_DF_PATH,
        "HU_Original": HU_ORIGINAL_DF_PATH,
        "LBP_Mask": LBP_MASK_DATASET,
        "LBP_Original": LBP_ORIGINAL_DATASET,
        "GLCM_Mask": GLCM_MASK_DF_PATH,
        "GLCM_Original": GLCM_ORIGINAL_DF_PATH,
    }
    
    os.makedirs(VISUALIZATION_PATH, exist_ok=True)
    
    for ds_name, ds_path in dataset_map.items():
        try:
            df = pd.read_pickle(os.path.join(DATASETS_PATH, ds_path))
            print(f"\n--- Processing Dataset: {ds_name} ---")
            print(df.head()) 
        except FileNotFoundError:
            print(f"Skipping {ds_name}: PKL file not found at {os.path.join(DATASETS_PATH, ds_path)}")
            continue

        for clf_name, classifier_func in classifier_map.items():
            full_name = f"{ds_name}_{clf_name}"
            classifier_func(df, full_name)


def _evaluate_classifier_on_df(estimator, df: pd.DataFrame, metric: str = "Accuracy"):
    try:
        X_train, X_test, y_train, y_test = prepare_dataset(df)
        clf = estimator
        
        
        if isinstance(clf, MLPClassifier):
            clf.fit(X_train, y_train)
        else:
            clf.fit(X_train, y_train)
            
        y_pred = clf.predict(X_test)
        
        if metric == "Accuracy":
            return accuracy_score(y_test, y_pred)
        if metric == "F1":
            return f1_score(y_test, y_pred, average='weighted', zero_division=0)
        if metric == "Recall":
            return recall_score(y_test, y_pred, average='weighted', zero_division=0)
        if metric == "Precision":
            return precision_score(y_test, y_pred, average='weighted', zero_division=0)
        return accuracy_score(y_test, y_pred)
    except Exception as e:
        print(f"Error evaluating classifier: {e}")
        return np.nan


def create_comparison_heatmap(metric: str = "Accuracy") -> None:
    os.makedirs(VISUALIZATION_PATH, exist_ok=True)
    classifier_builders = {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVM": SVC(kernel='rbf', probability=True),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        
        "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42),
    }
    dataset_map = {
        "Hu_Mask": HU_MASK_DF_PATH,
        "Hu_Original": HU_ORIGINAL_DF_PATH,
        "LBP_Mask": LBP_MASK_DATASET,
        "LBP_Original": LBP_ORIGINAL_DATASET,
        "GLCM_Mask": GLCM_MASK_DF_PATH,
        "GLCM_Original": GLCM_ORIGINAL_DF_PATH,
    }

    results = pd.DataFrame(index=list(classifier_builders.keys()), columns=list(dataset_map.keys()), dtype=float)

    for clf_name, estimator in classifier_builders.items():
        for ds_name, ds_path in dataset_map.items():
            try:
                df = pd.read_pickle(os.path.join(DATASETS_PATH, ds_path))
            except Exception:
                results.loc[clf_name, ds_name] = np.nan
                continue
            
            
            score = _evaluate_classifier_on_df(estimator, df, metric)
            results.loc[clf_name, ds_name] = score

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(results.astype(float).values, cmap='viridis', vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(results.columns)))
    ax.set_yticks(range(len(results.index)))
    ax.set_xticklabels(results.columns, rotation=45, ha='right')
    ax.set_yticklabels(results.index)
    for i in range(results.shape[0]):
        for j in range(results.shape[1]):
            val = results.iloc[i, j]
            txt = "nan" if np.isnan(val) else f"{val:.3f}"
            ax.text(j, i, txt, ha='center', va='center', color='white' if not np.isnan(val) and val < 0.6 else 'black')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(metric)
    plt.title(f"Comparação: Classificadores x Extratores ({metric})")
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_PATH, f"comparison_heatmap_{metric}.png"))
    plt.close()


if __name__ == "__main__":
    classify_datasets()
    create_comparison_heatmap("Accuracy")
    create_comparison_heatmap("F1") 