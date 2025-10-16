import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
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
    f1 = f1_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    metrics = {"Accuracy": acc, "F1-Score": f1, "Recall": rec}
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

def knn_classify(df: pd.DataFrame) -> object:
    X_train, X_test, y_train, y_test = prepare_dataset(df)
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    visualize_metrics("KNN", y_test, y_pred)
    return model

def svm_classify(df: pd.DataFrame) -> object:
    X_train, X_test, y_train, y_test = prepare_dataset(df)
    model = SVC(kernel='rbf', probability=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    visualize_metrics("SVM", y_test, y_pred)
    return model

def decision_tree_classify(df: pd.DataFrame) -> object:
    X_train, X_test, y_train, y_test = prepare_dataset(df)
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    visualize_metrics("DecisionTree", y_test, y_pred)
    return model

def random_forest_classify(df: pd.DataFrame) -> object:
    X_train, X_test, y_train, y_test = prepare_dataset(df)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    visualize_metrics("RandomForest", y_test, y_pred)
    return model

def mlp_classify(df: pd.DataFrame) -> object:
    X_train, X_test, y_train, y_test = prepare_dataset(df)
    model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
    history = {"loss": []}
    for i in range(model.max_iter):
        model.partial_fit(X_train, y_train, classes=np.unique(y_train))
        history["loss"].append(model.loss_)
    y_pred = model.predict(X_test)
    visualize_metrics("MLP", y_test, y_pred, history)
    return model

def classify_datasets() -> None:
    classifiers = [knn_classify, svm_classify, decision_tree_classify, random_forest_classify, mlp_classify]
    dataset_paths = [HU_MASK_DF_PATH, HU_ORIGINAL_DF_PATH, LBP_MASK_DATASET, LBP_ORIGINAL_DATASET, GLCM_MASK_DF_PATH, GLCM_ORIGINAL_DF_PATH]
    os.makedirs(VISUALIZATION_PATH, exist_ok=True)
    for classifier, dataset in zip(classifiers, dataset_paths):
        df = pd.read_pickle(os.path.join(DATASETS_PATH, dataset))
        print(df.head)
        classifier(df)

if __name__ == "__main__":
    classify_datasets()
