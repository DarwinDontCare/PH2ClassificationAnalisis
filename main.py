from build_dataset import build_datasets
from classify_data import classify_datasets, create_comparison_heatmap

if __name__ == "__main__":
    build_datasets()
    classify_datasets()
    create_comparison_heatmap("Accuracy")
