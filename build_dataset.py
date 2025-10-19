import pandas as pd
from data_extraction import extract_hu_moments, extract_LBP, extract_GLCM
import kagglehub
import os

OUTPUT_PATH = "datasets"

def load_ph2_dataset() -> pd.DataFrame:
    dataset_file_path = os.path.join(OUTPUT_PATH, "ph2_dataset.pkl")

    if not os.path.exists(dataset_file_path):
        path = kagglehub.dataset_download("spacesurfer/ph2-dataset")
        print(f"dataset path: {path}")

        df_path = os.path.join(path, "PH2Dataset", "PH2_dataset.txt")

        df = pd.read_csv(df_path,
            sep=r'\s*\|\|\s*',  
            engine='python',           
            skipinitialspace=True)
    
        if 'Unnamed: 0' in df.columns:
            df.drop(columns=['Unnamed: 0'], inplace=True)

        df.dropna(axis=1, how='all', inplace=True)
        print("samples: ", df.value_counts("Name"))
        df.columns = [col.strip() for col in df.columns]

        df = df[df['Name'].astype(str).str.contains('IMD', na=False)].reset_index(drop=True)
        df['Name'] = df['Name'].str.strip()

        df["Image"] = [""] * len(df["Name"])
        df["Mask"] = [""] * len(df["Name"])

        images_path = os.path.join(path, "PH2Dataset", "PH2 Dataset images")
        for i, name in enumerate(df["Name"]):
            if isinstance(name, str) and name != '':
                df.loc[i, "Image"] = os.path.join(images_path, name, f"{name}_Dermoscopic_Image", f"{name}.bmp")
                df.loc[i, "Mask"] = os.path.join(images_path, name, f"{name}_lesion", f"{name}_lesion.bmp")

        df["Classification"] = pd.to_numeric(df["Clinical Diagnosis"], errors='coerce').astype(int)

        df.to_pickle(dataset_file_path)
        return df
    else:
        return pd.read_pickle(dataset_file_path)

def generate_hu_moments_df(ph2_df: pd.DataFrame) -> None:
    hu_original_path = os.path.join(OUTPUT_PATH, "hu_original_dataset.pkl")
    hu_mask_path = os.path.join(OUTPUT_PATH, "hu_mask_dataset.pkl")

    if os.path.exists(hu_original_path) and os.path.exists(hu_mask_path):
        return

    hu_original_df = pd.DataFrame(columns=["Classification", "Values"])
    hu_mask_df = pd.DataFrame(columns=["Classification", "Values"])

    for classification, image, mask in zip(ph2_df["Clinical Diagnosis"], ph2_df["Image"], ph2_df["Mask"]):
        original_values = extract_hu_moments(image)
        mask_values = extract_hu_moments(mask)
        if original_values is not None:
            hu_original_df.loc[len(hu_original_df)] = [classification, original_values]
        if mask_values is not None:
            hu_mask_df.loc[len(hu_mask_df)] = [classification, mask_values]
    
    hu_original_df.to_pickle(hu_original_path)
    hu_mask_df.to_pickle(hu_mask_path)

def generate_LBP_df(ph2_df: pd.DataFrame) -> None:
    LBP_original_path = os.path.join(OUTPUT_PATH, "LBP_original_dataset.pkl")
    LBP_mask_path = os.path.join(OUTPUT_PATH, "LBP_mask_dataset.pkl")

    if os.path.exists(LBP_original_path) and os.path.exists(LBP_mask_path):
        return

    LBP_original_df = pd.DataFrame(columns=["Classification", "Values"])
    LBP_mask_df = pd.DataFrame(columns=["Classification", "Values"])

    for classification, image, mask in zip(ph2_df["Clinical Diagnosis"], ph2_df["Image"], ph2_df["Mask"]):
        original_values = extract_LBP(image)
        mask_values = extract_LBP(mask)
        if original_values is not None:
            LBP_original_df.loc[len(LBP_original_df)] = [classification, original_values.flatten()]
        if mask_values is not None:
            LBP_mask_df.loc[len(LBP_mask_df)] = [classification, mask_values.flatten()]
    
    LBP_original_df.to_pickle(LBP_original_path)
    LBP_mask_df.to_pickle(LBP_mask_path)

def generate_GLCM_df(ph2_df: pd.DataFrame) -> None:
    GLCM_original_path = os.path.join(OUTPUT_PATH, "GLCM_original_dataset.pkl")
    GLCM_mask_path = os.path.join(OUTPUT_PATH, "GLCM_mask_dataset.pkl")

    if os.path.exists(GLCM_original_path) and os.path.exists(GLCM_mask_path):
        return

    GLCM_original_df = pd.DataFrame(columns=["Classification", "Values"])
    GLCM_mask_df = pd.DataFrame(columns=["Classification", "Values"])

    for classification, image, mask in zip(ph2_df["Clinical Diagnosis"], ph2_df["Image"], ph2_df["Mask"]):
        original_values = extract_GLCM(image)
        mask_values = extract_GLCM(mask)
        if original_values is not None:
            GLCM_original_df.loc[len(GLCM_original_df)] = [classification, original_values]
        if mask_values is not None:
            GLCM_mask_df.loc[len(GLCM_mask_df)] = [classification, mask_values]
    
    GLCM_original_df.to_pickle(GLCM_original_path)
    GLCM_mask_df.to_pickle(GLCM_mask_path)

def build_datasets() -> None:
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    df = load_ph2_dataset()
    generate_hu_moments_df(df)
    generate_GLCM_df(df)
    generate_LBP_df(df)

if __name__ == "__main__":
    build_datasets()
