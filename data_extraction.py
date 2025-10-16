import cv2
import numpy as np
from skimage.measure import moments_hu, moments_normalized, moments_central
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

def extract_hu_moments(image_path="") -> list:
    if not valid_image(image_path):
        return None
    
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    
    image = np.array(image, dtype=np.float64)
    
    mu = moments_central(image)
    nu = moments_normalized(mu)
    hu = moments_hu(nu)
    
    return hu.tolist()

def extract_LBP(image_path="") -> np.ndarray:
    if not valid_image(image_path):
        return None
    
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    
    lbp = local_binary_pattern(image, P=8, R=1, method='uniform')
    return lbp

def extract_GLCM(image_path="") -> list:
    if not valid_image(image_path):
        return None
    
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    
    distances = [1, 2]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(image, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    
    contrast = graycoprops(glcm, 'contrast')
    energy = graycoprops(glcm, 'energy')
    correlation = graycoprops(glcm, 'correlation')
    dissimilarity = graycoprops(glcm, 'dissimilarity')
    homogeneity = graycoprops(glcm, 'homogeneity')
    ASM = graycoprops(glcm, 'ASM')
    
    features = [contrast, energy, correlation, dissimilarity, homogeneity, ASM]
    return [f.mean() for f in features]

def valid_image(image_path: str) -> bool:
    if (len(image_path) == 0):
        return False
    else:
        return True
    