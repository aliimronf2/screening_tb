import numpy as np
import cv2
import joblib
import streamlit as st
from skimage import measure, morphology, exposure, segmentation, img_as_float
from skimage.feature import graycomatrix, graycoprops, blob_log
from skimage.segmentation import felzenszwalb
from scipy import ndimage
from scipy.ndimage import gaussian_filter

# ==========================================
# 1. KONFIGURASI WARNA (FIX: HURUF KECIL)
# ==========================================
# Kunci harus huruf kecil ('infiltrate', dll) agar terbaca oleh app.py
LESION_CONFIG = {
    'infiltrate': {
        'color': [0.0, 1.0, 1.0], # Cyan
        'id': 1
    },
    'cavity': {
        'color': [1.0, 0.0, 0.0], # Merah
        'id': 2
    },
    'calcification': {
        'color': [1.0, 1.0, 0.0], # Kuning
        'id': 3
    },
    'effusion': { 
        'color': [0.0, 1.0, 0.0], # Hijau
        'id': 4
    }
}

# ==========================================
# 2. LOAD MODELS
# ==========================================
@st.cache_resource
def load_gusna_models():
    try:
        scaler = joblib.load('models/scaler_tbc_final_gusna.pkl')
        model = joblib.load('models/svm_model_tbc_final_gusna.pkl')
        return scaler, model
    except:
        return None, None

# ==========================================
# 3. SEGMENTASI (ALGORITMA GUSNA)
# ==========================================

def segment_body_robust(img_array, threshold=20):
    binary = img_array > threshold
    label_img = measure.label(binary)
    regions = measure.regionprops(label_img)

    if not regions:
        return img_array, np.zeros_like(img_array), img_array

    largest_region = max(regions, key=lambda x: x.area)
    body_mask = np.zeros_like(binary)
    for coords in largest_region.coords:
        body_mask[coords[0], coords[1]] = 1

    body_mask_filled = ndimage.binary_fill_holes(body_mask)
    body_mask_eroded = morphology.binary_erosion(body_mask_filled, morphology.disk(3))

    segmented_body = img_array.copy()
    segmented_body[body_mask_eroded == 0] = 0

    return img_array, body_mask_eroded, segmented_body

def segment_lungs_smart_fallback(img_input, body_mask):
    rows, cols = img_input.shape

    img_float = img_input.astype(float)
    img_gamma = 255 * (img_float / 255) ** 1.5
    img_gamma = img_gamma.astype(np.uint8)

    pixels_in_body = img_gamma[body_mask > 0]
    if len(pixels_in_body) == 0:
        return img_input, np.zeros_like(body_mask)

    mean_val = np.mean(pixels_in_body)
    std_val = np.std(pixels_in_body)
    thresh_val = mean_val - (0.3 * std_val)
    binary = (img_gamma < thresh_val) & (body_mask > 0)

    if np.sum(binary) < (np.sum(body_mask) * 0.05):
        thresh_val = np.percentile(pixels_in_body, 45)
        binary = (img_gamma < thresh_val) & (body_mask > 0)

    cutoff_row = int(rows * 0.12)
    binary[:cutoff_row, :] = 0

    binary = morphology.binary_closing(binary, morphology.disk(6))
    binary = morphology.binary_opening(binary, morphology.disk(4))

    mid_col = cols // 2
    binary[:, mid_col-3 : mid_col+3] = 0

    label_img = measure.label(binary)
    regions = measure.regionprops(label_img)
    candidates = [r for r in regions if r.area > 500 and r.centroid[0] > (rows * 0.12)]
    candidates.sort(key=lambda x: x.area, reverse=True)
    
    mask_convex_combined = np.zeros_like(binary)
    
    if not candidates:
        mask_convex_combined = morphology.binary_erosion(body_mask, morphology.disk(20))
        mask_convex_combined[:, mid_col-5:mid_col+5] = 0
    else:
        for region in candidates[:2]:
            temp_mask = np.zeros_like(binary)
            coords = region.coords
            temp_mask[coords[:,0], coords[:,1]] = 1
            hull = morphology.convex_hull_image(temp_mask)
            hull_clipped = np.logical_and(hull, body_mask)
            mask_convex_combined = np.logical_or(mask_convex_combined, hull_clipped)

    return img_gamma, mask_convex_combined


# 4. EKSTRAKSI FITUR (ALGORITMA GUSNA)
def extract_features_final(image):
    if image is None: return [0]*10
    image = image.astype(np.uint8)

    glcm = graycomatrix(image, distances=[1], angles=[0],
                        levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]

    glcm_norm = glcm / (np.sum(glcm) + 1e-10)
    entropy = -np.sum(glcm_norm * np.log2(glcm_norm + 1e-10))

    h, w = image.shape
    cutoff = int(h * 0.5)
    mean_upper = np.mean(image[:cutoff][image[:cutoff] > 0]) if np.any(image[:cutoff] > 0) else 0
    mean_lower = np.mean(image[cutoff:][image[cutoff:] > 0]) if np.any(image[cutoff:] > 0) else 0
    zonal_ratio = mean_upper / (mean_lower + 1e-5)

    blobs = blob_log(image, max_sigma=30, num_sigma=10, threshold=0.1)
    num_blobs = len(blobs)

    gy, gx = np.gradient(image)
    edge_mean = np.mean(np.hypot(gx, gy)[image > 0]) if np.any(image > 0) else 0

    return [contrast, homogeneity, energy, correlation, entropy,
            mean_upper, mean_lower, zonal_ratio, num_blobs, edge_mean]


# 5. DETEKSI LESI (ALGORITMA GUSNA - Felzenszwalb)
def analyze_lesions_gusna(img_original, lung_mask):
    img_float = img_as_float(img_original)
    img_masked = img_float.copy()
    img_masked[lung_mask == 0] = 0

    try:
        segments = felzenszwalb(img_masked, scale=40, sigma=0.5, min_size=50, channel_axis=None)
    except:
        segments = felzenszwalb(img_masked, scale=40, sigma=0.5, min_size=50)

    # Dictionary Masker Lesi (Gunakan keys huruf kecil)
    masks = {k: np.zeros_like(img_original, dtype=float) for k in LESION_CONFIG.keys()}

    lung_pixels = img_original[lung_mask > 0]
    if len(lung_pixels) == 0: return masks

    global_mean = np.mean(lung_pixels)
    global_std = np.std(lung_pixels)

    unique_segments = np.unique(segments)

    for seg_id in unique_segments:
        if seg_id == 0: continue
        segment_mask = (segments == seg_id)
        if np.sum(segment_mask & lung_mask) / np.sum(segment_mask) < 0.6: continue

        patch_vals = img_original[segment_mask]
        mean_val = np.mean(patch_vals)
        variance = np.var(patch_vals)

        # --- LOGIKA KLASIFIKASI (FIX: keys lowercase) ---
        
        # 1. Calcification
        if mean_val > (global_mean + 2.0 * global_std):
            masks['calcification'][segment_mask] = 1.0

        # 2. Effusion
        elif mean_val > (global_mean + 0.8 * global_std) and variance < 500:
            masks['effusion'][segment_mask] = 1.0

        # 3. Cavity
        elif mean_val < (global_mean - 0.9 * global_std):
            masks['cavity'][segment_mask] = 1.0

        # 4. Infiltrate
        elif mean_val > global_mean and variance >= 500:
            masks['infiltrate'][segment_mask] = 1.0

    return masks


# 6. FUNGSI UTAMA
def process_image(image_array):
    scaler, model = load_gusna_models()
    if not scaler or not model: 
        return {"error": "Model Gusna (scaler_tbc_final_gusna/svm_model_tbc_final_gusna) tidak ditemukan!"}

    # 1. Preprocess
    if len(image_array.shape) == 3:
        image_gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        image_gray = image_array
        
    img_512 = cv2.resize(image_gray, (512, 512))
    
    # 2. Segmentasi
    _, body_mask_eroded, body_img = segment_body_robust(img_512)
    img_enhanced, lung_mask = segment_lungs_smart_fallback(body_img, body_mask_eroded)
    
    if np.sum(lung_mask) == 0:
        return {"error": "Gagal segmentasi paru."}

    # 3. Predict SVM
    img_roi = img_512.copy()
    img_roi[lung_mask == 0] = 0
    
    features = extract_features_final(img_roi)
    features_scaled = scaler.transform([features])
    pred = model.predict(features_scaled)[0]
    conf = model.predict_proba(features_scaled)[0][pred]

    # 4. Lesion Analysis
    lesions = None
    stats = {}
    lung_area = np.sum(lung_mask)

    if pred == 1:
        lesions = analyze_lesions_gusna(img_512, lung_mask)
        # Hitung statistik (keys sudah lowercase sekarang)
        for name, mask in lesions.items():
            stats[name] = (np.sum(mask) / lung_area) * 100

    # 5. Visualisasi Overlay
    vis = cv2.cvtColor(img_512, cv2.COLOR_GRAY2RGB)
    overlay_lung = vis.copy()
    overlay_lung[lung_mask > 0] = [255, 0, 255] # Ungu
    vis = cv2.addWeighted(overlay_lung, 0.3, vis, 0.7, 0)

    if lesions:
        for name, config in LESION_CONFIG.items():
            mask = lesions[name]
            mask_smooth = gaussian_filter(mask, sigma=1.0)
            mask_bin = mask_smooth > 0.1
            
            if np.sum(mask_bin) > 0:
                color_rgb = [int(c*255) for c in config['color']]
                layer = vis.copy()
                layer[mask_bin] = color_rgb
                vis = cv2.addWeighted(layer, 0.5, vis, 0.5, 0)

    return {
        "prediction": "Tuberculosis" if pred == 1 else "Normal",
        "confidence": conf,
        "overlay": vis,
        "stats": stats
    }