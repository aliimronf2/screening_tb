import cv2
import numpy as np
import joblib
import streamlit as st
from skimage import filters, morphology, measure, exposure, segmentation, feature
from scipy.ndimage import binary_fill_holes
from skimage.measure import regionprops
from skimage.feature import graycomatrix, graycoprops, hog
from skimage.filters import rank
from sklearn.cluster import KMeans
from scipy.stats import skew, kurtosis

# ==========================================
# 1. KONFIGURASI & LOAD MODEL
# ==========================================

# PERBAIKAN: Keys menggunakan huruf kecil agar cocok dengan app.py dan fungsi deteksi
LESION_CONFIG = {
    'infiltrate': {'color': [0, 255, 255], 'id': 1},    # Cyan
    'cavity':     {'color': [255, 0, 0],   'id': 2},    # Merah
    'calcification': {'color': [255, 255, 0], 'id': 3}, # Kuning
    'effusion':   {'color': [0, 255, 0],   'id': 4}     # Hijau
}

@st.cache_resource
def load_my_models():
    try:
        # Pastikan nama file ini BENAR dan ada di folder 'models/'
        scaler = joblib.load('models/scaler_data2_3mini.pkl')
        model = joblib.load('models/svm_tb_data2_model3mini.pkl')
        return scaler, model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None, None

# ==========================================
# 2. SEGMENTASI PARU (V26 CORE)
# ==========================================
def advanced_preprocess(img):
    img_u8 = (img * 255).astype(np.uint8)
    denoised = cv2.bilateralFilter(img_u8, 9, 75, 75)
    img_clean = denoised.astype(np.float32) / 255.0
    return exposure.equalize_adapthist(img_clean, clip_limit=0.02)

def generate_torso_mask_safe(img):
    try: thr = filters.threshold_otsu(img)
    except: return np.ones_like(img, dtype=np.uint8)
    mask = img > thr * 0.5
    mask = morphology.remove_small_objects(mask, 5000)
    mask = morphology.binary_closing(mask, morphology.disk(5))
    mask = binary_fill_holes(mask)
    h, w = mask.shape
    mask[:, :int(w*0.05)] = 0
    mask[:, -int(w*0.05):] = 0
    return mask.astype(np.uint8)

def segment_kmeans(img, torso_mask):
    if np.sum(torso_mask) == 0: return np.zeros_like(img, dtype=np.uint8)
    masked_pixels = img[torso_mask > 0].reshape(-1, 1)
    if masked_pixels.shape[0] < 100: return np.zeros_like(img, dtype=np.uint8)
    
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=5).fit(masked_pixels)
    centers = kmeans.cluster_centers_
    lung_cluster_label = np.argmin(centers)
    
    labels = kmeans.labels_
    full_labels = np.zeros(img.shape, dtype=int)
    full_labels[torso_mask > 0] = labels
    
    return ((full_labels == lung_cluster_label) & (torso_mask > 0))

def get_clean_seed(raw_mask):
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 25))
    v_seed = cv2.morphologyEx(raw_mask.astype(np.uint8), cv2.MORPH_OPEN, v_kernel)
    try: v_seed = morphology.remove_small_objects(v_seed.astype(bool), 1000)
    except: pass
    
    severed_mask = morphology.binary_erosion(raw_mask, morphology.disk(10))
    labels = measure.label(severed_mask)
    clean_seed = np.zeros_like(raw_mask, dtype=bool)
    
    for region in measure.regionprops(labels):
        region_mask = (labels == region.label)
        if np.any(region_mask & v_seed):
            clean_seed = clean_seed | region_mask
    return clean_seed

def get_smart_barrier(img, torso_mask, clean_seed):
    if np.sum(torso_mask) == 0: return np.zeros_like(img)
    bright_thresh = np.percentile(img[torso_mask > 0], 60) 
    bright_areas = (img > bright_thresh) & (torso_mask > 0)
    skeleton = morphology.skeletonize(bright_areas)
    
    img_u8 = (img * 255).astype(np.uint8)
    canny_edges = cv2.Canny(img_u8, 40, 120) > 0
    
    raw_barrier = (skeleton | canny_edges)
    internal_zone = morphology.binary_dilation(clean_seed, morphology.disk(25))
    final_barrier = raw_barrier & (~internal_zone)
    
    bg_outside = (torso_mask == 0)
    final_barrier = final_barrier | bg_outside
    return final_barrier

def make_it_smooth(binary_mask):
    mask_u8 = (binary_mask * 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(mask_u8, (25, 25), 0)
    _, smooth_mask = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(smooth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_canvas = np.zeros_like(mask_u8)
    
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
    for cnt in contours:
        epsilon = 0.003 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        cv2.drawContours(final_canvas, [approx], -1, 255, -1)
    return final_canvas

def segment_lungs_v26_core(img_original):
    # Resize agar konsisten dengan training
    h_orig, w_orig = img_original.shape
    img_resized = cv2.resize(img_original, (512, 512))
    img_float = img_resized.astype(np.float32) / 255.0
    
    try:
        img_enhanced = advanced_preprocess(img_float)
        torso_mask = generate_torso_mask_safe(img_enhanced)
        raw_lung = segment_kmeans(img_enhanced, torso_mask)
        clean_seed = get_clean_seed(raw_lung)
        barrier_map = get_smart_barrier(img_enhanced, torso_mask, clean_seed)
        
        img_smooth = cv2.GaussianBlur(img_enhanced, (3, 3), 0)
        elevation_map = filters.sobel(img_smooth)
        
        markers = np.zeros_like(raw_lung, dtype=np.int32)
        markers[clean_seed == 1] = 1
        markers[barrier_map == 1] = 2
        
        watershed_result = segmentation.watershed(elevation_map, markers)
        lung_watershed = (watershed_result == 1)
        filled_lung = binary_fill_holes(lung_watershed)
        
        # Convex Smoothing
        final_mask = np.zeros_like(raw_lung, dtype=np.uint8)
        labels = measure.label(filled_lung)
        props = sorted(measure.regionprops(labels), key=lambda x: x.area, reverse=True)[:2]
        
        for prop in props:
            single_lung = (labels == prop.label)
            chull = morphology.convex_hull_image(single_lung)
            reference_shape = morphology.binary_dilation(single_lung, morphology.disk(10))
            smoothed_lung = chull & reference_shape
            final_mask[smoothed_lung] = 1
            
        final_mask_smooth = make_it_smooth(final_mask)
        
        # Return mask (512x512) dan enhanced image (512x512)
        return final_mask_smooth, img_enhanced
        
    except:
        return None, None

# ==========================================
# 3. EKSTRAKSI FITUR (NO LBP)
# ==========================================
def extract_intensity_features(img, mask):
    roi = img[mask > 0]
    if len(roi) == 0: return [0]*6
    return [np.mean(roi), np.std(roi), np.max(roi), np.min(roi), skew(roi), kurtosis(roi)]

def extract_glcm_features(img, mask):
    img_u8 = (img * 255).astype(np.uint8)
    rows, cols = np.where(mask > 0)
    if len(rows) == 0: return [0]*5
    roi = img_u8[np.min(rows):np.max(rows), np.min(cols):np.max(cols)]
    glcm = graycomatrix(roi, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], 256, symmetric=True, normed=True)
    return [np.mean(graycoprops(glcm, p)) for p in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']]

def extract_shape_features(mask):
    lbl_mask = measure.label(mask)
    props = regionprops(lbl_mask)
    if not props: return [0]*5
    
    total_area = sum([p.area for p in props])
    solidity = np.mean([p.solidity for p in props])
    extent = np.mean([p.extent for p in props])
    eccentricity = np.mean([p.eccentricity for p in props])
    perimeter = sum([p.perimeter for p in props])
    compactness = (4 * np.pi * total_area) / (perimeter ** 2) if perimeter > 0 else 0
    
    return [total_area, solidity, extent, eccentricity, compactness]

def extract_hog_features(img, mask):
    img_m = img.copy(); img_m[mask == 0] = 0
    fd = hog(cv2.resize(img_m, (128,128)), orientations=8, pixels_per_cell=(16,16), cells_per_block=(1,1))
    return [np.mean(fd), np.std(fd), np.max(fd), kurtosis(fd)]

# ==========================================
# 4. DETEKSI LESI (STRICT)
# ==========================================
def detect_lesions_strict(img_enhanced, lung_mask):
    # Output keys harus huruf kecil agar cocok dengan LESION_CONFIG dan app.py
    masks = {}
    lung_bool = lung_mask > 0
    roi = img_enhanced[lung_bool]
    if len(roi) == 0: return {}
    
    mean, std = np.mean(roi), np.std(roi)
    
    # Calcification
    masks['calcification'] = morphology.remove_small_objects((img_enhanced > mean + 3*std) & lung_bool, 10)
    
    # Cavity
    top_lung = lung_bool.copy(); top_lung[int(img_enhanced.shape[0]*0.6):, :] = 0
    raw_cav = (img_enhanced < mean - 1.5*std) & top_lung
    mask_cav = np.zeros_like(lung_mask)
    for p in regionprops(measure.label(raw_cav)):
        if p.area > 150 and p.eccentricity < 0.9 and p.solidity > 0.85: 
            mask_cav[measure.label(raw_cav) == p.label] = 1
    masks['cavity'] = mask_cav
    
    # Infiltrate
    try: entropy = rank.entropy((img_enhanced * 255).astype(np.uint8), morphology.disk(5), mask=lung_mask)
    except: entropy = np.zeros_like(img_enhanced)
    masks['infiltrate'] = morphology.binary_opening((entropy > 5.5) & lung_bool & (~masks['calcification']), morphology.disk(3))
    
    # Effusion
    btm_lung = lung_bool.copy(); btm_lung[:int(img_enhanced.shape[0]*0.75), :] = 0
    masks['effusion'] = np.zeros_like(lung_mask)
    if np.sum(btm_lung) > 0:
        lbl = measure.label(btm_lung)
        for p in regionprops(lbl):
            if p.area > 500:
                side = (lbl == p.label)
                cand = morphology.convex_hull_image(side) & (~side)
                masks['effusion'] |= morphology.binary_opening(morphology.remove_small_objects(cand, 1500), morphology.disk(5))
                
    return masks

# ==========================================
# 5. FUNGSI UTAMA (API UNTUK APP)
# ==========================================
def process_image(image_array):
    scaler, model = load_my_models()
    if not scaler or not model:
        return {"error": "Model Saya (scaler_3mini/svm_tb_model_3mini) tidak ditemukan!"}
        
    # 1. Preprocess Input (Streamlit kasih RGB/BGR)
    if len(image_array.shape) == 3:
        img_gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = image_array
        
    # 2. Segmentasi V26
    mask, img_enh = segment_lungs_v26_core(img_gray)
    
    if mask is None or np.sum(mask) == 0:
        return {"error": "Gagal segmentasi paru. Gambar mungkin tidak jelas."}
        
    # Normalize mask 0-255 (uint8)
    _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # 3. Feature Extraction
    img_512 = cv2.resize(img_gray, (512, 512))
    img_float = img_512.astype(np.float32) / 255.0
    
    f_int = extract_intensity_features(img_float, mask_bin)
    f_glcm = extract_glcm_features(img_float, mask_bin)
    f_shp = extract_shape_features(mask_bin) 
    
    img_masked = img_float.copy()
    img_masked[mask_bin == 0] = 0
    f_hog = extract_hog_features(img_masked, mask_bin) 
    
    features = f_int + f_glcm + f_shp + f_hog
    features_vector = np.array(features).reshape(1, -1)
    
    # 4. Predict SVM
    features_scaled = scaler.transform(features_vector)
    pred = model.predict(features_scaled)[0]
    conf = model.predict_proba(features_scaled)[0][pred]
    
    # 5. Lesion & Overlay
    lesions = None
    stats = {}
    
    # Hitung luas paru dalam PIKSEL (Bukan jumlah nilai intensitas)
    # --- PERBAIKAN UTAMA DI SINI ---
    lung_area_pixels = np.count_nonzero(mask_bin)
    # -------------------------------

    # Jika TB, deteksi lesi
    if pred == 1:
        lesions = detect_lesions_strict(img_enh, mask_bin)
        
        for k, v in lesions.items():
            # Hitung luas lesi dalam PIKSEL
            lesion_area_pixels = np.count_nonzero(v)
            
            # Hitung Persentase yang Benar
            if lung_area_pixels > 0:
                stats[k] = (lesion_area_pixels / lung_area_pixels) * 100
            else:
                stats[k] = 0
            
    # Buat Overlay Visualisasi
    vis = cv2.cvtColor(img_512, cv2.COLOR_GRAY2RGB)
    
    # Base Paru (Ungu)
    overlay_lung = vis.copy()
    overlay_lung[mask_bin > 0] = [255, 0, 255]
    vis = cv2.addWeighted(overlay_lung, 0.3, vis, 0.7, 0)
    
    # Overlay Lesi (Jika ada)
    if lesions:
        for name, config in LESION_CONFIG.items():
            if name in lesions:
                m = lesions[name]
                if np.sum(m) > 0:
                    color_rgb = config['color'] 
                    layer = vis.copy()
                    layer[m > 0] = color_rgb
                    vis = cv2.addWeighted(layer, 0.5, vis, 0.5, 0)
                
    return {
        "prediction": "Tuberculosis" if pred == 1 else "Normal",
        "confidence": conf,
        "overlay": vis,
        "stats": stats
    }