import cv2
import numpy as np
import joblib
from skimage import filters, morphology, measure, exposure, segmentation, feature
from scipy.ndimage import binary_fill_holes
from skimage.measure import regionprops
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, hog
from skimage.filters import rank
from sklearn.cluster import KMeans
from scipy.stats import skew, kurtosis

# --- LOAD MODELS ---
def load_my_models():
    try:
        # Pastikan nama file sesuai yang ada di folder models/
        scaler = joblib.load('models/scaler_data2.pkl')
        model = joblib.load('models/svm_tb_data2_model.pkl')
        return scaler, model
    except:
        return None, None


# SEGMENTASI PARU
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
    labels = kmeans.labels_
    full_labels = np.zeros(img.shape, dtype=int)
    full_labels[torso_mask > 0] = labels
    return ((full_labels == np.argmin(kmeans.cluster_centers_)) & (torso_mask > 0))

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
        if np.any(region_mask & v_seed): clean_seed |= region_mask
    return clean_seed

def get_smart_barrier(img, torso_mask, clean_seed):
    if np.sum(torso_mask) == 0: return np.zeros_like(img)
    bright_thresh = np.percentile(img[torso_mask > 0], 60)
    skeleton = morphology.skeletonize((img > bright_thresh) & (torso_mask > 0))
    img_u8 = (img * 255).astype(np.uint8)
    canny = cv2.Canny(img_u8, 40, 120) > 0
    internal_zone = morphology.binary_dilation(clean_seed, morphology.disk(25))
    return ((skeleton | canny) & (~internal_zone)) | (torso_mask == 0)

def segment_lungs_v26(img):
    try:
        img_enh = advanced_preprocess(img)
        torso_mask = generate_torso_mask_safe(img_enh)
        raw_lung = segment_kmeans(img_enh, torso_mask)
        clean_seed = get_clean_seed(raw_lung)
        barrier = get_smart_barrier(img_enh, torso_mask, clean_seed)
        
        img_smooth = cv2.GaussianBlur(img_enh, (3, 3), 0)
        markers = np.zeros_like(raw_lung, dtype=np.int32)
        markers[clean_seed == 1] = 1
        markers[barrier == 1] = 2
        
        ws = segmentation.watershed(filters.sobel(img_smooth), markers)
        filled = binary_fill_holes(ws == 1)
        
        final_mask = np.zeros_like(raw_lung, dtype=np.uint8)
        labels = measure.label(filled)
        for prop in sorted(measure.regionprops(labels), key=lambda x: x.area, reverse=True)[:2]:
            lung = (labels == prop.label)
            final_mask[morphology.convex_hull_image(lung) & morphology.binary_dilation(lung, morphology.disk(10))] = 1
            
        # Smoothing Akhir (V27 Logic)
        mask_u8 = (final_mask * 255).astype(np.uint8)
        _, smooth = cv2.threshold(cv2.GaussianBlur(mask_u8, (25, 25), 0), 127, 255, cv2.THRESH_BINARY)
        
        return smooth, img_enh
    except:
        return None, None


# BAGIAN 2: FEATURE EXTRACTION
def extract_features(img, mask):
    # Intensity
    roi = img[mask > 0]
    if len(roi) == 0: return np.zeros(29) # Total fitur
    
    feats = [np.mean(roi), np.std(roi), np.max(roi), np.min(roi), skew(roi), kurtosis(roi)]
    
    # GLCM
    img_u8 = (img * 255).astype(np.uint8)
    rows, cols = np.where(mask > 0)
    y1, y2, x1, x2 = np.min(rows), np.max(rows), np.min(cols), np.max(cols)
    glcm = graycomatrix(img_u8[y1:y2, x1:x2], [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], 256, symmetric=True, normed=True)
    feats += [np.mean(graycoprops(glcm, p)) for p in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']]
    
    # LBP
    lbp = local_binary_pattern(img_u8, 8, 1, 'uniform')
    hist, _ = np.histogram(lbp[mask > 0], bins=10, range=(0, 10), density=True)
    feats += list(hist) + [0]*(10-len(hist))
    
    # Shape
    props = regionprops(measure.label(mask))
    if props:
        feats += [sum(p.area for p in props), np.mean([p.solidity for p in props]), 
                  np.mean([p.extent for p in props]), np.mean([p.eccentricity for p in props]),
                  (4 * np.pi * sum(p.area for p in props)) / (sum(p.perimeter for p in props)**2 + 1e-5)]
    else: feats += [0]*5
    
    # HOG
    fd = hog(cv2.resize(img, (128, 128)), orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1))
    feats += [np.mean(fd), np.std(fd), np.max(fd), kurtosis(fd)]
    
    return np.array(feats).reshape(1, -1)


# LESION DETECTION
def detect_lesions_strict(img_enhanced, lung_mask):
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
        if p.area > 150 and p.eccentricity < 0.9 and p.solidity > 0.85: mask_cav[measure.label(raw_cav) == p.label] = 1
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


# API
def process_image(image_array):
    scaler, model = load_my_models()
    if not scaler or not model: return {"error": "Model Saya tidak ditemukan!"}
    
    # 1. Preprocess
    if len(image_array.shape) == 3: image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    img_512 = cv2.resize(image_array, (512, 512))
    img_float = img_512.astype(np.float32) / 255.0
    
    # 2. Segment
    mask, img_enh = segment_lungs_v26(img_512)
    if mask is None: return {"error": "Segmentasi Gagal"}
    _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # 3. Predict
    feats = extract_features(img_float, mask_bin)
    feats_scaled = scaler.transform(feats)
    pred = model.predict(feats_scaled)[0]
    conf = model.predict_proba(feats_scaled)[0][pred]
    
    # 4. Lesions & Stats
    lesions = detect_lesions_strict(img_enh, mask_bin) if pred == 1 else None
    
    # Hitung Persentase
    lung_area = np.sum(mask_bin > 0)
    stats = {}
    if lesions:
        for k, v in lesions.items():
            stats[k] = (np.sum(v > 0) / lung_area) * 100 if lung_area > 0 else 0
            
    # 5. Overlay
    vis = cv2.cvtColor(img_512, cv2.COLOR_GRAY2RGB)
    vis[mask_bin > 0] = [255, 0, 255] # Ungu Base
    vis = cv2.addWeighted(vis, 0.3, cv2.cvtColor(img_512, cv2.COLOR_GRAY2RGB), 0.7, 0)
    
    if lesions:
        if np.sum(lesions['infiltrate']) > 0: vis[lesions['infiltrate'] > 0] = [0, 255, 255] # Cyan
        if np.sum(lesions['effusion']) > 0: vis[lesions['effusion'] > 0] = [0, 255, 0] # Hijau
        if np.sum(lesions['calcification']) > 0: vis[lesions['calcification'] > 0] = [255, 255, 0] # Kuning
        if np.sum(lesions['cavity']) > 0: vis[lesions['cavity'] > 0] = [255, 0, 0] # Merah

    return {
        "prediction": "Tuberculosis" if pred == 1 else "Normal",
        "confidence": conf,
        "overlay": vis,
        "stats": stats
    }
