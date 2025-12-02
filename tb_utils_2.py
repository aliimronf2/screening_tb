import numpy as np
import cv2
import joblib
import streamlit as st
import math
from skimage import measure, morphology, img_as_float
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, hog, blob_log
from skimage.segmentation import felzenszwalb
from skimage.measure import regionprops, label
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from scipy.stats import skew, kurtosis, entropy


# 1. KONFIGURASI WARNA & MODEL
LESION_CONFIG = {
    'Infiltrate': {'color': [0.0, 1.0, 1.0], 'id': 1},    # Cyan
    'Cavity': {'color': [1.0, 0.0, 0.0], 'id': 2},        # Merah
    'Calcification': {'color': [1.0, 1.0, 0.0], 'id': 3}, # Kuning
    'Effusion': {'color': [0.0, 1.0, 0.0], 'id': 4}       # Hijau
}

@st.cache_resource
def load_sapto_models():
    try:
        # Nama file sesuai request: scaler_paru3 dan model_svm_paru3
        scaler = joblib.load('models/scaler_paru3.pkl')
        model = joblib.load('models/model_svm_paru3.pkl')
        return scaler, model
    except:
        return None, None

# 2. FUNGSI PENDUKUNG SEGMENTASI SAPTO
def clahe(input, clipLimit=2.0, tileGridSize=(8,8)):
    clahe_obj = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    return clahe_obj.apply(input)

def otsu(input):
    # Sapto pakai Binary Inv + Otsu
    ret, output = cv2.threshold(input, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return output

def make_round_frame(input_arr, mid_gap=35):
    H, W = input_arr.shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)
    
    # Radius default Sapto
    rad_tl, rad_tr, rad_bl, rad_br = 0, 0, 0, 0
    rad_gap = 0
    
    # Helper draw box sederhana
    def draw_simple_box(img, x0, y0, x1, y1):
        if x0 < x1 and y0 < y1:
            cv2.rectangle(img, (x0, y0), (x1-1, y1-1), 1, -1)

    y0_frame, y1_frame = 0, H
    x0_frame, x1_frame = 0, W
    
    if mid_gap <= 0:
        draw_simple_box(mask, x0_frame, y0_frame, x1_frame, y1_frame)
    else:
        center_x = W // 2
        half_gap = mid_gap // 2
        x1_left = center_x - half_gap
        x0_right = center_x + half_gap + (mid_gap % 2)
        
        draw_simple_box(mask, x0_frame, y0_frame, x1_left, y1_frame)
        draw_simple_box(mask, x0_right, y0_frame, x1_frame, y1_frame)

    return mask * 255

def body_mask(input):
    inverted_img = cv2.bitwise_not(input)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inverted_img, connectivity=8)
    
    largest_eroded = np.zeros_like(input)
    
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        max_label_idx = np.argmax(areas) + 1
        largest_mask = (labels == max_label_idx).astype("uint8") * 255
        
        contours, _ = cv2.findContours(largest_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = contours[0]
            hull = cv2.convexHull(cnt)
            hull_solid = np.zeros_like(largest_mask)
            cv2.drawContours(hull_solid, [hull], -1, 255, thickness=-1)
            
            # Erosi 10px (Kernel 21)
            kernel = np.ones((21, 21), np.uint8)
            largest_eroded = cv2.erode(hull_solid, kernel, iterations=1)
            
    return largest_eroded

def cca_dist(input_image):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(input_image, connectivity=8)
    if num_labels <= 1: return np.zeros_like(input_image)

    h, w = input_image.shape[:2]
    cx, cy = w // 2, h // 2
    candidates = []
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        c_x, c_y = centroids[i]
        dist = math.sqrt((c_x - cx)**2 + (c_y - cy)**2)
        candidates.append({'idx': i, 'area': area, 'dist': dist})
        
    # Sort area desc, ambil top 5, lalu sort dist asc, ambil top 2
    candidates.sort(key=lambda x: x['area'], reverse=True)
    top_cands = candidates[:5]
    top_cands.sort(key=lambda x: x['dist'])
    final = top_cands[:2]
    
    mask = np.zeros_like(input_image)
    for item in final:
        mask[labels == item['idx']] = 255
    return mask

def convex_hull_fill(input):
    contours, _ = cv2.findContours(input, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull_mask = np.zeros_like(input)
    for cnt in contours:
        hull = cv2.convexHull(cnt)
        cv2.drawContours(hull_mask, [hull], -1, 255, -1)
    return hull_mask

def remove_pad(input, padding_px=12):
    h, w = input.shape[:2]
    return input[padding_px : h - padding_px, padding_px : w - padding_px]

# 3. MASKING UTAMA (masking02)
def masking02(input_img):
    # Kernel Sapto
    kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    
    # Preprocess
    blur = cv2.GaussianBlur(input_img, (9,9), 0)
    cl = clahe(blur)
    otsu_res = otsu(cl)
    
    # Frame Cut
    frame = make_round_frame(otsu_res, mid_gap=35)
    masked_otsu = cv2.bitwise_and(frame, otsu_res)
    
    # Padding & Body Mask
    padded = cv2.copyMakeBorder(masked_otsu, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=[255])
    largest = body_mask(padded)
    
    out = cv2.bitwise_and(padded, largest)
    out = cv2.dilate(out, kernel5, iterations=3)
    out = cca_dist(out)
    
    # Morfologi Akhir
    morf = convex_hull_fill(out)
    output = remove_pad(morf)
    
    return output

# 4. EKSTRAKSI FITUR (extract_all_features)
def extract_all_features_sapto(image, mask):
    x, y, w, h = cv2.boundingRect(mask)
    if w == 0 or h == 0: 
        # Return dummy features (sesuai panjang fitur di training)
        # Intensity(4) + GLCM(6) + LBP(2) + Shape(3) + HOG(3) + Extra(2) = 20 fitur
        return np.zeros(20)
        
    roi_img = image[y:y+h, x:x+w]
    roi_mask = mask[y:y+h, x:x+w]
    masked_roi = cv2.bitwise_and(roi_img, roi_img, mask=roi_mask)
    lung_pixels = roi_img[roi_mask > 0]
    
    feats = {}
    
    # Intensity
    feats['int_mean'] = np.mean(lung_pixels) if len(lung_pixels)>0 else 0
    feats['int_std'] = np.std(lung_pixels) if len(lung_pixels)>0 else 0
    feats['int_skew'] = skew(lung_pixels) if len(lung_pixels)>0 else 0
    feats['int_kurtosis'] = kurtosis(lung_pixels) if len(lung_pixels)>0 else 0
    
    # GLCM
    glcm = graycomatrix(masked_roi, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
    feats['glcm_contrast'] = np.mean(graycoprops(glcm, 'contrast'))
    feats['glcm_dissimilarity'] = np.mean(graycoprops(glcm, 'dissimilarity'))
    feats['glcm_homogeneity'] = np.mean(graycoprops(glcm, 'homogeneity'))
    feats['glcm_energy'] = np.mean(graycoprops(glcm, 'energy'))
    feats['glcm_correlation'] = np.mean(graycoprops(glcm, 'correlation'))
    feats['glcm_ASM'] = np.mean(graycoprops(glcm, 'ASM'))
    
    # LBP
    lbp = local_binary_pattern(roi_img, 8, 1, method='uniform')
    lbp_valid = lbp[roi_mask > 0]
    if len(lbp_valid) > 0:
        hist, _ = np.histogram(lbp_valid, bins=np.arange(0, 11), density=True)
        feats['lbp_energy'] = np.sum(hist ** 2)
        feats['lbp_entropy'] = entropy(hist)
    else:
        feats['lbp_energy'] = 0; feats['lbp_entropy'] = 0
        
    # Shape
    lbl_mask = label(roi_mask)
    props = regionprops(lbl_mask)
    if props:
        p = max(props, key=lambda x: x.area)
        feats['shape_solidity'] = p.solidity
        feats['shape_eccentricity'] = p.eccentricity
        feats['shape_extent'] = p.extent
    else:
        feats['shape_solidity'] = 0; feats['shape_eccentricity'] = 0; feats['shape_extent'] = 0
        
    # HOG
    try:
        roi_res = cv2.resize(roi_img, (64, 128))
        fd, _ = hog(roi_res, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
        feats['hog_mean'] = np.mean(fd)
        feats['hog_std'] = np.std(fd)
        feats['hog_skew'] = skew(fd)
    except:
        feats['hog_mean'] = 0; feats['hog_std'] = 0; feats['hog_skew'] = 0
        
    # Extra
    bright = np.count_nonzero((roi_img > 200) & (roi_mask > 0))
    dark = np.count_nonzero((roi_img < 50) & (roi_mask > 0))
    area = np.count_nonzero(roi_mask)
    feats['calc_ratio'] = bright / area if area > 0 else 0
    feats['cavity_ratio'] = dark / area if area > 0 else 0
    
    # Return as array 1D
    return np.array(list(feats.values())).reshape(1, -1)


# 5. DETEKSI LESI (analyze_lesion_content)
def analyze_lesions_sapto(img, mask):
    img_float = img_as_float(img)
    img_masked = img_float.copy()
    img_masked[mask == 0] = 0
    
    try:
        segments = felzenszwalb(img_masked, scale=40, sigma=0.5, min_size=50, channel_axis=None)
    except:
        segments = felzenszwalb(img_masked, scale=40, sigma=0.5, min_size=50)
        
    masks = {k: np.zeros(img.shape, dtype=float) for k in LESION_CONFIG.keys()}
    lung_pixels = img[mask > 0]
    if len(lung_pixels) == 0: return masks
    
    g_mean = np.mean(lung_pixels)
    g_std = np.std(lung_pixels)
    
    for seg_id in np.unique(segments):
        if seg_id == 0: continue
        s_mask = (segments == seg_id)
        if np.sum(s_mask & (mask>0)) / np.sum(s_mask) < 0.6: continue
        
        vals = img[s_mask]
        m_val = np.mean(vals)
        var = np.var(vals)
        
        # Logika
        if m_val > (g_mean + 2.0 * g_std):
            masks['Calcification'][s_mask] = 1.0
        elif m_val > (g_mean + 0.8 * g_std) and var < 500:
            masks['Effusion'][s_mask] = 1.0
        elif m_val < (g_mean - 0.9 * g_std):
            masks['Cavity'][s_mask] = 1.0
        elif m_val > g_mean and var >= 500:
            masks['Infiltrate'][s_mask] = 1.0
            
    return masks

# 6. FUNGSI UTAMA (API)
def process_image(image_array):
    scaler, model = load_sapto_models()
    if not scaler or not model:
        return {"error": "Model Sapto (scaler_paru3/model_svm_paru3) tidak ditemukan!"}
        
    # 1. Preprocess
    if len(image_array.shape) == 3:
        img_gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = image_array
    
    img_512 = cv2.resize(img_gray, (512, 512))
    
    # 2. Segment
    mask = masking02(img_512)
    
    if np.sum(mask) == 0: return {"error": "Gagal segmentasi paru."}
    
    # 3. Predict
    # Masking image
    img_roi = img_512.copy()
    img_roi[mask == 0] = 0
    
    feats = extract_all_features_sapto(img_roi, mask)
    feats_scaled = scaler.transform(feats)
    pred = model.predict(feats_scaled)[0]
    conf = model.predict_proba(feats_scaled)[0][pred]
    
    # 4. Lesion & Overlay
    lesions = None
    stats = {}
    
    if pred == 1:
        lesions = analyze_lesions_sapto(img_512, mask)
        lung_area = np.sum(mask)
        for k, v in lesions.items():
            stats[k] = (np.sum(v) / lung_area) * 100
            
    # Visualisasi
    vis = cv2.cvtColor(img_512, cv2.COLOR_GRAY2RGB)
    vis[mask > 0] = [255, 0, 255] # Ungu
    vis = cv2.addWeighted(vis, 0.3, cv2.cvtColor(img_512, cv2.COLOR_GRAY2RGB), 0.7, 0)
    
    if lesions:
        for name, config in LESION_CONFIG.items():
            m = lesions[name]
            m_smooth = gaussian_filter(m, sigma=1.0) > 0.1
            if np.sum(m_smooth) > 0:
                col_rgb = [int(c*255) for c in config['color']]
                layer = vis.copy()
                layer[m_smooth] = col_rgb
                vis = cv2.addWeighted(layer, 0.5, vis, 0.5, 0)
                
    return {
        "prediction": "Tuberculosis" if pred == 1 else "Normal",
        "confidence": conf,
        "overlay": vis,
        "stats": stats
    }