import streamlit as st
import cv2
import numpy as np
import joblib
import json
import os
import math
from PIL import Image
from scipy import ndimage
from scipy.ndimage import binary_fill_holes, gaussian_filter
from scipy.stats import skew, kurtosis, entropy
from skimage import filters, morphology, measure, exposure, segmentation, img_as_float
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, hog, blob_log
from skimage.segmentation import felzenszwalb
from skimage.measure import regionprops, label
from skimage.filters import rank
from sklearn.cluster import KMeans

# ==========================================
# KONFIGURASI GLOBAL & USER DB
# ==========================================
st.set_page_config(page_title="TB Detective", page_icon="🫁", layout="wide")

USER_DB = 'users.json'

# Konfigurasi Warna Lesi (Format [R, G, B])
LESION_COLORS = {
    'infiltrate': [0, 255, 255],    # Cyan
    'cavity':     [255, 0, 0],      # Merah
    'calcification': [255, 255, 0], # Kuning
    'effusion':   [0, 255, 0]       # Hijau
}

# --- FUNGSI AUTH ---
def load_users():
    if not os.path.exists(USER_DB): return {}
    with open(USER_DB, 'r') as f: return json.load(f)

def save_user(u, p):
    users = load_users()
    users[u] = p
    with open(USER_DB, 'w') as f: json.dump(users, f)

def verify_login(u, p):
    users = load_users()
    return u in users and users[u] == p

# ==========================================
# MODEL 1: LOGIKA GUSNA (Class Wrapper)
# ==========================================
class ModelGusna:
    def __init__(self):
        self.scaler = None
        self.model = None
        
    def load_models(self):
        try:
            self.scaler = joblib.load('models/scaler_tbc_final_gusna.pkl')
            self.model = joblib.load('models/svm_model_tbc_final_gusna.pkl')
            return True
        except: return False

    def segment_body_robust(self, img_array, threshold=20):
        binary = img_array > threshold
        label_img = measure.label(binary)
        regions = measure.regionprops(label_img)
        if not regions: return img_array, np.zeros_like(img_array), img_array
        largest_region = max(regions, key=lambda x: x.area)
        body_mask = np.zeros_like(binary)
        for coords in largest_region.coords: body_mask[coords[0], coords[1]] = 1
        body_mask_filled = binary_fill_holes(body_mask)
        body_mask_eroded = morphology.binary_erosion(body_mask_filled, morphology.disk(3))
        segmented_body = img_array.copy()
        segmented_body[body_mask_eroded == 0] = 0
        return img_array, body_mask_eroded, segmented_body

    def segment_lungs_smart_fallback(self, img_input, body_mask):
        rows, cols = img_input.shape
        img_float = img_input.astype(float)
        img_gamma = (255 * (img_float / 255) ** 1.5).astype(np.uint8)
        pixels_in_body = img_gamma[body_mask > 0]
        if len(pixels_in_body) == 0: return img_input, np.zeros_like(body_mask)
        
        thresh_val = np.mean(pixels_in_body) - (0.3 * np.std(pixels_in_body))
        binary = (img_gamma < thresh_val) & (body_mask > 0)
        if np.sum(binary) < (np.sum(body_mask) * 0.05):
            binary = (img_gamma < np.percentile(pixels_in_body, 45)) & (body_mask > 0)
            
        binary[:int(rows * 0.12), :] = 0
        binary = morphology.binary_closing(binary, morphology.disk(6))
        binary = morphology.binary_opening(binary, morphology.disk(4))
        binary[:, cols//2-3 : cols//2+3] = 0
        
        label_img = measure.label(binary)
        regions = measure.regionprops(label_img)
        candidates = [r for r in regions if r.area > 500 and r.centroid[0] > (rows * 0.12)]
        candidates.sort(key=lambda x: x.area, reverse=True)
        
        mask_combined = np.zeros_like(binary)
        if not candidates:
            mask_combined = morphology.binary_erosion(body_mask, morphology.disk(20))
            mask_combined[:, cols//2-5:cols//2+5] = 0
        else:
            for region in candidates[:2]:
                temp = np.zeros_like(binary)
                temp[region.coords[:,0], region.coords[:,1]] = 1
                mask_combined |= (morphology.convex_hull_image(temp) & body_mask)
        return img_gamma, mask_combined

    def extract_features(self, image):
        if image is None: return [0]*10
        glcm = graycomatrix(image, [1], [0], 256, symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]
        correlation = graycoprops(glcm, 'correlation')[0, 0]
        entropy_val = -np.sum((glcm/(np.sum(glcm)+1e-10)) * np.log2((glcm/(np.sum(glcm)+1e-10)) + 1e-10))
        
        h = image.shape[0]
        cutoff = int(h * 0.5)
        mu = np.mean(image[:cutoff][image[:cutoff] > 0]) if np.any(image[:cutoff] > 0) else 0
        ml = np.mean(image[cutoff:][image[cutoff:] > 0]) if np.any(image[cutoff:] > 0) else 0
        
        blobs = blob_log(image, max_sigma=30, num_sigma=10, threshold=0.1)
        gy, gx = np.gradient(image)
        edge = np.mean(np.hypot(gx, gy)[image > 0]) if np.any(image > 0) else 0
        
        return [contrast, homogeneity, energy, correlation, entropy_val, mu, ml, mu/(ml+1e-5), len(blobs), edge]

    def analyze_lesions(self, img, mask):
        img_masked = img_as_float(img)
        img_masked[mask == 0] = 0
        try: segments = felzenszwalb(img_masked, scale=40, sigma=0.5, min_size=50)
        except: segments = np.zeros_like(mask)
            
        masks = {k: np.zeros_like(img, dtype=float) for k in LESION_COLORS.keys()}
        lung_pix = img[mask > 0]
        if len(lung_pix) == 0: return masks
        
        gm, gs = np.mean(lung_pix), np.std(lung_pix)
        for seg_id in np.unique(segments):
            if seg_id == 0: continue
            s_mask = (segments == seg_id)
            if np.sum(s_mask & mask)/np.sum(s_mask) < 0.6: continue
            
            vals = img[s_mask]
            m_val, var = np.mean(vals), np.var(vals)
            
            if m_val > (gm + 2.0 * gs): masks['calcification'][s_mask] = 1.0
            elif m_val > (gm + 0.8 * gs) and var < 500: masks['effusion'][s_mask] = 1.0
            elif m_val < (gm - 0.9 * gs): masks['cavity'][s_mask] = 1.0
            elif m_val > gm and var >= 500: masks['infiltrate'][s_mask] = 1.0
        return masks

    def process(self, img_array):
        if not self.load_models(): return {"error": "Model Gusna tidak ditemukan!"}
        
        if len(img_array.shape) == 3: img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else: img = img_array
        img = cv2.resize(img, (512, 512))
        
        _, body_mask, body_img = self.segment_body_robust(img)
        img_enh, mask = self.segment_lungs_smart_fallback(body_img, body_mask)
        
        if np.sum(mask) == 0: return {"error": "Segmentasi Gagal"}
        
        img_roi = img.copy(); img_roi[mask == 0] = 0
        feats = self.extract_features(img_roi)
        pred = self.model.predict(self.scaler.transform([feats]))[0]
        conf = self.model.predict_proba(self.scaler.transform([feats]))[0][pred]
        
        lesions = None
        stats = {}
        if pred == 1:
            lesions = self.analyze_lesions(img, mask)
            total = np.sum(mask)
            for k, v in lesions.items(): stats[k] = (np.sum(v)/total)*100
            
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        vis[mask > 0] = [255, 0, 255]
        vis = cv2.addWeighted(vis, 0.3, cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), 0.7, 0)
        
        if lesions:
            for k, v in lesions.items():
                sm = gaussian_filter(v, sigma=1.0) > 0.1
                if np.sum(sm) > 0:
                    vis[sm] = LESION_COLORS[k]
                    
        return {"prediction": "Tuberculosis" if pred == 1 else "Normal", "confidence": conf, "overlay": vis, "stats": stats}


# ==========================================
# MODEL 2: LOGIKA SAPTO (Class Wrapper)
# ==========================================
class ModelSapto:
    def __init__(self):
        self.scaler = None
        self.model = None

    def load_models(self):
        try:
            self.scaler = joblib.load('models/scaler_paru3_baru.pkl')
            self.model = joblib.load('models/model_svm_paru3_baru.pkl')
            return True
        except: return False

    def masking02(self, input_img):
        blur = cv2.GaussianBlur(input_img, (9,9), 0)
        cl = cv2.createCLAHE(2.0, (8,8)).apply(blur)
        _, otsu_res = cv2.threshold(cl, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Frame Cut logic simplified
        h, w = input_img.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(mask, (0,0), (w//2-17, h), 1, -1)
        cv2.rectangle(mask, (w//2+18,0), (w, h), 1, -1)
        masked_otsu = cv2.bitwise_and(mask*255, otsu_res)
        
        padded = cv2.copyMakeBorder(masked_otsu, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=[255])
        
        # Body Mask simplified
        inv = cv2.bitwise_not(padded)
        n, l, s, _ = cv2.connectedComponentsWithStats(inv)
        if n > 1:
            largest = (l == (np.argmax(s[1:, cv2.CC_STAT_AREA]) + 1)).astype(np.uint8)*255
            hull = cv2.convexHull(cv2.findContours(largest, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0])
            hull_img = np.zeros_like(largest); cv2.drawContours(hull_img, [hull], -1, 255, -1)
            largest = cv2.erode(hull_img, np.ones((21,21), np.uint8))
        else: largest = np.zeros_like(padded)
        
        out = cv2.bitwise_and(padded, largest)
        out = cv2.dilate(out, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations=3)
        
        # CCA Dist
        n, l, s, c = cv2.connectedComponentsWithStats(out)
        if n > 1:
            cands = [{'idx':i, 'area':s[i, cv2.CC_STAT_AREA], 'dist':math.hypot(c[i][0]-w/2, c[i][1]-h/2)} for i in range(1, n)]
            cands.sort(key=lambda x: x['area'], reverse=True); top = cands[:5]
            top.sort(key=lambda x: x['dist']); final = top[:2]
            out = np.zeros_like(out)
            for f in final: out[l == f['idx']] = 255
            
        # Hull Fill
        hull_mask = np.zeros_like(out)
        for cnt in cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            cv2.drawContours(hull_mask, [cv2.convexHull(cnt)], -1, 255, -1)
            
        return hull_mask[12:h+12, 12:w+12]

    def extract_features(self, img, mask):
        x, y, w, h = cv2.boundingRect(mask)
        if w == 0: return np.zeros(20)
        roi = img[y:y+h, x:x+w]; roi_mask = mask[y:y+h, x:x+w]
        masked_roi = cv2.bitwise_and(roi, roi, mask=roi_mask)
        pix = roi[roi_mask > 0]
        if len(pix) == 0: return np.zeros(20)
        
        feats = [np.mean(pix), np.std(pix), skew(pix), kurtosis(pix)]
        glcm = graycomatrix(masked_roi, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], 256, symmetric=True, normed=True)
        feats += [np.mean(graycoprops(glcm, p)) for p in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']]
        
        lbp = local_binary_pattern(roi, 8, 1, 'uniform')
        hist, _ = np.histogram(lbp[roi_mask > 0], bins=np.arange(0, 11), density=True)
        feats += [np.sum(hist**2), entropy(hist)]
        
        props = regionprops(label(roi_mask))
        if props: 
            p = max(props, key=lambda x: x.area)
            feats += [p.solidity, p.eccentricity, p.extent]
        else: feats += [0, 0, 0]
        
        try:
            fd = hog(cv2.resize(roi, (64, 128)), orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2))
            feats += [np.mean(fd), np.std(fd), skew(fd)]
        except: feats += [0, 0, 0]
        
        feats += [np.count_nonzero((roi>200)&(roi_mask>0))/len(pix), np.count_nonzero((roi<50)&(roi_mask>0))/len(pix)]
        return np.array(feats).reshape(1, -1)

    def process(self, img_array):
        if not self.load_models(): return {"error": "Model Sapto tidak ditemukan!"}
        
        if len(img_array.shape) == 3: img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else: img = img_array
        img = cv2.resize(img, (512, 512))
        
        mask = self.masking02(img)
        if np.sum(mask) == 0: return {"error": "Segmentasi Gagal"}
        
        img_roi = img.copy(); img_roi[mask == 0] = 0
        feats = self.extract_features(img_roi, mask)
        pred = self.model.predict(self.scaler.transform(feats))[0]
        conf = self.model.predict_proba(self.scaler.transform(feats))[0][pred]
        
        lesions = None
        stats = {}
        if pred == 1:
            # Reuse Gusna's lesion logic for Sapto as placeholder/standard
            gusna_logic = ModelGusna() 
            lesions = gusna_logic.analyze_lesions(img, mask)
            total = np.sum(mask)
            for k, v in lesions.items(): stats[k] = (np.sum(v)/total)*100
            
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        vis[mask > 0] = [255, 0, 255]
        vis = cv2.addWeighted(vis, 0.3, cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), 0.7, 0)
        
        if lesions:
            for k, v in lesions.items():
                if np.sum(v) > 0:
                    vis[gaussian_filter(v, 1.0) > 0.1] = LESION_COLORS[k]
                    
        return {"prediction": "Tuberculosis" if pred == 1 else "Normal", "confidence": conf, "overlay": vis, "stats": stats}

# ==========================================
# MODEL 3: LOGIKA SAYA (Class Wrapper)
# ==========================================
class ModelSaya:
    def __init__(self):
        self.scaler = None
        self.model = None

    def load_models(self):
        try:
            self.scaler = joblib.load('models/scaler_3mini.pkl')
            self.model = joblib.load('models/svm_tb_model_3mini.pkl')
            return True
        except: return False

    def segment_lungs_v26(self, img):
        # Simplified V26 Pipeline
        img_f = img.astype(float) / 255.0
        img_enh = exposure.equalize_adapthist(cv2.bilateralFilter(img, 9, 75, 75).astype(float)/255, clip_limit=0.02)
        
        try: thr = filters.threshold_otsu(img_enh)
        except: thr = 0.5
        mask = morphology.binary_closing(img_enh > thr, morphology.disk(5))
        mask = binary_fill_holes(mask)
        mask[:, :int(img.shape[1]*0.05)] = 0
        mask[:, -int(img.shape[1]*0.05):] = 0
        
        # Kmeans
        pix = img_enh[mask].reshape(-1, 1)
        if len(pix) < 100: return None, None
        km = KMeans(n_clusters=2, n_init=5).fit(pix)
        lung_raw = (mask) & (img_enh.reshape(-1) == km.cluster_centers_.argmin()).reshape(img.shape)
        
        # Seed & Watershed
        seed = morphology.binary_erosion(lung_raw, morphology.disk(10))
        markers = np.zeros_like(img, dtype=np.int32)
        markers[seed] = 1; markers[~mask] = 2
        ws = segmentation.watershed(filters.sobel(gaussian_filter(img_enh, 3)), markers)
        
        final = np.zeros_like(img, dtype=np.uint8)
        lbl = measure.label(binary_fill_holes(ws == 1))
        for p in sorted(measure.regionprops(lbl), key=lambda x: x.area, reverse=True)[:2]:
            l = (lbl == p.label)
            final[morphology.convex_hull_image(l) & morphology.binary_dilation(l, morphology.disk(10))] = 1
            
        # Smooth
        blur = cv2.GaussianBlur((final*255).astype(np.uint8), (25,25), 0)
        _, sm = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
        return sm, img_enh

    def extract_features(self, img, mask):
        roi = img[mask > 0]
        if len(roi) == 0: return np.zeros(22)
        
        f = [np.mean(roi), np.std(roi), np.max(roi), np.min(roi), skew(roi), kurtosis(roi)]
        
        img_u8 = (img * 255).astype(np.uint8)
        r, c = np.where(mask > 0)
        roi_crop = img_u8[r.min():r.max(), c.min():c.max()]
        glcm = graycomatrix(roi_crop, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], 256, symmetric=True, normed=True)
        f += [np.mean(graycoprops(glcm, p)) for p in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']]
        
        lbl = measure.label(mask)
        props = regionprops(lbl)
        if props:
            f += [sum(p.area for p in props), np.mean([p.solidity for p in props]), 
                  np.mean([p.extent for p in props]), np.mean([p.eccentricity for p in props]),
                  (4*np.pi*sum(p.area for p in props))/(sum(p.perimeter for p in props)**2 + 1e-5)]
        else: f += [0]*5
        
        img_h = img.copy(); img_h[mask == 0] = 0
        fd = hog(cv2.resize(img_h, (128, 128)), orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1))
        f += [np.mean(fd), np.std(fd), np.max(fd), kurtosis(fd)]
        
        return np.array(f).reshape(1, -1)

    def detect_lesions(self, img_enh, mask):
        masks = {}; roi = img_enh[mask > 0]; m, s = np.mean(roi), np.std(roi)
        masks['calcification'] = morphology.remove_small_objects((img_enh > m+3*s) & (mask>0), 10)
        
        top = (mask>0).copy(); top[int(img_enh.shape[0]*0.6):, :] = 0
        cav = (img_enh < m-1.5*s) & top
        masks['cavity'] = np.zeros_like(mask)
        for p in regionprops(measure.label(cav)):
            if p.area > 150 and p.eccentricity < 0.9 and p.solidity > 0.85: masks['cavity'][measure.label(cav)==p.label] = 1
            
        try: ent = rank.entropy((img_enh*255).astype(np.uint8), morphology.disk(5), mask=mask)
        except: ent = np.zeros_like(img_enh)
        masks['infiltrate'] = morphology.binary_opening((ent > 5.5) & (mask>0) & (~masks['calcification']), morphology.disk(3))
        
        btm = (mask>0).copy(); btm[:int(img_enh.shape[0]*0.75), :] = 0
        masks['effusion'] = np.zeros_like(mask)
        if np.sum(btm) > 0:
            lbl = measure.label(btm)
            for p in regionprops(lbl):
                if p.area > 500:
                    s = (lbl == p.label)
                    masks['effusion'] |= morphology.binary_opening(morphology.convex_hull_image(s) & (~s), morphology.disk(5))
        return masks

    def process(self, img_array):
        if not self.load_models(): return {"error": "Model Saya tidak ditemukan!"}
        
        if len(img_array.shape) == 3: img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else: img = img_array
        img = cv2.resize(img, (512, 512))
        
        mask, img_enh = self.segment_lungs_v26(img)
        if mask is None: return {"error": "Segmentasi Gagal"}
        _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        img_float = img.astype(np.float32)/255.0
        feats = self.extract_features(img_float, mask_bin)
        pred = self.model.predict(self.scaler.transform(feats))[0]
        conf = self.model.predict_proba(self.scaler.transform(feats))[0][pred]
        
        lesions = None; stats = {}
        if pred == 1:
            lesions = self.detect_lesions(img_enh, mask_bin)
            total = np.count_nonzero(mask_bin)
            for k, v in lesions.items(): stats[k] = (np.count_nonzero(v)/total)*100 if total>0 else 0
            
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        vis[mask_bin > 0] = [255, 0, 255]
        vis = cv2.addWeighted(vis, 0.3, cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), 0.7, 0)
        
        if lesions:
            for k, v in lesions.items():
                if np.sum(v) > 0:
                    vis[v > 0] = LESION_COLORS[k]
                    
        return {"prediction": "Tuberculosis" if pred == 1 else "Normal", "confidence": conf, "overlay": vis, "stats": stats}

# ==========================================
# STREAMLIT UI
# ==========================================

# Session Init
if 'page' not in st.session_state: st.session_state.page = 'Home'
if 'logged_in' not in st.session_state: st.session_state.logged_in = False
if 'username' not in st.session_state: st.session_state.username = None

# Navbar
with st.container():
    c1, c2, c3 = st.columns([6, 1, 1])
    with c1: st.markdown("## 🫁 TB Detective")
    with c2: 
        if st.button("🏠 Home", use_container_width=True): st.session_state.page = 'Home'; st.rerun()
    with c3:
        if st.session_state.logged_in:
            if st.button("🚪 Logout", use_container_width=True):
                st.session_state.logged_in = False; st.session_state.username = None; st.session_state.page = 'Home'; st.rerun()
        else:
            if st.button("🔐 Login", use_container_width=True): st.session_state.page = 'Login'; st.rerun()
    st.divider()

# Pages
def show_home():
    c1, c2 = st.columns([2, 1])
    with c1: 
        st.markdown("""
        ### Deteksi Dini Tuberkulosis Berbasis PCD
        
        Selamat datang di **TB Screening**. Aplikasi ini dirancang untuk membantu analisis citra X-Ray dada menggunakan Pengolahan Citra Digital.
        
        **Fitur:**
        * ✅ **Multi-Model:** Pilih algoritma deteksi dari berbagai pengembang.
        * ✅ **Analisis Lesi:** Mendeteksi Infiltrat, Kavitas, Kalsifikasi, dan Efusi secara visual.
        """)
        st.image("https://cdn.who.int/media/images/default-source/products/global-reports/tb-report/2025/black-tiles-(ig--fb)-(1).png", use_container_width=False, width=300)
    with c2:
        if st.session_state.logged_in:
            st.success(f"Login sebagai: {st.session_state.username}")
            if st.button("🚀 Mulai Deteksi", type="primary", use_container_width=True): st.session_state.page = 'Detect'; st.rerun()
        else:
            st.info("Silakan Login/Tamu untuk mulai."); 
            if st.button("Mulai", type="primary", use_container_width=True): st.session_state.page = 'Login'; st.rerun()

def show_login():
    t1, t2, t3 = st.tabs(["Login", "Daftar", "Tamu"])
    with t1:
        u = st.text_input("User"); p = st.text_input("Pass", type="password")
        if st.button("Masuk", use_container_width=True):
            if verify_login(u, p): st.session_state.logged_in=True; st.session_state.username=u; st.session_state.page='Detect'; st.rerun()
            else: st.error("Gagal Login")
    with t2:
        nu = st.text_input("New User"); np = st.text_input("New Pass", type="password")
        if st.button("Daftar", use_container_width=True):
            if nu and np: save_user(nu, np); st.success("Berhasil! Silakan Login.")
    with t3:
        if st.button("Masuk Tamu", use_container_width=True):
            st.session_state.logged_in=True; st.session_state.username="Tamu"; st.session_state.page='Detect'; st.rerun()

def show_detect():
    st.markdown(f"### Analisis X-Ray (User: {st.session_state.username})")
    c1, c2 = st.columns([1, 2])
    with c1:
        model_opt = st.selectbox("Pilih Model", ["Model Saya (V26)", "Model Gusna", "Model Sapto"])
        uploaded = st.file_uploader("Upload CXR", type=['jpg', 'png', 'jpeg'])
        btn = st.button("🔍 Deteksi", type="primary", use_container_width=True) if uploaded else None
        
    with c2:
        if uploaded:
            file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            img_cv = cv2.imdecode(file_bytes, 1)
            if not btn: st.image(img_cv, caption="Preview", width=300, channels="BGR")
            else:
                with st.spinner("Sedang memproses..."):
                    if model_opt == "Model Saya (V26)": runner = ModelSaya()
                    elif model_opt == "Model Gusna": runner = ModelGusna()
                    elif model_opt == "Model Sapto": runner = ModelSapto()
                    
                    res = runner.process(img_cv)
                    
                if "error" in res: st.error(res['error'])
                else:
                    st.divider()
                    rc1, rc2 = st.columns(2)
                    with rc1: st.image(res['overlay'], caption=f"Hasil {model_opt}", use_container_width=True)
                    with rc2:
                        lbl = res['prediction']
                        st.markdown(f"### {'🔴' if lbl=='Tuberculosis' else '🟢'} {lbl}")
                        st.progress(res['confidence']); st.caption(f"Confidence: {res['confidence']*100:.1f}%")
                        if res.get('stats'):
                            st.write("📊 **Deteksi Lesi (% Area):**")
                            sc1, sc2 = st.columns(2)
                            s = res['stats']
                            sc1.metric("Infiltrate (Cyan)", f"{s.get('infiltrate',0):.1f}%")
                            sc1.metric("Cavity (Merah)", f"{s.get('cavity',0):.1f}%")
                            sc2.metric("Calcification (Kuning)", f"{s.get('calcification',0):.1f}%")
                            sc2.metric("Effusion (Hijau)", f"{s.get('effusion',0):.1f}%")

# Routing
if st.session_state.page == 'Home': show_home()
elif st.session_state.page == 'Login': show_login()
elif st.session_state.page == 'Detect': 
    if st.session_state.logged_in: show_detect()
    else: st.session_state.page = 'Login'; st.rerun()
