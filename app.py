# app.py
import streamlit as st
import json
import os
import numpy as np
import cv2
from PIL import Image

# Import Modul Model
import tb_utils_1 as model_1
import tb_utils_2 as model_2
import tb_util_3b as model_3
# import tb_utils_3 as model_4

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="TB Detective",
    page_icon="🫁",
    layout="wide"
)

# --- USER MANAGEMENT (JSON BASED) ---
USER_DB = 'users.json'

def load_users():
    if not os.path.exists(USER_DB):
        return {}
    with open(USER_DB, 'r') as f:
        return json.load(f)

def save_user(username, password):
    users = load_users()
    users[username] = password
    with open(USER_DB, 'w') as f:
        json.dump(users, f)

def verify_login(username, password):
    users = load_users()
    if username in users and users[username] == password:
        return True
    return False

# --- SESSION STATE INITIALIZATION ---
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = None
if 'page' not in st.session_state:
    st.session_state['page'] = 'Home'

# --- NAVBAR COMPONENT ---
def render_navbar():
    st.markdown("""
        <style>
        .nav-btn { margin-top: 10px; }
        </style>
    """, unsafe_allow_html=True)

    with st.container():
        col1, col2, col3 = st.columns([6, 1, 1])
        
        with col1:
            st.markdown("## 🫁 TB Screening")
        
        with col2:
            if st.button("🏠 Home", use_container_width=True):
                st.session_state['page'] = 'Home'
                st.rerun()
                
        with col3:
            if st.session_state['logged_in']:
                if st.button("🚪 Logout", use_container_width=True):
                    st.session_state['logged_in'] = False
                    st.session_state['username'] = None
                    st.session_state['page'] = 'Home'
                    st.rerun()
            else:
                if st.button("🔐 Login", use_container_width=True):
                    st.session_state['page'] = 'Login'
                    st.rerun()
        
        st.divider()

# --- HALAMAN-HALAMAN ---

def show_home():
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Deteksi Dini Tuberkulosis Berbasis AI
        
        Selamat datang di **TB Screening**. Aplikasi ini dirancang untuk membantu analisis citra X-Ray dada menggunakan teknologi *Computer Vision* dan *Machine Learning* terkini.
        
        **Fitur Unggulan:**
        * ✅ **Multi-Model AI:** Pilih algoritma deteksi dari berbagai pengembang.
        * ✅ **Analisis Lesi:** Mendeteksi Infiltrat, Kavitas, Kalsifikasi, dan Efusi secara visual.
        * ✅ **Akurasi Klinis:** Dilatih dengan ribuan dataset terverifikasi.
        """)
        st.image("https://cdn.who.int/media/images/default-source/products/global-reports/tb-report/2025/black-tiles-(ig--fb)-(1).png", use_container_width=False, width=300)
    
    with col2:
        st.info("💡 **Status Pengguna**")
        if st.session_state['logged_in']:
            st.success(f"Anda login sebagai: **{st.session_state['username']}**")
            if st.button("🚀 Ke Halaman Deteksi", type="primary", use_container_width=True):
                st.session_state['page'] = 'Detect'
                st.rerun()
        else:
            st.warning("Anda belum login.")
            st.write("Silakan Login atau masuk sebagai Tamu untuk memulai.")
            if st.button("Mulai Sekarang", type="primary", use_container_width=True):
                st.session_state['page'] = 'Login'
                st.rerun()

def show_login():
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.subheader("Akses Sistem")
        tab1, tab2, tab3 = st.tabs(["Login Akun", "Registrasi Baru", "Mode Tamu"])
        
        with tab1: # LOGIN
            username = st.text_input("Username", key="login_user")
            password = st.text_input("Password", type="password", key="login_pass")
            if st.button("Masuk", use_container_width=True):
                if verify_login(username, password):
                    st.session_state['logged_in'] = True
                    st.session_state['username'] = username
                    st.session_state['page'] = 'Detect'
                    st.success("Login Berhasil!")
                    st.rerun()
                else:
                    st.error("Username atau Password salah.")
                    
        with tab2: # REGISTER
            new_user = st.text_input("Buat Username Baru", key="reg_user")
            new_pass = st.text_input("Buat Password Baru", type="password", key="reg_pass")
            if st.button("Daftar Akun", use_container_width=True):
                users = load_users()
                if new_user in users:
                    st.warning("Username sudah terpakai.")
                elif new_user and new_pass:
                    save_user(new_user, new_pass)
                    st.success("Registrasi berhasil! Silakan Login.")
                else:
                    st.warning("Mohon isi username dan password.")
                    
        with tab3: # TAMU
            st.write("Masuk sebagai tamu untuk mencoba fitur tanpa menyimpan riwayat.")
            if st.button("Masuk sebagai Tamu", type="primary", use_container_width=True):
                st.session_state['logged_in'] = True
                st.session_state['username'] = "Guest"
                st.session_state['page'] = 'Detect'
                st.rerun()

def show_detect():
    st.markdown(f"### 👋 Halo, {st.session_state['username']}")
    st.write("Silakan unggah citra X-Ray dan pilih model analisis.")
    
    # Layout Input
    with st.container():
        c1, c2 = st.columns([1, 2])
        
        with c1:
            st.info("⚙️ **Konfigurasi Model**")
            # 1. PILIH MODEL (Sekarang di halaman utama, bukan sidebar)
            model_option = st.selectbox(
                "Pilih Model AI", 
                ["Model 1", "Model 2", "Model 3"]
            )
            
            # 2. UPLOAD
            uploaded = st.file_uploader("Upload Citra CXR", type=['png', 'jpg', 'jpeg'])
            
            analyze_btn = False
            if uploaded:
                analyze_btn = st.button("🔍 DETEKSI SEKARANG", type="primary", use_container_width=True)

        with c2:
            if uploaded:
                # Convert ke array OpenCV
                file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
                img_cv = cv2.imdecode(file_bytes, 1) # BGR
                img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                
                # Preview kecil sebelum analisa
                if not analyze_btn: # Tampilkan preview jika belum dipencet
                    st.image(img_rgb, caption="Preview Citra", width=300)

                if analyze_btn:
                    with st.spinner(f"Sedang memproses dengan {model_option}..."):
                        
                        # Routing Logika Berdasarkan Pilihan
                        if model_option == "Model 1":
                            result = model_1.process_image(img_cv) 
                        elif model_option == "Model 2":
                            result = model_2.process_image(img_cv)
                        elif model_option == "Model 3":
                            result = model_3.process_image(img_cv)
                    
                    # TAMPILKAN HASIL
                    if "error" in result:
                        st.error(result['error'])
                    else:
                        st.divider()
                        st.subheader("📋 Hasil Analisis")
                        
                        res_col1, res_col2 = st.columns(2)
                        
                        with res_col1:
                            st.image(result['overlay'], caption="Overlay Lesi & Paru", use_container_width=True)
                        
                        with res_col2:
                            # Metrics Box
                            lbl = result['prediction']
                            color_emoji = "🔴" if lbl == "Tuberculosis" else "🟢"
                            
                            st.markdown(f"### {color_emoji} Prediksi: **{lbl}**")
                            st.progress(result['confidence'])
                            st.caption(f"Tingkat Keyakinan AI: {result['confidence']*100:.1f}%")
                            
                            st.divider()
                            
                            # Stats Lesi (Hanya jika ada)
                            if result.get('stats'):
                                st.markdown("##### 📊 Rincian Area Lesi")
                                st.caption("(% terhadap luas paru)")
                                
                                s_c1, s_c2 = st.columns(2)
                                stats = result['stats']
                                
                                with s_c1:
                                    st.metric("🔵 Infiltrate", f"{stats.get('infiltrate',0):.1f}%")
                                    st.metric("🔴 Cavity", f"{stats.get('cavity',0):.1f}%")
                                with s_c2:
                                    st.metric("🟡 Calcification", f"{stats.get('calcification',0):.1f}%")
                                    st.metric("🟢 Effusion", f"{stats.get('effusion',0):.1f}%")
                            else:
                                st.success("Tidak ditemukan lesi spesifik. Paru-paru tampak normal.")

# --- RENDER UI ---
render_navbar()

if st.session_state['page'] == 'Home':
    show_home()
elif st.session_state['page'] == 'Login':
    show_login()
elif st.session_state['page'] == 'Detect':
    if st.session_state['logged_in']:
        show_detect()
    else:
        st.warning("Akses ditolak. Silakan login terlebih dahulu.")
        st.session_state['page'] = 'Login'
        st.rerun()