import streamlit as st

# Thiết lập cấu hình giao diện
st.set_page_config(page_title="HearMe", page_icon="📷", layout="wide")

# Header với menu
st.markdown("""
    <style>
        .header-container {
            background-color: #1565C0;
            padding: 20px;
            color: white;
            text-align: center;
            font-size: 25px;
            border-radius: 5px;
        }
        .footer {
            text-align: center;
            padding: 10px;
            background-color: #1565C0;
            color: white;
            font-size: 15px;
            margin-top: 30px;
            border-radius: 5px;
        }
        .menu a {
            color: white;
            text-decoration: none;
            padding: 0 15px;
            font-size: 18px;
        }
    </style>
    <div class="header-container">
        <span>HearMe</span>
        <div class="menu" style="float:right; margin-top: -30px;">
            <a href="#">Home</a>
            <a href="#">About</a>
            <a href="#">Contact</a>
        </div>
    </div>
""", unsafe_allow_html=True)

# Section tiêu đề chính
st.markdown("""
    <div style="text-align: center; background-color: #E3F2FD; padding: 20px; margin: 10px 0; border-radius: 10px;">
        <h1>Sign Language Translator</h1>
        <p style="font-size: 18px; color: #546E7A;">Breaking barriers in communication</p>
    </div>
""", unsafe_allow_html=True)

# Tabs chuyển đổi
tab1, tab2 = st.tabs(["Sign to Language", "Language to Sign"])

with tab1:
    st.markdown("""
        <div style="border: 2px dashed #B0BEC5; padding: 50px; text-align: center; color: #90A4AE;">
            <p><i class="fas fa-camera"></i></p>
            <p>Upload a sign or use your camera to translate</p>
        </div>
    """, unsafe_allow_html=True)

with tab2:
    st.markdown("""
        <div style="border: 2px dashed #B0BEC5; padding: 50px; text-align: center; color: #90A4AE;">
            <p><i class="fas fa-pencil-alt"></i></p>
            <p>Type or paste your text to see the sign translation</p>
        </div>
    """, unsafe_allow_html=True)

# Input kết quả dịch
st.markdown("<h3>Translated Text</h3>", unsafe_allow_html=True)
st.text_area("Translated Text", placeholder="Translation will appear here...")

# Footer
st.markdown("""
    <div class="footer">
        <p>© 2023 HearMe. All rights reserved.</p>
        <p>
            <a href="#" style="color: white; padding-right: 15px;">Facebook</a>
            <a href="#" style="color: white; padding-right: 15px;">Instagram</a>
            <a href="#" style="color: white;">Twitter</a>
        </p>
    </div>
""", unsafe_allow_html=True)
