import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
from severity_model import get_severity
import plotly.graph_objects as go
import os
import time

# -------------------- Load Model --------------------
model = load_model("anemia_model_4class.h5")
class_labels = ['Eye_Anemic','Eye_Non_Anemic','Nail_Anemic','Nail_Non_Anemic']

# -------------------- Page Config --------------------
st.set_page_config(page_title="AI-Based Anemia Detection", layout="wide")

# -------------------- Animated Title --------------------
st.markdown("""
<style>
@keyframes fadeInDown {
    from {opacity:0; transform: translateY(-30px);}
    to {opacity:1; transform: translateY(0);}
}
.animated-title {
    font-size: 42px;
    font-weight: bold;
    text-align: center;
    background: linear-gradient(90deg, #00c6ff, #0072ff, #00c6ff);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: fadeInDown 1.5s ease-out,
               gradientMove 4s linear infinite;
}
@keyframes gradientMove {
    0% {background-position: 0% center;}
    100% {background-position: 200% center;}
}
.subtitle {
    text-align:center;
    font-size:18px;
    color:#dddddd;
}
.suggestion-box {
    background: rgba(255,255,255,0.08);
    padding: 20px;
    border-radius: 15px;
    margin-top: 15px;
    backdrop-filter: blur(10px);
}
.severe { border-left: 6px solid red; }
.moderate { border-left: 6px solid orange; }
.mild { border-left: 6px solid yellow; }
.normal { border-left: 6px solid green; }
</style>

<div class="animated-title">🩺 AI-Based Anemia Detection System</div>
<div class="subtitle">Upload Eye or Nail Images to detect Anemia Type, Confidence and Severity Level</div>
""", unsafe_allow_html=True)

# -------------------- Video Banner --------------------
import base64

video_path = "background.mp4"

if os.path.exists(video_path):
    with open(video_path, "rb") as f:
        video_bytes = f.read()
        video_base64 = base64.b64encode(video_bytes).decode()

    # HTML video player for autoplay, loop, muted, bigger height
    video_html = f"""
    <div style="width:100%; height:220px; overflow:hidden; border-radius:15px; margin:10px 0;">
        <video autoplay muted loop playsinline style="width:100%; height:100%; object-fit:cover;">
            <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
            Your browser does not support the video tag.
        </video>
    </div>
    """
    st.components.v1.html(video_html, height=230)  # height slightly bigger than div
else:
    st.warning("⚠ background.mp4 file not found in project folder.")

# -------------------- File Upload --------------------
uploaded_files = st.file_uploader(
    "📤 Upload Eye / Nail Image",
    type=["jpg", "png", "jpeg"],
    accept_multiple_files=True
)

if uploaded_files:
    results = []
    st.subheader("🖼 Uploaded Images Preview")
    cols = st.columns(3)

    for idx, uploaded_file in enumerate(uploaded_files):
        img = Image.open(uploaded_file).convert('RGB')

        # Crop & resize
        width, height = img.size
        img_cropped = img.crop((width*0.1, height*0.1, width*0.9, height*0.9))
        img_resized = img_cropped.resize((224,224))

        # Show preview
        with cols[idx % 3]:
            st.image(img, caption=uploaded_file.name, width=250)

        # Prediction
        img_array = np.expand_dims(np.array(img_resized)/255.0, axis=0)
        prediction = model.predict(img_array)[0]
        predicted_index = np.argmax(prediction)
        predicted_class = class_labels[predicted_index]
        confidence = float(prediction[predicted_index])
        accuracy_percent = round(confidence*100,2)

        # Severity
        if confidence < 0.7:
            img_type = "Unknown"
            severity = "Unknown"
        else:
            img_type = "Eye" if "eye" in predicted_class.lower() else "Nail"
            severity = get_severity(predicted_class, confidence)

        # ---------- Display Results ----------
        st.markdown(f"### 📊 Result for {uploaded_file.name}")
        st.write(f"Prediction: **{predicted_class}**")
        st.write(f"Confidence: **{accuracy_percent}%**")
        st.write(f"Severity: **{severity}**")

        # ---------- Confidence Chart with Plotly ----------
        colors = ['#ff4c4c','#28a745','#ffa500','#1f77b4']  # Red, Green, Orange, Blue
        fig = go.Figure()

        # Start bars at 0 for animation
        fig.add_trace(go.Bar(
            x=class_labels,
            y=[0,0,0,0],
            marker_color=colors,
            marker_line_color='black',
            marker_line_width=1.5,
            opacity=0.9,
            text=[f'{p*100:.2f}%' for p in prediction],
            textposition='outside'
        ))

        st_plot = st.plotly_chart(fig, use_container_width=True)

        # Animate bars growing
        for i in range(1, 101):
            fig.data[0].y = [p*i/100 for p in prediction]
            st_plot.plotly_chart(fig, use_container_width=True)
            time.sleep(0.01)  # controls animation speed

        # ---------- Suggestion Box ----------
        if severity.lower() == "severe anemia":
            st.markdown("""
            <div class="suggestion-box severe">
            <b>⚠ Severe Anemia - Immediate Action Required</b><br>
            • Immediately consult a doctor<br>
            • Blood test (Hb, Iron, Ferritin)<br>
            • Iron supplements as prescribed<br>
            • Eat spinach, beetroot, pomegranate, dates<br>
            • Take Vitamin C with iron<br>
            </div>
            """, unsafe_allow_html=True)

        elif severity.lower() == "moderate anemia":
            st.markdown("""
            <div class="suggestion-box moderate">
            <b>⚠ Moderate Anemia</b><br>
            • Include iron rich foods daily<br>
            • Green leafy vegetables<br>
            • Lentils, beans<br>
            • Avoid tea/coffee after meals<br>
            </div>
            """, unsafe_allow_html=True)

        elif severity.lower() == "mild anemia":
            st.markdown("""
            <div class="suggestion-box mild">
            <b>⚠ Mild Anemia</b><br>
            • Improve diet<br>
            • Eat fruits and vegetables<br>
            • Regular checkup<br>
            </div>
            """, unsafe_allow_html=True)

        elif severity.lower() == "normal":
            st.markdown("""
            <div class="suggestion-box normal">
            <b>✅ No Anemia Detected</b><br>
            • Maintain balanced diet<br>
            • Regular health checkup<br>
            </div>
            """, unsafe_allow_html=True)

        # Save results
        results.append({
            "Filename": uploaded_file.name,
            "Type": img_type,
            "Predicted Class": predicted_class,
            "Confidence (%)": accuracy_percent,
            "Severity": severity
        })

    # ---------- Summary Table ----------
    st.subheader("📋 Summary Table")
    st.table(results)