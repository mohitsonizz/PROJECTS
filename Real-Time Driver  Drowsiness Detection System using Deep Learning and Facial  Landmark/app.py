import streamlit as st
import cv2
import dlib
import numpy as np
from tensorflow.keras.models import load_model
import time
import winsound
import math

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Driver Drowsiness System", page_icon="🚗", layout="wide")

# --- CUSTOM CSS FOR STYLING ---
st.markdown("""
    <style>
    .big-font { font-size:25px !important; font-weight: bold; }
    .alert-font { font-size:30px !important; font-weight: bold; color: red; }
    .mar-font { font-size:22px !important; font-weight: bold; color: purple; }
    .score-font { font-size:20px !important; font-weight: bold; color: teal; }
    </style>
    """, unsafe_allow_html=True)

st.title("🚗 Real-Time Driver Drowsiness Detection System")
st.markdown("---")

# --- UI LAYOUT (2 Columns) ---
col1, col2 = st.columns([6, 4])

with col1:
    st.markdown("### 📷 Live Camera Feed")
    FRAME_WINDOW = st.image([]) 

with col2:
    st.markdown("### 📊 Real-Time Metrics")
    status_box = st.empty()
    score_box = st.empty()     # Naya box: L aur R ka exact score dikhane ke liye
    mar_box = st.empty()       
    blink_box = st.empty()
    yawn_box = st.empty()
    alert_box = st.empty()

def crop_eye(frame, landmarks, points):
    x_coords = [landmarks.part(i).x for i in points]
    y_coords = [landmarks.part(i).y for i in points]
    x_min, x_max = max(0, min(x_coords) - 5), max(x_coords) + 5
    y_min, y_max = max(0, min(y_coords) - 5), max(y_coords) + 5
    return frame[y_min:y_max, x_min:x_max], (x_min, y_min, x_max, y_max)

def calculate_mar(landmarks):
    p50, p51, p52 = (landmarks.part(50).x, landmarks.part(50).y), (landmarks.part(51).x, landmarks.part(51).y), (landmarks.part(52).x, landmarks.part(52).y)
    p58, p57, p56 = (landmarks.part(58).x, landmarks.part(58).y), (landmarks.part(57).x, landmarks.part(57).y), (landmarks.part(56).x, landmarks.part(56).y)
    p48, p54 = (landmarks.part(48).x, landmarks.part(48).y), (landmarks.part(54).x, landmarks.part(54).y)
    
    d1, d2, d3 = math.dist(p50, p58), math.dist(p51, p57), math.dist(p52, p56)
    d4 = math.dist(p48, p54)
    return (d1 + d2 + d3) / (3.0 * d4)

@st.cache_resource
def load_all_models():
    model = load_model('blink_model.h5')
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    return model, detector, predictor

model, detector, predictor = load_all_models()

# --- SIDEBAR SETTINGS (LIVE CALIBRATION) ---
st.sidebar.markdown("### ⚙️ Live Settings / Calibration")
# Default value ko 0.40 kar diya hai. Agar phir bhi problem aaye, toh live slider aage-peeche karein
EYE_THRESH = st.sidebar.slider("Eye Closed Sensitivity", min_value=0.05, max_value=0.90, value=0.40, step=0.05, help="Band aankh ko open bata raha ho, toh isse thoda badhayein.")
YAWN_THRESH = st.sidebar.slider("Yawning Limit (MAR)", min_value=0.30, max_value=0.80, value=0.45, step=0.05)
st.sidebar.markdown("---")

# --- START SYSTEM BUTTON ---
run = st.sidebar.checkbox("🟢 START SYSTEM")

if run:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened() or not cap.read()[0]:
        cap = cv2.VideoCapture(1)

    blink_count, yawn_count = 0, 0
    state, is_yawning = "Open", False
    eye_closed_start_time = None
    ALARM_DURATION = 2.0
    
    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Camera not working!")
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), dtype=np.uint8)
        
        faces = detector(gray)
        alert_msg = ""
        current_mar = 0.0  
        l_pred, r_pred = 1.0, 1.0  # Default value

        for face in faces:
            landmarks = predictor(gray, face)
            try:
                # 1. Aankh ka logic
                left_eye_img, left_box = crop_eye(gray, landmarks, range(36, 42))
                right_eye_img, right_box = crop_eye(gray, landmarks, range(42, 48))

                l_eye = cv2.resize(left_eye_img, (24, 24)) / 255.0
                l_eye = l_eye.reshape(1, 24, 24, 1)
                l_pred = model.predict(l_eye, verbose=0)[0][0]
                
                r_eye = cv2.resize(right_eye_img, (24, 24)) / 255.0
                r_eye = r_eye.reshape(1, 24, 24, 1)
                r_pred = model.predict(r_eye, verbose=0)[0][0]
                
                cv2.rectangle(frame_rgb, (left_box[0], left_box[1]), (left_box[2], left_box[3]), (0, 255, 0), 2)
                cv2.rectangle(frame_rgb, (right_box[0], right_box[1]), (right_box[2], right_box[3]), (0, 255, 0), 2)
                
                # Dynamic Threshold ka use kar rahe hain
                if l_pred < EYE_THRESH and r_pred < EYE_THRESH:
                    state = "Closed (Drowsy)"
                    if eye_closed_start_time is None:
                        eye_closed_start_time = time.time()
                    elif time.time() - eye_closed_start_time >= ALARM_DURATION:
                        alert_msg = "⚠️ CRITICAL: DROWSINESS DETECTED!"
                        winsound.Beep(2500, 300)
                else:
                    state = "Open (Active)"
                    if eye_closed_start_time is not None:
                        if time.time() - eye_closed_start_time < ALARM_DURATION:
                            blink_count += 1
                        eye_closed_start_time = None
                    
                # 2. Munh ka logic
                current_mar = calculate_mar(landmarks)
                mouth_x = [landmarks.part(i).x for i in range(48, 68)]
                mouth_y = [landmarks.part(i).y for i in range(48, 68)]
                cv2.rectangle(frame_rgb, (min(mouth_x)-5, min(mouth_y)-5), (max(mouth_x)+5, max(mouth_y)+5), (255, 255, 0), 2)

                if current_mar > YAWN_THRESH:
                    is_yawning = True
                    alert_msg = "🥱 WARNING: YAWNING DETECTED!"
                elif is_yawning:
                    yawn_count += 1
                    is_yawning = False

            except Exception:
                pass
                
        # --- UI UPDATE ---
        FRAME_WINDOW.image(frame_rgb, channels="RGB")
        
        state_color = "green" if "Active" in state else "red"
        
        status_box.markdown(f"<p class='big-font'>Driver State: <span style='color:{state_color};'>{state}</span></p>", unsafe_allow_html=True)
        
        # Nayi Line: L aur R ka exact AI score dikhayegi
        score_box.markdown(f"<p class='score-font'>👁️ AI Score -> Left Eye: {l_pred:.2f} | Right Eye: {r_pred:.2f}</p>", unsafe_allow_html=True)
        
        mar_box.markdown(f"<p class='mar-font'>👄 Live MAR (Mouth Khulna): {current_mar:.2f}</p>", unsafe_allow_html=True)
        
        blink_box.markdown(f"<p class='big-font'>👁️ Total Blinks: <span style='color:blue;'>{blink_count}</span></p>", unsafe_allow_html=True)
        yawn_box.markdown(f"<p class='big-font'>🥱 Total Yawns: <span style='color:orange;'>{yawn_count}</span></p>", unsafe_allow_html=True)
        
        if alert_msg:
            alert_box.markdown(f"<p class='alert-font'>{alert_msg}</p>", unsafe_allow_html=True)
        else:
            alert_box.empty()

    cap.release()
else:
    st.info("👈 Check the 'START SYSTEM' box in the sidebar to turn on the camera.")