"""
ASL (American Sign Language) Finger-Spelling Recognition App
Upload an image of an ASL hand sign and get the predicted letter.
Uses MediaPipe hand landmarks (Tasks API) + a geometric classifier.
"""

import os
import math
import streamlit as st
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw

# ──────────────────────────────────────────────
# MediaPipe Tasks API setup
# ──────────────────────────────────────────────
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

# ──────────────────────────────────────────────
# Landmark indices
# ──────────────────────────────────────────────
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

# Hand connections for drawing
HAND_CONNECTIONS = [
    (WRIST, THUMB_CMC), (THUMB_CMC, THUMB_MCP), (THUMB_MCP, THUMB_IP), (THUMB_IP, THUMB_TIP),
    (WRIST, INDEX_MCP), (INDEX_MCP, INDEX_PIP), (INDEX_PIP, INDEX_DIP), (INDEX_DIP, INDEX_TIP),
    (WRIST, MIDDLE_MCP), (MIDDLE_MCP, MIDDLE_PIP), (MIDDLE_PIP, MIDDLE_DIP), (MIDDLE_DIP, MIDDLE_TIP),
    (WRIST, RING_MCP), (RING_MCP, RING_PIP), (RING_PIP, RING_DIP), (RING_DIP, RING_TIP),
    (WRIST, PINKY_MCP), (PINKY_MCP, PINKY_PIP), (PINKY_PIP, PINKY_DIP), (PINKY_DIP, PINKY_TIP),
    (INDEX_MCP, MIDDLE_MCP), (MIDDLE_MCP, RING_MCP), (RING_MCP, PINKY_MCP),
]


# ──────────────────────────────────────────────
# Helper geometry functions
# ──────────────────────────────────────────────
def _dist(a, b):
    """Euclidean distance between two landmarks (NormalizedLandmark)."""
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def _is_finger_extended(landmarks, tip, dip, pip_idx, mcp):
    """Check if a finger (non-thumb) is extended."""
    return _dist(landmarks[tip], landmarks[WRIST]) > _dist(landmarks[pip_idx], landmarks[WRIST])


def _is_thumb_extended(landmarks):
    """Check if the thumb is extended (away from the palm)."""
    return _dist(landmarks[THUMB_TIP], landmarks[PINKY_MCP]) > _dist(landmarks[THUMB_IP], landmarks[PINKY_MCP])


def _fingers_extended(landmarks):
    """Return list of booleans: [thumb, index, middle, ring, pinky] extended."""
    return [
        _is_thumb_extended(landmarks),
        _is_finger_extended(landmarks, INDEX_TIP, INDEX_DIP, INDEX_PIP, INDEX_MCP),
        _is_finger_extended(landmarks, MIDDLE_TIP, MIDDLE_DIP, MIDDLE_PIP, MIDDLE_MCP),
        _is_finger_extended(landmarks, RING_TIP, RING_DIP, RING_PIP, RING_MCP),
        _is_finger_extended(landmarks, PINKY_TIP, PINKY_DIP, PINKY_PIP, PINKY_MCP),
    ]


def _fingers_touching(landmarks, tip_a, tip_b, threshold=0.05):
    """Check if two finger tips are close together."""
    return _dist(landmarks[tip_a], landmarks[tip_b]) < threshold


def _angle_between(a, b, c):
    """Angle at point b formed by a-b-c (in degrees)."""
    ba = np.array([a.x - b.x, a.y - b.y, a.z - b.z])
    bc = np.array([c.x - b.x, c.y - b.y, c.z - b.z])
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return math.degrees(math.acos(np.clip(cos_angle, -1.0, 1.0)))


# ──────────────────────────────────────────────
# ASL Letter classifier (rule-based heuristics)
# ──────────────────────────────────────────────
def classify_asl_letter(landmarks):
    """
    Classify an ASL static hand sign into a letter (A-Z)
    using geometric rules on hand landmarks.

    NOTE: J and Z involve motion and cannot be classified from
    a single static image.
    """
    lm = landmarks
    ext = _fingers_extended(lm)
    thumb, index, middle, ring, pinky = ext
    num_extended = sum(ext)

    # ── 0 fingers extended ──────────────────────
    if num_extended == 0:
        thumb_above = lm[THUMB_TIP].y < lm[INDEX_MCP].y
        thumb_between_idx_mid = (
            min(lm[INDEX_PIP].x, lm[MIDDLE_PIP].x)
            < lm[THUMB_TIP].x
            < max(lm[INDEX_PIP].x, lm[MIDDLE_PIP].x)
        )
        if thumb_between_idx_mid:
            return "T"
        if thumb_above:
            return "A"
        tips_near_palm = (
            _dist(lm[INDEX_TIP], lm[INDEX_MCP]) < 0.06
            and _dist(lm[MIDDLE_TIP], lm[MIDDLE_MCP]) < 0.06
        )
        if tips_near_palm:
            return "E"
        thumb_under_three = (
            lm[THUMB_TIP].y > lm[INDEX_PIP].y
            and lm[THUMB_TIP].y > lm[MIDDLE_PIP].y
            and lm[THUMB_TIP].y > lm[RING_PIP].y
        )
        if thumb_under_three:
            return "M"
        thumb_under_two = (
            lm[THUMB_TIP].y > lm[INDEX_PIP].y
            and lm[THUMB_TIP].y > lm[MIDDLE_PIP].y
        )
        if thumb_under_two:
            return "N"
        return "S"

    # ── 1 finger extended ───────────────────────
    if num_extended == 1:
        if index:
            index_horizontal = abs(lm[INDEX_TIP].y - lm[INDEX_MCP].y) < abs(
                lm[INDEX_TIP].x - lm[INDEX_MCP].x
            )
            if index_horizontal:
                return "G"
            if _fingers_touching(lm, THUMB_TIP, MIDDLE_TIP, 0.06):
                return "D"
            index_bent = _angle_between(lm[INDEX_MCP], lm[INDEX_PIP], lm[INDEX_TIP]) < 140
            if index_bent:
                return "X"
            return "D"
        if pinky:
            return "I"
        if thumb:
            return "A"

    # ── 2 fingers extended ──────────────────────
    if num_extended == 2:
        if thumb and index:
            angle = _angle_between(lm[THUMB_TIP], lm[WRIST], lm[INDEX_TIP])
            if angle > 40:
                return "L"
            return "G"
        if index and middle:
            index_x = lm[INDEX_TIP].x
            middle_x = lm[MIDDLE_TIP].x
            index_base_x = lm[INDEX_MCP].x
            middle_base_x = lm[MIDDLE_MCP].x
            crossed = (index_x - middle_x) * (index_base_x - middle_base_x) < 0
            if crossed:
                return "R"
            pointing_sideways = abs(lm[INDEX_TIP].y - lm[INDEX_MCP].y) < abs(
                lm[INDEX_TIP].x - lm[INDEX_MCP].x
            )
            if pointing_sideways:
                return "H"
            fingers_spread = _dist(lm[INDEX_TIP], lm[MIDDLE_TIP]) > 0.06
            if fingers_spread:
                return "V"
            return "U"

    # ── 3 fingers extended ──────────────────────
    if num_extended == 3:
        if index and middle and ring:
            return "W"
        if thumb and index and middle:
            return "K"

    # ── 4 fingers extended ──────────────────────
    if num_extended == 4:
        if not thumb:
            return "B"
        if not pinky:
            return "F"

    # ── 5 fingers extended ──────────────────────
    if num_extended == 5:
        return "B"

    # ── Special shapes ──────────────────────────
    if _fingers_touching(lm, THUMB_TIP, INDEX_TIP, 0.04) and not middle and not ring and not pinky:
        return "O"
    if (
        not _fingers_touching(lm, THUMB_TIP, INDEX_TIP, 0.04)
        and _dist(lm[THUMB_TIP], lm[INDEX_TIP]) < 0.12
        and num_extended >= 3
    ):
        return "C"
    if _fingers_touching(lm, THUMB_TIP, INDEX_TIP, 0.04) and middle and ring and pinky:
        return "F"

    return "?"


# ──────────────────────────────────────────────
# Draw hand landmarks on image
# ──────────────────────────────────────────────
def draw_landmarks_on_image(image, hand_landmarks, img_width, img_height):
    """Draw hand landmarks and connections on a PIL Image copy."""
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    # Draw connections
    for start_idx, end_idx in HAND_CONNECTIONS:
        start = hand_landmarks[start_idx]
        end = hand_landmarks[end_idx]
        x1, y1 = int(start.x * img_width), int(start.y * img_height)
        x2, y2 = int(end.x * img_width), int(end.y * img_height)
        draw.line([(x1, y1), (x2, y2)], fill=(0, 255, 0), width=2)

    # Draw landmarks
    for lm in hand_landmarks:
        cx, cy = int(lm.x * img_width), int(lm.y * img_height)
        r = 4
        draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)], fill=(255, 0, 0), outline=(255, 255, 255))

    return annotated


# ──────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="ASL Sign Language Detector",
    page_icon="🤟",
    layout="centered",
)

st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color: #888;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .result-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin: 1.5rem 0;
    }
    .result-letter {
        font-size: 6rem;
        font-weight: 800;
        color: #fff;
        line-height: 1;
    }
    .result-label {
        font-size: 1.1rem;
        color: rgba(255,255,255,0.85);
        margin-top: 0.5rem;
    }
    .info-card {
        background: #f0f2f6;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 1rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">🤟 ASL Sign Language Detector</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Upload an image of an ASL hand sign to detect the letter</div>',
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.write(
        "This app uses **MediaPipe** hand landmark detection and geometric "
        "heuristics to recognise ASL finger-spelled letters."
    )
    st.markdown("---")
    st.subheader("Supported Letters")
    st.write("Static signs: **A–Z** (J and Z are motion-based and may be inaccurate)")
    st.markdown("---")
    st.subheader("Tips for Best Results")
    st.markdown(
        """
        - Use a **clear, well-lit** photo  
        - Show **one hand** only  
        - Keep the hand **facing the camera**  
        - Avoid **cluttered backgrounds**
        """
    )

# ── File uploader ───────────────────────────────
uploaded_file = st.file_uploader(
    "Choose an image…",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    help="Upload a photo of an ASL hand sign",
)

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    img_height, img_width = img_array.shape[:2]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Uploaded Image")
        try:
            st.image(image, caption="Uploaded image", use_container_width=True)
        except TypeError:
            st.image(image, caption="Uploaded image", use_column_width=True)

    # ── Run MediaPipe HandLandmarker (Tasks API) ──
    if not os.path.exists(MODEL_PATH):
        st.error(
            f"Hand landmarker model not found at `{MODEL_PATH}`. "
            "Please download it from: "
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/latest/hand_landmarker.task"
        )
        st.stop()

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    with HandLandmarker.create_from_options(options) as landmarker:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_array)
        result = landmarker.detect(mp_image)

    if result.hand_landmarks:
        hand_landmarks = result.hand_landmarks[0]  # list of NormalizedLandmark

        # Draw landmarks
        annotated_image = draw_landmarks_on_image(image, hand_landmarks, img_width, img_height)

        with col2:
            st.subheader("🖐️ Detected Hand")
            try:
                st.image(annotated_image, caption="Hand landmarks", use_container_width=True)
            except TypeError:
                st.image(annotated_image, caption="Hand landmarks", use_column_width=True)

        # Classify letter
        predicted_letter = classify_asl_letter(hand_landmarks)

        # Show result
        st.markdown(
            f"""
            <div class="result-box">
                <div class="result-letter">{predicted_letter}</div>
                <div class="result-label">Predicted ASL Letter</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Landmark details
        with st.expander("🔍 Landmark Details"):
            ext = _fingers_extended(hand_landmarks)
            finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
            for name, is_ext in zip(finger_names, ext):
                status = "✅ Extended" if is_ext else "✊ Curled"
                st.write(f"**{name}**: {status}")
    else:
        with col2:
            st.subheader("⚠️ No Hand Detected")
        st.warning(
            "No hand was detected in the image. Please try another image with a clearly visible hand sign."
        )
else:
    st.markdown(
        """
        <div class="info-card">
            <h4>👆 Upload an image to get started</h4>
            <p>Take a photo of an ASL hand sign or upload an existing one.
            The app will detect the hand and predict the corresponding letter.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### ASL Alphabet Reference")
    st.markdown(
        "Learn more about the ASL alphabet at "
        "[handspeak.com](https://www.handspeak.com/spell/)"
    )
