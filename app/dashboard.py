import streamlit as st
import requests

# --- Configuration ---
st.set_page_config(page_title="DeepSight Dashboard", layout="centered")

# --- Constants ---
MAX_FILE_SIZE_MB = 100
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


# --- Display Results ---
def display_results(result: dict):
    verdict = result.get("verdict", "N/A")
    avg_prob = result.get("average_fake_probability", 0.0)
    faces_detected = result.get("faces_detected", 0)

    st.subheader("Analysis Results")
    if verdict == "FAKE":
        st.error(f"**Verdict: {verdict}** 🔴")
    elif verdict == "REAL":
        st.success(f"**Verdict: {verdict}** 🟢")
    else:
        st.warning(f"**Verdict: {verdict}** 🟡")

    st.markdown("---")
    st.subheader("Details")

    st.markdown("**Average Fake Probability**")
    # avg_prob should be between 0 and 1
    try:
        st.progress(float(avg_prob), text=f"{float(avg_prob):.2%}")
    except Exception:
        st.write(f"{avg_prob}")

    st.metric("Faces Detected", f"{faces_detected}")

    if verdict != "UNCERTAIN":
        st.caption(
            f"This score represents the model's average confidence that the detected "
            f"face(s) are synthetically generated. Scores above "
            f"{result.get('prediction_threshold', 0.5):.0%} are classified as FAKE."
        )

    if "reason" in result:
        st.info(f"Reason: {result['reason']}")


# --- Analyze Video (send original upload) ---
def analyze_video(video_bytes: bytes, filename: str, mime_type: str, api_url: str):
    st.info("Analyzing video... This may take a moment.")

    files = {"video": (filename, video_bytes, mime_type or "application/octet-stream")}
    try:
        response = requests.post(api_url, files=files, timeout=300)
        if response.status_code == 200:
            st.success("Analysis complete!")
            result = response.json()
            display_results(result)
        else:
            try:
                detail = response.json().get("detail", response.text)
            except Exception:
                detail = response.text
            st.error(f"API Error {response.status_code}: {detail}")
    except requests.exceptions.RequestException as e:
        st.error(f"Could not reach the API: {e}")


# --- Streamlit UI ---

st.title("DeepSight: Deepfake Detection")
st.markdown("""
Upload a video file to analyze it for deepfake content. The system will sample frames,
detect faces, and predict whether the video is real or fake.
""")

# --- Sidebar ---
st.sidebar.title("Configuration")
api_url = st.sidebar.text_input(
    "Backend API URL",
    value="http://127.0.0.1:8000/predict"
)

st.sidebar.title("How to Use")
st.sidebar.info(
    """
    1. **Start the Backend API:**
       ```bash
       uvicorn app.main:app --reload
       ```
    2. **Run this Dashboard:**
       ```bash
       streamlit run dashboard.py
       ```
    3. **Upload a Video:** Use the file uploader.
    4. **Analyze:** Click the 'Analyze Video' button.
    """
)

# --- Session State for video ---
if "video_bytes" not in st.session_state:
    st.session_state["video_bytes"] = None
if "video_mime" not in st.session_state:
    st.session_state["video_mime"] = None
if "video_name" not in st.session_state:
    st.session_state["video_name"] = None

# --- Main Page ---
uploaded_file = st.file_uploader(
    "Choose a video file",
    type=["mp4", "mov", "avi"],
    help=f"Maximum file size: {MAX_FILE_SIZE_MB}MB"
)

if uploaded_file is not None:
    # Size check
    if uploaded_file.size > MAX_FILE_SIZE_BYTES:
        st.error(
            f"File is too large ({uploaded_file.size / (1024*1024):.2f}MB). "
            f"Maximum allowed size is {MAX_FILE_SIZE_MB}MB."
        )
    else:
        # Read bytes once per upload
        video_bytes = uploaded_file.read()
        st.session_state["video_bytes"] = video_bytes
        st.session_state["video_mime"] = uploaded_file.type or "video/mp4"
        st.session_state["video_name"] = uploaded_file.name

# Display video if stored
if st.session_state["video_bytes"]:
    # Use bytes + correct MIME so Streamlit/browser can handle different formats
    st.video(
        st.session_state["video_bytes"],
        format=st.session_state["video_mime"]
    )

    # Analyze button
    if st.button("Analyze Video", type="primary"):
        with st.spinner("Sending video to backend..."):
            analyze_video(
                st.session_state["video_bytes"],
                st.session_state["video_name"],
                st.session_state["video_mime"],
                api_url
            )
