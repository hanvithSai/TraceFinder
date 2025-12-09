import streamlit as st
import requests
from PIL import Image
import json
import os
import socket
from urllib.parse import urlparse

# Use env var as default but allow user override in UI
API_URL = os.getenv("TRACEFINDER_API", "http://localhost:8000/predict")

# Page configuration
st.set_page_config(page_title="TraceFinder — Scanner Detection", layout="wide", page_icon="🔎")

def render_header():
    st.markdown(
        """
        <div style='display:flex;align-items:center;gap:16px'>
            <div style='font-size:42px'>🔎</div>
            <div>
                <h1 style='margin:0'>TraceFinder</h1>
                <p style='margin:0;color:gray'>Scanner identification & trace source detection </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def check_api_available(api_url, timeout=2):
    """
    Quick TCP check to see if host:port is reachable.
    Returns (available: bool, msg: str)
    """
    try:
        p = urlparse(api_url)
        host = p.hostname or "localhost"
        port = p.port or (443 if p.scheme == "https" else 80)
        socket.create_connection((host, port), timeout=timeout).close()
        return True, f"Reachable: {host}:{port}"
    except Exception as e:
        return False, str(e)


def sidebar_info():
    st.sidebar.title("Project")
    st.sidebar.markdown(
        """
        **TraceFinder** is a lightweight tool to detect the source scanner/model from scanned images.

        - Backend models: `xgboost` (150/300 DPI variants) and `cnn`.
        - API endpoint: `/predict` (default `http://localhost:8000/predict`).
        - Upload scanned images (tiff/jpg/png) and pick a model to predict.
        """
    )

    st.sidebar.markdown("---")
    st.sidebar.header("About")
    st.sidebar.info(
        "TraceFinder was developed as part of a research project to showcase scanner/source detection using classical and deep-learning approaches.\n\nBuilt with Python, Streamlit, XGBoost and Keras."
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("Repository: `TraceFinder` — contact: maintainer in README")

    # API config + quick test
    st.sidebar.markdown("---")
    st.sidebar.header("API configuration")
    api_input = st.sidebar.text_input("Prediction API URL", value=API_URL, help="Full URL to /predict endpoint (e.g. http://localhost:8000/predict)")
    if st.sidebar.button("Test API connection"):
        ok, msg = check_api_available(api_input)
        if ok:
            st.sidebar.success(f"OK — {msg}")
        else:
            st.sidebar.error("Cannot reach API: " + msg)
            st.sidebar.markdown(
                """
                Quick troubleshooting:
                - Make sure the backend is running (common command: `uvicorn main:app --reload --host 0.0.0.0 --port 8000`).
                - If backend runs in WSL/docker, use the correct host (try 127.0.0.1 or container host).
                - Check firewall or antivirus blocking port 8000.
                - Test with curl: `curl -v <API_URL>` to see connection details.
                """
            )

    # New: detailed backend run instructions & troubleshooting helper
    with st.sidebar.expander("How to run the backend (commands & tips)", expanded=False):
        st.markdown("Run these from the backend project folder.")
        st.markdown("Python (virtualenv + uvicorn):")
        st.code(
            "python -m venv .venv\n"
            "# Windows\n"
            ".\\.venv\\Scripts\\activate\n"
            "# macOS / Linux\n"
            "source .venv/bin/activate\n"
            "pip install -r requirements.txt  # or pip install fastapi uvicorn\n"
            "uvicorn main:app --reload --host 0.0.0.0 --port 8000\n",
            language="bash",
        )
        st.markdown("Docker (if a Dockerfile or docker-compose.yml exists):")
        st.code(
            "docker-compose up --build\n\n"
            "# or build/run manually\n"
            "docker build -t tracefinder-backend .\n"
            "docker run -p 8000:8000 tracefinder-backend\n",
            language="bash",
        )
        st.markdown("Quick tests and tips:")
        st.markdown(
            "- Test connection: `curl -v http://localhost:8000/predict` (any response shows the port is reachable).\n"
            "- Visit `http://localhost:8000/docs` to see OpenAPI UI if FastAPI is used.\n"
            "- If using WSL, ensure the server binds `0.0.0.0` and use `localhost` from Windows.\n"
            "- If connection is refused, check that the process is running and that firewall/AV is not blocking port 8000."
        )

    return api_input


def pretty_upload_area():
    st.subheader("Upload & Predict")

    # Responsive styling and centered container
    st.markdown(
        """
        <style>
        /* Main container sizing and generous padding */
        .block-container{max-width:1200px;margin:0 auto 48px auto;padding-top:2rem;padding-left:2rem;padding-right:2rem}

        /* Header spacing */
        h1{margin-top:0.2rem;margin-bottom:0.3rem}
        h2, h3{margin-top:0.8rem;margin-bottom:0.6rem}

        /* Tweak buttons and form elements */
        .stButton>button{border-radius:10px;padding:10px 18px}
        [data-testid="stForm"]{padding:14px;border-radius:10px;border:1px solid rgba(255,255,255,0.04);background-color:rgba(255,255,255,0.01)}

        /* File uploader: more breathing room */
        [data-testid="stFileUploader"]{padding:8px;margin-top:8px}

        /* Column gap for large screens */
        .stColumns {gap:3rem}

        /* Make the preview box stand out slightly */
        [data-testid="stInfoBox"]{padding:18px;border-radius:10px;background-color:rgba(15,40,60,0.25)}

        /* Responsive adjustments */
        @media (max-width: 900px) {
            .block-container{padding-left:1rem;padding-right:1rem}
            .stColumns {gap:1rem}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Two-column responsive layout: controls on left, preview on right
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        with st.form(key="predict_form"):
            model_choice = st.selectbox("Choose model", ["xgboost", "cnn"])
            uploaded = st.file_uploader("Upload image (drag & drop works)", type=["tif", "tiff", "png", "jpg", "jpeg"])
            st.write("")
            predict_btn = st.form_submit_button("Predict")

    with col2:
        placeholder = st.empty()
        placeholder.info("Upload an image to preview and run predictions.")

    return model_choice, uploaded, predict_btn, placeholder


def show_prediction_results(out, placeholder):
    # Use a container for nicer layout
    with st.container():
        if out.get("model") == "cnn":
            st.subheader("🧠 CNN Prediction")
            st.success(f"Label: {out.get('label')}")
            st.write(f"Confidence: {out.get('confidence', 0) * 100:.2f}%")
            st.write("Probability Distribution:")
            st.bar_chart(out.get("probs", []))

        elif out.get("model") == "xgboost":
            st.subheader("🌲 XGBoost Predictions")
            cols = st.columns(2)
            with cols[0]:
                st.markdown("**150 DPI**")
                st.success(f"Label: {out['150dpi'].get('label')}")
                st.write(f"Confidence: {out['150dpi'].get('confidence', 0) * 100:.2f}%")
                st.bar_chart(out['150dpi'].get('probs', []))

            with cols[1]:
                st.markdown("**300 DPI**")
                st.success(f"Label: {out['300dpi'].get('label')}")
                st.write(f"Confidence: {out['300dpi'].get('confidence', 0) * 100:.2f}%")
                st.bar_chart(out['300dpi'].get('probs', []))

        else:
            st.error("Unexpected response format from the API.")


def main():
    render_header()
    # call sidebar_info and retrieve API URL typed by user
    api_from_sidebar = sidebar_info()

    model_choice, uploaded, predict_btn, placeholder = pretty_upload_area()

    if uploaded:
        try:
            img = Image.open(uploaded)
            # place preview in the right column if present
            placeholder.image(img, caption=f"Preview: {uploaded.name}", use_column_width=True)
            # show some file meta under preview
            placeholder.caption(f"Filename: {uploaded.name} — Type: {uploaded.type} — Size: {uploaded.size} bytes")
        except Exception as e:
            st.error(f"Unable to open image: {e}")

    if predict_btn:
        if not uploaded:
            st.warning("Please upload an image before predicting.")
            return

        # use API URL from sidebar (fallback to default)
        api_to_use = api_from_sidebar or API_URL

        with st.spinner("Sending image to prediction API..."):
            try:
                files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
                data = {"model_choice": model_choice}
                res = requests.post(api_to_use, files=files, data=data, timeout=30)
            except requests.exceptions.ConnectionError as e:
                st.error(
                    "Failed to reach API (connection error). Possible causes:\n"
                    "- Backend not running on the configured host/port.\n"
                    "- Wrong API URL (use http://127.0.0.1:8000/predict if needed).\n"
                    "- Firewall or network blocking the port.\n\n"
                    "Try:\n"
                    "- Start the backend (e.g. `uvicorn main:app --reload --host 0.0.0.0 --port 8000`).\n"
                    "- Test with curl: `curl -v {}`".format(api_to_use)
                )
                return
            except Exception as e:
                st.error(f"Failed to reach API: {e}")
                return

        if res.status_code == 200:
            try:
                out = res.json()
            except Exception:
                st.error("API returned non-JSON response")
                st.code(res.text)
                return

            show_prediction_results(out, placeholder)

            # debug / raw view
            with st.expander("Raw response (debug)"):
                st.json(out)

        else:
            st.error(f"Prediction failed — status {res.status_code}")
            st.code(res.text)


if __name__ == "__main__":
    main()
