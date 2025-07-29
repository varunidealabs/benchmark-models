import streamlit as st
import re
import base64
import json
import requests
from vertexai.generative_models import GenerativeModel
import vertexai
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request

st.set_page_config(page_title="SVG Generator", layout="centered")

PROMPT = """Create a simple SVG illustration of a house with the sun in the sky. Requirements:
- House: rectangular base, triangular roof, square door, rectangular window
- Sun: yellow circle with simple rays (lines or triangles)
- Use bright, cheerful colors
- Include a ground line (green rectangle or line)
- Keep shapes simple and geometric

Return ONLY the complete SVG code, starting with <svg> and ending with </svg>. No other text."""

def init_vertex_ai():
    creds_info = dict(st.secrets["google_creds"])
    credentials = Credentials.from_authorized_user_info(creds_info)
    project_id = creds_info["quota_project_id"]
    vertexai.init(project=project_id, location="us-central1", credentials=credentials)

def generate_gemini_svg():
    init_vertex_ai()
    model = GenerativeModel("gemini-2.5-pro")
    response = model.generate_content(PROMPT)
    
    svg_pattern = r'<svg[^>]*>.*?</svg>'
    match = re.search(svg_pattern, response.text, re.DOTALL | re.IGNORECASE)
    return match.group(0) if match else None

def generate_deepseek_svg():
    try:
        creds_info = dict(st.secrets["google_creds"])
        credentials = Credentials.from_authorized_user_info(creds_info)
        project_id = creds_info["quota_project_id"]
        
        # Get access token
        credentials.refresh(Request())
        access_token = credentials.token
        
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json; charset=utf-8"
        }
        
        data = {
            "model": "deepseek-ai/deepseek-r1-0528-maas",
            "messages": [{"role": "user", "content": PROMPT}],
            "stream": False
        }
        
        # Try the MaaS endpoint
        url = f"https://us-central1-aiplatform.googleapis.com/v1/projects/{project_id}/locations/us-central1/endpoints/openapi/chat/completions"
        
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 404:
            st.error("DeepSeek R1-0528 model not available in your project. Please check if the model is enabled in your Google Cloud project.")
            return None
            
        response.raise_for_status()
        result = response.json()
        
        content = result["choices"][0]["message"]["content"]
        svg_pattern = r'<svg[^>]*>.*?</svg>'
        match = re.search(svg_pattern, content, re.DOTALL | re.IGNORECASE)
        return match.group(0) if match else None
        
    except Exception as e:
        st.error(f"DeepSeek API Error: {str(e)}")
        return None

def display_svg(svg_content):
    if svg_content:
        if 'viewBox' not in svg_content:
            svg_content = svg_content.replace('<svg', '<svg viewBox="0 0 400 300" width="400" height="300"')
        svg_b64 = base64.b64encode(svg_content.encode('utf-8')).decode('utf-8')
        st.markdown(f'<img src="data:image/svg+xml;base64,{svg_b64}" width="400">', unsafe_allow_html=True)

st.title("SVG Generator")

# Sidebar model selection
st.sidebar.markdown("### 🤖 Select Models")
gemini_selected = st.sidebar.checkbox("**Gemini 2.5 Pro**", value=True)
deepseek_selected = st.sidebar.checkbox("**DeepSeek R1-0528**", value=True)

if not gemini_selected and not deepseek_selected:
    st.warning("⚠️ Please select at least one model.")
else:
    if st.button("Generate Images"):
        if gemini_selected and deepseek_selected:
            # Both models - side by side
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Gemini 2.5 Pro")
                gemini_svg = generate_gemini_svg()
                if gemini_svg:
                    display_svg(gemini_svg)
                    with st.expander("Gemini SVG Code"):
                        st.code(gemini_svg, language='xml')
                else:
                    st.error("Gemini failed to generate SVG")
            
            with col2:
                st.subheader("DeepSeek R1-0528")
                deepseek_svg = generate_deepseek_svg()
                if deepseek_svg:
                    display_svg(deepseek_svg)
                    with st.expander("DeepSeek SVG Code"):
                        st.code(deepseek_svg, language='xml')
                else:
                    st.error("DeepSeek failed to generate SVG")
        
        elif gemini_selected:
            # Only Gemini
            st.subheader("Gemini 2.5 Pro")
            gemini_svg = generate_gemini_svg()
            if gemini_svg:
                display_svg(gemini_svg)
                with st.expander("SVG Code"):
                    st.code(gemini_svg, language='xml')
            else:
                st.error("Failed to generate SVG")
        
        elif deepseek_selected:
            # Only DeepSeek
            st.subheader("DeepSeek R1-0528")
            deepseek_svg = generate_deepseek_svg()
            if deepseek_svg:
                display_svg(deepseek_svg)
                with st.expander("SVG Code"):
                    st.code(deepseek_svg, language='xml')
            else:
                st.error("Failed to generate SVG")
