import streamlit as st
import re
import base64
import json
from vertexai.generative_models import GenerativeModel
import vertexai
from google.oauth2 import service_account

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
    credentials = service_account.Credentials.from_service_account_info(creds_info)
    project_id = creds_info["quota_project_id"]
    vertexai.init(project=project_id, location="us-central1", credentials=credentials)

def generate_svg():
    init_vertex_ai()
    model = GenerativeModel("gemini-2.5-pro")
    response = model.generate_content(PROMPT)
    
    svg_pattern = r'<svg[^>]*>.*?</svg>'
    match = re.search(svg_pattern, response.text, re.DOTALL | re.IGNORECASE)
    return match.group(0) if match else None

def display_svg(svg_content):
    if svg_content:
        if 'viewBox' not in svg_content:
            svg_content = svg_content.replace('<svg', '<svg viewBox="0 0 400 300" width="400" height="300"')
        svg_b64 = base64.b64encode(svg_content.encode('utf-8')).decode('utf-8')
        st.markdown(f'<img src="data:image/svg+xml;base64,{svg_b64}" width="400">', unsafe_allow_html=True)

st.title("SVG Generator")

if st.button("Generate Image"):
    svg = generate_svg()
    if svg:
        display_svg(svg)
        with st.expander("SVG Code"):
            st.code(svg, language='xml')
    else:
        st.error("Failed to generate SVG")