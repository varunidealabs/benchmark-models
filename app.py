import streamlit as st
import requests
import re
import base64
import time
import json
from vertexai.generative_models import GenerativeModel
import vertexai
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request

st.set_page_config(page_title="Image Generation Benchmark", page_icon="🖼️", layout="wide", initial_sidebar_state="expanded")

MODELS = [
    {"id": "gemini-2.5-pro", "name": "Gemini 2.5 Pro", "type": "vertex"},
    {"id": "claude-3-5-sonnet-20241022", "name": "Claude 3.5 Sonnet", "type": "anthropic"},
    {"id": "gpt-4", "name": "Azure GPT-4", "type": "azure"},
    {"id": "deepseek-chat", "name": "DeepSeek R1-0528", "type": "deepseek"}
]

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

def call_gemini(prompt):
    init_vertex_ai()
    model = GenerativeModel("gemini-2.5-pro")
    response = model.generate_content(prompt)
    return {"choices": [{"message": {"content": response.text}}]}

def call_claude(prompt):
    headers = {
        "x-api-key": st.secrets["ANTHROPIC_API_KEY"],
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01"
    }
    data = {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 4000,
        "messages": [{"role": "user", "content": prompt}]
    }
    response = requests.post("https://api.anthropic.com/v1/messages", json=data, headers=headers)
    result = response.json()
    return {"choices": [{"message": {"content": result["content"][0]["text"]}}]}

def call_azure_gpt4(prompt):
    headers = {
        "api-key": st.secrets["AZURE_OPENAI_KEY"],
        "Content-Type": "application/json"
    }
    data = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 4000,
        "temperature": 0.7
    }
    response = requests.post(
        "https://access-01.openai.azure.com/openai/deployments/gpt-4/chat/completions?api-version=2025-01-01-preview",
        json=data, headers=headers
    )
    response.raise_for_status()
    return response.json()

def call_deepseek(prompt):
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
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        }
        
        # Try the MaaS endpoint
        url = f"https://us-central1-aiplatform.googleapis.com/v1/projects/{project_id}/locations/us-central1/endpoints/openapi/chat/completions"
        
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 404:
            st.error("DeepSeek R1-0528 model not available in your project. Please check if the model is enabled in your Google Cloud project.")
            return {"choices": [{"message": {"content": ""}}]}
            
        response.raise_for_status()
        return response.json()
        
    except Exception as e:
        st.error(f"DeepSeek API Error: {str(e)}")
        return {"choices": [{"message": {"content": ""}}]}

def extract_svg_content(text):
    if not text:
        return None
    svg_pattern = r'<svg[^>]*>.*?</svg>'
    match = re.search(svg_pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        svg_content = match.group(0)
        return svg_content
    return None

def test_model(model_info):
    try:
        if model_info["type"] == "vertex":
            response = call_gemini(PROMPT)
        elif model_info["type"] == "anthropic":
            response = call_claude(PROMPT)
        elif model_info["type"] == "azure":
            response = call_azure_gpt4(PROMPT)
        elif model_info["type"] == "deepseek":
            response = call_deepseek(PROMPT)
        
        
        content = response["choices"][0]["message"]["content"]
        svg_content = extract_svg_content(content)
        
        
        return {
            "model_name": model_info["name"],
            "success": bool(svg_content),
            "svg_content": svg_content,
            "error": None if svg_content else "No valid SVG generated"
        }
    except Exception as e:
        return {
            "model_name": model_info["name"],
            "success": False,
            "error": str(e),
            "svg_content": None
        }

def display_svg(svg_content, width=350):
    if svg_content:
        # Don't modify viewBox if it already exists
        if 'viewBox' not in svg_content and 'width' not in svg_content:
            svg_content = svg_content.replace('<svg', '<svg viewBox="0 0 400 300" width="400" height="300"')
        svg_b64 = base64.b64encode(svg_content.encode('utf-8')).decode('utf-8')
        st.markdown(f'<div style="text-align: center; margin: 20px 0;"><img src="data:image/svg+xml;base64,{svg_b64}" width="{width}"></div>', unsafe_allow_html=True)
    else:
        st.error("❌ No valid SVG content to display")

def main():
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="color: #2563eb; margin-bottom: 0.5rem;">🖼️ Image Generation Benchmark</h1>
        <p style="font-size: 1.2rem; color: #64748b; margin-bottom: 2rem;">Which AI model generates the best SVG for the current prompt?</p>
    </div>
    """, unsafe_allow_html=True)
    
    
    st.sidebar.markdown("### 🤖 Select Models to Test")
    selected_models = []
    for model in MODELS:
        if st.sidebar.checkbox(f"**{model['name']}**", value=True, key=model['id']):
            selected_models.append(model)
    
    if not selected_models:
        st.warning("⚠️ Please select at least one model to test.")
        return
    
    st.sidebar.markdown("### 📖 About the Benchmark")
    st.sidebar.markdown("""
    This benchmark tests AI models with a creative challenge: generating SVG code for **"a simple house with the sun in the sky."**
    """)
    
    if st.button("🚀 Run Benchmark", type="primary", use_container_width=True):
        run_benchmark(selected_models)
    
    if 'benchmark_results' in st.session_state:
        display_results()

def run_benchmark(selected_models):
    st.markdown("### Running Benchmark...")
    progress_bar = st.progress(0)
    status_text = st.empty()
    results = []
    
    for i, model in enumerate(selected_models):
        status_text.markdown(f"**Testing {model['name']}...** ⏳")
        result = test_model(model)
        results.append(result)
        progress_bar.progress((i + 1) / len(selected_models))
    
    st.session_state.benchmark_results = results
    status_text.markdown("")
    progress_bar.progress(1.0)
    
    successful_models = [r for r in results if r["success"]]
    if len(successful_models) > 0:
        st.balloons()

def display_results():
    results = st.session_state.benchmark_results
    st.markdown("---")
    
    successful_results = [r for r in results if r["success"]]
    failed_results = [r for r in results if not r["success"]]
    
    if successful_results:
        st.markdown("### ✅ Generated Images")
        for i in range(0, len(successful_results), 2):
            cols = st.columns(2)
            for j, col in enumerate(cols):
                if i + j < len(successful_results):
                    result = successful_results[i + j]
                    with col:
                        st.markdown(f"<div style=\"text-align: center; margin-bottom: 1rem;\"><h4 style=\"margin-bottom: 0.5rem;\">{result['model_name']}</h4></div>", unsafe_allow_html=True)
                        display_svg(result['svg_content'])
    
    if failed_results:
        for result in failed_results:
            st.error(f"❌ **{result['model_name']}** - {result['error']}")
    
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🗑️ Clear Results & Run New Test", use_container_width=True):
            if 'benchmark_results' in st.session_state:
                del st.session_state['benchmark_results']
            st.rerun()

if __name__ == "__main__":
    main()
