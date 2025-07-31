from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import requests
import google.generativeai as genai
import time
import threading
from functools import lru_cache
import concurrent.futures

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend-backend communication

# Configure API keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or ""
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or ""
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama3-70b-8192"

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# Supported languages mapping
LANGUAGE_MAP = {
    'hindi': 'Hindi',
    'tamil': 'Tamil',
    'telugu': 'Telugu'
}

# Cache translations to avoid redundant API calls
@lru_cache(maxsize=1000)
def cached_groq_translation(user_input, target_language):
    """Cached version of Groq translation function"""
    return get_groq_translation(user_input, target_language)

@lru_cache(maxsize=1000)
def cached_gemini_translation(user_input, target_language):
    """Cached version of Gemini translation function"""
    return get_gemini_translation(user_input, target_language)

def get_groq_translation(user_input, target_language):
    """Get translation from Groq (Llama) model"""
    prompt = f"""
You are an expert translator specializing in {target_language}-English code-switched text. Follow these rules:

1. PRESERVE:
   - English words exactly as-is
   - Names/places (e.g., "Hyderabad" stays "Hyderabad")
   - Original tone (casual/formal)

2. TRANSLATE:
   - Only the {target_language} portions to English
   - Maintain natural flow (e.g., "చాలా బాగుంది" → "Very nice")
   - Handle mixed grammar (e.g., "రాత్రి dinner" → "Night dinner")

3. OUTPUT FORMAT:
Language Detected: {target_language}
English Translation: <full translation>
{target_language} Translation: <back-translation if needed>

Example {target_language} Output:
Input: "ఈ movie చాలా interesting ఉంది"
Output:
Language Detected: Telugu
English Translation: This movie is very interesting
Telugu Translation: ఈ సినిమా చాలా ఆసక్తికరంగా ఉంది

Now translate:
Input: "{user_input}"
Output:
"""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": f"You are a professional {target_language} translator"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 500
    }

    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        reply = response.json()['choices'][0]['message']['content']
        
        # Parse the response
        result = {
            'detected_language': target_language,
            'english_translation': '',
            'native_translation': ''
        }
        
        # Extract translations from response
        lines = [line.strip() for line in reply.split('\n') if line.strip()]
        for line in lines:
            if line.startswith('English Translation:'):
                result['english_translation'] = line.replace('English Translation:', '').strip()
            elif line.startswith(f'{target_language} Translation:'):
                result['native_translation'] = line.replace(f'{target_language} Translation:', '').strip()
        
        return result
    except requests.exceptions.Timeout:
        return {'error': 'Groq API timeout'}
    except Exception as e:
        return {'error': f'Groq API error: {str(e)}'}

def get_gemini_translation(user_input, target_language):
    """Get translation from Gemini model"""
    prompt = f"""
You are an expert translator specializing in {target_language}-English code-switched text. Follow these rules:

1. PRESERVE:
   - English words exactly as-is when appropriate
   - Names/places (e.g., "Hyderabad" stays "Hyderabad")
   - Original tone (casual/formal)

2. TRANSLATE:
   - Only the {target_language} portions to English
   - Maintain natural flow (e.g., "చాలా బాగుంది" → "Very nice")
   - Handle mixed grammar (e.g., "రాత్రి dinner" → "Night dinner")

3. OUTPUT FORMAT:
Language Detected: {target_language}
English Translation: <full translation>
{target_language} Translation: <back-translation if needed>

Example {target_language} Output:
Input: "ఈ movie చాలా interesting ఉంది"
Output:
Language Detected: Telugu
English Translation: This movie is very interesting
Telugu Translation: ఈ సినిమా చాలా ఆసక్తికరంగా ఉంది

Now translate:
Input: "{user_input}"
Output:
"""

    try:
        response = gemini_model.generate_content(prompt)
        
        # Parse the response
        result = {
            'detected_language': target_language,
            'english_translation': '',
            'native_translation': ''
        }
        
        # Extract translations from response
        lines = [line.strip() for line in response.text.split('\n') if line.strip()]
        for line in lines:
            if line.startswith('English Translation:'):
                result['english_translation'] = line.replace('English Translation:', '').strip()
            elif line.startswith(f'{target_language} Translation:'):
                result['native_translation'] = line.replace(f'{target_language} Translation:', '').strip()
        
        return result
    except Exception as e:
        return {'error': f'Gemini API error: {str(e)}'}

@app.route('/translate', methods=['POST'])
def translate():
    data = request.json
    user_input = data.get('text', '')
    selected_lang = data.get('lang', 'hindi').lower()

    if selected_lang not in LANGUAGE_MAP:
        return jsonify({
            'error': f"Unsupported language. Choose from: {', '.join(LANGUAGE_MAP.keys())}"
        }), 400
    
    if not user_input:
        return jsonify({'error': 'No text provided'}), 400

    target_language = LANGUAGE_MAP[selected_lang]
    
    try:
        start_time = time.time()
        
        # Use ThreadPoolExecutor to run both translations in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            groq_future = executor.submit(cached_groq_translation, user_input, target_language)
            gemini_future = executor.submit(cached_gemini_translation, user_input, target_language)
            
            groq_result = groq_future.result()
            gemini_result = gemini_future.result()
        
        execution_time = time.time() - start_time
        
        return jsonify({
            'groq': groq_result,
            'gemini': gemini_result,
            'execution_time': round(execution_time, 2),
            'cache_hit': {
                'groq': groq_result is not None and 'error' not in groq_result,
                'gemini': gemini_result is not None and 'error' not in gemini_result
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)