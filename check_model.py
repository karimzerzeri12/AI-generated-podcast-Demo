import os
from dotenv import load_dotenv
# Load API key from .env
load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY')

if not api_key:
    print("❌ GOOGLE_API_KEY not found in .env file!")
    print("Make sure your .env file has: GOOGLE_API_KEY=your_key_here")
else:
    print(f"✓ API Key loaded: {api_key[:10]}...")
    
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    
    print("\nAvailable embedding models:")
    for model in genai.list_models():
        if 'embed' in model.name.lower():
            print(f"  ✓ {model.name}")