import streamlit as st
import os
from dotenv import load_dotenv
import subprocess
import wave
import io
import re
import html
import time

import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
import pymupdf4llm
from langsmith import traceable

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title='Podcast Generator - Food Flavour Design',
    page_icon='🎙️',
    layout='wide',
    initial_sidebar_state='collapsed'
)

# Custom CSS
st.markdown("""
    <style>
    .step-container {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        border-left: 5px solid #1f77b4;
    }
    .step-header {
        font-size: 20px;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== SESSION STATE ====================
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'form_data' not in st.session_state:
    st.session_state.form_data = {
        'duration': 10,
        'language': 'English',
        'chapter': None,
        'style': None,
        'personalization': '',
        'speaker_voices': {},
        'generate_audio': True
    }
if 'generated_script' not in st.session_state:
    st.session_state.generated_script = None
if 'generated_audio' not in st.session_state:
    st.session_state.generated_audio = None
if 'audio_file_name' not in st.session_state:
    st.session_state.audio_file_name = None
if 'detected_speakers' not in st.session_state:
    st.session_state.detected_speakers = []

# ==================== CONFIGURATION DATA ====================
LANGUAGES = [
    'English', 'Spanish', 'French', 'German', 'Italian', 'Portuguese',
    'Japanese', 'Chinese (Mandarin)', 'Chinese (Cantonese)', 'Korean',
    'Arabic', 'Hindi', 'Russian', 'Turkish', 'Dutch', 'Swedish'
]

CHAPTERS = {
    'Module 1 - Flavour Generation': [
        '1.0 Historical Background – Summarising Flavour Generation',
        '1.1 Lipid Oxidation in a multiphase food system and strategies for prevention',
        '1.2 Maillard Reaction - flavour generation during thermal processing',
        '1.2a Maillard Chemistry',
        '1.3 Fermentation a natural route for flavour generation'
    ],
    'Module 2 - Flavour and food matrix interactions': [
        '2.1 The physics behind flavours and the importance of product formulation on flavour release',
        '2.2 Impact of food product architecture and cross-modal interactions on flavour release & perception',
        '2.3 Flavour Application: composition & format of flavourings, selection and regulatory aspects to consider'
    ],
    'Module 3 - Drivers of Flavour Perception': [
        '3.1 Taste and Odour Receptors',
        '3.2 Impact of oral processing behaviour and physiology'
    ],
    'Module 4 - Measuring Flavours in foods': [
        '4.1 Static and dynamic sensory methods for flavour assessment',
        '4.2 Static and dynamic flavour analytics – methodologies for key aroma compounds identification and quantification'
    ]
}

PODCAST_STYLES = {
    'Conversational & Educational': 'A casual, friendly dialogue between a knowledgeable host and an engaging guest. Use analogies and real-world examples.',
    'Academic & In-Depth': 'A formal, research-focused discussion with detailed explanations. Include technical terminology and comprehensive coverage.',
    'Storytelling & Narrative': 'A narrative-driven format with engaging stories and anecdotes. Make concepts memorable through narrative structure.',
    'Interview Format': 'A Q&A style interview where the host asks probing questions and the guest provides expert responses.',
    'Debate & Multiple Perspectives': 'Present multiple viewpoints on the topic with balanced arguments from different speakers.'
}

PIPER_VOICES = {
    'Lessac (Warm Male)': {
        'model': 'piper_models/en_US-lessac-medium.onnx',
        'config': 'piper_models/en_US-lessac-medium.onnx.json',
        'gender': 'Male',
        'description': 'Warm, natural, professional - Perfect for Host'
    },
    'LibriTTS (Clear Female)': {
        'model': 'piper_models/en_US-libritts-high.onnx',
        'config': 'piper_models/en_US-libritts-high.onnx.json',
        'gender': 'Female',
        'description': 'Clear, articulate, educational - Great for Guest'
    },
    'Ryan (Deep Male)': {
        'model': 'piper_models/en_US-ryan-medium.onnx',
        'config': 'piper_models/en_US-ryan-medium.onnx.json',
        'gender': 'Male',
        'description': 'Deep, authoritative, confident - Perfect for expert content'
    },
    'Hfc (Bright Female)': {
        'model': 'piper_models/en_US-hfc_female-medium.onnx',
        'config': 'piper_models/en_US-hfc_female-medium.onnx.json',
        'gender': 'Female',
        'description': 'Bright, energetic, enthusiastic - Great for engaging content'
    },
    'Arctic (Neutral Male)': {
        'model': 'piper_models/en_US-arctic-medium.onnx',
        'config': 'piper_models/en_US-arctic-medium.onnx.json',
        'gender': 'Male',
        'description': 'Neutral, calm, steady - Good for narration'
    }
}

DEFAULT_SPEAKER_VOICES = {
    'Host': 'Lessac (Warm Male)',
    'Guest': 'LibriTTS (Clear Female)',
    'Speaker 1': 'Ryan (Deep Male)',
    'Speaker 2': 'Hfc (Bright Female)',
    'Narrator': 'Arctic (Neutral Male)'
}

# ==================== HELPER FUNCTIONS ====================
@st.cache_resource
def initialize_rag(pdf_path, chapter_info):
    """Initialize RAG system with PDF and chapter context"""
    try:
        md_text = pymupdf4llm.to_markdown(pdf_path)
        docs = [Document(page_content=md_text)]
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        return vectorstore.as_retriever(search_kwargs={"k": 15})
    except Exception as e:
        st.error(f"Error initializing RAG: {e}")
        return None

def calculate_word_count(duration_minutes):
    """Estimate word count based on podcast duration"""
    words_per_minute = 130
    return duration_minutes * words_per_minute

def format_docs(docs):
    """Format retrieved documents for context block"""
    return "\n\n".join(doc.page_content for doc in docs)

@traceable
def generate_podcast_script(retriever, user_input, form_data):
    """Generate the podcast script using pure LCEL with strict word count enforcement"""
    try:
        language = form_data['language']
        duration = form_data['duration']
        style_key = form_data['style']
        style_desc = PODCAST_STYLES[style_key]
        personalization = form_data['personalization']
        word_count = calculate_word_count(duration)
        min_words = int(word_count * 0.85)
        max_words = int(word_count * 1.15)
        
        system_prompt = f"""You are an expert podcast scriptwriter specializing in educational content.

Language: Write in {language}
Podcast Style: {style_key}
Style Description: {style_desc}
Target Duration: {duration} minutes
EXACT TARGET WORD COUNT: {word_count} words (MINIMUM: {min_words}, MAXIMUM: {max_words})
{f'Additional Notes: {personalization}' if personalization else ''}

CRITICAL FORMATTING RULES:
- Format as a DIALOGUE script with CLEAR speaker labels
- Each line MUST start with the speaker name followed by a colon
- Use ONLY these speaker labels: Host, Guest, Speaker 1, Speaker 2, Narrator
- Example format:
  Host: Welcome to today's episode!
  Guest: Thanks for having me.
  Host: Let's dive into the topic.
- Maintain natural conversation flow with distinct personalities per speaker
- Include transitions, questions, and reactions between speakers
- AVOID: bullet points, markdown formatting, special characters like *, #, _, backticks
- Write plain text only — no formatting, no headers, no lists
- Count words carefully to hit the target duration

Context from course materials:
\n\n
{{context}}"""

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash', temperature=0.7)
        
        rag_chain = (
            {
                "context": retriever | format_docs,
                "input": RunnablePassthrough()
            }
            | prompt_template
            | llm
        )
        
        response = rag_chain.invoke(user_input)
        return response.content
    
    except Exception as e:
        st.error(f"Error generating script: {e}")
        return None

# ==================== TEXT CLEANING FOR TTS ====================
def clean_text_for_tts(text):
    """
    Clean text to prevent Piper from reading escape characters, markdown, etc.
    """
    # Replace literal backslash-n with space
    text = text.replace('\\n', ' ')
    text = text.replace('\\t', ' ')
    text = text.replace('\\r', ' ')
    
    # Remove markdown formatting
    text = re.sub(r'\*\*?(.*?)\*\*?', r'\1', text)
    text = re.sub(r'__(.*?)__', r'\1', text)
    text = re.sub(r'`(.*?)`', r'\1', text)
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    
    # Remove special characters that TTS might read aloud
    text = text.replace('&', 'and')
    text = text.replace('%', 'percent')
    text = text.replace('$', 'dollars')
    text = text.replace('@', 'at')
    text = text.replace('#', 'number')
    text = text.replace('/', ' ')
    text = text.replace('\\', ' ')
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove HTML entities
    text = html.unescape(text)
    
    return text.strip()

# ==================== SPEAKER & AUDIO FUNCTIONS ====================
def extract_speakers_from_script(script):
    """Extract unique speaker names from the script"""
    pattern = r'^\s*([A-Za-z\s\d]+?):\s'
    matches = re.findall(pattern, script, re.MULTILINE)
    
    speakers = []
    for match in set(matches):
        speaker = match.strip()
        if speaker and speaker not in speakers:
            speakers.append(speaker)
    
    speakers.sort()
    return speakers

def parse_script_by_speaker(script):
    """Parse script into list of (speaker, text) tuples"""
    lines = script.split('\n')
    segments = []
    current_speaker = None
    current_text = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        match = re.match(r'^([A-Za-z\s\d]+?):\s*(.*)', line)
        
        if match:
            if current_speaker and current_text:
                full_text = ' '.join(current_text)
                cleaned = clean_text_for_tts(full_text)
                if cleaned:
                    segments.append((current_speaker, cleaned))
            
            current_speaker = match.group(1).strip()
            current_text = [match.group(2)]
        else:
            if current_speaker:
                current_text.append(line)
    
    if current_speaker and current_text:
        full_text = ' '.join(current_text)
        cleaned = clean_text_for_tts(full_text)
        if cleaned:
            segments.append((current_speaker, cleaned))
    
    return segments

def generate_piper_tts(text, model_path, config_path):
    """
    Generate audio using Piper TTS.
    FIXED: Uses echo piped to piper stdin instead of --file to avoid path reading issues.
    """
    try:
        # Clean text thoroughly first
        text = clean_text_for_tts(text)
        
        if not text.strip():
            return None
        
        # Split into manageable chunks (Piper works best with ~400 chars)
        max_chars = 400
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < max_chars:
                current_chunk += " " + sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        all_audio = b''
        
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            
            output_file = os.path.join(os.getcwd(), f"piper_out_{i}.wav")
            
            # FIXED: Use echo command piped to piper instead of --file
            # This avoids any file path interpretation issues
            echo_cmd = f'echo {subprocess.list2cmdline([chunk])}'
            
            # On Windows, use cmd /c to pipe echo to piper
            if os.name == 'nt':  # Windows
                # Write to a simple file first, then use type command
                temp_txt = f"piper_text_{i}.txt"
                with open(temp_txt, 'w', encoding='utf-8') as f:
                    f.write(chunk)
                
                # Use type command (Windows equivalent of cat) piped to piper
                cmd = f'type {temp_txt} | piper --model {os.path.abspath(model_path)} --config {os.path.abspath(config_path)} --output_file {output_file}'
                
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
                
                # Cleanup temp text file
                if os.path.exists(temp_txt):
                    try:
                        os.remove(temp_txt)
                    except:
                        pass
            else:  # Linux/Mac
                cmd = f'echo {subprocess.list2cmdline([chunk])} | piper --model {os.path.abspath(model_path)} --config {os.path.abspath(config_path)} --output_file {output_file}'
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                st.error(f"❌ Piper error: {result.stderr}")
                if os.path.exists(output_file):
                    try:
                        os.remove(output_file)
                    except:
                        pass
                continue
            
            # Read the generated WAV file
            with open(output_file, 'rb') as f:
                all_audio += f.read()
            
            # Cleanup output file
            if os.path.exists(output_file):
                try:
                    os.remove(output_file)
                except:
                    pass
        
        return all_audio if all_audio else None
    
    except subprocess.TimeoutExpired:
        st.error("❌ Piper TTS timed out")
        return None
    except FileNotFoundError:
        st.error("❌ Piper not found. Install with: pip install piper-tts")
        return None
    except Exception as e:
        st.error(f"❌ Error generating speech: {str(e)}")
        return None

def add_silence_to_wav(wav_bytes, silence_ms=400):
    """Add silence to WAV audio for natural pauses between speakers."""
    try:
        with io.BytesIO(wav_bytes) as wav_io:
            with wave.open(wav_io, 'rb') as wav_in:
                nchannels = wav_in.getnchannels()
                sampwidth = wav_in.getsampwidth()
                framerate = wav_in.getframerate()
                audio_data = wav_in.readframes(wav_in.getnframes())
        
        # Create silence
        num_silence_frames = int(framerate * silence_ms / 1000)
        silence_data = b'\x00' * (num_silence_frames * nchannels * sampwidth)
        
        # Combine
        with io.BytesIO() as output:
            with wave.open(output, 'wb') as wav_out:
                wav_out.setnchannels(nchannels)
                wav_out.setsampwidth(sampwidth)
                wav_out.setframerate(framerate)
                wav_out.writeframes(audio_data + silence_data)
            return output.getvalue()
    
    except Exception:
        return wav_bytes

def concatenate_wav_segments(wav_segments):
    """Concatenate multiple WAV byte segments into one WAV file."""
    if not wav_segments:
        return None
    
    try:
        # Parse first segment to get parameters
        with io.BytesIO(wav_segments[0]) as first_io:
            with wave.open(first_io, 'rb') as first_wav:
                nchannels = first_wav.getnchannels()
                sampwidth = first_wav.getsampwidth()
                framerate = first_wav.getframerate()
        
        # Combine all audio data
        all_frames = b''
        for seg in wav_segments:
            try:
                with io.BytesIO(seg) as seg_io:
                    with wave.open(seg_io, 'rb') as seg_wav:
                        all_frames += seg_wav.readframes(seg_wav.getnframes())
            except Exception:
                continue
        
        # Write combined WAV
        with io.BytesIO() as output:
            with wave.open(output, 'wb') as wav_out:
                wav_out.setnchannels(nchannels)
                wav_out.setsampwidth(sampwidth)
                wav_out.setframerate(framerate)
                wav_out.writeframes(all_frames)
            return output.getvalue()
    
    except Exception as e:
        st.error(f"Error concatenating audio: {e}")
        return None

def generate_multi_voice_podcast(script, speaker_voices):
    """Generate full podcast audio with multiple voices using Piper TTS"""
    segments = parse_script_by_speaker(script)
    
    if not segments:
        st.error("No speaker segments found in script")
        return None
    
    total_segments = len(segments)
    all_wav_segments = []
    
    for i, (speaker, text) in enumerate(segments):
        st.write(f"🎙️ Processing {speaker}... ({i+1}/{total_segments}) — {len(text.split())} words")
        
        voice_name = speaker_voices.get(speaker, 'Lessac (Warm Male)')
        voice_config = PIPER_VOICES[voice_name]
        
        # Generate audio for this segment
        segment_audio = generate_piper_tts(
            text,
            voice_config['model'],
            voice_config['config']
        )
        
        if segment_audio:
            # Add silence for natural pause
            segment_with_pause = add_silence_to_wav(segment_audio, silence_ms=400)
            all_wav_segments.append(segment_with_pause)
        else:
            st.warning(f"⚠️ Failed to generate audio for {speaker}, skipping...")
    
    if not all_wav_segments:
        return None
    
    # Concatenate all WAV segments
    combined = concatenate_wav_segments(all_wav_segments)
    return combined

# ==================== UI LAYOUT ====================
st.title('🎙️ AI Podcast Generator - Food Flavour Design')
st.markdown('*Transform Food Flavour Design topics into interactive multi-voice podcasts*')

# Progress indicator (6 steps)
col1, col2, col3, col4, col5, col6 = st.columns(6)
steps = ['Duration', 'Language', 'Chapter', 'Style', 'Voices', 'Generate']
colors = ['#1f77b4' if st.session_state.step >= i+1 else '#cccccc' for i in range(6)]

for idx, (col, step) in enumerate(zip([col1, col2, col3, col4, col5, col6], steps)):
    with col:
        st.markdown(f"""
        <div style='text-align: center; padding: 10px; border-radius: 5px; 
                    background-color: {colors[idx]}; color: white; font-weight: bold; font-size: 0.8rem;'>
            {idx+1}. {step}
        </div>
        """, unsafe_allow_html=True)

st.divider()

# ==================== STEP 1: DURATION ====================
if st.session_state.step >= 1:
    with st.container():
        st.markdown("""
        <div class='step-container'>
            <div class='step-header'>Step 1: Select Podcast Duration</div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            duration = st.slider(
                'How long should the podcast be?',
                min_value=5,
                max_value=20,
                value=st.session_state.form_data['duration'],
                step=1
            )
            word_estimate = calculate_word_count(duration)
            st.info(f'📊 Target: ~{word_estimate} words')
        
        st.session_state.form_data['duration'] = duration
        
        if st.button('✓ Continue to Language', key='btn_step1'):
            st.session_state.step = 2
            st.rerun()

# ==================== STEP 2: LANGUAGE ====================
if st.session_state.step >= 2:
    st.divider()
    with st.container():
        st.markdown("""
        <div class='step-container'>
            <div class='step-header'>Step 2: Select Language</div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            language = st.selectbox(
                'In which language should the podcast be?',
                options=LANGUAGES,
                index=LANGUAGES.index(st.session_state.form_data['language'])
            )
        
        st.session_state.form_data['language'] = language
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button('← Back', key='btn_back2'):
                st.session_state.step = 1
                st.rerun()
        with col2:
            if st.button('✓ Continue to Chapter', key='btn_step2'):
                st.session_state.step = 3
                st.rerun()

# ==================== STEP 3: CHAPTER ====================
if st.session_state.step >= 3:
    st.divider()
    with st.container():
        st.markdown("""
        <div class='step-container'>
            <div class='step-header'>Step 3: Select Module & Chapter</div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            module = st.selectbox(
                'Select a Module:',
                options=list(CHAPTERS.keys())
            )
        
        with col2:
            chapter = st.selectbox(
                'Select a Chapter:',
                options=CHAPTERS[module]
            )
        
        st.session_state.form_data['chapter'] = f"{module} - {chapter}"
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button('← Back', key='btn_back3'):
                st.session_state.step = 2
                st.rerun()
        with col2:
            if st.button('✓ Continue to Podcast Style', key='btn_step3'):
                st.session_state.step = 4
                st.rerun()

# ==================== STEP 4: PODCAST STYLE ====================
if st.session_state.step >= 4:
    st.divider()
    with st.container():
        st.markdown("""
        <div class='step-container'>
            <div class='step-header'>Step 4: Choose Podcast Style & Tone</div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            style = st.selectbox(
                'Select a Podcast Style:',
                options=list(PODCAST_STYLES.keys()),
                index=list(PODCAST_STYLES.keys()).index(st.session_state.form_data['style']) if st.session_state.form_data['style'] else 0
            )
        
        with col2:
            st.markdown("**Style Description:**")
            st.info(PODCAST_STYLES[style])
        
        st.session_state.form_data['style'] = style
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button('← Back', key='btn_back4'):
                st.session_state.step = 3
                st.rerun()
        with col2:
            if st.button('✓ Continue to Voice Selection', key='btn_step4'):
                st.session_state.step = 5
                st.rerun()

# ==================== STEP 5: VOICE ASSIGNMENT ====================
if st.session_state.step >= 5:
    st.divider()
    with st.container():
        st.markdown("""
        <div class='step-container'>
            <div class='step-header'>Step 5: Assign Voices to Speakers</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Check if Piper models exist
        models_exist = all(os.path.exists(v['model']) for v in PIPER_VOICES.values())
        if not models_exist:
            st.warning("""
            ⚠️ **Piper model files not found!**
            
            Download models from: https://huggingface.co/rhasspy/piper-voices/tree/main
            
            Create a `piper_models` folder and download these files:
            - `en_US-lessac-medium.onnx` + `.json`
            - `en_US-libritts-high.onnx` + `.json`
            - `en_US-ryan-medium.onnx` + `.json`
            - `en_US-hfc_female-medium.onnx` + `.json`
            - `en_US-arctic-medium.onnx` + `.json`
            """)
        
        if st.session_state.generated_script:
            speakers = extract_speakers_from_script(st.session_state.generated_script)
            st.session_state.detected_speakers = speakers
            st.success(f"🎭 Detected {len(speakers)} speakers: {', '.join(speakers)}")
        else:
            speakers = ['Host', 'Guest']
            st.info("🎭 Default speakers: Host and Guest")
        
        st.divider()
        st.markdown("**Assign a voice to each speaker:**")
        
        for speaker in speakers:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown(f"### 🎤 {speaker}")
            
            with col2:
                default_voice = DEFAULT_SPEAKER_VOICES.get(speaker, 'Lessac (Warm Male)')
                current_selection = st.session_state.form_data['speaker_voices'].get(speaker, default_voice)
                
                available_voices = list(PIPER_VOICES.keys())
                
                voice = st.selectbox(
                    f'Select voice for {speaker}:',
                    options=available_voices,
                    index=available_voices.index(current_selection) if current_selection in available_voices else 0,
                    key=f'voice_{speaker}'
                )
                
                voice_info = PIPER_VOICES[voice]
                st.caption(f"{voice_info['gender']} | {voice_info['description']}")
                
                if not os.path.exists(voice_info['model']):
                    st.error(f"❌ Model file not found: {voice_info['model']}")
                
                st.session_state.form_data['speaker_voices'][speaker] = voice
        
        st.divider()
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write('**Audio Generation:**')
            st.write('Convert to multi-voice audio using Piper TTS (Free, local, open-source)')
        with col2:
            generate_audio = st.checkbox(
                'Generate Audio',
                value=st.session_state.form_data['generate_audio']
            )
            st.session_state.form_data['generate_audio'] = generate_audio
        
        st.divider()
        st.write('**Optional: Add any specific requirements:**')
        personalization = st.text_area(
            'Personalization Notes (optional)',
            value=st.session_state.form_data['personalization'],
            placeholder='e.g., "Focus on practical applications"',
            height=100
        )
        st.session_state.form_data['personalization'] = personalization
        
        st.markdown('### 📋 Configuration Summary:')
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"**Duration:** {st.session_state.form_data['duration']} min\n\n**Language:** {st.session_state.form_data['language']}")
        with c2:
            st.markdown(f"**Chapter:** {st.session_state.form_data['chapter']}\n\n**Style:** {st.session_state.form_data['style']}")
        with c3:
            voices_summary = "\n".join([f"**{s}:** {v}" for s, v in st.session_state.form_data['speaker_voices'].items()])
            st.markdown(f"**Voices:**\n{voices_summary}")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if st.button('← Back', key='btn_back5'):
                st.session_state.step = 4
                st.rerun()
        with col2:
            btn_label = '🎙️ Generate Podcast Script & Audio' if generate_audio else '📝 Generate Podcast Script Only'
            if st.button(btn_label, key='btn_generate', use_container_width=True):
                with st.spinner('🔄 Generating your podcast...'):
                    pdf_path = "C:\\Users\\Karim\\Downloads\\Insights into flavor and key influencing factors of Maillard reaction products_ A recent update - fnut-09-973677.pdf"
                    
                    retriever = initialize_rag(pdf_path, st.session_state.form_data['chapter'])
                    
                    if retriever:
                        script = generate_podcast_script(
                            retriever,
                            f"Create a podcast about {st.session_state.form_data['chapter']}",
                            st.session_state.form_data
                        )
                        
                        if script:
                            # Verify word count
                            word_count = len(script.split())
                            target = calculate_word_count(st.session_state.form_data['duration'])
                            st.session_state.generated_script = script
                            st.success(f'✓ Script generated: {word_count} words (target: ~{target})')
                            
                            detected = extract_speakers_from_script(script)
                            st.session_state.detected_speakers = detected
                            
                            for speaker in detected:
                                if speaker not in st.session_state.form_data['speaker_voices']:
                                    st.session_state.form_data['speaker_voices'][speaker] = DEFAULT_SPEAKER_VOICES.get(speaker, 'Lessac (Warm Male)')
                            
                            if st.session_state.form_data['generate_audio']:
                                with st.spinner(f'🎵 Generating multi-voice audio with {len(detected)} speakers...'):
                                    audio_bytes = generate_multi_voice_podcast(
                                        script,
                                        st.session_state.form_data['speaker_voices']
                                    )
                                    
                                    if audio_bytes:
                                        st.session_state.generated_audio = audio_bytes
                                        st.session_state.audio_file_name = f"podcast_{st.session_state.form_data['chapter'].replace(' - ', '_').replace(' ', '_')[:50]}.wav"
                                        st.success('✅ Multi-voice audio generated!')
                                        st.balloons()
                            else:
                                st.balloons()
                            
                            st.session_state.step = 6
                            st.rerun()

# ==================== STEP 6: DISPLAY OUTPUT ====================
if st.session_state.step >= 6 and st.session_state.generated_script:
    st.divider()
    st.markdown('### 🎬 Generated Podcast')
    
    # Show word count info
    actual_words = len(st.session_state.generated_script.split())
    target_words = calculate_word_count(st.session_state.form_data['duration'])
    st.info(f"📊 Script: {actual_words} words | Target: ~{target_words} words | Duration estimate: ~{actual_words // 130} minutes")
    
    tab1, tab2 = st.tabs(['📝 Script', '🎵 Audio'])
    
    with tab1:
        st.markdown(f"""
        **Duration:** {st.session_state.form_data['duration']} minutes | 
        **Language:** {st.session_state.form_data['language']} | 
        **Style:** {st.session_state.form_data['style']}
        """)
        
        if st.session_state.detected_speakers:
            st.markdown("**🎭 Speaker Voices:**")
            cols = st.columns(len(st.session_state.detected_speakers))
            for col, speaker in zip(cols, st.session_state.detected_speakers):
                voice = st.session_state.form_data['speaker_voices'].get(speaker, 'Lessac (Warm Male)')
                col.markdown(f"**{speaker}:** {voice}")
        
        st.text_area(
            'Podcast Script:',
            value=st.session_state.generated_script,
            height=500,
            disabled=True
        )
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label='📥 Download Script',
                data=st.session_state.generated_script,
                file_name=f"podcast_script_{st.session_state.form_data['chapter'].replace(' - ', '_').replace(' ', '_')[:50]}.txt",
                mime='text/plain'
            )
    
    with tab2:
        if st.session_state.generated_audio:
            st.markdown('### 🎧 Your Multi-Voice Podcast')
            st.audio(st.session_state.generated_audio, format='audio/wav')
            st.info(f'✅ Audio ready! Speakers: {", ".join(st.session_state.detected_speakers)}')
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label='📥 Download Audio (WAV)',
                    data=st.session_state.generated_audio,
                    file_name=st.session_state.audio_file_name,
                    mime='audio/wav'
                )
        else:
            st.info('No audio generated. Go back to Step 5 to enable audio generation.')
    
    st.divider()
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button('🔄 Generate Another', key='btn_new'):
            st.session_state.step = 1
            st.session_state.generated_script = None
            st.session_state.generated_audio = None
            st.session_state.detected_speakers = []
            st.rerun()
    
    with col2:
        if st.button('📝 Modify Settings', key='btn_modify'):
            st.session_state.step = 5
            st.rerun()
    
    with col3:
        if st.button('🎵 Regenerate Audio Only', key='btn_regen_audio'):
            if st.session_state.generated_script:
                with st.spinner('🎵 Regenerating multi-voice audio...'):
                    detected = extract_speakers_from_script(st.session_state.generated_script)
                    st.session_state.detected_speakers = detected
                    
                    audio_bytes = generate_multi_voice_podcast(
                        st.session_state.generated_script,
                        st.session_state.form_data['speaker_voices']
                    )
                    if audio_bytes:
                        st.session_state.generated_audio = audio_bytes
                        st.success('✅ Audio regenerated!')
                        st.rerun()


