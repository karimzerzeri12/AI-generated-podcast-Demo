import streamlit as st
import os
from dotenv import load_dotenv
import io
import re
import html
import time
import base64

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
ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY')
COURSE_PDF_PATH = os.getenv('COURSE_PDF_PATH')

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
        'generate_audio': True,
        'tts_provider': 'ElevenLabs'
    }
if 'generated_script' not in st.session_state:
    st.session_state.generated_script = None
if 'generated_audio' not in st.session_state:
    st.session_state.generated_audio = None
if 'audio_file_name' not in st.session_state:
    st.session_state.audio_file_name = None
if 'detected_speakers' not in st.session_state:
    st.session_state.detected_speakers = []
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'rag_initialized' not in st.session_state:
    st.session_state.rag_initialized = False

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

# ==================== ELEVENLABS VOICES ====================
ELEVENLABS_VOICES = {
    'Rachel': {
        'id': 'EXAVITQu4vr4xnSDxMaL',
        'description': 'Warm, friendly, professional - Great for educational content',
        'gender': 'Female'
    },
    'Chris': {
        'id': 'iP95p4xoKVk53GoZ742B',
        'description': 'Clear, engaging, confident tone',
        'gender': 'Male'
    },
    'Bella': {
        'id': 'XB0fDUnXU5powFXDhCwa',
        'description': 'Bright, energetic, enthusiastic',
        'gender': 'Female'
    },
    'Giovanni': {
        'id': 'zcAOhNBS3c14rBihAFp1',
        'description': 'Deep, authoritative, expert tone',
        'gender': 'Male'
    },
    'Ethan': {
        'id': 'g5CIjZEefAph4nQFvHAz',
        'description': 'Young, friendly, relatable',
        'gender': 'Male'
    },
    'Sofia': {
        'id': 'EB1811eVaBlNVzP59GaM',
        'description': 'Calm, clear, educational',
        'gender': 'Female'
    }
}

DEFAULT_SPEAKER_VOICES = {
    'Host': 'Rachel',
    'Guest': 'Giovanni',
    'Speaker 1': 'Chris',
    'Speaker 2': 'Sofia',
    'Narrator': 'Rachel'
}

# ==================== HELPER FUNCTIONS ====================
@st.cache_resource
def initialize_rag(pdf_path):
    """Initialize RAG system with PDF from env path"""
    try:
        if not pdf_path or not os.path.exists(pdf_path):
            return None

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
    """Generate the podcast script using RAG with strict word count enforcement"""
    try:
        language = form_data['language']
        duration = form_data['duration']
        style_key = form_data['style']
        style_desc = PODCAST_STYLES[style_key]
        personalization = form_data['personalization']
        word_count = calculate_word_count(duration)
        min_words = int(word_count * 0.90)
        max_words = int(word_count * 1.05)

        if retriever:
            context_docs = retriever.invoke(user_input)
            context = format_docs(context_docs)
            context_block = f"""Context from course materials:
{context}"""
        else:
            context_block = f"""Topic: {form_data['chapter']}
This is a podcast about Food Flavour Design. Please generate educational content based on your knowledge of this topic.
No course materials are available — use your general knowledge about food science, flavour chemistry, and sensory perception."""

        system_prompt = f"""You are an expert podcast scriptwriter specializing in educational content.

Language: Write in {language}
Podcast Style: {style_key}
Style Description: {style_desc}
Target Duration: EXACTLY {duration} minutes
TARGET WORD COUNT: EXACTLY {word_count} words
HARD MINIMUM: {min_words} words
HARD MAXIMUM: {max_words} words
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
- COUNT WORDS CAREFULLY: The script MUST be between {min_words} and {max_words} words
- If you need to adjust length, add or remove conversational filler, examples, or transitions
- Do NOT exceed {max_words} words under any circumstances

{context_block}"""

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash', temperature=0.7)

        if retriever:
            rag_chain = (
                {
                    "context": retriever | format_docs,
                    "input": RunnablePassthrough()
                }
                | prompt_template
                | llm
            )
            response = rag_chain.invoke(user_input)
        else:
            messages = prompt_template.format_messages(input=user_input)
            response = llm.invoke(messages)

        script = response.content

        # Post-process: enforce word count
        words = script.split()
        actual_count = len(words)

        if actual_count > max_words:
            truncated = ' '.join(words[:max_words])
            last_period = truncated.rfind('.')
            if last_period > max_words * 0.8:
                script = truncated[:last_period + 1]
            else:
                script = truncated
            st.warning(f"⚠️ Script was truncated from {actual_count} to ~{max_words} words to match duration")
        elif actual_count < min_words:
            st.warning(f"⚠️ Script is {actual_count} words (target: {word_count}). Consider regenerating with more context.")

        return script

    except Exception as e:
        st.error(f"Error generating script: {e}")
        return None

# ==================== TEXT CLEANING FOR TTS ====================
def clean_text_for_tts(text):
    """Clean text to prevent TTS from reading escape characters, markdown, etc."""
    text = text.replace('\n', ' ')
    text = text.replace('\t', ' ')
    text = text.replace('\r', ' ')

    text = re.sub(r'\*\*?(.*?)\*\*?', r'\1', text)
    text = re.sub(r'__(.*?)__', r'\1', text)
    text = re.sub(r'`(.*?)`', r'\1', text)
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)

    text = text.replace('&', 'and')
    text = text.replace('%', 'percent')
    text = text.replace('$', 'dollars')
    text = text.replace('@', 'at')
    text = text.replace('#', 'number')
    text = text.replace('/', ' ')
    text = text.replace('\\', ' ')

    text = re.sub(r'\s+', ' ', text)
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
    """Parse script into list of (speaker, text) tuples - strips labels for audio"""
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

# ==================== ELEVENLABS TTS ====================
def generate_elevenlabs_tts(text, voice_id):
    """Generate audio using ElevenLabs API with high quality model."""
    if not ELEVENLABS_API_KEY:
        st.error("❌ ElevenLabs API key not found. Add ELEVENLABS_API_KEY to .env")
        return None

    text = clean_text_for_tts(text)
    if not text or not text.strip():
        st.warning("⚠️ Empty text segment, skipping...")
        return None

    try:
        import requests

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

        headers = {
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json"
        }

        max_chunk = 2500
        chunks = []
        current_chunk = ""
        sentences = re.split(r'(?<=[.!?])\s+', text)

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(current_chunk) + len(sentence) + 1 <= max_chunk:
                current_chunk += sentence + " "
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        if not chunks:
            st.warning("⚠️ No valid text chunks after splitting")
            return None

        all_audio = []

        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue

            st.write(f"📝 Processing audio chunk {i+1} of {len(chunks)}...")

            payload = {
                "text": chunk,
                "model_id": "eleven_multilingual_v2",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.75
                }
            }

            response = requests.post(url, json=payload, headers=headers, timeout=120)

            if response.status_code != 200:
                st.error(f"❌ ElevenLabs API error: {response.status_code}")
                st.error(response.text)
                return None

            all_audio.append(response.content)

        if not all_audio:
            return None

        try:
            from pydub import AudioSegment
            combined = AudioSegment.empty()
            for seg in all_audio:
                audio = AudioSegment.from_mp3(io.BytesIO(seg))
                combined += audio + AudioSegment.silent(duration=400)

            combined = combined[:-400] if len(combined) > 400 else combined

            buf = io.BytesIO()
            combined.export(buf, format="mp3", bitrate="192k")
            return buf.getvalue()

        except ImportError:
            st.warning("pydub not installed, using basic concatenation (may have glitches)")
            silent_frame = b'\xff\xfb\x90\x00' + b'\x00' * 418
            result = b''
            for i, seg in enumerate(all_audio):
                result += seg
                if i < len(all_audio) - 1:
                    result += silent_frame
            return result

    except Exception as e:
        st.error(f"❌ ElevenLabs error: {str(e)}")
        return None

# ==================== AUDIO GENERATION ====================
def generate_multi_voice_podcast(script, speaker_voices):
    """Generate full podcast audio with multiple voices using ElevenLabs."""
    segments = parse_script_by_speaker(script)

    if not segments:
        st.error("No speaker segments found in script")
        return None

    total_segments = len(segments)
    all_audio_segments = []

    for i, (speaker, text) in enumerate(segments):
        word_count = len(text.split()) if text else 0
        st.write(f"🎙️ Processing {speaker}... ({i+1}/{total_segments}) — {word_count} words")

        voice_name = speaker_voices.get(speaker, 'Rachel')

        if voice_name not in ELEVENLABS_VOICES:
            st.warning(f"⚠️ Voice '{voice_name}' not found, using default 'Rachel'")
            voice_name = 'Rachel'

        voice_config = ELEVENLABS_VOICES[voice_name]

        segment_audio = generate_elevenlabs_tts(text, voice_config['id'])

        if segment_audio:
            all_audio_segments.append(segment_audio)
        else:
            st.warning(f"⚠️ Failed to generate audio for {speaker}, skipping...")

    if not all_audio_segments:
        return None

    try:
        from pydub import AudioSegment
        combined = AudioSegment.empty()
        for seg in all_audio_segments:
            audio = AudioSegment.from_mp3(io.BytesIO(seg))
            combined += audio + AudioSegment.silent(duration=500)

        combined = combined[:-500] if len(combined) > 500 else combined

        buf = io.BytesIO()
        combined.export(buf, format="mp3", bitrate="192k")
        return buf.getvalue()

    except ImportError:
        st.warning("pydub not installed, using basic concatenation")
        return b''.join(all_audio_segments)

# ==================== UI LAYOUT ====================
st.title('🎙️ AI Podcast Generator - Food Flavour Design')
st.markdown('*Transform Food Flavour Design topics into interactive multi-voice podcasts*')

# Auto-initialize RAG on first load if PDF path is set
if not st.session_state.rag_initialized and COURSE_PDF_PATH:
    with st.spinner('📚 Loading course materials from environment...'):
        retriever = initialize_rag(COURSE_PDF_PATH)
        if retriever:
            st.session_state.retriever = retriever
            st.session_state.rag_initialized = True
        else:
            st.session_state.retriever = None
            st.session_state.rag_initialized = True

# Progress indicator (5 steps)
col1, col2, col3, col4, col5 = st.columns(5)
steps = ['Duration', 'Language', 'Chapter', 'Style', 'Generate']
colors = ['#1f77b4' if st.session_state.step >= i+1 else '#cccccc' for i in range(5)]

for idx, (col, step) in enumerate(zip([col1, col2, col3, col4, col5], steps)):
    with col:
        st.markdown(f"""
        <div style='text-align: center; padding: 10px; border-radius: 5px; 
                    background-color: {colors[idx]}; color: white; font-weight: bold; font-size: 0.8rem;'>
            {idx+1}. {step}
        </div>
        """, unsafe_allow_html=True)

# RAG status indicator
if COURSE_PDF_PATH and st.session_state.retriever:
    st.success(f"📚 RAG Active: Using course PDF from `{COURSE_PDF_PATH}`")
elif COURSE_PDF_PATH and not st.session_state.retriever:
    st.warning(f"⚠️ PDF path set but file not found: `{COURSE_PDF_PATH}`")
else:
    st.info("📚 No COURSE_PDF_PATH set — using general knowledge")

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
            st.info(f'📊 Target: ~{word_estimate} words ({duration} min × 130 WPM)')

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

# ==================== STEP 4: PODCAST STYLE & VOICES ====================
if st.session_state.step >= 4:
    st.divider()
    with st.container():
        st.markdown("""
        <div class='step-container'>
            <div class='step-header'>Step 4: Choose Podcast Style & Assign Voices</div>
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

        st.divider()

        # ElevenLabs info
        st.markdown("### 🎛️ Text-to-Speech Provider")
        st.info("Using **ElevenLabs** for high-quality, natural-sounding multi-voice audio generation.")

        if not ELEVENLABS_API_KEY:
            st.error("❌ ElevenLabs API key not found! Add ELEVENLABS_API_KEY to your .env file")

        st.divider()

        # Voice Selection
        if st.session_state.generated_script:
            speakers = extract_speakers_from_script(st.session_state.generated_script)
            st.session_state.detected_speakers = speakers
            st.success(f"🎭 Detected {len(speakers)} speakers: {', '.join(speakers)}")
        else:
            speakers = ['Host', 'Guest']
            st.info("🎭 Default speakers: Host and Guest")

        st.markdown("**Assign a voice to each speaker:**")

        for speaker in speakers:
            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown(f"### 🎤 {speaker}")

            with col2:
                default_voice = DEFAULT_SPEAKER_VOICES.get(speaker, list(ELEVENLABS_VOICES.keys())[0])
                current_selection = st.session_state.form_data['speaker_voices'].get(speaker, default_voice)

                voice = st.selectbox(
                    f'Select voice for {speaker}:',
                    options=list(ELEVENLABS_VOICES.keys()),
                    index=list(ELEVENLABS_VOICES.keys()).index(current_selection) if current_selection in ELEVENLABS_VOICES else 0,
                    key=f'voice_{speaker}'
                )

                voice_info = ELEVENLABS_VOICES[voice]
                st.caption(f"{voice_info['gender']} | {voice_info['description']}")

                st.session_state.form_data['speaker_voices'][speaker] = voice

        st.divider()
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write('**Audio Generation:**')
            st.write('Convert to multi-voice audio using ElevenLabs')
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
            st.markdown(f"**TTS:** ElevenLabs")
            if st.session_state.retriever:
                st.markdown("✅ **RAG:** Course PDF active")
            else:
                st.markdown("📚 **RAG:** General knowledge")

        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if st.button('← Back', key='btn_back4'):
                st.session_state.step = 3
                st.rerun()
        with col2:
            btn_label = '🎙️ Generate Podcast Script & Audio' if generate_audio else '📝 Generate Podcast Script Only'

            if st.button(btn_label, key='btn_generate', use_container_width=True):
                with st.spinner('🔄 Generating your podcast...'):
                    retriever = st.session_state.retriever

                    script = generate_podcast_script(
                        retriever,
                        f"Create a podcast about {st.session_state.form_data['chapter']}",
                        st.session_state.form_data
                    )

                    if script:
                        word_count = len(script.split())
                        target = calculate_word_count(st.session_state.form_data['duration'])
                        st.session_state.generated_script = script
                        st.success(f'✓ Script generated: {word_count} words (target: ~{target})')

                        detected = extract_speakers_from_script(script)
                        st.session_state.detected_speakers = detected

                        for speaker in detected:
                            if speaker not in st.session_state.form_data['speaker_voices']:
                                st.session_state.form_data['speaker_voices'][speaker] = DEFAULT_SPEAKER_VOICES.get(speaker, list(ELEVENLABS_VOICES.keys())[0])

                        if st.session_state.form_data['generate_audio']:
                            with st.spinner(f'🎵 Generating multi-voice audio with {len(detected)} speakers...'):
                                audio_bytes = generate_multi_voice_podcast(
                                    script,
                                    st.session_state.form_data['speaker_voices']
                                )

                                if audio_bytes:
                                    st.session_state.generated_audio = audio_bytes
                                    st.session_state.audio_file_name = f"podcast_{st.session_state.form_data['chapter'].replace(' - ', '_').replace(' ', '_')[:50]}.mp3"
                                    st.success('✅ Multi-voice audio generated!')
                                    st.balloons()
                        else:
                            st.balloons()

                        st.session_state.step = 5
                        st.rerun()

# ==================== STEP 5: DISPLAY OUTPUT ====================
if st.session_state.step >= 5 and st.session_state.generated_script:
    st.divider()
    st.markdown('### 🎬 Generated Podcast')

    actual_words = len(st.session_state.generated_script.split())
    target_words = calculate_word_count(st.session_state.form_data['duration'])
    duration_estimate = actual_words // 130
    st.info(f"📊 Script: {actual_words} words | Target: ~{target_words} words | Estimated audio: ~{duration_estimate} minutes")

    tab1, tab2 = st.tabs(['📝 Script', '🎵 Audio'])

    with tab1:
        st.markdown(f"""
        **Duration:** {st.session_state.form_data['duration']} minutes | 
        **Language:** {st.session_state.form_data['language']} | 
        **Style:** {st.session_state.form_data['style']} |
        **TTS:** ElevenLabs
        """)

        if st.session_state.detected_speakers:
            st.markdown("**🎭 Speaker Voices:**")
            cols = st.columns(len(st.session_state.detected_speakers))
            for col, speaker in zip(cols, st.session_state.detected_speakers):
                voice = st.session_state.form_data['speaker_voices'].get(speaker, 'Default')
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
            st.audio(st.session_state.generated_audio, format='audio/mp3')
            st.info(f'✅ Audio ready! Speakers: {", ".join(st.session_state.detected_speakers)}')

            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label='📥 Download Audio (MP3)',
                    data=st.session_state.generated_audio,
                    file_name=st.session_state.audio_file_name,
                    mime='audio/mp3'
                )
        else:
            st.info('No audio generated. Go back to Step 4 to enable audio generation.')

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
            st.session_state.step = 4
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

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown('### 📚 About')
    st.markdown("""
    **AI Podcast Generator - Food Flavour Design**

    Multi-voice podcasts powered by ElevenLabs!
    """)

    st.divider()
    st.markdown('### ⚙️ API Status')
    if ELEVENLABS_API_KEY:
        st.success('✅ ElevenLabs API Key found')
    else:
        st.error('❌ ElevenLabs API Key not set! Add ELEVENLABS_API_KEY to your .env file')

    st.markdown("""
    **Required in .env:**
    ```
    GOOGLE_API_KEY=your_key_here
    ELEVENLABS_API_KEY=your_key_here
    COURSE_PDF_PATH=/path/to/your/course.pdf
    ```
    """)

    st.divider()
    st.markdown('### 💰 ElevenLabs Plans')
    st.markdown("""
    | Plan | Price | Characters |
    |------|-------|------------|
    | Free | $0 | 10k/mo |
    | Starter | $5/mo | 30k/mo |
    | Creator | $22/mo | 100k/mo |
    """)

    st.divider()
    st.markdown('### 📦 Dependencies')
    st.markdown("""
    Make sure you have these installed:
    ```bash
    pip install pydub pymupdf4llm langchain-huggingface langchain-community faiss-cpu
    ```
    """)
