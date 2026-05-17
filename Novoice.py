import streamlit as st
import os
from dotenv import load_dotenv

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

# Custom CSS for better visual hierarchy
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

# ==================== SESSION STATE INITIALIZATION ====================
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'form_data' not in st.session_state:
    st.session_state.form_data = {
        'duration': 10,
        'language': 'English',
        'chapter': None,
        'style': None,
        'personalization': ''
    }
if 'generated_script' not in st.session_state:
    st.session_state.generated_script = None

# ==================== CONFIGURATION DATA ====================
LANGUAGES = [
    'English', 'Spanish', 'French', 'German', 'Italian', 'Portuguese',
    'Japanese', 'Chinese (Mandarin)', 'Chinese (Cantonese)', 'Korean',
    'Arabic', 'Hindi', 'Russian', 'Turkish', 'Dutch', 'Swedish'
]

# ==================== FOOD FLAVOUR DESIGN COURSE ====================
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

# ==================== HELPER FUNCTIONS ====================
@st.cache_resource
def initialize_rag(pdf_path, chapter_info):
    """Initialize RAG system with PDF and chapter context"""
    try:
        md_text = pymupdf4llm.to_markdown("C:\\Users\\Karim\\Downloads\\Maillard Reaction.pdf")
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
    words_per_minute = 140
    return duration_minutes * words_per_minute

def format_docs(docs):
    """Format retrieved documents for context block"""
    return "\n\n".join(doc.page_content for doc in docs)

@traceable
def generate_podcast_script(retriever, user_input, form_data):
    """Generate the podcast script using pure LCEL (no langchain.chains)"""
    try:
        language = form_data['language']
        duration = form_data['duration']
        style_key = form_data['style']
        style_desc = PODCAST_STYLES[style_key]
        personalization = form_data['personalization']
        word_count = calculate_word_count(duration)
        
        system_prompt = f"""You are an expert podcast scriptwriter specializing in educational content.

Language: Write in {language}
Podcast Style: {style_key}
Style Description: {style_desc}
Target Duration: {duration} minutes (approximately {word_count} words)
{f'Additional Notes: {personalization}' if personalization else ''}

Guidelines:
- Create an engaging, {duration}-minute podcast script
- Use the provided context as the primary source material
- Focus ONLY on facts and information from the context
- Make it suitable for students/learners
- Include natural transitions between sections
- Format as a dialogue script with speaker labels (Host/Guest/Speaker 1, etc.)
- Ensure the content fits the specified time duration
- Maintain an educational yet engaging tone

Context from course materials:
\n\n
{{context}}"""

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash', temperature=0.7)
        
        # Pure LCEL Pipeline
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

# ==================== UI LAYOUT ====================
st.title('🎙️ AI Podcast Generator - Food Flavour Design')
st.markdown('*Transform Food Flavour Design topics into engaging podcast episodes*')

# Progress indicator
col1, col2, col3, col4, col5 = st.columns(5)
steps = ['Duration', 'Language', 'Chapter', 'Style', 'Generate']
colors = ['#1f77b4' if st.session_state.step >= i+1 else '#cccccc' for i in range(5)]

for idx, (col, step) in enumerate(zip([col1, col2, col3, col4, col5], steps)):
    with col:
        st.markdown(f"""
        <div style='text-align: center; padding: 10px; border-radius: 5px; 
                    background-color: {colors[idx]}; color: white; font-weight: bold;'>
            Step {idx+1}: {step}
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
                step=1,
                help='Select a duration between 5-20 minutes'
            )
            word_estimate = calculate_word_count(duration)
            st.info(f'📊 Estimated word count: ~{word_estimate} words')
        
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
                index=LANGUAGES.index(st.session_state.form_data['language']),
                help='Choose from 16+ languages'
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
                options=list(CHAPTERS.keys()),
                help='Choose your module'
            )
        
        with col2:
            chapter = st.selectbox(
                'Select a Chapter:',
                options=CHAPTERS[module],
                help='Choose a specific chapter'
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
        
        # Redesigned to match Step 3 style - clean dropdown instead of cards
        col1, col2 = st.columns(2)
        with col1:
            style = st.selectbox(
                'Select a Podcast Style:',
                options=list(PODCAST_STYLES.keys()),
                index=list(PODCAST_STYLES.keys()).index(st.session_state.form_data['style']) if st.session_state.form_data['style'] else 0,
                help='Choose a style that fits your learning preference'
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
            if st.button('✓ Continue to Generation', key='btn_step4'):
                st.session_state.step = 5
                st.rerun()

# ==================== STEP 5: PERSONALIZATION & GENERATION ====================
if st.session_state.step >= 5:
    st.divider()
    with st.container():
        st.markdown("""
        <div class='step-container'>
            <div class='step-header'>Step 5: Personalization & Generate</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.write('**Optional: Add any specific requirements or focus areas:**')
        personalization = st.text_area(
            'Personalization Notes (optional)',
            value=st.session_state.form_data['personalization'],
            placeholder='e.g., "Focus on practical applications", "Include more examples for beginners", "Add industry case studies"',
            height=100
        )
        
        st.session_state.form_data['personalization'] = personalization
        
        st.markdown('### 📋 Your Podcast Configuration:')
        summary_col1, summary_col2 = st.columns(2)
        
        with summary_col1:
            st.markdown(f"""
            **Duration:** {st.session_state.form_data['duration']} minutes
            
            **Language:** {st.session_state.form_data['language']}
            
            **Chapter:** {st.session_state.form_data['chapter']}
            """)
        
        with summary_col2:
            st.markdown(f"""
            **Style:** {st.session_state.form_data['style']}
            
            **Personalization:** {personalization if personalization else '(None)'}
            """)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            if st.button('← Back', key='btn_back5'):
                st.session_state.step = 4
                st.rerun()
        
        with col2:
            if st.button('🎙️ Generate Podcast Script', key='btn_generate', use_container_width=True):
                with st.spinner('🔄 Generating your podcast script...'):
                    pdf_path = "C:\\Users\\Karim\\Downloads\\Insights into flavor and key influencing factors of Maillard reaction products_ A recent update - fnut-09-973677.pdf"
                    
                    retriever = initialize_rag(pdf_path, st.session_state.form_data['chapter'])
                    
                    if retriever:
                        script = generate_podcast_script(
                            retriever,
                            f"Create a podcast about {st.session_state.form_data['chapter']}",
                            st.session_state.form_data
                        )
                        
                        if script:
                            st.session_state.generated_script = script
                            st.success('✓ Script generated successfully!')
                            st.balloons()

# ==================== DISPLAY GENERATED SCRIPT ====================
if st.session_state.generated_script:
    st.divider()
    st.markdown('### 🎬 Generated Podcast Script')
    
    with st.container():
        st.markdown(f"""
        **Duration:** {st.session_state.form_data['duration']} minutes | 
        **Language:** {st.session_state.form_data['language']} | 
        **Style:** {st.session_state.form_data['style']}
        """)
        
        st.text_area(
            'Podcast Script:',
            value=st.session_state.generated_script,
            height=500,
            disabled=True
        )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.download_button(
                label='📥 Download as Text',
                data=st.session_state.generated_script,
                file_name=f"podcast_script_{st.session_state.form_data['chapter'].replace(' - ', '_').replace(' ', '_')[:50]}.txt",
                mime='text/plain'
            )
        
        with col2:
            if st.button('🔄 Generate Another', key='btn_new'):
                st.session_state.step = 1
                st.session_state.generated_script = None
                st.rerun()
        
        with col3:
            if st.button('📝 Modify & Regenerate', key='btn_modify'):
                st.session_state.step = 4
                st.rerun()