#Introduction 
import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# 1. Setup AP Keys
load_dotenv()

# 2. Configure the Streamlit Page
st.set_page_config(page_title='Podcast Generator Demo', page_icon='🎙️')
st.title('🎙️ AI Podcast Script Generator')
st.write('Enter a Topic below to generate a short podcast script.')

# 3. Sidebar for Settings (Character Limit)
with st.sidebar:
    st.header('Settings')
    char_limit = st.slider('Max Characters for Prompt',50,500,250)

# 4. User Input With Character Limit
user_input = st.text_area('What the podcast should be about?', max_chars=char_limit, placeholder=' e.g., The impact of AI on marine Biology....')

# 5. The Logic: Connect to gemini 
if st.button('Generate Script'):
    if user_input:
        try:
            #Initialize the Free Gemini Flah model
            llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

            with st.spinner('writing your script....'):
                #Simple prompt to get a dialogue
                prompt=f'create a short 2 person podcast dialogue about: {user_input}. Format it as Host: and Guest:'
                response=llm.invoke(prompt)

                st.subheader('Generated Script')
                st.write(response.content)

                # We save this in session state so we can use it for audio later
                st.session_state['script'] = response.content


        except Exception as e:
            st.error(f'An error occured: {e}')
    else:
        st.warning('please enter a topic first!')        