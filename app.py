import validators 
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import UnstructuredURLLoader, YoutubeLoader
import re 
from youtube_transcript_api import YouTubeTranscriptApi


# Streamlit app
st.set_page_config(page_title="URL and Youtube Summarize")
st.title("Summarize Website and Youtube Videos 📝")
st.subheader("Link URL")

# Setting Groq API and URL content to summarize
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type="password")

    if not groq_api_key:
        st.error("Please enter your API Key")

content_url = st.text_input("Input Website or Youtube URL", label_visibility="collapsed")

prompt_template="""
You as assistant that has task to provide summary and explain of the following content in between 500 until 1000 words:
Content:{text}

"""
# gemma2-9b-it LLM use Groq
llm = ChatGroq(model="gemma2-9b-it", groq_api_key=groq_api_key)
prompts = PromptTemplate(template=prompt_template, input_variables=["text"])

if st.button("Summarize"):
    if not groq_api_key.strip() or not content_url.strip():
        st.error("Please input API Key and URL")
    elif not validators.url(content_url):
        st.error("Please enter a valid URL")

    else:
        try:
            with st.spinner("Process..."):

                if "www.youtube.com" in content_url or "youtu.be" in content_url:
                    loader = YoutubeLoader.from_youtube_url(content_url)
                    
                else:
                    loader=UnstructuredURLLoader(urls=[content_url],
                                                headers={
                                                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"
                                                })
                docs = loader.load()

                if not docs:
                    st.warning("Failed to load content please try another link or enter the correct URL")
                    st.stop()

                # Setting chain for summarize
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompts)
                summary = chain.run(docs)
                st.success(summary)

        except Exception as e:
            st.exception(f"Exception:{e}")


