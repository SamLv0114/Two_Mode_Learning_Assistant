"""
Streamlit UI for the AI Learning Assistant
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
from datetime import datetime
from src.pipelines import DailyFeedPipeline, QAAssistant
from src.utils.config import settings

st.set_page_config(
    page_title="AI Learning Assistant",
    page_icon="📚",
    layout="wide"
)

# Check for API key
if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY == "your_openai_api_key_here":
    st.error("⚠️ OPENAI_API_KEY not configured!")
    st.markdown("""
    **Please set up your API key:**
    
    1. Create a `.env` file in the project root
    2. Add your OpenAI API key:
       ```
       OPENAI_API_KEY=sk-your-actual-key-here
       ```
    3. Restart Streamlit
    
    Get your API key from: https://platform.openai.com/api-keys
    """)
    st.stop()

# Initialize session state
if "pipeline" not in st.session_state:
    try:
        st.session_state.pipeline = DailyFeedPipeline()
    except Exception as e:
        st.error(f"Error initializing pipeline: {e}")
        st.stop()
if "qa_assistant" not in st.session_state:
    try:
        st.session_state.qa_assistant = QAAssistant()
    except Exception as e:
        st.error(f"Error initializing Q&A assistant: {e}")
        st.stop()

# Sidebar
st.sidebar.title("📚 AI Learning Assistant")
st.sidebar.markdown("Two-Mode System for ML Research")

mode = st.sidebar.radio(
    "Select Mode",
    ["Daily Feed", "Q&A Assistant"]
)

# Main content
if mode == "Daily Feed":
    st.title("📰 Daily ML Reading Feed")
    st.markdown("Get personalized ML research recommendations")
    
    if st.button("Generate Daily Feed", type="primary"):
        with st.spinner("Generating your daily feed... This may take a few minutes."):
            result = st.session_state.pipeline.run()
            formatted = st.session_state.pipeline.format_for_display(result)
            
            st.success("Daily feed generated!")
            st.markdown("---")
            st.text(formatted)
            
            # Display papers
            if result["papers"]:
                st.subheader(f"📄 Research Papers ({len(result['papers'])})")
                for paper in result["papers"]:
                    with st.expander(f"{paper['rank']}. {paper['title']}"):
                        st.markdown(f"**arXiv ID:** {paper['arxiv_id']}")
                        st.markdown(f"**Citations:** {paper['citation_count']}")
                        st.markdown(f"**Summary:** {paper['summary']}")
                        st.markdown(f"**Relevance Score:** {paper['relevance_score']:.3f}")
                        st.markdown(f"[View Paper]({paper['url']})")
            
            # Display articles
            if result["articles"]:
                st.subheader(f"📰 Tech Articles ({len(result['articles'])})")
                for article in result["articles"]:
                    with st.expander(f"{article['rank']}. {article['title']}"):
                        st.markdown(f"**Source:** {article['source']}")
                        st.markdown(f"**Upvotes:** {article['upvotes']}")
                        st.markdown(f"**Summary:** {article['summary']}")
                        st.markdown(f"**Relevance Score:** {article['relevance_score']:.3f}")
                        st.markdown(f"[Read Article]({article['url']})")

elif mode == "Q&A Assistant":
    st.title("❓ Q&A Assistant")
    st.markdown("Ask questions about your knowledge base")
    
    # Question input
    question = st.text_input(
        "Enter your question:",
        placeholder="e.g., Explain the attention mechanism in transformers"
    )
    
    n_context = st.slider("Number of context documents", 3, 10, 5)
    
    if st.button("Ask Question", type="primary") and question:
        with st.spinner("Searching knowledge base and generating answer..."):
            result = st.session_state.qa_assistant.answer_question(
                question=question,
                n_context=n_context
            )
        
        st.success("Answer generated!")
        st.markdown("---")
        
        # Display answer
        st.subheader("Answer")
        st.markdown(result["answer"])
        
        # Display citations
        if result["citations"]:
            st.subheader("Sources")
            for i, citation in enumerate(result["citations"], 1):
                with st.expander(f"Source {i}: {citation['title']}"):
                    if citation['type'] == 'paper':
                        st.markdown(f"**Type:** Research Paper")
                        st.markdown(f"**arXiv ID:** {citation.get('arxiv_id', 'N/A')}")
                        st.markdown(f"[View Paper]({citation.get('url', '#')})")
                    else:
                        st.markdown(f"**Type:** Article")
                        st.markdown(f"**Source:** {citation.get('source', 'N/A')}")
                        st.markdown(f"[Read Article]({citation.get('url', '#')})")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### Configuration")
st.sidebar.markdown(f"**User Interests:** {', '.join(settings.USER_INTERESTS[:3])}...")
st.sidebar.markdown(f"**LLM Provider:** {settings.LLM_PROVIDER}")
st.sidebar.markdown(f"**Model:** {settings.LLM_MODEL}")

