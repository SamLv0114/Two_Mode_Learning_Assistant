"""
Streamlit UI for the AI Learning Assistant
"""
import sys
from pathlib import Path
import json
import time

import streamlit as st
import streamlit.components.v1 as components

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.pipelines import DailyFeedPipeline, QAAssistant
from src.utils.config import settings
from src.utils.preprocessing import extract_text_from_pdf


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
if "feed_result" not in st.session_state:
    st.session_state.feed_result = None
if "time_window_days" not in st.session_state:
    st.session_state.time_window_days = 7
if "focus_areas" not in st.session_state:
    st.session_state.focus_areas = []
if "interest_phrases" not in st.session_state:
    st.session_state.interest_phrases = []
if "trainer" not in st.session_state:
    st.session_state.trainer = st.session_state.pipeline.trainer


# Helper utilities for consistent interaction tracking
def open_external_link(url: str):
    """Open a link in a new tab; includes a timestamp to avoid caching the iframe content."""
    safe_url = json.dumps(url)
    components.html(
        f"<script>window.open({safe_url}, '_blank');</script><div>{int(time.time()*1000)}</div>",
        height=0,
        width=0,
    )


def record_latest_interaction(item_type: str, item_id: int, interaction_type: str) -> bool:
    """
    Record/update the latest interaction for an item.
    Returns True if the interaction changed or was newly recorded.
    """
    state_key = f"last_interaction_{item_type}_{item_id}"
    last = st.session_state.get(state_key)
    if last == interaction_type:
        return False
    st.session_state.trainer.record_interaction(item_type, item_id, interaction_type)
    st.session_state[state_key] = interaction_type
    return True


# Sidebar
st.sidebar.title("📚 AI Learning Assistant")
st.sidebar.markdown("Two-Mode System for ML Research")

mode = st.sidebar.radio(
    "Select Mode",
    ["Daily Feed", "Q&A Assistant"]
)

# Main content
if mode == "Daily Feed":
    st.title("📪 Daily ML Reading Feed")
    st.markdown("Get personalized ML research recommendations")

    # User filters
    time_window_label = {
        "1 Day": 1,
        "1 Week": 7,
        "1 Month": 30,
        "1 Year": 365,
    }
    selected_window = st.selectbox(
        "Time window",
        list(time_window_label.keys()),
        index=list(time_window_label.keys()).index("1 Week"),
        help="Limit recommendations to content published within this window.",
    )
    st.session_state.time_window_days = time_window_label[selected_window]

    available_areas = ["NLP", "ML", "AI", "DL", "CV"]
    selected_areas = st.multiselect(
        "Focus areas (optional)",
        options=available_areas,
        default=st.session_state.focus_areas or ["ML", "AI", "DL"],
        help="Choose one or more areas to prioritize. Leave empty to use default interests.",
    )
    st.session_state.focus_areas = selected_areas

    st.markdown("**Example interests**")
    example_interests = [
        "machine learning and deep learning",
        "natural language processing and transformers",
        "computer vision and image recognition",
        "reinforcement learning and agents",
    ]
    selected_examples = st.multiselect(
        "Select from examples (optional)",
        options=example_interests,
        help="Choose any examples that match your interests; you can also add your own below.",
    )

    custom_interests = st.text_area(
        "Enter your interests (one per line)",
        placeholder="e.g.\nmachine learning and deep learning\nnatural language processing and transformers",
        help="Used to compute semantic similarity; longer, specific phrases work best.",
    )
    user_interest_lines = [line.strip() for line in custom_interests.splitlines() if line.strip()]
    combined_interests = list(dict.fromkeys(selected_examples + user_interest_lines))
    st.session_state.interest_phrases = combined_interests
    
    if st.button("Generate Daily Feed", type="primary"):
        with st.spinner("Generating your daily feed... This may take a few minutes."):
            result = st.session_state.pipeline.run(
                time_window_days=st.session_state.time_window_days,
                focus_areas=st.session_state.interest_phrases or st.session_state.focus_areas,
            )
            st.session_state.feed_result = result  # Store for interaction tracking
            formatted = st.session_state.pipeline.format_for_display(result)
            
            st.success("Daily feed generated!")
            st.markdown("---")
            st.text(formatted)
    
    # Display feed results with interaction buttons
    if st.session_state.feed_result:
        result = st.session_state.feed_result
        
        # Display papers
        if result["papers"]:
            st.subheader(f"📓 Research Papers ({len(result['papers'])})")
            for paper in result["papers"]:
                with st.expander(f"{paper['rank']}. {paper['title']}"):
                    st.markdown(f"**arXiv ID:** {paper['arxiv_id']}")
                    if paper.get("citation_count"):
                        st.markdown(f"**Citations:** {paper['citation_count']}")
                    elif paper.get("impact_score"):
                        st.markdown(f"**Impact score:** {paper['impact_score']}")
                    st.markdown(f"**Summary:** {paper['summary']}")
                    st.markdown(f"**Relevance Score:** {paper['relevance_score']:.3f}")
                
                if paper.get("db_id"):
                    col1, col2, col3 = st.columns(3)
                    view_state_key = f"viewed_paper_{paper['db_id']}"
                    with col1:
                        view_button_key = f"view_paper_{paper['db_id']}"
                        if st.button("📓 View & Open", key=view_button_key):
                            if record_latest_interaction("paper", paper["db_id"], "viewed"):
                                st.session_state[view_state_key] = True
                            open_external_link(paper["url"])
                        if st.session_state.get(view_state_key):
                            st.caption("View logged. Reopening won't add another interaction.")
                    with col2:
                        if st.button("💾 Save", key=f"save_paper_{paper['db_id']}"):
                            if record_latest_interaction("paper", paper["db_id"], "saved"):
                                st.success("Saved! This helps improve recommendations.")
                            else:
                                st.info("Already saved. No duplicate interaction recorded.")
                    with col3:
                        if st.button("✖ Dismiss", key=f"dismiss_paper_{paper['db_id']}"):
                            if record_latest_interaction("paper", paper["db_id"], "dismissed"):
                                st.info("Dismissed. We'll learn from this.")
                            else:
                                st.info("Already dismissed. No duplicate interaction recorded.")
                    
                    st.markdown(f"[Open in new tab]({paper['url']})")
                    st.caption("Use the buttons above to log interactions; the link is an untracked fallback.")
        
        # Display articles
        if result["articles"]:
            st.subheader(f"📰 Tech Articles ({len(result['articles'])})")
            for article in result["articles"]:
                with st.expander(f"{article['rank']}. {article['title']}"):
                    st.markdown(f"**Source:** {article['source']}")
                    st.markdown(f"**Upvotes:** {article['upvotes']}")
                    st.markdown(f"**Summary:** {article['summary']}")
                    st.markdown(f"**Relevance Score:** {article['relevance_score']:.3f}")
                
                if article.get("db_id"):
                    col1, col2, col3 = st.columns(3)
                    view_state_key = f"viewed_article_{article['db_id']}"
                    with col1:
                        view_button_key = f"view_article_{article['db_id']}"
                        if st.button("📰 Read & Open", key=view_button_key):
                            if record_latest_interaction("article", article["db_id"], "viewed"):
                                st.session_state[view_state_key] = True
                            open_external_link(article["url"])
                        if st.session_state.get(view_state_key):
                            st.caption("View logged. Reopening won't add another interaction.")
                    with col2:
                        if st.button("💾 Save", key=f"save_article_{article['db_id']}"):
                            if record_latest_interaction("article", article["db_id"], "saved"):
                                st.success("Saved! This helps improve recommendations.")
                            else:
                                st.info("Already saved. No duplicate interaction recorded.")
                    with col3:
                        if st.button("✖ Dismiss", key=f"dismiss_article_{article['db_id']}"):
                            if record_latest_interaction("article", article["db_id"], "dismissed"):
                                st.info("Dismissed. We'll learn from this.")
                            else:
                                st.info("Already dismissed. No duplicate interaction recorded.")
                    
                    st.markdown(f"[Open in new tab]({article['url']})")
                    st.caption("Use the buttons above to log interactions; the link is an untracked fallback.")
        
        # Manual retrain button
        st.markdown("---")
        interaction_count = st.session_state.trainer.get_interaction_count()
        st.markdown(f"**Total Interactions:** {interaction_count}")
        
        if interaction_count >= 50:
            if st.button("🔄 Retrain Model", type="secondary"):
                with st.spinner("Retraining model with your interactions..."):
                    success = st.session_state.trainer.retrain_model(
                        st.session_state.pipeline.recommender, 
                        min_interactions=50
                    )
                    if success:
                        st.success(f"Model retrained successfully with {interaction_count} interactions!")
                    else:
                        st.warning("Not enough interactions to retrain. Need at least 50.")
        else:
            st.info(f"Collect {50 - interaction_count} more interactions to enable model retraining.")

elif mode == "Q&A Assistant":
    st.title("❓Q&A Assistant")
    st.markdown("Ask questions about your knowledge base")

    st.subheader("Upload study materials")
    uploaded_files = st.file_uploader(
        "Upload .txt, .md, or .pdf files",
        type=["txt", "md", "pdf"],
        accept_multiple_files=True
    )
    if st.button("Add to Knowledge Base", key="upload_docs"):
        if not uploaded_files:
            st.warning("Please upload at least one .txt or .md file.")
        else:
            total_chunks = 0
            for uploaded_file in uploaded_files:
                try:
                    if uploaded_file.type == "application/pdf" or uploaded_file.name.lower().endswith(".pdf"):
                        content = extract_text_from_pdf(uploaded_file.getvalue())
                    else:
                        content = uploaded_file.getvalue().decode("utf-8", errors="ignore")
                except Exception as e:
                    st.warning(f"Failed to read {uploaded_file.name}: {e}")
                    content = ""
                result = st.session_state.qa_assistant.add_user_document(
                    title=uploaded_file.name,
                    content=content,
                    source=uploaded_file.name
                )
                total_chunks += result.get("chunks", 0)
            st.success(f"Added {total_chunks} chunks from {len(uploaded_files)} file(s).")
    
    # Question input
    question = st.text_input(
        "Enter your question:",
        placeholder="e.g., Explain the attention mechanism in transformers"
    )
    
    n_context = st.slider("Number of context documents", 3, 10, 5)
    use_uploads_only = st.checkbox("Use only my uploaded documents", value=False)
    
    if st.button("Ask Question", type="primary") and question:
        with st.spinner("Searching knowledge base and generating answer..."):
            result = st.session_state.qa_assistant.answer_question(
                question=question,
                n_context=n_context,
                filter_type="user_doc" if use_uploads_only else None
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
                    elif citation['type'] == 'article':
                        st.markdown(f"**Type:** Article")
                        st.markdown(f"**Source:** {citation.get('source', 'N/A')}")
                        st.markdown(f"[Read Article]({citation.get('url', '#')})")
                    else:
                        st.markdown("**Type:** User Document")
                        st.markdown(f"**Source:** {citation.get('source', 'N/A')}")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### Statistics")
interaction_count = st.session_state.trainer.get_interaction_count()
st.sidebar.markdown(f"**Interactions:** {interaction_count}")
if interaction_count >= 50:
    st.sidebar.success("✅Ready to retrain!")
else:
    st.sidebar.info(f"📈 Need {50 - interaction_count} more for retraining")

st.sidebar.markdown("---")
st.sidebar.markdown("### Configuration")
st.sidebar.markdown(f"**User Interests:** {', '.join(settings.USER_INTERESTS[:3])}...")
st.sidebar.markdown(f"**LLM Provider:** {settings.LLM_PROVIDER}")
st.sidebar.markdown(f"**Model:** {settings.LLM_MODEL}")
