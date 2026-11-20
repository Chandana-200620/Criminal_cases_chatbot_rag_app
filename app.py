
import streamlit as st
from rag_engine import load_base, ask_rag

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="CASE_CHATBOT",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>
[data-testid="stSidebar"] {
    background-color: #EAF2F8;
    min-width: 260px;
}
[data-testid="stSidebar"] button {
    width: 100%;
    border-radius: 8px;
    font-size: 16px;
}
[data-testid="stSidebar"] h1 {
    color: #154360;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SESSION STATE ----------------

if "page" not in st.session_state:
    st.session_state.page = "Chatbot"

if "info_page" not in st.session_state:
    st.session_state.info_page = "Description"

if "chat_hist" not in st.session_state:
    st.session_state.chat_hist = []

# ---------------- SIDEBAR ----------------

st.sidebar.title("⚖️ CASE CHATBOT")
st.sidebar.caption("Criminal Cases in India")

if st.sidebar.button("💬 Chatbot", use_container_width=True):
    st.session_state.page = "Chatbot"

if st.sidebar.button("🔄 RAG Pipeline", use_container_width=True):
    st.session_state.page = "Pipeline"

if st.sidebar.button("ℹ️ Info", use_container_width=True):
    st.session_state.page = "Info"

# Info submenu

if st.session_state.page == "Info":

    st.sidebar.divider()
    st.sidebar.subheader("Information")

    if st.sidebar.button("📖 Description", use_container_width=True):
        st.session_state.info_page = "Description"

    if st.sidebar.button("💻 Technologies Used", use_container_width=True):
        st.session_state.info_page = "Technologies"

# ---------------- CHATBOT PAGE ----------------

if st.session_state.page == "Chatbot about Criminal cases":

    st.title("TOP 10 CRIMINAL CASES IN INDIA")

    st.caption(
        "Explore important criminal cases, investigations, "
        "court proceedings and judgments in India."
    )

    st.caption("RAG Pipeline with Multiple PDF Files")
    

    mykey = st.secrets["GROQ_API_KEY"]

    @st.cache_resource
    def initiate_rag():
        return load_base("datafiles_docs")

    index, embed_model, chunks = initiate_rag()

    # Display chat history

    for qa in st.session_state.chat_hist:

        with st.chat_message(qa["role"]):
            st.write(qa["content"])

    # Ask question

    question = st.chat_input(
        "Ask anything about criminal cases in India..."
    )

    if question:

        st.session_state.chat_hist.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):

            with st.spinner("Searching case documents..."):

                answer = ask_rag(
                    question,
                    embed_model,
                    chunks,
                    index,
                    api_key=mykey
                )

                st.write(answer)

                st.session_state.chat_hist.append({
                    "role": "assistant",
                    "content": answer
                })

    # Clear chat

    if st.button("Clear Chat"):

        st.session_state.chat_hist = []
        st.rerun()


# ---------------- RAG PIPELINE PAGE ----------------

elif st.session_state.page == "Pipeline":

    st.title("🔄 RAG PIPELINE")

    st.subheader("Retrieval Augmented Generation")

    st.write("""
    This pipeline explains how the chatbot processes
    multiple criminal case documents and generates
    answers based on retrieved information.
    """)

    st.graphviz_chart("""
    digraph RAG {

        rankdir=TB;

        node [
            shape=box,
            style="rounded,filled",
            fillcolor="#D6EAF8",
            color="#2874A6",
            fontname="Arial"
        ];

        A [label="Criminal Case PDF Documents"];
        B [label="Extract Text"];
        C [label="Text Chunking"];
        D [label="Generate Embeddings"];
        E [label="Store Vectors in FAISS"];

        F [label="User Question"];
        G [label="Question Embedding"];
        H [label="Similarity Search"];
        I [label="Retrieve Relevant Information"];
        J [label="Groq LLM"];
        K [label="Generate Final Answer"];

        A -> B -> C -> D -> E;
        F -> G -> H -> I;
        E -> H;
        I -> J -> K;
    }
    """)


# ---------------- INFO PAGE ----------------

elif st.session_state.page == "Info":

    # PROJECT DESCRIPTION

    if st.session_state.info_page == "Description":

        st.title("📖 PROJECT DESCRIPTION")

        st.subheader("Top 10 Criminal Cases in India")

        st.write("""
        The Top 10 Criminal Cases in India chatbot is an
        AI-powered application developed using
        Retrieval Augmented Generation (RAG).

        The main objective of this project is to provide
        information about notable criminal cases in
        Indian history.

        The application uses multiple PDF documents
        containing information about different cases.

        Users can ask questions related to:

        1. Background of criminal cases
        2. Important events and timelines
        3. Investigation and evidence
        4. Court proceedings
        5. Judgments and legal decisions
        6. Legal significance of cases

        The RAG system retrieves relevant information
        from uploaded documents and uses a Large
        Language Model to generate answers.

        The application is designed for educational
        and legal awareness purposes.
        """)

        st.info(
            "Note: The chatbot provides educational information, "
            "not legal advice. It distinguishes allegations from "
            "court-established findings."
        )

    # TECHNOLOGIES USED

    elif st.session_state.info_page == "Technologies":

        st.title("💻 TECHNOLOGIES USED")

        st.subheader("1. Programming Language")
        st.write("Python")

        st.subheader("2. Frontend Framework")
        st.write("Streamlit")

        st.subheader("3. AI and Machine Learning")

        st.write("""
        - Sentence Transformers
        - Text Embeddings
        - Retrieval Augmented Generation
        """)

        st.subheader("4. Vector Database")
        st.write("FAISS")

        st.subheader("5. Large Language Model")
        st.write("Groq API")

        st.subheader("6. Supporting Libraries")

        st.write("""
        - NumPy
        - PyPDF
        - Streamlit
        - Sentence Transformers
        """)

        st.subheader("7. Document Processing")

        st.write("""
        - Multiple PDF loading
        - Text extraction
        - Text chunking
        - Semantic search
        - Context retrieval
        """)
