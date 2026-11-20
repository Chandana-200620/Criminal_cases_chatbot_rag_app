
import streamlit as st
from rag_engine import load_base, ask_rag

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="CASE CHATBOT",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>
    [data-testid="stSidebar"] {
        min-width: 260px;
        max-width: 260px;
    }

    .main-title {
        text-align: center;
        font-size: 32px;
        font-weight: bold;
        color: #2563eb;
    }

    .sub-title {
        text-align: center;
        font-size: 16px;
        color: gray;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# --------------------------------------------------
# GET GROQ API KEY
# --------------------------------------------------

def get_api_key():
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return ""


# --------------------------------------------------
# LOAD RAG DATABASE
# --------------------------------------------------

@st.cache_resource(show_spinner=False)
def initialize_rag():
    return load_base("datafiles_docs")


# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.title("⚖️ CASE CHATBOT")
st.sidebar.caption("TOP 10 CRIMINAL CASES IN INDIA")

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    ["Chatbot", "RAG Pipeline", "Info"],
    key="main_navigation"
)

info_page = None

if page == "Info":
    info_page = st.sidebar.radio(
        "Information",
        ["Description", "Technologies Used"],
        key="info_navigation"
    )

st.sidebar.divider()

st.sidebar.caption("RAG Based Question Answering System")


# --------------------------------------------------
# CHATBOT PAGE
# --------------------------------------------------

if page == "Chatbot":

    st.markdown(
        '<p class="main-title">TOP 10 CRIMINAL CASES IN INDIA</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="sub-title">AI Powered Retrieval Augmented Generation Chatbot</p>',
        unsafe_allow_html=True
    )

    st.divider()

    st.subheader("💬 Ask Your Question")

    st.caption(
        "Ask questions about the criminal cases available in the uploaded PDF documents."
    )

    # Get API key
    api_key = get_api_key()

    if not api_key:
        st.error(
            "Groq API key is missing. Add GROQ_API_KEY to "
            ".streamlit/secrets.toml."
        )

    else:

        # Load RAG resources
        try:
            with st.spinner("Loading PDF documents and RAG database..."):
                index, embed_model, chunks = initialize_rag()

            st.success("RAG database loaded successfully.")

            # Display previous chat messages
            for message in st.session_state.chat_history:

                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Chat input
            user_question = st.chat_input(
                "Enter your question here..."
            )

            if user_question:

                # Display user question
                with st.chat_message("user"):
                    st.markdown(user_question)

                st.session_state.chat_history.append({
                    "role": "user",
                    "content": user_question
                })

                # Generate answer
                with st.chat_message("assistant"):

                    with st.spinner("Searching documents and generating answer..."):

                        try:
                            answer = ask_rag(
                                question=user_question,
                                embed_model=embed_model,
                                chunks=chunks,
                                index=index,
                                api_key=api_key,
                                num_chunks=3
                            )

                            st.markdown(answer)

                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": answer
                            })

                        except Exception as e:
                            st.error(f"Error generating answer: {e}")

        except Exception as e:
            st.error(f"Unable to load the RAG database: {e}")

    # Clear chat button
    if st.session_state.chat_history:

        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()


# --------------------------------------------------
# RAG PIPELINE PAGE
# --------------------------------------------------

elif page == "RAG Pipeline":

    st.title("⚙️ RAG PIPELINE")

    st.write(
        "The following diagram explains how the chatbot "
        "processes PDF documents and answers user questions."
    )

    st.divider()

    pipeline = """
    digraph RAG {
        rankdir=TB;
        bgcolor="transparent";

        node [
            shape=box,
            style="rounded,filled",
            fillcolor="#DBEAFE",
            color="#2563EB",
            fontname="Arial",
            fontsize=12,
            margin=0.2
        ];

        edge [
            color="#64748B",
            arrowsize=0.8
        ];

        A [label="PDF Documents"];
        B [label="Text Extraction"];
        C [label="Text Chunking"];
        D [label="Text Embeddings"];
        E [label="FAISS Vector Database"];

        F [label="User Question"];
        G [label="Question Embedding"];
        H [label="Similarity Search"];
        I [label="Relevant Chunks"];
        J [label="Groq LLM"];
        K [label="Generated Answer"];

        A -> B -> C -> D -> E;
        F -> G -> H;
        E -> H;
        H -> I -> J -> K;
    }
    """

    st.graphviz_chart(
        pipeline,
        use_container_width=True
    )

    st.divider()

    st.subheader("Pipeline Explanation")

    steps = [
        ("1. PDF Loading", "Loads criminal case information from PDF documents."),
        ("2. Text Extraction", "Extracts readable text from the PDF files using PyPDF."),
        ("3. Text Chunking", "Divides extracted text into smaller overlapping chunks."),
        ("4. Embedding Generation", "Converts text chunks into numerical vectors using Sentence Transformers."),
        ("5. FAISS Indexing", "Stores vectors in a FAISS index for similarity searching."),
        ("6. User Query", "Accepts a question from the user."),
        ("7. Similarity Search", "Finds the most relevant document chunks."),
        ("8. Context Retrieval", "Provides the retrieved information as context to the language model."),
        ("9. Answer Generation", "Groq generates an answer using the retrieved context.")
    ]

    for heading, description in steps:
        st.markdown(f"**{heading}**")
        st.write(description)


# --------------------------------------------------
# INFO PAGE
# --------------------------------------------------

elif page == "Info":

    if info_page == "Description":

        st.title("📘 PROJECT DESCRIPTION")

        st.divider()

        st.subheader("About the Project")

        st.write("""
        The Top 10 Criminal Cases in India RAG Chatbot is an
        AI-powered question-answering application.

        It uses Retrieval Augmented Generation (RAG) to retrieve
        relevant information from PDF documents and generate
        answers to user questions.

        Instead of relying only on the language model's
        general knowledge, the application uses information
        retrieved from the uploaded documents.
        """)

        st.subheader("Main Features")

        st.markdown("""
        - PDF document loading
        - Text extraction and chunking
        - Text embedding generation
        - FAISS similarity search
        - Relevant information retrieval
        - AI-powered answer generation
        - Interactive chatbot interface
        - Chat history management
        """)

        st.subheader("Criminal Case Documents")

        cases = [
            "Nirbhaya Case",
            "Jessica Lal Case",
            "Aarushi Talwar Case",
            "Priyadarshini Mattoo Case",
            "Bilkis Bano Case",
            "Satyam Scam",
            "Sister Abhaya Case",
            "Ruchika Girhotra Case",
            "Best Bakery Case",
            "Rajiv Gandhi Assassination Case"
        ]

        for number, case in enumerate(cases, start=1):
            st.write(f"{number}. {case}")

        st.info(
            "This project is intended for educational and "
            "informational purposes only. It is not a source "
            "of legal advice."
        )

    # ----------------------------------------------
    # TECHNOLOGIES USED
    # ----------------------------------------------

    elif info_page == "Technologies Used":

        st.title("💻 TECHNOLOGIES USED")

        st.divider()

        technologies = [
            ("Python", "Main programming language."),
            ("Streamlit", "Used to build the web application interface."),
            ("PyPDF", "Extracts text from PDF documents."),
            ("Sentence Transformers", "Generates text embeddings."),
            ("FAISS", "Stores and searches vector embeddings."),
            ("NumPy", "Performs numerical operations on embeddings."),
            ("Groq API", "Connects the application to the language model."),
            ("RAG", "Combines document retrieval with answer generation.")
        ]

        for technology, description in technologies:

            st.subheader(technology)
            st.write(description)
            st.divider()
