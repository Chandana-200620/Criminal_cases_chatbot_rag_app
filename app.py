# ---------------- SIDEBAR ----------------

st.sidebar.title("⚖️ CASE CHATBOT")
st.sidebar.caption("TOP 10 CRIMINAL CASES IN INDIA")

page = st.sidebar.radio(
    "Navigation",
    [
        "Chatbot",
        "RAG Pipeline",
        "Info"
    ],
    key="navigation"
)

if page == "Info":

    info_page = st.sidebar.radio(
        "Information",
        [
            "Description",
            "Technologies Used"
        ],
        key="info_navigation"
    )
