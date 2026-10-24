import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage 

# ──────────────────────────────────────────────────────────────
# Page configuration
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Custom Assistant",
    page_icon=":)",
    layout="centered",
    initial_sidebar_state="expanded",
) 
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

# ──────────────────────────────────────────────────────────────
# Session state
# ──────────────────────────────────────────────────────────────
st.session_state.setdefault("messages", [])      # chat history (Human / AI only)
st.session_state.setdefault("api_key", "")       # OpenAI key supplied by the user
st.session_state.setdefault("auth_error", "")

is_authed = bool(st.session_state.api_key)

# ──────────────────────────────────────────────────────────────
# Styling
# ──────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --bg: #0b0f1a;
    --surface: #121829;
    --surface-2: #182036;
    --border: rgba(255,255,255,0.08);
    --text: #e8ecf5;
    --muted: #8b95b0;
    --accent: #7c8cff;
    --accent-2: #b48cff;
    --success: #4ade80;
}

html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif;
}

/* App background */
.stApp {
    background:
        radial-gradient(900px 500px at 10% -10%, rgba(124,140,255,0.14), transparent 60%),
        radial-gradient(700px 450px at 100% 0%, rgba(180,140,255,0.10), transparent 55%),
        var(--bg);
    color: var(--text);
}

/* Hide default Streamlit chrome */
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
.block-container { padding-top: 2.5rem; padding-bottom: 7rem; max-width: 780px; }

/* Hero */
.hero { text-align: center; margin-bottom: 1.8rem; }
.hero .badge {
    display: inline-flex; align-items: center; gap: .45rem;
    padding: .3rem .8rem; border-radius: 999px;
    background: rgba(124,140,255,0.12); border: 1px solid rgba(124,140,255,0.28);
    color: var(--accent); font-size: .74rem; font-weight: 600; letter-spacing: .06em;
    text-transform: uppercase; margin-bottom: 1rem;
}
.hero h1 {
    font-size: 2.5rem; font-weight: 700; letter-spacing: -0.03em; margin: 0 0 .4rem 0;
    background: linear-gradient(120deg, #ffffff 20%, var(--accent) 60%, var(--accent-2) 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero p { color: var(--muted); font-size: 1rem; margin: 0; }

/* Auth card */
.auth-card {
    background: linear-gradient(180deg, var(--surface-2), var(--surface));
    border: 1px solid var(--border); border-radius: 18px;
    padding: 1.6rem 1.8rem 0.4rem 1.8rem;
    box-shadow: 0 20px 50px rgba(0,0,0,0.35);
    margin-bottom: .6rem;
}
.auth-card h3 { margin: 0 0 .3rem 0; font-size: 1.15rem; font-weight: 600; color: var(--text); }
.auth-card p { margin: 0 0 .6rem 0; color: var(--muted); font-size: .9rem; line-height: 1.5; }
.privacy-note { color: var(--muted); font-size: .78rem; text-align: center; margin-top: .9rem; }

/* Inputs */
.stTextInput input, .stTextArea textarea {
    background: var(--bg) !important; color: var(--text) !important;
    border: 1px solid var(--border) !important; border-radius: 12px !important;
    padding: .7rem .9rem !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(124,140,255,0.18) !important;
}

/* Buttons */
.stButton > button, .stFormSubmitButton > button {
    width: 100%; border-radius: 12px; border: 1px solid var(--border);
    background: var(--surface-2); color: var(--text); font-weight: 500;
    padding: .6rem 1rem; transition: all .2s ease;
}
.stButton > button:hover, .stFormSubmitButton > button:hover {
    border-color: var(--accent); color: #fff; transform: translateY(-1px);
}
.stFormSubmitButton > button {
    background: linear-gradient(120deg, var(--accent), var(--accent-2));
    border: none; color: #0b0f1a; font-weight: 600;
}
.stFormSubmitButton > button:hover { color: #0b0f1a; filter: brightness(1.08); }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--surface); border-right: 1px solid var(--border);
}
.side-title { font-size: .72rem; font-weight: 600; letter-spacing: .09em;
    text-transform: uppercase; color: var(--muted); margin: .4rem 0 .5rem 0; }
.status-pill {
    display: inline-flex; align-items: center; gap: .5rem; padding: .35rem .8rem;
    border-radius: 999px; font-size: .8rem; font-weight: 500;
    background: rgba(74,222,128,0.10); border: 1px solid rgba(74,222,128,0.28); color: var(--success);
}
.status-pill.off { background: rgba(139,149,176,0.10); border-color: var(--border); color: var(--muted); }
.dot { width: 7px; height: 7px; border-radius: 50%; background: currentColor; }

/* Chat messages */
[data-testid="stChatMessage"] {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 16px; padding: 1rem 1.2rem; margin-bottom: .8rem;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: linear-gradient(180deg, rgba(124,140,255,0.10), rgba(124,140,255,0.04));
    border-color: rgba(124,140,255,0.22);
}

/* Empty state */
.empty {
    text-align: center; padding: 3rem 1rem; color: var(--muted);
    border: 1px dashed var(--border); border-radius: 18px; background: rgba(255,255,255,0.015);
}
.empty .icon { font-size: 2rem; margin-bottom: .5rem; color: var(--accent); }
.empty b { color: var(--text); font-weight: 600; }

/* Chat input (the "search bar") */
[data-testid="stChatInput"] {
    border: 1px solid var(--border); border-radius: 16px; background: var(--surface-2);
    box-shadow: 0 10px 30px rgba(0,0,0,0.35);
}
[data-testid="stChatInput"]:focus-within {
    border-color: var(--accent); box-shadow: 0 0 0 3px rgba(124,140,255,0.18);
}
[data-testid="stChatInput"] textarea { color: var(--text) !important; }
[data-testid="stBottom"] > div { background: transparent; }
</style>
""",
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ✦ Assistant")

    if is_authed:
        st.markdown(
            '<span class="status-pill"><span class="dot"></span>API key connected</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="status-pill off"><span class="dot"></span>Not connected</span>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="side-title">Behaviour</div>', unsafe_allow_html=True)
    system_message = st.text_area(
        "System role",
        value=DEFAULT_SYSTEM_PROMPT,
        height=120,
        disabled=not is_authed,
        help="Defines how the assistant should behave. Applied to every request.",
    )

    st.markdown('<div class="side-title">Model</div>', unsafe_allow_html=True)
    model_name = st.text_input("Model name", value="gpt-5.2", disabled=not is_authed)
    temperature = st.slider(
        "Creativity", 0.0, 1.0, 0.5, 0.05, disabled=not is_authed,
        help="Lower = focused and deterministic, higher = more creative.",
    )

    st.markdown('<div class="side-title">Session</div>', unsafe_allow_html=True)
    if st.button("Clear conversation", disabled=not st.session_state.messages):
        st.session_state.messages = []
        st.rerun()
    if is_authed and st.button("Change API key"):
        st.session_state.api_key = ""
        st.session_state.messages = []
        st.rerun()

# ──────────────────────────────────────────────────────────────
# Hero header
# ──────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="hero">
    <div class="badge">✦ AI Workspace</div>
    <h1>Your Custom Assistant</h1>
    <p>A focused, private space to think, draft and explore ideas.</p>
</div>
""",
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────
# Step 1 – API key gate
# ──────────────────────────────────────────────────────────────
if not is_authed:
    st.markdown(
        """
<div class="auth-card">
    <h3>🔐 Connect your OpenAI account</h3>
    <p>Enter your API key to unlock the assistant. The key is kept only in this
    browser session and is never written to disk.</p>
</div>
""",
        unsafe_allow_html=True,
    )

    with st.form("api_key_form", clear_on_submit=True):
        key_input = st.text_input(
            "OpenAI API key",
            type="password",
            placeholder="sk-...",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Unlock assistant →")

    if submitted:
        key = key_input.strip()
        if len(key) < 20 or not key.startswith("sk-"):
            st.session_state.auth_error = (
                "That doesn't look like a valid OpenAI key. It should start with “sk-”."
            )
        else:
            st.session_state.api_key = key
            st.session_state.auth_error = ""
            st.rerun()

    if st.session_state.auth_error:
        st.error(st.session_state.auth_error)

    st.markdown(
        '<div class="privacy-note">Get a key at platform.openai.com/api-keys</div>',
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────────────────────
# Step 2 – Chat (unlocked only after a key is provided)
# ──────────────────────────────────────────────────────────────
else:
    if not st.session_state.messages:
        st.markdown(
            """
<div class="empty">
    <div class="icon">✦</div>
    <b>You're all set.</b><br>
    Ask a question below to start the conversation.
</div>
""",
            unsafe_allow_html=True,
        )

    for msg in st.session_state.messages:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user", avatar="🧑"):
                st.write(msg.content)
        elif isinstance(msg, AIMessage):
            with st.chat_message("assistant", avatar="✦"):
                st.write(msg.content)

# The search / message bar – disabled until an API key is supplied
user_prompt = st.chat_input(
    "Ask anything…" if is_authed else "Enter your API key above to start chatting",
    disabled=not is_authed,
)

if is_authed and user_prompt:
    st.session_state.messages.append(HumanMessage(content=user_prompt))
    with st.chat_message("user", avatar="🧑"):
        st.write(user_prompt)

    try:
        chat = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=st.session_state.api_key,
        )
        # System message is prepended per request so edits in the sidebar apply instantly
        payload = [SystemMessage(content=system_message or DEFAULT_SYSTEM_PROMPT)]
        payload += st.session_state.messages

        with st.chat_message("assistant", avatar="✦"):
            with st.spinner("Thinking…"):
                response = chat.invoke(payload)
            st.write(response.content)

        st.session_state.messages.append(AIMessage(content=response.content))

    except Exception as e:
        # Roll back the unanswered user turn and surface a friendly error
        st.session_state.messages.pop()
        err = str(e)
        if "401" in err or "invalid_api_key" in err or "Incorrect API key" in err:
            st.session_state.api_key = ""
            st.session_state.auth_error = "Your API key was rejected. Please enter a valid key."
            st.rerun()
        else:
            st.error(f"Something went wrong: {err}")
