import streamlit as st
import tempfile
import os
import re
import plotly.graph_objects as go
from rag import load_and_index_pdf, get_qa_chain

st.set_page_config(
    page_title="FinSight",
    page_icon=None,
    layout="wide"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .block-container { padding-top: 2rem; }
    .chat-user {
        background: #1c1e2e;
        border: 1px solid #2a2d3e;
        padding: 14px 18px;
        border-radius: 10px;
        margin: 10px 0;
        font-size: 0.92rem;
        color: #e5e7eb;
    }
    .chat-assistant {
        background: #111318;
        border: 1px solid #1e3a5f;
        border-left: 3px solid #0866ff;
        padding: 14px 18px;
        border-radius: 10px;
        margin: 10px 0;
        font-size: 0.92rem;
        color: #d1d5db;
        line-height: 1.7;
    }
    .source-chip {
        display: inline-block;
        background: #1c1e2e;
        color: #6b7280;
        font-size: 0.72rem;
        padding: 3px 10px;
        border-radius: 20px;
        margin: 3px 2px;
        border: 1px solid #2a2d3e;
    }
    .section-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #6b7280;
        margin-bottom: 12px;
    }
    .divider { border: none; border-top: 1px solid #2a2d3e; margin: 24px 0; }
</style>
""", unsafe_allow_html=True)

if "indexed" not in st.session_state:
    st.session_state.indexed = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def try_make_chart(answer):
    lines = answer.split('\n')
    labels, values = [], []
    for line in lines:
        match = re.search(r'(.+?):\s*\$?([\d,]+\.?\d*)', line)
        if match:
            labels.append(match.group(1).strip())
            try:
                values.append(float(match.group(2).replace(',', '')))
            except:
                pass
    if len(labels) >= 2:
        fig = go.Figure(go.Bar(
            x=labels,
            y=values,
            marker_color='#0866ff',
            marker_line_width=0
        ))
        fig.update_layout(
            paper_bgcolor='#111318',
            plot_bgcolor='#111318',
            font=dict(color='#6b7280', family='Inter'),
            margin=dict(l=20, r=20, t=30, b=20),
            height=280,
            xaxis=dict(gridcolor='#1c1e2e'),
            yaxis=dict(gridcolor='#1c1e2e')
        )
        return fig
    return None

# Sidebar
with st.sidebar:
    st.markdown('<div class="section-label">Document</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload PDF", type="pdf", label_visibility="collapsed")

    if uploaded_file and not st.session_state.indexed:
        with st.spinner("Indexing..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            load_and_index_pdf(tmp_path)
            os.unlink(tmp_path)
            st.session_state.indexed = True
        st.success("Ready")

    if st.session_state.indexed:
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        if st.button("New Document", use_container_width=True):
            st.session_state.indexed = False
            st.session_state.chat_history = []
            st.rerun()

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Quick Questions</div>', unsafe_allow_html=True)
    quick = [
        "What was the total revenue?",
        "How did net income change?",
        "What was the gross margin?",
        "What are the main risk factors?",
        "How did operating expenses change?"
    ]
    for q in quick:
        if st.button(q, key=q, use_container_width=True):
            st.session_state.pending_question = q

# Main
st.markdown("## FinSight")
st.markdown('<div style="color:#6b7280; font-size:0.9rem; margin-bottom:2rem;">Financial document intelligence</div>', unsafe_allow_html=True)

if st.session_state.indexed:
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f'<div class="chat-user">{message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-assistant">{message["content"]}</div>', unsafe_allow_html=True)
            if "sources" in message:
                source_html = " ".join([f'<span class="source-chip">p. {s}</span>' for s in message["sources"]])
                st.markdown(source_html, unsafe_allow_html=True)
            if "chart" in message and message["chart"]:
                st.plotly_chart(message["chart"], use_container_width=True)

    question = st.chat_input("Ask about the report...")

    if "pending_question" in st.session_state:
        question = st.session_state.pending_question
        del st.session_state.pending_question

    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.spinner("Thinking..."):
            retriever, chain = get_qa_chain()
            answer = chain.invoke(question)
            sources = retriever.invoke(question)
            pages = list(set([str(doc.metadata.get('page', '?')) for doc in sources]))
            chart = try_make_chart(answer)
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": pages,
            "chart": chart
        })
        st.rerun()

else:
    st.markdown("""
    <div style="text-align:center; padding:100px 0; color:#3d4152;">
        <div style="font-size:0.95rem; letter-spacing:0.5px;">Upload a PDF to begin</div>
    </div>
    """, unsafe_allow_html=True)