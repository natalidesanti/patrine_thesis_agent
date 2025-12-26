"""
Streamlit app for the thesis agent

Loads ONLY the preprocessed thesis index and answers questions locally using Ollama.

"""

#------------------
# Libraries
#------------------

import json
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from ollama import Client
from pathlib import Path
import re

#------------------
# Data
#------------------

@st.cache_resource
def load_index(index_dir: str):
    '''Load preprocessed data'''
    index_dir = Path(index_dir)
    chunks = json.loads((index_dir / "chunks.json").read_text(encoding = "utf-8"))
    embeddings = np.load(index_dir / "embeddings.npy")
    config = json.loads((index_dir / "config.json").read_text(encoding = "utf-8"))
    embedder = SentenceTransformer(config["embedding_model"])
    return chunks, embeddings, embedder, config

def retrieve_chunks(question, chunks, embeddings, embedder, top_k = 5):
    '''Retrieve data'''
    q_emb = embedder.encode([question], convert_to_numpy = True, normalize_embeddings = True)
    sims = cosine_similarity(q_emb, embeddings)[0]
    idx = sims.argsort()[-top_k:][::-1]
    return [chunks[i] for i in idx]

#------------------
# Thesis agent
#------------------

ollama_client = Client(host = "http://localhost:11434")

SYSTEM_PROMPT = (
    "You are an expert assistant trained exclusively on Natalí Soler Matubaro de Santi's PhD thesis."
    "Use ONLY the provided thesis excerpts."
    "Do NOT use external knowledge or speculation."
    "If the answer is not present, say: 'This is not discussed in the thesis.'"
)

def generate_answer(question, context, model = "llama3"):
    prompt = (SYSTEM_PROMPT + f"Thesis excerpts: {context}"
        f"Question: {question} Answer:" )
    response = ollama_client.generate(model = model, prompt = prompt, stream = False)
    return response["response"].strip()

#------------------
# Verifier
#------------------

VERIFICATION_PROMPT = (
    "You are a strict but fair scientific reviewer.\n"
    "Your task is to assess whether each STATEMENT is supported by the EXCERPTS.\n\n"
    "Rules:\n"
    "- Use ONLY the provided excerpts.\n"
    "- Do NOT require exact wording match.\n"
    "- A statement is SUPPORTED if it is a reasonable paraphrase or inference.\n"
    "- A statement is NOT_SUPPORTED only if it introduces new facts, methods, results, or claims.\n\n"
    "Respond in JSON:\n"
    "{\n"
    "  \"verdict\": \"SUPPORTED\" | \"PARTIALLY_SUPPORTED\" | \"NOT_SUPPORTED\",\n"
    "  \"explanation\": \"short explanation\"\n"
    "}\n"
)

def split_sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.!?])\\s+', text) if len(s.strip()) > 10]

def verify_answer(answer, context, model = "llama3"):

    sentences = split_sentences(answer)

    results = []
    for s in sentences:
        prompt = (VERIFICATION_PROMPT + f"\nEXCERPTS:\n{context}\n\n" f"STATEMENT:\n{s}\n")

        response = ollama_client.generate(model = model, prompt = prompt, stream = False)

        try:
            verdict = json.loads(response["response"].strip())
        except Exception:
            verdict = {"verdict": "UNKNOWN", "explanation": "Parsing failed"}

        verdict["sentence"] = s
        results.append(verdict)

    return results

#------------------
# Streamlit
#------------------

st.set_page_config(page_title = "Thesis Agent", layout = "wide")
st.title("📚 Patrine: Natalí de Santi's PhD Thesis Agent")

st.markdown(f"Hello there, I am **Patrine**, Natalí de Santi's PhD Thesis Agent!")

THESIS_PDF_URL = "https://www.teses.usp.br/teses/disponiveis/43/43134/tde-15072024-101341/publico/tesenatalisolermatubarodesanti.pdf"
st.markdown(f"📄 **Thesis PDF:** Download [here]({THESIS_PDF_URL})", unsafe_allow_html = False)

st.markdown("I assistant answer questions *only* based on the content of the thesis, and I might hallucinate a bit 🤪")

AUTHOR_EMAIL = "natalidesanti@gmail.com"
st.markdown(f"For clarifications, interpretations, or discussions please contact Natalí directly at **{AUTHOR_EMAIL}**")

with st.sidebar:
    st.image(
        "data/patrine_agent.png",
        use_container_width = True
    )
    st.markdown(
        "**Patrine: Natalí de Santi's Thesis Agent**  \n"
        "I answer questions strictly grounded to Natalís de Santi's PhD thesis"
    )

    st.divider()

    st.markdown("### Agent Settings")
    index_dir = st.text_input("Thesis index directory", value = "data/thesis_index")
    model = st.selectbox("Ollama model", ["llama3", "mistral"])
    top_k = st.slider("Retrieved chunks", 2, 10, 5)

try:
    chunks, embeddings, embedder, config = load_index(index_dir)
    st.caption(f"Loaded {config['num_chunks']} chunks from thesis")
except Exception as e:
    st.error(f"Failed to load thesis index: {e}")
    st.stop()

question = st.text_input("Ask a question about the thesis")

if question:
    with st.spinner("Retrieving and answering"):
        retrieved = retrieve_chunks(question, chunks, embeddings, embedder, top_k = top_k)
        context = "".join([f"[Excerpt {i+1}] {c}" for i, c in enumerate(retrieved)])
        answer = generate_answer(question, context, model=model)
        verifications = verify_answer(answer, context, model=model)

        supported = sum(v["verdict"] == "SUPPORTED" for v in verifications)
        partial = sum(v["verdict"] == "PARTIALLY_SUPPORTED" for v in verifications)
        unsupported = sum(v["verdict"] == "NOT_SUPPORTED" for v in verifications)

        if unsupported == 0 and partial == 0:
            st.success("✅ All statements supported by the thesis.")
        elif unsupported == 0:
            st.warning("⚠️ Some statements are only partially supported.")
        else:
            st.error("❌ Some statements are not supported by the thesis.")

    st.subheader("Answer")
    st.markdown(answer)

    with st.expander("Retrieved excerpts"):
        st.text(context)
