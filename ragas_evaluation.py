from git import Repo
import os
import streamlit as st
import requests
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAI
from sentence_transformers import CrossEncoder
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.text_splitter import SemanticChunker
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
import time
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import GoogleEmbeddings
from ragas.metrics import faithfulness, context_precision, answer_correctness
from langchain_classic.docstore.document import Document
from typing import List
import torch


st.title("CODEBASE ANALYZER 🤖")



# === Load API keys ===
load_dotenv()

main_llm = GoogleGenerativeAI(model='gemini-2.0-pro', temperature=0.1, max_output_tokens=8192) 
#main_llm=ChatGroq(model='openai/gpt-oss-120b', temperature=0.1, max_tokens=8192)
# === DIRECT GITHUB LOADER (NO CLONING!) ===
embedding = HuggingFaceEmbeddings(model_name='BAAI/bge-base-en-v1.5')

def load_repo_from_github_url(repo_url: str, branch: str = "main") -> List[Document]:
 
    # Extract owner/repo from URL
    if "github.com" not in repo_url:
        raise ValueError("Invalid GitHub URL")
    
    parts = repo_url.strip("/").split("github.com/")[-1]
    if "/" not in parts:
        raise ValueError("URL must contain owner/repo")
    
    owner_repo = parts.split("/", 1)
    if len(owner_repo) != 2:
        raise ValueError("Could not parse owner/repo from URL")
    
    owner, repo = owner_repo[0], owner_repo[1].split("/", 1)[0]
    
    api_url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
    headers = {
        "Accept": "application/vnd.github.v3+json",
        # Optional: Add token to avoid rate limits
        # "Authorization": "token YOUR_GITHUB_TOKEN"
    }
    
    print(f"Fetching file list from {owner}/{repo} ({branch} branch)...")
    response = requests.get(api_url, headers=headers)
    
    if response.status_code != 200:
        error_msg = response.json().get("message", "Unknown error")
        raise Exception(f"GitHub API Error: {error_msg} (Status: {response.status_code})")
    
    tree = response.json().get("tree", [])
    documents = []
    
    # File extensions to include
    INCLUDE_EXT = {".py", ".md", ".txt", ".js", ".jsx", ".ts", ".tsx", 
                   ".html", ".css", ".json", ".yaml", ".yml", ".env", ".gitignore", ".ipynb"}
    
    print(f"Found {len(tree)} items. Loading text files...")
    
    for item in tree:
        if item["type"] != "blob":
            continue
        path = item["path"]
        if not any(path.endswith(ext) for ext in INCLUDE_EXT):
            continue
        
        raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"
        try:
            content_resp = requests.get(raw_url, headers=headers)
            time.sleep(0.1) 
            if content_resp.status_code != 200:
                print(f"Skipped {path} (failed to fetch)")
                continue
                
            content = content_resp.text
            combined_text=f"## FILE: {path}\n\n{content}"
            
            text_for_embedding=(
                  f"File name: {os.path.basename(path)}\n"
                  f"File path: {path}\n\n"
                  f"Repository: {owner}/{repo}\n"
                  f"Branch: {branch}\n\n"
                  f"{combined_text}"
            )
            doc = Document(
                page_content=text_for_embedding,
                metadata={
                    "source": f"https://github.com/{owner}/{repo}/blob/{branch}/{path}",
                    "file_path": path,
                    "file_name": os.path.basename(path),
                    "repo": f"{owner}/{repo}",
                    "branch": branch
                }
            )
            documents.append(doc)
            print(f"Loaded {path} ({len(content.split())} words)")
            
        except Exception as e:
            print(f"Error loading {path}: {e}")
    
    print(f"\nSuccessfully loaded {len(documents)} files from {repo_url}\n")
    return documents



with st.sidebar.expander("Load GitHub Repo"):
    repo_url = st.text_input("Enter GitHub repo URL ..").strip()
    if st.button("Load Repo"):
        if repo_url:
            st.write("📡 Loading repo directly from GitHub ...")
        try:
            document = load_repo_from_github_url(repo_url, branch="main")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
            doc = text_splitter.split_documents(document)

            st.session_state.vector_db = Chroma.from_documents(doc, embedding)
            st.session_state.retriever = st.session_state.vector_db.as_retriever(search_kwargs={"k": 50})

            llm = main_llm
            prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert software engineer analyzing Python code from a repository.
When the user asks about a specific function, follow this exact response structure:
1. **Show the function  code first**  
   - Find the *complete function* in the provided context.  
   - Print the full function using a fenced code block:
     ```python
     <function code>
     ```
   - Include decorators, docstrings, and helper definitions if directly tied to that function.
2. **Explain the function after showing it**
    - Step through the code logically: parameters → control flow → purpose.  
    - Describe what the function does, why each part exists, and how it interacts with other components.  
    - Mention dependencies or related functions/classes when relevant.
3. **If the question is general** (e.g., architecture, purpose of the repo):
    - Summarize the main logic or structure clearly.
4. **If the function is not found in the provided context**, say exactly:
    `Not found in context.`
Rules:
- Never show the entire file unless the user explicitly asks for it.
- Always wrap any code snippets in proper ```python``` fences.
- Keep your explanations precise and technically accurate.
- Assume the user already understands programming — skip basic syntax explanations."""),
    
    ("human", """Use the provided code context to answer the question below.
<context>
{context}
</context>
Question: {input}""")
])

            document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
            st.session_state.retrieval_chain = create_retrieval_chain(st.session_state.retriever, document_chain)

            st.success(f"✅ Repo loaded successfully with {len(document)} files.")

            # 🔍 Display repo file list
            file_paths = [doc.metadata.get("file_path", doc.metadata.get("source", "Unknown")) for doc in document]

            with st.expander("📁 View Loaded Files", expanded=False):
                for path in sorted(file_paths):
                    st.markdown(f"- `{path}`")
        except Exception as e:
            st.error(f"❌ Error loading repo: {e}")
        else:
          st.warning("⚠️ Please enter a valid GitHub repo URL first.") 
 

st.session_state.query = st.text_input("Ask about the repo:").strip()

if st.button("Get Answer"):
    if not st.session_state.query:
        st.warning("⚠️ Please enter a question.")
    elif "retrieval_chain" not in st.session_state:
        st.warning("⚠️ Please load a repo first.")
    else:
        with st.spinner("🔎 Retrieving..."):
            try:
                response = st.session_state.retrieval_chain.invoke({"input": st.session_state.query})
                st.session_state.answer = response["answer"]
                st.markdown(f"**Answer:** {st.session_state.answer}")
            except Exception as e:
                st.error(f"Error: {e}")

if st.button("Check Answer Quality"):
    if not st.session_state.query:
        st.warning("⚠️ Please ask a question first.")
    else:
        try:
            st.write("Evaluating answer quality...")
            retrieved = st.session_state.retriever.invoke(st.session_state.query)
            contexts = [doc.page_content for doc in retrieved]
            answer = st.session_state.answer

            eval_data = Dataset.from_dict({
                "question": [st.session_state.query],
                "answer": [answer],
                "contexts": [contexts],
                "reference": [answer]
            })

            ragas_llm = LangchainLLMWrapper(main_llm)
            ragas_embeddings = LangchainLLMWrapper(embedding)
            results = evaluate(
                dataset=eval_data,
                llm=ragas_llm,
                embeddings=ragas_embeddings,
                metrics=[faithfulness, context_precision, answer_correctness],
                raise_exceptions=False,
            )

            def get(val):
                return val[0] if isinstance(val, list) else val

            st.write(f"**Faithfulness:** {get(results['faithfulness']):.4f}")
            st.write(f"**Context Precision:** {get(results['context_precision']):.4f}")
            st.write(f"**Answer Correctness:** {get(results['answer_correctness']):.4f}")

        except Exception as e:
            st.error(f"Error: {e}")
        