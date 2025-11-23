# Codebase Analyzer 🤖

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python)](https://python.org)
[![Gemini](https://img.shields.io/badge/Gemini_2.0_Pro-34A853?style=for-the-badge&logo=google&logoColor=white&labelColor=001d35)](https://deepmind.google/technologies/gemini/)
[![Hugging Face](https://img.shields.io/badge/🤗_Hugging_Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Live_Demo-ff4b4b?style=for-the-badge&logo=streamlit)](https://your-streamlit-app-link.streamlit.app)

**Ask anything about any public GitHub repository — instantly explained with real code.**

A powerful **AI-powered code assistant** built with **Streamlit + LangChain + Google Gemini** that lets you load any public GitHub repo (without cloning!) and ask questions like:

- "Explain the `create_retrieval_chain` function"
- "Show me how authentication works"
- "What does this repo actually do?"
- "Find and explain the main FastAPI router"

…and get **accurate, code-first, expert-level answers** with full function definitions automatically included.

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live_Demo-ff4b4b?style=for-the-badge&logo=streamlit)](https://your-streamlit-app-link.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

## Features

- **No git clone required** — loads directly from GitHub API
- Supports `.py`, `.js`, `.ts`, `.md`, `.json`, `.html`, `.yaml`, and more
- Smart chunking with **2000-token chunks + 500 overlap**
- Powered by **Google Gemini 2.0 Pro** (or Groq Llama 3.1 70B)
- **RAGAS evaluation** — check Faithfulness, Context Precision & Answer Correctness
- Beautiful chat interface with full conversation history
- Shows **exact function code first**, then explains it step-by-step
- Metadata-rich answers (file path, repo, branch, source link)

## Live Demo

Try it now → https://huggingface.co/spaces/jagadishwar/Codebase
## Screenshots

<img width="1892" height="847" alt="Screenshot 2025-11-23 172754" src="https://github.com/user-attachments/assets/8e08dd63-05f3-4077-83c7-a25aefd21e54" />


## Quick Start

### 1. Clone & Enter Directory
```bash
git clone https://github.com/yourusername/codebase-analyzer.git
cd codebase-analyzer
