# Web Search Agent — Agno chat app

This agent uses the **Agno** framework plus a Groq or Mistral model. It does not use a database, local memory folder, or PDFs. It supports general live search through DDGS and in-depth research through Agno's `TavilyTools`.

## Create and activate the project environment

Run these commands once in the Visual Studio terminal. The virtual environment prevents conflicts with globally installed Python packages.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## Save your API key once

Copy `.env.example` and rename the copy to `.env`. Open `.env` and paste your real `GROQ_API_KEY`. The app loads this file automatically, so you do not need to enter the key in the terminal again. Keep `.env` private; it is already excluded by `.gitignore`.

For Tavily research, also add `TAVILY_API_KEY=your_key` to `.env`. Without this key, the app continues to use the general DDGS web search.

## If you get a `proxies` error

Your Visual Studio interpreter is using an old global Groq package. In the project terminal, run this exact command once to install fresh packages into the project environment:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade --force-reinstall -r requirements.txt
```

Then select `.venv\Scripts\python.exe` as Visual Studio's Python interpreter and start the app with `.\.venv\Scripts\python.exe server.py`.

## Run with Groq

```powershell
python -m pip install -r requirements.txt
python server.py
```

## Run with Mistral

```powershell
$env:MODEL_PROVIDER = "mistral"
$env:MISTRAL_API_KEY = "your_mistral_key"
python -m pip install -r requirements.txt
python server.py
```

Open `http://localhost:8000` after starting the server. API keys remain only in your terminal/server; never add them to frontend files.

## Conversation history

Each browser chat session is saved in one Markdown file in `conversation_history/`. The same file is updated after every response and contains the complete conversation.
