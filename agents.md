# Hooplytics — AI Agent Guide

## Project Summary

Hooplytics is an AI-powered NBA assistant that combines **Google Gemini 2.5 Flash** (LLM), **Model Context Protocol** (MCP) for real-time NBA data, and a **PyTorch neural network** (96.2% accuracy) for player performance tier classification. Users ask basketball questions via a React chat UI, and the system orchestrates LLM reasoning, live NBA stats, and ML inference to produce answers.

## Architecture

```
User → React Frontend (Vite + TypeScript + Tailwind)
     → FastAPI Backend (Gemini LLM orchestration)
     → MCP Server (NBA API tools + ML classifier)
```

### Key Components

| Component | Location | Purpose |
|-----------|----------|---------|
| Backend API | `backend/main.py` | FastAPI server, Gemini chat loop, MCP client |
| JSON Sanitizer | `backend/json_sanitizer.py` | Converts JSON tool responses to natural language (prevents Gemini content policy blocks) |
| MCP Server | `mcp-server/nba_server.py` | 6 MCP tools: live games, standings, player stats, rosters, ML classification, roster aggregation |
| ML Model | `classification/player_classifier_model.py` | 3-layer MLP (13→64→32→16→5) with BatchNorm + Dropout |
| Training Pipeline | `classification/train_classifier.py` | PyTorch training with early stopping and visualization |
| Data Preprocessing | `classification/data_preprocessing.py` | Feature engineering + 5-tier labeling from composite score |
| Frontend UI | `frontend/src/components/ChatInterface.tsx` | Two-column glassmorphic chat interface with markdown rendering |
| Trained Model | `data/models/player_classifier.pth` | Serialized PyTorch weights (~27KB) |
| Scaler Params | `data/models/scaler_params.json` | StandardScaler mean/std for feature normalization |

## Tech Stack

- **Backend:** Python 3.13, FastAPI, google-generativeai 0.8.3, MCP 1.1.2
- **Frontend:** React 19, TypeScript 5.9, Vite 7, Tailwind CSS 4
- **ML:** PyTorch 2.9, scikit-learn, pandas, numpy
- **Data:** nba_api 1.5.2 (NBA stats), 5 seasons of training data (2021-2026)

## Development Setup

```bash
# Backend (requires GOOGLE_API_KEY in backend/.env)
cd backend && pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Frontend
cd frontend && npm install
npm run dev    # http://localhost:5173

# ML training pipeline
cd classification && bash setup_classification.sh
```

## Build & Lint Commands

```bash
# Frontend build (TypeScript compile + Vite bundle)
cd frontend && npm run build

# Frontend lint
cd frontend && npm run lint
```

There are no automated test suites. Testing is manual via API calls and the chat UI.

## Critical Patterns

### 1. Gemini Content Policy Mitigation

ALL MCP tool responses must pass through `sanitize_tool_response()` in `backend/json_sanitizer.py` before being sent to Gemini. Raw JSON triggers `finish_reason: 12` blocks. The sanitizer converts JSON to natural language summaries.

### 2. Tool Orchestration Loop

`backend/main.py` runs an iterative loop (max 20 iterations) where Gemini can request multiple tool calls sequentially. The `tools=` parameter must be included in every `chat.send_message()` call. Conversation history is capped at 10 messages.

### 3. Unicode Name Normalization

Player names like Jokić and Dončić require NFD normalization before matching. All name lookups in `mcp-server/nba_server.py` strip diacritics via `unicodedata.normalize('NFD', ...)`.

### 4. ML Inference Flow

`classify_player_tier()` in the MCP server: load model (lazy) → fetch player stats from NBA API → extract 13 features → normalize with saved scaler params → run inference → return tier + confidence + probability distribution.

## 5-Tier Classification System

| Tier | Composite Score | Description |
|------|----------------|-------------|
| Elite | ≥0.85 | MVP candidates, superstars |
| All-Star | ≥0.70 | Top-tier caliber |
| Starter | ≥0.50 | Quality starters |
| Rotation | ≥0.30 | Regular bench contributors |
| Bench | <0.30 | Limited role players |

Composite score weights: 35% PTS, 20% AST, 15% REB, 15% FG%, 10% +/-, 5% STL+BLK.

## Adding New MCP Tools

1. Add function with `@mcp.tool()` decorator in `mcp-server/nba_server.py`
2. Add sanitization handler in `backend/json_sanitizer.py`
3. Update the Gemini system prompt in `backend/main.py` (lines ~120-270) with tool usage examples
4. Test with multi-step queries through the chat UI

## Common Pitfalls

- **Large payloads cause Gemini blocks:** Use `aggregate_roster_classifications()` instead of classifying players one-by-one for roster-wide queries.
- **Player not found:** Player may be inactive in 2025-26, or name needs full Unicode spelling.
- **UI layout:** Header uses `sticky top-0`; right panel needs `pt-[195px]` top padding to align with left panel.
- **MyBinder SSL:** `pip-system-certs>=5.0` must be in `binder/environment.yml`.

## Google Gemini Credentials — Dos and Don'ts

### Do

- Store `GOOGLE_API_KEY` in `backend/.env` and load it via environment variables at runtime
- Use `backend/.env.example` as the template — it contains the expected variable names without real values
- Rotate your API key immediately if you suspect it has been exposed
- Use a separate API key per environment (local dev, staging, production) when possible
- Restrict your key in the Google Cloud Console to only the Generative Language API
- Set usage quotas and billing alerts in the Google Cloud Console to catch runaway costs

### Don't

- **Never** hardcode `GOOGLE_API_KEY` (or any API key) directly in source code
- **Never** commit `backend/.env` or any file containing a real API key to version control — `.gitignore` already excludes `.env`
- **Never** log, print, or expose the API key in error messages, API responses, or frontend code
- **Never** share a single API key across multiple contributors — each developer should use their own key
- **Never** embed the key in frontend bundles, environment variables accessible to the browser, or client-side config files
- **Never** disable or weaken Gemini safety settings beyond the project's current configuration without team review — `backend/main.py` already sets `HarmBlockThreshold` to a deliberate level

## Code Style

- **Backend:** PEP 8, type hints, try/except with dict returns, `print()` for logging
- **Frontend:** React hooks (no Redux), Tailwind utility classes, relative API path `./api/chat`
- **ML:** CPU-only inference, `model.eval()` + `torch.no_grad()`, saved scaler params (never refit)
