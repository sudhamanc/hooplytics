# Hooplytics AI Agent Instructions

## Global GitHub Copilot Instructions

You are an expert Senior Software Engineer specializing in Python and Java. You prioritize accruacy,usability,maintainability, security, and performance over quick fixes.

### 1. General Code Quality & Structure

- **Separation of Concerns**: Enforce strict separation between logic, data access, and presentation. Never mix these concerns in a single class or function.
- **Function Size**: Keep functions and methods small (ideally under 20 lines). Each function must do exactly one thing. Refactor immediately if a function grows too large.
- **No Monkey Patching**: Never use monkey patching (runtime modification of classes/modules) for logic fixes. Use dependency injection, inheritance, or proper design patterns instead.
- **Modularity**: Break complex tasks into smaller, self-contained modules or classes.
- **DRY Principle**: Don't Repeat Yourself. Extract shared logic into utility functions or base classes.
- **Performance**: Avoid N+1 queries, unoptimized loops, and excessive memory usage.

### 2. Language-Specific Guidelines

**Python**
- **Standards**: Follow PEP 8 strictly.
- **Typing**: Use strict static typing (typing module) for all function arguments and return values.
- **Modern Syntax**: Use the latest stable Python features (e.g., match/case, f-strings, type unions).
- **Libraries**: Prefer standard libraries where possible. If external libraries are needed, use the latest stable versions.
- **Async**: Use asyncio for I/O-bound tasks where appropriate.

**Java**
- **Version**: Target Java 17 or 21 (LTS).
- **Style**: Follow Google Java Style Guide.
- **Streams**: Use Java Streams API for collection processing where readable and performant.
- **Immutability**: Prefer immutable data structures (Records, final keywords).
- **Exceptions**: Use unchecked exceptions for logic errors; checked exceptions only for recoverable I/O.

### 3. Testing

- **Mandatory Tests**: Every code generation request must include corresponding unit tests.
- **Frameworks**:
  - Python: Use pytest.
  - Java: Use JUnit 5 and Mockito.
- **Coverage**: Test happy paths, edge cases, and failure modes.
- **Isolation**: Tests must not depend on external systems (DB, API) unless explicitly asked for integration tests. Mock all external dependencies.

### 4. Security (Non-Negotiable)

- **Vulnerabilities**: Scan generated code for OWASP Top 10 vulnerabilities (Injection, Broken Access Control, etc.).
- **Input Validation**: Validate all inputs at the boundary. Never trust user input.
- **Secrets**: NEVER hardcode API keys, passwords, or secrets. Use environment variables.
- **Dependencies**: Do not suggest deprecated or known-vulnerable libraries.

### 5. Documentation

- **Docstrings**: Include comprehensive docstrings for all public classes and methods.
- **README**: When generating a new project or module, always provide a README.md that includes:
  - Project Title & Description
  - Installation Instructions
  - Usage Examples
  - Testing Instructions (how to run tests)

### 6. Response Format

- **No Fluff**: Do not be chatty. Provide the code and a brief explanation of why architectural decisions were made.
- **Filepaths**: Always include the recommended file path and name for the code generated.

---

## Project Overview
Hooplytics is a hybrid intelligence NBA assistant combining three technologies:
1. **Google Gemini 2.5 Flash** (LLM) - Natural language understanding and orchestration
2. **Model Context Protocol (MCP)** - Standardized tool integration for real-time NBA data
3. **PyTorch Neural Network** - ML-based player performance tier classification (96.2% accuracy)

**Architecture Pattern:** User → React Frontend → FastAPI Backend → Gemini LLM → MCP Tools (NBA API + ML Classifier)

## Critical Implementation Patterns

### 1. Gemini Content Policy Mitigation
**Problem:** Gemini 2.5 Flash triggers `finish_reason: 12` (UNEXPECTED_TOOL_CALL) when processing large JSON responses, causing conversation blocks.

**Solution:** `backend/json_sanitizer.py` converts ALL tool responses from JSON → natural language BEFORE sending to Gemini:
```python
# ❌ WRONG - Gemini sees raw JSON → BLOCKED
{"team": "Lakers", "wins": 12}

# ✅ RIGHT - Gemini sees natural language → SUCCESS
"The Lakers have 12 wins and 5 losses"
```

**Rule:** ALL MCP tool responses MUST pass through `sanitize_tool_response(tool_name, result_text)` before returning to Gemini.

**Recent Improvements:**
- Live games now include scores and status: "Game 1: TeamA 110 vs TeamB 105 (Final)"
- Conversation history limited to last 10 messages to prevent max_iterations errors
- Source attribution tracks ML classifier usage: "NBA API + ML Classifier"

### 2. Tool Orchestration Loop
**Pattern:** Backend handles multi-step tool calls via iterative loop (max 20 iterations):
1. User sends query → Gemini analyzes → determines required tool(s)
2. Backend executes MCP tool → sanitizes response → sends back to Gemini
3. Gemini processes result → may request additional tools OR return final answer
4. Loop continues until Gemini produces text response or hits max iterations

**Location:** `backend/main.py` lines 340-390 (tool execution while loop)

**Key Details:**
- `tools=[combined_tool]` MUST be included in EVERY `chat.send_message()` call to allow continued tool orchestration
- Conversation history limited to last 10 messages (prevents context overflow)
- `classifier_used` flag tracks if ML model was invoked (for source attribution)

### 3. Unicode Name Normalization
**Problem:** NBA player names contain Unicode (Jokić, Dončić, Antetokounmpo) causing exact-match failures.

**Solution:** `mcp-server/nba_server.py` uses NFD normalization:
```python
import unicodedata
def normalize_name(name):
    nfd = unicodedata.normalize('NFD', name)
    return ''.join(c for c in nfd if unicodedata.category(c) != 'Mn').lower()
# "Nikola Jokić" → "nikola jokic" (matches successfully)
```

**Pattern:** ALL player name lookups go through normalization before matching against NBA API rosters.

### 4. ML Model Inference
**Workflow:** Classification happens in `mcp-server/nba_server.py` via `classify_player_tier()`:
1. Load PyTorch model from `data/models/player_classifier.pth` (lazy loading on first call)
2. Retrieve player stats from NBA API (`leaguedashplayerstats` endpoint)
3. Extract 13 features (GP, MIN, PTS, REB, AST, FG%, 3P%, FT%, STL, BLK, TOV, PF, +/-)
4. Normalize using saved `scaler_params.json` (StandardScaler mean/std)
5. Run inference → get 5-class probabilities (Bench, Rotation, Starter, All-Star, Elite)
6. Return tier + confidence + full probability distribution

**Model Architecture:** 3-layer MLP (13→64→32→16→5) with BatchNorm + Dropout(0.3)

## Development Workflows

### Local Development Setup
```bash
# Backend (Terminal 1)
cd backend
python3.13 -m venv venv  # Fresh venv in Assignment5/backend
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Frontend (Terminal 2)
cd frontend
npm install
npm run dev  # Runs on port 5173
```

**Environment:** Requires `backend/.env` with `GOOGLE_API_KEY=your-key-here`

### Model Training Pipeline
```bash
# 1. Download multi-season data (2021-2026)
python classification/download_nba_data.py

# 2. Preprocess and create tier labels
python classification/data_preprocessing.py

# 3. Train neural network (93 epochs with early stopping)
python classification/train_classifier.py
# Outputs: player_classifier.pth, training_history.png, confusion_matrix.png
```

**Tier Labels:** Derived from composite score (35% PTS, 20% AST, 15% REB, 15% FG%, 10% +/-, 5% STL+BLK)
- Elite: ≥0.85, All-Star: ≥0.70, Starter: ≥0.50, Rotation: ≥0.30, Bench: <0.30

### MyBinder Deployment
**Configuration:** `binder/environment.yml` (conda), `binder/postBuild` (frontend build), `binder/start_mybinder.sh` (startup)

**Critical:** `pip-system-certs>=5.0` MUST be in environment.yml to avoid SSL cert errors in MyBinder.

**Startup:** Run `./binder/start_mybinder.sh` which starts both backend + serves frontend static files from `backend/main.py` (routes 85-99 handle static file serving).

## File Architecture & Data Flow

### Backend Entry Points
- `backend/main.py` (lines 27-56): MCP session initialization via lifespan context manager
- `backend/main.py` (lines 112-379): `/api/chat` endpoint - main conversation handler
- `backend/json_sanitizer.py`: Tool response sanitization (prevents Gemini blocks)

### MCP Server Tools
- `mcp-server/nba_server.py`:
  - `get_live_games()`: Today's scores from nba_api.live.nba.endpoints.scoreboard
  - `get_standings()`: League rankings from nba_api.stats.endpoints.leaguestandings
  - `get_player_stats()`: Career stats from nba_api.stats.endpoints.playercareerstats
  - `get_team_roster()`: 2025-26 rosters from nba_api.stats.endpoints.commonteamroster
  - `classify_player_tier()`: ML inference (PyTorch model + NBA stats lookup)
  - `aggregate_roster_classifications()`: **NEW** - Efficient roster-wide tier classification (gets roster + classifies all players + aggregates counts in one call)
  - `get_current_games_with_rosters()`: Composite tool (games + all rosters in one call)

### Classification Module
- `classification/player_classifier_model.py`: PyTorch nn.Module definition (PlayerClassifierNN)
- `classification/train_classifier.py`: Training loop with early stopping, visualization
- `classification/data_preprocessing.py`: Feature engineering + tier labeling logic
- `data/models/`: Contains `player_classifier.pth` (27KB) and `scaler_params.json` (843B)

### Frontend
- `frontend/src/components/ChatInterface.tsx`:
  - Lines 133-147: Sticky header (Hooplytics branding always visible)
  - Lines 280-295: Chat input with `pt-[195px]` top padding
  - Lines 300-340: Quick action buttons (9 pre-configured queries covering all 6 scenarios)
  - Lines 58-110: Message submission handler (API call to `./api/chat`)

## Common Pitfalls & Solutions

### Issue: finish_reason 12 on Multi-Game Queries
**Symptom:** "Which game today has the most Elite players?" fails with UNEXPECTED_TOOL_CALL error.

**Root Cause:** Large response payloads (16+ team rosters + 200+ classifications) trigger Gemini content policy even with sanitization.

**Solution:** Use `aggregate_roster_classifications(team_name)` for roster-wide queries instead of sequential `classify_player_tier()` calls. Works: "How many Elite players does #1 seed have?" Fails: "Classify all players in all 8 games."

**Recent Fix:** Conversation history now limited to 10 messages to prevent hitting max_iterations limit.

### Issue: UI Header Scrolls Out of View
**Symptom:** Hooplytics branding disappears after sending message.

**Solution:** Header has `sticky top-0` class (`ChatInterface.tsx` line 133) to stay fixed during scroll. If missing, add to header element.

### Issue: Chat Input Box Cut Off
**Symptom:** Input box positioned too high, overlapping with content.

**Solution:** Right panel container has `pt-[195px]` top padding (`ChatInterface.tsx` line 280) to align with left panel header height.

### Issue: Player Not Found
**Symptom:** `classify_player_tier()` returns "Player X not found in 2025-26 season data."

**Causes:**
1. Player inactive in 2025-26 (retired, injured, G-League)
2. Unicode name mismatch (use full name: "Nikola Jokić" not "Jokic")
3. Rookie with <10 games played (limited stats)

**Debug:** Check `nba_server.py` normalization logic (lines 290-295).

## Testing Scenarios
Six query categories test different code paths (see `GAME_CONTEXT_QUESTIONS.md`):

1. **NBA API Only:** "What are today's games?" → `get_live_games()` (now includes scores)
2. **Gemini Knowledge:** "Who won 2020 championship?" → No tools, LLM knowledge
3. **API + Gemini:** "Any upsets today?" → `get_live_games()` + `get_standings()` + upset analysis
4. **API + Gemini + Classifier:** "Give me counts by classification for all players in the #1 seeded team" → `get_standings()` → `aggregate_roster_classifications()`
5. **Classifier Only:** "Classify LeBron James" → Direct ML inference
6. **Guardrails:** "What's the weather?" → Rejection (basketball-only policy)

**Test Command (Category 4):**
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Give me counts by classification for all the players in the #1 seeded team"}]}' | jq .
```

## Code Style Conventions

### Backend
- **Error handling:** Try/except with dict returns (`{"error": "message"}`) rather than exceptions
- **Tool responses:** ALWAYS return valid JSON strings (parsed on frontend)
- **Logging:** `print()` statements for tool calls (visible in uvicorn logs)

### Frontend
- **State management:** React useState hooks (no Redux/Context)
- **Styling:** Tailwind utility classes (dark theme: `bg-[#0f1014]`, gradients: `from-blue-400 to-purple-400`)
- **API calls:** Relative path `./api/chat` (works in MyBinder proxy + local dev)

### ML Code
- **Device:** Always `cpu` (no CUDA requirement for deployment)
- **Eval mode:** `model.eval()` + `torch.no_grad()` for all inference
- **Normalization:** Use saved scaler params (never refit on new data)

## When Modifying This Codebase

### Adding New MCP Tools
1. Add tool function to `mcp-server/nba_server.py` with `@mcp.tool()` decorator
2. Add sanitization logic to `backend/json_sanitizer.py` (convert JSON → text)
3. Update Gemini system prompt in `backend/main.py` (lines 120-270) with tool usage examples
4. Test tool orchestration with multi-step queries

### Changing ML Model
1. Modify `classification/player_classifier_model.py` architecture
2. Retrain via `classification/train_classifier.py`
3. Update `data/models/player_classifier.pth` (ensure <50MB for Git)
4. Restart backend to reload model

### UI Layout Changes
1. Header modifications: `ChatInterface.tsx` lines 133-147
2. Chat feed: Lines 148-245 (left column scroll container)
3. Control panel: Lines 247-390 (right column sticky panel)
4. Maintain `sticky top-0` on header and `pt-[195px]` on right panel

## External Dependencies

### Critical APIs
- **Google Gemini API:** Free tier = 60 req/min, requires API key from https://makersuite.google.com/app/apikey
- **NBA Stats API:** No auth required, rate-limited to ~30 req/min, provided by `nba_api` package

### Python Packages (backend/requirements.txt)
- `fastapi` + `uvicorn` - Web framework
- `google-generativeai==0.8.3` - Gemini SDK
- `mcp` - Model Context Protocol client/server
- `torch==2.5.1` - PyTorch (CPU-only)
- `nba_api` - NBA statistics wrapper
- `scikit-learn` - Feature scaling utilities

### npm Packages (frontend/package.json)
- `react@18` + `react-dom@18` - UI framework
- `typescript@5` - Type safety
- `vite@6` - Build tool + dev server
- `tailwindcss@4` - Utility CSS
- `react-markdown` + `remark-gfm` - Markdown rendering
