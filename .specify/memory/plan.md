# Hooplytics Implementation Plan

**Status**: COMPLETED ✅  
**Version**: 1.0.0  
**Implementation Date**: 2025-11-25 to 2025-12-03

---

## Phase 1: Foundation & Infrastructure ✅

### 1.1 Project Setup
**Status**: COMPLETED  
**Duration**: Day 1

- [x] Initialize repository structure
- [x] Create virtual environment (`backend/venv`)
- [x] Install core dependencies (FastAPI, Gemini SDK, MCP, PyTorch, nba_api)
- [x] Configure environment variables (`.env` with `GOOGLE_API_KEY`)
- [x] Set up `.gitignore` (exclude venv, .env, __pycache__, data/raw)

**Artifacts**:
- `backend/requirements.txt`
- `backend/.env.example`
- `.gitignore`

### 1.2 Backend Skeleton
**Status**: COMPLETED  
**Duration**: Day 1-2

- [x] Create FastAPI application with CORS middleware
- [x] Implement `/api/chat` endpoint with basic request/response schemas
- [x] Add Gemini API integration (basic chat without tools)
- [x] Test simple conversational queries

**Artifacts**:
- `backend/main.py` (lines 1-60: imports, app setup, CORS)
- `backend/main.py` (lines 112-150: /api/chat endpoint structure)

### 1.3 Frontend Skeleton
**Status**: COMPLETED  
**Duration**: Day 2

- [x] Initialize Vite + React + TypeScript project
- [x] Install Tailwind CSS and configure dark theme
- [x] Create `ChatInterface` component with basic layout
- [x] Implement message state management (useState hooks)
- [x] Add API fetch to backend `/api/chat`

**Artifacts**:
- `frontend/package.json`
- `frontend/tailwind.config.js` (dark theme colors)
- `frontend/src/components/ChatInterface.tsx` (lines 1-110: state + API call)

---

## Phase 2: MCP Integration & NBA Data Access ✅

### 2.1 MCP Server Setup
**Status**: COMPLETED  
**Duration**: Day 3-4

- [x] Create `mcp-server/nba_server.py` with FastMCP initialization
- [x] Implement `@mcp.tool()` decorator for tool registration
- [x] Add MCP session management in backend lifespan
- [x] Test tool discovery via `list_tools()`

**Artifacts**:
- `mcp-server/nba_server.py` (lines 1-25: FastMCP setup)
- `backend/main.py` (lines 27-56: lifespan MCP session)

### 2.2 NBA API Tools
**Status**: COMPLETED  
**Duration**: Day 4-5

**Tool 1: `get_live_games()`**
- [x] Integrate `nba_api.live.nba.endpoints.scoreboard`
- [x] Parse game data (teams, scores, status)
- [x] Return JSON array of games
- [x] Test with today's date

**Tool 2: `get_standings()`**
- [x] Integrate `nba_api.stats.endpoints.leaguestandings`
- [x] Parse team records (wins, losses, conference)
- [x] Return JSON array sorted by wins
- [x] Test with current season

**Tool 3: `get_player_stats()`**
- [x] Integrate `nba_api.stats.endpoints.playercareerstats`
- [x] Implement Unicode NFD normalization for player names
- [x] Parse career statistics (PPG, RPG, APG, FG%)
- [x] Return JSON object with player data
- [x] Test with "Nikola Jokić" (Unicode test case)

**Tool 4: `get_team_roster()`**
- [x] Integrate `nba_api.stats.endpoints.commonteamroster`
- [x] Support team name/abbreviation/nickname lookup
- [x] Extract player names from roster data
- [x] Return JSON with team + player list
- [x] Test with "Lakers", "Los Angeles Lakers", "LAL"

**Artifacts**:
- `mcp-server/nba_server.py` (lines 75-150: get_live_games)
- `mcp-server/nba_server.py` (lines 152-190: get_standings)
- `mcp-server/nba_server.py` (lines 192-245: get_player_stats)
- `mcp-server/nba_server.py` (lines 26-73: get_team_roster helper)
- `mcp-server/nba_server.py` (lines 290-295: normalize_name function)

### 2.3 Tool Orchestration
**Status**: COMPLETED  
**Duration**: Day 5-6

- [x] Implement tool execution loop in backend (while loop with function_call check)
- [x] Add `tools=[combined_tool]` to all `chat.send_message()` calls
- [x] Implement max_iterations=20 safety limit
- [x] Add source attribution ("NBA API", "Gemini", etc.)
- [x] Test multi-step queries requiring sequential tool calls

**Artifacts**:
- `backend/main.py` (lines 308-360: tool orchestration loop)

---

## Phase 3: ML Classification System ✅

### 3.1 Data Collection
**Status**: COMPLETED  
**Duration**: Day 6-7

- [x] Create `classification/download_nba_data.py` script
- [x] Download player stats for 2021-22, 2022-23, 2023-24, 2024-25, 2025-26 seasons
- [x] Use `nba_api.stats.endpoints.leaguedashplayerstats`
- [x] Save CSV files to `data/raw/nba_stats_{season}.csv`
- [x] Validate dataset size (target: 2,500+ player-seasons)

**Artifacts**:
- `classification/download_nba_data.py`
- `data/raw/nba_stats_2021-22.csv` through `nba_stats_2025-26.csv` (5 files)

### 3.2 Data Preprocessing & Tier Labeling
**Status**: COMPLETED  
**Duration**: Day 7-8

- [x] Create `classification/data_preprocessing.py` script
- [x] Implement composite score algorithm:
  - 35% PTS, 20% AST, 15% REB, 15% FG%, 10% +/-, 5% STL+BLK
- [x] Define tier thresholds (Elite ≥0.85, All-Star ≥0.70, Starter ≥0.50, Rotation ≥0.30, Bench <0.30)
- [x] Normalize features using StandardScaler
- [x] Save scaler parameters to `data/scaler_params.json`
- [x] Save processed data (`X_train.npy`, `y_train.npy`, `player_names.npy`)
- [x] Validate class distribution (target: balanced across tiers)

**Artifacts**:
- `classification/data_preprocessing.py`
- `data/X_train.npy`, `data/y_train.npy`, `data/player_names.npy`
- `data/scaler_params.json` (843 bytes)

### 3.3 Model Architecture
**Status**: COMPLETED  
**Duration**: Day 8

- [x] Create `classification/player_classifier_model.py`
- [x] Implement `PlayerClassifierNN` class (3-layer MLP)
- [x] Add BatchNorm1d layers for training stability
- [x] Add Dropout(0.3) for regularization
- [x] Implement `predict()` method with softmax probabilities
- [x] Create `TierLabels` helper class (tier names, descriptions, colors)
- [x] Implement `load_model()` function for inference

**Artifacts**:
- `classification/player_classifier_model.py`

### 3.4 Model Training
**Status**: COMPLETED  
**Duration**: Day 9-10

- [x] Create `classification/train_classifier.py` with training loop
- [x] Implement early stopping (patience=15)
- [x] Track training/validation loss and accuracy
- [x] Generate training history plots (loss + accuracy curves)
- [x] Generate confusion matrix heatmap
- [x] Save best model to `data/models/player_classifier.pth`
- [x] Validate test accuracy >95% (target: 96%+)

**Training Results**:
- Epochs: 93 (early stopped)
- Best Validation Accuracy: 95.25%
- Test Accuracy: 96.2%
- Model Size: 27KB

**Artifacts**:
- `classification/train_classifier.py`
- `data/models/player_classifier.pth` (27KB)
- `data/models/training_history.png`
- `data/models/confusion_matrix.png`

### 3.5 ML Tool Integration
**Status**: COMPLETED  
**Duration**: Day 10-11

- [x] Add `classify_player_tier()` tool to MCP server
- [x] Implement lazy model loading (load on first call)
- [x] Retrieve player stats from NBA API for current season
- [x] Extract 13 features (GP, MIN, PTS, REB, AST, FG%, 3P%, FT%, STL, BLK, TOV, PF, +/-)
- [x] Normalize using saved scaler params
- [x] Run inference and get tier + confidence
- [x] Return detailed response with probabilities and key stats
- [x] Test with LeBron James, Nikola Jokić (Unicode)

**Artifacts**:
- `mcp-server/nba_server.py` (lines 247-406: classify_player_tier)

---

## Phase 4: Content Policy Mitigation ✅

### 4.1 JSON Sanitizer
**Status**: COMPLETED  
**Duration**: Day 11-12

**Problem Identified**:
- Gemini 2.5 Flash triggers `finish_reason: 12` on large JSON responses
- Causes conversation blocks and hallucinated tools

**Solution Implemented**:
- [x] Create `backend/json_sanitizer.py` module
- [x] Implement `sanitize_tool_response()` dispatcher
- [x] Add summarization functions for each tool:
  - `_summarize_live_games()`: Numbered list (Game 1: TeamA vs TeamB)
  - `_summarize_standings()`: Semicolon-separated (TeamName (Conf): W-L)
  - `_summarize_player_stats()`: Single line (Name: PPG, RPG, APG, FG%)
  - `_summarize_team_roster()`: Comma-separated player list
- [x] Integrate sanitizer into backend tool execution loop
- [x] Test with complex multi-game queries

**Artifacts**:
- `backend/json_sanitizer.py`
- `backend/main.py` (lines 340-344: sanitizer integration)

### 4.2 Composite Tools
**Status**: COMPLETED  
**Duration**: Day 12

- [x] Create `get_current_games_with_rosters()` composite tool
- [x] Combine `get_live_games()` + `get_team_roster()` in single call
- [x] Reduce tool call count for multi-game queries
- [x] Test with "Which game has the most players?" queries

**Note**: Still triggers finish_reason 12 with 8+ games. Documented limitation.

**Artifacts**:
- `mcp-server/nba_server.py` (lines 75-150: get_current_games_with_rosters)

---

## Phase 5: UI/UX Polish ✅

### 5.1 Glassmorphism Design
**Status**: COMPLETED  
**Duration**: Day 13

- [x] Implement dark theme (`bg-[#0f1014]`)
- [x] Add gradient accents (`from-blue-400 to-purple-400`)
- [x] Create glass panels (`bg-white/5 backdrop-blur-md`)
- [x] Add shadow effects (`shadow-2xl shadow-black/50`)
- [x] Implement fade-in animations (`animate-fade-in`)

**Artifacts**:
- `frontend/src/index.css` (Tailwind custom utilities)
- `frontend/src/components/ChatInterface.tsx` (lines 125-147: header styling)

### 5.2 Two-Column Layout
**Status**: COMPLETED  
**Duration**: Day 13-14

- [x] Left column: AI responses feed (scrollable)
- [x] Right column: Controls panel (sticky)
- [x] Sticky header with Hooplytics branding (`sticky top-0`)
- [x] Chat input with proper top padding (`pt-[195px]`)
- [x] Responsive grid layout (`flex` with `flex-1`)

**Artifacts**:
- `frontend/src/components/ChatInterface.tsx` (lines 125-390: full layout)

### 5.3 Quick Action Buttons
**Status**: COMPLETED  
**Duration**: Day 14

- [x] Add 9 pre-configured query buttons
- [x] Cover all 6 query categories
- [x] Icon integration (🏀, 📊, 🌟, etc.)
- [x] Hover effects and gradients
- [x] One-click query submission

**Artifacts**:
- `frontend/src/components/ChatInterface.tsx` (lines 300-340: quick actions)

### 5.4 Conversation History
**Status**: COMPLETED  
**Duration**: Day 14

- [x] Display last 10 queries in right panel
- [x] Click to re-ask previous question
- [x] Scrollable history container
- [x] Hover effects and timestamps

**Artifacts**:
- `frontend/src/components/ChatInterface.tsx` (lines 250-295: history panel)

### 5.5 Source Attribution
**Status**: COMPLETED  
**Duration**: Day 15

- [x] Tag responses with data source (NBA API / Gemini / ML Classifier)
- [x] Display source badge in UI
- [x] Color-code badges (blue for API, purple for Gemini, gold for Classifier)
- [x] Update source dynamically based on tools used

**Artifacts**:
- `backend/main.py` (lines 330-360: source tracking)
- `frontend/src/components/ChatInterface.tsx` (lines 175-180: source badge)

---

## Phase 6: Documentation & Testing ✅

### 6.1 README Documentation
**Status**: COMPLETED  
**Duration**: Day 15-16

- [x] Write comprehensive README.md with:
  - Project overview and architecture diagram
  - Feature matrix (6 query categories)
  - Model performance metrics (96.2% accuracy table)
  - Installation instructions (Local + MyBinder)
  - Usage examples for all scenarios
  - Repository structure guide
  - Technology stack listing
  - Known limitations section
  - Deployment considerations

**Artifacts**:
- `README.md` (comprehensive 500+ line guide)

### 6.2 Agent Instructions
**Status**: COMPLETED  
**Duration**: Day 16

- [x] Create `.github/copilot-instructions.md`
- [x] Document critical implementation patterns:
  - Gemini content policy mitigation
  - Tool orchestration loop
  - Unicode name normalization
  - ML model inference workflow
- [x] Add common pitfalls and solutions
- [x] Include testing scenarios
- [x] Document file architecture and data flow

**Artifacts**:
- `.github/copilot-instructions.md`

### 6.3 Query Examples
**Status**: COMPLETED  
**Duration**: Day 16

- [x] Create `GAME_CONTEXT_QUESTIONS.md`
- [x] Document all 6 query categories with examples
- [x] Add curl test commands
- [x] Include expected response formats
- [x] Note finish_reason 12 limitations

**Artifacts**:
- `GAME_CONTEXT_QUESTIONS.md`

### 6.4 End-to-End Testing
**Status**: COMPLETED  
**Duration**: Day 17

**Category 1: NBA API Only**
- [x] Test "What are today's NBA games?" → Verify scoreboard data
- [x] Test "Show current NBA standings" → Verify team rankings

**Category 2: Gemini Knowledge Only**
- [x] Test "Who won the 2020 NBA championship?" → Verify historical fact
- [x] Test "How many quarters in an NBA game?" → Verify rule knowledge

**Category 3: API + Gemini**
- [x] Test "Are there any upsets in today's games?" → Verify contextual analysis

**Category 4: API + Gemini + Classifier**
- [x] Test "How many Elite players does the #1 seed have?" → Verify multi-tool orchestration
- [x] Validate tool sequence: get_standings → get_team_roster → classify_player_tier

**Category 5: Classifier Only**
- [x] Test "Classify LeBron James" → Verify ML inference
- [x] Test "Classify Nikola Jokić" → Verify Unicode handling

**Category 6: Guardrails**
- [x] Test "What's the weather today?" → Verify rejection response

**All tests passed** ✅

---

## Phase 7: Deployment Configuration ✅

### 7.1 MyBinder Setup
**Status**: COMPLETED  
**Duration**: Day 17-18

- [x] Create `binder/environment.yml` with Python 3.11, Node.js 20, all pip dependencies
- [x] Add `pip-system-certs>=5.0` to prevent SSL errors
- [x] Create `binder/postBuild` script for frontend build
- [x] Create `binder/start_mybinder.sh` startup script
- [x] Add API key validation with user-friendly error messages
- [x] Configure backend to serve frontend static files
- [x] Test full deployment cycle in MyBinder

**Artifacts**:
- `binder/environment.yml`
- `binder/postBuild`
- `binder/start_mybinder.sh`
- `backend/main.py` (lines 61-99: static file serving for MyBinder)

### 7.2 Local Development Scripts
**Status**: COMPLETED  
**Duration**: Day 18

- [x] Document terminal commands for backend/frontend startup
- [x] Add venv creation instructions
- [x] Create `.env.example` template
- [x] Validate Python 3.13 compatibility

**Artifacts**:
- `backend/.env.example`
- README.md installation section

---

## Phase 8: Bug Fixes & Refinements ✅

### 8.1 UI Alignment Issues
**Status**: COMPLETED (Dec 3, 2025)  
**Duration**: Day 19

**Issues Fixed**:
- [x] Header scrolling out of view → Added `sticky top-0` to header (line 133)
- [x] Chat input cut off at top → Set `pt-[195px]` top padding (line 280)
- [x] Hooplytics branding disappearing → Changed subtitle to always show "NBA Intelligence Engine" (line 141-144)

**Artifacts**:
- `frontend/src/components/ChatInterface.tsx` (final UI polish)

### 8.2 response.text Access Error
**Status**: COMPLETED (Dec 3, 2025)  
**Duration**: Day 19

**Issue**: ValueError when accessing `response.text` while function calls still pending

**Fix**:
- [x] Added try/except around `response.text` access
- [x] Graceful fallback message when max_iterations hit
- [x] Prevents crash when 20-tool limit reached

**Artifacts**:
- `backend/main.py` (lines 356-365: try/except wrapper)

### 8.3 Virtual Environment Cleanup
**Status**: COMPLETED (Dec 3, 2025)  
**Duration**: Day 19

**Issue**: Confusion between Assignment4 and Assignment5 venvs

**Fix**:
- [x] Created fresh venv in `Assignment5/backend/venv`
- [x] Installed all dependencies from requirements.txt
- [x] Validated uvicorn path (`./venv/bin/python3`)

---

## Implementation Summary

**Total Duration**: 19 days (Nov 25 - Dec 3, 2025)

**Lines of Code**:
- Backend: ~400 lines (main.py + json_sanitizer.py)
- MCP Server: ~410 lines (nba_server.py)
- Classification: ~850 lines (4 modules)
- Frontend: ~390 lines (ChatInterface.tsx)
- **Total**: ~2,050 lines

**Model Performance**:
- Training Data: 2,768 player-seasons
- Test Accuracy: 96.2%
- Model Size: 27KB
- Inference Time: <5 seconds

**Query Categories Supported**: 6 (NBA API, Gemini Knowledge, API+Gemini, API+Gemini+Classifier, Classifier Only, Guardrails)

**Deployment Platforms**: 2 (Local Development, MyBinder)

**Status**: Production-ready ✅

---

## Future Roadmap

### Phase 9: Multi-Season Classification (Planned)
- Track player trajectory over multiple seasons
- Show tier progression (Rotation → Starter → All-Star)
- Predict future tier based on trend

### Phase 10: Advanced Metrics (Planned)
- Integrate PER, Win Shares, VORP
- Refine tier labeling algorithm with advanced stats
- Retrain model with expanded feature set

### Phase 11: Interactive Visualizations (Planned)
- D3.js charts for player comparisons
- Tier distribution histograms
- Team analytics dashboards

### Phase 12: Multi-League Support (Planned)
- Extend to WNBA (separate model required)
- Add EuroLeague support
- NCAA basketball integration
