# Hooplytics Technical Specification

**Version**: 1.0.0  
**Status**: Implemented  
**Last Updated**: 2025-12-03

## 1. System Overview

Hooplytics is a hybrid intelligence NBA assistant that combines three technologies to provide accurate, real-time basketball insights:

1. **Google Gemini 2.5 Flash (LLM)** - Natural language understanding and query orchestration
2. **Model Context Protocol (MCP)** - Standardized tool integration for NBA data access
3. **PyTorch Neural Network** - ML-based player performance tier classification (96.2% accuracy)

**Architecture Pattern**: User → React Frontend → FastAPI Backend → Gemini LLM → MCP Tools (NBA API + ML Classifier)

## 2. Core Components

### 2.1 Backend Architecture (`backend/main.py`)

#### 2.1.1 MCP Session Management
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Connect to MCP Server on startup"""
    global mcp_session, mcp_tools
    
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(mcp_server_path)],
        env=os.environ.copy()
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            mcp_session = session
            result = await session.list_tools()
            mcp_tools = result.tools
            yield
```

**Requirements**:
- MCP session MUST initialize on application startup
- Session MUST persist for application lifetime
- Tools MUST be discovered dynamically via `list_tools()`
- Server path: `{project_root}/mcp-server/nba_server.py`

#### 2.1.2 Chat Endpoint (`/api/chat`)

**Request Schema**:
```json
{
  "messages": [
    {"role": "user", "content": "What are today's NBA games?"}
  ]
}
```

**Response Schema**:
```json
{
  "role": "assistant",
  "content": "Today's NBA games:\n• Lakers vs Warriors\n• Celtics vs Heat",
  "source": "NBA API"
}
```

**Tool Orchestration Loop** (lines 308-360):
1. Send user message to Gemini with `tools=[combined_tool]`
2. Check if response contains `function_call`
3. If yes:
   - Execute MCP tool via `mcp_session.call_tool()`
   - Sanitize response via `sanitize_tool_response()`
   - Send result back to Gemini with `tools=[combined_tool]` (enables continued orchestration)
   - Repeat until text response or max 20 iterations
4. Return final text response with source attribution

**Critical**: `tools=[combined_tool]` MUST be included in EVERY `chat.send_message()` call to allow multi-step tool orchestration.

#### 2.1.3 Gemini System Prompt (lines 120-270)

**Key Instructions**:
- Use `classify_player_tier()` for ANY question about player tiers/performance levels
- NEVER override ML classification with basketball knowledge
- Format responses as clean Markdown tables (no raw JSON or pipe-delimited strings)
- Reject non-basketball questions with standard response

**Safety Settings**:
```python
safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
}
```

### 2.2 JSON Sanitizer (`backend/json_sanitizer.py`)

**Purpose**: Prevent Gemini `finish_reason: 12` blocks by converting structured JSON to natural language.

**Pattern**:
```python
def sanitize_tool_response(tool_name: str, result_text: str) -> str:
    """
    Convert tool response JSON to natural language summary.
    
    ❌ WRONG: {"team": "Lakers", "wins": 12}
    ✅ RIGHT: "The Lakers have 12 wins and 5 losses"
    """
    try:
        result_data = json.loads(result_text)
        
        if tool_name == "get_live_games":
            return _summarize_live_games(result_data)
        elif tool_name == "get_standings":
            return _summarize_standings(result_data)
        # ... other tools
    except json.JSONDecodeError:
        return result_text[:1000]
```

**Summarization Rules**:
- **Live Games**: Numbered list format (Game 1: TeamA vs TeamB), all games shown
- **Standings**: Semicolon-separated list (TeamName (Conf): W-L), limit to 15 teams
- **Player Stats**: Single line summary (Name: PPG, RPG, APG, FG%)
- **Team Roster**: Comma-separated player list, limit to 20 players
- **Aggregate Roster Classifications** [NEW]: Natural language tier summary + grouped player lists
  - Format: "The [Team] have X Elite, Y All-Star, Z Starter, A Rotation, and B Bench players"
  - Includes named player lists for each non-empty tier
  - Rotation/Bench lists limited to 5 players + "and X more" to avoid verbosity
- **Classifier**: Already text format, pass through unchanged

### 2.3 MCP Server (`mcp-server/nba_server.py`)

#### 2.3.1 Available Tools

**1. `get_live_games()`**
- **Source**: `nba_api.live.nba.endpoints.scoreboard`
- **Returns**: List of today's games with scores and status
- **Format**: JSON array of game objects

**2. `get_standings()`**
- **Source**: `nba_api.stats.endpoints.leaguestandings`
- **Returns**: League rankings with wins, losses, win%, conference
- **Format**: JSON array of team objects

**3. `get_player_stats(player_name: str)`**
- **Source**: `nba_api.stats.endpoints.playercareerstats`
- **Returns**: Career statistics for specified player
- **Unicode**: Player name MUST go through NFD normalization before lookup

**4. `get_team_roster(team_name: str)`**
- **Source**: `nba_api.stats.endpoints.commonteamroster`
- **Season**: 2025-26 only
- **Returns**: List of player names on roster

**5. `classify_player_tier(player_name: str)`**
- **Workflow**:
  1. Load PyTorch model from `data/models/player_classifier.pth` (lazy loading)
  2. Retrieve player stats from NBA API (`leaguedashplayerstats`)
  3. Extract 13 features: GP, MIN, PTS, REB, AST, FG%, 3P%, FT%, STL, BLK, TOV, PF, +/-
  4. Normalize using `data/scaler_params.json` (StandardScaler mean/std)
  5. Run inference → get 5-class probabilities (Bench, Rotation, Starter, All-Star, Elite)
  6. Return tier + confidence + probability distribution + key stats

**6. `aggregate_roster_classifications(team_name: str)` [NEW]**
- **Purpose**: Efficient roster-wide tier classification (avoids finish_reason 12 issues)
- **Workflow**:
  1. Get team roster via internal `_get_team_roster_data()` helper
  2. Retrieve current season stats for all players
  3. Classify each player using the same ML model as `classify_player_tier()`
  4. Aggregate results into tier buckets (Elite, All-Star, Starter, Rotation, Bench)
  5. Return structured JSON with tier counts + player lists grouped by tier
- **Sanitization**: Converts to natural language summary (e.g., "The Thunder have 0 Elite, 2 All-Star, 3 Starter, 5 Rotation, and 8 Bench players")
- **Use Cases**: "Give me tier counts for [team]", "How many Elite players does [team] have?"
- **Advantages**: Single tool call instead of 15+ sequential `classify_player_tier()` calls

**7. `get_current_games_with_rosters()`**
- **Composite Tool**: Combines `get_live_games()` + `get_team_roster()` for all teams
- **Purpose**: Reduce tool call count for multi-game queries
- **Warning**: Large response payload may trigger finish_reason 12 if >8 games

#### 2.3.2 Unicode Normalization

```python
import unicodedata

def normalize_name(name: str) -> str:
    """Convert Unicode names to ASCII for matching"""
    nfd = unicodedata.normalize('NFD', name)
    return ''.join(c for c in nfd if unicodedata.category(c) != 'Mn').lower()

# "Nikola Jokić" → "nikola jokic" → exact match
```

**Applied To**:
- All player name lookups in `classify_player_tier()`
- All player name lookups in `get_player_stats()`
- Team roster matching (case-insensitive)

### 2.4 ML Classification Module

#### 2.4.1 Model Architecture (`classification/player_classifier_model.py`)

```python
class PlayerClassifierNN(nn.Module):
    """
    3-layer MLP: 13 → 64 → 32 → 16 → 5
    - Input: 13 features (player statistics)
    - Hidden layers: ReLU + BatchNorm + Dropout(0.3)
    - Output: 5 classes (Bench, Rotation, Starter, All-Star, Elite)
    """
    def __init__(self):
        self.fc1 = nn.Linear(13, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(32, 16)
        self.bn3 = nn.BatchNorm1d(16)
        self.dropout3 = nn.Dropout(0.3)
        
        self.fc4 = nn.Linear(16, 5)
```

**Total Parameters**: 3,813  
**Model Size**: 27KB  
**Device**: CPU-only (no CUDA requirement)

#### 2.4.2 Training Pipeline

**1. Data Collection** (`classification/download_nba_data.py`):
- Seasons: 2021-22, 2022-23, 2023-24, 2024-25, 2025-26
- Source: `nba_api.stats.endpoints.leaguedashplayerstats`
- Output: CSV files in `data/raw/nba_stats_{season}.csv`

**2. Preprocessing** (`classification/data_preprocessing.py`):
- Tier Labeling Algorithm:
  ```python
  composite_score = (
      0.35 * normalized_pts +      # Points (35% weight)
      0.20 * normalized_ast +      # Assists (20%)
      0.15 * normalized_reb +      # Rebounds (15%)
      0.15 * normalized_fg_pct +   # Shooting efficiency (15%)
      0.10 * normalized_plus_minus + # Impact (10%)
      0.05 * normalized_stl_blk    # Defense (5%)
  )
  
  if score >= 0.85: tier = 'Elite'
  elif score >= 0.70: tier = 'All-Star'
  elif score >= 0.50: tier = 'Starter'
  elif score >= 0.30: tier = 'Rotation'
  else: tier = 'Bench'
  ```
- Output: `X_train.npy`, `y_train.npy`, `scaler_params.json`

**3. Training** (`classification/train_classifier.py`):
- **Dataset**: 2,768 player-seasons
- **Train/Val Split**: 80/20 (2,214 / 554 samples)
- **Batch Size**: 32
- **Learning Rate**: 0.001 (Adam optimizer)
- **Early Stopping**: Patience = 15 epochs
- **Training Duration**: 93 epochs
- **Best Validation Accuracy**: 95.25%
- **Test Accuracy**: 96.2%

**4. Outputs**:
- `data/models/player_classifier.pth` (27KB)
- `data/models/training_history.png` (loss/accuracy curves)
- `data/models/confusion_matrix.png` (classification heatmap)

#### 2.4.3 Performance Metrics

| Tier | Precision | Recall | F1-Score | Support |
|------|-----------|--------|----------|---------|
| Elite | 0.94 | 0.96 | 0.95 | 145 |
| All-Star | 0.93 | 0.92 | 0.93 | 203 |
| Starter | 0.97 | 0.96 | 0.96 | 312 |
| Rotation | 0.96 | 0.97 | 0.97 | 428 |
| Bench | 0.98 | 0.97 | 0.98 | 680 |
| **Overall** | **0.962** | **0.962** | **0.962** | **1768** |

### 2.5 Frontend Architecture (`frontend/src/components/ChatInterface.tsx`)

#### 2.5.1 Layout Structure

**Two-Column Design**:
- **Left Column** (AI Responses):
  - Sticky header with Hooplytics branding (lines 133-147)
  - Scrollable chat feed (lines 148-245)
  - Markdown rendering with `react-markdown` + `remark-gfm`
  
- **Right Column** (Controls):
  - Chat input box with 195px top padding (line 280)
  - Conversation history (last 10 queries)
  - Quick action buttons (9 pre-configured queries)

#### 2.5.2 Critical CSS Classes

```tsx
// Sticky header (MUST NOT scroll out of view)
<header className="sticky top-0 p-8 bg-[#0f1014]/95 backdrop-blur-md z-20">

// Chat input alignment (MUST align with header)
<div className="pt-[195px]">

// Dark theme gradient
<div className="bg-gradient-to-br from-blue-500/10 to-purple-600/10">
```

#### 2.5.3 API Integration

```tsx
const endpoint = apiUrl ? `${apiUrl}/api/chat` : './api/chat';

const response = await fetch(endpoint, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    messages: [...messages, userMessage].map(m => ({ 
      role: m.role, 
      content: m.content 
    })),
  }),
});
```

**Relative Path**: Uses `./api/chat` (works in MyBinder proxy + local dev)  
**CORS**: Backend allows `http://localhost:5173` (local) or `*` (MyBinder)

#### 2.5.4 Quick Actions

Pre-configured queries covering all 6 scenarios:
1. "What are today's NBA games?" (API Only)
2. "Show current NBA standings" (API Only)
3. "Who won the 2020 NBA championship?" (Gemini Knowledge)
4. "Are there any upsets in today's games?" (API + Gemini)
5. "#1 Seed Elite" → "How many Elite players does the #1 seed have?" (API + Gemini + Classifier)
6. "Classify LeBron" → "Classify LeBron James" (Classifier Only)
7. "Player Stats" → "Show LeBron James career stats" (API Only)
8. "Team Roster" → "Show Lakers roster" (API Only)
9. "Weather?" → "What's the weather today?" (Guardrails)

## 3. Data Flow

### 3.1 Typical Query Flow (Category 4 Example)

**User**: "How many Elite players does the #1 seed have?"

**Step 1**: Frontend → Backend
```json
POST /api/chat
{"messages": [{"role": "user", "content": "How many Elite players does the #1 seed have?"}]}
```

**Step 2**: Backend → Gemini
```python
response = chat.send_message(
    "How many Elite players does the #1 seed have?",
    tools=[combined_tool]
)
```

**Step 3**: Gemini → Function Call
```python
response.parts[0].function_call = {
    "name": "get_standings",
    "args": {}
}
```

**Step 4**: Backend → MCP Tool
```python
result = await mcp_session.call_tool("get_standings", arguments={})
```

**Step 5**: MCP → NBA API
```python
standings = leaguestandings.LeagueStandings().get_data_frames()[0]
# Returns: DataFrame with team stats
```

**Step 6**: MCP → Backend (JSON)
```json
[
  {"TeamName": "Oklahoma City Thunder", "Conference": "West", "WINS": 20, "LOSSES": 5, ...}
]
```

**Step 7**: Backend → JSON Sanitizer
```python
sanitized = sanitize_tool_response("get_standings", result_text)
# Output: "Standings: Oklahoma City Thunder (West): 20-5; ..."
```

**Step 8**: Backend → Gemini (Natural Language)
```python
response = chat.send_message(
    protos.Content(parts=[protos.Part(
        function_response=protos.FunctionResponse(
            name="get_standings",
            response={"result": sanitized}
        )
    )]),
    tools=[combined_tool]  # Enable continued orchestration
)
```

**Step 9**: Gemini → Function Call (get_team_roster)
```python
response.parts[0].function_call = {
    "name": "get_team_roster",
    "args": {"team_name": "Oklahoma City Thunder"}
}
```

**Step 10-12**: Repeat steps 4-8 for get_team_roster
```
Result: "Oklahoma City Thunder roster (15 players): Jalen Williams, Shai Gilgeous-Alexander, ..."
```

**Step 13**: Gemini → Function Call (classify_player_tier)
```python
response.parts[0].function_call = {
    "name": "classify_player_tier",
    "args": {"player_name": "Jalen Williams"}
}
```

**Step 14**: MCP → ML Model
```python
# 1. Load model (lazy)
model = load_model("data/models/player_classifier.pth")

# 2. Get player stats from NBA API
stats = leaguedashplayerstats.LeagueDashPlayerStats(season='2025-26')

# 3. Extract 13 features
features = [GP, MIN, PTS, REB, AST, FG%, 3P%, FT%, STL, BLK, TOV, PF, +/-]

# 4. Normalize
normalized = (features - scaler_mean) / scaler_std

# 5. Inference
with torch.no_grad():
    logits = model(torch.tensor(normalized))
    probs = F.softmax(logits, dim=1)
    tier_id = torch.argmax(probs).item()

# tier_id = 3 → "All-Star"
```

**Step 15**: MCP → Backend (Text)
```
"Jalen Williams is classified as All-Star tier with 92.3% confidence.
Tier Probabilities: All-Star 92.3%, Starter 6.1%, Elite 1.2%, Rotation 0.3%, Bench 0.1%
Key Stats: 19.1 PPG, 4.0 RPG, 4.3 APG, 48.2% FG"
```

**Step 16**: Backend → Gemini (No sanitization needed, already text)

**Step 17**: Gemini → Final Response (Text)
```
"The #1 seed, the Oklahoma City Thunder, does not have any Elite tier players. 
Their highest-ranked player is Jalen Williams, classified as an All-Star tier 
(92.3% confidence)."
```

**Step 18**: Backend → Frontend
```json
{
  "role": "assistant",
  "content": "The #1 seed, the Oklahoma City Thunder, does not have any Elite tier players...",
  "source": "NBA API + Gemini + ML Classifier"
}
```

**Step 19**: Frontend → User (Markdown Rendering)

### 3.2 Tool Orchestration Patterns

**Pattern 1**: Single Tool (Category 1, 5)
```
User → Gemini → Tool → Result → Gemini → Response
```

**Pattern 2**: Sequential Tools (Category 4)
```
User → Gemini → Tool1 → Result1 → Gemini → Tool2 → Result2 → Gemini → Tool3 → Result3 → Gemini → Response
```

**Pattern 3**: No Tools (Category 2)
```
User → Gemini → Response (from training data)
```

**Pattern 4**: Tool + Knowledge (Category 3)
```
User → Gemini → Tool → Result → Gemini (synthesizes with basketball knowledge) → Response
```

## 4. Deployment Configurations

### 4.1 Local Development

**Backend Requirements**:
- Python 3.13+
- Virtual environment in `Assignment5/backend/venv`
- `.env` file with `GOOGLE_API_KEY=your-key-here`

**Frontend Requirements**:
- Node.js 18+
- npm or yarn
- Vite 6+ for dev server

**Startup Commands**:
```bash
# Terminal 1 - Backend
cd backend
source venv/bin/activate
uvicorn main:app --reload --port 8000

# Terminal 2 - Frontend
cd frontend
npm run dev  # Port 5173
```

**Access**: http://localhost:5173

### 4.2 MyBinder Deployment

**Configuration Files**:
- `binder/environment.yml`: Conda environment (Python 3.11, Node.js 20, pip dependencies)
- `binder/postBuild`: Frontend build script (`npm install && npm run build`)
- `binder/start_mybinder.sh`: Startup script (validates API key, starts uvicorn)

**Critical Requirements**:
- `pip-system-certs>=5.0` in environment.yml (prevents SSL errors)
- API key validation before server start
- Frontend served as static files from `backend/main.py` (routes 85-99)

**Resource Constraints**:
- RAM: ~2GB limit
- CPU: Shared resources (2-3x slower than local)
- Timeout: 10 minutes inactivity
- Build Time: 5-10 minutes first time (cached after)

**Startup Flow**:
1. User clicks Binder badge → environment builds
2. User exports `GOOGLE_API_KEY` in JupyterLab terminal
3. User runs `./binder/start_mybinder.sh`
4. Script validates API key → starts uvicorn on port 8000
5. Backend serves frontend static files from `frontend/dist`
6. User clicks generated proxy URL

## 5. Known Limitations

### 5.1 Gemini finish_reason 12

**Issue**: Large tool response payloads trigger content policy blocks.

**Affected Queries**:
- ❌ "Which game today has the most Elite tier players?" (16+ teams, 200+ classifications)
- ❌ "Compare all teams in today's games by average tier"

**Working Queries**:
- ✅ "How many Elite players does the #1 seed have?" (1 team, ~15 players)
- ✅ "Which team in the top 5 has the most All-Star tier players?" (5 teams max)

**Mitigation**: Scope queries to 1-2 teams maximum.

### 5.2 Max Tool Iterations

**Limit**: 20 sequential tool calls

**Affected Scenarios**: Deep standings analysis requiring 20+ lookups

**Solution**: Backend returns partial results with message when limit hit.

### 5.3 Player Name Matching

**Unicode Edge Cases**:
- Very similar names (Marcus Morris vs Markieff Morris) → may match wrong player
- Retired players (not in 2025-26 roster) → "Player not found"

**Accuracy**: ~98% for active players with NFD normalization

### 5.4 ML Model Constraints

**Training Data**: 2021-2026 seasons only (2,768 player-seasons)

**Edge Cases**:
- Rookies with <10 games: Low confidence predictions
- Injury returns: Misclassification until stats stabilize
- Position-agnostic: Doesn't account for position-specific stat distributions

**Tier Boundary Sensitivity**: Players near thresholds (49% Starter, 48% Rotation) may fluctuate

### 5.5 Real-Time Data Lag

**NBA API Delay**: Live scores update every 30-60 seconds (not truly real-time)

**Roster Updates**: Trades/signings may lag 1-2 days behind official announcements

**Offseason**: Limited data June-September (preseason incomplete)

## 6. Testing Strategy

### 6.1 Category-Based Testing

Every feature change must demonstrate functionality across all 6 categories:

**Category 1**: NBA API Only
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "What are today'\''s NBA games?"}]}'
```

**Category 2**: Gemini Knowledge Only
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Who won the 2020 NBA championship?"}]}'
```

**Category 3**: API + Gemini
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Are there any upsets in today'\''s games?"}]}'
```

**Category 4**: API + Gemini + Classifier
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "How many Elite players does the #1 seed have?"}]}'
```

**Category 5**: Classifier Only
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Classify LeBron James"}]}'
```

**Category 6**: Guardrails
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "What'\''s the weather today?"}]}'
```

### 6.2 Performance Benchmarks

- **Simple Queries** (Category 1-2): <2 seconds
- **ML Classification** (Category 5): <5 seconds
- **Complex Orchestration** (Category 4): <10 seconds
- **Tool Iterations**: Must complete within 20 loops

### 6.3 UI Regression Tests

- Header must remain visible after scrolling (sticky positioning)
- Chat input must align with header (195px top padding)
- Hooplytics branding must always show ("NBA Intelligence Engine" subtitle)
- Markdown tables must render correctly (no pipe-delimited raw text)
- Quick action buttons must trigger correct queries

## 7. Future Enhancements

### 7.1 Multi-Season Classification
Track player trajectory over time by classifying each season separately.

### 7.2 Advanced Metrics Integration
Incorporate PER, Win Shares, VORP for more sophisticated tier labeling.

### 7.3 Playoff Mode
Separate classification for postseason performance (higher variance, smaller sample).

### 7.4 Position-Specific Models
Specialized classifiers for Guards, Forwards, Centers (account for position-specific stat distributions).

### 7.5 Interactive Visualizations
D3.js charts for player comparisons, tier distributions, team analytics.

### 7.6 Multi-League Support
Extend to WNBA, EuroLeague, NCAA (requires separate models and data pipelines).

---

**Specification Status**: This document reflects the IMPLEMENTED state of Hooplytics as of 2025-12-03. All components described are functional and deployed.
