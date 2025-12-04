# Hooplytics Improvement Roadmap

**Document Version**: 1.0.0  
**Created**: 2025-12-03  
**Status**: STRATEGIC PLANNING

---

## Executive Summary

This roadmap provides **strategic recommendations** for enhancing Hooplytics based on the current implementation (v1.0.0, 96.2% ML accuracy, 2,050 LOC). Improvements are prioritized by **impact** (user value) and **feasibility** (implementation effort).

**Current Strengths**:
- ✅ Solid hybrid intelligence architecture (LLM + MCP + ML)
- ✅ High ML accuracy (96.2% test accuracy)
- ✅ Content policy mitigation working (JSON sanitization)
- ✅ Professional UI with glassmorphism design
- ✅ Multi-platform deployment (local + MyBinder)

**Key Limitations** (opportunities for improvement):
1. **Scalability**: finish_reason 12 blocks on large multi-team queries
2. **ML Coverage**: Limited to 2021-2026 training data, no historical players
3. **Feature Depth**: Only 13 statistical features, missing advanced metrics (PER, VORP, Win Shares)
4. **User Experience**: No data visualizations, saved conversations, or personalization
5. **Domain Scope**: NBA-only, no WNBA/EuroLeague/NCAA support

---

## Priority Matrix

| Priority | Impact | Effort | Timeline |
|----------|--------|--------|----------|
| **P0** - Critical | High | Low-Med | 1-2 weeks |
| **P1** - High | High | Medium | 2-4 weeks |
| **P2** - Medium | Med | Medium | 1 month |
| **P3** - Low | Low-Med | High | 2+ months |

---

## Phase 9: Performance & Scalability (P0)

**Goal**: Eliminate finish_reason 12 blocks and support large-scale queries

### 9.1 Streaming Response Architecture ⭕
**Problem**: Large JSON payloads from multi-team queries trigger Gemini content policy  
**Current**: Entire tool response sent at once → blocks on 16+ teams  
**Solution**: Implement incremental streaming for tool responses

**Implementation**:
1. **Backend Changes** (`backend/main.py`):
   - Add SSE (Server-Sent Events) endpoint: `/api/chat/stream`
   - Process tool results in chunks (max 2 teams per iteration)
   - Stream partial results to frontend as they complete
   - Update Gemini with incremental context, not full payload

2. **Frontend Changes** (`ChatInterface.tsx`):
   - Replace fetch with EventSource for SSE
   - Display progressive results (loading states per team)
   - Show "Processing team 3 of 8..." progress indicator

3. **Sanitizer Updates** (`json_sanitizer.py`):
   - Add `chunk_response(data, chunk_size=2)` function
   - Stream summaries instead of batching

**Acceptance Criteria**:
- ✅ "Which game today has the most Elite players?" works without blocking
- ✅ Response time <15 seconds for 8-game analysis
- ✅ No finish_reason 12 errors on multi-team queries

**Effort**: Medium (2-3 days)  
**Impact**: High (unlocks Category 4 complex queries)

---

### 9.2 Caching Layer for NBA API ⭕
**Problem**: Repeated API calls for same data (rosters, standings) within 24 hours  
**Current**: Every query re-fetches from NBA API  
**Solution**: In-memory cache with TTL (time-to-live)

**Implementation**:
1. **Cache Module** (`backend/nba_cache.py`):
   ```python
   from datetime import datetime, timedelta
   from typing import Optional, Dict, Any
   
   class NBACache:
       def __init__(self):
           self._cache: Dict[str, tuple[Any, datetime]] = {}
           self._ttl = timedelta(hours=6)  # 6-hour TTL
       
       def get(self, key: str) -> Optional[Any]:
           if key in self._cache:
               data, timestamp = self._cache[key]
               if datetime.now() - timestamp < self._ttl:
                   return data
           return None
       
       def set(self, key: str, value: Any):
           self._cache[key] = (value, datetime.now())
   ```

2. **MCP Server Updates** (`mcp-server/nba_server.py`):
   - Wrap each tool with cache check before NBA API call
   - Cache keys: `f"standings:{date}"`, `f"roster:{team_name}:{season}"`, `f"live_games:{date}"`
   - Invalidate cache on date change

3. **Performance Gains**:
   - 80% reduction in NBA API calls for repeat queries
   - <100ms response time for cached data vs 2-3s for API call

**Effort**: Low (1 day)  
**Impact**: Medium (faster responses, reduced API load)

---

### 9.3 Parallel Tool Execution ⭕
**Problem**: Sequential tool calls slow down multi-step orchestration  
**Current**: Tool 1 → wait → Tool 2 → wait → Tool 3 (15+ seconds total)  
**Solution**: Execute independent tools in parallel

**Implementation**:
1. **Dependency Graph Analysis**:
   - Parse Gemini function_call to identify dependencies
   - Example: `get_standings()` and `get_live_games()` are independent → parallel
   - Example: `get_standings()` must complete before `get_team_roster(#1_seed)` → sequential

2. **Backend Updates** (`backend/main.py`):
   ```python
   import asyncio
   
   async def execute_tools_parallel(tool_calls: list):
       tasks = []
       for tool_call in tool_calls:
           tasks.append(mcp_session.call_tool(tool_call.name, tool_call.args))
       results = await asyncio.gather(*tasks)
       return results
   ```

3. **Orchestration Loop**:
   - Detect when Gemini returns multiple function_calls
   - Group by dependencies (topological sort)
   - Execute groups in parallel, wait between dependency levels

**Acceptance Criteria**:
- ✅ "Show standings AND live games" completes in <3 seconds (vs 5 seconds sequential)
- ✅ Complex queries 2-3x faster

**Effort**: Medium (2 days)  
**Impact**: Medium (better UX for multi-tool queries)

---

## Phase 10: Advanced ML Features (P1)

**Goal**: Improve classification accuracy and expand model capabilities

### 10.1 Multi-Season Tier Tracking ⭕
**Problem**: Cannot track player development over time  
**Current**: Single tier classification based on current season  
**Solution**: Track tier progression across 5 seasons (2021-2026)

**Implementation**:
1. **Data Preprocessing** (`classification/data_preprocessing.py`):
   - Group player stats by season
   - Create time-series features: tier_2021, tier_2022, ..., tier_2025
   - Calculate tier delta (improvement/decline)

2. **New Tool**: `get_player_tier_history(player_name: str)` in `nba_server.py`:
   ```python
   @mcp.tool()
   def get_player_tier_history(player_name: str) -> str:
       """
       Returns player tier classification history across 2021-2026 seasons.
       Useful for analyzing career trajectory and development.
       """
       # Load multi-season model
       # Classify for each season
       # Return: {season: {tier, confidence, stats}}
   ```

3. **Frontend Visualization** (`ChatInterface.tsx`):
   - Add tier progression chart (line graph)
   - Color-coded by tier (Bench → Elite gradient)
   - Display when user asks "How has LeBron's performance changed?"

**Example Query**: "Show me Giannis Antetokounmpo's tier progression from 2021 to 2025"

**Acceptance Criteria**:
- ✅ Track tier changes across 5 seasons
- ✅ Identify breakout players (Rotation → All-Star jumps)
- ✅ Visualization renders in <2 seconds

**Effort**: High (4-5 days)  
**Impact**: High (unique insight, storytelling capability)

---

### 10.2 Advanced Statistical Features ⭕
**Problem**: Model uses only basic stats (PPG, RPG, APG, FG%, etc.)  
**Current**: 13 features, missing advanced metrics  
**Solution**: Expand to 25+ features including PER, VORP, Win Shares, True Shooting %

**New Features**:
- **PER** (Player Efficiency Rating): Overall player productivity per minute
- **VORP** (Value Over Replacement Player): Win contribution vs average player
- **Win Shares**: Estimated wins contributed
- **True Shooting %**: Shooting efficiency including 3PT and FT
- **Usage Rate**: % of team plays used while on court
- **Defensive Rating**: Points allowed per 100 possessions
- **Offensive Rating**: Points scored per 100 possessions
- **Box Plus/Minus**: Overall impact on team performance

**Implementation**:
1. **Data Collection** (`classification/download_nba_data.py`):
   - Use `nba_api.stats.endpoints.playerdashboardbyyearoveryear` for advanced stats
   - Calculate custom metrics (TS%, Usage Rate) from raw data

2. **Model Retraining** (`classification/train_classifier.py`):
   - Update architecture: 25 input features → 128 → 64 → 32 → 5
   - Retrain with early stopping
   - Target: 97%+ test accuracy (vs current 96.2%)

3. **Feature Importance Analysis**:
   - Use SHAP (SHapley Additive exPlanations) to explain model decisions
   - Display top 5 features contributing to tier classification

**Example Response**:
```
LeBron James: **Starter Tier** (68.3% confidence)

Top Contributing Features:
1. VORP: 3.2 (91st percentile) → +15% All-Star probability
2. Usage Rate: 28.4% (High) → +12% Starter probability
3. Win Shares: 5.8 (Above avg) → +8% confidence
4. True Shooting: 61.2% (Excellent) → +6% Elite probability
5. Defensive Rating: 109.3 (Good) → +5% confidence
```

**Acceptance Criteria**:
- ✅ Test accuracy improves to ≥97%
- ✅ Model <100MB (still Git-compatible)
- ✅ Tier explanations include top features

**Effort**: High (1 week: data collection + model training + integration)  
**Impact**: High (better accuracy, explainable AI)

---

### 10.3 Injury Impact Prediction ⭕
**Problem**: Cannot predict tier changes due to injuries  
**Solution**: Add injury history as feature, predict recovery tier

**Implementation**:
1. **Injury Data Source**: NBA injury reports (via `nba_api` or web scraping)
2. **Feature Engineering**:
   - Days injured current season
   - Injury type (ankle, knee, back, etc.)
   - Games missed vs games played
   - Pre-injury tier vs post-injury tier

3. **New Model**: `InjuryImpactPredictor` (separate from tier classifier):
   - Input: Player stats + injury history
   - Output: Expected tier after recovery (with uncertainty range)

4. **Use Case**: "If Kawhi Leonard returns from knee injury, what tier will he be?"

**Acceptance Criteria**:
- ✅ Predict tier within ±1 tier 80% of time
- ✅ Integrate with tier history tool

**Effort**: Very High (2 weeks: data collection + model design + validation)  
**Impact**: Medium (niche use case, but unique value)  
**Priority**: P2 (defer until Phase 11+)

---

## Phase 11: Interactive Visualizations (P1)

**Goal**: Transform raw data into visual insights

### 11.1 Player Comparison Radar Charts ⭕
**Problem**: Hard to compare players across multiple dimensions  
**Current**: Text-only stat comparison  
**Solution**: D3.js radar charts for visual comparison

**Implementation**:
1. **Backend API** (`backend/main.py`):
   ```python
   @app.post("/api/visualize/player_comparison")
   async def player_comparison(players: list[str]):
       # Fetch stats for each player
       # Normalize to 0-100 scale
       # Return JSON for radar chart
       return {
           "players": [
               {"name": "LeBron", "stats": {"scoring": 85, "rebounding": 70, ...}},
               {"name": "Giannis", "stats": {"scoring": 90, "rebounding": 95, ...}}
           ]
       }
   ```

2. **Frontend Component** (`ChatInterface.tsx`):
   - Install `recharts` library: `npm install recharts`
   - Create `<RadarChart>` component
   - Trigger on queries like "Compare LeBron vs Giannis"

3. **Gemini Integration**:
   - Update system prompt: "When user asks to compare players, call player_comparison API and display radar chart"
   - Include chart in markdown response: `![Comparison](api/visualize/player_comparison?players=LeBron,Giannis)`

**Example Visual**:
```
         Scoring (85)
              /\
             /  \
    Passing/    \Rebounding
      (90) |     | (70)
           |     |
           -------
          Defense (75)
```

**Acceptance Criteria**:
- ✅ Compare up to 5 players simultaneously
- ✅ Chart renders in <1 second
- ✅ Interactive (hover for exact values)

**Effort**: Medium (3-4 days)  
**Impact**: High (visual storytelling, shareable insights)

---

### 11.2 Tier Distribution Histograms ⭕
**Problem**: No way to see league-wide tier breakdown  
**Solution**: Bar chart showing tier distribution across NBA

**Implementation**:
1. **Backend Aggregation** (`mcp-server/nba_server.py`):
   ```python
   @mcp.tool()
   def get_league_tier_distribution() -> str:
       """
       Returns tier distribution for all active NBA players.
       Useful for understanding league talent distribution.
       """
       # Classify all active players (450+ players)
       # Count by tier
       # Return: {tier: count, percentage}
   ```

2. **Frontend Chart** (`ChatInterface.tsx`):
   - Use `recharts` BarChart
   - Color-coded bars (Bench=gray, Rotation=blue, Starter=green, All-Star=gold, Elite=red)

3. **Caching Strategy**:
   - Cache league-wide classification for 7 days
   - Pre-compute during off-hours (cron job)

**Example Query**: "What's the tier breakdown for the 2025-26 NBA season?"

**Acceptance Criteria**:
- ✅ Classify 450+ players in <30 seconds (with caching)
- ✅ Chart updates when user clicks tier (drill-down to player list)

**Effort**: Medium (3 days)  
**Impact**: Medium (league insights, analyst tool)

---

### 11.3 Live Game Momentum Tracker ⭕
**Problem**: Only shows final scores, no in-game insights  
**Solution**: Real-time score tracking with momentum visualization

**Implementation**:
1. **Live Score Updates** (`mcp-server/nba_server.py`):
   - Poll NBA API every 30 seconds during game time
   - Track score changes, lead changes, largest lead

2. **Frontend Live Feed** (`ChatInterface.tsx`):
   - Add "Live Scores" panel (auto-refresh)
   - Show momentum arrows (🔥 for team on run, ❄️ for cold streak)
   - Display quarter-by-quarter scoring

3. **WebSocket Integration** (optional):
   - Replace polling with WebSocket for real-time updates
   - Backend pushes score changes to connected clients

**Example Display**:
```
Lakers 98 🔥 (+8 run)  |  Warriors 92 ❄️
Q3: 7:34 remaining
Momentum: Lakers (last 5 min: +12)
```

**Acceptance Criteria**:
- ✅ Updates every 30 seconds during live games
- ✅ No refresh button needed
- ✅ Works for all simultaneous games (up to 15)

**Effort**: High (5-6 days with WebSocket)  
**Impact**: Medium (nice-to-have, not core feature)  
**Priority**: P2 (defer until core features complete)

---

## Phase 12: Multi-League Support (P2)

**Goal**: Expand beyond NBA to other basketball leagues

### 12.1 WNBA Integration ⭕
**Why**: 30% of basketball fans follow WNBA  
**Data Source**: `wnba_api` (community library) or web scraping

**Implementation**:
1. **New MCP Server** (`mcp-server/wnba_server.py`):
   - Clone structure from `nba_server.py`
   - Implement same 6 tools (games, standings, stats, roster, classification, composite)

2. **WNBA ML Model**:
   - Train separate model on WNBA data (2019-2025 seasons)
   - Different tier thresholds (WNBA scoring averages lower than NBA)
   - Expected accuracy: 94%+ (smaller league, less data)

3. **Backend Routing** (`backend/main.py`):
   - Detect league from query ("WNBA" keyword)
   - Route to appropriate MCP server
   - Update Gemini system prompt with league context

**Example Queries**:
- "What are today's WNBA games?"
- "Classify A'ja Wilson (WNBA)"
- "Compare NBA vs WNBA tier distributions"

**Acceptance Criteria**:
- ✅ All 6 query categories work for WNBA
- ✅ Cross-league comparisons supported
- ✅ Model accuracy ≥94%

**Effort**: Very High (2 weeks: data collection + model training + integration)  
**Impact**: Medium (expands audience, differentiator)

---

### 12.2 EuroLeague Support ⭕
**Challenge**: No official API, requires web scraping  
**Data Source**: EuroLeague official website

**Implementation**:
1. **Scraper Module** (`mcp-server/euroleague_scraper.py`):
   - BeautifulSoup or Scrapy for game data
   - Respect robots.txt and rate limits
   - Cache aggressively (24-hour TTL)

2. **ML Model**:
   - Train on EuroLeague stats (different play style than NBA)
   - Adjust tier thresholds for international game

**Effort**: Very High (3+ weeks)  
**Impact**: Low-Medium (niche audience)  
**Priority**: P3 (future consideration)

---

## Phase 13: User Experience Enhancements (P1)

**Goal**: Personalization and user engagement

### 13.1 Conversation Memory & Export ⭕
**Problem**: Conversations lost on page refresh  
**Solution**: Save conversation history to browser LocalStorage or backend DB

**Implementation**:
1. **LocalStorage Persistence** (`ChatInterface.tsx`):
   ```typescript
   useEffect(() => {
       const saved = localStorage.getItem('hooplytics_history');
       if (saved) setMessages(JSON.parse(saved));
   }, []);
   
   useEffect(() => {
       localStorage.setItem('hooplytics_history', JSON.stringify(messages));
   }, [messages]);
   ```

2. **Export Feature**:
   - Add "Export Chat" button → downloads JSON or Markdown
   - Include timestamps, sources, query/response pairs

3. **Session Management** (optional):
   - Backend DB (SQLite) to store sessions
   - User login (OAuth) to sync across devices

**Acceptance Criteria**:
- ✅ Conversations persist across page refreshes
- ✅ Export includes all metadata
- ✅ Option to clear history

**Effort**: Low (1 day for LocalStorage, 1 week for backend DB)  
**Impact**: High (improves retention, enables research use case)

---

### 13.2 Favorite Players Dashboard ⭕
**Problem**: Users repeatedly ask about same players  
**Solution**: Pin favorite players for quick access

**Implementation**:
1. **UI Component** (`ChatInterface.tsx`):
   - "⭐ Favorites" section in right panel
   - Add player button (appears after classifying a player)
   - Display: Player name, tier badge, last update timestamp

2. **Quick Actions**:
   - Click player → auto-populate input with "Classify [Player]"
   - Right-click → "Compare with..." (select another favorite)

3. **LocalStorage**:
   - Store favorites as JSON array
   - Max 10 favorites

**Example Display**:
```
⭐ Favorites
• LeBron James [Starter] Updated 2h ago
• Giannis Antetokounmpo [Elite] Updated 1d ago
• Stephen Curry [All-Star] Updated 5h ago
```

**Acceptance Criteria**:
- ✅ Add/remove favorites with one click
- ✅ Quick re-classification for stale data
- ✅ Sync favorites via LocalStorage

**Effort**: Low (2 days)  
**Impact**: Medium (power user feature)

---

### 13.3 Voice Input/Output ⭕
**Problem**: Typing queries is slow, not accessible  
**Solution**: Web Speech API for voice commands

**Implementation**:
1. **Voice Input** (`ChatInterface.tsx`):
   ```typescript
   const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
   const recognition = new SpeechRecognition();
   
   recognition.onresult = (event) => {
       const transcript = event.results[0][0].transcript;
       setInput(transcript);
       handleSubmit();
   };
   ```

2. **Voice Output** (Text-to-Speech):
   ```typescript
   const speakResponse = (text: string) => {
       const utterance = new SpeechSynthesisUtterance(text);
       window.speechSynthesis.speak(utterance);
   };
   ```

3. **UI Controls**:
   - 🎤 Microphone button (toggle listening)
   - 🔊 Speaker button (read last response)

**Acceptance Criteria**:
- ✅ Voice recognition accuracy >85%
- ✅ Works in Chrome, Safari, Edge
- ✅ Fallback to text input if unsupported

**Effort**: Medium (3 days)  
**Impact**: Medium (accessibility, hands-free use)

---

## Phase 14: Production Readiness (P0-P1)

**Goal**: Deploy to production with monitoring and security

### 14.1 Rate Limiting & API Key Management ⭕
**Problem**: No protection against abuse  
**Solution**: Rate limiting per IP/API key

**Implementation**:
1. **Rate Limiter** (`backend/main.py`):
   ```python
   from slowapi import Limiter, _rate_limit_exceeded_handler
   from slowapi.util import get_remote_address
   
   limiter = Limiter(key_func=get_remote_address)
   app.state.limiter = limiter
   
   @app.post("/api/chat")
   @limiter.limit("10/minute")  # 10 requests per minute
   async def chat_endpoint(...):
       ...
   ```

2. **API Key Authentication** (optional):
   - Require user-provided Gemini API key (removes shared key risk)
   - Store in browser LocalStorage (encrypted)
   - Validate on each request

**Acceptance Criteria**:
- ✅ Rate limit: 10 req/min per IP
- ✅ Graceful error message when exceeded
- ✅ API key per user (optional)

**Effort**: Low (1 day)  
**Impact**: High (prevents abuse, protects API quota)

---

### 14.2 Error Tracking & Monitoring ⭕
**Problem**: No visibility into production errors  
**Solution**: Sentry or LogRocket integration

**Implementation**:
1. **Sentry Setup** (backend):
   ```python
   import sentry_sdk
   sentry_sdk.init(dsn="your-sentry-dsn")
   ```

2. **Frontend Sentry**:
   ```typescript
   import * as Sentry from "@sentry/react";
   Sentry.init({ dsn: "your-sentry-dsn" });
   ```

3. **Custom Logging**:
   - Log finish_reason 12 errors with query context
   - Track tool execution times
   - Monitor ML inference latency

**Acceptance Criteria**:
- ✅ All errors captured in Sentry dashboard
- ✅ Performance metrics tracked
- ✅ Alerts for high error rates

**Effort**: Low (1 day)  
**Impact**: High (essential for production)

---

### 14.3 Automated Testing Suite ⭕
**Problem**: No unit or integration tests  
**Solution**: pytest for backend, Jest for frontend

**Implementation**:
1. **Backend Tests** (`backend/tests/`):
   ```python
   # tests/test_json_sanitizer.py
   def test_sanitize_live_games():
       raw_json = '{"games": [{"home": "Lakers", ...}]}'
       result = sanitize_tool_response("get_live_games", raw_json)
       assert "Lakers" in result
       assert "{" not in result  # No JSON in output
   
   # tests/test_ml_inference.py
   def test_classify_player_tier():
       result = classify_player_tier("LeBron James")
       assert result in ["Bench", "Rotation", "Starter", "All-Star", "Elite"]
   ```

2. **Frontend Tests** (`frontend/src/__tests__/`):
   ```typescript
   // ChatInterface.test.tsx
   import { render, screen } from '@testing-library/react';
   
   test('renders header', () => {
       render(<ChatInterface />);
       expect(screen.getByText(/Hooplytics/i)).toBeInTheDocument();
   });
   ```

3. **CI/CD Integration** (GitHub Actions):
   ```yaml
   # .github/workflows/test.yml
   name: Test
   on: [push, pull_request]
   jobs:
     test:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v2
         - run: pip install -r backend/requirements.txt
         - run: pytest backend/tests/
         - run: npm install && npm test
   ```

**Acceptance Criteria**:
- ✅ 80%+ code coverage
- ✅ All 6 query categories tested
- ✅ Tests run on every PR

**Effort**: High (1 week)  
**Impact**: High (prevents regressions, enables safe refactoring)

---

## Implementation Priority Ranking

### Immediate (Next Sprint - 1-2 Weeks)

| Task | Priority | Impact | Effort | Owner |
|------|----------|--------|--------|-------|
| 9.2 Caching Layer | P0 | High | Low | Backend |
| 14.1 Rate Limiting | P0 | High | Low | Backend |
| 13.1 Conversation Memory | P0 | High | Low | Frontend |
| 14.2 Error Tracking | P0 | High | Low | DevOps |

**Rationale**: Quick wins that improve performance, prevent abuse, and enhance UX.

---

### Short-Term (1-2 Months)

| Task | Priority | Impact | Effort | Owner |
|------|----------|--------|--------|-------|
| 9.1 Streaming Responses | P1 | High | Medium | Backend |
| 10.2 Advanced ML Features | P1 | High | High | ML Team |
| 11.1 Radar Charts | P1 | High | Medium | Frontend |
| 14.3 Testing Suite | P1 | High | High | QA |

**Rationale**: Core feature improvements that unlock new query types and visualizations.

---

### Medium-Term (2-4 Months)

| Task | Priority | Impact | Effort | Owner |
|------|----------|--------|--------|-------|
| 10.1 Multi-Season Tracking | P1 | High | High | ML Team |
| 11.2 Tier Histograms | P2 | Medium | Medium | Frontend |
| 13.2 Favorites Dashboard | P2 | Medium | Low | Frontend |
| 12.1 WNBA Support | P2 | Medium | Very High | Full Stack |

**Rationale**: Expand capabilities and audience, but not critical for MVP+1.

---

### Long-Term (4+ Months)

| Task | Priority | Impact | Effort | Owner |
|------|----------|--------|--------|-------|
| 9.3 Parallel Tool Execution | P2 | Medium | Medium | Backend |
| 10.3 Injury Prediction | P2 | Medium | Very High | ML Team |
| 11.3 Live Momentum | P2 | Medium | High | Full Stack |
| 13.3 Voice I/O | P3 | Medium | Medium | Frontend |
| 12.2 EuroLeague | P3 | Low | Very High | Full Stack |

**Rationale**: Nice-to-haves that enhance experience but not essential.

---

## Risk Assessment

### Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Gemini API rate limits hit with streaming | Medium | High | Implement exponential backoff, queue requests |
| Advanced stats API unstable | Medium | Medium | Fallback to basic stats, cache aggressively |
| WNBA API unavailable | High | Medium | Web scraping fallback, manual data collection |
| Model retraining degrades accuracy | Low | High | A/B test new model before deployment, keep v1.0 fallback |

### Business Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| User adoption low for visualizations | Medium | Low | User testing before full implementation |
| WNBA users expect different UX | Medium | Medium | Separate UI themes per league |
| Cost of advanced APIs prohibitive | Low | High | Start with free tiers, upgrade based on ROI |

---

## Success Metrics

### Phase 9 (Performance)
- **Goal**: 90% reduction in finish_reason 12 errors
- **KPIs**:
  - Average response time <5 seconds (vs current 10s)
  - Cache hit rate >70%
  - Support 8-game queries without errors

### Phase 10 (ML)
- **Goal**: 97%+ test accuracy, explainable tiers
- **KPIs**:
  - Test accuracy ≥97% (vs current 96.2%)
  - Top-5 feature importance displayed for every classification
  - Multi-season tracking for 450+ active players

### Phase 11 (Visualizations)
- **Goal**: 50% of queries result in visual output
- **KPIs**:
  - 5+ chart types available (radar, bar, line, heatmap, scatter)
  - Chart render time <1 second
  - User engagement +30% (time on page, queries per session)

### Phase 12 (Multi-League)
- **Goal**: 20% of queries are WNBA-related
- **KPIs**:
  - WNBA model accuracy ≥94%
  - Cross-league comparisons supported
  - Zero cross-contamination (NBA queries don't return WNBA data)

### Phase 13 (UX)
- **Goal**: 80% user retention (return within 7 days)
- **KPIs**:
  - Average session duration >5 minutes
  - 3+ queries per session
  - Conversation export used by 15% of users

### Phase 14 (Production)
- **Goal**: Zero critical errors in production
- **KPIs**:
  - 99.9% uptime
  - Error rate <0.1%
  - Test coverage ≥80%

---

## Appendix: Alternative Approaches Considered

### A. Gemini Pro 1.5 Instead of 2.5 Flash
**Pros**: Larger context window (1M tokens vs 32K)  
**Cons**: Higher latency (2-3x slower), more expensive  
**Decision**: Stay with 2.5 Flash, implement streaming to handle large payloads

### B. RAG (Retrieval-Augmented Generation) for Historical Stats
**Pros**: Could answer "What was Kobe's stats in 2008 Finals?"  
**Cons**: Requires vector DB (Pinecone, Weaviate), high setup cost  
**Decision**: Defer to Phase 15+, use Gemini knowledge for historical queries

### C. Custom UI Framework (Svelte, Solid.js) Instead of React
**Pros**: Smaller bundle size, faster rendering  
**Cons**: Smaller ecosystem, team familiarity with React  
**Decision**: Stay with React 18, optimize with code splitting

### D. WebSocket for All API Calls (Not Just Live Scores)
**Pros**: Real-time updates, persistent connection  
**Cons**: Complex state management, server resource usage  
**Decision**: Use WebSocket only for live game updates (Phase 11.3)

---

## Conclusion

This roadmap provides a **strategic path forward** for Hooplytics based on:
1. **Current Limitations**: Address finish_reason 12, expand ML capabilities
2. **User Value**: Prioritize visualizations, multi-season tracking, conversation memory
3. **Technical Feasibility**: Balance impact vs effort, sequence dependencies logically

**Recommended Next Steps**:
1. **Week 1-2**: Implement P0 quick wins (caching, rate limiting, conversation memory, error tracking)
2. **Month 1**: Tackle streaming architecture (eliminates main blocker)
3. **Month 2**: Expand ML model with advanced features (biggest differentiation)
4. **Month 3+**: Add visualizations, multi-league support based on user feedback

**Total Effort Estimate**: 6-8 weeks for Phases 9-11 (core improvements)  
**Expected Outcome**: 97%+ ML accuracy, 3x faster responses, 50% visual queries, production-ready deployment

---

**Document Owner**: Architecture Team  
**Review Cycle**: Quarterly (every 3 months)  
**Last Updated**: 2025-12-03
