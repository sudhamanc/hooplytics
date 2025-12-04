# Hooplytics Constitution

## Core Principles

### I. Hybrid Intelligence Architecture
**Multi-Modal Integration**: Every feature must leverage the three-tier architecture:
- **LLM Layer** (Gemini 2.5 Flash): Natural language understanding and orchestration
- **MCP Layer**: Standardized tool integration for real-time NBA data
- **ML Layer** (PyTorch): Domain-specific classification and predictions

**Separation of Concerns**: Never mix LLM logic, MCP tool execution, or ML inference within the same module. Each tier must be independently testable and deployable.

### II. Content Policy First (NON-NEGOTIABLE)
**JSON Sanitization Mandatory**: ALL tool responses must pass through `sanitize_tool_response()` before returning to Gemini LLM to prevent `finish_reason: 12` blocks.

**Natural Language Protocol**: 
- MCP tools → JSON responses (structured data)
- JSON Sanitizer → Natural language summaries (LLM-safe)
- Gemini → Markdown responses (user-facing)

**Never Return Raw JSON to Gemini**: This is the #1 cause of conversation blocks. Violating this principle breaks the entire application.

### III. Basketball-Only Policy
**Strict Domain Boundaries**: The application ONLY answers basketball-related questions (NBA, WNBA, basketball history, players, teams, games, scores from any date).

**Guardrails Enforcement**: Non-basketball queries must be rejected with the standard response. No exceptions, no scope creep into general knowledge domains.

### IV. ML Model Integrity
**Classification Truth**: When `classify_player_tier()` returns a tier, that classification is TRUTH. Never override with LLM basketball knowledge.

**Model-First Decisions**: For any question about player performance levels, tiers, or quality:
1. ALWAYS use the ML classifier
2. Report the exact tier returned (Bench, Rotation, Starter, All-Star, Elite)
3. Include confidence scores and probability distributions
4. Never second-guess the model's output

**Training Data Constraints**: Model trained on 2021-2026 seasons only. Classification accuracy is 96.2% within this range. Outside this range, acknowledge limitations.

### V. Developer Experience & Maintainability
**PEP 8 Strict Compliance**: All Python code must pass PEP 8 linting. No exceptions.

**Type Safety**: Use strict static typing (`typing` module) for all function arguments and return values. This includes MCP tools, backend routes, and ML model interfaces.

**Small Functions**: Functions must be <20 lines. If a function grows beyond this, refactor immediately into smaller, single-purpose functions.

**DRY Principle**: Extract shared logic into utility functions. Examples:
- Unicode normalization in `nba_server.py`
- JSON sanitization in `json_sanitizer.py`
- Feature extraction in classification module

### VI. Unicode & International Support
**NFD Normalization**: ALL player name lookups must use Unicode NFD normalization to handle special characters (Jokić, Dončić, Antetokounmpo).

**Case-Insensitive Matching**: Player and team name matching must be case-insensitive after normalization.

### VII. Testing & Quality Gates
**Category-Based Testing**: Every PR must demonstrate functionality across all 6 query categories:
1. NBA API Only
2. Gemini Knowledge Only  
3. API + Gemini
4. API + Gemini + Classifier
5. Classifier Only
6. Guardrails

**Integration Tests Required**: Changes to MCP tools, LLM orchestration, or ML inference must include integration tests.

**Test Command**: `curl -X POST http://localhost:8000/api/chat` with sample queries from `GAME_CONTEXT_QUESTIONS.md`

## Technology Stack Requirements

### Backend
- **Python**: 3.13+ (latest stable)
- **FastAPI**: Web framework for /api/chat endpoint
- **Google Gemini API**: 2.5 Flash model only (60 req/min free tier)
- **MCP**: Model Context Protocol client/server
- **PyTorch**: 2.5.1+ (CPU-only, no CUDA requirement)
- **nba_api**: Wrapper for NBA Stats API (no auth required)

### Frontend  
- **React**: 18+ with TypeScript
- **Vite**: 6+ for build tooling
- **Tailwind CSS**: 4+ for styling (dark theme: `bg-[#0f1014]`)
- **react-markdown**: For rich response rendering

### ML Training
- **scikit-learn**: Feature scaling (StandardScaler)
- **matplotlib/seaborn**: Training visualizations
- **PyTorch**: Model training with early stopping

## Security Standards (NON-NEGOTIABLE)

### API Key Management
**Environment Variables Only**: NEVER hardcode `GOOGLE_API_KEY`. Must be in `backend/.env` (gitignored).

**MyBinder Validation**: Startup scripts must validate API key presence with user-friendly error messages.

### Input Validation
**NBA API Inputs**: Validate team names, player names, and season parameters before making API calls.

**LLM Inputs**: All user messages must be sanitized before sending to Gemini (prevent prompt injection).

### Dependency Security
**No Deprecated Packages**: Use latest stable versions. Check for known vulnerabilities before adding dependencies.

**SSL Certificates**: MyBinder requires `pip-system-certs>=5.0` in `environment.yml` to avoid SSL errors.

## Performance Standards

### Response Times
- **Simple Queries** (Category 1-2): <2 seconds
- **ML Classification** (Category 5): <5 seconds  
- **Complex Orchestration** (Category 4): <10 seconds
- **Max Tool Iterations**: 20 (safety limit to prevent infinite loops)

### Resource Constraints
- **Model Size**: PyTorch model must be <50MB (Git compatibility)
- **Memory**: Application must run in 2GB RAM (MyBinder constraint)
- **Scaler Params**: JSON file <1KB

### Query Scope Limits
**finish_reason 12 Mitigation**: Scope complex queries to 1-2 teams maximum. Queries requiring 16+ team rosters + 200+ classifications will fail due to Gemini content policy.

**Working Pattern**: "How many Elite players does #1 seed have?" ✅  
**Failing Pattern**: "Classify all players in all 8 games today" ❌

## Development Workflow

### Local Development
1. **Backend**: `cd backend && source venv/bin/activate && uvicorn main:app --reload --port 8000`
2. **Frontend**: `cd frontend && npm run dev` (port 5173)
3. **Environment**: Requires `backend/.env` with `GOOGLE_API_KEY`

### Model Training Pipeline
1. Download: `python classification/download_nba_data.py` (2021-2026 seasons)
2. Preprocess: `python classification/data_preprocessing.py` (tier labeling)
3. Train: `python classification/train_classifier.py` (93 epochs with early stopping)

### MyBinder Deployment
- **Config**: `binder/environment.yml` (conda), `binder/postBuild` (frontend build)
- **Startup**: `./binder/start_mybinder.sh` (validates API key, starts servers)
- **Static Files**: Backend serves frontend from `backend/main.py` (routes 85-99)

### UI Layout Standards
- **Header**: Must have `sticky top-0` class (line 133 in `ChatInterface.tsx`)
- **Chat Input**: Must have `pt-[195px]` top padding (line 280) to align with header
- **Branding**: Hooplytics logo must always be visible (no conditional rendering)

## Code Review Requirements

### MCP Tool Changes
1. Update tool function in `mcp-server/nba_server.py` with `@mcp.tool()` decorator
2. Add sanitization logic to `backend/json_sanitizer.py`
3. Update Gemini system prompt in `backend/main.py` (lines 120-270)
4. Test with multi-step queries requiring tool orchestration

### ML Model Changes
1. Modify architecture in `classification/player_classifier_model.py`
2. Retrain via `classification/train_classifier.py`
3. Validate test accuracy ≥95%
4. Update `data/models/player_classifier.pth` (<50MB)
5. Restart backend to reload model

### Frontend Changes
1. Maintain two-column layout (left: AI feed, right: controls)
2. Preserve sticky header and chat input alignment
3. Test on mobile (Tailwind responsive classes)
4. Verify markdown rendering (tables, code blocks, emojis)

## Governance

### Constitution Authority
This constitution supersedes all other development practices, coding standards, and architectural decisions. When conflicts arise, constitution principles take precedence.

### Amendment Process
1. Propose amendment with rationale and migration plan
2. Validate against existing codebase
3. Update `.github/copilot-instructions.md` to reflect changes
4. Increment version number

### Compliance Verification
- All PRs must demonstrate adherence to Core Principles I-VII
- Code reviews must explicitly check Content Policy First (Principle II)
- Integration tests must cover all 6 query categories
- Performance benchmarks must meet response time standards

### Runtime Guidance
Use `.github/copilot-instructions.md` for detailed implementation patterns, common pitfalls, and debugging workflows.

**Version**: 1.0.0 | **Ratified**: 2025-12-03 | **Last Amended**: 2025-12-03
