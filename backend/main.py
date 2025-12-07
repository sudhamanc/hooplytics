import os
import sys
from typing import List, Optional, Any
from datetime import datetime
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

import google.generativeai as genai
from google.generativeai.types import Tool, FunctionDeclaration
from google.generativeai import protos
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Add backend directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from json_sanitizer import sanitize_tool_response

load_dotenv()

# Global state for MCP
mcp_session: Optional[ClientSession] = None
mcp_tools: List[Any] = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Connect to MCP Server on startup"""
    global mcp_session, mcp_tools
    
    # Get the path to the MCP server (parent directory of backend)
    backend_dir = Path(__file__).parent
    mcp_server_path = backend_dir.parent / "mcp-server" / "nba_server.py"
    
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(mcp_server_path)],
        env=os.environ.copy()
    )
    
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                mcp_session = session
                
                result = await session.list_tools()
                mcp_tools = result.tools
                
                yield
    except Exception as e:
        yield

app = FastAPI(lifespan=lifespan)

# Determine if running in MyBinder environment
IS_MYBINDER = os.getenv("JUPYTERHUB_SERVICE_PREFIX") is not None

# Set CORS based on environment
if IS_MYBINDER:
    # MyBinder: Allow all origins (static files served from same origin)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    # Local dev: Only allow Vite dev server
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Serve static files only in MyBinder environment
static_dir = Path(__file__).parent.parent / "frontend" / "dist"
if IS_MYBINDER and static_dir.exists():
    # Mount assets directory
    app.mount("/assets", StaticFiles(directory=static_dir / "assets"), name="assets")
    
    @app.get("/vite.svg")
    async def serve_vite_svg():
        """Serve vite.svg if it exists"""
        svg_path = static_dir / "vite.svg"
        if svg_path.exists():
            return FileResponse(svg_path)
        return {"error": "Not found"}
    
    @app.get("/")
    async def serve_root():
        """Serve the frontend index.html (MyBinder only)"""
        return FileResponse(static_dir / "index.html")
elif IS_MYBINDER:
    @app.get("/")
    async def root():
        return {"error": "Frontend not built. Run './start_mybinder.sh' to build and serve the app."}
else:
    @app.get("/")
    async def root():
        return {"message": "Hooplytics NBA Assistant API", "status": "running", "mode": "development"}

# Configure Gemini
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

genai.configure(api_key=api_key)

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

class ChatResponse(BaseModel):
    role: str
    content: str
    source: Optional[str] = None

def create_combined_tool() -> Tool:
    """Create a combined tool with Google Search + NBA API functions"""
    # Build function declarations from MCP tools
    function_declarations = []
    
    print(f"DEBUG: Creating combined tool from {len(mcp_tools)} MCP tools")
    
    for mcp_tool in mcp_tools:
        print(f"DEBUG: Registering tool: {mcp_tool.name}")
        # Convert MCP tool schema to Gemini FunctionDeclaration
        params = mcp_tool.inputSchema.copy()
        
        # Remove problematic fields
        params.pop("title", None)
        params.pop("$schema", None)
        
        # Clean nested properties
        if "properties" in params:
            for prop_value in params["properties"].values():
                prop_value.pop("title", None)
        
        # Create FunctionDeclaration
        func_decl = FunctionDeclaration(
            name=mcp_tool.name,
            description=mcp_tool.description,
            parameters=params
        )
        function_declarations.append(func_decl)
    
    # Create tool with Google Search + NBA API functions
    # Note: For google-generativeai 0.8.x, google_search parameter is not yet supported
    # Google Search grounding works automatically with gemini-2.5-flash
    combined_tool = Tool(function_declarations=function_declarations)
    return combined_tool

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    global mcp_session
    
    try:
        # Prepare conversation history (keep last 10 messages to avoid context overflow)
        all_messages = request.messages[:-1]
        history = []
        for msg in all_messages[-10:]:  # Limit to last 10 messages
            role = "user" if msg.role == "user" else "model"
            history.append({"role": role, "parts": [msg.content]})
        
        # System instruction
        current_date = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        system_instruction = f"""You are an expert NBA assistant with access to comprehensive basketball knowledge. Today's date is {current_date}.

You have FIVE specialized NBA API tools for specific real-time data:
- get_live_games() - for TODAY'S live games only
- get_standings() - for current season standings only  
- get_player_stats(player_name) - for career stats only
- get_team_roster(team_name) - for current team rosters
- classify_player_tier(player_name) - for ML-based player performance tier classification

TOOL USAGE: Use the NBA API tools strategically:
- Today's live games/scores → use get_live_games()
- Current league standings → use get_standings()
- A player's career statistics → use get_player_stats(player_name)
- Classify a player's performance tier → use classify_player_tier(player_name)
- Current season team records → use get_standings()
- Questions like "how many games did [player] win this season?" → use get_standings() to get their team's record

PLAYER CLASSIFICATION TOOL:
The classify_player_tier(player_name) tool uses a neural network trained on 5 seasons of NBA data (2021-2026) to classify players into performance tiers:
- Elite: Superstar level (MVP candidates, best players in the league)
- All-Star: All-Star caliber players
- Starter: Quality starting players
- Rotation: Solid rotation/bench players
- Bench: Limited role players

⚠️ WHEN TO USE classify_player_tier():
Use this tool for ANY question about player tiers, performance levels, or quality comparisons:
- "Who are the elite players?" → classify multiple players, filter for Elite
- "Is [player] a superstar?" → classify that player
- "Which players from the last 4 seasons were elite?" → classify players from those seasons
- "Top players on [team]" → get roster, classify each player
- "Compare [player A] vs [player B]" → classify both
- "Best player in tonight's game" → classify all players, find highest tier

DO NOT use your basketball knowledge to answer tier/quality questions - ALWAYS use the tool.

⚠️ CRITICAL: When using classify_player_tier():
- You MUST use the EXACT tier returned by the tool, NOT your own basketball knowledge
- The tool classifies based on season performance data (2021-2026), not current real-time stats
- NEVER override or modify the classification result - report it exactly as returned
- The tool works for any player who played during 2021-2026 seasons

MULTI-PLAYER CLASSIFICATION WORKFLOW:
When questions require classifying or comparing multiple players or getting tier counts for a team:

⭐ USE aggregate_roster_classifications(team_name) when:
- User asks for tier counts for a team (e.g., "Give me counts by classification for all players in the #1 seeded team")
- User asks "How many [tier] players does [team] have?"
- User wants a roster breakdown by tier
- User wants to see all player tiers for a team

Steps when using aggregate_roster_classifications:
1. If question mentions "#1 seed", "#2 seed", etc. → first call get_standings() to find the team name
2. Then call aggregate_roster_classifications(team_name) once
3. This single call will get the roster AND classify all players AND return aggregated counts
4. Respond with the tier counts and player lists provided by the tool

⚠️ For individual player comparisons or single player classification:
1. If player names are mentioned directly → call classify_player_tier() for each player
2. For comparing specific players → call classify_player_tier() once per player

⚠️ IMPORTANT: Do NOT use get_current_games_with_rosters() for classification queries - it returns too much data.
Only use get_current_games_with_rosters() when you need games AND rosters together for informational purposes (not for classification).

⚠️ CRITICAL OUTPUT FORMATTING:
When you make multiple tool calls (especially classify_player_tier), do NOT echo or display the raw tool responses.
Instead:
1. Collect all the tool results silently
2. Process and analyze the data
3. Provide ONLY your final synthesized answer

NEVER include raw JSON objects or function response dictionaries in your response.
The user should only see your natural language answer, not the intermediate tool outputs.

Examples requiring tool calls:
- "Give me counts by classification for all players in the #1 seeded team" → get_standings() to find #1 seed → aggregate_roster_classifications(team_name)
- "How many Elite players does the #1 seed have?" → get_standings() to find #1 seed → aggregate_roster_classifications(team_name) → report Elite count
- "Which team has more Elite tier players, Lakers or Celtics?" → aggregate_roster_classifications("Lakers") → aggregate_roster_classifications("Celtics") → compare Elite counts
- "Compare LeBron and Curry" → classify_player_tier("LeBron James") → classify_player_tier("Stephen Curry")
- "Top players on the Warriors" → aggregate_roster_classifications("Warriors") → report Elite + All-Star players

For basketball questions NOT about player tiers/quality, use your extensive basketball knowledge combined with available tools to provide comprehensive answers. When contextual data would enhance accuracy (like team records, current standings, or live scores), proactively use the appropriate tools before answering.

⚠️ UPSETS ANALYSIS WORKFLOW:
When asked about "upsets" in games:
1. Call get_live_games() to get today's scores
2. Call get_standings() to get team records and win percentages
3. Analyze: An upset occurs when a team with a significantly worse record defeats a team with a better record
4. Use these guidelines:
   - Compare win percentages: 10+ percentage point difference suggests potential upset
   - Lower-seeded team beating higher-seeded team = upset
   - Sub-.500 team beating playoff team = notable upset
5. Provide specific analysis citing team records and final scores

Example: "The Pistons (8-21, .276) defeating the Heat (14-13, .519) by 138-135 is an upset because the Pistons have a significantly worse record."

DO NOT say you "can't determine upsets" - you have the tools and knowledge to analyze them.

🚨 CRITICAL FORMATTING RULE 🚨
When you receive JSON from a tool, you MUST:
1. Parse the JSON completely
2. Extract the relevant data fields
3. Format as a clean Markdown table
4. NEVER EVER output raw JSON text or pipe-delimited strings

**WRONG - DO NOT DO THIS:**
| Team | Conference | Wins | Losses | Win % | Conference Rank |-----|-----|-----|-----|-----|
| Cleveland Cavaliers | East | 11 | 6 | 64.7% | 3 |

**RIGHT - DO THIS:**
| Team | Conference | Wins | Losses | Win % | Rank |
|------|------------|------|--------|-------|------|
| Cleveland Cavaliers | East | 11 | 6 | 64.7% | 3 |

**For standings (get_standings):**
Parse the JSON response, extract: TeamCity, TeamName, Conference, WINS, LOSSES, WinPCT, ConferenceRecord
Create a properly formatted Markdown table with headers on one line, separator on the next line, then data rows.

**For live games (get_live_games):**
Parse the JSON games array and format each game with team names, scores, and status.

**For player stats (get_player_stats):**
Parse the JSON stats data and create a clean table with key career statistics.

BASKETBALL-ONLY POLICY: You ONLY answer basketball-related questions (NBA, WNBA, basketball history, culture, players, teams, games, scores from any date).

IMPORTANT: Questions about NBA games, scores, or results from ANY date (today, yesterday, last week, last season, etc.) ARE basketball questions and should be answered normally.

If a user asks about anything not related to basketball (politics, weather, cooking, general news, etc.), politely respond ONLY with:
"I'm Hooplytics, your NBA assistant. I can only help with basketball-related questions. Ask me about NBA games, players, stats, teams, or basketball history!"

Do NOT use this rejection response for ANY basketball-related questions including past games, scores, or historical NBA data."""

        # Create model with safety settings to reduce blocking
        from google.generativeai.types import HarmCategory, HarmBlockThreshold
        
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
        }
        
        model = genai.GenerativeModel(
            'gemini-2.5-flash',
            system_instruction=system_instruction,
            safety_settings=safety_settings
        )
        
        chat = model.start_chat(history=history)
        
        # Create combined tool
        combined_tool = create_combined_tool()
        
        # Send message with tools
        last_message = request.messages[-1].content
        
        response = chat.send_message(
            last_message,
            tools=[combined_tool]
        )
        
        source = "Gemini"
        classifier_used = False  # Track if classifier was used
        
        # Handle function calls (tool execution loop)
        max_iterations = 20  # Reasonable limit for tool orchestration
        iteration = 0
        
        while response.parts and hasattr(response.parts[0], 'function_call') and response.parts[0].function_call:
            iteration += 1
            if iteration > max_iterations:
                print(f"WARNING: Reached maximum tool call iterations ({max_iterations})")
                break
                
            fc = response.parts[0].function_call
            tool_name = fc.name
            args = dict(fc.args)
            
            print(f"Tool call #{iteration}: {tool_name}({args})")
            
            if not mcp_session:
                raise HTTPException(status_code=500, detail="MCP Session not active")
            
            # Execute tool via MCP
            tool_result = await mcp_session.call_tool(tool_name, arguments=args)
            result_text = tool_result.content[0].text
            
            # Track if classifier is used
            if tool_name in ["classify_player_tier", "aggregate_roster_classifications"]:
                classifier_used = True
            
            # Update source attribution
            if classifier_used:
                source = "NBA API + ML Classifier"
            else:
                source = "NBA API"
            
            # Sanitize response
            summarized_result = sanitize_tool_response(tool_name, result_text)
            
            # Send tool result back to Gemini as a proper function response
            # CRITICAL: Include tools parameter so Gemini knows it can continue making tool calls
            try:
                response = chat.send_message(
                    protos.Content(
                        parts=[protos.Part(
                            function_response=protos.FunctionResponse(
                                name=tool_name,
                                response={"result": summarized_result}
                            )
                        )]
                    ),
                    tools=[combined_tool]  # Must include tools to enable continued tool calling
                )
            except Exception as e:
                # Handle cases where Gemini blocks the response
                if "StopCandidateException" in str(type(e)) or "finish_reason" in str(e):
                    print(f"WARNING: Response blocked after tool call, returning partial result")
                    return {
                        "role": "assistant",
                        "content": f"I retrieved data from {tool_name}: {summarized_result}",
                        "source": source
                    }
                raise
            
            # Check if response has more function calls (continue loop) or final answer
            if not response.parts or not any(hasattr(part, 'function_call') and part.function_call for part in response.parts):
                # No more function calls - return the final response
                break
        
        # Return final response text
        # Check if response has text (no pending function calls)
        try:
            response_text = response.text
        except ValueError:
            # Response still has pending function calls - this means we hit max_iterations
            response_text = (
                "I encountered a processing limit while handling your request. "
                "This usually happens with very long conversations. "
                "Try refreshing the page and asking your question again, or ask a more focused question."
            )
        
        return {
            "role": "assistant",
            "content": response_text,
            "source": source
        }

    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n{traceback.format_exc()}"
        print(f"ERROR in chat_endpoint: {error_detail}")
        raise HTTPException(status_code=500, detail=str(e))
