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
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

# Global state for MCP
mcp_session: Optional[ClientSession] = None
mcp_tools: List[Any] = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Connect to MCP Server on startup"""
    global mcp_session, mcp_tools
    
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["mcp-server/nba_server.py"],
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
        return {"message": "Hoop.io NBA Assistant API", "status": "running", "mode": "development"}

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
    
    for mcp_tool in mcp_tools:
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
        # Prepare conversation history
        history = []
        for msg in request.messages[:-1]:
            role = "user" if msg.role == "user" else "model"
            history.append({"role": role, "parts": [msg.content]})
        
        # System instruction
        current_date = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        system_instruction = f"""You are an expert NBA assistant with access to comprehensive basketball knowledge. Today's date is {current_date}.

You have three specialized NBA API tools for specific real-time data:
- get_live_games() - for TODAY'S live games only
- get_standings() - for current season standings only  
- get_player_stats(player_name) - for career stats only

TOOL USAGE: Use the NBA API tools strategically:
- Today's live games/scores → use get_live_games()
- Current league standings → use get_standings()
- A player's career statistics → use get_player_stats(player_name)
- Current season team records → use get_standings()
- Questions like "how many games did [player] win this season?" → use get_standings() to get their team's record

For ALL other basketball questions (yesterday's games, past scores, recent news, historical data, injuries, trades, playoff results, draft information, etc.), use your extensive basketball knowledge to provide accurate answers.

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
"I'm Hoop.io, your NBA assistant. I can only help with basketball-related questions. Ask me about NBA games, players, stats, teams, or basketball history!"

Do NOT use this rejection response for ANY basketball-related questions including past games, scores, or historical NBA data."""

        # Create model with combined tool
        model = genai.GenerativeModel(
            'gemini-2.5-flash',
            system_instruction=system_instruction
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
        
        # Handle function calls (tool execution loop)
        while response.parts and hasattr(response.parts[0], 'function_call') and response.parts[0].function_call:
            fc = response.parts[0].function_call
            tool_name = fc.name
            args = dict(fc.args)
            
            if not mcp_session:
                raise HTTPException(status_code=500, detail="MCP Session not active")
            
            # Execute tool via MCP
            tool_result = await mcp_session.call_tool(tool_name, arguments=args)
            result_text = tool_result.content[0].text
            
            # Mark as NBA API source
            source = "NBA API"
            
            # Send result back to model
            response = chat.send_message(
                genai.protos.Content(
                    parts=[genai.protos.Part(
                        function_response=genai.protos.FunctionResponse(
                            name=tool_name,
                            response={'result': result_text}
                        )
                    )]
                )
            )
        
        return {
            "role": "assistant",
            "content": response.text,
            "source": source
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
