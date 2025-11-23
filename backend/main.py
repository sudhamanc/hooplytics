import os
import sys
from typing import List, Optional, Any
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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
    
    print("Connecting to MCP Server...")
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                mcp_session = session
                
                result = await session.list_tools()
                mcp_tools = result.tools
                
                print(f"Loaded {len(mcp_tools)} tools from MCP Server")
                yield
    except Exception as e:
        print(f"Failed to connect to MCP Server: {e}")
        yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    print("✓ Created tool with NBA API functions (Google Search auto-enabled for gemini-2.5-flash)")
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
        system_instruction = f"""You are an expert NBA assistant. Today's date is {current_date}.

You have access to three NBA API tools for specific queries:
1. get_live_games() - for TODAY'S games only
2. get_standings() - for current season standings only  
3. get_player_stats(player_name) - for career stats only

IMPORTANT: Only use these tools when the query specifically asks for:
- Today's games/scores
- Current league standings  
- A player's career statistics

For everything else (team records, recent games, season stats, news, historical facts), use your general knowledge and search capabilities. Do NOT call NBA API tools unnecessarily.

Only answer basketball-related questions (NBA, WNBA, basketball history/culture).

Be helpful and provide accurate, detailed NBA information."""

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
            
            print(f"→ Calling tool: {tool_name} with args: {args}")
            
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
        
        # Log the source
        if source == "NBA API":
            print("✓ Response used NBA API tools")
        else:
            print("✓ Response from Gemini (LLM knowledge + web search)")
        
        return {
            "role": "assistant",
            "content": response.text,
            "source": source
        }

    except Exception as e:
        import traceback
        print(f"✗ Error in chat_endpoint: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
