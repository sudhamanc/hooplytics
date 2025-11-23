import os
import asyncio
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from contextlib import asynccontextmanager

load_dotenv()

# Global state for MCP
mcp_session: Optional[ClientSession] = None
mcp_tools: List[Any] = []
gemini_tools: List[Any] = []

import sys

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Connect to MCP Server
    global mcp_session, mcp_tools, gemini_tools
    
    # Define server parameters
    server_params = StdioServerParameters(
        command=sys.executable, # Use the same python interpreter (venv)
        args=["mcp-server/nba_server.py"],
        env=os.environ.copy()
    )
    
    print("Connecting to MCP Server...")
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                mcp_session = session
                
                # List tools
                result = await session.list_tools()
                mcp_tools = result.tools
                
                # Convert to Gemini Tools format
                gemini_tools = []
                for tool in mcp_tools:
                    schema = tool.inputSchema.copy()
                    if "title" in schema:
                        del schema["title"]
                    # Also remove title from properties if present
                    if "properties" in schema:
                        for prop in schema["properties"].values():
                            if "title" in prop:
                                del prop["title"]
                                
                    gemini_tools.append({
                        "function_declarations": [{
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": schema
                        }]
                    })
                
                print(f"Loaded {len(mcp_tools)} tools from MCP Server")
                yield
    except Exception as e:
        print(f"Failed to connect to MCP Server: {e}")
        yield

app = FastAPI(lifespan=lifespan)

# Configure CORS
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
# Use a model that supports function calling
model = genai.GenerativeModel('gemini-2.0-flash')

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

class ChatResponse(BaseModel):
    role: str
    content: str
    source: Optional[str] = None

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    global mcp_session
    
    try:
        # 1. Prepare History
        history = []
        for msg in request.messages[:-1]:
            role = "user" if msg.role == "user" else "model"
            history.append({"role": role, "parts": [msg.content]})
        
        # 2. Create model with system instruction
        system_instruction = """You are an expert NBA assistant. You have comprehensive knowledge about NBA history, teams, players, and statistics.

For general NBA knowledge questions (like historical facts, all-time records, team histories), use your training data to provide accurate answers.

For real-time or current data (like today's games, current standings, current season stats), use the available tools to fetch live data.

Be helpful and provide detailed, accurate information about the NBA."""

        chat_model = genai.GenerativeModel(
            'gemini-2.0-flash',
            system_instruction=system_instruction
        )
        
        chat = chat_model.start_chat(history=history)
        
        # 3. Send user's message to Gemini with available tools
        last_message = request.messages[-1].content
        
        response = chat.send_message(
            last_message,
            tools=gemini_tools if gemini_tools else None
        )
        
        source = "Gemini LLM"
        final_content = ""

        # Check for function calls
        # This loop handles multiple tool calls if needed
        while response.parts and response.parts[0].function_call:
            fc = response.parts[0].function_call
            tool_name = fc.name
            args = dict(fc.args)
            
            print(f"Gemini requested tool: {tool_name} with args: {args}")
            
            if not mcp_session:
                raise HTTPException(status_code=500, detail="MCP Session not active")
            
            # Execute tool via MCP
            tool_result = await mcp_session.call_tool(tool_name, arguments=args)
            result_text = tool_result.content[0].text
            
            # Set source to NBA API if a tool was called
            source = "NBA API"
            
            # Send result back to Gemini
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
            
        final_content = response.text
        return {"role": "assistant", "content": final_content, "source": source}

    except Exception as e:
        import traceback
        print(f"Error in chat_endpoint: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
