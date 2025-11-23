# 🏀 Hoop.io - AI-Powered NBA Assistant

<div align="center">

![Hoop.io Banner](https://img.shields.io/badge/NBA-Assistant-orange?style=for-the-badge&logo=basketball)
![Python](https://img.shields.io/badge/Python-3.13-blue?style=for-the-badge&logo=python)
![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi)

**Your intelligent companion for NBA stats, live scores, and basketball knowledge**

[Features](#-features) • [Installation](#-installation) • [Usage](#️-usage) • [Technologies](#-technologies)

</div>

---

## 📖 Overview

**Hoop.io** is a sophisticated AI-powered chatbot that combines Google Gemini 2.5 Flash with real-time NBA data through the Model Context Protocol (MCP). Ask about historical NBA facts, get live game scores, check player statistics, explore team standings, or inquire about recent NBA news - all through a beautiful, modern chat interface.

The application intelligently decides when to use dedicated NBA API tools versus Gemini's extensive capabilities (which include LLM knowledge and web search), providing accurate and up-to-date information for all your basketball queries.

---

## ✨ Features

### 🤖 **Intelligent Query Handling**
- **Dual-Source Intelligence**: Automatically switches between NBA API tools and Gemini's knowledge base
- **Gemini 2.5 Flash**: Advanced LLM with built-in web search capabilities for current information
- **Contextual Conversations**: Maintains conversation history for natural, multi-turn dialogues
- **Smart Tool Selection**: Intelligently decides when to use NBA API tools vs general knowledge/web search

### 📊 **Real-Time NBA Data**
- **Live Game Scores**: Get current scores and game status for today's matches via NBA API
- **League Standings**: Check current NBA standings for both conferences
- **Player Statistics**: Fetch detailed career stats for any NBA player
- **Current Season Data**: Gemini provides team records, recent games, and up-to-date news
- **Historical Knowledge**: Ask about NBA history, records, and all-time achievements

### 🎨 **Premium User Interface**
- **Modern Design**: Glassmorphism-inspired dark theme with NBA color accents
- **Two-Column Layout**: AI responses on the left, chat controls on the right
- **Source Attribution**: See whether data came from NBA API or Gemini
- **Quick Actions**: One-click access to popular queries
- **Conversation History**: Review and re-ask previous questions

---

## 🚀 Installation

### Prerequisites
- Python 3.13+
- Node.js 18+
- npm or yarn
- Google Gemini API key ([Get one here](https://makersuite.google.com/app/apikey))

### Step 1: Clone the Repository
```bash
git clone <your-repo-url>
cd Assignment4
```

### Step 2: Backend Setup
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r backend/requirements.txt

# Configure environment variables
cd backend
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

### Step 3: Frontend Setup
```bash
cd frontend
npm install
```

### Step 4: Start the Application

**Terminal 1 - Backend:**
```bash
# From project root
./venv/bin/uvicorn backend.main:app --reload --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

Visit **http://localhost:5173** in your browser! 🎉

### 🌐 Deploy on MyBinder (Alternative)

Want to try Hoop.io without installing anything? Launch it on MyBinder!

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/sudhamanc/Hoop.io/main)

**Steps:**
1. Click the "launch binder" badge above
2. Wait for the environment to build (first time takes ~5-10 minutes)
3. Once loaded, open a terminal in JupyterLab
4. Run: `./start.sh`
5. Click on the proxy URLs to access:
   - Frontend: Port 5173
   - Backend API: Port 8000

> ⚠️ **Note**: You'll need to set your `GOOGLE_API_KEY` environment variable in the Binder terminal:
> ```bash
> export GOOGLE_API_KEY="your_key_here"
> ```

**Binder Configuration Files:**
- `binder/environment.yml` - Conda environment with Python, Node.js, and dependencies
- `binder/postBuild` - Post-build script to install frontend packages
- `start.sh` - Startup script to launch both servers

---

## 🛠️ Usage

### Basic Queries

**Historical Questions** (uses Gemini's knowledge):
```
"Which team has won the most NBA titles?"
"Who is the all-time leading scorer?"
"Tell me about Michael Jordan's career"
```

**Live Data** (uses NBA API):
```
"What are today's games?"
"Show me current standings"
"Get Stephen Curry's career stats"
```

**Recent/Current Season** (uses Gemini):
```
"How many wins does the Lakers have this season?"
"Who won last night's game between Warriors and Celtics?"
"Latest NBA trade news"
```

### Example Conversation
```
You: Who won the most NBA championships?
Hoop.io: The Boston Celtics have won 17 NBA championships.
         Source: Gemini

You: What about the Lakers?
Hoop.io: The Los Angeles Lakers have also won 17 championships, 
         tied with the Celtics for the most all-time.
         Source: Gemini

You: Show me LeBron James' career stats
Hoop.io: [Detailed career statistics from NBA API]
         Source: NBA API

You: How many wins does the Lakers have this season?
Hoop.io: [Current season record from Gemini's knowledge/web search]
         Source: Gemini
```

---

## 📦 Technologies

### Backend
- **FastAPI** - Modern Python web framework
- **Google Gemini 2.5 Flash** - Advanced LLM with built-in web search capabilities
- **MCP (Model Context Protocol)** - Tool integration framework
- **FastMCP** - Python MCP server implementation
- **nba_api** - Official NBA statistics API wrapper
- **Python 3.13** - Latest Python runtime

### Frontend
- **React 18** - UI framework
- **TypeScript** - Type-safe JavaScript
- **Vite** - Next-generation frontend tooling
- **Tailwind CSS v4** - Utility-first CSS framework
- **react-markdown** - Markdown rendering for rich responses

### Architecture
- **MCP Protocol** - Standardized tool calling interface
- **Async/Await** - Non-blocking I/O for better performance
- **RESTful API** - Clean HTTP interface between frontend and backend

---

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the `backend/` directory:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

**Getting Your Gemini API Key:**
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the key and paste it into your `.env` file

> ⚠️ **Security Note**: Never commit your `.env` file to version control. The `.gitignore` is already configured to exclude it.

### System Instruction (Optional)

You can customize the AI's behavior by editing the system instruction in `backend/main.py`:

```python
system_instruction = """You are an expert NBA assistant..."""
```

---

## ✅ Requirements

### Python Dependencies
```
fastapi>=0.115.0
uvicorn>=0.32.0
google-generativeai>=0.8.3
mcp>=1.1.2
nba_api>=1.5.2
python-dotenv>=1.0.1
httpx>=0.27.2
```

### Node Dependencies
```json
{
  "react": "^18.3.1",
  "vite": "^6.0.1",
  "tailwindcss": "^4.0.0",
  "react-markdown": "^9.0.1"
}
```

---

## 🗂️ Repository Structure

```
Assignment4/
├── backend/                    # FastAPI backend server
│   ├── main.py                # Main application entry point
│   │                          # - FastAPI app configuration
│   │                          # - MCP client setup and lifespan management
│   │                          # - Chat endpoint with Gemini 2.5 Flash
│   │                          # - Combined tool with Google Search + NBA API
│   │                          # - Tool calling and response handling
│   ├── requirements.txt       # Python dependencies
│   ├── .env.example          # Environment variables template
│   ├── .env                  # Your API keys (gitignored)
│   └── .gitignore            # Git ignore rules
│
├── frontend/                  # React frontend application
│   ├── src/
│   │   ├── components/
│   │   │   └── ChatInterface.tsx  # Main chat UI component
│   │   │                          # - Message display and management
│   │   │                          # - User input handling
│   │   │                          # - History and quick actions
│   │   ├── App.tsx               # Root React component
│   │   ├── main.tsx              # React entry point
│   │   └── index.css             # Global styles and Tailwind config
│   ├── package.json              # Node dependencies
│   ├── tailwind.config.js        # Tailwind CSS configuration
│   ├── postcss.config.js         # PostCSS configuration
│   └── vite.config.ts            # Vite build configuration
│
├── mcp-server/                # MCP server for NBA data
│   └── nba_server.py         # FastMCP server implementation
│                             # - get_live_games() tool
│                             # - get_standings() tool
│                             # - get_player_stats() tool
│
├── venv/                      # Python virtual environment (gitignored)
└── README.md                 # This file
```

### Key Components Explained

**`backend/main.py`**
- **Lifespan Manager**: Connects to MCP server on startup, loads NBA tools
- **Chat Endpoint**: Handles user messages, manages conversation history
- **Combined Tool**: Creates unified tool with Google Search grounding + NBA API functions
- **Tool Integration**: Converts MCP tools to Gemini format, executes tool calls
- **Response Handling**: Tracks data source (NBA API, Google Search, or Gemini LLM)

**`mcp-server/nba_server.py`**
- **FastMCP Server**: Exposes NBA data as MCP tools
- **NBA API Integration**: Uses `nba_api` library for live data
- **Tool Definitions**: Three main tools with clear descriptions and schemas

**`frontend/src/components/ChatInterface.tsx`**
- **State Management**: Tracks messages, input, loading states
- **API Communication**: Sends requests to backend, handles responses
- **UI Rendering**: Two-column layout with glassmorphism design
- **User Interactions**: Input handling, quick actions, history management

---

## 🔗 Flow Chart

```mermaid
graph TB
    subgraph "User Interface"
        A[User enters question] --> B[ChatInterface.tsx]
        B --> C[Send to Backend API]
    end
    
    subgraph "Backend Processing"
        C --> D[FastAPI /api/chat endpoint]
        D --> E[Build conversation history]
        E --> F[Create Gemini model with system instruction]
        F --> G[Send message with available tools]
    end
    
    subgraph "Gemini Decision"
        G --> H{Does question need live data?}
        H -->|No| I[Use general knowledge]
        H -->|Yes| J[Call appropriate tool]
    end
    
    subgraph "Tool Execution"
        J --> K[Gemini returns function_call]
        K --> L[Backend detects function_call]
        L --> M[Execute tool via MCP session]
        M --> N[MCP Server calls NBA API]
        N --> O[Return data to backend]
        O --> P[Send result back to Gemini]
        P --> Q[Gemini formats response]
    end
    
    I --> R[Return response to frontend]
    Q --> R
    R --> S[Display in chat UI]
    S --> T[Show source attribution]
    
    style A fill:#4CAF50
    style S fill:#2196F3
    style H fill:#FF9800
    style N fill:#F44336
```

### Data Flow Explanation

1. **User Input** → User types question in chat interface
2. **Frontend** → Sends message + history to backend API
3. **Backend** → Prepares context, creates combined tool (Google Search + NBA functions), sends to Gemini
4. **Gemini Analysis** → Decides whether to use LLM knowledge, Google Search, or NBA API tools
5. **Tool Execution** (if needed) → Backend executes NBA MCP calls or processes Google Search results
6. **Response** → Gemini formats data → Backend adds source attribution (NBA API/Google Search/Gemini) → Frontend displays with source tag

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes** and test thoroughly
4. **Commit your changes**: `git commit -m 'Add amazing feature'`
5. **Push to the branch**: `git push origin feature/amazing-feature`
6. **Open a Pull Request**

### Development Guidelines
- Follow PEP 8 for Python code
- Use TypeScript for new React components
- Add comments for complex logic
- Test with multiple query types before submitting

---

## 📄 Documentation

### Additional Resources
- [Google Gemini API Documentation](https://ai.google.dev/docs)
- [Model Context Protocol Specification](https://modelcontextprotocol.io/)
- [NBA API Documentation](https://github.com/swar/nba_api)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://react.dev/)

### Key Concepts
- **MCP (Model Context Protocol)**: A standardized way for LLMs to interact with external tools
- **Function Calling**: Gemini's ability to recognize when to use tools vs general knowledge
- **Google Search Grounding**: Real-time web search integration for up-to-date information beyond training data
- **Tool Orchestration**: Combining multiple data sources (NBA API + Google Search + LLM) in a single unified interface
- **Agentic Behavior**: The LLM acts as an intelligent agent, making decisions about tool usage

---

## ❤️ Acknowledgements

- **Google Gemini** - For the powerful language model and function calling capabilities
- **Anthropic** - For pioneering the Model Context Protocol
- **nba_api** - For providing easy access to NBA statistics
- **FastAPI** - For the excellent Python web framework
- **React Team** - For the amazing frontend library
- **Tailwind CSS** - For the utility-first CSS framework

Special thanks to the open-source community for making projects like this possible!

---

## 📝 Changelog

### Version 1.0.0 (Current)
- ✅ Initial release with full functionality
- ✅ Integrated Google Gemini 2.5 Flash with MCP and Google Search grounding
- ✅ Implemented three NBA API tools (live games, standings, player stats)
- ✅ Built premium glassmorphism UI with two-column layout
- ✅ Added conversation history and context management
- ✅ Implemented triple-source attribution (NBA API, Google Search, Gemini LLM)
- ✅ Added quick actions and suggested queries
- ✅ Created comprehensive documentation

### Future Enhancements
- 🔄 Add more NBA tools (team stats, game highlights, playoff brackets)
- 🔄 Implement user authentication and saved conversations
- 🔄 Add data visualization for statistics
- 🔄 Support for multiple sports leagues
- 🔄 Voice input/output capabilities

---

<div align="center">

**Built with ❤️ for basketball fans everywhere**


</div>
