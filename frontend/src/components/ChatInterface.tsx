import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  source?: string;
}

const ChatInterface: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);



  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      role: 'user',
      content: input,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      // Use environment variable or default to relative path for production builds
      const apiUrl = import.meta.env.VITE_API_URL || '';
      // Use relative path without leading slash to stay within proxy context
      const endpoint = apiUrl ? `${apiUrl}/api/chat` : './api/chat';
      
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          messages: [...messages, userMessage].map(m => ({ role: m.role, content: m.content })),
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Failed to fetch response: ${response.status}`);
      }

      const data = await response.json();
      const assistantMessage: Message = {
        role: 'assistant',
        content: data.content,
        timestamp: new Date(),
        source: data.source
      };
      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: "Sorry, I encountered an error connecting to the NBA server.",
        timestamp: new Date(),
        source: "System"
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex h-screen bg-[#0f1014] text-white font-sans overflow-hidden relative selection:bg-purple-500/30">
      {/* Background Gradients */}
      <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-purple-600/20 rounded-full blur-[120px] pointer-events-none" />
      <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-blue-600/20 rounded-full blur-[120px] pointer-events-none" />

      {/* Left Column: AI Responses / "The Feed" */}
      <div className="flex-1 flex flex-col relative z-10 h-full overflow-hidden">
        {/* Fixed Header Section */}
        <header className="flex-none p-6 border-b border-white/5 bg-[#0f1014]/95 backdrop-blur-md z-20 shrink-0">
          <div className="flex items-center gap-4 mb-4">
            <div className="w-10 h-10 rounded-2xl bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center shadow-lg shadow-purple-500/20">
              <span className="text-white font-bold text-lg">IO</span>
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight">
                Hoop<span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-400">.io</span>
              </h1>
              <p className="text-slate-400 text-xs font-medium tracking-wide uppercase">NBA Intelligence Engine</p>
            </div>
          </div>

          {/* Persistent Welcome Banner - Visible only when chatting */}
          {messages.length > 0 && (
            <div className="flex items-center gap-4 p-3 rounded-xl bg-white/5 border border-white/5">
              <div className="w-10 h-10 rounded-full bg-gradient-to-tr from-blue-500/10 to-purple-500/10 flex items-center justify-center border border-white/5 shrink-0">
                <div className="text-lg">🏀</div>
              </div>
              <div className="min-w-0">
                <h2 className="text-sm font-bold text-white mb-0.5">
                  <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-400 mr-1">Hoop.io</span>
                  Ready for Tip-Off
                </h2>
                <p className="text-xs text-slate-400">Ask about live scores, player stats, or historical data.</p>
              </div>
            </div>
          )}
        </header>

        {/* Scrollable Chat Feed */}
        <div className="flex-1 overflow-y-auto px-8 py-6 space-y-8 scrollbar-thin scrollbar-thumb-slate-700 min-h-0">
          {messages.length === 0 && (
            <div className="h-full flex flex-col items-center justify-center text-center animate-fade-in">
              <div className="w-24 h-24 rounded-full bg-gradient-to-tr from-blue-500/10 to-purple-500/10 flex items-center justify-center mb-6 border border-white/5 backdrop-blur-sm shadow-2xl shadow-blue-500/20">
                <div className="text-5xl">🏀</div>
              </div>
              <h2 className="text-4xl font-bold mb-4">
                <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-400">Hoop.io</span>
              </h2>
              <h3 className="text-2xl font-semibold text-white mb-3">Ready for Tip-Off</h3>
              <p className="text-slate-400 max-w-md leading-relaxed text-lg">
                Your AI-powered NBA companion. Ask about live scores, player stats, historical data, or general basketball knowledge.
              </p>
            </div>
          )}

          {messages.filter(m => m.role === 'assistant').map((msg, index) => (
            <div key={index} className="animate-slide-up">
              <div className="glass-panel rounded-3xl p-8 shadow-2xl shadow-black/50">
                <div className="flex items-center gap-3 mb-6 border-b border-white/5 pb-4">
                  <div className="w-6 h-6 rounded-full bg-gradient-to-r from-blue-500 to-purple-500 flex items-center justify-center">
                    <svg className="w-3 h-3 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M13 10V3L4 14h7v7l9-11h-7z" />
                    </svg>
                  </div>
                  <span className="text-sm font-medium text-slate-300">Insight Generated</span>

                  {/* Source Indicator */}
                  {msg.source && (
                    <span className="ml-2 text-xs text-slate-500 italic border border-white/10 px-2 py-0.5 rounded-full">
                      Source: {msg.source}
                    </span>
                  )}

                  <span className="ml-auto text-xs text-slate-500 font-mono">
                    {msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </span>
                </div>
                <div className="prose prose-invert max-w-none prose-p:text-slate-300 prose-p:leading-loose prose-headings:text-white prose-strong:text-blue-300 prose-a:text-purple-400">
                  <ReactMarkdown>{msg.content}</ReactMarkdown>
                </div>
              </div>
            </div>
          ))}

          {isLoading && (
            <div className="flex items-center gap-4 p-4 opacity-70">
              <div className="flex gap-1.5">
                <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce [animation-delay:0ms]" />
                <div className="w-2 h-2 bg-purple-400 rounded-full animate-bounce [animation-delay:200ms]" />
                <div className="w-2 h-2 bg-white rounded-full animate-bounce [animation-delay:400ms]" />
              </div>
              <span className="text-sm font-medium text-slate-400 tracking-wide">Processing Data...</span>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Right Column: Control Panel */}
      <div className="w-[450px] bg-[#13141c]/90 backdrop-blur-2xl border-l border-white/5 flex flex-col z-20 shadow-[-20px_0_40px_rgba(0,0,0,0.5)] h-full">
        {/* Chat Input Form */}
        <div className="flex-none p-8 pt-40 border-b border-white/5">
          <form onSubmit={handleSubmit} className="relative group">
            <div className="absolute -inset-0.5 bg-gradient-to-r from-blue-500 to-purple-600 rounded-2xl opacity-20 group-hover:opacity-40 transition duration-500 blur"></div>
            <div className="relative bg-[#0f1014] rounded-xl overflow-hidden border border-white/5">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSubmit(e);
                  }
                }}
                placeholder="Ask a question..."
                className="w-full bg-transparent text-white p-4 min-h-[140px] focus:outline-none resize-none placeholder:text-slate-600 text-lg leading-relaxed"
                disabled={isLoading}
              />
              <div className="flex justify-between items-center px-4 pb-3 pt-2 border-t border-white/5 bg-white/5">
                <div className="flex gap-2">
                  <button type="button" className="p-2 text-slate-400 hover:text-white transition-colors">
                    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" /></svg>
                  </button>
                  <button type="button" className="p-2 text-slate-400 hover:text-white transition-colors">
                    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" /></svg>
                  </button>
                </div>
                <button
                  type="submit"
                  disabled={isLoading || !input.trim()}
                  className="bg-white text-black hover:bg-blue-50 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg px-6 py-2 font-semibold text-sm transition-all transform active:scale-95"
                >
                  Send
                </button>
              </div>
            </div>
          </form>
        </div>

        {/* Chat away with Hoop.io - Between Send and History */}
        <div className="flex-none px-8 py-6 border-b border-white/5">
          <h2 className="text-xl font-bold text-white mb-1">
            Chat away with <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-400">Hoop.io</span>
          </h2>
          <p className="text-xs font-medium text-slate-500 uppercase tracking-wider">AI-Powered NBA Assistant</p>
        </div>

        {/* History Section */}
        <div className="flex-1 overflow-y-auto p-8">
          <h3 className="text-xs font-bold text-slate-500 uppercase tracking-[0.2em] mb-6">History</h3>
          <div className="space-y-4">
            {messages.filter(m => m.role === 'user').slice().reverse().map((msg, i) => (
              <div key={i} className="group cursor-pointer" onClick={() => setInput(msg.content)}>
                <div className="p-4 rounded-xl bg-white/5 hover:bg-white/10 border border-white/5 transition-all duration-300 group-hover:border-blue-500/30">
                  <p className="text-sm text-slate-300 line-clamp-2 group-hover:text-white transition-colors">
                    {msg.content}
                  </p>
                  <div className="mt-3 flex items-center justify-between">
                    <span className="text-[10px] text-slate-600 font-mono">
                      {msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </span>
                    <span className="opacity-0 group-hover:opacity-100 text-blue-400 text-xs transition-opacity">
                      Re-ask →
                    </span>
                  </div>
                </div>
              </div>
            ))}
            {messages.filter(m => m.role === 'user').length === 0 && (
              <div className="text-center py-10">
                <p className="text-slate-700 text-sm">No recent queries</p>
              </div>
            )}
          </div>
        </div>

        {/* Quick Actions */}
        <div className="flex-none p-8 border-t border-white/5 bg-black/20">
          <h3 className="text-xs font-bold text-slate-500 uppercase tracking-[0.2em] mb-4">Quick Actions</h3>
          <div className="grid grid-cols-2 gap-3">
            {[
              { label: "Live Games", icon: "⏱️" },
              { label: "Standings", icon: "🏆" },
              { label: "LeBron Stats", icon: "👑" },
              { label: "Curry 3PM", icon: "👌" }
            ].map((item) => (
              <button
                key={item.label}
                onClick={() => setInput(item.label)}
                className="flex items-center gap-3 p-3 rounded-lg bg-white/5 hover:bg-white/10 border border-white/5 hover:border-purple-500/30 transition-all text-left group"
              >
                <span className="text-lg grayscale group-hover:grayscale-0 transition-all">{item.icon}</span>
                <span className="text-xs font-medium text-slate-400 group-hover:text-white transition-colors">{item.label}</span>
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChatInterface;
