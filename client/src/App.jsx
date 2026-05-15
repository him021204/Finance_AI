import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import {
  MessageSquare,
  Plus,
  Send,
  Upload,
  FileText,
  ChevronDown,
  ChevronUp,
  Loader,
  X,
  CheckCircle,
  AlertCircle,
  Sparkles,
  ArrowUp,
} from 'lucide-react';
import './index.css';

const API_BASE = 'http://localhost:8000';

function App() {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState('knowledge_base');

  // ── Upload state ───────────────────────────────────────────────
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [isUploading, setIsUploading] = useState(false);

  // ── Chat history (persisted) ───────────────────────────────────
  const [chatHistory, setChatHistory] = useState(() => {
    try {
      const saved = localStorage.getItem('financeai_chat_history');
      return saved ? JSON.parse(saved) : [];
    } catch { return []; }
  });

  // ── Toast ──────────────────────────────────────────────────────
  const [toast, setToast] = useState(null);

  const fileInputRef = useRef(null);
  const chatBottomRef = useRef(null);
  const inputRef = useRef(null);

  // Persist chat history
  useEffect(() => {
    localStorage.setItem('financeai_chat_history', JSON.stringify(chatHistory));
  }, [chatHistory]);

  useEffect(() => {
    chatBottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  useEffect(() => {
    if (!toast) return;
    const timer = setTimeout(() => setToast(null), 3000);
    return () => clearTimeout(timer);
  }, [toast]);

  const showToast = (type, message) => setToast({ type, message });

  // ── Save current chat to history ───────────────────────────────
  const saveCurrentChat = () => {
    if (messages.length === 0) return;
    const firstUserMsg = messages.find(m => m.sender === 'user');
    const title = firstUserMsg
      ? firstUserMsg.text.slice(0, 40) + (firstUserMsg.text.length > 40 ? '…' : '')
      : 'New chat';
    const chat = {
      id: Date.now(),
      title,
      messages: [...messages],
      uploadedFiles: [...uploadedFiles],
      timestamp: new Date().toISOString(),
    };
    setChatHistory(prev => [chat, ...prev.slice(0, 49)]); // keep last 50
  };

  // ── New chat ───────────────────────────────────────────────────
  const handleNewChat = () => {
    saveCurrentChat();
    setMessages([]);
    setUploadedFiles([]);
    inputRef.current?.focus();
  };

  // ── Load a past chat ───────────────────────────────────────────
  const handleLoadChat = (chat) => {
    saveCurrentChat();
    setMessages(chat.messages);
    setUploadedFiles(chat.uploadedFiles || []);
  };

  // ── Delete a past chat ─────────────────────────────────────────
  const handleDeleteChat = (e, chatId) => {
    e.stopPropagation();
    setChatHistory(prev => prev.filter(c => c.id !== chatId));
  };

  // ── File upload ────────────────────────────────────────────────
  const handleFileUpload = async (e) => {
    const files = Array.from(e.target.files);
    if (!files.length) return;

    const validFiles = files.filter(f => {
      const name = f.name.toLowerCase();
      return name.endsWith('.pdf') || name.endsWith('.zip');
    });

    if (!validFiles.length) {
      showToast('error', 'Only PDF and ZIP files are supported.');
      e.target.value = '';
      return;
    }

    setIsUploading(true);
    const formData = new FormData();
    validFiles.forEach(f => formData.append('files', f));

    try {
      const res = await fetch(`${API_BASE}/upload-multiple`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }

      const data = await res.json();
      setSessionId(data.session_id);

      const newFiles = data.files_processed.map(name => ({ name }));
      setUploadedFiles(prev => [...prev, ...newFiles]);

      if (data.files_processed.length > 0) {
        showToast('success', `${data.files_processed.length} file(s) uploaded successfully!`);
      }
      if (data.files_failed.length > 0) {
        showToast('error', `${data.files_failed.length} file(s) failed to process.`);
      }
    } catch (error) {
      showToast('error', `Upload failed: ${error.message}`);
    } finally {
      setIsUploading(false);
      e.target.value = '';
    }
  };

  // ── Send message ──────────────────────────────────────────────
  const handleSendMessage = async () => {
    const question = inputValue.trim();
    if (!question || isLoading) return;

    const userMessage = { sender: 'user', text: question };
    const updatedMessages = [...messages, userMessage];
    setMessages(updatedMessages);
    setInputValue('');
    setIsLoading(true);

    try {
      const history = updatedMessages
        .filter(m => m.sender === 'user' || m.sender === 'bot')
        .map(m => ({
          role: m.sender === 'user' ? 'user' : 'assistant',
          content: m.text,
        }));

      const res = await fetch(`${API_BASE}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId || '',
          question,
          history,
        }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }

      const data = await res.json();
      setMessages(prev => [...prev, {
        sender: 'bot',
        text: data.answer || 'No response generated.',
        sources: data.sources || [],
      }]);
    } catch (error) {
      setMessages(prev => [...prev, {
        sender: 'bot',
        text: `Error: ${error.message}`,
        isError: true,
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div className="app-root">
      {/* ── Toast ─────────────────────────────────────────────── */}
      {toast && (
        <div className={`toast toast-${toast.type}`}>
          {toast.type === 'success' ? <CheckCircle size={16} /> : <AlertCircle size={16} />}
          <span>{toast.message}</span>
          <button className="toast-close" onClick={() => setToast(null)}><X size={14} /></button>
        </div>
      )}

      {/* ── Sidebar ───────────────────────────────────────────── */}
      <aside className="sidebar">
        <button className="new-chat-btn" onClick={handleNewChat}>
          <Plus size={18} />
          <span>New chat</span>
        </button>

        {/* Upload button */}
        <input
          type="file"
          ref={fileInputRef}
          style={{ display: 'none' }}
          onChange={handleFileUpload}
          accept=".pdf,.zip"
          multiple
        />
        <button
          className="upload-btn"
          onClick={() => fileInputRef.current.click()}
          disabled={isUploading}
        >
          {isUploading ? (
            <><Loader size={16} className="spin" /> Processing…</>
          ) : (
            <><Upload size={16} /> Upload reports</>
          )}
        </button>

        {/* Uploaded files */}
        {uploadedFiles.length > 0 && (
          <div className="uploaded-section">
            <span className="uploaded-label">Uploads</span>
            {uploadedFiles.map((f, i) => (
              <div key={i} className="uploaded-item">
                <FileText size={13} />
                <span title={f.name}>{f.name}</span>
              </div>
            ))}
          </div>
        )}

        {/* Chat history */}
        <div className="chat-history-section">
          {chatHistory.length > 0 && (
            <span className="history-label">Recent</span>
          )}
          {chatHistory.map(chat => (
            <div
              key={chat.id}
              className="history-item"
              onClick={() => handleLoadChat(chat)}
            >
              <MessageSquare size={14} />
              <span className="history-title">{chat.title}</span>
              <button
                className="history-delete"
                onClick={(e) => handleDeleteChat(e, chat.id)}
              >
                <X size={12} />
              </button>
            </div>
          ))}
        </div>
      </aside>

      {/* ── Main ──────────────────────────────────────────────── */}
      <main className="main">
        {messages.length === 0 ? (
          <div className="welcome">
            <div className="welcome-icon">
              <Sparkles size={36} />
            </div>
            <h1>What can I help with?</h1>
            <div className="suggestions">
              {[
                "What was the revenue growth this year?",
                "Summarise the key financial highlights.",
                "What are the major risk factors?",
                "What is the debt-to-equity ratio?",
              ].map(text => (
                <button
                  key={text}
                  className="suggestion-chip"
                  onClick={() => setInputValue(text)}
                >
                  {text}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="messages">
            {messages.map((msg, idx) => (
              <MessageBubble key={idx} msg={msg} />
            ))}

            {isLoading && (
              <div className="msg msg-bot">
                <div className="msg-avatar bot-avatar">
                  <Sparkles size={18} />
                </div>
                <div className="msg-content">
                  <div className="typing-indicator">
                    <span></span><span></span><span></span>
                  </div>
                </div>
              </div>
            )}

            <div ref={chatBottomRef} />
          </div>
        )}

        {/* ── Input area ──────────────────────────────────────── */}
        <div className="input-area">
          <div className="input-box">
            <textarea
              ref={inputRef}
              className="chat-input"
              placeholder="Message Finance AI…"
              value={inputValue}
              onChange={e => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
              rows={1}
              onInput={e => {
                e.target.style.height = 'auto';
                e.target.style.height = Math.min(e.target.scrollHeight, 200) + 'px';
              }}
            />
            <button
              className="send-btn"
              onClick={handleSendMessage}
              disabled={isLoading || !inputValue.trim()}
            >
              <ArrowUp size={18} />
            </button>
          </div>
          <p className="disclaimer">Finance AI can make mistakes. Verify important information.</p>
        </div>
      </main>
    </div>
  );
}


// ── Message bubble ───────────────────────────────────────────────
function MessageBubble({ msg }) {
  const [showSources, setShowSources] = useState(false);

  if (msg.sender === 'system') {
    return <div className={`system-msg ${msg.isError ? 'error' : ''}`}>{msg.text}</div>;
  }

  const isUser = msg.sender === 'user';

  return (
    <div className={`msg ${isUser ? 'msg-user' : 'msg-bot'}`}>
      {!isUser && (
        <div className="msg-avatar bot-avatar">
          <Sparkles size={18} />
        </div>
      )}
      <div className="msg-content">
        {isUser ? (
          <p>{msg.text}</p>
        ) : (
          <div className="markdown-body">
            <ReactMarkdown>{msg.text}</ReactMarkdown>
          </div>
        )}

        {msg.sources && msg.sources.length > 0 && (
          <div className="sources">
            <button className="sources-btn" onClick={() => setShowSources(v => !v)}>
              <FileText size={12} />
              {showSources ? 'Hide' : 'Show'} {msg.sources.length} source{msg.sources.length > 1 ? 's' : ''}
              {showSources ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
            </button>
            {showSources && (
              <div className="sources-list">
                {msg.sources.map((src, i) => (
                  <div key={i} className="source-chip">
                    <span className="source-num">{i + 1}</span>
                    <p>{src}</p>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
      {isUser && (
        <div className="msg-avatar user-avatar">
          <span>Y</span>
        </div>
      )}
    </div>
  );
}

export default App;