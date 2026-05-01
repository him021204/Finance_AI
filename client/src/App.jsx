import React, { useState, useRef, useEffect } from 'react';
import {
  Bot,
  Settings,
  Database,
  Cpu,
  Zap,
  TrendingUp,
  PieChart,
  BarChart3,
  Lightbulb,
  Send,
  Paperclip,
  FileText,
  CheckCircle,
  AlertCircle,
  ChevronDown,
  ChevronUp,
} from 'lucide-react';
import './index.css';

const API_BASE = 'https://9047-34-10-184-233.ngrok-free.app';

function App() {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [attachedFile, setAttachedFile] = useState(null);

  // ── FIX 1: Store session_id returned by /upload ──────────────
  const [sessionId, setSessionId] = useState(null);
  const [uploadStatus, setUploadStatus] = useState(null); // 'uploading' | 'ready' | 'error'
  const [uploadedFileName, setUploadedFileName] = useState(null);

  const fileInputRef = useRef(null);
  const chatBottomRef = useRef(null);

  // Auto-scroll to latest message
  useEffect(() => {
    chatBottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  const handleSuggestionClick = (text) => setInputValue(text);

  // ── FIX 1: /upload — capture & store session_id ──────────────
  const handleFileChange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setAttachedFile(file);
    setUploadStatus('uploading');

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch(`${API_BASE}/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }

      const data = await res.json();
      // ✅ Save session_id for all subsequent /query calls
      setSessionId(data.session_id);
      setUploadedFileName(file.name);
      setUploadStatus('ready');

      // Show a system message in chat
      setMessages(prev => [...prev, {
        sender: 'system',
        text: `📄 "${file.name}" uploaded — ${data.num_chunks} chunks indexed. You can now ask questions about this report.`,
      }]);
    } catch (error) {
      console.error('Upload error:', error);
      setUploadStatus('error');
      setAttachedFile(null);
      setMessages(prev => [...prev, {
        sender: 'system',
        text: `❌ Upload failed: ${error.message}`,
        isError: true,
      }]);
    }

    // Reset file input so the same file can be re-selected if needed
    e.target.value = '';
  };

  // In App.jsx, replace handleSendMessage with this:

  const handleSendMessage = async () => {
    const question = inputValue.trim();
    if (!question) return;

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
          session_id: sessionId || '',   // ✅ empty string if no doc uploaded — backend handles it
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

  return (
    <div className="app-container">
      {/* ── Sidebar ─────────────────────────────────────────── */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <Bot size={24} />
          <span>Finance AI</span>
        </div>

        <div className="model-info-section">
          <div className="model-info-header">
            <Settings size={16} />
            <span>Model Info</span>
          </div>

          <div className="info-row">
            <span className="info-label"><Cpu size={14} /> Model</span>
            <span className="info-value">qwen-finance-7b</span>
          </div>

          <div className="info-row">
            <span className="info-label"><Database size={14} /> Provider</span>
            <span className="info-value">Hugging Face</span>
          </div>

          <div className="info-row">
            <span className="info-label"><Zap size={14} /> Type</span>
            <span className="info-value">Fine-tuned LLM</span>
          </div>

          <div className="model-desc">
            Fine-tuned finance LLM for investment insights, market analysis, and financial advisory.
          </div>
        </div>

        {/* ── Upload status badge ────────────────────────────── */}
        {uploadStatus && (
          <div className={`upload-status ${uploadStatus}`}>
            {uploadStatus === 'uploading' && <span>⏳ Indexing PDF…</span>}
            {uploadStatus === 'ready' && (
              <>
                <CheckCircle size={14} />
                <span>{uploadedFileName}</span>
              </>
            )}
            {uploadStatus === 'error' && (
              <>
                <AlertCircle size={14} />
                <span>Upload failed</span>
              </>
            )}
          </div>
        )}
      </aside>

      {/* ── Main chat area ───────────────────────────────────── */}
      <main className="main-area">
        {messages.length === 0 ? (
          <div className="hero-section">
            <div className="hero-icon"><BarChart3 size={40} /></div>
            <h1 className="hero-title">Finance AI Assistant</h1>
            <p className="hero-subtitle">
              Upload an Annual Report PDF, then ask anything about it.
            </p>

            <div className="suggestions-grid">
              {[
                { icon: <TrendingUp size={16} />, text: "What was the revenue growth this year?" },
                { icon: <PieChart size={16} />, text: "Summarise the key financial highlights." },
                { icon: <BarChart3 size={16} />, text: "What are the major risk factors mentioned?" },
                { icon: <Lightbulb size={16} />, text: "What is the company's debt-to-equity ratio?" },
              ].map(({ icon, text }) => (
                <div key={text} className="suggestion-card" onClick={() => handleSuggestionClick(text)}>
                  <span className="suggestion-icon">{icon}</span>
                  <span className="suggestion-text">{text}</span>
                </div>
              ))}
            </div>
          </div>
        ) : (
          <div className="chat-history">
            {messages.map((msg, idx) => (
              <MessageBubble key={idx} msg={msg} />
            ))}

            {isLoading && (
              <div className="chat-message bot">
                <Bot size={20} style={{ minWidth: 20 }} />
                <div className="thinking-dots">
                  Thinking<span>.</span><span>.</span><span>.</span>
                </div>
              </div>
            )}

            <div ref={chatBottomRef} />
          </div>
        )}

        {/* ── Input bar ─────────────────────────────────────── */}
        <div className="input-container">
          {attachedFile && uploadStatus === 'uploading' && (
            <div className="file-pill uploading">
              <FileText size={14} /> Indexing {attachedFile.name}…
            </div>
          )}

          <div className="input-wrapper">
            <input
              type="file"
              ref={fileInputRef}
              style={{ display: 'none' }}
              onChange={handleFileChange}
              accept=".pdf"
            />
            <label
              className="file-upload-label"
              title="Upload Annual Report PDF"
              onClick={() => fileInputRef.current.click()}
            >
              <Paperclip size={20} />
            </label>

            <input
              type="text"
              className="chat-input"
              placeholder={'Ask a finance question, or upload a report for specific analysis…'}
              value={inputValue}
              onChange={e => setInputValue(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && !isLoading && handleSendMessage()}
              // disabled={isLoading}
            />  

            <button
              className="icon-btn send-btn"
              onClick={handleSendMessage}
              disabled={isLoading || !inputValue.trim()}
            >
              <Send size={20} />
            </button>
          </div>

          <div className="footer-text">
            Powered by <span>Himanshu2124/qwen-finance-7b</span> on Hugging Face
          </div>
        </div>
      </main>
    </div>
  );
}

// ── Message bubble with collapsible sources ──────────────────────
function MessageBubble({ msg }) {
  const [showSources, setShowSources] = useState(false);

  if (msg.sender === 'system') {
    return (
      <div className={`system-message ${msg.isError ? 'error' : ''}`}>
        {msg.text}
      </div>
    );
  }

  return (
    <div className={`chat-message ${msg.sender}`}>
      {msg.sender === 'bot' && <Bot size={20} style={{ minWidth: 20 }} />}
      <div style={{ flex: 1 }}>
        <p style={{ margin: 0, whiteSpace: 'pre-wrap' }}>{msg.text}</p>

        {/* ── Source citations (collapsible) ─────────────────── */}
        {msg.sources && msg.sources.length > 0 && (
          <div className="sources-section">
            <button
              className="sources-toggle"
              onClick={() => setShowSources(v => !v)}
            >
              <FileText size={13} />
              {showSources ? 'Hide' : 'Show'} {msg.sources.length} source{msg.sources.length > 1 ? 's' : ''}
              {showSources ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
            </button>

            {showSources && (
              <div className="sources-list">
                {msg.sources.map((src, i) => (
                  <div key={i} className="source-item">
                    <span className="source-index">{i + 1}</span>
                    <p>{src}</p>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;