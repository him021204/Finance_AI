import React, { useState, useRef } from 'react';
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
  FileText
} from 'lucide-react';
import './index.css';

function App() {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [attachedFile, setAttachedFile] = useState(null);
  const fileInputRef = useRef(null);

  const handleSuggestionClick = (text) => {
    setInputValue(text);
  };

  const handleFileChange = async (e) => {
    const file = e.target.files[0];
    if (file) {
      setAttachedFile(file);
      // Automatically upload file when selected
      const formData = new FormData();
      formData.append('file', file);
      
      try {
        const response = await fetch('http://localhost:5000/api/upload', {
          method: 'POST',
          body: formData,
        });
        if (response.ok) {
          console.log('File uploaded successfully');
        }
      } catch (error) {
        console.error('Error uploading file:', error);
      }
    }
  };

  const handleSendMessage = async () => {
    if (!inputValue.trim() && !attachedFile) return;

    const userMessage = { text: inputValue, sender: 'user', file: attachedFile?.name };
    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setAttachedFile(null);
    setIsLoading(true);

    try {
      const response = await fetch('http://localhost:5000/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userMessage.text })
      });
      const data = await response.json();
      
      setMessages(prev => [...prev, { text: data.response || "No response generated.", sender: 'bot' }]);
    } catch (error) {
      console.error('Error chatting:', error);
      setMessages(prev => [...prev, { text: "Error connecting to the backend server. Make sure it is running.", sender: 'bot' }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-container">
      {/* Sidebar */}
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
            <span className="info-label">
              <Cpu size={14} /> Model
            </span>
            <span className="info-value">qwen-finance-7b</span>
          </div>
          
          <div className="info-row">
            <span className="info-label">
              <Database size={14} /> Provider
            </span>
            <span className="info-value">Hugging Face</span>
          </div>
          
          <div className="info-row">
            <span className="info-label">
              <Zap size={14} /> Type
            </span>
            <span className="info-value">Fine-tuned LLM</span>
          </div>

          <div className="model-desc">
            Fine-tuned finance LLM for investment insights, market analysis, and financial advisory.
          </div>
        </div>
      </aside>

      {/* Main Chat Area */}
      <main className="main-area">
        {messages.length === 0 ? (
          <div className="hero-section">
            <div className="hero-icon">
              <BarChart3 size={40} />
            </div>
            <h1 className="hero-title">Finance AI Assistant</h1>
            <p className="hero-subtitle">
              Ask questions about financial markets, investment strategies, portfolio management, and more.
            </p>

            <div className="suggestions-grid">
              <div className="suggestion-card" onClick={() => handleSuggestionClick("What factors affect stock market volatility?")}>
                <TrendingUp size={16} className="suggestion-icon" />
                <span className="suggestion-text">What factors affect stock market volatility?</span>
              </div>
              <div className="suggestion-card" onClick={() => handleSuggestionClick("Explain the concept of portfolio diversification")}>
                <PieChart size={16} className="suggestion-icon" />
                <span className="suggestion-text">Explain the concept of portfolio diversification</span>
              </div>
              <div className="suggestion-card" onClick={() => handleSuggestionClick("How do interest rate changes impact bond prices?")}>
                <BarChart3 size={16} className="suggestion-icon" />
                <span className="suggestion-text">How do interest rate changes impact bond prices?</span>
              </div>
              <div className="suggestion-card" onClick={() => handleSuggestionClick("What is the difference between ETFs and mutual funds?")}>
                <Lightbulb size={16} className="suggestion-icon" />
                <span className="suggestion-text">What is the difference between ETFs and mutual funds?</span>
              </div>
            </div>
          </div>
        ) : (
          <div className="chat-history">
            {messages.map((msg, idx) => (
              <div key={idx} className={`chat-message ${msg.sender}`}>
                {msg.sender === 'bot' && <Bot size={20} style={{minWidth: '20px'}} />}
                <div>
                  {msg.file && (
                    <div style={{fontSize: '0.8rem', color: '#8b949e', display: 'flex', alignItems: 'center', gap: '4px', marginBottom: '4px'}}>
                      <FileText size={12}/> Attached file: {msg.file}
                    </div>
                  )}
                  {msg.text}
                </div>
              </div>
            ))}
            {isLoading && (
              <div className="chat-message bot">
                <Bot size={20} />
                <div style={{display: 'flex', gap: '4px', alignItems: 'center'}}>
                  Thinking<span style={{animation: 'pulse 1s infinite'}}>...</span>
                </div>
              </div>
            )}
          </div>
        )}

        <div className="input-container">
          {attachedFile && (
             <div style={{position: 'absolute', top: '-10px', left: '2rem', background: '#238636', padding: '4px 12px', borderRadius: '12px', fontSize: '0.8rem', display: 'flex', alignItems: 'center', gap: '8px'}}>
               <FileText size={14}/> {attachedFile.name}
               <span style={{cursor: 'pointer'}} onClick={() => setAttachedFile(null)}>×</span>
             </div>
          )}
          <div className="input-wrapper">
            <input 
              type="file" 
              ref={fileInputRef} 
              style={{display: 'none'}} 
              onChange={handleFileChange}
              accept=".pdf,.txt,.csv"
            />
            <label className="file-upload-label" onClick={() => fileInputRef.current.click()}>
              <Paperclip size={20} />
            </label>
            <input 
              type="text" 
              className="chat-input"
              placeholder="Ask a finance question..." 
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSendMessage()}
            />
            <button className="icon-btn send-btn" onClick={handleSendMessage}>
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

export default App;
