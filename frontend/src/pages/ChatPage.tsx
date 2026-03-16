import { useState, useRef, useEffect, type FormEvent, type KeyboardEvent } from 'react';
import { useAuth } from '../auth-context';
import { chat as chatApi, user as userApi, ApiError } from '../api';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

function generateId() {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

function generateSessionId() {
  return `session-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

export default function ChatPage() {
  const { user, logout } = useAuth();
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [sending, setSending] = useState(false);
  const [sessionId] = useState(generateSessionId);
  const [error, setError] = useState('');
  const [deleting, setDeleting] = useState(false);
  const [showImportModal, setShowImportModal] = useState(false);
  const [importUrl, setImportUrl] = useState('');
  const [importing, setImporting] = useState(false);
  const [importResult, setImportResult] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 160)}px`;
    }
  }, [input]);

  async function handleSend(e?: FormEvent) {
    e?.preventDefault();
    const trimmed = input.trim();
    if (!trimmed || sending) return;

    setError('');
    const userMsg: Message = {
      id: generateId(),
      role: 'user',
      content: trimmed,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMsg]);
    setInput('');
    setSending(true);

    try {
      const response = await chatApi.send(sessionId, trimmed);
      const assistantMsg: Message = {
        id: generateId(),
        role: 'assistant',
        content: response.assistant_message,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err) {
      const errMsg = err instanceof ApiError ? err.message : 'Failed to get response';
      setError(errMsg);
    } finally {
      setSending(false);
      textareaRef.current?.focus();
    }
  }

  function handleKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  async function handleClearMemories() {
    if (!confirm('⚠️ This will delete ALL your memories, entities, and conversation history. This cannot be undone. Continue?')) {
      return;
    }

    setDeleting(true);
    setError('');

    try {
      await userApi.deleteAllMemories();
      setMessages([]);
      alert('✅ All memories cleared successfully');
    } catch (err) {
      const errMsg = err instanceof ApiError ? err.message : 'Failed to clear memories';
      setError(errMsg);
    } finally {
      setDeleting(false);
    }
  }

  async function handleImportMemories() {
    if (!importUrl.trim()) {
      setImportResult('Please enter a valid URL');
      return;
    }

    setImporting(true);
    setImportResult(null);
    setError('');

    try {
      const result = await userApi.ingestMemories({
        source_type: 'chatgpt',
        source: importUrl.trim(),
      });

      if (result.status === 'success') {
        setImportResult(
          `✅ Successfully imported ${result.memories_created} memories from ${result.chunks_parsed} conversation chunks!`
        );
        setImportUrl('');
        
        if (result.errors.length > 0) {
          setImportResult(prev => 
            `${prev}\n\n⚠️ ${result.errors.length} errors occurred:\n${result.errors.slice(0, 3).join('\n')}`
          );
        }
      } else {
        setImportResult(`❌ Import failed: ${result.errors.join(', ')}`);
      }
    } catch (err) {
      const errMsg = err instanceof ApiError ? err.message : 'Failed to import memories';
      setImportResult(`❌ ${errMsg}`);
    } finally {
      setImporting(false);
    }
  }

  function closeImportModal() {
    setShowImportModal(false);
    setImportUrl('');
    setImportResult(null);
  }

  return (
    <div className="h-full flex flex-col bg-surface-950">
      {/* Header */}
      <header className="flex-none border-b border-surface-800 bg-surface-900/80 backdrop-blur-md">
        <div className="max-w-4xl mx-auto w-full flex items-center justify-between px-6 py-3">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-primary-600/20 flex items-center justify-center">
              <span className="text-lg font-bold text-primary-400">D</span>
            </div>
            <div>
              <h1 className="text-white font-semibold text-sm leading-tight">DAPPY</h1>
              <p className="text-surface-400 text-xs">Cognitive Memory</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <span className="text-surface-400 text-sm">
              {user?.display_name || user?.username}
            </span>
            <button
              onClick={() => setShowImportModal(true)}
              className="px-3 py-1.5 text-xs font-medium text-blue-400 hover:text-blue-300 bg-surface-800 hover:bg-blue-900/30 rounded-lg transition-colors"
              title="Import memories from ChatGPT"
            >
              Import Memories
            </button>
            <button
              onClick={handleClearMemories}
              disabled={deleting}
              className="px-3 py-1.5 text-xs font-medium text-red-400 hover:text-red-300 bg-surface-800 hover:bg-red-900/30 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              title="Clear all memories"
            >
              {deleting ? 'Clearing...' : 'Clear Memories'}
            </button>
            <button
              onClick={logout}
              className="px-3 py-1.5 text-xs font-medium text-surface-400 hover:text-white bg-surface-800 hover:bg-surface-700 rounded-lg transition-colors"
            >
              Sign Out
            </button>
          </div>
        </div>
      </header>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto">
        <div className="max-w-4xl mx-auto w-full px-6 py-6">
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center h-full min-h-[60vh] text-center">
              <div className="w-20 h-20 rounded-2xl bg-primary-600/10 flex items-center justify-center mb-6">
                <span className="text-4xl font-bold text-primary-500/60">D</span>
              </div>
              <h2 className="text-xl font-semibold text-white mb-2">Ready to chat!</h2>
              <p className="text-surface-400 max-w-sm text-sm leading-relaxed">
                Tell me anything you'd like me to remember, or ask me questions.
                I'll help you keep track of important information.
              </p>
            </div>
          )}

          {messages.map((msg) => (
            <div
              key={msg.id}
              className={`flex mb-4 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`max-w-[80%] rounded-2xl px-5 py-3 ${
                  msg.role === 'user'
                    ? 'bg-primary-600 text-white rounded-br-md'
                    : 'bg-surface-800 text-surface-100 border border-surface-700/50 rounded-bl-md'
                }`}
              >
                <p className="text-sm leading-relaxed whitespace-pre-wrap">{msg.content}</p>
                <p
                  className={`text-[10px] mt-1.5 ${
                    msg.role === 'user' ? 'text-primary-200/60' : 'text-surface-500'
                  }`}
                >
                  {msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </p>
              </div>
            </div>
          ))}

          {sending && (
            <div className="flex justify-start mb-4">
              <div className="bg-surface-800 border border-surface-700/50 rounded-2xl rounded-bl-md px-5 py-3">
                <div className="flex gap-1.5">
                  <span className="w-2 h-2 bg-surface-500 rounded-full animate-bounce [animation-delay:0ms]" />
                  <span className="w-2 h-2 bg-surface-500 rounded-full animate-bounce [animation-delay:150ms]" />
                  <span className="w-2 h-2 bg-surface-500 rounded-full animate-bounce [animation-delay:300ms]" />
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Error banner */}
      {error && (
        <div className="flex-none border-t border-red-500/20 bg-red-500/5 px-6 py-2">
          <p className="max-w-4xl mx-auto text-red-400 text-sm">{error}</p>
        </div>
      )}

      {/* Input */}
      <div className="flex-none border-t border-surface-800 bg-surface-900/80 backdrop-blur-md">
        <form
          onSubmit={handleSend}
          className="max-w-4xl mx-auto w-full px-6 py-4 flex items-end gap-3"
        >
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            rows={1}
            placeholder="Type a message..."
            disabled={sending}
            className="flex-1 resize-none rounded-xl bg-surface-800 border border-surface-700/50 text-white placeholder-surface-500 px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500/40 focus:border-primary-500/40 disabled:opacity-50 transition-all"
          />
          <button
            type="submit"
            disabled={!input.trim() || sending}
            className="flex-none w-11 h-11 rounded-xl bg-primary-600 hover:bg-primary-500 disabled:bg-surface-700 text-white flex items-center justify-center transition-colors disabled:cursor-not-allowed"
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 12L3.269 3.126A59.768 59.768 0 0121.485 12 59.77 59.77 0 013.27 20.876L5.999 12zm0 0h7.5" />
            </svg>
          </button>
        </form>
      </div>

      {showImportModal && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4 z-50">
          <div className="bg-surface-900 border border-surface-700 rounded-2xl shadow-2xl max-w-lg w-full p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold text-white">Import Memories</h2>
              <button
                onClick={closeImportModal}
                className="text-surface-400 hover:text-white transition-colors"
              >
                <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            <p className="text-surface-400 text-sm mb-4">
              Paste a ChatGPT shared conversation link to import memories from it.
            </p>

            <input
              type="text"
              value={importUrl}
              onChange={(e) => setImportUrl(e.target.value)}
              placeholder="https://chatgpt.com/share/..."
              disabled={importing}
              className="w-full px-4 py-3 rounded-lg bg-surface-800 border border-surface-700/50 text-white placeholder-surface-500 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500/40 focus:border-primary-500/40 disabled:opacity-50 mb-4"
            />

            {importResult && (
              <div className={`mb-4 p-3 rounded-lg text-sm whitespace-pre-wrap ${
                importResult.startsWith('✅') 
                  ? 'bg-green-900/20 border border-green-700/30 text-green-300'
                  : 'bg-red-900/20 border border-red-700/30 text-red-300'
              }`}>
                {importResult}
              </div>
            )}

            <div className="flex gap-3">
              <button
                onClick={closeImportModal}
                disabled={importing}
                className="flex-1 px-4 py-2.5 rounded-lg bg-surface-800 hover:bg-surface-700 text-surface-300 font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Cancel
              </button>
              <button
                onClick={handleImportMemories}
                disabled={!importUrl.trim() || importing}
                className="flex-1 px-4 py-2.5 rounded-lg bg-primary-600 hover:bg-primary-500 text-white font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {importing ? 'Importing...' : 'Import'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
