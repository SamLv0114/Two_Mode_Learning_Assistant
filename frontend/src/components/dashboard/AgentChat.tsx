'use client';

import { useEffect, useRef, useState } from 'react';
import dynamic from 'next/dynamic';
import { qaApi, chatApi, AgentEvent, errorMessage, readEventStream } from '@/lib/api';
import toast from 'react-hot-toast';
import { BookOpen, X, RefreshCw, Upload, Trash2, ChevronDown, ChevronUp, Bot, Send } from 'lucide-react';
import { ChatContext } from './shared';

// The Markdown renderer (react-markdown + remark) is ~45 kB; keep it out of the
// dashboard's first load. AgentChat preloads it on mount, before any answer arrives.
const loadMarkdown = () => import('../Markdown').then((module) => module.Markdown);
const Markdown = dynamic(loadMarkdown, { ssr: false });

const INTENT_COLORS: Record<string, string> = {
  research_qa: 'bg-blue-100 text-blue-700',
  recommendation: 'bg-green-100 text-green-700',
  document_management: 'bg-amber-100 text-amber-700',
  general_chat: 'bg-gray-100 text-gray-600',
};

const INTENT_LABELS: Record<string, string> = {
  research_qa: 'Research Q&A',
  recommendation: 'Recommendations',
  document_management: 'Documents',
  general_chat: 'General Chat',
};

const TOOL_LABELS: Record<string, string> = {
  search_knowledge_base: 'Search KB',
  get_personalized_feed: 'Get Feed',
  list_user_documents: 'List Docs',
  search_user_documents: 'Search Docs',
};

const TOOL_ICONS: Record<string, string> = {
  search_knowledge_base: '🔍',
  get_personalized_feed: '📰',
  list_user_documents: '📁',
  search_user_documents: '🔎',
};

export interface ResearchTask {
  id: number;
  title: string;
  intent: string;
  status: 'pending' | 'running' | 'done';
  citations?: number;
}
export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  intent?: string;
  intentMethod?: string;
  intentConfidence?: number;
  agentUsed?: string;
  toolCalls?: { tool: string; count: number }[];
  citations?: { title: string; url: string; type: string }[];
  isStreaming?: boolean;
  researchPlan?: ResearchTask[];
  researchGaps?: string[];
  researchRunId?: string;
  researchComplete?: boolean;
  reflectionStage?: 'critiquing' | 'refining';
  reflectionScore?: number;
  isDraft?: boolean;
}

const CHAT_SESSION_STORAGE_KEY = 'chat_session_id';

/* ── Agent Chat ── */
export function AgentChat({
  chatContext = null,
  onClearContext,
  onPractice,
}: {
  chatContext?: ChatContext | null;
  onClearContext?: () => void;
  onPractice?: (context: ChatContext) => void;
}) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(() =>
    typeof window !== 'undefined' ? localStorage.getItem(CHAT_SESSION_STORAGE_KEY) : null
  );
  const [showKB, setShowKB] = useState(false);
  const [pausedRun, setPausedRun] = useState<{ run_id: string; question: string } | null>(null);
  const [kbDocs, setKbDocs] = useState<any[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const contextInjected = useRef(false);

  useEffect(() => { loadMarkdown(); }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    loadKbDocs();
    chatApi.getResearchRuns().then(({ runs }) => {
      const paused = runs.find(run => run.resumable);
      if (paused) setPausedRun(paused);
    }).catch(() => {});
    return () => {
      abortRef.current?.abort();
    };
  }, []);

  // Restore the previous conversation on mount (page refresh, or switching
  // back to this tab after AgentChat was unmounted) using whatever
  // session_id survived in localStorage. The backend keeps history for this
  // session_id in Redis for 24h regardless of whether the frontend remembers
  // it — this just gives the frontend a way to ask for it back.
  useEffect(() => {
    if (!sessionId) return;
    chatApi.getHistory(sessionId)
      .then((data) => {
        if (!data.messages || data.messages.length === 0) return;
        setMessages(
          data.messages.map((m, i) => ({
            id: `restored-${i}`,
            role: m.role,
            content: m.content,
          }))
        );
      })
      .catch(() => {
        // Session expired in Redis (past the 24h TTL) or otherwise
        // unreachable — nothing to restore, start fresh silently.
        localStorage.removeItem(CHAT_SESSION_STORAGE_KEY);
        setSessionId(null);
      });
    // Intentionally only on mount — this restores once, not on every
    // sessionId change (a live chat already has its messages in state).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Keep localStorage in sync whenever the active session changes, so a
  // page refresh or a return trip to this tab always has the latest id.
  useEffect(() => {
    if (sessionId) {
      localStorage.setItem(CHAT_SESSION_STORAGE_KEY, sessionId);
    } else {
      localStorage.removeItem(CHAT_SESSION_STORAGE_KEY);
    }
  }, [sessionId]);

  useEffect(() => {
    contextInjected.current = false;
  }, [chatContext]);

  const loadKbDocs = async () => {
    try {
      const data = await qaApi.listDocuments();
      setKbDocs(data.documents || data || []);
    } catch {}
  };

  const handleKbUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;
    setIsUploading(true);
    let uploaded = 0;
    for (let i = 0; i < files.length; i++) {
      try {
        await qaApi.uploadDocument(files[i], files[i].name);
        uploaded++;
      } catch (err) {
        toast.error(`Failed to upload ${files[i].name}: ${errorMessage(err, 'Unknown error')}`);
      }
    }
    if (uploaded > 0) {
      toast.success(`Uploaded ${uploaded} document(s) to knowledge base`);
      loadKbDocs();
    }
    setIsUploading(false);
    e.target.value = '';
  };

  const handleKbDelete = async (docId: number, title: string) => {
    try {
      await qaApi.deleteDocument(docId);
      toast.success(`Deleted "${title}"`);
      loadKbDocs();
    } catch {
      toast.error('Failed to delete document');
    }
  };

  const sendMessage = async (resumeRunId?: string) => {
    if ((!input.trim() && !resumeRunId) || isStreaming) return;

    const userMsg: ChatMessage = { id: `u-${Date.now()}`, role: 'user', content: resumeRunId ? 'Resume research' : input };
    const assistantId = `a-${Date.now() + 1}`;
    const assistantMsg: ChatMessage = {
      id: assistantId,
      role: 'assistant',
      content: '',
      isStreaming: true,
      toolCalls: [],
    };

    setMessages(prev => [...prev, userMsg, assistantMsg]);
    let text = input;
    if (chatContext && !contextInjected.current) {
      text = `I want to ask about this research paper:\n\nTitle: ${chatContext.title}\narXiv: ${chatContext.arxivId || 'Unknown'}\nURL: ${chatContext.arxivUrl}\n\nAbstract: ${chatContext.abstract?.slice(0, 800) || 'Not available'}\n\n---\n\nMy question: ${input}`;
      contextInjected.current = true;
    }
    setInput('');
    if (resumeRunId) setPausedRun(null);
    setIsStreaming(true);

    abortRef.current = new AbortController();

    try {
      const applyEvent = (event: AgentEvent) => {
        if (event.type === 'session') {
          setSessionId(event.session_id);
          return;
        }
        setMessages(prev =>
          prev.map(m => {
            if (m.id !== assistantId) return m;
            switch (event.type) {
              case 'run':
                return { ...m, researchRunId: event.run_id };
              case 'intent':
                return { ...m, intent: event.value, intentMethod: event.method, intentConfidence: event.confidence };
              case 'agent':
                return { ...m, agentUsed: event.value };
              case 'tool_call':
                return { ...m, toolCalls: [...(m.toolCalls || []), { tool: event.tool, count: -1 }] };
              case 'tool_result': {
                const updated = (m.toolCalls || []).map(tc =>
                  tc.tool === event.tool && tc.count === -1 ? { ...tc, count: event.count } : tc
                );
                return { ...m, toolCalls: updated };
              }
              case 'token':
                return { ...m, content: m.content + event.value };
              case 'draft_token':
                return { ...m, content: m.content + event.value, isDraft: true };
              case 'replace':
                return { ...m, content: event.value, isDraft: false };
              case 'done':
                return { ...m, isStreaming: false, citations: event.citations,
                  researchComplete: Boolean(m.researchRunId || event.run_id) && !event.resumable, isDraft: false };
              case 'error':
                return { ...m, isStreaming: false, content: m.content || `Error: ${event.value}` };
              case 'cancelled':
                return { ...m, isStreaming: false, content: `${m.content}\n\n[Generation stopped]`.trim() };
              case 'plan': {
                // A replan re-sends the whole plan; keep the progress of tasks already shown.
                const previous = new Map((m.researchPlan || []).map(t => [t.id, t]));
                return {
                  ...m,
                  researchPlan: event.tasks.map(t => {
                    const known = previous.get(t.id);
                    return { ...t, status: known?.status ?? ('pending' as const), citations: known?.citations };
                  }),
                };
              }
              case 'gaps':
                return { ...m, researchGaps: event.gaps.map(g => g.title) };
              case 'task_started': {
                const updated = (m.researchPlan || []).map(t =>
                  t.id === event.id ? { ...t, status: 'running' as const } : t
                );
                return { ...m, researchPlan: updated };
              }
              case 'task_done': {
                const updated = (m.researchPlan || []).map(t =>
                  t.id === event.id ? { ...t, status: 'done' as const, citations: event.citations } : t
                );
                return { ...m, researchPlan: updated };
              }
              case 'critiquing':
                return { ...m, reflectionStage: 'critiquing' };
              case 'critique_result':
                return { ...m, reflectionScore: event.aggregate };
              case 'refining':
                return { ...m, reflectionStage: 'refining' };
              default:
                return m;
            }
          })
        );
      };

      await readEventStream<AgentEvent>('/chat/stream', {
        method: 'POST',
        body: { message: resumeRunId ? '' : text, session_id: sessionId, resume_run_id: resumeRunId },
        signal: abortRef.current.signal,
      }, applyEvent);
    } catch (err) {
      if ((err as Error).name !== 'AbortError') {
        setMessages(prev =>
          prev.map(m =>
            m.id === assistantId
              ? { ...m, isStreaming: false, content: m.content || 'Failed to get a response. Please try again.' }
              : m
          )
        );
      }
    } finally {
      setIsStreaming(false);
      setMessages(prev =>
        prev.map(m => m.id === assistantId ? { ...m, isStreaming: false } : m)
      );
    }
  };

  const stopGeneration = () => {
    abortRef.current?.abort();
    setMessages(prev => prev.map(m =>
      m.isStreaming ? { ...m, isStreaming: false, content: `${m.content}\n\n[Generation stopped]`.trim() } : m
    ));
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const clearChat = () => {
    abortRef.current?.abort();
    setMessages([]);
    setSessionId(null);
    setIsStreaming(false);
  };

  return (
    <div className="flex flex-col" style={{ height: 'calc(100vh - 320px)', minHeight: 480 }}>
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-lg font-semibold text-gray-900 flex items-center gap-2">
            <Bot className="w-5 h-5 text-primary-600" />
            Agent Chat
          </h2>
          <p className="text-xs text-gray-400 mt-0.5">
            Powered by multi-agent routing — ask research questions, get recommendations, or manage documents
          </p>
        </div>
        <button
          onClick={clearChat}
          className="text-xs text-gray-400 hover:text-gray-600 flex items-center gap-1 px-2 py-1 rounded hover:bg-gray-100"
        >
          <X className="w-3.5 h-3.5" />
          New Chat
        </button>
      </div>

      {pausedRun && (
        <div className="mb-3 flex items-center justify-between gap-3 rounded-lg border border-indigo-200 bg-indigo-50 px-3 py-2 text-xs">
          <span className="truncate text-indigo-800">Unfinished research: {pausedRun.question}</span>
          <button type="button" disabled={isStreaming} onClick={() => sendMessage(pausedRun.run_id)}
            className="shrink-0 font-medium text-indigo-700 underline disabled:opacity-50">Resume</button>
        </div>
      )}

      {/* Selected discovery item */}
      {chatContext && (
        <div className="shrink-0 mb-3 flex items-start gap-3 bg-sky-50 border border-sky-200 rounded-xl px-4 py-3">
          <BookOpen className="w-4 h-4 text-sky-600 shrink-0 mt-0.5" />
          <div className="flex-1 min-w-0">
            <p className="text-xs font-semibold text-sky-700 mb-0.5">Discussing paper{chatContext.origin ? ` · from ${chatContext.origin}` : ''}</p>
            <p className="text-sm text-sky-900 font-medium leading-snug line-clamp-2">{chatContext.title}</p>
            {chatContext.arxivUrl && (
              <a href={chatContext.arxivUrl} target="_blank" rel="noopener noreferrer" className="text-xs text-sky-600 hover:underline">
                arXiv:{chatContext.arxivId || 'source'} →
              </a>
            )}
            {onPractice && chatContext.arxivId && <button type="button" onClick={() => onPractice(chatContext)} className="ml-3 text-xs font-medium text-sky-700 underline">Build algorithm →</button>}
          </div>
          <button
            onClick={onClearContext}
            className="text-sky-400 hover:text-sky-600 p-1 shrink-0 transition-colors"
            title="Clear selected context"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      )}

      {/* Knowledge Base Panel */}
      <div className="shrink-0 mb-3 border border-gray-200 rounded-xl overflow-hidden">
        <button
          onClick={() => setShowKB(!showKB)}
          className="w-full flex items-center justify-between px-4 py-2.5 bg-gray-50 hover:bg-gray-100 text-sm font-medium text-gray-700 transition-colors"
        >
          <div className="flex items-center gap-2">
            <Upload className="w-4 h-4 text-gray-500" />
            Knowledge Base
            {kbDocs.length > 0 && (
              <span className="text-xs bg-primary-100 text-primary-700 px-2 py-0.5 rounded-full font-medium">
                {kbDocs.length} doc{kbDocs.length !== 1 ? 's' : ''}
              </span>
            )}
          </div>
          {showKB ? <ChevronUp className="w-4 h-4 text-gray-400" /> : <ChevronDown className="w-4 h-4 text-gray-400" />}
        </button>
        {showKB && (
          <div className="p-4 space-y-3 bg-white">
            <label className="inline-flex items-center gap-2 btn-secondary text-sm cursor-pointer">
              <Upload className="w-4 h-4" />
              {isUploading ? 'Uploading...' : 'Upload Files (.txt, .md, .pdf)'}
              <input
                type="file"
                accept=".txt,.md,.pdf"
                multiple
                onChange={handleKbUpload}
                disabled={isUploading}
                className="hidden"
              />
            </label>
            {kbDocs.length > 0 ? (
              <div className="space-y-1.5 max-h-36 overflow-y-auto">
                {kbDocs.map((doc: any) => (
                  <div key={doc.id} className="flex items-center justify-between bg-gray-50 rounded-lg px-3 py-1.5">
                    <div className="min-w-0">
                      <span className="text-sm text-gray-700 truncate block">{doc.title}</span>
                      {doc.chunk_count > 0 && (
                        <span className="text-xs text-gray-400">{doc.chunk_count} chunks</span>
                      )}
                    </div>
                    <button
                      onClick={() => handleKbDelete(doc.id, doc.title)}
                      className="text-gray-300 hover:text-rose-400 p-1 ml-2 shrink-0 transition-colors"
                      title="Delete document"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-xs text-gray-400">No documents yet. Uploaded files will be searchable in Agent Chat.</p>
            )}
          </div>
        )}
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto space-y-4 pr-1">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-center py-12">
            <Bot className="w-12 h-12 text-gray-200 mb-3" />
            <p className="text-gray-500 font-medium">Ask me anything</p>
            <p className="text-sm text-gray-400 mt-1 max-w-sm">
              Research questions, paper recommendations, or searches across your uploaded documents
            </p>
            <div className="mt-4 flex flex-wrap gap-2 justify-center">
              {/* Only questions the chat agents can ground: the knowledge base, a paper's full
                  text, the web, your feed and your uploads. */}
              {(chatContext ? [
                "What is this paper's main contribution?",
                'Which details of the method are hardest to reproduce?',
                'Find related papers in the knowledge base',
              ] : [
                'What are recent advances in retrieval-augmented generation?',
                'Compare LoRA and QLoRA for fine-tuning LLMs',
                'What should I read next?',
                'List my uploaded documents',
              ]).map(ex => (
                <button
                  key={ex}
                  onClick={() => setInput(ex)}
                  className="text-xs bg-gray-50 border border-gray-200 rounded-full px-3 py-1.5 text-gray-600 hover:bg-primary-50 hover:border-primary-200 hover:text-primary-700 transition-colors"
                >
                  {ex}
                </button>
              ))}
            </div>
          </div>
        )}
        {messages.map(msg => (
          <ChatBubble key={msg.id} message={msg} onResume={(runId) => sendMessage(runId)} canResume={!isStreaming} />
        ))}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="flex gap-2 mt-4 pt-4 border-t">
        <input
          type="text"
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask a question..."
          className="input-field flex-1"
          disabled={isStreaming}
        />
        <button
          onClick={() => sendMessage()}
          disabled={isStreaming || !input.trim()}
          className="btn-primary flex items-center gap-2 disabled:opacity-50 shrink-0"
        >
          {isStreaming ? (
            <RefreshCw className="w-4 h-4 animate-spin" />
          ) : (
            <Send className="w-4 h-4" />
          )}
          Send
        </button>
        {isStreaming && (
          <button
            onClick={stopGeneration}
            className="px-3 py-2 rounded border border-gray-300 text-sm text-gray-700 hover:bg-gray-100"
            type="button"
          >
            Stop
          </button>
        )}
      </div>
    </div>
  );
}

/* ── Chat Bubble ── */
export function ChatBubble({ message, onResume, canResume }: {
  message: ChatMessage;
  onResume: (runId: string) => void;
  canResume: boolean;
}) {
  if (message.role === 'user') {
    return (
      <div className="flex justify-end">
        <div className="bg-primary-600 text-white rounded-2xl rounded-tr-sm px-4 py-3 max-w-[80%] text-sm leading-relaxed">
          {message.content}
        </div>
      </div>
    );
  }

  return (
    <div className="flex justify-start">
      <div className="flex-1 max-w-[88%] space-y-2">
        {/* Intent + agent badges */}
        {(message.intent || message.agentUsed) && (
          <div className="flex flex-wrap items-center gap-1.5">
            {message.intent && (
              <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${INTENT_COLORS[message.intent] || 'bg-gray-100 text-gray-600'}`}>
                {INTENT_LABELS[message.intent] || message.intent}
              </span>
            )}
            {message.agentUsed && (
              <span className="text-xs px-2 py-0.5 rounded-full bg-purple-100 text-purple-700 font-medium">
                {message.agentUsed}
              </span>
            )}
            {message.reflectionScore !== undefined && (
              <span
                className={`text-xs px-2 py-0.5 rounded-full border font-medium ${
                  message.reflectionScore >= 0.7
                    ? 'bg-green-50 border-green-200 text-green-700'
                    : 'bg-amber-50 border-amber-200 text-amber-700'
                }`}
              >
                quality {message.reflectionScore.toFixed(2)}
              </span>
            )}
            {message.intentMethod && message.intentConfidence !== undefined && (
              <span className="text-xs text-gray-400">
                via {message.intentMethod} · {Math.round(message.intentConfidence * 100)}%
              </span>
            )}
          </div>
        )}

        {/* Tool calls */}
        {message.toolCalls && message.toolCalls.length > 0 && (
          <div className="flex flex-wrap gap-1.5">
            {message.toolCalls.map((tc, i) => (
              <span
                key={i}
                className={`text-xs px-2 py-0.5 rounded-full border font-medium ${
                  tc.count === -1
                    ? 'bg-yellow-50 border-yellow-200 text-yellow-700 animate-pulse'
                    : 'bg-green-50 border-green-200 text-green-700'
                }`}
              >
                {TOOL_ICONS[tc.tool] || '⚙'} {TOOL_LABELS[tc.tool] || tc.tool}
                {tc.count >= 0 && ` (${tc.count})`}
              </span>
            ))}
          </div>
        )}

        {/* Deep research plan */}
        {message.researchPlan && message.researchPlan.length > 0 && (
          <div className="bg-indigo-50 border border-indigo-100 rounded-xl px-3 py-2 space-y-1">
            <p className="text-xs font-semibold text-indigo-600 uppercase tracking-wide mb-1.5">Research Plan</p>
            {message.researchPlan.map(task => (
              <div key={task.id} className="flex items-center gap-2 text-xs">
                {task.status === 'done' ? (
                  <span className="text-green-500">✓</span>
                ) : task.status === 'running' ? (
                  <RefreshCw className="w-3 h-3 text-indigo-500 animate-spin shrink-0" />
                ) : (
                  <span className="w-3 h-3 rounded-full border border-gray-300 shrink-0 inline-block" />
                )}
                <span className={`flex-1 ${task.status === 'done' ? 'text-gray-500 line-through' : task.status === 'running' ? 'text-indigo-700 font-medium' : 'text-gray-500'}`}>
                  {task.title}
                </span>
                {task.status === 'done' && task.citations !== undefined && task.citations > 0 && (
                  <span className="text-gray-400">{task.citations} src</span>
                )}
              </div>
            ))}
            {message.researchGaps && message.researchGaps.length > 0 && (
              <p className="text-xs text-amber-700 pt-1">
                Evidence gaps found, researching further: {message.researchGaps.join('; ')}
              </p>
            )}
          </div>
        )}

        {/* Reply */}
        <div className="bg-white border border-gray-100 rounded-2xl rounded-tl-sm px-4 py-3 shadow-sm">
          {message.isDraft && <p className="mb-2 text-xs text-amber-700">Draft answer — checking evidence</p>}
          {message.content ? (
            <Markdown text={message.content} codeCitations={false} />
          ) : (
            <div className="flex items-center gap-2 text-gray-400 text-sm">
              <RefreshCw className="w-3.5 h-3.5 animate-spin" />
              {message.reflectionStage === 'critiquing'
                ? 'Checking answer quality...'
                : message.reflectionStage === 'refining'
                ? 'Refining the answer...'
                : 'Thinking...'}
            </div>
          )}
          {message.isStreaming && message.content && (
            <span className="inline-block w-0.5 h-4 bg-gray-400 animate-pulse align-middle ml-0.5" />
          )}
        </div>

        {message.researchRunId && !message.researchComplete && !message.isStreaming && (
          <button type="button" disabled={!canResume}
            onClick={() => onResume(message.researchRunId!)}
            className="text-xs text-indigo-700 underline disabled:opacity-50">
            Resume research
          </button>
        )}

        {/* Citations */}
        {message.citations && message.citations.length > 0 && (
          <div className="flex flex-wrap gap-1.5">
            {message.citations.map((c, i) => (
              <a
                key={i}
                href={c.url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-xs px-2 py-0.5 rounded bg-blue-50 text-blue-600 hover:bg-blue-100 border border-blue-100 max-w-[200px] truncate"
              >
                {c.title || 'Source'} &rarr;
              </a>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
