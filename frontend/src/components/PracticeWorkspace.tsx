'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import type { Origin } from './dashboard/shared';
import { formatDistanceToNow } from 'date-fns';
import toast from 'react-hot-toast';
import { errorMessage, practiceApi, PracticeEvent, PracticeRun, PracticeRunSummary, RepoToolStep } from '@/lib/api';
import { ActivityFeed } from './ActivityFeed';
import { CodeViewer } from './CodeViews';
import { PlanReview } from './PlanReview';
import { ExecutionData, PaperRunView, PaperTab, RepoRunView, RepoTab, RunDiscussion } from './PracticeRunViews';
import { runStatus, StatusPill } from './practiceLabels';

type RunTab = PaperTab | RepoTab | 'discuss';
const ago = (iso?: string | null) => { try { return iso ? formatDistanceToNow(new Date(iso.endsWith('Z') || iso.includes('+') ? iso : `${iso}Z`), { addSuffix: true }) : ''; } catch { return ''; } };

const ACTIVE = ['queued', 'running'];

export type PracticeContext = {
  kind: 'paper' | 'repo'; source: string; title?: string; description?: string;
  url?: string; origin?: Origin;
};

function tabsFor(run: PracticeRun): { id: RunTab; label: string; count?: number; disabled?: boolean }[] {
  const finished = run.status === 'completed';
  const messages = run.messages?.length || 0;
  return run.kind === 'paper'
    ? [{ id: 'overview', label: 'Overview' },
      { id: 'code', label: 'Code', disabled: !run.result.code },
      { id: 'practice', label: 'Practice', disabled: !finished || !run.result.code },
      { id: 'discuss', label: 'Discuss', count: messages, disabled: !finished || !run.result.code }]
    : [{ id: 'overview', label: 'Overview' },
      { id: 'tour', label: 'Reading tour', count: run.result.tour?.length, disabled: !run.result.tour?.length },
      { id: 'files', label: 'Files', disabled: !run.result.file_tree?.length },
      { id: 'discuss', label: 'Discuss', count: messages, disabled: !finished }];
}

export default function PracticeWorkspace({ context, onAskAgent }: {
  context: PracticeContext | null; onAskAgent?: (context: PracticeContext) => void;
}) {
  const [mode, setMode] = useState<'paper' | 'repo'>('paper');
  const [source, setSource] = useState('');
  const [objective, setObjective] = useState('');
  const [capabilities, setCapabilities] = useState({
    enabled: false, paper_enabled: false, paper_execution_enabled: false, repo_enabled: false, reason: '',
  });
  const [runs, setRuns] = useState<PracticeRunSummary[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [run, setRun] = useState<PracticeRun | null>(null);
  const [loadError, setLoadError] = useState('');
  const [busy, setBusy] = useState(false);
  const [editedCode, setEditedCode] = useState('');
  const [messageInput, setMessageInput] = useState('');
  const [guidedQuestion, setGuidedQuestion] = useState('');
  const [sending, setSending] = useState(false);
  const [lastExecution, setLastExecution] = useState<ExecutionData>(null);
  const [comparing, setComparing] = useState(false);
  const [confirmPlan, setConfirmPlan] = useState(true);
  const [events, setEvents] = useState<PracticeEvent[]>([]);
  const [followKey, setFollowKey] = useState(0);
  const [liveSteps, setLiveSteps] = useState<RepoToolStep[]>([]);
  const [pendingPrompt, setPendingPrompt] = useState('');
  const [viewer, setViewer] = useState<{ path: string; start: number; end: number } | null>(null);
  const [planBusy, setPlanBusy] = useState(false);
  const [runTab, setRunTab] = useState<RunTab>('overview');
  const lastStatus = useRef<{ id: string; status: string } | null>(null);

  useEffect(() => { setRunTab('overview'); }, [selected]);

  // Tell the learner when a run they are watching finishes or needs them.
  useEffect(() => {
    if (!run) return;
    const before = lastStatus.current;
    if (before?.id === run.run_id && before.status !== run.status && ACTIVE.includes(before.status)) {
      if (run.status === 'waiting') toast('The agent needs you to review its plan', { icon: '✋' });
      else if (run.status === 'completed') toast.success(run.kind === 'repo' ? 'Repository overview is ready' : 'Your program is ready');
      else if (run.status === 'failed') toast.error('The run failed');
    }
    lastStatus.current = { id: run.run_id, status: run.status };
  }, [run?.run_id, run?.status, run?.kind]);

  useEffect(() => {
    if (context) {
      setMode(context.kind);
      setSource(context.source);
      setSelected(null);
      setRun(null);
      setLoadError('');
    }
  }, [context]);

  const refreshList = useCallback(async () => {
    try { setRuns(await practiceApi.list()); } catch (error) { toast.error(errorMessage(error)); }
  }, []);

  useEffect(() => {
    let mounted = true;
    practiceApi.capabilities()
      .then((data) => { if (mounted) setCapabilities({ ...data, reason: data.reason || '' }); })
      .catch((error) => { if (mounted) setCapabilities((old) => ({ ...old, reason: errorMessage(error) })); });
    refreshList();
    return () => { mounted = false; };
  }, [refreshList]);

  // Follow a run live: load it, stream its progress events while it works, reload when the
  // stream ends (finished, failed, or waiting for the learner). Falls back to polling on errors.
  useEffect(() => {
    if (!selected) return;
    setLoadError('');
    let active = true;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const controller = new AbortController();
    const follow = async () => {
      let current: PracticeRun;
      try {
        current = await practiceApi.get(selected);
      } catch (error) {
        if (active) {
          const message = errorMessage(error);
          setLoadError(message);
          toast.error(message);
        }
        return;
      }
      if (!active) return;
      setRun(current);
      setEvents(current.events || []);
      setLoadError('');
      if (ACTIVE.includes(current.status)) {
        const after = current.events?.length ? current.events[current.events.length - 1].id : 0;
        let streamed = true;
        try {
          await practiceApi.streamEvents(selected, after, (event) => {
            if (!active) return;
            if (event.type === 'event') {
              setEvents((old) => old.some((item) => item.id === event.id) ? old
                : [...old, { id: event.id, kind: event.kind, data: event.data, at: event.at }]);
            } else if (event.type === 'status') {
              setRun((old) => old && old.run_id === selected ? { ...old, status: event.status, stage: event.stage } : old);
            }
          }, controller.signal);
        } catch {
          streamed = false;  // proxy or network trouble: poll instead
        }
        if (active) timer = setTimeout(follow, streamed ? 200 : 2000);
      } else if (ACTIVE.includes(current.comparison_status || '')) {
        timer = setTimeout(follow, 2000);
      } else {
        refreshList();
      }
    };
    follow();
    return () => { active = false; controller.abort(); if (timer) clearTimeout(timer); };
  }, [selected, refreshList, followKey]);

  useEffect(() => {
    if (run) {
      setEditedCode(run.kind === 'paper' ? run.result.code || '' : '');
      setLastExecution(null);
    }
  }, [run?.run_id, run?.result.code]);

  const create = async () => {
    if (busy) return;
    setBusy(true);
    try {
      const result = mode === 'paper'
        ? await practiceApi.createPaper(source.trim(), objective.trim(), confirmPlan)
        : await practiceApi.createRepo(source.trim());
      setRun(null);
      setLoadError('');
      setSelected(result.run_id);
      await refreshList();
    } catch (error) {
      toast.error(errorMessage(error));
    } finally {
      setBusy(false);
    }
  };

  const sendMessage = async () => {
    if (!run || !messageInput.trim() || sending) return;
    const text = messageInput.trim();
    setSending(true);
    setPendingPrompt(text);
    setLiveSteps([]);
    let failure = '';
    try {
      await practiceApi.streamMessage(run.run_id, text, guidedQuestion, (event) => {
        if (event.type === 'step') setLiveSteps((old) => [...old, event]);
        if (event.type === 'error') failure = event.detail || 'Could not generate a reply';
      });
      if (failure) throw new Error(failure);
      setRun(await practiceApi.get(run.run_id));
      setMessageInput('');
      setGuidedQuestion('');
    } catch (error) {
      toast.error(errorMessage(error));
    } finally {
      setSending(false);
      setPendingPrompt('');
      setLiveSteps([]);
    }
  };

  const submitPlan = async (plan: { choices: Record<string, string>; simplifications: string[]; instructions: string }) => {
    if (!run || planBusy) return;
    setPlanBusy(true);
    try {
      await practiceApi.confirmPlan(run.run_id, plan);
      setRun({ ...run, status: 'queued', stage: 'plan_confirmed' });
      setFollowKey((key) => key + 1);
    } catch (error) {
      toast.error(errorMessage(error));
    } finally {
      setPlanBusy(false);
    }
  };

  const openFile = (path: string, start = 1, end = 0) => setViewer({ path, start, end: end || start });
  const focusDiscussion = () => setTimeout(() => document.getElementById('run-discussion-input')?.focus(), 50);
  const answerTour = (question: string) => { setGuidedQuestion(question); setMessageInput(''); setRunTab('discuss'); focusDiscussion(); };
  const askAbout = (reference: string) => { setGuidedQuestion(''); setMessageInput((old) => `${reference}${old}`); setRunTab('discuss'); focusDiscussion(); };

  const executeEdit = async () => {
    if (!run || busy) return;
    setBusy(true);
    try {
      const response = await practiceApi.execute(run.run_id, editedCode, crypto.randomUUID());
      setLastExecution(response.result);
    } catch (error) {
      toast.error(errorMessage(error));
    } finally {
      setBusy(false);
    }
  };

  const cancelRun = async () => {
    if (!run || busy) return;
    setBusy(true);
    try {
      await practiceApi.cancel(run.run_id);
      setRun({ ...run, status: 'cancelled', stage: 'cancelled' });
      await refreshList();
    } catch (error) {
      toast.error(errorMessage(error));
    } finally {
      setBusy(false);
    }
  };

  const switchSource = (kind: 'paper' | 'repo', value: string) => {
    setViewer(null);
    setMode(kind);
    setSource(value);
    setSelected(null);
    setRun(null);
    setLoadError('');
    setMessageInput('');
  };

  const compare = async () => {
    if (!run || comparing) return;
    setComparing(true);
    try {
      await practiceApi.compare(run.run_id);
      setRun(await practiceApi.get(run.run_id));
    } catch (error) {
      toast.error(errorMessage(error));
    } finally {
      setComparing(false);
    }
  };

  const editedLines = run?.kind === 'paper' && run.result.code && editedCode !== run.result.code
    ? editedCode.split('\n')
      .map((line, index) => line !== run.result.code?.split('\n')[index] ? index + 1 : null)
      .filter((line): line is number => line !== null).slice(0, 12)
    : [];
  const modeEnabled = mode === 'paper' ? capabilities.paper_enabled : capabilities.repo_enabled;

  return <section className="space-y-5 animate-in">
    <div>
      <h2 className="text-lg font-bold text-gray-900">Coding Agent</h2>
      <p className="text-sm text-gray-500">Build and inspect a paper algorithm, or explore a public repository with an agent.</p>
    </div>
    {context && mode === context.kind && source === context.source && <div className="notice notice-info !p-4">
      <p className="text-xs font-semibold text-sky-700">From {context.origin || 'Discovery'} · {context.kind === 'paper' ? 'Paper' : 'Repository'}</p>
      <p className="mt-1 font-medium text-sky-950">{context.title || context.source}</p>
      {context.description && <p className="mt-1 text-sm text-sky-800 line-clamp-2">{context.description}</p>}
      <div className="mt-3 flex gap-3 text-xs">{context.url && <a href={context.url} target="_blank" rel="noopener noreferrer" className="underline">Open source ↗</a>}{onAskAgent && context.kind === 'paper' && <button onClick={() => onAskAgent(context)} className="underline">Ask Agent →</button>}</div>
    </div>}
    {!capabilities.enabled && <p role="status" className="notice notice-warn">Coding Agent is unavailable: {capabilities.reason || 'checking configuration'}.</p>}
    {capabilities.enabled && mode === 'paper' && !capabilities.paper_execution_enabled && <p className="notice notice-warn">Draft mode: code execution and repair require the isolated sandbox worker.</p>}
    <div className="card space-y-3">
      <div className="segmented"><button onClick={() => { setMode('paper'); setSource(''); }} className={`segmented-item ${mode === 'paper' ? 'segmented-item-active' : ''}`}>Paper</button><button onClick={() => { setMode('repo'); setSource(''); }} className={`segmented-item ${mode === 'repo' ? 'segmented-item-active' : ''}`}>Repository</button></div>
      <label htmlFor="practice-source" className="text-sm font-medium">{mode === 'paper' ? 'arXiv ID' : 'Public GitHub URL'}</label>
      <input id="practice-source" value={source} onChange={(event) => setSource(event.target.value)} placeholder={mode === 'paper' ? '2608.14407' : 'https://github.com/owner/repo  (or …/tree/<branch>/<folder>)'} className="input-field text-sm" />
      {mode === 'paper' && <input value={objective} onChange={(event) => setObjective(event.target.value)} placeholder="Learning objective (optional)" className="input-field text-sm" />}
      {mode === 'paper' && <label className="flex items-center gap-2 text-xs text-gray-600"><input type="checkbox" checked={confirmPlan} onChange={(event) => setConfirmPlan(event.target.checked)} />Let me review the plan and answer the agent&apos;s questions before it writes code</label>}
      <button onClick={create} disabled={!modeEnabled || busy || !source.trim()} className="btn-primary text-sm">{busy ? 'Working…' : mode === 'paper' ? 'Build algorithm' : 'Explore code'}</button>
    </div>
    {!!runs.length && <div className="card !p-4">
      <h3 className="mb-2 text-sm font-semibold">Your recent runs</h3>
      <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">{runs.map((item) => {
        const status = runStatus(item);
        return <button key={item.run_id} onClick={() => { setSelected(item.run_id); setRun(null); setLoadError(''); }}
          className={`rounded-xl border p-3 text-left transition hover:border-primary-300 ${selected === item.run_id ? 'border-primary-500 bg-primary-50/40' : item.status === 'waiting' ? 'border-violet-300 bg-violet-50/40' : 'border-gray-200'}`}>
          <p className="text-[10px] font-semibold uppercase tracking-wide text-gray-400">{item.kind === 'paper' ? `Paper · ${item.source}` : 'Repository'}</p>
          <p className="mt-0.5 line-clamp-2 text-sm font-medium text-gray-900">{item.title || item.source}</p>
          <div className="mt-1.5 flex items-center justify-between gap-2"><StatusPill label={item.status === 'completed' ? 'Done' : status.label.split(' · ')[0]} tone={status.tone} /><span className="text-[11px] text-gray-400">{ago(item.created_at)}</span></div>
        </button>;
      })}</div>
    </div>}
    {selected && !run && (loadError
      ? <div role="alert" className="notice notice-error">Could not load this Coding Agent run: {loadError}</div>
      : <p className="text-sm text-gray-500">Loading Coding Agent run…</p>)}
    {run && <div className="card space-y-4">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div className="min-w-0"><p className="text-[11px] font-semibold uppercase tracking-wide text-gray-400">{run.kind === 'paper' ? `Paper · ${run.source}` : 'Repository'}</p>
          <h3 className="font-semibold text-gray-900">{run.kind === 'paper' ? run.result.title || run.source : run.source.replace('https://github.com/', '')}</h3></div>
        <div className="flex items-center gap-2"><StatusPill {...runStatus({ ...run, stop_reason: run.result.stop_reason })} />{['queued', 'running', 'waiting'].includes(run.status) && <button onClick={cancelRun} className="btn-secondary btn-sm">Cancel</button>}</div>
      </div>
      {run.error && <p role="alert" className="text-sm text-rose-700">{run.error}</p>}
      <ActivityFeed events={events} live={ACTIVE.includes(run.status)} />
      {run.status === 'waiting' && run.kind === 'paper' && <PlanReview key={run.run_id} run={run} busy={planBusy} onSubmit={submitPlan} />}
      {run.status !== 'waiting' && <nav className="flex gap-1 overflow-x-auto border-b border-gray-200" aria-label="Run sections">
        {tabsFor(run).map((item) => <button key={item.id} type="button" disabled={item.disabled} onClick={() => setRunTab(item.id)}
          className={`relative whitespace-nowrap px-3 py-2 text-sm disabled:cursor-not-allowed disabled:opacity-40 ${runTab === item.id ? 'font-semibold text-primary-700' : 'text-gray-500 hover:text-gray-900'}`}>
          {item.label}{item.count ? <span className="ml-1 rounded-full bg-gray-100 px-1.5 text-[10px] text-gray-600">{item.count}</span> : null}
          {runTab === item.id && <span className="absolute inset-x-0 bottom-0 h-0.5 rounded-t-full bg-primary-600" />}
        </button>)}
      </nav>}
      {runTab === 'discuss'
        ? (run.status === 'completed' && (run.kind === 'repo' || !!run.result.code)
          ? <RunDiscussion run={run} guidedQuestion={guidedQuestion} setGuidedQuestion={setGuidedQuestion} messageInput={messageInput} setMessageInput={setMessageInput} sending={sending} onSend={sendMessage}
              liveSteps={liveSteps} pendingPrompt={pendingPrompt} onOpenFile={run.kind === 'repo' ? openFile : undefined} />
          : <p className="text-sm text-gray-500">The discussion opens when the run is finished.</p>)
        : run.kind === 'paper'
          ? <PaperRunView run={run} tab={(['overview', 'code', 'practice'].includes(runTab) ? runTab : 'overview') as PaperTab} onTab={setRunTab}
              editedCode={editedCode} setEditedCode={setEditedCode} editedLines={editedLines} executionEnabled={capabilities.paper_execution_enabled} busy={busy} lastExecution={lastExecution} onExecute={executeEdit} onSwitchSource={switchSource} onCompare={compare} comparing={comparing}
              onAnswerQuestion={(question) => { setGuidedQuestion(question); setMessageInput(''); setRunTab('discuss'); focusDiscussion(); }} />
          : <RepoRunView run={run} tab={(['overview', 'tour', 'files'].includes(runTab) ? runTab : 'overview') as RepoTab} onSwitchSource={switchSource} onOpenFile={openFile} onAnswerTour={answerTour} />}
    </div>}
    {run && viewer && run.kind === 'repo' && <CodeViewer runId={run.run_id} target={viewer} onClose={() => setViewer(null)} onAsk={askAbout} />}
  </section>;
}
