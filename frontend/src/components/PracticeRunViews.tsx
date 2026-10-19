'use client';

import { useState } from 'react';
import { PracticeRun, RepoToolStep } from '@/lib/api';
import { AnnotatedCode } from './CodeViews';
import { ExercisePanel } from './ExercisePanel';
import { FileTree } from './FileTree';
import { LatexText } from './LatexText';
import { Markdown } from './Markdown';
import { MermaidDiagram, paperDiagram, repoDiagram } from './MermaidDiagram';
import { STOP_REASONS, StatusPill } from './practiceLabels';

type OpenFile = (path: string, start?: number, end?: number) => void;
type PaperResult = PracticeRun['result'];
export type PaperTab = 'overview' | 'code' | 'practice';
export type RepoTab = 'overview' | 'tour' | 'files';

/** "12-14, 20" -> [12, 13, 14, 20] */
function lineList(spec: string): number[] {
  return spec.split(/[,;]/).flatMap((part) => {
    const range = part.match(/(\d+)\s*[-–]\s*(\d+)/);
    if (range) {
      const [a, b] = [Number(range[1]), Number(range[2])];
      return b >= a && b - a < 80 ? Array.from({ length: b - a + 1 }, (_, i) => a + i) : [a];
    }
    const single = part.match(/\d+/);
    return single ? [Number(single[0])] : [];
  }).filter((n) => n > 0);
}

export type ExecutionData = { passed?: boolean; stdout?: string; stderr?: string; error?: string; timed_out?: boolean } | null;

export function ExecutionResult({ result, title }: { result?: ExecutionData; title?: string }) {
  if (!result) return null;
  return <div className="rounded-xl border border-gray-200 bg-gray-50 p-3 text-sm" aria-live="polite">
    <p className={result.passed ? 'font-semibold text-emerald-700' : 'font-semibold text-amber-700'}>
      {title ? `${title}: ` : ''}{result.passed ? 'self-check passed on synthetic data' : result.timed_out ? 'timed out' : 'did not pass'}
    </p>
    {(result.stdout || result.stderr || result.error) && <pre className="mt-2 max-h-56 overflow-auto whitespace-pre-wrap break-all font-mono text-xs text-gray-700">{[result.stdout, result.stderr, result.error].filter(Boolean).join('\n')}</pre>}
  </div>;
}

const CHECK_BADGES = {
  implemented: ['Matches', 'bg-emerald-100 text-emerald-800'],
  contradicted: ['Contradicts', 'bg-red-100 text-red-800'],
  missing: ['Missing', 'bg-amber-100 text-amber-800'],
  simplified: ['Simplified', 'bg-sky-100 text-sky-800'],
} as const;

function DraftReview({ review, revision, edited, repaired, onShowLines }: { review: NonNullable<PaperResult['detail_review']>;
  revision?: PaperResult['draft_revision']; edited: boolean; repaired: boolean; onShowLines: (lines: number[]) => void }) {
  const issues = review.issue_count;
  return <div className="rounded-xl border border-gray-200 p-3 text-sm">
    <p className="font-semibold">Does the code follow the paper? · {issues ? `${issues} open issue${issues === 1 ? '' : 's'}` : 'no mismatch found'}</p>
    <p className="text-xs text-gray-500">A model compared the code with the cited excerpts before anything ran. It is a careful reading, not a proof.
      {review.round > 1 ? ' Re-checked after one revision.' : ''}
      {repaired ? ' Later fixes from failed runs were not re-checked.' : ''}
      {edited ? ' It describes the agent’s version, not your edits.' : ''}</p>
    {!review.core_mechanism.implemented && <p className="mt-2 rounded-lg bg-red-50 p-2 text-red-800"><strong>The central computation looks like a placeholder.</strong> <LatexText text={review.core_mechanism.note} /></p>}
    {revision && <div className="mt-2 rounded-lg bg-gray-50 p-2">
      <strong>{revision.accepted ? 'Revised once to fix:' : 'A revision was tried but not kept:'}</strong>
      <ul className="list-disc pl-5">{revision.issues.map((issue, index) => <li key={index}><LatexText text={issue} /></li>)}</ul>
      {revision.reason && <p className="mt-1 text-xs text-gray-600">{revision.reason}</p>}
      {revision.diff && <details className="mt-1"><summary className="cursor-pointer text-xs">What changed</summary><pre className="max-h-60 overflow-auto whitespace-pre-wrap break-all text-xs">{revision.diff}</pre></details>}
    </div>}
    {!!review.static_issues?.length && <div className="mt-2 rounded-lg bg-amber-50 p-2 text-amber-900">
      <strong>Code checks</strong> <span className="text-xs">(rules applied to the code, not a model’s reading)</span>
      <ul className="list-disc pl-5 text-xs">{review.static_issues.map((issue, index) => <li key={index}>{issue}</li>)}</ul>
    </div>}
    {!!review.checks.length && <ul className="mt-2 space-y-1.5">{review.checks.map((check, index) => {
      const [label, style] = CHECK_BADGES[check.status];
      return <li key={index} className="flex items-start gap-2">
        <span className={`mt-0.5 w-20 shrink-0 rounded px-1.5 py-0.5 text-center text-[10px] font-semibold ${style}`}>{label}</span>
        <span><LatexText text={check.detail} /> <span className="text-xs text-gray-500">({check.evidence_ids.join(', ')}{check.code_lines && lineList(check.code_lines).length
          ? <> · <button type="button" onClick={() => onShowLines(lineList(check.code_lines))} className="text-sky-700 underline">line {check.code_lines}</button></> : ''})</span>
          {check.note && <span className="block text-xs text-gray-600"><LatexText text={check.note} /></span>}</span>
      </li>;
    })}</ul>}
  </div>;
}

function EvidenceList({ result, onShowInCode }: { result: PaperResult; onShowInCode: (lines: number[]) => void }) {
  const linesFor = (id: string) => (result.code_map?.links || []).filter((link) => link.evidence_id === id).flatMap((link) => link.lines);
  const code = (result.evidence || []).filter((item) => item.kind === 'repo_code').length;
  return <div className="text-sm text-gray-600">
    <p className="font-semibold text-gray-800">Paper evidence</p>
    <p className="text-xs text-gray-500">{result.evidence!.length} excerpts{result.evidence_source ? ` from the paper's ${result.evidence_source === 'latex' ? 'LaTeX source' : 'PDF'}` : ''}{code ? `, ${code} from the linked repository` : ''}</p>
    <ol className="mt-2 space-y-2">{result.evidence!.map((item, index) => <li key={index} className="rounded-lg border border-gray-100 bg-white p-2">
      <div className="flex flex-wrap items-center gap-2 text-xs">
        <strong className={item.kind === 'repo_code' ? 'text-emerald-700' : 'text-violet-700'}>{item.id || `E${index + 1}`}</strong>
        {item.kind === 'repo_code' && <span className="rounded bg-emerald-100 px-1 text-[10px] font-semibold text-emerald-800">code from linked repository</span>}
        {!!linesFor(item.id).length && <button type="button" onClick={() => onShowInCode(linesFor(item.id))} className="text-sky-700 underline">Show in code</button>}
        <a href={item.source} target="_blank" rel="noopener noreferrer" className="ml-auto text-gray-400 hover:text-primary-600">source ↗</a>
      </div>
      {item.found_for && <p className="text-[11px] text-gray-500">Found when the agent searched for “{item.found_for}”</p>}
      {item.kind === 'repo_code'
        ? <pre className="mt-1 max-h-48 overflow-auto rounded bg-gray-900 p-2 font-mono text-[11px] text-gray-100">{item.text}</pre>
        : <p className="mt-1 text-xs"><LatexText text={item.text} /></p>}
    </li>)}</ol>
  </div>;
}

function AgentWork({ result }: { result: PaperResult }) {
  const plan = result.plan;
  const gathered = result.gathered || [];
  const repo = result.repo_evidence;
  if (!plan && !gathered.length && !repo && !result.personalized) return null;
  return <div className="space-y-2 rounded-xl border border-gray-200 p-3 text-sm">
    {plan && (!!plan.choices.length || plan.instructions) && <div>
      <p className="font-semibold">Your plan</p>
      <ul className="list-disc pl-5 text-xs">{plan.choices.map((choice) => <li key={choice.id}><LatexText text={choice.question} /> → <strong><LatexText text={choice.answer} /></strong>{choice.recommended ? ' (suggested)' : ''}</li>)}
        {plan.instructions && <li>Instructions: {plan.instructions}</li>}</ul>
    </div>}
    {!!gathered.length && <div>
      <p className="font-semibold">What the agent looked up</p>
      <ul className="list-disc pl-5 text-xs">{gathered.flatMap((round) => (round.paper || []).map((search, index) => <li key={`${round.round}-${index}`}>“{search.query}” — {search.found ? `found in “${search.found}”` : 'not stated in the paper'}</li>))}</ul>
    </div>}
    {repo && <details className="text-xs"><summary className="cursor-pointer text-sm font-semibold">Repository agent: {repo.error ? 'linked repository unavailable' : `${repo.used || 0} verified code excerpt${repo.used === 1 ? '' : 's'} from ${repo.repo_url.replace('https://github.com/', '')}`}</summary>
      {repo.error ? <p className="mt-1 text-amber-700">{repo.error}</p> : <>
        <p className="mt-1 text-gray-500">{repo.label} · only lines the agent actually read and cited became code excerpts.</p>
        {repo.answer && <div className="mt-1 rounded-lg bg-gray-50 p-2"><Markdown text={repo.answer} citations={repo.citations || []} /></div>}
        {!!repo.trace?.length && <StepList steps={repo.trace} />}
      </>}
    </details>}
    {result.personalized && <p className="text-xs text-gray-500">Comments and explanations were adapted to your learning profile from Agent Chat.</p>}
  </div>;
}

function StepList({ steps }: { steps: RepoToolStep[] }) {
  return <ol className="mt-1 list-decimal space-y-0.5 pl-5 text-xs text-gray-500">{steps.map((step, index) => <li key={index}><code className="break-all">{step.tool}({Object.values(step.args || {}).join(', ')})</code> → {step.summary}</li>)}</ol>;
}

export function PaperRunView({ run, tab, onTab, editedCode, setEditedCode, editedLines, executionEnabled,
  busy, lastExecution, onExecute, onSwitchSource, onCompare, comparing, onAnswerQuestion }:
  { run: PracticeRun; tab: PaperTab; onTab: (tab: PaperTab | 'discuss') => void; editedCode: string;
    setEditedCode: (code: string) => void; editedLines: number[]; executionEnabled: boolean; busy: boolean;
    lastExecution: ExecutionData; onExecute: () => void; onSwitchSource: (kind: 'paper' | 'repo', value: string) => void;
    onCompare: () => void; comparing: boolean; onAnswerQuestion: (question: string) => void }) {
  const result = run.result;
  const edited = editedCode !== result.code;
  const [view, setView] = useState<'annotated' | 'edit'>('annotated');
  const [highlight, setHighlight] = useState<number[]>([]);
  const showInCode = (lines: number[]) => { setView('annotated'); setHighlight(lines); onTab('code'); };
  const diagram = result.code_map ? paperDiagram(result.code_map, result.manifest?.known_details || []) : null;
  const stop = STOP_REASONS[result.stop_reason || ''];
  const download = () => {
    const url = URL.createObjectURL(new Blob([editedCode], { type: 'text/x-python' }));
    const link = document.createElement('a');
    link.href = url;
    link.download = `paper-algorithm-${run.source.replace(/[^a-zA-Z0-9.-]/g, '')}.py`;
    link.click();
    URL.revokeObjectURL(url);
  };

  if (tab === 'code') {
    if (!result.code) return <p className="text-sm text-gray-500">No program yet{run.status === 'completed' ? ': the agent did not write code for this paper. See the overview for why.' : '.'}</p>;
    return <div className="space-y-4">
      <div className="grid gap-4 lg:grid-cols-5">
        <div className="space-y-2 lg:col-span-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <span className="text-sm font-semibold">{view === 'edit' && edited ? 'Your edited version' : 'Agent’s version'}{result.stop_reason === 'generated_unexecuted' ? ' · draft, not run' : ''}</span>
            <div className="flex items-center gap-2"><div className="segmented">
              <button type="button" onClick={() => setView('annotated')} className={`segmented-item !py-1 !text-xs ${view === 'annotated' ? 'segmented-item-active' : ''}`}>Annotated</button>
              <button type="button" onClick={() => setView('edit')} className={`segmented-item !py-1 !text-xs ${view === 'edit' ? 'segmented-item-active' : ''}`}>Edit</button>
              </div><button type="button" onClick={download} className="btn-secondary btn-sm">Download .py</button>
            </div>
          </div>
          {view === 'annotated'
            ? <><AnnotatedCode code={result.code} evidence={result.evidence || []} highlight={highlight} />
              <p className="text-[11px] text-gray-500">Click an <span className="rounded bg-violet-100 px-1 text-violet-800">E</span> or <span className="rounded bg-emerald-100 px-1 text-emerald-800">C</span> tag to read the excerpt a line implements.</p></>
            : <textarea id="paper-code" aria-label="Edit the program" value={editedCode} onChange={(event) => setEditedCode(event.target.value)} spellCheck={false} className="h-[28rem] w-full rounded-xl border border-gray-300 p-3 font-mono text-xs" />}
          {edited && <p role="status" className="rounded-lg border border-amber-200 bg-amber-50 p-2 text-xs text-amber-900">You changed {editedLines.length ? `line${editedLines.length === 1 ? '' : 's'} ${editedLines.join(', ')}${editedLines.length === 12 ? '…' : ''}` : 'the program'}. The evidence and review describe the agent’s version only.</p>}
          {executionEnabled && <button type="button" disabled={busy || run.status !== 'completed'} onClick={onExecute} className="btn-primary text-sm">{busy ? 'Running…' : 'Run my edited code'}</button>}
          {lastExecution && <ExecutionResult result={lastExecution} title="Your latest edit" />}
        </div>
        {!!result.evidence?.length && <aside className="max-h-[36rem] overflow-y-auto rounded-xl bg-gray-50 p-3 lg:col-span-2"><EvidenceList result={result} onShowInCode={showInCode} /></aside>}
      </div>
      {result.detail_review && <DraftReview review={result.detail_review} revision={result.draft_revision}
        edited={edited} repaired={(result.timeline?.length ?? 0) > 1} onShowLines={showInCode} />}
      {diagram && <MermaidDiagram definition={diagram} caption="How the paper maps to the code: each excerpt (left) points to the function whose lines cite it." />}
    </div>;
  }

  if (tab === 'practice') {
    return <div className="space-y-4">
      {run.status === 'completed' && result.code
        ? <ExercisePanel runId={run.run_id} evidence={result.evidence || []} executionEnabled={executionEnabled} />
        : <p className="text-sm text-gray-500">Practice opens once the program is ready.</p>}
      {!!result.questions?.length && <div className="notice notice-info">
        <h4 className="font-semibold text-sky-900">Check your understanding</h4>
        <ol className="mt-2 space-y-2">{result.questions.map((question, index) => <li key={index} className="flex items-start justify-between gap-3">
          <span>{index + 1}. {question}</span>
          <button type="button" onClick={() => onAnswerQuestion(question)} className="shrink-0 text-xs text-sky-700 underline">Answer →</button>
        </li>)}</ol>
      </div>}
    </div>;
  }

  return <div className="space-y-4">
    {stop && run.status === 'completed' && <div className={`rounded-xl border p-3 text-sm ${stop.tone === 'amber' ? 'border-amber-200 bg-amber-50 text-amber-900' : stop.tone === 'emerald' ? 'border-emerald-200 bg-emerald-50 text-emerald-900' : 'border-sky-200 bg-sky-50 text-sky-950'}`}>
      <p className="font-semibold">{stop.label}</p>
      {result.stop_reason === 'generated_unexecuted' && <p className="mt-0.5 text-xs">The code has not been executed, so nothing here confirms it reproduces the paper’s results. It shows the mechanism at toy scale.</p>}
      {result.stop_reason !== 'method_detail_insufficient' && result.code && <div className="mt-2 flex flex-wrap gap-2 text-xs">
        <button type="button" onClick={() => onTab('code')} className="rounded-lg bg-white px-2.5 py-1 font-medium shadow-sm">Read the code →</button>
        <button type="button" onClick={() => onTab('practice')} className="rounded-lg bg-white px-2.5 py-1 font-medium shadow-sm">Practice →</button>
        <button type="button" onClick={() => onTab('discuss')} className="rounded-lg bg-white px-2.5 py-1 font-medium shadow-sm">Ask a question →</button>
      </div>}
    </div>}
    {result.manifest && run.status !== 'waiting' && <div className="rounded-xl border border-sky-200 bg-white p-3 text-sm text-gray-800">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="font-semibold">{result.manifest.feasible ? result.manifest.method : 'This paper does not give enough detail for a single-file example'}</p>
        {result.source_url && <a href={result.source_url} target="_blank" rel="noopener noreferrer" className="text-xs text-primary-600 underline">Paper ↗</a>}
      </div>
      {result.manifest.reason && <p className="mt-1 text-gray-600">{result.manifest.reason}</p>}
      {!!result.manifest.known_details?.length && <><p className="mt-2 section-label">What the paper specifies</p>
        <ul className="mt-1 list-disc space-y-1 pl-5">{result.manifest.known_details.map((item, index) => <li key={index}><LatexText text={item.detail} /> <span className="text-xs text-gray-500">({item.evidence_ids.join(', ')})</span>{item.quote && <span className="block border-l-2 border-sky-300 pl-2 text-xs italic text-sky-800">“<LatexText text={item.quote} />”</span>}</li>)}</ul></>}
      {!!result.manifest.unknown_details?.length && <p className="mt-2 text-xs"><strong>Not stated in the excerpts:</strong> <LatexText text={result.manifest.unknown_details.join('; ')} /></p>}
      {!!result.manifest.simplifications?.length && <><p className="mt-2 section-label">Simplified for teaching</p>
        <ul className="mt-1 list-disc pl-5 text-xs">{result.manifest.simplifications.map((item, index) => <li key={index}><LatexText text={item} /></li>)}</ul></>}
      {!!result.method_notes?.length && <p className="mt-2 text-xs text-gray-600" title="General notes on how this kind of method is usually implemented and gets wrong. They guide the code and the review; the paper's excerpts always take precedence.">
        <strong>Also used general implementation notes on:</strong> {result.method_notes.join(', ')} (background knowledge, not from this paper)</p>}
      {!result.manifest.feasible && <p className="mt-2 text-xs">Try asking Agent Chat to explain the method, or pick a paper with clearer implementation detail.</p>}
    </div>}
    <AgentWork result={result} />
    {result.explanation && result.explanation !== result.manifest?.reason && <p className="text-sm text-gray-600">{result.explanation}</p>}
    {result.faithfulness && (edited
      ? <p role="status" className="notice notice-warn">Your edited version has not been compared with the paper.</p>
      : <div className="notice notice-accent"><strong>{result.faithfulness.status === 'PASS' ? 'A model review found the code consistent with the cited excerpts' : 'Fidelity to the paper is not confirmed'}</strong> · {result.faithfulness.reason}{!!result.faithfulness.evidence_ids?.length && <span> ({result.faithfulness.evidence_ids.join(', ')})</span>}</div>)}
    {result.execution && <p className="text-xs text-gray-500">The agent ran the program {result.attempts ?? 0} time{result.attempts === 1 ? '' : 's'} (at most 5).</p>}
    <ExecutionResult result={result.execution} title="Agent’s last run" />
    {!!result.timeline?.length && <details className="rounded-xl border border-gray-200 p-3 text-sm"><summary className="cursor-pointer font-semibold">Build and repair history ({result.timeline.length})</summary><ol className="mt-3 space-y-3">{result.timeline.map((step) => <li key={step.attempt} className="rounded-lg bg-gray-50 p-3"><strong>Run {step.attempt}</strong><p>Diagnosis: {step.diagnosis}</p><ExecutionResult result={step.execution} />{!!step.change_basis?.length && <ul className="mt-2 list-disc pl-5">{step.change_basis.map((change, index) => <li key={index}>{change.change} · {change.kind === 'paper_supported' ? `based on ${change.evidence_ids.join(', ')}` : 'engineering fix not stated in the paper'}</li>)}</ul>}{step.diff && <details className="mt-2"><summary className="cursor-pointer">Code changes</summary><pre className="max-h-60 overflow-auto whitespace-pre-wrap break-all text-xs">{step.diff}</pre></details>}</li>)}</ol></details>}
    {!!run.user_executions?.length && <details className="text-xs text-gray-600"><summary className="cursor-pointer">Your saved runs ({run.user_executions.length})</summary><div className="mt-2 space-y-2">{run.user_executions.map((item, index) => <ExecutionResult key={index} result={item.result} title={item.evidence_status === 'user_modified_unverified' ? 'Edited version' : 'Agent’s version'} />)}</div></details>}
    {result.linked_repo?.url && <div className="rounded-xl border border-gray-200 p-3 text-sm"><strong>Related repository</strong><p className="text-xs text-gray-600">{result.linked_repo.label} · <a href={result.linked_repo.provenance} target="_blank" rel="noopener noreferrer" className="underline">source</a></p><p className="text-xs text-gray-500">Linked from the arXiv {result.linked_repo.source_section || 'page'}; a link does not prove authorship.</p><div className="mt-2 flex flex-wrap gap-3"><button type="button" onClick={() => onSwitchSource('repo', result.linked_repo!.url)} className="btn-link">Explore code →</button>{result.code && <button type="button" onClick={onCompare} disabled={comparing || ['queued', 'running'].includes(run.comparison_status || '') || run.status !== 'completed'} className="text-sky-700 underline disabled:opacity-40">{comparing || ['queued', 'running'].includes(run.comparison_status || '') ? 'Comparing…' : run.comparison_status === 'failed' ? 'Retry comparison' : 'Compare with the repository’s code'}</button>}</div>{run.comparison_error && <p role="alert" className="mt-1 text-xs text-rose-700">{run.comparison_error}</p>}</div>}
    {run.comparison && <div className="rounded-xl border border-gray-200 p-3 text-sm"><h4 className="font-semibold">Teaching code vs repository module</h4><p>{run.comparison.summary}</p><p className="mt-1 text-xs">Compared with <a href={run.comparison.module_url} target="_blank" rel="noopener noreferrer" className="underline">{run.comparison.module_path} ↗</a> at {run.comparison.commit_sha.slice(0, 12)} · chosen because: {run.comparison.selection_reason}</p><ul className="mt-2 space-y-2">{run.comparison.differences.map((item, index) => <li key={index} className="grid gap-2 rounded-lg bg-gray-50 p-2 md:grid-cols-3"><div><strong>Teaching version</strong><p>{item.teaching}</p></div><div><strong>Repository version</strong><p>{item.repository}</p></div><div><strong>Why they differ</strong><p>{item.reason || 'Not established from the inspected code.'}</p></div></li>)}</ul><p className="mt-2 text-xs text-gray-500">A model comparison of one file, not proof of an official implementation. {run.comparison.limitations}</p></div>}
  </div>;
}

const ENTRY_LABELS: Record<string, string> = {
  console_script: 'Command', npm_bin: 'Command', package_api: 'Library API', package_main: 'Package main',
  script: 'Script', npm_script: 'npm script', go_main: 'Go main', rust_main: 'Rust binary',
};

function PathLink({ path, start, end, onOpenFile, label }: { path: string; start?: number; end?: number; onOpenFile: OpenFile; label?: string }) {
  return <button type="button" onClick={() => onOpenFile(path, start, end)} className="break-all text-left font-mono text-xs text-sky-700 hover:underline">{label || path}</button>;
}

export function RepoRunView({ run, tab, onSwitchSource, onOpenFile, onAnswerTour }: {
  run: PracticeRun; tab: RepoTab; onSwitchSource: (kind: 'paper' | 'repo', value: string) => void;
  onOpenFile: OpenFile; onAnswerTour: (question: string) => void }) {
  const result = run.result;
  const overview = result.overview;
  const diagram = overview ? repoDiagram(overview) : null;

  if (tab === 'tour') {
    if (!result.tour?.length) return <p className="text-sm text-gray-500">{run.status === 'completed' ? 'No reading tour was produced for this repository.' : 'The tour appears once the overview is ready.'}</p>;
    return <div className="space-y-3 text-sm">
      <p className="text-xs text-gray-600">Open each stop, read the lines, then answer the question. The agent checks your answer against the code.</p>
      <ol className="space-y-3">{result.tour.map((stop, index) => <li key={index} className="flex gap-3 rounded-xl border border-emerald-200 bg-emerald-50/40 p-3">
        <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-emerald-600 text-xs font-semibold text-white">{index + 1}</span>
        <div className="min-w-0 space-y-1">
          <PathLink path={stop.path} start={stop.start_line} end={stop.end_line} onOpenFile={onOpenFile} label={`${stop.path}:${stop.start_line}-${stop.end_line}`} />
          <p className="text-gray-700">{stop.focus}</p>
          <p className="text-xs"><strong>Question:</strong> {stop.question}</p>
          <div className="flex gap-3 text-xs">
            <button type="button" onClick={() => onOpenFile(stop.path, stop.start_line, stop.end_line)} className="text-emerald-700 underline">Read these lines</button>
            {run.status === 'completed' && <button type="button" onClick={() => onAnswerTour(stop.question)} className="text-emerald-700 underline">Answer →</button>}
          </div>
        </div>
      </li>)}</ol>
    </div>;
  }

  if (tab === 'files') {
    return <FileTree paths={result.file_tree || []} focus={result.focus} total={overview?.total_files} onOpenFile={(path) => onOpenFile(path)}
      highlight={[...(overview?.entry_points || []).map((e) => e.path), ...(overview?.central_modules || []).map((c) => c.path)]} />;
  }

  return <div className="space-y-4">
    {result.commit_sha && <p className="text-xs text-gray-500">Pinned commit <code>{result.commit_sha.slice(0, 12)}</code>{result.focus ? <> · folder <code>{result.focus}/</code></> : null}</p>}
    {overview && overview.total_files > 0 && overview.text_files * 2 < overview.total_files && <div role="status" className="notice notice-warn">
      <p className="font-semibold">This repository is too large to load completely ({Math.round(100 * overview.text_files / overview.total_files)}% of {overview.total_files.toLocaleString()} files)</p>
      <p className="mt-0.5 text-xs">The overview and code search cover the core source first; any other file can still be opened and is fetched when read.{overview.focus_suggestions?.length ? ' For full coverage, study one folder:' : ''}</p>
      {!!overview.focus_suggestions?.length && result.repo_url && result.commit_sha && <div className="mt-2 flex flex-wrap gap-2">
        {overview.focus_suggestions.map((item) => <button key={item.path} type="button" onClick={() => onSwitchSource('repo', `${result.repo_url}/tree/${result.commit_sha}/${item.path}`)}
          className="rounded-lg border border-amber-300 bg-white px-2 py-1 font-mono text-xs hover:bg-amber-100">{item.path}/ <span className="font-sans text-amber-700">· {item.source_files} source files</span></button>)}
      </div>}
    </div>}
    {result.architecture && <Markdown text={result.architecture} />}
    {overview && (!!overview.entry_points.length || !!overview.central_modules.length) && <div className="grid gap-3 md:grid-cols-2">
      {!!overview.entry_points.length && <div className="rounded-xl border border-gray-200 p-3 text-sm">
        <h4 className="font-semibold">How it starts</h4>
        <ul className="mt-1 space-y-1">{overview.entry_points.map((entry, index) => <li key={index}>
          <span className="mr-1 rounded bg-gray-100 px-1.5 py-0.5 text-[10px] font-semibold uppercase text-gray-600">{ENTRY_LABELS[entry.kind] || entry.kind}</span>
          <PathLink path={entry.path} onOpenFile={onOpenFile} /> <span className="text-xs text-gray-500">{entry.detail}</span></li>)}</ul>
        <p className="mt-1 text-[11px] text-gray-400">From packaging metadata and the code itself, not guessed by a model.</p>
      </div>}
      {!!overview.central_modules.length && <div className="rounded-xl border border-gray-200 p-3 text-sm">
        <h4 className="font-semibold">Core modules</h4>
        <ul className="mt-1 space-y-1">{overview.central_modules.map((module) => <li key={module.path}>
          <PathLink path={module.path} onOpenFile={onOpenFile} /> <span className="text-xs text-gray-500">imported by {module.imported_by} file{module.imported_by === 1 ? '' : 's'}</span></li>)}</ul>
        <p className="mt-1 text-[11px] text-gray-400">Ranked by how many non-test files import them.</p>
      </div>}
    </div>}
    {diagram && <MermaidDiagram definition={diagram} caption="Arrows point from a file to a module it imports; blue boxes are entry points. Built from the import graph, not by a model." />}
    {!!result.entry_points?.length && <p className="text-sm"><strong>Suggested reading order:</strong> {result.entry_points.map((path, index) => <span key={path}>{index ? ' → ' : ''}<PathLink path={path} onOpenFile={onOpenFile} /></span>)}</p>}
    {!!result.file_roles?.length && <div><h4 className="text-sm font-semibold">Key files</h4>{result.file_roles.map((item) => <p key={item.path} className="text-sm"><PathLink path={item.path} onOpenFile={onOpenFile} /> — {item.role}</p>)}</div>}
    {overview
      ? <p className="text-xs text-gray-500">{overview.text_files} of {overview.total_files} files are searchable in the discussion{overview.skipped_files ? `; the other ${overview.skipped_files} are binary, very large, or outside the snapshot budget (text files can still be opened)` : ''}.{overview.archive_truncated ? ' The archive exceeded the download limit and was read only in part.' : ''}</p>
      : result.tree_truncated && <p className="text-xs text-amber-700">Only part of the file tree was inspected.</p>}
    {result.linked_paper?.arxiv_id && <div className="rounded-xl border p-3 text-sm"><strong>Related paper</strong><p className="text-xs">{result.linked_paper.label}</p><button onClick={() => onSwitchSource('paper', result.linked_paper!.arxiv_id)} className="btn-link mt-2">Build algorithm →</button></div>}
  </div>;
}

/** Starter questions built from the deterministic overview, so a learner does not face an empty box. */
export function repoSuggestions(result: PaperResult): string[] {
  const overview = result.overview;
  if (!overview) return [];
  const out: string[] = [];
  const command = overview.entry_points.find((e) => e.kind === 'console_script' || e.kind === 'npm_bin');
  if (command) out.push(`What happens when I run ${command.detail.match(/`([^`]+)`/)?.[1] || 'the command'}? Trace the call path.`);
  const api = overview.entry_points.find((e) => e.kind === 'package_api');
  if (api) out.push(`What does ${api.detail.match(/`import ([^`]+)`/)?.[1] || 'the package'} expose, and where are its main classes implemented?`);
  const script = overview.entry_points.find((e) => e.kind === 'script');
  if (script && out.length < 2) out.push(`Walk me through ${script.path} step by step.`);
  overview.central_modules.slice(0, 2).forEach((module) => out.push(`What does ${module.path} do, and which parts of the code depend on it?`));
  return out.slice(0, 4);
}

export function RunDiscussion({ run, guidedQuestion, setGuidedQuestion, messageInput, setMessageInput, sending, onSend,
  liveSteps, pendingPrompt, onOpenFile }:
  { run: PracticeRun; guidedQuestion: string; setGuidedQuestion: (value: string) => void; messageInput: string;
    setMessageInput: (value: string) => void; sending: boolean; onSend: () => void;
    liveSteps: RepoToolStep[]; pendingPrompt: string; onOpenFile?: OpenFile }) {
  const repo = run.kind === 'repo';
  const suggestions = repo ? repoSuggestions(run.result) : [];
  const empty = !run.messages?.length && !sending;
  return <div id="run-discussion" className="space-y-3">
    {repo && <p className="text-xs text-gray-500">The agent searches, reads and follows calls in the pinned snapshot (up to 8 tool calls per question), cites file:line ranges you can open, and remembers what earlier answers established.</p>}
    {!repo && !!run.result.questions?.length && <div className="space-y-1.5">
      <p className="text-xs font-semibold text-gray-500">Answer a question about the program</p>
      {run.result.questions.map((question, index) => <button key={index} onClick={() => { setGuidedQuestion(question); setMessageInput(''); }} className={`block w-full rounded-lg border px-3 py-2 text-left text-sm ${guidedQuestion === question ? 'border-sky-500 bg-sky-50' : 'border-sky-200'}`}>{index + 1}. {question}</button>)}
    </div>}
    {run.messages?.map((item) => <div key={item.id} className="space-y-2">
      <div className="ml-auto max-w-[85%] rounded-2xl rounded-br-sm bg-primary-600 px-3 py-2 text-sm text-white">
        {item.guided_question && <p className="text-[11px] text-primary-100">Answer to: {item.guided_question}</p>}
        <p className="whitespace-pre-wrap">{item.prompt}</p>
      </div>
      <div className="max-w-[95%] rounded-2xl rounded-bl-sm bg-gray-50 px-3 py-2">
        {item.response
          ? <Markdown text={item.response} citations={repo ? item.citations || [] : undefined} onOpenFile={onOpenFile} />
          : <p className="text-sm text-gray-500">Reply in progress…</p>}
        {!!item.trace?.length && <details className="mt-1"><summary className="cursor-pointer text-xs text-gray-500">How the agent found this ({item.trace.length} steps)</summary><StepList steps={item.trace} /></details>}
      </div>
    </div>)}
    {sending && pendingPrompt && <div className="space-y-2" aria-live="polite">
      <div className="ml-auto max-w-[85%] rounded-2xl rounded-br-sm bg-primary-600 px-3 py-2 text-sm text-white"><p className="whitespace-pre-wrap">{pendingPrompt}</p></div>
      <div className="max-w-[95%] rounded-2xl rounded-bl-sm border border-sky-100 bg-sky-50/50 px-3 py-2">
        {!!liveSteps.length && <StepList steps={liveSteps} />}
        <p className="mt-1 flex items-center gap-2 text-xs text-sky-700"><span className="h-2 w-2 animate-pulse rounded-full bg-sky-500" />{repo ? (liveSteps.length ? 'Investigating…' : 'Planning the investigation…') : 'Thinking…'}</p>
      </div>
    </div>}
    {empty && !!suggestions.length && <div className="flex flex-wrap gap-2">
      {suggestions.map((question) => <button key={question} type="button" onClick={() => { setGuidedQuestion(''); setMessageInput(question); }}
        className="rounded-full border border-sky-200 bg-white px-3 py-1 text-left text-xs text-sky-800 hover:bg-sky-50">{question}</button>)}
    </div>}
    {guidedQuestion && <p className="flex items-center justify-between gap-2 rounded-lg bg-sky-50 p-2 text-xs text-sky-800"><span>Answering: {guidedQuestion}</span><button type="button" onClick={() => setGuidedQuestion('')} className="underline">clear</button></p>}
    <div className="flex items-end gap-2">
      <textarea id="run-discussion-input" value={messageInput} onChange={(event) => setMessageInput(event.target.value)}
        onKeyDown={(event) => { if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) { event.preventDefault(); onSend(); } }}
        placeholder={guidedQuestion ? 'Write your answer…' : repo ? 'Ask about the code, or select lines in a file and ask about them' : 'Ask about the program…'} maxLength={2000}
        className="input-field h-20 flex-1 text-sm" />
      <button onClick={onSend} disabled={sending || !messageInput.trim()} title="Send (⌘/Ctrl + Enter)" className="btn-primary text-sm">{sending ? '…' : 'Send'}</button>
    </div>
  </div>;
}

export { StatusPill };
