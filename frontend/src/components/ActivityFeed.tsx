'use client';

import { useEffect, useRef } from 'react';
import { PracticeEvent } from '@/lib/api';

type Tone = 'info' | 'ok' | 'warn' | 'wait';

const short = (url: string) => url.replace(/^https:\/\/github\.com\//, '');
const list = (items: unknown, n = 3) => (Array.isArray(items) ? items.slice(0, n).join(', ') : '');

/** One readable line per progress event the Coding Agent emits. */
export function describeEvent(event: PracticeEvent): { text: string; tone: Tone } {
  const d = event.data || {};
  switch (event.kind) {
    case 'started': return { text: 'Started', tone: 'info' };
    case 'searching_paper': return { text: "Fetching the paper's full text", tone: 'info' };
    case 'evidence':
      return d.found
        ? { text: `Selected ${d.found} excerpts from the ${d.source === 'latex' ? 'LaTeX source' : d.source === 'pdf' ? 'PDF' : 'paper'}${d.linked_repo ? ` · linked repository ${short(d.linked_repo)}` : ''}`, tone: 'ok' }
        : { text: `No usable full text: ${d.reason || 'unknown reason'}`, tone: 'warn' };
    case 'manifest': {
      const open = Array.isArray(d.unknown) && d.unknown.length ? `; still open: ${list(d.unknown, 2)}` : '';
      const choices = d.choices ? `; ${d.choices} choice${d.choices === 1 ? '' : 's'} for you` : '';
      const when = d.round ? ` after search ${d.round}` : '';
      return d.feasible
        ? { text: `Method check${when}: ${d.method} — ${d.known} cited details${open}${choices}`, tone: 'ok' }
        : { text: `Method check${when}: not enough detail yet${open}`, tone: 'warn' };
    }
    case 'evidence_search': return { text: `Searching the paper for “${d.query}”`, tone: 'info' };
    case 'evidence_result':
      return d.found ? { text: `Found in section “${d.section}”`, tone: 'ok' } : { text: 'Not stated in the paper text', tone: 'warn' };
    case 'repo_search': return { text: `Asking the repository agent to read ${short(d.repo_url || '')} @ ${d.commit}`, tone: 'info' };
    case 'repo_step': return { text: `Repository agent · ${d.tool}(${Object.values(d.args || {}).join(', ')}) → ${d.summary}`, tone: 'info' };
    case 'repo_result':
      return { text: d.used ? `Added ${d.used} verified code excerpt${d.used === 1 ? '' : 's'} from the repository` : 'The repository agent found no verifiable code lines', tone: d.used ? 'ok' : 'warn' };
    case 'repo_unavailable': return { text: `Linked repository unavailable: ${d.reason}`, tone: 'warn' };
    case 'evidence_unchanged': return { text: 'No new evidence found', tone: 'warn' };
    case 'awaiting_confirmation': return { text: 'Waiting for you to review the plan', tone: 'wait' };
    case 'plan_submitted': return { text: 'You confirmed the plan', tone: 'ok' };
    case 'resumed': return { text: 'Resumed with your plan', tone: 'info' };
    case 'plan_confirmed': return { text: `Applying your ${d.choices || 0} choice${d.choices === 1 ? '' : 's'}${d.instructions ? ' and instructions' : ''}`, tone: 'info' };
    case 'method_notes': return { text: `Using general implementation notes on ${list(d.families)}`, tone: 'info' };
    case 'generating': return { text: `Writing the ${d.method || 'teaching'} program`, tone: 'info' };
    case 'static_check': return { text: `Static check failed: ${d.issue}`, tone: 'warn' };
    case 'draft': return { text: `Draft written (${d.lines} lines)`, tone: 'ok' };
    case 'review':
      return { text: `Code-vs-evidence review${d.round > 1 ? ' of the revision' : ''}: ${d.issues ? `${d.issues} issue${d.issues === 1 ? '' : 's'}` : 'no mismatch'}${d.core_ok === false ? ' · core mechanism looks like a placeholder' : ''}`, tone: d.issues ? 'warn' : 'ok' };
    case 'revision':
      return d.accepted ? { text: `Revised: ${list(d.issues, 2)}`, tone: 'info' } : { text: 'Revision rejected by the static check; kept the reviewed draft', tone: 'warn' };
    case 'execution': return { text: `Run ${d.attempt}: ${d.passed ? 'self-check passed' : 'did not pass'}`, tone: d.passed ? 'ok' : 'warn' };
    case 'pinning': return { text: 'Pinning the repository commit', tone: 'info' };
    case 'snapshot': return { text: `Snapshot ${d.commit}: ${d.searchable} of ${d.files} files searchable${d.focus ? ` · folder ${d.focus}/` : ''}`, tone: 'ok' };
    case 'overview': return { text: `Entry points: ${list(d.entry_points) || 'none found'} · core: ${list(d.core_modules) || 'none found'}`, tone: 'ok' };
    case 'explaining': return { text: `Reading outlines of ${(d.files || []).length} files and writing the overview`, tone: 'info' };
    case 'finished':
      return d.stop_reason !== undefined
        ? { text: `Finished (${String(d.stop_reason).replace(/_/g, ' ')})`, tone: 'ok' }
        : { text: `Overview ready${d.tour_stops ? ` with a ${d.tour_stops}-stop reading tour` : ''}`, tone: 'ok' };
    case 'exercise_graded': return { text: `Exercise checked: ${d.correct}/${d.blanks} correct`, tone: 'ok' };
    default: return { text: event.kind.replace(/_/g, ' '), tone: 'info' };
  }
}

const DOT: Record<Tone, string> = { info: 'bg-sky-400', ok: 'bg-emerald-500', warn: 'bg-amber-500', wait: 'bg-violet-500' };

export function ActivityFeed({ events, live }: { events: PracticeEvent[]; live: boolean }) {
  const end = useRef<HTMLDivElement>(null);
  useEffect(() => { if (live) end.current?.scrollIntoView({ block: 'nearest' }); }, [events.length, live]);
  if (!events.length) return live ? <p className="text-xs text-gray-500">Starting…</p> : null;
  const items = <ol className="max-h-64 space-y-1 overflow-y-auto pr-1 text-xs">
    {events.map((event) => {
      const { text, tone } = describeEvent(event);
      return <li key={event.id} className={`flex items-start gap-2 ${event.kind === 'repo_step' ? 'pl-4 text-gray-500' : 'text-gray-700'}`}>
        <span className={`mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full ${DOT[tone]}`} /><span className="break-words">{text}</span></li>;
    })}
    <div ref={end} />
  </ol>;
  return live
    ? <div className="rounded-xl border border-sky-100 bg-sky-50/40 p-3"><p className="mb-2 flex items-center gap-2 text-xs font-semibold text-sky-800"><span className="h-2 w-2 animate-pulse rounded-full bg-sky-500" />Agent activity</p>{items}</div>
    : <details className="rounded-xl border border-gray-100 p-3"><summary className="cursor-pointer text-xs font-semibold text-gray-600">Agent activity ({events.length} steps)</summary><div className="mt-2">{items}</div></details>;
}
