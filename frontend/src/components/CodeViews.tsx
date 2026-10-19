'use client';

import { Fragment, useEffect, useRef, useState } from 'react';
import { errorMessage, practiceApi, RepoFileView } from '@/lib/api';
import { LatexText } from './LatexText';

type Evidence = { id: string; text: string; kind?: string; source?: string };
const MARKER = /\[((?:E|C)\d+)\]/g;

function EvidenceBody({ item }: { item: Evidence }) {
  return item.kind === 'repo_code'
    ? <pre className="max-h-48 overflow-auto whitespace-pre rounded bg-gray-900 p-2 font-mono text-[11px] text-gray-100">{item.text}</pre>
    : <p className="text-xs text-gray-700"><LatexText text={item.text} /></p>;
}

/** The generated program with clickable evidence tags; lines citing a selected excerpt are highlighted. */
export function AnnotatedCode({ code, evidence, highlight }: { code: string; evidence: Evidence[]; highlight: number[] }) {
  const [open, setOpen] = useState<{ line: number; id: string } | null>(null);
  const first = useRef<HTMLDivElement>(null);
  const byId = new Map(evidence.map((item) => [item.id, item]));
  useEffect(() => { first.current?.scrollIntoView({ block: 'center', behavior: 'smooth' }); }, [highlight.join(',')]);
  const lines = code.split('\n');
  return <div className="max-h-[28rem] overflow-auto rounded-xl border border-gray-200 bg-gray-50 py-2 font-mono text-xs">
    {lines.map((line, index) => {
      const number = index + 1;
      const hashAt = line.indexOf('#');
      const ids = hashAt >= 0 ? Array.from(new Set(Array.from(line.slice(hashAt).matchAll(MARKER), (m) => m[1]))) : [];
      const lit = highlight.includes(number);
      return <Fragment key={number}>
        <div ref={lit && number === Math.min(...highlight) ? first : undefined} className={`flex gap-3 px-2 ${lit ? 'bg-amber-100' : ''}`}>
          <span className="w-8 shrink-0 select-none text-right text-gray-400">{number}</span>
          <span className="whitespace-pre-wrap break-all text-gray-800">{line || ' '}</span>
          {ids.map((id) => <button key={id} type="button" onClick={() => setOpen(open?.line === number && open.id === id ? null : { line: number, id })}
            className={`ml-auto h-fit shrink-0 rounded px-1.5 text-[10px] font-semibold ${id.startsWith('C') ? 'bg-emerald-100 text-emerald-800' : 'bg-violet-100 text-violet-800'}`}>{id}</button>)}
        </div>
        {open?.line === number && byId.get(open.id) && <div className="mx-2 my-1 rounded-lg border border-violet-200 bg-white p-2 font-sans">
          <p className="mb-1 text-[11px] font-semibold text-gray-500">{open.id} · {open.id.startsWith('C') ? 'code from the linked repository' : 'paper excerpt'}</p>
          <EvidenceBody item={byId.get(open.id)!} />
        </div>}
      </Fragment>;
    })}
  </div>;
}

/** In-page reader for one file of a repository snapshot; select lines to ask the agent about them. */
export function CodeViewer({ runId, target, onClose, onAsk }: {
  runId: string; target: { path: string; start: number; end: number };
  onClose: () => void; onAsk: (reference: string) => void;
}) {
  const [file, setFile] = useState<RepoFileView | null>(null);
  const [error, setError] = useState('');
  const [selection, setSelection] = useState<[number, number]>([target.start, target.end]);
  const first = useRef<HTMLDivElement>(null);
  useEffect(() => {
    let active = true;
    setFile(null);
    setError('');
    setSelection([target.start, target.end]);
    practiceApi.file(runId, target.path, target.start, target.end)
      .then((data) => { if (active) { setFile(data); setSelection(data.highlight); } })
      .catch((reason) => { if (active) setError(errorMessage(reason, 'Could not load this file')); });
    return () => { active = false; };
  }, [runId, target.path, target.start, target.end]);
  useEffect(() => { first.current?.scrollIntoView({ block: 'center' }); }, [file]);
  useEffect(() => {
    const close = (event: KeyboardEvent) => { if (event.key === 'Escape') onClose(); };
    window.addEventListener('keydown', close);
    return () => window.removeEventListener('keydown', close);
  }, [onClose]);
  const pick = (line: number, extend: boolean) => setSelection(extend ? [Math.min(selection[0], line), Math.max(selection[0], line)] : [line, line]);
  return <div role="dialog" aria-modal="true" aria-label={target.path} className="fixed inset-0 z-50 flex items-center justify-center bg-black/30 p-4" onClick={onClose}>
    <div className="flex max-h-[85vh] w-full max-w-4xl flex-col rounded-2xl bg-white shadow-xl" onClick={(event) => event.stopPropagation()}>
      <div className="flex items-center justify-between gap-3 border-b px-4 py-2">
        <p className="break-all font-mono text-sm">{target.path}{file ? <span className="text-gray-400"> · {file.total} lines</span> : null}</p>
        <div className="flex shrink-0 items-center gap-3 text-xs">
          {file && <a href={file.url} target="_blank" rel="noopener noreferrer" className="text-sky-700 hover:underline">GitHub ↗</a>}
          <button type="button" onClick={onClose} className="btn-secondary btn-sm">Close</button>
        </div>
      </div>
      <div className="flex-1 overflow-auto py-2 font-mono text-xs">
        {error && <p className="px-4 text-rose-700">{error}</p>}
        {!file && !error && <p className="px-4 text-gray-400">Loading…</p>}
        {file?.lines.map((line, index) => {
          const number = file.start + index;
          const selected = number >= selection[0] && number <= selection[1];
          return <div key={number} ref={number === selection[0] ? first : undefined} className={`flex gap-3 px-2 ${selected ? 'bg-amber-100' : ''}`}>
            <button type="button" onClick={(event) => pick(number, event.shiftKey)} title="Click to select, Shift+click to extend"
              className="w-10 shrink-0 select-none text-right text-gray-400 hover:text-sky-700">{number}</button>
            <span className="whitespace-pre-wrap break-all text-gray-800">{line || ' '}</span>
          </div>;
        })}
      </div>
      <div className="flex items-center justify-between gap-3 border-t px-4 py-2 text-xs">
        <span className="text-gray-500">Click a line number to select it, Shift+click to extend.</span>
        <button type="button" disabled={!file} onClick={() => { onAsk(`[${target.path}:${selection[0]}-${selection[1]}] `); onClose(); }}
          className="btn-primary btn-sm">Ask about lines {selection[0]}–{selection[1]}</button>
      </div>
    </div>
  </div>;
}
