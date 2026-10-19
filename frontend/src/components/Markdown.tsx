'use client';

import { ReactNode } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { RepoCitation } from '@/lib/api';

type OpenFile = (path: string, start?: number, end?: number) => void;

const CITATION = /\[([^[\]\s:]+):(\d+)(?:-(\d+))?\](?!\()/g;
// Fenced and inline code are left alone: `a[i:3]` in code is an index, not a citation.
const CODE = /(```[\s\S]*?(?:```|$)|`[^`\n]*`)/g;

/** Turn [path:12-30] citations into links the renderer can recognise, outside code only. */
function linkCitations(text: string): string {
  return text.split(CODE).map((part, index) => index % 2 === 1 ? part
    : part.replace(CITATION, (match) => `[${match.slice(1, -1)}](#cite:${encodeURIComponent(match)})`)).join('');
}

/**
 * Agent answers are Markdown (headings, lists, code fences). Raw HTML is not rendered, so text
 * from repositories or papers quoted in an answer cannot inject markup.
 * `codeCitations` turns [path:12-30] into code links; turn it off for answers that are not about a repository.
 */
export function Markdown({ text, citations, onOpenFile, codeCitations = true }: {
  text: string; citations?: RepoCitation[]; onOpenFile?: OpenFile; codeCitations?: boolean;
}) {
  const byText = new Map((citations || []).map((item) => [item.text, item]));
  const cite = (raw: string, children: ReactNode) => {
    const citation = byText.get(raw);
    if (citation?.verified && citation.url) {
      return onOpenFile
        ? <button type="button" onClick={() => onOpenFile(citation.path, citation.start, citation.end)} className="rounded bg-sky-50 px-1 font-mono text-[0.8em] text-sky-700 hover:underline">{children}</button>
        : <a href={citation.url} target="_blank" rel="noopener noreferrer" className="rounded bg-sky-50 px-1 font-mono text-[0.8em] text-sky-700 hover:underline">{children}</a>;
    }
    return <span title="Not among the lines the agent read for this answer" className="rounded bg-amber-50 px-1 font-mono text-[0.8em] text-amber-800">{children} (unverified)</span>;
  };
  return <div className="space-y-2 break-words text-sm leading-relaxed text-gray-700">
    <ReactMarkdown remarkPlugins={[remarkGfm]} urlTransform={(url) => url} components={{
      a: ({ href, children }) => href?.startsWith('#cite:')
        ? cite(decodeURIComponent(href.slice(6)), children)
        : /^https?:\/\//.test(href || '')
          ? <a href={href} target="_blank" rel="noopener noreferrer" className="text-sky-700 underline">{children}</a>
          : <span>{children}</span>,
      h1: ({ children }) => <h3 className="pt-1 text-base font-semibold text-gray-900">{children}</h3>,
      h2: ({ children }) => <h3 className="pt-1 text-base font-semibold text-gray-900">{children}</h3>,
      h3: ({ children }) => <h4 className="pt-1 text-sm font-semibold text-gray-900">{children}</h4>,
      h4: ({ children }) => <h4 className="pt-1 text-sm font-semibold text-gray-900">{children}</h4>,
      ul: ({ children }) => <ul className="list-disc space-y-1 pl-5">{children}</ul>,
      ol: ({ children }) => <ol className="list-decimal space-y-1 pl-5">{children}</ol>,
      blockquote: ({ children }) => <blockquote className="border-l-2 border-gray-300 pl-3 text-gray-600">{children}</blockquote>,
      pre: ({ children }) => <pre className="overflow-x-auto rounded-lg bg-gray-900 p-3 text-xs leading-snug text-gray-100 [&_code]:bg-transparent [&_code]:p-0 [&_code]:text-gray-100">{children}</pre>,
      code: ({ children }) => <code className="rounded bg-gray-100 px-1 py-0.5 font-mono text-[0.85em] text-gray-800">{children}</code>,
      table: ({ children }) => <div className="overflow-x-auto"><table className="min-w-full border-collapse text-xs">{children}</table></div>,
      th: ({ children }) => <th className="border border-gray-200 bg-gray-50 px-2 py-1 text-left font-semibold">{children}</th>,
      td: ({ children }) => <td className="border border-gray-200 px-2 py-1 align-top">{children}</td>,
    }}>{codeCitations ? linkCitations(text) : text}</ReactMarkdown>
    {!!citations?.length && <p className="text-[11px] text-gray-400">{citations.filter((c) => c.verified).length} of {citations.length} references match lines the agent actually read at the pinned commit.</p>}
  </div>;
}
