'use client';

import katex from 'katex';
import { useMemo } from 'react';

// Paper excerpts can be raw LaTeX source. Math is rendered with KaTeX; prose keeps
// its words but loses citation and cross-reference commands that mean nothing here.

type Segment = { math: boolean; value: string; display?: boolean };

const ENV_WRAP: Record<string, string> = { align: 'aligned', alignat: 'alignedat', eqnarray: 'aligned',
  gather: 'gathered', multline: 'gathered', split: 'aligned' };

// $$..$$, \[..\], \(..\), display environments, then $..$ (no space just inside the dollars,
// so prices such as "$5 and $10" stay text).
const MATH = new RegExp([
  String.raw`\$\$([\s\S]+?)\$\$`,
  String.raw`\\\[([\s\S]+?)\\\]`,
  String.raw`\\\(([\s\S]+?)\\\)`,
  String.raw`\\begin\{(equation|align|alignat|eqnarray|gather|multline|split)\*?\}([\s\S]+?)\\end\{\4\*?\}`,
  String.raw`\$(?!\s)((?:\\\$|[^$])+?)(?<!\s)\$`,
].join('|'), 'g');

function split(text: string): Segment[] {
  const segments: Segment[] = [];
  let last = 0;
  for (const match of Array.from(text.matchAll(MATH))) {
    const start = match.index ?? 0;
    if (start > last) segments.push({ math: false, value: text.slice(last, start) });
    const [, dollars, bracket, paren, env, envBody, inline] = match;
    let value = dollars ?? bracket ?? paren ?? envBody ?? inline ?? '';
    if (env && ENV_WRAP[env]) value = `\\begin{${ENV_WRAP[env]}}${value}\\end{${ENV_WRAP[env]}}`;
    segments.push({ math: true, value, display: inline === undefined && paren === undefined });
    last = start + match[0].length;
  }
  if (last < text.length) segments.push({ math: false, value: text.slice(last) });
  return segments;
}

function cleanProse(text: string): string {
  return text
    .replace(/\\(?:cite[tp]?|citeauthor|label|footnote)\*?(?:\[[^\]]*\])*\{[^}]*\}/g, '')
    .replace(/\\(?:eq|auto|c|C)?ref\{[^}]*\}/g, '(ref.)')
    .replace(/\\[vh]space\*?\{[^}]*\}|\\(?:noindent|centering|maketitle|small|footnotesize|newline)\b/g, '')
    .replace(/\\(?:emph|textbf|textit|texttt|textrm|textsc|underline|(?:sub)*section|paragraph|parhead)\*?\{([^}]*)\}/g, '$1')
    .replace(/\\begin\{(?:itemize|enumerate|description|algorithmic|algorithm)\}(?:\[[^\]]*\])?|\\end\{(?:itemize|enumerate|description|algorithmic|algorithm)\}/g, '')
    .replace(/\\item\s*/g, '• ')
    .replace(/~/g, ' ')
    .replace(/\\\\/g, ' ')
    .replace(/\\([%&_#$])/g, '$1')
    // LaTeX source wraps lines arbitrarily; only a blank line is a paragraph break
    .replace(/\n\s*\n/g, '\u2029').replace(/[ \t]*\n[ \t]*/g, ' ').replace(/\u2029/g, '\n\n');
}

function renderMath(value: string, display: boolean): string | null {
  try {
    return katex.renderToString(value.replace(/\\(?:label\{[^}]*\}|nonumber|notag)/g, ''),
      { displayMode: display, throwOnError: true, strict: 'ignore', trust: false, maxExpand: 500 });
  } catch {
    return null;  // unknown paper macro: show the source instead of a red error
  }
}

export function LatexText({ text, className }: { text: string; className?: string }) {
  const parts = useMemo(() => split(text || '').map((segment) => segment.math
    ? { ...segment, html: renderMath(segment.value, !!segment.display) }
    : { ...segment, value: cleanProse(segment.value), html: null }), [text]);
  return <span className={className}>
    {parts.map((part, index) => {
      if (!part.math) return <span key={index} className="whitespace-pre-line">{part.value}</span>;
      if (part.html === null) return <code key={index} className="break-all rounded bg-gray-100 px-1 text-xs">{part.value}</code>;
      return <span key={index} className={part.display ? 'my-1 block overflow-x-auto' : undefined}
        dangerouslySetInnerHTML={{ __html: part.html }} />;
    })}
  </span>;
}
