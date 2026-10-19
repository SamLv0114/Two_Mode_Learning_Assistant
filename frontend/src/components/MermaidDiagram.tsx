'use client';

import { useEffect, useState } from 'react';
import { CodeMap, RepoOverview } from '@/lib/api';

// Labels come from repository paths and paper text: keep only plain characters so nothing in them
// can be read as Mermaid syntax or markup (Mermaid's strict mode sanitizes too).
const label = (text: string, max = 42) => {
  const clean = text.replace(/\\[a-zA-Z]+/g, ' ').replace(/[^\w ./:+=,()-]/g, ' ').replace(/\s+/g, ' ').trim();
  return clean.length > max ? `${clean.slice(0, max - 1)}…` : clean;
};
const shortPath = (path: string) => path.split('/').slice(-2).join('/');

/** Modules as boxes, arrows from a file to a module it imports; entry points highlighted. */
export function repoDiagram(overview: RepoOverview): string | null {
  const entries = overview.entry_points.filter((e) => e.kind !== 'npm_script').map((e) => e.path);
  const nodes = Array.from(new Set([...entries, ...overview.central_modules.map((c) => c.path)])).slice(0, 14);
  const edges = overview.import_edges.filter(([a, b]) => nodes.includes(a) && nodes.includes(b));
  if (!edges.length) return null;
  const used = nodes.filter((node) => edges.some(([a, b]) => a === node || b === node));
  const id = (path: string) => `n${used.indexOf(path)}`;
  return ['graph LR',
    ...used.map((path) => `  ${id(path)}["${label(shortPath(path))}"]`),
    ...edges.map(([a, b]) => `  ${id(a)} --> ${id(b)}`),
    '  classDef entry fill:#e0f2fe,stroke:#0284c7,color:#0c4a6e',
    ...(used.some((p) => entries.includes(p)) ? [`  class ${used.filter((p) => entries.includes(p)).map(id).join(',')} entry`] : []),
  ].join('\n');
}

/** Paper excerpts on the left, the functions whose lines cite them on the right. */
export function paperDiagram(map: CodeMap, details: { detail: string; evidence_ids: string[] }[]): string | null {
  const pairs = Array.from(new Set(map.links.map((link) => `${link.evidence_id}|${link.function}`))).map((pair) => pair.split('|'));
  if (!pairs.length) return null;
  const evidence = Array.from(new Set(pairs.map(([e]) => e))).slice(0, 12);
  const functions = Array.from(new Set(pairs.filter(([e]) => evidence.includes(e)).map(([, f]) => f)));
  const describe = (id: string) => details.find((d) => d.evidence_ids.includes(id))?.detail || '';
  return ['graph LR',
    ...evidence.map((e, i) => `  e${i}["${label(`${e} ${describe(e)}`, 48)}"]`),
    ...functions.map((f, i) => `  f${i}["${label(f === '(module level)' ? 'module level' : `${f}()`)}"]`),
    ...pairs.filter(([e]) => evidence.includes(e)).map(([e, f]) => `  e${evidence.indexOf(e)} --> f${functions.indexOf(f)}`),
    '  classDef paper fill:#f5f3ff,stroke:#7c3aed,color:#3b0764',
    `  class ${evidence.map((_, i) => `e${i}`).join(',')} paper`,
  ].join('\n');
}

let counter = 0;

export function MermaidDiagram({ definition, caption }: { definition: string; caption: string }) {
  const [svg, setSvg] = useState('');
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    let active = true;
    import('mermaid').then(async ({ default: mermaid }) => {
      mermaid.initialize({ startOnLoad: false, securityLevel: 'strict', theme: 'neutral', flowchart: { htmlLabels: false } });
      const { svg: rendered } = await mermaid.render(`diagram-${++counter}`, definition);
      if (active) setSvg(rendered);
    }).catch(() => { if (active) setFailed(true); });
    return () => { active = false; };
  }, [definition]);
  if (failed) return null;
  return <figure className="rounded-xl border border-gray-200 p-3">
    {svg ? <div className="overflow-x-auto [&_svg]:mx-auto [&_svg]:max-w-full" dangerouslySetInnerHTML={{ __html: svg }} />
      : <p className="text-xs text-gray-400">Drawing diagram…</p>}
    <figcaption className="mt-1 text-[11px] text-gray-500">{caption}</figcaption>
  </figure>;
}
