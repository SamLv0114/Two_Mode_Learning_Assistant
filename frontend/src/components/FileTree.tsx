'use client';

import { useMemo, useState } from 'react';

type Node = { name: string; path: string; children: Map<string, Node>; files: number };

function buildTree(paths: string[]): Node {
  const root: Node = { name: '', path: '', children: new Map(), files: 0 };
  for (const path of paths) {
    let node = root;
    node.files += 1;
    const parts = path.split('/');
    parts.forEach((part, index) => {
      const childPath = parts.slice(0, index + 1).join('/');
      if (!node.children.has(part)) node.children.set(part, { name: part, path: childPath, children: new Map(), files: 0 });
      node = node.children.get(part)!;
      if (index < parts.length - 1) node.files += 1;
    });
  }
  return root;
}

function Branch({ node, depth, open, toggle, onOpenFile, highlight }: {
  node: Node; depth: number; open: Set<string>; toggle: (path: string) => void;
  onOpenFile: (path: string) => void; highlight: Set<string>;
}) {
  const entries = Array.from(node.children.values()).sort((a, b) =>
    (a.children.size ? 0 : 1) - (b.children.size ? 0 : 1) || a.name.localeCompare(b.name));
  return <ul>{entries.map((child) => {
    const folder = child.children.size > 0;
    const expanded = open.has(child.path);
    return <li key={child.path}>
      <button type="button" onClick={() => folder ? toggle(child.path) : onOpenFile(child.path)}
        style={{ paddingLeft: `${depth * 14 + 4}px` }}
        className={`flex w-full items-center gap-1.5 rounded py-0.5 pr-2 text-left hover:bg-gray-100 ${highlight.has(child.path) ? 'font-semibold text-sky-800' : 'text-gray-700'}`}>
        <span className="w-3 shrink-0 text-gray-400">{folder ? (expanded ? '▾' : '▸') : ''}</span>
        <span className="truncate">{child.name}{folder ? '/' : ''}</span>
        {folder && <span className="ml-auto shrink-0 text-[10px] text-gray-400">{child.files}</span>}
      </button>
      {folder && expanded && <Branch node={child} depth={depth + 1} open={open} toggle={toggle} onOpenFile={onOpenFile} highlight={highlight} />}
    </li>;
  })}</ul>;
}

/** Collapsible file tree of the pinned snapshot; core modules and entry points are emphasised. */
export function FileTree({ paths, focus, highlight, onOpenFile, total }: {
  paths: string[]; focus?: string; highlight: string[]; onOpenFile: (path: string) => void; total?: number;
}) {
  const tree = useMemo(() => buildTree(paths), [paths]);
  const marked = useMemo(() => new Set(highlight.flatMap((path) =>
    path.split('/').map((_, i, parts) => parts.slice(0, i + 1).join('/')))), [highlight]);
  // Start with the focus folder and the folders holding the core modules open.
  const [open, setOpen] = useState<Set<string>>(() => new Set(Array.from(marked).filter((p) => !highlight.includes(p))
    .concat(focus ? focus.split('/').map((_, i, parts) => parts.slice(0, i + 1).join('/')) : [])));
  const [filter, setFilter] = useState('');
  const toggle = (path: string) => setOpen((old) => { const next = new Set(old); if (next.has(path)) next.delete(path); else next.add(path); return next; });
  const matches = filter.trim() ? paths.filter((path) => path.toLowerCase().includes(filter.trim().toLowerCase())).slice(0, 200) : [];
  return <div className="space-y-2 text-xs">
    <input value={filter} onChange={(event) => setFilter(event.target.value)} placeholder="Filter files, e.g. lora/layer"
      className="input-field !py-1.5 text-xs" aria-label="Filter files" />
    <div className="max-h-[28rem] overflow-y-auto rounded-lg border border-gray-100 p-1 font-mono">
      {filter.trim()
        ? matches.length
          ? <ul>{matches.map((path) => <li key={path}><button type="button" onClick={() => onOpenFile(path)} className="w-full truncate rounded px-1 py-0.5 text-left text-gray-700 hover:bg-gray-100">{path}</button></li>)}</ul>
          : <p className="p-2 font-sans text-gray-400">No file matches.</p>
        : <Branch node={tree} depth={0} open={open} toggle={toggle} onOpenFile={onOpenFile} highlight={marked} />}
    </div>
    <p className="text-gray-500">{total && total > paths.length ? `Showing ${paths.length} of ${total} files. ` : ''}Bold: entry points and core modules. Click a file to read it.</p>
  </div>;
}
