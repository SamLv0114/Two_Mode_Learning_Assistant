'use client';

import { useState } from 'react';
import toast from 'react-hot-toast';
import { ExternalLink, Route, Save } from 'lucide-react';
import { errorMessage, profileApi, ReadingPath as Path } from '@/lib/api';

const STAGE_STYLE: Record<string, string> = {
  foundation: 'bg-amber-50 text-amber-700',
  core: 'bg-sky-50 text-sky-700',
  frontier: 'bg-violet-50 text-violet-700',
};

/** On demand: an ordered path of papers from what the reader knows to a topic they name. */
export function ReadingPathPanel({ onSave }: { onSave: (paperId: number) => void }) {
  const [topic, setTopic] = useState('');
  const [path, setPath] = useState<Path | null>(null);
  const [building, setBuilding] = useState(false);

  const build = async () => {
    setBuilding(true);
    setPath(null);
    try {
      setPath(await profileApi.readingPath(topic.trim()));
    } catch (error) {
      toast.error(errorMessage(error, 'Could not build a reading path'));
    } finally {
      setBuilding(false);
    }
  };

  return (
    <div className="card mt-8">
      <div className="flex items-center gap-2 mb-1">
        <Route className="w-4 h-4 text-primary-600" />
        <h2 className="text-base font-bold text-gray-900">Plan a reading path</h2>
      </div>
      <p className="text-sm text-gray-500 mb-3">
        Name something you want to learn. The agent searches, reads what the central papers cite, and orders
        5 to 8 papers from foundations to the current frontier, skipping what your profile says you know.
      </p>
      <div className="flex gap-2">
        <input value={topic} onChange={(e) => setTopic(e.target.value)} className="input-field text-sm py-2"
          placeholder="e.g. understand diffusion models well enough to implement DDPM"
          onKeyDown={(e) => { if (e.key === 'Enter' && topic.trim().length >= 3 && !building) build(); }} />
        <button type="button" onClick={build} disabled={building || topic.trim().length < 3} className="btn-primary text-sm shrink-0">
          <Route className={`w-4 h-4 ${building ? 'animate-pulse' : ''}`} /> {building ? 'Building…' : 'Build path'}
        </button>
      </div>
      {building && <p className="text-xs text-gray-400 mt-2">Searching and reading references takes 20 to 60 seconds.</p>}

      {path && (
        <div className="mt-5 space-y-3">
          {path.library_only && (
            <p className="notice notice-warn">
              Semantic Scholar could not be reached, so this path only uses papers already in this library, which are
              mostly recent. Foundational papers may be missing. Try again in a minute.
            </p>
          )}
          {path.ordered_by === 'year' && (
            <p className="notice notice-info">The ordering step failed, so these are in publication order without explanations.</p>
          )}
          {path.steps.length === 0 && <p className="text-sm text-gray-500">No papers found for this topic.</p>}
          <ol className="space-y-3">
            {path.steps.map((step, i) => (
              <li key={`${step.paper.title}-${i}`} className="panel flex gap-3">
                <span className="w-6 h-6 shrink-0 rounded-full bg-gray-100 text-gray-600 text-xs font-bold flex items-center justify-center">{i + 1}</span>
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className={`rounded-full px-2 py-0.5 text-[11px] font-semibold capitalize ${STAGE_STYLE[step.stage]}`}>{step.stage}</span>
                    <a href={step.paper.url} target="_blank" rel="noopener noreferrer"
                      className="font-semibold text-sm text-gray-900 hover:text-primary-600">{step.paper.title}</a>
                  </div>
                  <p className="text-xs text-gray-400 mt-0.5">
                    {step.paper.year ?? 'year unknown'} · {step.paper.citation_count.toLocaleString()} citations
                  </p>
                  {step.why && <p className="text-sm text-gray-700 mt-1">{step.why}</p>}
                </div>
                <div className="flex shrink-0 items-start gap-1">
                  <a href={step.paper.url} target="_blank" rel="noopener noreferrer" className="p-1.5 text-gray-400 hover:text-gray-700" aria-label="Open paper">
                    <ExternalLink className="w-4 h-4" />
                  </a>
                  {step.paper.db_id != null && (
                    <button type="button" onClick={() => onSave(step.paper.db_id as number)} className="p-1.5 text-gray-400 hover:text-emerald-600" aria-label="Save paper">
                      <Save className="w-4 h-4" />
                    </button>
                  )}
                </div>
              </li>
            ))}
          </ol>
          {path.skipped && <p className="text-xs text-gray-500">Skipped: {path.skipped}</p>}
        </div>
      )}
    </div>
  );
}
