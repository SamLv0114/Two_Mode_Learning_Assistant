'use client';

import { useEffect, useState } from 'react';
import { feedApi, Paper } from '@/lib/api';
import { Bookmark, Bot, Code2, ExternalLink } from 'lucide-react';

/** Papers the learner saved, with the same next steps a feed card offers. */
export function SavedItems({ onAskAgent, onPractice, onGoToFeed }: {
  onAskAgent: (paper: Paper) => void;
  onPractice: (paper: Paper) => void;
  onGoToFeed: () => void;
}) {
  const [papers, setPapers] = useState<Paper[] | null>(null);

  useEffect(() => {
    feedApi.getSaved()
      .then((data) => setPapers(data.papers))
      .catch((error) => { console.error('Failed to load saved papers:', error); setPapers([]); });
  }, []);

  if (papers === null) {
    return (
      <div className="text-center py-12">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600 mx-auto"></div>
      </div>
    );
  }

  if (papers.length === 0) {
    return (
      <div className="text-center py-12">
        <Bookmark className="w-10 h-10 text-gray-200 mx-auto mb-3" />
        <p className="font-medium text-gray-600">No saved papers yet</p>
        <p className="text-sm text-gray-400 mt-1">Saving a paper also teaches your feed what you want more of.</p>
        <button type="button" onClick={onGoToFeed} className="btn-primary btn-sm mt-4">Go to Daily Feed</button>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <h2 className="text-base font-bold text-gray-900">
        Saved papers <span className="ml-1 text-sm font-normal text-gray-400">({papers.length})</span>
      </h2>
      {papers.map((paper) => (
        <div key={paper.id} className="card">
          <h3 className="font-semibold text-gray-900">{paper.title}</h3>
          <p className="mt-0.5 text-xs text-gray-400">arXiv:{paper.arxiv_id}{paper.authors ? ` · ${paper.authors.split(',').slice(0, 3).join(',')}` : ''}</p>
          {paper.abstract && <p className="mt-2 text-sm text-gray-500 line-clamp-2">{paper.abstract}</p>}
          <div className="mt-3 flex flex-wrap gap-2 border-t border-gray-100 pt-3">
            <button type="button" onClick={() => onAskAgent(paper)} className="btn-secondary text-xs py-1.5"><Bot className="w-3.5 h-3.5" /> Ask Agent</button>
            <button type="button" onClick={() => onPractice(paper)} className="btn-secondary text-xs py-1.5"><Code2 className="w-3.5 h-3.5" /> Build algorithm</button>
            {paper.arxiv_url && (
              <a href={paper.arxiv_url} target="_blank" rel="noopener noreferrer" className="btn-link text-xs inline-flex items-center gap-1">
                Open paper <ExternalLink className="w-3 h-3" />
              </a>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}
