'use client';

import { HFPaper, GithubRepo, WhatsHotData } from '@/lib/api';
import { RefreshCw, ExternalLink, Bot, Code2 } from 'lucide-react';
import { arxivIdFromHotPaper } from './shared';

export function WhatsHotSection({
  data,
  isLoading,
  isRefreshing,
  onRefresh,
  onAskPaper,
  onPracticePaper,
  onPracticeRepo,
}: {
  data: WhatsHotData | null;
  isLoading: boolean;
  isRefreshing: boolean;
  onRefresh: () => void;
  onAskPaper: (paper: HFPaper) => void;
  onPracticePaper: (paper: HFPaper) => void;
  onPracticeRepo: (repo: GithubRepo) => void;
}) {
  return (
    <div className="animate-in">
      {/* Page header — states plainly that this surface is not personalized,
          which is the whole reason it lives apart from Daily Feed. */}
      <div className="flex items-start justify-between gap-4 mb-5">
        <div className="flex items-start gap-3">
          <div className="w-10 h-10 rounded-xl bg-orange-100 flex items-center justify-center text-xl flex-shrink-0">
            🔥
          </div>
          <div>
            <h2 className="text-lg font-bold text-gray-900 leading-tight">What&apos;s Hot</h2>
            <p className="text-sm text-gray-500 mt-0.5">
              What the ML community is reading and building right now — the same for
              everyone, not tailored to you.
              {data && (
                <span className="text-gray-400">
                  {' '}Tools ranked by stars gained{' '}
                  {data.github_window === 'daily' ? 'today' : 'this week'}.
                </span>
              )}
            </p>
          </div>
        </div>
        <button
          onClick={onRefresh}
          disabled={isRefreshing}
          title="Refresh What's Hot"
          className="p-2 rounded-lg text-gray-400 hover:text-orange-500 hover:bg-orange-50 transition-colors disabled:opacity-40 flex-shrink-0"
        >
          <RefreshCw className={`w-4 h-4 ${isRefreshing ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {/* Loading skeleton */}
      {isLoading && !data && (
        <div className="space-y-3">
          {[...Array(3)].map((_, i) => (
            <div key={i} className="bg-white rounded-2xl border border-gray-100 shadow-sm p-4 animate-pulse">
              <div className="h-4 bg-gray-100 rounded w-3/4 mb-2" />
              <div className="h-3 bg-gray-100 rounded w-full mb-1" />
              <div className="h-3 bg-gray-100 rounded w-5/6" />
            </div>
          ))}
        </div>
      )}

      {data && (
        <div className="space-y-6">
          {/* HuggingFace Trending Papers */}
          {data.hf_papers.length > 0 && (
            <div>
              <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-3 flex items-center gap-1.5">
                <span>🤗</span> Trending Papers
              </h3>
              <div className="space-y-3">
                {data.hf_papers.map((paper) => (
                  <HFPaperCard key={paper.id} paper={paper} onAsk={onAskPaper} onPractice={onPracticePaper} />
                ))}
              </div>
            </div>
          )}

          {/* GitHub Trending Repos */}
          {data.github_repos.length > 0 && (
            <div>
              <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-3 flex items-center gap-1.5">
                <span>🛠️</span> Trending Tools
              </h3>
              <div className="space-y-3">
                {data.github_repos.map((repo) => (
                  <GithubRepoCard key={repo.full_name} repo={repo} window={data.github_window} onPractice={onPracticeRepo} />
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
export function HFPaperCard({ paper, onAsk, onPractice }: { paper: HFPaper; onAsk: (paper: HFPaper) => void; onPractice: (paper: HFPaper) => void }) {
  return (
    <div className="bg-white rounded-2xl border border-gray-100 shadow-sm p-4 hover:border-orange-200 transition-colors">
      <div className="flex items-start justify-between gap-3 mb-2">
        <h4 className="text-sm font-semibold text-gray-900 leading-snug flex-1">{paper.title}</h4>
        {paper.upvotes > 0 && (
          <span className="flex-shrink-0 flex items-center gap-1 text-xs font-medium text-orange-600 bg-orange-50 px-2 py-0.5 rounded-full">
            ▲ {paper.upvotes}
          </span>
        )}
      </div>
      {paper.digest && (
        <p className="text-sm text-gray-500 leading-relaxed mb-3">{paper.digest}</p>
      )}
      <div className="flex items-center gap-3 text-xs text-gray-400">
        <span className="font-medium text-gray-500">HuggingFace</span>
        <a
          href={paper.url}
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-0.5 hover:text-orange-500 transition-colors"
        >
          HF <ExternalLink className="w-3 h-3" />
        </a>
        {paper.arxiv_url && (
          <a
            href={paper.arxiv_url}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-0.5 hover:text-orange-500 transition-colors"
          >
            arXiv <ExternalLink className="w-3 h-3" />
          </a>
        )}
        {paper.authors.length > 0 && (
          <span className="truncate max-w-[200px]">{paper.authors.slice(0, 2).join(', ')}</span>
        )}
      </div>
      <div className="mt-3 flex flex-wrap gap-2 border-t border-gray-100 pt-3">
        <button type="button" onClick={() => onAsk(paper)} className="btn-secondary text-xs py-1.5"><Bot className="w-3.5 h-3.5" /> Ask Agent</button>
        <button type="button" onClick={() => onPractice(paper)} disabled={!arxivIdFromHotPaper(paper)} title={arxivIdFromHotPaper(paper) ? 'Build a runnable algorithm from this paper' : 'An arXiv ID is needed to build from this paper'} className="btn-secondary text-xs py-1.5 disabled:cursor-not-allowed disabled:opacity-40"><Code2 className="w-3.5 h-3.5" /> Build algorithm</button>
      </div>
    </div>
  );
}
const compact = (n: number) => (n >= 1000 ? `${(n / 1000).toFixed(1)}k` : `${n}`);

export function GithubRepoCard({ repo, window, onPractice }: { repo: GithubRepo; window?: string; onPractice: (repo: GithubRepo) => void }) {
  return (
    <div className="bg-white rounded-2xl border border-gray-100 shadow-sm p-4 hover:border-orange-200 transition-colors">
      <div className="flex items-start justify-between gap-3 mb-2">
        <div className="flex-1 min-w-0">
          <a
            href={repo.url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-sm font-semibold text-gray-900 hover:text-orange-600 transition-colors flex items-center gap-1"
          >
            {repo.name}
            <ExternalLink className="w-3 h-3 flex-shrink-0" />
          </a>
          <span className="text-xs text-gray-400">{repo.full_name.split('/')[0]}</span>
        </div>
        <div className="flex items-center gap-2 flex-shrink-0">
          {repo.language && (
            <span className="text-xs font-medium px-2 py-0.5 rounded-full bg-sky-50 text-sky-600">
              {repo.language}
            </span>
          )}
          {repo.stars_gained ? (
            <span className="text-xs font-semibold px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-700">
              +{compact(repo.stars_gained)} {window === 'daily' ? 'today' : 'this week'}
            </span>
          ) : repo.trend_basis === 'new' ? (
            <span className="text-xs font-semibold px-2 py-0.5 rounded-full bg-violet-50 text-violet-700" title="Created in the last 30 days">
              New
            </span>
          ) : null}
          <span className="flex items-center gap-1 text-xs text-yellow-600 font-medium" title="Total stars">
            ★ {compact(repo.stars)}
          </span>
        </div>
      </div>
      {repo.description && (
        <p className="text-sm text-gray-500 leading-relaxed">{repo.description}</p>
      )}
      <div className="mt-3 flex flex-wrap gap-2 border-t border-gray-100 pt-3">
        {/* One way into a repository: the repository agent reads the code and cites what it read. */}
        <button type="button" onClick={() => onPractice(repo)} className="btn-secondary text-xs py-1.5"><Code2 className="w-3.5 h-3.5" /> Explore code</button>
        <span className="self-center text-xs text-gray-400">Ask questions about the code there</span>
      </div>
    </div>
  );
}
