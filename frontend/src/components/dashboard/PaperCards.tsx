'use client';

import { useState } from 'react';
import { Paper, PaperReasons } from '@/lib/api';
import { Save, X, ExternalLink, Bot, Code2, MessageSquare, Send } from 'lucide-react';

const FIT_BADGE: Record<number, { label: string; style: string }> = {
  3: { label: 'On your goal', style: 'bg-emerald-50 text-emerald-700 border-emerald-100' },
  2: { label: 'Useful for your goal', style: 'bg-sky-50 text-sky-700 border-sky-100' },
  1: { label: 'Nearby topic', style: 'bg-gray-50 text-gray-600 border-gray-200' },
  0: { label: 'Off your goal', style: 'bg-rose-50 text-rose-700 border-rose-100' },
};
const CHANNEL_LABEL: Record<string, string> = {
  content: 'Profile match', graph: 'Citation graph', trending: 'Trending', latest: 'New',
};
const READINESS_LABEL: Record<string, string> = { too_basic: 'Likely familiar', too_advanced: 'Advanced read' };

/** Why the paper is in this feed: the model's reason and what found it (feed/judge.py). */
export function WhyThisPaper({ reasons }: { reasons: PaperReasons }) {
  const fit = reasons.goal_fit != null ? FIT_BADGE[reasons.goal_fit] : null;
  return (
    <div className="mb-3 rounded-xl border border-indigo-100 bg-indigo-50/50 px-3 py-2.5">
      <div className="flex flex-wrap items-center gap-1.5">
        <span className="text-xs font-semibold text-indigo-900">Why this paper</span>
        {fit && <span className={`rounded-full border px-2 py-0.5 text-[11px] font-semibold ${fit.style}`}>{fit.label}</span>}
        {reasons.readiness && READINESS_LABEL[reasons.readiness] && (
          <span className="rounded-full bg-amber-50 px-2 py-0.5 text-[11px] font-medium text-amber-700">{READINESS_LABEL[reasons.readiness]}</span>
        )}
        {reasons.paper_type && <span className="rounded-full bg-white px-2 py-0.5 text-[11px] capitalize text-gray-500">{reasons.paper_type}</span>}
        {reasons.channels.map((c) => (
          <span key={c} className="rounded-full bg-white px-2 py-0.5 text-[11px] text-indigo-600">{CHANNEL_LABEL[c] || c}</span>
        ))}
      </div>
      {reasons.reason && <p className="mt-1.5 text-sm text-gray-800">{reasons.reason}</p>}
      {reasons.evidence.map((e) => <p key={e} className="mt-0.5 text-xs text-gray-500">{e}</p>)}
    </div>
  );
}

export function FormattedSummary({ text }: { text: string }) {
  // Parse the summary into sections based on common patterns
  // The LLM returns: **Key Insight:** ..., **Why They Should Care:** ..., **Relation to Their Interests:** ...
  const sections: { label: string; content: string; color: string }[] = [];
  let remaining = text;

  const patterns = [
    { regex: /\*\*Key Insight:\*\*\s*/i, label: 'Key Insight', color: 'text-amber-700 bg-amber-50' },
    { regex: /\*\*Why (?:They |You )?(Should )?Care:\*\*\s*/i, label: 'Why It Matters', color: 'text-blue-700 bg-blue-50' },
    { regex: /\*\*(?:Relation to (?:Their |Your )?Interests|How [Ii]t [Rr]elates).*?:\*\*\s*/i, label: 'Relevance to You', color: 'text-green-700 bg-green-50' },
  ];

  // Try to split into structured sections
  const allPatternStarts: { idx: number; len: number; patternIdx: number }[] = [];
  for (let p = 0; p < patterns.length; p++) {
    const match = patterns[p].regex.exec(remaining);
    if (match && match.index !== undefined) {
      allPatternStarts.push({ idx: match.index, len: match[0].length, patternIdx: p });
    }
  }

  if (allPatternStarts.length >= 2) {
    // Sort by position
    allPatternStarts.sort((a, b) => a.idx - b.idx);

    for (let i = 0; i < allPatternStarts.length; i++) {
      const start = allPatternStarts[i].idx + allPatternStarts[i].len;
      const end = i + 1 < allPatternStarts.length ? allPatternStarts[i + 1].idx : remaining.length;
      const content = remaining.slice(start, end).trim();
      const pat = patterns[allPatternStarts[i].patternIdx];
      sections.push({ label: pat.label, content, color: pat.color });
    }
  }

  // If we couldn't parse structured sections, show as-is with basic bold rendering
  if (sections.length === 0) {
    return (
      <div className="text-sm text-gray-700 leading-relaxed" dangerouslySetInnerHTML={{
        __html: text
          .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
          .replace(/\n/g, '<br/>')
      }} />
    );
  }

  return (
    <div className="space-y-2">
      {sections.map((section, i) => (
        <div key={i} className="flex gap-2 items-start">
          <span className={`shrink-0 text-xs font-semibold px-2 py-0.5 rounded ${section.color}`}>
            {section.label}
          </span>
          <p className="text-sm text-gray-700 leading-relaxed">{section.content}</p>
        </div>
      ))}
    </div>
  );
}

/* ── Paper Card ── */
export function PaperCard({
  paper,
  onInteraction,
  onAskAgent,
  onPractice,
  onFeedback,
}: {
  paper: Paper;
  onInteraction: (paperId: number, interaction: 'viewed' | 'saved' | 'dismissed') => void;
  onAskAgent: (ctx: { title: string; abstract: string | null; arxivId: string; arxivUrl: string }) => void;
  onPractice: () => void;
  /** Plain-language feedback about this paper, applied to the research profile. Hidden when absent. */
  onFeedback?: (text: string, paperId: number) => Promise<boolean>;
}) {
  const [feedbackOpen, setFeedbackOpen] = useState(false);
  const [feedback, setFeedback] = useState('');
  const [sending, setSending] = useState(false);
  const sendFeedback = async () => {
    if (!onFeedback || feedback.trim().length < 3) return;
    setSending(true);
    if (await onFeedback(feedback.trim(), paper.id)) {
      setFeedback('');
      setFeedbackOpen(false);
    }
    setSending(false);
  };
  return (
    <div
      className="card-lift animate-in"
      style={{ borderLeft: '3px solid #0ea5e9' }}
    >
      <div className="flex-1">
        <div className="flex items-start justify-between gap-3 mb-2">
          <h3 className="font-semibold text-gray-900 leading-snug">
            <span className="text-gray-400 mr-1.5">#{paper.rank}</span>
            {paper.title}
          </h3>
          {paper.citation_count > 50 && (
            <span className="shrink-0 text-xs font-semibold px-2 py-0.5 rounded-full bg-amber-50 text-amber-700 border border-amber-100">
              ⭐ {paper.citation_count} citations
            </span>
          )}
        </div>
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-gray-400 mb-3">
          <span>arXiv:{paper.arxiv_id}</span>
          <span className="text-primary-600 font-medium">Score: {paper.relevance_score.toFixed(3)}</span>
          {paper.impact_score != null && (
            <span>Impact: {paper.impact_score.toFixed(2)}</span>
          )}
        </div>
        {paper.reasons && <WhyThisPaper reasons={paper.reasons} />}
        {paper.summary && <FormattedSummary text={paper.summary} />}
      </div>
      {feedbackOpen && (
        <div className="mt-3 flex gap-2">
          <input autoFocus value={feedback} onChange={(e) => setFeedback(e.target.value)} className="input-field text-sm py-2"
            placeholder="What is off about this one? e.g. “too theoretical”, “I already know this”"
            onKeyDown={(e) => { if (e.key === 'Enter') sendFeedback(); }} />
          <button onClick={sendFeedback} disabled={sending || feedback.trim().length < 3} className="btn-secondary btn-sm shrink-0">
            <Send className={`w-3.5 h-3.5 ${sending ? 'animate-pulse' : ''}`} /> {sending ? 'Applying…' : 'Apply'}
          </button>
        </div>
      )}
      <div className="flex items-center gap-2 mt-4 pt-4 border-t border-gray-100">
        <a
          href={paper.arxiv_url}
          target="_blank"
          rel="noopener noreferrer"
          onClick={() => onInteraction(paper.id, 'viewed')}
          className="btn-secondary text-sm py-1.5"
        >
          <ExternalLink className="w-3.5 h-3.5" />
          View paper
        </a>
        <button
          onClick={() => onInteraction(paper.id, 'saved')}
          className="btn-secondary text-sm py-1.5"
        >
          <Save className="w-3.5 h-3.5" />
          Save
        </button>
        <button
          onClick={() => onAskAgent({
            title: paper.title,
            abstract: paper.abstract ?? null,
            arxivId: paper.arxiv_id,
            arxivUrl: paper.arxiv_url ?? '',
          })}
          className="btn-secondary text-sm py-1.5 text-primary-600 border-primary-200 hover:bg-primary-50"
        >
          <Bot className="w-3.5 h-3.5" />
          Ask Agent
        </button>
        <button onClick={onPractice} className="btn-secondary text-sm py-1.5"><Code2 className="w-3.5 h-3.5" /> Build algorithm</button>
        {onFeedback && (
          <button onClick={() => setFeedbackOpen(!feedbackOpen)} className="btn-secondary text-sm py-1.5" title="Tell the recommender what is off">
            <MessageSquare className="w-3.5 h-3.5" /> Not quite right?
          </button>
        )}
        <button
          onClick={() => onInteraction(paper.id, 'dismissed')}
          className="ml-auto text-gray-300 hover:text-rose-400 p-2 rounded-lg hover:bg-rose-50 transition-colors"
          title="Dismiss"
        >
          <X className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
}
