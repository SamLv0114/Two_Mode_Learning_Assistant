'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/lib/auth';
import Link from 'next/link';

const DEMO_PAPERS = [
  {
    title: 'Attention Is All You Need',
    source: 'ArXiv · cs.LG',
    score: '0.941',
    summary: {
      insight: 'Introduces the Transformer — replacing recurrence entirely with self-attention for sequence modeling.',
      why: 'Transformers are now the backbone of every major LLM including GPT and BERT.',
      relevance: 'Directly relevant to your interest in NLP and deep learning architectures.',
    },
    badge: { label: '⭐ High Impact', color: 'bg-amber-100 text-amber-800' },
  },
  {
    title: 'LightGBM: A Highly Efficient Gradient Boosting Decision Tree',
    source: 'ArXiv · cs.LG',
    score: '0.887',
    summary: {
      insight: 'Histogram-based algorithms that dramatically speed up gradient boosting with lower memory.',
      why: 'LightGBM is the go-to model for tabular data and learning-to-rank in production ML.',
      relevance: 'Matches your interests in efficient ML and recommendation systems.',
    },
    badge: { label: '💻 Code Available', color: 'bg-emerald-100 text-emerald-800' },
  },
];

const DEMO_ARTICLE = {
  title: 'How Retrieval-Augmented Generation Actually Works',
  source: 'Hacker News · 342 points',
  score: '0.863',
  summary: {
    insight: 'Covers the full RAG pipeline from document chunking to vector retrieval and LLM augmentation.',
    why: 'RAG is the standard approach for Q&A systems over private knowledge bases.',
    relevance: 'Highly relevant to your work on knowledge base assistants.',
  },
  badge: { label: '🔥 Trending', color: 'bg-red-100 text-red-800' },
};

const FEATURES = [
  {
    icon: '📚',
    title: 'Personalized Daily Feed',
    description: 'ML-ranked papers from ArXiv and tech articles from Hacker News — tailored to your research focus.',
    accent: 'from-sky-400 to-primary-500',
  },
  {
    icon: '🧠',
    title: 'Learns From You',
    description: 'Your own LightGBM model trains on your saves and dismissals. The more you interact, the smarter it gets.',
    accent: 'from-violet-400 to-indigo-500',
  },
  {
    icon: '💬',
    title: 'Q&A Over Your Documents',
    description: 'Upload papers and notes. Ask questions and get cited answers backed by your personal knowledge base.',
    accent: 'from-emerald-400 to-teal-500',
  },
];

const HOW_IT_WORKS = [
  { step: '01', title: 'Set your interests', body: 'Choose research areas like NLP, CV, or RL. The system builds a semantic profile from your selections.' },
  { step: '02', title: 'Generate your feed', body: 'ChromaDB HNSW retrieval finds semantically relevant papers across 20k+ indexed ArXiv papers in milliseconds.' },
  { step: '03', title: 'Save and dismiss', body: 'Your interactions train a personal LightGBM ranking model. After 50 interactions, your model kicks in.' },
];

export default function Home() {
  const { isAuthenticated, isLoading, fetchProfile } = useAuth();
  const router = useRouter();

  useEffect(() => { fetchProfile(); }, [fetchProfile]);
  useEffect(() => {
    if (!isLoading && isAuthenticated) router.push('/dashboard');
  }, [isAuthenticated, isLoading, router]);

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-slate-50">
        <div className="animate-spin rounded-full h-10 w-10 border-[3px] border-gray-200 border-t-primary-600" />
      </div>
    );
  }

  return (
    <main className="min-h-screen bg-slate-50">
      {/* ── Hero ── */}
      <div className="relative overflow-hidden">
        {/* Animated gradient background */}
        <div className="absolute inset-0 bg-animated opacity-90" />
        {/* Grid overlay */}
        <div
          className="absolute inset-0 opacity-[0.12]"
          style={{
            backgroundImage:
              'linear-gradient(rgba(255,255,255,.4) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,.4) 1px, transparent 1px)',
            backgroundSize: '48px 48px',
          }}
        />
        {/* Bottom fade */}
        <div className="absolute bottom-0 left-0 right-0 h-24 bg-gradient-to-t from-slate-50 to-transparent" />

        <div className="relative max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-28 text-center">
          {/* Brand */}
          <div className="inline-flex items-center gap-2 bg-white/20 backdrop-blur-sm text-white text-sm font-medium px-4 py-1.5 rounded-full mb-8 border border-white/30">
            <span>🔬</span> ResearchMate
          </div>

          <h1 className="text-5xl sm:text-6xl font-extrabold text-white mb-6 leading-tight tracking-tight">
            Your ML research feed,<br />
            <span className="text-primary-100">ranked by your own AI</span>
          </h1>

          <p className="text-lg sm:text-xl text-white/80 mb-10 max-w-2xl mx-auto leading-relaxed">
            Personalized paper recommendations from ArXiv, a knowledge base you can ask questions to,
            and a learning-to-rank model that trains on your behavior.
          </p>

          <div className="flex flex-col sm:flex-row gap-3 justify-center">
            <Link
              href="/register"
              className="inline-flex items-center justify-center gap-2 bg-white text-primary-700 font-semibold px-8 py-3.5 rounded-xl shadow-lg hover:shadow-xl hover:bg-primary-50 transition-all duration-150 text-base"
            >
              Get started — free →
            </Link>
            <Link
              href="/login"
              className="inline-flex items-center justify-center gap-2 bg-white/10 border border-white/30 backdrop-blur-sm text-white font-medium px-8 py-3.5 rounded-xl hover:bg-white/20 transition-all duration-150 text-base"
            >
              Sign in
            </Link>
          </div>

          {/* Social proof chips */}
          <div className="flex flex-wrap justify-center gap-3 mt-10">
            {['20k+ papers indexed', 'ChromaDB HNSW retrieval', 'Personal LightGBM model', 'RAG Q&A'].map((t) => (
              <span key={t} className="bg-white/15 border border-white/25 text-white/90 text-xs font-medium px-3.5 py-1.5 rounded-full backdrop-blur-sm">
                {t}
              </span>
            ))}
          </div>
        </div>
      </div>

      {/* ── Features ── */}
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
        <div className="text-center mb-14">
          <p className="text-sm font-semibold text-primary-600 uppercase tracking-widest mb-3">What you get</p>
          <h2 className="text-3xl sm:text-4xl font-bold text-gray-900">Built for ML researchers</h2>
        </div>

        <div className="grid md:grid-cols-3 gap-6">
          {FEATURES.map((f) => (
            <div key={f.title} className="card-lift group">
              <div className={`w-12 h-12 rounded-2xl bg-gradient-to-br ${f.accent} flex items-center justify-center text-2xl mb-5 shadow-md group-hover:scale-105 transition-transform duration-200`}>
                {f.icon}
              </div>
              <h3 className="text-lg font-bold text-gray-900 mb-2">{f.title}</h3>
              <p className="text-gray-500 text-sm leading-relaxed">{f.description}</p>
            </div>
          ))}
        </div>
      </div>

      {/* ── How it works ── */}
      <div className="bg-white border-y border-gray-100 py-20">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-14">
            <p className="text-sm font-semibold text-primary-600 uppercase tracking-widest mb-3">How it works</p>
            <h2 className="text-3xl sm:text-4xl font-bold text-gray-900">Simple, powerful, and personal</h2>
          </div>

          <div className="grid sm:grid-cols-3 gap-8">
            {HOW_IT_WORKS.map((s) => (
              <div key={s.step} className="text-center sm:text-left">
                <div className="text-5xl font-black text-gradient mb-4">{s.step}</div>
                <h3 className="font-bold text-gray-900 mb-2">{s.title}</h3>
                <p className="text-sm text-gray-500 leading-relaxed">{s.body}</p>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* ── Demo Feed ── */}
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
        <div className="text-center mb-12">
          <p className="text-sm font-semibold text-primary-600 uppercase tracking-widest mb-3">Live preview</p>
          <h2 className="text-3xl sm:text-4xl font-bold text-gray-900">Your daily feed</h2>
          <p className="text-gray-500 mt-3">Papers and articles ranked by your personal ML model</p>
        </div>

        <div className="space-y-4">
          {[...DEMO_PAPERS, DEMO_ARTICLE].map((item, i) => (
            <DemoCard key={i} item={item} type={i < DEMO_PAPERS.length ? 'paper' : 'article'} />
          ))}
        </div>

        <p className="text-center text-xs text-gray-400 mt-5">
          ↑ Sample content — your real feed is ranked by your trained model and semantic interests
        </p>
      </div>

      {/* ── CTA ── */}
      <div className="bg-animated py-20 relative overflow-hidden">
        <div
          className="absolute inset-0 opacity-[0.10]"
          style={{
            backgroundImage:
              'linear-gradient(rgba(255,255,255,.4) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,.4) 1px, transparent 1px)',
            backgroundSize: '48px 48px',
          }}
        />
        <div className="relative max-w-3xl mx-auto px-4 text-center">
          <h2 className="text-3xl sm:text-4xl font-bold text-white mb-4">
            Start building your research feed today
          </h2>
          <p className="text-white/75 text-lg mb-8">Free to use. No credit card required.</p>
          <Link
            href="/register"
            className="inline-flex items-center gap-2 bg-white text-primary-700 font-semibold px-10 py-3.5 rounded-xl shadow-lg hover:shadow-xl hover:bg-primary-50 transition-all duration-150 text-base"
          >
            Create free account →
          </Link>
        </div>
      </div>

      {/* ── Footer ── */}
      <footer className="bg-slate-900 py-10">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row justify-between items-center gap-4">
          <div className="flex items-center gap-2">
            <span className="text-xl">🔬</span>
            <span className="text-white font-bold">ResearchMate</span>
          </div>
          <p className="text-gray-500 text-sm">
            Built with FastAPI · Next.js · LightGBM · ChromaDB · PostgreSQL
          </p>
          <div className="flex gap-6 text-sm">
            <Link href="/login" className="text-gray-400 hover:text-white transition-colors">Sign in</Link>
            <Link href="/register" className="text-gray-400 hover:text-white transition-colors">Register</Link>
          </div>
        </div>
      </footer>
    </main>
  );
}

function DemoCard({ item, type }: { item: typeof DEMO_PAPERS[0]; type: 'paper' | 'article' }) {
  const accentColor = type === 'paper' ? '#0ea5e9' : '#8b5cf6';
  return (
    <div
      className="bg-white rounded-2xl border border-gray-100 shadow-sm p-5 transition-all duration-200 hover:shadow-md hover:-translate-y-px"
      style={{ borderLeft: `3px solid ${accentColor}` }}
    >
      <div className="flex items-start justify-between gap-4 mb-3.5">
        <div>
          <h3 className="font-semibold text-gray-900">{item.title}</h3>
          <div className="flex items-center gap-2 mt-0.5">
            <p className="text-xs text-gray-400">{item.source}</p>
            <span className="text-xs text-gray-300">·</span>
            <p className="text-xs text-gray-400">Score: {item.score}</p>
          </div>
        </div>
        <span className={`text-xs font-semibold px-2.5 py-1 rounded-full whitespace-nowrap ${item.badge.color}`}>
          {item.badge.label}
        </span>
      </div>

      <div className="space-y-2 mb-4">
        <SummaryRow label="Key Insight" content={item.summary.insight} labelColor="text-amber-700 bg-amber-50" />
        <SummaryRow label="Why It Matters" content={item.summary.why} labelColor="text-blue-700 bg-blue-50" />
        <SummaryRow label="Relevance" content={item.summary.relevance} labelColor="text-emerald-700 bg-emerald-50" />
      </div>

      <div className="flex gap-2 pt-3 border-t border-gray-100">
        <button disabled className="flex-1 text-sm py-1.5 rounded-lg bg-primary-50 text-primary-600 font-medium opacity-75 cursor-default">✓ Save</button>
        <button disabled className="flex-1 text-sm py-1.5 rounded-lg bg-gray-100 text-gray-400 font-medium cursor-default">✕ Dismiss</button>
        <button disabled className="flex-1 text-sm py-1.5 rounded-lg bg-gray-100 text-gray-400 font-medium cursor-default">Read →</button>
      </div>
    </div>
  );
}

function SummaryRow({ label, content, labelColor }: { label: string; content: string; labelColor: string }) {
  return (
    <div className="flex gap-2 items-start">
      <span className={`shrink-0 text-xs font-semibold px-2 py-0.5 rounded-md ${labelColor}`}>{label}</span>
      <p className="text-sm text-gray-600 leading-relaxed">{content}</p>
    </div>
  );
}
