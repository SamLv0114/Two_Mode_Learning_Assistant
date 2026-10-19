'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { useAuth } from '@/lib/auth';
import { roundedCount, usePublicStats } from '@/lib/usePublicStats';
import { TryDemoButton } from '@/components/landing/TryDemoButton';

const FEATURES = [
  {
    icon: '🧪',
    title: 'Paper → runnable code',
    description: 'The Coding Agent reads a paper\'s full LaTeX, asks you to settle what the paper leaves open, '
      + 'and writes a small teaching implementation. Every key line is tagged with the excerpt it implements '
      + 'and checked against the paper before you read it.',
    accent: 'from-violet-400 to-indigo-500',
  },
  {
    icon: '🗂️',
    title: 'Understand any repository',
    description: 'Point it at a public GitHub repo. It pins a snapshot, maps entry points, core modules and the '
      + 'call graph, gives you a reading tour, and answers questions with file:line citations it checks '
      + 'against what it actually read.',
    accent: 'from-emerald-400 to-teal-500',
  },
  {
    icon: '💬',
    title: 'Research agents that cite',
    description: 'Questions are routed to the right agent. Deep research splits a question into sub-questions, '
      + 'researches them in parallel, looks again where evidence is missing, and writes a cited report — '
      + 'resumable if it is interrupted.',
    accent: 'from-sky-400 to-primary-500',
  },
  {
    icon: '📚',
    title: 'A feed that learns your taste',
    description: 'Hybrid search over arXiv finds papers for your interests, with a summary of why each matters '
      + 'to you. Your saves and dismissals train a personal LightGBM ranker. What\'s Hot tracks the papers '
      + 'and repositories the community is picking up this week.',
    accent: 'from-amber-400 to-orange-500',
  },
];

const HOW_IT_WORKS = [
  { step: '01', title: 'Tell it what you study', body: 'Pick focus areas and rate a few papers. That is enough for a first personalised feed.' },
  { step: '02', title: 'Read, ask, go deeper', body: 'Ask about any paper in chat, or run deep research across the knowledge base, full texts and the web.' },
  { step: '03', title: 'Build it yourself', body: 'Turn a paper into code you can trace back to the text, then rebuild its key lines as an exercise.' },
];

const HIGHLIGHTS = [
  { value: '0.44 → 0.69', label: 'Recall@5 from vector search to hybrid BM25 + vector + reranking', note: '120 labelled queries' },
  { value: '98%', label: 'intent routing accuracy, 63% of requests routed without an LLM call', note: '108 labelled queries' },
  { value: '67% → 0%', label: 'success of constructed prompt-injection attacks once sources are fenced', note: '6 attack cases' },
];

/** A faithful sketch of the Coding Agent's annotated program view — not live data. */
function CodeIllustration() {
  const lines: [string, string?][] = [
    ['class LoRALinear(nn.Module):'],
    ['    def __init__(self, base, r=8, alpha=16):'],
    ['        self.A = randn(r, base.in_features) * 0.01', 'E2'],
    ['        self.B = zeros(base.out_features, r)', 'E3'],
    ['        self.scale = alpha / r', 'E4'],
    ['    def forward(self, x):'],
    ['        return base(x) + (x @ self.A.T @ self.B.T) * self.scale', 'E1'],
  ];
  return (
    <div className="rounded-2xl border border-gray-200 bg-white shadow-sm overflow-hidden text-left">
      <div className="flex items-center justify-between border-b border-gray-100 px-4 py-2 text-xs text-gray-500">
        <span className="font-mono">lora_teaching.py</span>
        <span>Illustration of the Coding Agent view</span>
      </div>
      <div className="bg-gray-50 py-2 font-mono text-[12px] leading-6">
        {lines.map(([code, tag], i) => (
          <div key={i} className={`flex gap-3 px-3 ${tag === 'E3' ? 'bg-amber-100' : ''}`}>
            <span className="w-5 shrink-0 select-none text-right text-gray-400">{i + 1}</span>
            <span className="whitespace-pre text-gray-800 overflow-hidden">{code}</span>
            {tag && <span className="ml-auto h-fit shrink-0 rounded bg-violet-100 px-1.5 text-[10px] font-semibold text-violet-800">{tag}</span>}
          </div>
        ))}
      </div>
      <div className="m-3 rounded-lg border border-violet-200 bg-white p-3 text-xs text-gray-700">
        <p className="mb-1 font-semibold text-gray-500">E3 · paper excerpt</p>
        “We use a random Gaussian initialization for A and zero for B, so ΔW = BA is zero at the beginning of training.”
      </div>
    </div>
  );
}

export default function Home() {
  const { isAuthenticated, isLoading, fetchProfile } = useAuth();
  const router = useRouter();
  const stats = usePublicStats();

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

  const chips = [
    stats ? `${roundedCount(stats.papers_indexed)} arXiv papers indexed` : null,
    'Hybrid retrieval + reranking', 'LangGraph agents', 'Verified citations',
  ].filter(Boolean) as string[];

  return (
    <main className="min-h-screen bg-slate-50">
      {/* ── Hero ── */}
      <div className="relative overflow-hidden">
        <div className="absolute inset-0 bg-animated opacity-90" />
        <div
          className="absolute inset-0 opacity-[0.12]"
          style={{
            backgroundImage:
              'linear-gradient(rgba(255,255,255,.4) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,.4) 1px, transparent 1px)',
            backgroundSize: '48px 48px',
          }}
        />
        <div className="absolute bottom-0 left-0 right-0 h-24 bg-gradient-to-t from-slate-50 to-transparent" />

        <div className="relative max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-24 text-center">
          <div className="inline-flex items-center gap-2 bg-white/20 backdrop-blur-sm text-white text-sm font-medium px-4 py-1.5 rounded-full mb-8 border border-white/30">
            <span>🔬</span> ResearchMate
          </div>

          <h1 className="text-4xl sm:text-6xl font-extrabold text-white mb-6 leading-tight tracking-tight">
            Read ML research.<br />
            <span className="text-primary-100">Then build it yourself.</span>
          </h1>

          <p className="text-lg sm:text-xl text-white/80 mb-10 max-w-2xl mx-auto leading-relaxed">
            A research assistant that finds the papers worth your time, answers with citations,
            and turns a paper or a repository into code you can trace back to its source.
          </p>

          <div className="flex flex-col sm:flex-row gap-3 justify-center">
            {stats?.demo_available && (
              <TryDemoButton className="inline-flex items-center justify-center gap-2 bg-white text-primary-700 font-semibold px-8 py-3.5 rounded-xl shadow-lg hover:shadow-xl hover:bg-primary-50 transition-all duration-150 text-base disabled:opacity-70" />
            )}
            <Link
              href="/register"
              className={stats?.demo_available
                ? 'inline-flex items-center justify-center gap-2 bg-white/10 border border-white/30 backdrop-blur-sm text-white font-medium px-8 py-3.5 rounded-xl hover:bg-white/20 transition-all duration-150 text-base'
                : 'inline-flex items-center justify-center gap-2 bg-white text-primary-700 font-semibold px-8 py-3.5 rounded-xl shadow-lg hover:shadow-xl hover:bg-primary-50 transition-all duration-150 text-base'}
            >
              Create a free account
            </Link>
            <Link
              href="/login"
              className="inline-flex items-center justify-center gap-2 text-white/90 font-medium px-6 py-3.5 rounded-xl hover:bg-white/10 transition-all duration-150 text-base"
            >
              Sign in
            </Link>
          </div>

          <div className="flex flex-wrap justify-center gap-3 mt-10">
            {chips.map((t) => (
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
          <p className="text-sm font-semibold text-primary-600 uppercase tracking-widest mb-3">What it does</p>
          <h2 className="text-3xl sm:text-4xl font-bold text-gray-900">From finding a paper to implementing it</h2>
        </div>
        <div className="grid md:grid-cols-2 gap-6">
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

      {/* ── Showcase ── */}
      <div className="bg-white border-y border-gray-100 py-20">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 grid lg:grid-cols-2 gap-12 items-center">
          <div>
            <p className="text-sm font-semibold text-primary-600 uppercase tracking-widest mb-3">Verifiable, not just plausible</p>
            <h2 className="text-3xl font-bold text-gray-900 mb-4">Every claim points back to its source</h2>
            <p className="text-gray-500 leading-relaxed mb-4">
              Generated code carries tags like <span className="rounded bg-violet-100 px-1 font-mono text-xs text-violet-800">E3</span> that
              open the exact paper passage a line implements. A reviewer checks each tagged detail against the text before
              anything runs, and repository answers only count a file:line citation as verified if the agent read those lines.
            </p>
            <p className="text-gray-500 leading-relaxed">
              When the paper leaves a choice open — an initial value, a schedule — the agent asks you instead of guessing.
            </p>
          </div>
          <CodeIllustration />
        </div>
      </div>

      {/* ── How it works ── */}
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
        <div className="text-center mb-14">
          <p className="text-sm font-semibold text-primary-600 uppercase tracking-widest mb-3">How you use it</p>
          <h2 className="text-3xl sm:text-4xl font-bold text-gray-900">Three steps</h2>
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

      {/* ── Engineering ── */}
      <div className="bg-slate-900 py-20">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <p className="text-sm font-semibold text-primary-300 uppercase tracking-widest mb-3">Measured, not assumed</p>
            <h2 className="text-3xl font-bold text-white">Evaluated on labelled data</h2>
          </div>
          <div className="grid md:grid-cols-3 gap-6">
            {HIGHLIGHTS.map((h) => (
              <div key={h.label} className="rounded-2xl border border-white/10 bg-white/5 p-6">
                <p className="text-3xl font-extrabold text-white mb-2">{h.value}</p>
                <p className="text-sm text-gray-300 leading-relaxed">{h.label}</p>
                <p className="mt-3 text-xs text-gray-500">{h.note}</p>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* ── CTA ── */}
      <div className="bg-animated py-20 relative overflow-hidden">
        <div className="relative max-w-3xl mx-auto px-4 text-center">
          <h2 className="text-3xl sm:text-4xl font-bold text-white mb-4">See it on a real paper</h2>
          <p className="text-white/75 text-lg mb-8">Free to use. No credit card.</p>
          <div className="flex flex-col sm:flex-row gap-3 justify-center">
            {stats?.demo_available && (
              <TryDemoButton className="inline-flex items-center justify-center gap-2 bg-white text-primary-700 font-semibold px-10 py-3.5 rounded-xl shadow-lg hover:shadow-xl hover:bg-primary-50 transition-all duration-150 text-base disabled:opacity-70" />
            )}
            <Link
              href="/register"
              className="inline-flex items-center justify-center gap-2 bg-white/10 border border-white/30 text-white font-semibold px-10 py-3.5 rounded-xl hover:bg-white/20 transition-all duration-150 text-base"
            >
              Create free account →
            </Link>
          </div>
        </div>
      </div>

      {/* ── Footer ── */}
      <footer className="bg-slate-900 py-10">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row justify-between items-center gap-4">
          <div className="flex items-center gap-2">
            <span className="text-xl">🔬</span>
            <span className="text-white font-bold">ResearchMate</span>
          </div>
          <p className="text-gray-500 text-sm text-center">
            FastAPI · Next.js · LangGraph · OpenAI · ChromaDB · PostgreSQL · LightGBM · Docker
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
