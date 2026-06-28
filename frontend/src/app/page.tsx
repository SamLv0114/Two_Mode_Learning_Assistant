'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/lib/auth';
import Link from 'next/link';

const DEMO_PAPERS = [
  {
    title: 'Attention Is All You Need',
    source: 'ArXiv · cs.LG',
    summary: {
      insight: 'Introduces the Transformer architecture, replacing recurrence with self-attention for sequence modeling.',
      why: 'Transformers are now the backbone of every major LLM including GPT and BERT — understanding this paper is foundational.',
      relevance: 'Directly relevant to your interest in NLP and deep learning architectures.',
    },
    badge: '⭐ High Impact',
    badgeColor: 'bg-amber-100 text-amber-800',
  },
  {
    title: 'LightGBM: A Highly Efficient Gradient Boosting Decision Tree',
    source: 'ArXiv · cs.LG',
    summary: {
      insight: 'Proposes histogram-based algorithms that dramatically speed up gradient boosting with lower memory usage.',
      why: 'LightGBM is the go-to model for tabular data and ranking tasks in production ML systems.',
      relevance: 'Matches your interests in efficient ML and recommendation systems.',
    },
    badge: '💻 Code Available',
    badgeColor: 'bg-green-100 text-green-800',
  },
];

const DEMO_ARTICLES = [
  {
    title: 'How Retrieval-Augmented Generation Actually Works',
    source: 'Hacker News · 342 points',
    summary: {
      insight: 'Explains the full RAG pipeline from document chunking to vector retrieval and LLM augmentation.',
      why: 'RAG is now the standard approach for building Q&A systems over private knowledge bases.',
      relevance: 'Highly relevant to your work on knowledge base assistants.',
    },
    badge: '🔥 Trending',
    badgeColor: 'bg-red-100 text-red-800',
  },
];

const DEMO_QA = {
  question: 'What is the difference between RAG and fine-tuning?',
  answer: `Based on your uploaded documents:

**RAG (Retrieval-Augmented Generation)** retrieves relevant passages from your knowledge base at query time and passes them as context to the LLM. It requires no model training and your knowledge base can be updated instantly.

**Fine-tuning** trains the model weights directly on your data, baking knowledge into the model itself. It requires GPU resources and retraining whenever your data changes.

**When to use each:**
- Use RAG when your knowledge changes frequently or you need citations
- Use fine-tuning when you need the model to adopt a specific style or reasoning pattern

*Sources: "Survey of LLM Adaptation Methods" (uploaded), "RAG vs Fine-tuning Benchmark" (uploaded)*`,
};

export default function Home() {
  const { isAuthenticated, isLoading, fetchProfile } = useAuth();
  const router = useRouter();

  useEffect(() => {
    fetchProfile();
  }, [fetchProfile]);

  useEffect(() => {
    if (!isLoading && isAuthenticated) {
      router.push('/dashboard');
    }
  }, [isAuthenticated, isLoading, router]);

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
      </div>
    );
  }

  return (
    <main className="min-h-screen">
      {/* Hero */}
      <div className="bg-gradient-to-b from-primary-50 to-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-24">
          <div className="text-center">
            <h1 className="text-5xl font-bold text-gray-900 mb-6">ResearchMate</h1>
            <p className="text-xl text-gray-600 mb-8 max-w-2xl mx-auto">
              Your personalized ML research companion. Get daily paper recommendations,
              track your interests, and ask questions about your knowledge base.
            </p>
            <div className="flex gap-4 justify-center">
              <Link href="/register" className="btn-primary text-lg px-8 py-3">
                Get Started — Free
              </Link>
              <Link href="/login" className="btn-secondary text-lg px-8 py-3">
                Sign In
              </Link>
            </div>
          </div>
        </div>
      </div>

      {/* Feature Cards */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
        <h2 className="text-3xl font-bold text-center text-gray-900 mb-12">Features</h2>
        <div className="grid md:grid-cols-3 gap-8">
          <FeatureCard
            title="Personalized Daily Feed"
            description="ML-ranked papers and articles from ArXiv, Hacker News, and more — tailored to your interests."
            icon="📚"
          />
          <FeatureCard
            title="Learns From You"
            description="Your own LightGBM model trains on your saves and dismissals. The more you use it, the smarter it gets."
            icon="🧠"
          />
          <FeatureCard
            title="Q&A Over Your Documents"
            description="Upload papers and notes. Ask questions and get cited answers from your personal knowledge base."
            icon="💬"
          />
        </div>
      </div>

      {/* Demo: Feed */}
      <div className="bg-gray-50 py-16">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-10">
            <span className="text-sm font-semibold text-primary-600 uppercase tracking-wide">Live Demo</span>
            <h2 className="text-3xl font-bold text-gray-900 mt-2">Your Daily Feed</h2>
            <p className="text-gray-500 mt-2">Papers and articles ranked by your personal ML model</p>
          </div>

          <div className="space-y-4">
            {[...DEMO_PAPERS, ...DEMO_ARTICLES].map((item, i) => (
              <DemoFeedCard key={i} item={item} />
            ))}
          </div>

          <p className="text-center text-sm text-gray-400 mt-6">
            ↑ Sample content — your real feed is ranked by your own trained model
          </p>
        </div>
      </div>

      {/* Demo: Q&A */}
      <div className="py-16">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-10">
            <span className="text-sm font-semibold text-primary-600 uppercase tracking-wide">Live Demo</span>
            <h2 className="text-3xl font-bold text-gray-900 mt-2">Q&A Over Your Knowledge Base</h2>
            <p className="text-gray-500 mt-2">Upload papers and documents — then ask anything</p>
          </div>

          <div className="card border border-gray-200 shadow-sm">
            {/* Question */}
            <div className="flex gap-3 mb-6">
              <div className="w-8 h-8 rounded-full bg-primary-100 flex items-center justify-center flex-shrink-0 text-sm font-bold text-primary-700">
                You
              </div>
              <div className="bg-primary-50 rounded-lg px-4 py-3 text-gray-800 text-sm flex-1">
                {DEMO_QA.question}
              </div>
            </div>

            {/* Answer */}
            <div className="flex gap-3">
              <div className="w-8 h-8 rounded-full bg-gray-100 flex items-center justify-center flex-shrink-0 text-sm">
                🤖
              </div>
              <div className="flex-1 text-sm text-gray-700 space-y-2">
                {DEMO_QA.answer.split('\n\n').map((para, i) => (
                  <p key={i} className={para.startsWith('**') ? 'font-semibold text-gray-900' : ''}>
                    {para.replace(/\*\*/g, '')}
                  </p>
                ))}
              </div>
            </div>
          </div>

          <p className="text-center text-sm text-gray-400 mt-6">
            ↑ Sample response — answers are grounded in your uploaded documents
          </p>
        </div>
      </div>

      {/* CTA */}
      <div className="bg-primary-600 py-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h2 className="text-3xl font-bold text-white mb-4">
            Start building your personalized research feed
          </h2>
          <p className="text-primary-100 mb-8 text-lg">
            Free to use. No credit card required.
          </p>
          <Link href="/register" className="bg-white text-primary-600 font-semibold px-8 py-3 rounded-lg hover:bg-primary-50 transition-colors text-lg">
            Create Account
          </Link>
        </div>
      </div>

      {/* Footer */}
      <footer className="bg-gray-50 border-t">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <p className="text-center text-gray-500 text-sm">
            ResearchMate · Built with FastAPI, Next.js, LightGBM, and ChromaDB
          </p>
        </div>
      </footer>
    </main>
  );
}

function FeatureCard({ title, description, icon }: { title: string; description: string; icon: string }) {
  return (
    <div className="card text-center">
      <div className="text-4xl mb-4">{icon}</div>
      <h3 className="text-xl font-semibold text-gray-900 mb-2">{title}</h3>
      <p className="text-gray-600">{description}</p>
    </div>
  );
}

function DemoFeedCard({ item }: { item: typeof DEMO_PAPERS[0] }) {
  return (
    <div className="card border border-gray-200 shadow-sm">
      <div className="flex items-start justify-between gap-4 mb-3">
        <div>
          <h3 className="font-semibold text-gray-900 text-base">{item.title}</h3>
          <p className="text-xs text-gray-400 mt-0.5">{item.source}</p>
        </div>
        <span className={`text-xs font-medium px-2 py-1 rounded-full whitespace-nowrap ${item.badgeColor}`}>
          {item.badge}
        </span>
      </div>

      <div className="space-y-2 mb-4">
        <div className="rounded-md px-3 py-2 bg-amber-50">
          <span className="text-xs font-bold text-amber-700 uppercase tracking-wide">Key Insight</span>
          <p className="text-sm text-gray-700 mt-0.5">{item.summary.insight}</p>
        </div>
        <div className="rounded-md px-3 py-2 bg-blue-50">
          <span className="text-xs font-bold text-blue-700 uppercase tracking-wide">Why It Matters</span>
          <p className="text-sm text-gray-700 mt-0.5">{item.summary.why}</p>
        </div>
        <div className="rounded-md px-3 py-2 bg-green-50">
          <span className="text-xs font-bold text-green-700 uppercase tracking-wide">Relevance to You</span>
          <p className="text-sm text-gray-700 mt-0.5">{item.summary.relevance}</p>
        </div>
      </div>

      <div className="flex gap-2">
        <button disabled className="flex-1 text-sm py-1.5 rounded-lg bg-primary-50 text-primary-600 font-medium cursor-default opacity-70">
          ✓ Save
        </button>
        <button disabled className="flex-1 text-sm py-1.5 rounded-lg bg-gray-100 text-gray-500 font-medium cursor-default opacity-70">
          ✕ Dismiss
        </button>
        <button disabled className="flex-1 text-sm py-1.5 rounded-lg bg-gray-100 text-gray-500 font-medium cursor-default opacity-70">
          Read More
        </button>
      </div>
    </div>
  );
}
