'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { useAuth } from '@/lib/auth';
import toast from 'react-hot-toast';

export default function LoginPage() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const { login } = useAuth();
  const router = useRouter();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    try {
      await login(email, password);
      toast.success('Welcome back!');
      router.push('/dashboard');
    } catch (error: any) {
      toast.error(error.response?.data?.detail || 'Login failed');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen flex">
      {/* Left — brand panel */}
      <div className="hidden lg:flex lg:w-[45%] bg-gradient-to-br from-slate-900 via-primary-900 to-indigo-900 flex-col justify-between p-12 relative overflow-hidden">
        {/* Subtle grid overlay */}
        <div
          className="absolute inset-0 opacity-[0.07]"
          style={{
            backgroundImage:
              'linear-gradient(rgba(255,255,255,.4) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,.4) 1px, transparent 1px)',
            backgroundSize: '44px 44px',
          }}
        />
        {/* Glow orb */}
        <div className="absolute top-1/3 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[480px] h-[480px] bg-primary-500 rounded-full opacity-[0.15] blur-3xl pointer-events-none" />

        {/* Brand */}
        <div className="relative z-10 flex items-center gap-2.5">
          <span className="text-3xl">🔬</span>
          <span className="text-white font-bold text-xl tracking-tight">ResearchMate</span>
        </div>

        {/* Headline + bullets */}
        <div className="relative z-10 space-y-7">
          <h2 className="text-white text-4xl font-bold leading-tight">
            Your personalized<br />ML research companion
          </h2>
          <ul className="space-y-3.5">
            {[
              'Daily paper feed ranked by your own ML model',
              'Semantic search across 20,000+ ArXiv papers',
              'RAG-powered Q&A over your uploaded documents',
            ].map((text) => (
              <li key={text} className="flex items-start gap-3 text-primary-100 text-sm leading-relaxed">
                <span className="text-primary-400 mt-0.5 shrink-0 text-base">✦</span>
                {text}
              </li>
            ))}
          </ul>
        </div>

        {/* Footer tech stack */}
        <div className="relative z-10">
          <p className="text-xs text-primary-400 tracking-wide">
            FastAPI · Next.js · LightGBM · ChromaDB · PostgreSQL
          </p>
        </div>
      </div>

      {/* Right — form */}
      <div className="flex-1 flex items-center justify-center p-6 sm:p-12 bg-slate-50">
        <div className="w-full max-w-sm animate-in">
          {/* Mobile logo */}
          <div className="flex items-center gap-2.5 mb-10 lg:hidden">
            <span className="text-2xl">🔬</span>
            <span className="font-bold text-xl text-gray-900 tracking-tight">ResearchMate</span>
          </div>

          <h1 className="text-2xl font-bold text-gray-900 mb-1">Welcome back</h1>
          <p className="text-sm text-gray-500 mb-8">
            New here?{' '}
            <Link href="/register" className="text-primary-600 hover:text-primary-700 font-semibold">
              Create a free account
            </Link>
          </p>

          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1.5">
                Email address
              </label>
              <input
                type="email"
                autoComplete="email"
                required
                className="input-field"
                placeholder="you@example.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1.5">
                Password
              </label>
              <input
                type="password"
                autoComplete="current-password"
                required
                className="input-field"
                placeholder="••••••••"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
            </div>

            <button
              type="submit"
              disabled={isSubmitting}
              className="w-full btn-primary py-2.5 mt-2"
            >
              {isSubmitting ? (
                <>
                  <svg className="animate-spin w-4 h-4" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                  </svg>
                  Signing in…
                </>
              ) : (
                'Sign in'
              )}
            </button>
          </form>

          <p className="mt-10 text-center">
            <Link href="/" className="text-sm text-gray-400 hover:text-gray-600 transition-colors">
              ← Back to home
            </Link>
          </p>
        </div>
      </div>
    </div>
  );
}
