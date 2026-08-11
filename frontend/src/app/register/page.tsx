'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { useAuth } from '@/lib/auth';
import toast from 'react-hot-toast';

const FOCUS_OPTIONS = ['ML', 'NLP', 'CV', 'AI', 'DL'];
const DEFAULT_INTERESTS = [
  'machine learning',
  'deep learning',
  'natural language processing',
  'computer vision',
  'reinforcement learning',
];

function PasswordStrength({ password }: { password: string }) {
  if (!password) return null;
  const checks = [
    { label: '8+ characters', ok: password.length >= 8 },
    { label: 'Uppercase', ok: /[A-Z]/.test(password) },
    { label: 'Lowercase', ok: /[a-z]/.test(password) },
    { label: 'Number', ok: /[0-9]/.test(password) },
  ];
  const passed = checks.filter((c) => c.ok).length;
  const color = passed <= 1 ? 'bg-red-400' : passed <= 2 ? 'bg-amber-400' : passed <= 3 ? 'bg-yellow-400' : 'bg-green-500';
  return (
    <div className="mt-2 space-y-1.5">
      <div className="flex gap-1">
        {[0, 1, 2, 3].map((i) => (
          <div key={i} className={`h-1 flex-1 rounded-full transition-colors duration-200 ${i < passed ? color : 'bg-gray-200'}`} />
        ))}
      </div>
      <div className="flex flex-wrap gap-x-3 gap-y-0.5">
        {checks.map((c) => (
          <span key={c.label} className={`text-xs ${c.ok ? 'text-green-600' : 'text-gray-400'}`}>
            {c.ok ? '✓' : '·'} {c.label}
          </span>
        ))}
      </div>
    </div>
  );
}

export default function RegisterPage() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [fullName, setFullName] = useState('');
  const [focusAreas, setFocusAreas] = useState<string[]>(['ML', 'AI']);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');
  const { register } = useAuth();
  const router = useRouter();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setErrorMsg('');
    setIsSubmitting(true);
    try {
      await register({
        email,
        password,
        full_name: fullName || undefined,
        interests: DEFAULT_INTERESTS,
        focus_areas: focusAreas,
      });
      toast.success('Account created! Welcome to ResearchMate.');
      router.push('/dashboard');
    } catch (error: any) {
      setErrorMsg(error.response?.data?.detail || 'Registration failed');
    } finally {
      setIsSubmitting(false);
    }
  };

  const toggleFocusArea = (area: string) => {
    setFocusAreas((prev) =>
      prev.includes(area) ? prev.filter((a) => a !== area) : [...prev, area]
    );
  };

  return (
    <div className="min-h-screen flex">
      {/* Left — brand panel */}
      <div className="hidden lg:flex lg:w-[45%] bg-gradient-to-br from-slate-900 via-primary-900 to-indigo-900 flex-col justify-between p-12 relative overflow-hidden">
        <div
          className="absolute inset-0 opacity-[0.07]"
          style={{
            backgroundImage:
              'linear-gradient(rgba(255,255,255,.4) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,.4) 1px, transparent 1px)',
            backgroundSize: '44px 44px',
          }}
        />
        <div className="absolute top-1/3 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[480px] h-[480px] bg-primary-500 rounded-full opacity-[0.15] blur-3xl pointer-events-none" />

        <div className="relative z-10 flex items-center gap-2.5">
          <span className="text-3xl">🔬</span>
          <span className="text-white font-bold text-xl tracking-tight">ResearchMate</span>
        </div>

        <div className="relative z-10 space-y-7">
          <h2 className="text-white text-4xl font-bold leading-tight">
            Start learning<br />smarter today
          </h2>
          <ul className="space-y-3.5">
            {[
              'Get a personalized feed of ML papers every day',
              'Your model trains on your saves — gets smarter over time',
              'Chat with an AI agent over your research documents',
            ].map((text) => (
              <li key={text} className="flex items-start gap-3 text-primary-100 text-sm leading-relaxed">
                <span className="text-primary-400 mt-0.5 shrink-0 text-base">✦</span>
                {text}
              </li>
            ))}
          </ul>
        </div>

        <div className="relative z-10">
          <p className="text-xs text-primary-400 tracking-wide">
            FastAPI · Next.js · LightGBM · ChromaDB · PostgreSQL
          </p>
        </div>
      </div>

      {/* Right — form */}
      <div className="flex-1 flex items-center justify-center p-6 sm:p-12 bg-slate-50 overflow-y-auto">
        <div className="w-full max-w-sm animate-in py-4">
          <div className="flex items-center gap-2.5 mb-10 lg:hidden">
            <span className="text-2xl">🔬</span>
            <span className="font-bold text-xl text-gray-900 tracking-tight">ResearchMate</span>
          </div>

          <h1 className="text-2xl font-bold text-gray-900 mb-1">Create your account</h1>
          <p className="text-sm text-gray-500 mb-8">
            Already have an account?{' '}
            <Link href="/login" className="text-primary-600 hover:text-primary-700 font-semibold">
              Sign in
            </Link>
          </p>

          {errorMsg && (
            <div className="mb-5 bg-red-50 border border-red-200 text-red-700 text-sm rounded-xl px-4 py-3">
              {errorMsg}
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1.5">
                Email address <span className="text-red-400">*</span>
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
                Password <span className="text-red-400">*</span>
              </label>
              <input
                type="password"
                autoComplete="new-password"
                required
                className="input-field"
                placeholder="At least 8 characters"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
              <PasswordStrength password={password} />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1.5">
                Full name <span className="text-gray-400 font-normal">(optional)</span>
              </label>
              <input
                type="text"
                autoComplete="name"
                className="input-field"
                placeholder="John Doe"
                value={fullName}
                onChange={(e) => setFullName(e.target.value)}
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Research focus areas
              </label>
              <div className="flex flex-wrap gap-2">
                {FOCUS_OPTIONS.map((area) => (
                  <button
                    key={area}
                    type="button"
                    onClick={() => toggleFocusArea(area)}
                    className={`px-3.5 py-1.5 rounded-full text-sm font-medium transition-all duration-150 ${
                      focusAreas.includes(area)
                        ? 'bg-primary-600 text-white shadow-sm'
                        : 'bg-white text-gray-600 border border-gray-200 hover:border-primary-300 hover:text-primary-600'
                    }`}
                  >
                    {area}
                  </button>
                ))}
              </div>
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
                  Creating account…
                </>
              ) : (
                'Create account'
              )}
            </button>
          </form>

          <p className="mt-8 text-center">
            <Link href="/" className="text-sm text-gray-400 hover:text-gray-600 transition-colors">
              ← Back to home
            </Link>
          </p>
        </div>
      </div>
    </div>
  );
}
