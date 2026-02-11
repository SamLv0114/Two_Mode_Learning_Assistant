'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/lib/auth';
import Link from 'next/link';

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
      {/* Hero Section */}
      <div className="bg-gradient-to-b from-primary-50 to-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-24">
          <div className="text-center">
            <h1 className="text-5xl font-bold text-gray-900 mb-6">
              Learning Assistant
            </h1>
            <p className="text-xl text-gray-600 mb-8 max-w-2xl mx-auto">
              Your personalized ML research companion. Get daily paper recommendations,
              track your interests, and ask questions about your knowledge base.
            </p>
            <div className="flex gap-4 justify-center">
              <Link
                href="/register"
                className="btn-primary text-lg px-8 py-3"
              >
                Get Started
              </Link>
              <Link
                href="/login"
                className="btn-secondary text-lg px-8 py-3"
              >
                Sign In
              </Link>
            </div>
          </div>
        </div>
      </div>

      {/* Features Section */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-24">
        <h2 className="text-3xl font-bold text-center text-gray-900 mb-12">
          Features
        </h2>
        <div className="grid md:grid-cols-3 gap-8">
          <FeatureCard
            title="Daily Feed"
            description="Get personalized paper and article recommendations based on your interests and reading history."
            icon="📚"
          />
          <FeatureCard
            title="Smart Learning"
            description="The more you interact, the better the recommendations. Your personal ML model learns from your preferences."
            icon="🧠"
          />
          <FeatureCard
            title="Q&A Assistant"
            description="Ask questions about your knowledge base. Upload documents and get instant answers."
            icon="❓"
          />
        </div>
      </div>

      {/* Footer */}
      <footer className="bg-gray-50 border-t">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <p className="text-center text-gray-500">
            Powered by FastAPI, Next.js, and LightGBM
          </p>
        </div>
      </footer>
    </main>
  );
}

function FeatureCard({
  title,
  description,
  icon,
}: {
  title: string;
  description: string;
  icon: string;
}) {
  return (
    <div className="card text-center">
      <div className="text-4xl mb-4">{icon}</div>
      <h3 className="text-xl font-semibold text-gray-900 mb-2">{title}</h3>
      <p className="text-gray-600">{description}</p>
    </div>
  );
}
