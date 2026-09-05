'use client';

import { useEffect, useRef, useState } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/lib/auth';
import {
  feedApi, interactionsApi, qaApi, authApi, chatApi, whatsHotApi,
  Paper, Article, FeedResponse, InteractionStats, FeedJobStatus, AgentEvent,
  HFPaper, GithubRepo, WhatsHotData,
  ColdStartSeed, ColdStartResponse,
  API_BASE_URL,
} from '@/lib/api';
import toast from 'react-hot-toast';
import { BookOpen, FileText, Save, X, RefreshCw, ExternalLink, MessageCircle, Upload, Trash2, Search, Settings, ChevronDown, ChevronUp, Bot, Send, Sparkles, Flame } from 'lucide-react';

const AVAILABLE_AREAS = ['ML', 'NLP', 'CV', 'AI', 'DL'];
const EXAMPLE_INTERESTS = [
  'machine learning and deep learning',
  'natural language processing and transformers',
  'computer vision and image recognition',
  'reinforcement learning and agents',
  'generative AI and large language models',
];

export default function DashboardPage() {
  const { user, isAuthenticated, isLoading, fetchProfile, logout } = useAuth();
  const router = useRouter();

  const [feed, setFeed] = useState<FeedResponse | null>(null);
  const [stats, setStats] = useState<InteractionStats | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [whatsHot, setWhatsHot] = useState<WhatsHotData | null>(null);
  const [isLoadingHot, setIsLoadingHot] = useState(false);
  const [isRefreshingHot, setIsRefreshingHot] = useState(false);
  const [isRefining, setIsRefining] = useState(false);
  const [refineResult, setRefineResult] = useState<{ score: number; rounds: number; passed: boolean } | null>(null);
  const [coldStart, setColdStart] = useState<ColdStartResponse | null>(null);
  const [ratedSeeds, setRatedSeeds] = useState<Record<number, 'saved' | 'dismissed'>>({});
  const [feedProgress, setFeedProgress] = useState('');
  const [timeWindow, setTimeWindow] = useState(7);
  const [feedMode, setFeedMode] = useState<'recommended' | 'latest'>('recommended');
  const [activeTab, setActiveTab] = useState<'feed' | 'hot' | 'saved' | 'chat' | 'settings'>('feed');
  const [activePaperContext, setActivePaperContext] = useState<{
    title: string;
    abstract: string | null;
    arxivId: string;
    arxivUrl: string;
  } | null>(null);
  const feedPollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    fetchProfile();
  }, [fetchProfile]);

  useEffect(() => {
    if (!isLoading && !isAuthenticated) {
      router.push('/login');
    }
  }, [isAuthenticated, isLoading, router]);

  useEffect(() => {
    if (isAuthenticated) {
      loadStats();
      loadColdStart();
    }
  }, [isAuthenticated]);

  // What's Hot has its own tab now, so fetch it when that tab is first opened
  // rather than on every dashboard load. Kept in state after the first visit.
  useEffect(() => {
    if (isAuthenticated && activeTab === 'hot' && !whatsHot && !isLoadingHot) {
      loadWhatsHot();
    }
  }, [isAuthenticated, activeTab]);

  useEffect(() => {
    return () => {
      if (feedPollRef.current) clearInterval(feedPollRef.current);
    };
  }, []);

  const loadStats = async () => {
    try {
      const data = await interactionsApi.getStats();
      setStats(data);
    } catch (error) {
      console.error('Failed to load stats:', error);
    }
  };

  const loadWhatsHot = async () => {
    setIsLoadingHot(true);
    try {
      const data = await whatsHotApi.get();
      setWhatsHot(data);
    } catch (error) {
      console.error('Failed to load What\'s Hot:', error);
    } finally {
      setIsLoadingHot(false);
    }
  };

  const loadColdStart = async () => {
    try {
      const data = await feedApi.getColdStartSeeds(12);
      // Only surface onboarding when the backend says the profile is still thin.
      setColdStart(data.onboarding_recommended ? data : null);
    } catch (error) {
      console.error('Failed to load onboarding seeds:', error);
    }
  };

  const rateSeed = async (seed: ColdStartSeed, verdict: 'saved' | 'dismissed') => {
    setRatedSeeds((prev) => ({ ...prev, [seed.db_id]: verdict }));
    try {
      await interactionsApi.create({
        item_type: 'paper',
        item_id: seed.db_id,
        interaction_type: verdict,
      });
      loadStats();
    } catch (error) {
      // Roll the choice back so the card does not look recorded when it is not.
      setRatedSeeds((prev) => {
        const next = { ...prev };
        delete next[seed.db_id];
        return next;
      });
      toast.error('Could not save that rating');
    }
  };

  const refreshWhatsHot = async () => {
    setIsRefreshingHot(true);
    try {
      const data = await whatsHotApi.refresh();
      setWhatsHot(data);
      toast.success('What\'s Hot refreshed!');
    } catch (error) {
      toast.error('Failed to refresh');
    } finally {
      setIsRefreshingHot(false);
    }
  };

  const generateFeed = async (forceRefresh = false) => {
    if (feedPollRef.current) clearInterval(feedPollRef.current);
    setIsGenerating(true);
    setFeedProgress(forceRefresh ? 'Drawing a new batch...' : 'Loading your feed...');
    setFeed(null);
    setRefineResult(null);
    try {
      const { job_id } = await feedApi.generate({
        time_window_days: timeWindow,
        focus_areas: user?.focus_areas,
        use_ml: true,
        mode: feedMode,
        force_refresh: forceRefresh,
      });

      feedPollRef.current = setInterval(async () => {
        try {
          const status: FeedJobStatus = await feedApi.getJobStatus(job_id);

          const msgMap: Record<string, string> = {
            generating: 'Loading your feed...',
            collecting: 'Collecting papers...',
            ranking: 'Ranking and summarizing content (this takes 1-2 min)...',
            done: 'Feed ready!',
            error: status.message || 'Pipeline error',
            not_found: 'Job expired',
          };
          setFeedProgress(msgMap[status.status] || status.message || status.status);

          if (status.status === 'done') {
            clearInterval(feedPollRef.current!);
            feedPollRef.current = null;
            // Articles retired in V4 — trending content now comes from What's Hot.
            // No count — the server decides how large a feed is, so this does
            // not need updating if that size changes.
            const papers = await feedApi.getPapers();
            setFeed({
              papers,
              articles: [],
              generated_at: new Date().toISOString(),
              time_window_days: timeWindow,
              focus_areas: user?.focus_areas || [],
              used_ml_ranking: status.used_ml_ranking ?? false,
              total_papers_considered: status.papers_count ?? papers.length,
              total_articles_considered: 0,
            });
            toast.success(status.reused ? "Showing today's feed" : 'Feed ready!');
            loadStats();
            setIsGenerating(false);
          } else if (status.status === 'error' || status.status === 'not_found') {
            clearInterval(feedPollRef.current!);
            feedPollRef.current = null;
            toast.error(status.message || 'Feed generation failed');
            setIsGenerating(false);
          }
        } catch (_) {
          // transient error — keep polling
        }
      }, 2500);
    } catch (error: any) {
      toast.error(error.response?.data?.detail || 'Failed to start feed generation');
      setIsGenerating(false);
    }
  };

  // V4: Evaluator-Optimizer pass — re-scores the current feed and swaps out
  // weak papers without re-running the whole pipeline.
  const handleRefine = async () => {
    setIsRefining(true);
    try {
      const result = await feedApi.refine();
      setFeed((prev) => (prev ? { ...prev, papers: result.papers } : prev));
      setRefineResult({ score: result.score, rounds: result.rounds, passed: result.passed });
      toast.success(result.message || 'Feed refined!');
    } catch (error: any) {
      toast.error(error.response?.data?.detail || 'Refinement failed');
    } finally {
      setIsRefining(false);
    }
  };

  const handleInteraction = async (
    itemType: 'paper' | 'article',
    itemId: number,
    interactionType: 'viewed' | 'saved' | 'dismissed'
  ) => {
    try {
      await interactionsApi.create({
        item_type: itemType,
        item_id: itemId,
        interaction_type: interactionType,
      });
      toast.success(
        interactionType === 'saved' ? 'Saved!' :
        interactionType === 'dismissed' ? 'Dismissed' :
        'Viewed'
      );
      loadStats();
    } catch (error) {
      toast.error('Failed to record interaction');
    }
  };

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-slate-50">
        <div className="animate-spin rounded-full h-10 w-10 border-[3px] border-gray-200 border-t-primary-600" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-100 sticky top-0 z-20">
        <div className="h-0.5 bg-gradient-to-r from-primary-500 via-indigo-500 to-violet-500" />
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3.5 flex justify-between items-center">
          <div className="flex items-center gap-2.5">
            <span className="text-xl">🔬</span>
            <h1 className="text-lg font-bold text-gray-900 tracking-tight">ResearchMate</h1>
          </div>
          <div className="flex items-center gap-3">
            <span className="text-sm text-gray-400 hidden sm:block">{user?.email}</span>
            <div className="w-8 h-8 rounded-full bg-primary-100 text-primary-700 flex items-center justify-center text-sm font-bold select-none">
              {(user?.full_name?.[0] ?? user?.email?.[0] ?? '?').toUpperCase()}
            </div>
            <button
              onClick={logout}
              className="text-sm text-gray-400 hover:text-gray-700 px-3 py-1.5 rounded-lg hover:bg-gray-100 transition-colors"
            >
              Sign out
            </button>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Stats Bar */}
        <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 mb-6">
          <div className="bg-white rounded-2xl border border-gray-100 shadow-sm p-4">
            <p className="text-xs font-medium text-gray-400 uppercase tracking-wide mb-1">Total</p>
            <p className="text-2xl font-bold text-gray-800">{stats?.total || 0}</p>
          </div>
          <div className="bg-emerald-50 rounded-2xl border border-emerald-100 shadow-sm p-4">
            <p className="text-xs font-medium text-emerald-600 uppercase tracking-wide mb-1">Saved</p>
            <p className="text-2xl font-bold text-emerald-700">{stats?.saved || 0}</p>
          </div>
          <div className="bg-sky-50 rounded-2xl border border-sky-100 shadow-sm p-4">
            <p className="text-xs font-medium text-sky-600 uppercase tracking-wide mb-1">Viewed</p>
            <p className="text-2xl font-bold text-sky-700">{stats?.viewed || 0}</p>
          </div>
          <div className="bg-rose-50 rounded-2xl border border-rose-100 shadow-sm p-4">
            <p className="text-xs font-medium text-rose-500 uppercase tracking-wide mb-1">Dismissed</p>
            <p className="text-2xl font-bold text-rose-600">{stats?.dismissed || 0}</p>
          </div>
          <div className="bg-white rounded-2xl border border-gray-100 shadow-sm p-4 flex items-center">
            {stats?.ready_for_training ? (
              <span className="inline-flex items-center gap-1.5 text-sm font-semibold text-emerald-700">
                <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                ML model active
              </span>
            ) : (
              <span className="text-xs text-gray-400 leading-relaxed">
                <span className="font-semibold text-gray-600">{stats?.interactions_until_training ?? 50}</span> more interactions to enable ML ranking
              </span>
            )}
          </div>
        </div>

        {/* Tabs */}
        <div className="flex border-b border-gray-200 mb-6 -mx-0 overflow-x-auto scrollbar-none">
          {(
            [
              { id: 'feed', label: 'Daily Feed' },
              { id: 'hot', label: "What's Hot", Icon: Flame },
              { id: 'saved', label: 'Saved Items' },
              { id: 'chat', label: 'Agent Chat', Icon: Bot },
              { id: 'settings', label: 'Settings' },
            ] as { id: string; label: string; Icon?: typeof Bot }[]
          ).map(({ id, label, Icon }) => (
            <button
              key={id}
              onClick={() => setActiveTab(id as typeof activeTab)}
              className={`relative flex items-center gap-1.5 px-4 py-3 text-sm font-medium whitespace-nowrap transition-colors duration-150 ${
                activeTab === id
                  ? 'text-primary-600'
                  : 'text-gray-500 hover:text-gray-900'
              }`}
            >
              {Icon && <Icon className="w-4 h-4" />}
              {label}
              {activeTab === id && (
                <span className="absolute bottom-0 left-0 right-0 h-0.5 bg-primary-600 rounded-t-full" />
              )}
            </button>
          ))}
        </div>

        {activeTab === 'feed' && (
          <>
            {/* Personalization + Feed Controls */}
            <FeedControls
              user={user}
              timeWindow={timeWindow}
              setTimeWindow={setTimeWindow}
              feedMode={feedMode}
              setFeedMode={setFeedMode}
              isGenerating={isGenerating}
              feedProgress={feedProgress}
              onGenerate={() => generateFeed(false)}
              onProfileUpdate={fetchProfile}
            />

            {/* Onboarding — only while the profile is still thin */}
            {coldStart && coldStart.seeds.length > 0 && (
              <ColdStartSection
                data={coldStart}
                rated={ratedSeeds}
                onRate={rateSeed}
                onDismiss={() => setColdStart(null)}
              />
            )}


            {/* Feed Results */}
            {feed && (
              <div className="space-y-6 mt-6">
                {/* Papers */}
                {feed.papers.length > 0 && (
                  <div>
                    <div className="flex items-center justify-between gap-3 mb-4">
                      <div className="flex items-center gap-2.5">
                        <div className="w-8 h-8 rounded-xl bg-sky-100 flex items-center justify-center">
                          <BookOpen className="w-4 h-4 text-sky-600" />
                        </div>
                        <h2 className="text-base font-bold text-gray-900">
                          Research Papers
                          <span className="ml-2 text-sm font-normal text-gray-400">({feed.papers.length})</span>
                        </h2>
                        {refineResult && (
                          <span
                            className={`text-xs font-medium px-2 py-0.5 rounded-full ${
                              refineResult.passed
                                ? 'bg-emerald-50 text-emerald-700'
                                : 'bg-amber-50 text-amber-700'
                            }`}
                          >
                            quality {refineResult.score.toFixed(2)} · {refineResult.rounds} round
                            {refineResult.rounds > 1 ? 's' : ''}
                          </span>
                        )}
                      </div>
                      <div className="flex items-center gap-2">
                        <button
                          onClick={handleRefine}
                          disabled={isRefining || isGenerating}
                          title="Re-score this feed and swap out weak matches"
                          className="flex items-center gap-1.5 text-xs font-medium px-3 py-1.5 rounded-lg border border-gray-200 text-gray-600 hover:text-primary-600 hover:border-primary-200 hover:bg-primary-50 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                        >
                          <Sparkles className={`w-3.5 h-3.5 ${isRefining ? 'animate-pulse' : ''}`} />
                          {isRefining ? 'Refining…' : 'Refine'}
                        </button>
                        {/* Separate from Generate: this is the one action that
                            deliberately replaces today's feed with lower-ranked
                            papers, so it should be chosen, not stumbled into. */}
                        <button
                          onClick={() => generateFeed(true)}
                          disabled={isGenerating || isRefining}
                          title="Replace today's feed with the next set of papers"
                          className="flex items-center gap-1.5 text-xs font-medium px-3 py-1.5 rounded-lg border border-gray-200 text-gray-600 hover:text-gray-900 hover:border-gray-300 hover:bg-gray-50 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                        >
                          <RefreshCw className={`w-3.5 h-3.5 ${isGenerating ? 'animate-spin' : ''}`} />
                          New batch
                        </button>
                      </div>
                    </div>
                    <div className="space-y-4">
                      {feed.papers.map((paper) => (
                        <PaperCard
                          key={paper.id}
                          paper={paper}
                          onInteraction={handleInteraction}
                          onAskAgent={(ctx) => { setActivePaperContext(ctx); setActiveTab('chat'); }}
                        />
                      ))}
                    </div>
                  </div>
                )}

                {/* Meta info */}
                <div className="flex items-center justify-center gap-2 pt-2">
                  <span className={`inline-flex items-center gap-1.5 text-xs font-medium px-3 py-1.5 rounded-full ${feed.used_ml_ranking ? 'bg-emerald-50 text-emerald-700' : 'bg-gray-100 text-gray-500'}`}>
                    {feed.used_ml_ranking ? '🧠 Personalized ML ranking' : '📊 Heuristic ranking — interact more to unlock ML'}
                  </span>
                  <span className="text-xs text-gray-400">
                    {feed.total_papers_considered} papers considered
                  </span>
                </div>
              </div>
            )}

            {!feed && !isGenerating && (
              <div className="text-center py-10 animate-in">
                <div className="text-5xl mb-4">🔬</div>
                <h3 className="text-lg font-semibold text-gray-700 mb-2">Generate your personalized feed</h3>
                <p className="text-sm text-gray-400 max-w-sm mx-auto">
                  Click <strong>Generate Feed</strong> to discover research papers tailored to your interests.
                </p>
                <button
                  onClick={() => setActiveTab('hot')}
                  className="mt-4 text-sm text-orange-600 hover:text-orange-700 font-medium transition-colors"
                >
                  Or see what the community is reading &rarr;
                </button>
              </div>
            )}
          </>
        )}

        {activeTab === 'hot' && (
          <WhatsHotSection
            data={whatsHot}
            isLoading={isLoadingHot}
            isRefreshing={isRefreshingHot}
            onRefresh={refreshWhatsHot}
          />
        )}

        {activeTab === 'saved' && <SavedItems />}

        {activeTab === 'chat' && (
          <AgentChat
            paperContext={activePaperContext}
            onClearPaperContext={() => setActivePaperContext(null)}
          />
        )}

        {activeTab === 'settings' && <UserSettings user={user} onUpdate={fetchProfile} />}
      </main>
    </div>
  );
}

/* ── Feed Controls with inline personalization ── */
function FeedControls({
  user,
  timeWindow,
  setTimeWindow,
  feedMode,
  setFeedMode,
  isGenerating,
  feedProgress,
  onGenerate,
  onProfileUpdate,
}: {
  user: any;
  timeWindow: number;
  setTimeWindow: (v: number) => void;
  feedMode: 'recommended' | 'latest';
  setFeedMode: (v: 'recommended' | 'latest') => void;
  isGenerating: boolean;
  feedProgress: string;
  onGenerate: () => void;
  onProfileUpdate: () => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const [focusAreas, setFocusAreas] = useState<string[]>(user?.focus_areas || []);
  const [selectedExamples, setSelectedExamples] = useState<string[]>(
    (user?.interests || []).filter((i: string) => EXAMPLE_INTERESTS.includes(i))
  );
  const [customInterests, setCustomInterests] = useState(
    (user?.interests || []).filter((i: string) => !EXAMPLE_INTERESTS.includes(i)).join('\n')
  );
  const [isSaving, setIsSaving] = useState(false);

  // Sync state when user prop changes
  useEffect(() => {
    if (user) {
      setFocusAreas(user.focus_areas || []);
      setSelectedExamples(
        (user.interests || []).filter((i: string) => EXAMPLE_INTERESTS.includes(i))
      );
      setCustomInterests(
        (user.interests || []).filter((i: string) => !EXAMPLE_INTERESTS.includes(i)).join('\n')
      );
    }
  }, [user]);

  const toggleArea = (area: string) => {
    setFocusAreas((prev: string[]) =>
      prev.includes(area) ? prev.filter((a: string) => a !== area) : [...prev, area]
    );
  };

  const toggleExample = (interest: string) => {
    setSelectedExamples((prev: string[]) =>
      prev.includes(interest) ? prev.filter((i: string) => i !== interest) : [...prev, interest]
    );
  };

  const handleSavePreferences = async () => {
    setIsSaving(true);
    try {
      const customLines = customInterests
        .split('\n')
        .map((l: string) => l.trim())
        .filter((l: string) => l.length > 0);
      const allInterests = Array.from(new Set([...selectedExamples, ...customLines]));

      await authApi.updateProfile({
        interests: allInterests,
        focus_areas: focusAreas,
      });
      toast.success('Preferences saved!');
      onProfileUpdate();
    } catch (error: any) {
      toast.error(error.response?.data?.detail || 'Failed to save preferences');
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className="card mb-6">
      {/* Top row: mode + time window + generate */}
      <div className="flex flex-wrap items-center gap-4">
        <div>
          <label className="block text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1.5">Mode</label>
          <div className="flex rounded-xl border border-gray-200 overflow-hidden bg-gray-50 p-0.5 gap-0.5">
            <button
              onClick={() => setFeedMode('recommended')}
              className={`px-3 py-1.5 text-sm font-medium transition-all duration-150 rounded-lg ${
                feedMode === 'recommended'
                  ? 'bg-white text-primary-600 shadow-sm'
                  : 'text-gray-500 hover:text-gray-700'
              }`}
            >
              Recommended
            </button>
            <button
              onClick={() => setFeedMode('latest')}
              className={`px-3 py-1.5 text-sm font-medium transition-all duration-150 rounded-lg ${
                feedMode === 'latest'
                  ? 'bg-white text-primary-600 shadow-sm'
                  : 'text-gray-500 hover:text-gray-700'
              }`}
            >
              Latest
            </button>
          </div>
        </div>

        {feedMode === 'latest' && (
          <div>
            <label className="block text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1.5">
              Time window
            </label>
            <select
              value={timeWindow}
              onChange={(e) => setTimeWindow(Number(e.target.value))}
              className="input-field w-40"
            >
              <option value={1}>1 day</option>
              <option value={7}>1 week</option>
              <option value={30}>1 month</option>
              <option value={365}>1 year</option>
            </select>
          </div>
        )}

        {/* Current focus areas preview */}
        <div className="flex-1 min-w-0">
          <label className="block text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1.5">Focus areas</label>
          <div className="flex flex-wrap gap-1.5">
            {(user?.focus_areas || []).length > 0 ? (
              (user.focus_areas as string[]).map((area: string) => (
                <span key={area} className="px-2.5 py-0.5 rounded-full text-xs font-medium bg-primary-100 text-primary-700">
                  {area}
                </span>
              ))
            ) : (
              <span className="text-xs text-gray-400">None selected</span>
            )}
            {(user?.interests || []).length > 0 && (
              <span className="text-xs text-gray-400 ml-1">
                + {(user.interests as string[]).length} interest{(user.interests as string[]).length !== 1 ? 's' : ''}
              </span>
            )}
          </div>
        </div>

        <div className="flex items-end gap-2">
          <button
            onClick={() => setExpanded(!expanded)}
            className="btn-secondary text-sm flex items-center gap-1"
          >
            <Settings className="w-4 h-4" />
            {expanded ? 'Hide' : 'Edit'}
            {expanded ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
          </button>
          <button
            onClick={onGenerate}
            disabled={isGenerating}
            className="btn-primary flex items-center gap-2 disabled:opacity-50"
          >
            {isGenerating ? (
              <>
                <RefreshCw className="w-4 h-4 animate-spin" />
                Generating...
              </>
            ) : (
              <>
                <RefreshCw className="w-4 h-4" />
                Generate Feed
              </>
            )}
          </button>
        </div>
      </div>

      {/* Feed generation progress */}
      {isGenerating && feedProgress && (
        <div className="mt-4 flex items-start gap-3 bg-primary-50 border border-primary-100 rounded-xl px-4 py-3">
          <RefreshCw className="w-4 h-4 text-primary-600 animate-spin shrink-0 mt-0.5" />
          <div>
            <p className="text-sm font-semibold text-primary-800">{feedProgress}</p>
            <p className="text-xs text-primary-400 mt-0.5">Generation takes 30–90 seconds — this runs in the background</p>
          </div>
        </div>
      )}

      {/* Expandable personalization section */}
      {expanded && (
        <div className="mt-6 pt-6 border-t space-y-5">
          {/* Focus Areas */}
          <div>
            <h3 className="text-sm font-semibold text-gray-900 mb-2">Focus Areas</h3>
            <p className="text-xs text-gray-500 mb-3">Select areas to prioritize in your feed</p>
            <div className="flex flex-wrap gap-2">
              {AVAILABLE_AREAS.map((area) => (
                <button
                  key={area}
                  onClick={() => toggleArea(area)}
                  className={`px-4 py-1.5 rounded-full text-sm font-medium transition-colors ${
                    focusAreas.includes(area)
                      ? 'bg-primary-600 text-white'
                      : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                  }`}
                >
                  {area}
                </button>
              ))}
            </div>
          </div>

          {/* Research Interests */}
          <div>
            <h3 className="text-sm font-semibold text-gray-900 mb-2">Research Interests</h3>
            <p className="text-xs text-gray-500 mb-3">
              Used for semantic similarity matching and personalized summaries. Specific phrases work best.
            </p>

            <div className="flex flex-wrap gap-2 mb-3">
              {EXAMPLE_INTERESTS.map((interest) => (
                <button
                  key={interest}
                  onClick={() => toggleExample(interest)}
                  className={`px-3 py-1 rounded-full text-xs font-medium transition-colors ${
                    selectedExamples.includes(interest)
                      ? 'bg-primary-600 text-white'
                      : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                  }`}
                >
                  {interest}
                </button>
              ))}
            </div>

            <textarea
              value={customInterests}
              onChange={(e) => setCustomInterests(e.target.value)}
              rows={2}
              placeholder="Custom interests (one per line), e.g. graph neural networks"
              className="input-field w-full text-sm"
            />
          </div>

          {/* Save button */}
          <button
            onClick={handleSavePreferences}
            disabled={isSaving}
            className="btn-primary text-sm flex items-center gap-2 disabled:opacity-50"
          >
            {isSaving ? (
              <>
                <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                Saving...
              </>
            ) : (
              <>
                <Save className="w-3.5 h-3.5" />
                Save Preferences
              </>
            )}
          </button>
        </div>
      )}
    </div>
  );
}

/* ── Formatted Summary ── */
function FormattedSummary({ text }: { text: string }) {
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
function PaperCard({
  paper,
  onInteraction,
  onAskAgent,
}: {
  paper: Paper;
  onInteraction: (type: 'paper' | 'article', id: number, interaction: 'viewed' | 'saved' | 'dismissed') => void;
  onAskAgent: (ctx: { title: string; abstract: string | null; arxivId: string; arxivUrl: string }) => void;
}) {
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
        {paper.summary && <FormattedSummary text={paper.summary} />}
      </div>
      <div className="flex items-center gap-2 mt-4 pt-4 border-t border-gray-100">
        <a
          href={paper.arxiv_url}
          target="_blank"
          rel="noopener noreferrer"
          onClick={() => onInteraction('paper', paper.id, 'viewed')}
          className="btn-secondary text-sm py-1.5"
        >
          <ExternalLink className="w-3.5 h-3.5" />
          View paper
        </a>
        <button
          onClick={() => onInteraction('paper', paper.id, 'saved')}
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
        <button
          onClick={() => onInteraction('paper', paper.id, 'dismissed')}
          className="ml-auto text-gray-300 hover:text-rose-400 p-2 rounded-lg hover:bg-rose-50 transition-colors"
          title="Dismiss"
        >
          <X className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
}

/* ── Digest Card ("What's Hot in CS") ── */
function DigestCard({
  article,
  onInteraction,
}: {
  article: Article;
  onInteraction: (type: 'paper' | 'article', id: number, interaction: 'viewed' | 'saved' | 'dismissed') => void;
}) {
  return (
    <div
      className="card-lift animate-in"
      style={{ borderLeft: '3px solid #f97316' }}
    >
      <div className="flex-1">
        {/* Title + source */}
        <div className="flex items-start justify-between gap-3 mb-1">
          <h3 className="font-semibold text-gray-900 leading-snug">
            <span className="text-gray-400 mr-1.5">#{article.rank}</span>
            {article.title}
          </h3>
        </div>
        <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-gray-400 mb-3">
          <span className="font-medium text-orange-500">{article.source}</span>
        </div>

        {/* Digest summary — the main content */}
        {article.digest_summary ? (
          <p className="text-sm text-gray-700 leading-relaxed mb-3">
            {article.digest_summary}
          </p>
        ) : article.summary ? (
          <FormattedSummary text={article.summary} />
        ) : null}

        {/* Source citation link */}
        <div className="flex items-center gap-1 text-xs text-gray-400">
          <span>Source:</span>
          <a
            href={article.url}
            target="_blank"
            rel="noopener noreferrer"
            onClick={() => onInteraction('article', article.id, 'viewed')}
            className="text-orange-500 hover:text-orange-600 hover:underline truncate max-w-[300px]"
          >
            {article.url.replace(/^https?:\/\/(www\.)?/, '').split('/')[0]}
          </a>
        </div>
      </div>

      <div className="flex items-center gap-2 mt-4 pt-4 border-t border-gray-100">
        <a
          href={article.url}
          target="_blank"
          rel="noopener noreferrer"
          onClick={() => onInteraction('article', article.id, 'viewed')}
          className="btn-secondary text-sm py-1.5"
        >
          <ExternalLink className="w-3.5 h-3.5" />
          Read full article
        </a>
        <button
          onClick={() => onInteraction('article', article.id, 'saved')}
          className="btn-secondary text-sm py-1.5"
        >
          <Save className="w-3.5 h-3.5" />
          Save
        </button>
        <button
          onClick={() => onInteraction('article', article.id, 'dismissed')}
          className="ml-auto text-gray-300 hover:text-rose-400 p-2 rounded-lg hover:bg-rose-50 transition-colors"
          title="Dismiss"
        >
          <X className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
}

/* ── User Settings (profile only) ── */
function UserSettings({ user, onUpdate }: { user: any; onUpdate: () => void }) {
  const [focusAreas, setFocusAreas] = useState<string[]>(user?.focus_areas || []);
  const [selectedExamples, setSelectedExamples] = useState<string[]>(
    (user?.interests || []).filter((i: string) => EXAMPLE_INTERESTS.includes(i))
  );
  const [customInterests, setCustomInterests] = useState(
    (user?.interests || []).filter((i: string) => !EXAMPLE_INTERESTS.includes(i)).join('\n')
  );
  const [fullName, setFullName] = useState(user?.full_name || '');
  const [isSaving, setIsSaving] = useState(false);

  const toggleArea = (area: string) => {
    setFocusAreas((prev: string[]) =>
      prev.includes(area) ? prev.filter((a: string) => a !== area) : [...prev, area]
    );
  };

  const toggleExample = (interest: string) => {
    setSelectedExamples((prev: string[]) =>
      prev.includes(interest) ? prev.filter((i: string) => i !== interest) : [...prev, interest]
    );
  };

  const handleSave = async () => {
    setIsSaving(true);
    try {
      const customLines = customInterests
        .split('\n')
        .map((l: string) => l.trim())
        .filter((l: string) => l.length > 0);
      const allInterests = Array.from(new Set([...selectedExamples, ...customLines]));

      await authApi.updateProfile({
        full_name: fullName || undefined,
        interests: allInterests,
        focus_areas: focusAreas,
      });
      toast.success('Settings saved!');
      onUpdate();
    } catch (error: any) {
      toast.error(error.response?.data?.detail || 'Failed to save settings');
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Profile */}
      <div className="card">
        <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
          <Settings className="w-5 h-5" />
          Profile
        </h2>
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Email</label>
            <input
              type="text"
              value={user?.email || ''}
              disabled
              className="input-field w-full bg-gray-100 cursor-not-allowed"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Full name</label>
            <input
              type="text"
              value={fullName}
              onChange={(e) => setFullName(e.target.value)}
              placeholder="Your name"
              className="input-field w-full"
            />
          </div>
        </div>
      </div>

      {/* Focus Areas */}
      <div className="card">
        <h2 className="text-lg font-semibold text-gray-900 mb-2">Focus Areas</h2>
        <p className="text-sm text-gray-500 mb-4">Select areas to prioritize in your feed</p>
        <div className="flex flex-wrap gap-2">
          {AVAILABLE_AREAS.map((area) => (
            <button
              key={area}
              onClick={() => toggleArea(area)}
              className={`px-4 py-2 rounded-full text-sm font-medium transition-colors ${
                focusAreas.includes(area)
                  ? 'bg-primary-600 text-white'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              {area}
            </button>
          ))}
        </div>
      </div>

      {/* Interests */}
      <div className="card">
        <h2 className="text-lg font-semibold text-gray-900 mb-2">Research Interests</h2>
        <p className="text-sm text-gray-500 mb-4">
          Used to compute semantic similarity for recommendations. Longer, specific phrases work best.
        </p>

        <h3 className="text-sm font-medium text-gray-700 mb-2">Select from examples</h3>
        <div className="flex flex-wrap gap-2 mb-4">
          {EXAMPLE_INTERESTS.map((interest) => (
            <button
              key={interest}
              onClick={() => toggleExample(interest)}
              className={`px-3 py-1.5 rounded-full text-xs font-medium transition-colors ${
                selectedExamples.includes(interest)
                  ? 'bg-primary-600 text-white'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              {interest}
            </button>
          ))}
        </div>

        <h3 className="text-sm font-medium text-gray-700 mb-2">Custom interests (one per line)</h3>
        <textarea
          value={customInterests}
          onChange={(e) => setCustomInterests(e.target.value)}
          rows={4}
          placeholder={"e.g.\ngraph neural networks\nfederated learning\nAI for healthcare"}
          className="input-field w-full"
        />
      </div>

      {/* Save Button */}
      <button
        onClick={handleSave}
        disabled={isSaving}
        className="btn-primary flex items-center gap-2 disabled:opacity-50"
      >
        {isSaving ? (
          <>
            <RefreshCw className="w-4 h-4 animate-spin" />
            Saving...
          </>
        ) : (
          <>
            <Save className="w-4 h-4" />
            Save Settings
          </>
        )}
      </button>
    </div>
  );
}

// ── Agent Chat constants ──────────────────────────────────────────────────────

const INTENT_COLORS: Record<string, string> = {
  research_qa: 'bg-blue-100 text-blue-700',
  recommendation: 'bg-green-100 text-green-700',
  document_management: 'bg-amber-100 text-amber-700',
  general_chat: 'bg-gray-100 text-gray-600',
};

const INTENT_LABELS: Record<string, string> = {
  research_qa: 'Research Q&A',
  recommendation: 'Recommendations',
  document_management: 'Documents',
  general_chat: 'General Chat',
};

const TOOL_LABELS: Record<string, string> = {
  search_knowledge_base: 'Search KB',
  get_personalized_feed: 'Get Feed',
  list_user_documents: 'List Docs',
  search_user_documents: 'Search Docs',
};

const TOOL_ICONS: Record<string, string> = {
  search_knowledge_base: '🔍',
  get_personalized_feed: '📰',
  list_user_documents: '📁',
  search_user_documents: '🔎',
};

interface ResearchTask {
  id: number;
  title: string;
  intent: string;
  status: 'pending' | 'running' | 'done';
  citations?: number;
}

interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  intent?: string;
  intentMethod?: string;
  intentConfidence?: number;
  agentUsed?: string;
  toolCalls?: { tool: string; count: number }[];
  citations?: { title: string; url: string; type: string }[];
  isStreaming?: boolean;
  researchPlan?: ResearchTask[];
}

/* ── Agent Chat ── */
function AgentChat({
  paperContext = null,
  onClearPaperContext,
}: {
  paperContext?: { title: string; abstract: string | null; arxivId: string; arxivUrl: string } | null;
  onClearPaperContext?: () => void;
}) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [showKB, setShowKB] = useState(false);
  const [kbDocs, setKbDocs] = useState<any[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const paperContextInjected = useRef(false);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    loadKbDocs();
    return () => {
      abortRef.current?.abort();
    };
  }, []);

  useEffect(() => {
    paperContextInjected.current = false;
  }, [paperContext]);

  const loadKbDocs = async () => {
    try {
      const data = await qaApi.listDocuments();
      setKbDocs(data.documents || data || []);
    } catch {}
  };

  const handleKbUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;
    setIsUploading(true);
    let uploaded = 0;
    for (let i = 0; i < files.length; i++) {
      try {
        await qaApi.uploadDocument(files[i], files[i].name);
        uploaded++;
      } catch (err: any) {
        toast.error(`Failed to upload ${files[i].name}: ${err.response?.data?.detail || 'Unknown error'}`);
      }
    }
    if (uploaded > 0) {
      toast.success(`Uploaded ${uploaded} document(s) to knowledge base`);
      loadKbDocs();
    }
    setIsUploading(false);
    e.target.value = '';
  };

  const handleKbDelete = async (docId: number, title: string) => {
    try {
      await qaApi.deleteDocument(docId);
      toast.success(`Deleted "${title}"`);
      loadKbDocs();
    } catch {
      toast.error('Failed to delete document');
    }
  };

  const sendMessage = async () => {
    if (!input.trim() || isStreaming) return;

    const userMsg: ChatMessage = { id: `u-${Date.now()}`, role: 'user', content: input };
    const assistantId = `a-${Date.now() + 1}`;
    const assistantMsg: ChatMessage = {
      id: assistantId,
      role: 'assistant',
      content: '',
      isStreaming: true,
      toolCalls: [],
    };

    setMessages(prev => [...prev, userMsg, assistantMsg]);
    let text = input;
    if (paperContext && !paperContextInjected.current) {
      const abstractPreview = paperContext.abstract
        ? paperContext.abstract.slice(0, 800)
        : 'Not available';
      text = `I want to ask about this research paper:\n\nTitle: ${paperContext.title}\narXiv: ${paperContext.arxivId}\n\nAbstract: ${abstractPreview}\n\n---\n\nMy question: ${input}`;
      paperContextInjected.current = true;
    }
    setInput('');
    setIsStreaming(true);

    abortRef.current = new AbortController();

    try {
      const token = chatApi.getAuthToken();
      const res = await fetch(chatApi.getStreamUrl(), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify({ message: text, session_id: sessionId }),
        signal: abortRef.current.signal,
      });

      if (!res.ok || !res.body) {
        throw new Error(`HTTP ${res.status}`);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = '';

      const applyEvent = (event: AgentEvent) => {
        setMessages(prev =>
          prev.map(m => {
            if (m.id !== assistantId) return m;
            switch (event.type) {
              case 'session':
                setSessionId(event.session_id);
                return m;
              case 'intent':
                return { ...m, intent: event.value, intentMethod: event.method, intentConfidence: event.confidence };
              case 'agent':
                return { ...m, agentUsed: event.value };
              case 'tool_call':
                return { ...m, toolCalls: [...(m.toolCalls || []), { tool: event.tool, count: -1 }] };
              case 'tool_result': {
                const updated = (m.toolCalls || []).map(tc =>
                  tc.tool === event.tool && tc.count === -1 ? { ...tc, count: event.count } : tc
                );
                return { ...m, toolCalls: updated };
              }
              case 'token':
                return { ...m, content: m.content + event.value };
              case 'done':
                return { ...m, isStreaming: false, citations: event.citations };
              case 'error':
                return { ...m, isStreaming: false, content: m.content || `Error: ${event.value}` };
              case 'plan':
                return {
                  ...m,
                  researchPlan: event.tasks.map(t => ({ ...t, status: 'pending' as const })),
                };
              case 'task_started': {
                const updated = (m.researchPlan || []).map(t =>
                  t.id === event.id ? { ...t, status: 'running' as const } : t
                );
                return { ...m, researchPlan: updated };
              }
              case 'task_done': {
                const updated = (m.researchPlan || []).map(t =>
                  t.id === event.id ? { ...t, status: 'done' as const, citations: event.citations } : t
                );
                return { ...m, researchPlan: updated };
              }
              default:
                return m;
            }
          })
        );
      };

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        let idx: number;
        while ((idx = buf.indexOf('\n')) !== -1) {
          const line = buf.slice(0, idx);
          buf = buf.slice(idx + 1);
          if (line.startsWith('data: ')) {
            try {
              applyEvent(JSON.parse(line.slice(6)) as AgentEvent);
            } catch (_) { /* skip malformed */ }
          }
        }
      }
    } catch (err: any) {
      if (err.name !== 'AbortError') {
        setMessages(prev =>
          prev.map(m =>
            m.id === assistantId
              ? { ...m, isStreaming: false, content: m.content || 'Failed to get a response. Please try again.' }
              : m
          )
        );
      }
    } finally {
      setIsStreaming(false);
      setMessages(prev =>
        prev.map(m => m.id === assistantId ? { ...m, isStreaming: false } : m)
      );
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const clearChat = () => {
    abortRef.current?.abort();
    setMessages([]);
    setSessionId(null);
    setIsStreaming(false);
  };

  return (
    <div className="flex flex-col" style={{ height: 'calc(100vh - 320px)', minHeight: 480 }}>
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-lg font-semibold text-gray-900 flex items-center gap-2">
            <Bot className="w-5 h-5 text-primary-600" />
            Agent Chat
          </h2>
          <p className="text-xs text-gray-400 mt-0.5">
            Powered by multi-agent routing — ask research questions, get recommendations, or manage documents
          </p>
        </div>
        <button
          onClick={clearChat}
          className="text-xs text-gray-400 hover:text-gray-600 flex items-center gap-1 px-2 py-1 rounded hover:bg-gray-100"
        >
          <X className="w-3.5 h-3.5" />
          New Chat
        </button>
      </div>

      {/* Paper Context Banner */}
      {paperContext && (
        <div className="shrink-0 mb-3 flex items-start gap-3 bg-sky-50 border border-sky-200 rounded-xl px-4 py-3">
          <BookOpen className="w-4 h-4 text-sky-600 shrink-0 mt-0.5" />
          <div className="flex-1 min-w-0">
            <p className="text-xs font-semibold text-sky-700 mb-0.5">Discussing paper</p>
            <p className="text-sm text-sky-900 font-medium leading-snug line-clamp-2">{paperContext.title}</p>
            {paperContext.arxivUrl && (
              <a
                href={paperContext.arxivUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="text-xs text-sky-600 hover:underline"
              >
                arXiv:{paperContext.arxivId} →
              </a>
            )}
          </div>
          <button
            onClick={onClearPaperContext}
            className="text-sky-400 hover:text-sky-600 p-1 shrink-0 transition-colors"
            title="Clear paper context"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      )}

      {/* Knowledge Base Panel */}
      <div className="shrink-0 mb-3 border border-gray-200 rounded-xl overflow-hidden">
        <button
          onClick={() => setShowKB(!showKB)}
          className="w-full flex items-center justify-between px-4 py-2.5 bg-gray-50 hover:bg-gray-100 text-sm font-medium text-gray-700 transition-colors"
        >
          <div className="flex items-center gap-2">
            <Upload className="w-4 h-4 text-gray-500" />
            Knowledge Base
            {kbDocs.length > 0 && (
              <span className="text-xs bg-primary-100 text-primary-700 px-2 py-0.5 rounded-full font-medium">
                {kbDocs.length} doc{kbDocs.length !== 1 ? 's' : ''}
              </span>
            )}
          </div>
          {showKB ? <ChevronUp className="w-4 h-4 text-gray-400" /> : <ChevronDown className="w-4 h-4 text-gray-400" />}
        </button>
        {showKB && (
          <div className="p-4 space-y-3 bg-white">
            <label className="inline-flex items-center gap-2 btn-secondary text-sm cursor-pointer">
              <Upload className="w-4 h-4" />
              {isUploading ? 'Uploading...' : 'Upload Files (.txt, .md, .pdf)'}
              <input
                type="file"
                accept=".txt,.md,.pdf"
                multiple
                onChange={handleKbUpload}
                disabled={isUploading}
                className="hidden"
              />
            </label>
            {kbDocs.length > 0 ? (
              <div className="space-y-1.5 max-h-36 overflow-y-auto">
                {kbDocs.map((doc: any) => (
                  <div key={doc.id} className="flex items-center justify-between bg-gray-50 rounded-lg px-3 py-1.5">
                    <div className="min-w-0">
                      <span className="text-sm text-gray-700 truncate block">{doc.title}</span>
                      {doc.chunk_count > 0 && (
                        <span className="text-xs text-gray-400">{doc.chunk_count} chunks</span>
                      )}
                    </div>
                    <button
                      onClick={() => handleKbDelete(doc.id, doc.title)}
                      className="text-gray-300 hover:text-rose-400 p-1 ml-2 shrink-0 transition-colors"
                      title="Delete document"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-xs text-gray-400">No documents yet. Uploaded files will be searchable in Agent Chat.</p>
            )}
          </div>
        )}
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto space-y-4 pr-1">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-center py-12">
            <Bot className="w-12 h-12 text-gray-200 mb-3" />
            <p className="text-gray-500 font-medium">Ask me anything</p>
            <p className="text-sm text-gray-400 mt-1 max-w-sm">
              Research questions, paper recommendations, or searches across your uploaded documents
            </p>
            <div className="mt-4 flex flex-wrap gap-2 justify-center">
              {(paperContext ? [
                '这篇论文的核心贡献是什么？',
                '它有哪些局限性和不足？',
                '帮我找引用了这篇论文的相关工作',
              ] : [
                'What are recent advances in transformers?',
                'Recommend papers on reinforcement learning',
                'List my uploaded documents',
              ]).map(ex => (
                <button
                  key={ex}
                  onClick={() => setInput(ex)}
                  className="text-xs bg-gray-50 border border-gray-200 rounded-full px-3 py-1.5 text-gray-600 hover:bg-primary-50 hover:border-primary-200 hover:text-primary-700 transition-colors"
                >
                  {ex}
                </button>
              ))}
            </div>
          </div>
        )}
        {messages.map(msg => (
          <ChatBubble key={msg.id} message={msg} />
        ))}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="flex gap-2 mt-4 pt-4 border-t">
        <input
          type="text"
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask a question..."
          className="input-field flex-1"
          disabled={isStreaming}
        />
        <button
          onClick={sendMessage}
          disabled={isStreaming || !input.trim()}
          className="btn-primary flex items-center gap-2 disabled:opacity-50 shrink-0"
        >
          {isStreaming ? (
            <RefreshCw className="w-4 h-4 animate-spin" />
          ) : (
            <Send className="w-4 h-4" />
          )}
          Send
        </button>
      </div>
    </div>
  );
}

/* ── Chat Bubble ── */
function ChatBubble({ message }: { message: ChatMessage }) {
  if (message.role === 'user') {
    return (
      <div className="flex justify-end">
        <div className="bg-primary-600 text-white rounded-2xl rounded-tr-sm px-4 py-3 max-w-[80%] text-sm leading-relaxed">
          {message.content}
        </div>
      </div>
    );
  }

  return (
    <div className="flex justify-start">
      <div className="flex-1 max-w-[88%] space-y-2">
        {/* Intent + agent badges */}
        {(message.intent || message.agentUsed) && (
          <div className="flex flex-wrap items-center gap-1.5">
            {message.intent && (
              <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${INTENT_COLORS[message.intent] || 'bg-gray-100 text-gray-600'}`}>
                {INTENT_LABELS[message.intent] || message.intent}
              </span>
            )}
            {message.agentUsed && (
              <span className="text-xs px-2 py-0.5 rounded-full bg-purple-100 text-purple-700 font-medium">
                {message.agentUsed}
              </span>
            )}
            {message.intentMethod && message.intentConfidence !== undefined && (
              <span className="text-xs text-gray-400">
                via {message.intentMethod} · {Math.round(message.intentConfidence * 100)}%
              </span>
            )}
          </div>
        )}

        {/* Tool calls */}
        {message.toolCalls && message.toolCalls.length > 0 && (
          <div className="flex flex-wrap gap-1.5">
            {message.toolCalls.map((tc, i) => (
              <span
                key={i}
                className={`text-xs px-2 py-0.5 rounded-full border font-medium ${
                  tc.count === -1
                    ? 'bg-yellow-50 border-yellow-200 text-yellow-700 animate-pulse'
                    : 'bg-green-50 border-green-200 text-green-700'
                }`}
              >
                {TOOL_ICONS[tc.tool] || '⚙'} {TOOL_LABELS[tc.tool] || tc.tool}
                {tc.count >= 0 && ` (${tc.count})`}
              </span>
            ))}
          </div>
        )}

        {/* Deep research plan */}
        {message.researchPlan && message.researchPlan.length > 0 && (
          <div className="bg-indigo-50 border border-indigo-100 rounded-xl px-3 py-2 space-y-1">
            <p className="text-xs font-semibold text-indigo-600 uppercase tracking-wide mb-1.5">Research Plan</p>
            {message.researchPlan.map(task => (
              <div key={task.id} className="flex items-center gap-2 text-xs">
                {task.status === 'done' ? (
                  <span className="text-green-500">✓</span>
                ) : task.status === 'running' ? (
                  <RefreshCw className="w-3 h-3 text-indigo-500 animate-spin shrink-0" />
                ) : (
                  <span className="w-3 h-3 rounded-full border border-gray-300 shrink-0 inline-block" />
                )}
                <span className={`flex-1 ${task.status === 'done' ? 'text-gray-500 line-through' : task.status === 'running' ? 'text-indigo-700 font-medium' : 'text-gray-500'}`}>
                  {task.title}
                </span>
                {task.status === 'done' && task.citations !== undefined && task.citations > 0 && (
                  <span className="text-gray-400">{task.citations} src</span>
                )}
              </div>
            ))}
          </div>
        )}

        {/* Reply */}
        <div className="bg-white border border-gray-100 rounded-2xl rounded-tl-sm px-4 py-3 shadow-sm">
          {message.content ? (
            <p className="text-sm text-gray-800 whitespace-pre-wrap leading-relaxed">{message.content}</p>
          ) : (
            <div className="flex items-center gap-2 text-gray-400 text-sm">
              <RefreshCw className="w-3.5 h-3.5 animate-spin" />
              Thinking...
            </div>
          )}
          {message.isStreaming && message.content && (
            <span className="inline-block w-0.5 h-4 bg-gray-400 animate-pulse align-middle ml-0.5" />
          )}
        </div>

        {/* Citations */}
        {message.citations && message.citations.length > 0 && (
          <div className="flex flex-wrap gap-1.5">
            {message.citations.map((c, i) => (
              <a
                key={i}
                href={c.url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-xs px-2 py-0.5 rounded bg-blue-50 text-blue-600 hover:bg-blue-100 border border-blue-100 max-w-[200px] truncate"
              >
                {c.title || 'Source'} &rarr;
              </a>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

/* ── Saved Items ── */
function SavedItems() {
  const [saved, setSaved] = useState<{ papers: Paper[]; articles: Article[] } | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    loadSaved();
  }, []);

  const loadSaved = async () => {
    try {
      const data = await feedApi.getSaved();
      setSaved(data);
    } catch (error) {
      console.error('Failed to load saved items:', error);
    } finally {
      setIsLoading(false);
    }
  };

  if (isLoading) {
    return (
      <div className="text-center py-12">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600 mx-auto"></div>
      </div>
    );
  }

  if (!saved || (saved.papers.length === 0 && saved.articles.length === 0)) {
    return (
      <div className="text-center py-12 text-gray-500">
        No saved items yet. Save papers and articles from your feed!
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {saved.papers.length > 0 && (
        <div>
          <h2 className="text-xl font-semibold text-gray-900 mb-4">
            Saved Papers ({saved.papers.length})
          </h2>
          <div className="space-y-4">
            {saved.papers.map((paper) => (
              <div key={paper.id} className="card">
                <h3 className="font-semibold text-gray-900 mb-1">{paper.title}</h3>
                <p className="text-sm text-gray-500 mb-2">arXiv:{paper.arxiv_id}</p>
                <a
                  href={paper.arxiv_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-primary-600 hover:text-primary-700 text-sm"
                >
                  Open paper &rarr;
                </a>
              </div>
            ))}
          </div>
        </div>
      )}

      {saved.articles.length > 0 && (
        <div>
          <h2 className="text-xl font-semibold text-gray-900 mb-4">
            Saved Articles ({saved.articles.length})
          </h2>
          <div className="space-y-4">
            {saved.articles.map((article) => (
              <div key={article.id} className="card">
                <h3 className="font-semibold text-gray-900 mb-1">{article.title}</h3>
                <p className="text-sm text-gray-500 mb-2">{article.source}</p>
                <a
                  href={article.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-primary-600 hover:text-primary-700 text-sm"
                >
                  Read article &rarr;
                </a>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

/* ── Cold-start onboarding ── */
function ColdStartSection({
  data,
  rated,
  onRate,
  onDismiss,
}: {
  data: ColdStartResponse;
  rated: Record<number, 'saved' | 'dismissed'>;
  onRate: (seed: ColdStartSeed, verdict: 'saved' | 'dismissed') => void;
  onDismiss: () => void;
}) {
  // Progress counts ratings already on the server plus the ones made in this
  // session. Counting only this session would restart the bar at zero for a user
  // who already has interactions, showing a target they have partly met.
  const target = 5;
  const ratedCount = Math.min(
    target,
    data.interactions_recorded + Object.keys(rated).length
  );
  const remaining = Math.max(0, target - ratedCount);
  const pct = Math.min(100, (ratedCount / target) * 100);

  // Rated cards drop out so what remains is always what still needs a decision.
  const pending = data.seeds.filter((s) => !(s.db_id in rated));

  return (
    <div className="bg-white rounded-2xl border border-gray-100 shadow-sm p-5 mb-6">
      <div className="flex items-start justify-between gap-3 mb-1">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-xl bg-violet-100 flex items-center justify-center text-lg">
            🎯
          </div>
          <h2 className="text-base font-bold text-gray-900">Teach your feed</h2>
        </div>
        <button
          onClick={onDismiss}
          className="text-xs text-gray-400 hover:text-gray-600 px-2 py-1 rounded-lg hover:bg-gray-100 transition-colors"
        >
          Skip
        </button>
      </div>

      <p className="text-sm text-gray-500 mb-4 max-w-2xl leading-relaxed">
        {remaining > 0 ? (
          <>
            Rate <strong>{remaining} more</strong> {remaining === 1 ? 'paper' : 'papers'} to
            bootstrap your profile. Saves and dismissals both help &mdash; a dismissal tells
            the ranker what to avoid just as clearly as a save tells it what to find.
          </>
        ) : (
          <>Enough signal to personalize. Generate a feed to see it applied.</>
        )}
      </p>

      <div className="h-1.5 bg-gray-100 rounded-full overflow-hidden mb-4">
        <div
          className="h-full bg-violet-500 rounded-full transition-all duration-300"
          style={{ width: `${pct}%` }}
        />
      </div>

      {pending.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {pending.slice(0, 6).map((seed) => (
            <div
              key={seed.db_id}
              className="border border-gray-100 rounded-xl p-3.5 hover:border-violet-200 transition-colors flex flex-col"
            >
              <a
                href={seed.url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-sm font-semibold text-gray-900 hover:text-violet-600 leading-snug mb-1.5 transition-colors"
              >
                {seed.title}
              </a>
              <p className="text-xs text-gray-500 leading-relaxed mb-3 line-clamp-3 flex-1">
                {seed.abstract}
              </p>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => onRate(seed, 'saved')}
                  className="flex items-center gap-1.5 text-xs font-medium px-2.5 py-1.5 rounded-lg border border-emerald-200 text-emerald-700 hover:bg-emerald-50 transition-colors"
                >
                  <Save className="w-3.5 h-3.5" /> Interested
                </button>
                <button
                  onClick={() => onRate(seed, 'dismissed')}
                  className="flex items-center gap-1.5 text-xs font-medium px-2.5 py-1.5 rounded-lg border border-gray-200 text-gray-500 hover:bg-gray-50 transition-colors"
                >
                  <X className="w-3.5 h-3.5" /> Not for me
                </button>
                {seed.citation_count ? (
                  <span className="ml-auto text-xs text-gray-400">
                    ⭐ {seed.citation_count}
                  </span>
                ) : null}
              </div>
            </div>
          ))}
        </div>
      ) : (
        <p className="text-sm text-gray-400 py-2">
          All set — that&apos;s everything from this batch.
        </p>
      )}
    </div>
  );
}

/* ── What's Hot Section ── */
function WhatsHotSection({
  data,
  isLoading,
  isRefreshing,
  onRefresh,
}: {
  data: WhatsHotData | null;
  isLoading: boolean;
  isRefreshing: boolean;
  onRefresh: () => void;
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
                  {' '}Tools from the last{' '}
                  {data.github_window === 'daily' ? 'day' : 'week'}.
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
                  <HFPaperCard key={paper.id} paper={paper} />
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
                  <GithubRepoCard key={repo.full_name} repo={repo} />
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function HFPaperCard({ paper }: { paper: HFPaper }) {
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
    </div>
  );
}

function GithubRepoCard({ repo }: { repo: GithubRepo }) {
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
          <span className="flex items-center gap-1 text-xs text-yellow-600 font-medium">
            ★ {repo.stars >= 1000 ? `${(repo.stars / 1000).toFixed(1)}k` : repo.stars}
          </span>
        </div>
      </div>
      {repo.description && (
        <p className="text-sm text-gray-500 leading-relaxed">{repo.description}</p>
      )}
    </div>
  );
}
