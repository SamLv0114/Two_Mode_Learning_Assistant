'use client';

import { useEffect, useRef, useState } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/lib/auth';
import dynamic from 'next/dynamic';
import type { PracticeContext } from '@/components/PracticeWorkspace';
import { errorMessage, feedApi, interactionsApi, whatsHotApi, practiceApi, profileApi, FeedResponse, InteractionStats, ModelStatus, FeedJobStatus, WhatsHotData, ColdStartSeed, ColdStartResponse, ResearchProfile } from '@/lib/api';
import toast from 'react-hot-toast';
import { formatDistanceToNow } from 'date-fns';
import { BookOpen, RefreshCw, Settings, Bot, Sparkles, Flame, Code2 } from 'lucide-react';
import { ChatContext, arxivIdFromHotPaper } from '@/components/dashboard/shared';
import { FeedControls } from '@/components/dashboard/FeedControls';
import { PaperCard } from '@/components/dashboard/PaperCards';
import { UserSettings } from '@/components/dashboard/UserSettings';
import { AgentChat } from '@/components/dashboard/AgentChat';
import { SavedItems } from '@/components/dashboard/SavedItems';
import { ColdStartSection } from '@/components/dashboard/ColdStartSection';
import { WhatsHotSection } from '@/components/dashboard/WhatsHot';
import { ProfileCard, ProfileOnboarding } from '@/components/dashboard/ResearchProfile';
import { ReadingPathPanel } from '@/components/dashboard/ReadingPath';
// Loaded on first visit to the Coding Agent tab: it brings Markdown, KaTeX and Mermaid with it.
const PracticeWorkspace = dynamic(() => import('@/components/PracticeWorkspace'), {
  ssr: false, loading: () => <p className="text-sm text-gray-500">Loading Coding Agent…</p>,
});

export default function DashboardPage() {
  const { user, isAuthenticated, isLoading, fetchProfile, logout } = useAuth();
  const router = useRouter();
  const [authChecked, setAuthChecked] = useState(false);

  const [feed, setFeed] = useState<FeedResponse | null>(null);
  const [stats, setStats] = useState<InteractionStats | null>(null);
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);
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
  const [activeTab, setActiveTab] = useState<'feed' | 'hot' | 'saved' | 'chat' | 'practice' | 'settings'>('feed');
  const [practiceAlerts, setPracticeAlerts] = useState(0);
  const [feedEditorSignal, setFeedEditorSignal] = useState(0);
  // The research profile drives retrieval and the model's judgements. undefined while loading.
  const [profile, setProfile] = useState<ResearchProfile | null | undefined>(undefined);
  const [profileEditKey, setProfileEditKey] = useState(0);
  const profileReady = !!profile?.set_up;
  const needsOnboarding = profile !== undefined && !profileReady;
  const activeTabRef = useRef(activeTab);
  activeTabRef.current = activeTab;
  const practiceSeen = useRef<Record<string, string>>({});

  // Coding Agent runs take minutes: keep an eye on them from any tab, often only while one is working.
  useEffect(() => {
    let stopped = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const check = async () => {
      try {
        const runs = await practiceApi.list();
        if (stopped) return;
        for (const item of runs) {
          const before = practiceSeen.current[item.run_id];
          if (before && ['queued', 'running'].includes(before) && before !== item.status && activeTabRef.current !== 'practice') {
            const name = item.title || item.source;
            if (item.status === 'waiting') toast(`Coding Agent needs your review: ${name}`, { icon: '✋' });
            else if (item.status === 'completed') toast.success(`Coding Agent finished: ${name}`);
          }
          practiceSeen.current[item.run_id] = item.status;
        }
        setPracticeAlerts(runs.filter((item) => item.status === 'waiting').length);
        timer = setTimeout(check, runs.some((item) => ['queued', 'running'].includes(item.status)) ? 10000 : 60000);
      } catch {
        if (!stopped) timer = setTimeout(check, 60000);
      }
    };
    check();
    return () => { stopped = true; if (timer) clearTimeout(timer); };
  }, []);
  const [practiceContext, setPracticeContext] = useState<PracticeContext | null>(null);
  const [chatContext, setChatContext] = useState<ChatContext | null>(null);
  const openPractice = (context: PracticeContext) => { setPracticeContext(context); setActiveTab('practice'); };
  const openChat = (context: ChatContext) => {
    // A new source starts its own conversation so earlier paper/repo context cannot leak into it.
    localStorage.removeItem('chat_session_id');
    setChatContext(context);
    setActiveTab('chat');
  };
  // Only papers go to Agent Chat; a repository's questions are answered by the repository agent.
  const askFromPractice = (context: PracticeContext) => {
    if (context.kind === 'paper') openChat({ kind: 'paper', title: context.title || context.source, abstract: context.description || null, arxivId: context.source, arxivUrl: context.url || `https://arxiv.org/abs/${context.source}`, origin: 'Coding Agent' });
  };
  const feedPollRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const feedPollRunRef = useRef(0);
  const feedGenerationActiveRef = useRef(false);

  useEffect(() => {
    let mounted = true;
    fetchProfile().finally(() => {
      if (mounted) setAuthChecked(true);
    });
    return () => { mounted = false; };
  }, [fetchProfile]);

  // Set when a demo visitor leaves to register, so signing out does not bounce them to /login.
  const leavingForRegister = useRef(false);
  useEffect(() => {
    if (authChecked && !isLoading && !isAuthenticated && !leavingForRegister.current) {
      router.replace('/login');
    }
  }, [authChecked, isAuthenticated, isLoading, router]);
  const registerFromDemo = () => {
    leavingForRegister.current = true;
    logout();
    router.push('/register');
  };

  useEffect(() => {
    if (authChecked && isAuthenticated) {
      loadStats();
      loadColdStart();
      loadCurrentFeed();
    }
  }, [authChecked, isAuthenticated]);

  // Open on the feed the user already has. Someone who has never had one gets their first
  // generated straight away once there is a profile to build it from; until then the
  // onboarding card is the first step (it can also skip to a general feed).
  const loadCurrentFeed = async () => {
    try {
      const [current, saved] = await Promise.all([feedApi.getCurrent(), profileApi.get().catch(() => null)]);
      setProfile(saved);
      if (!current.generated_at) {
        if (saved?.set_up || user?.is_demo) generateFeed(false);
        return;
      }
      if (feedGenerationActiveRef.current) return;
      if (current.feed_mode) setFeedMode(current.feed_mode);
      if (current.time_window_days) setTimeWindow(current.time_window_days);
      setFeed({
        papers: current.papers,
        generated_at: current.generated_at,
        time_window_days: current.time_window_days || 7,
        focus_areas: user?.focus_areas || [],
        used_ml_ranking: current.ranking_source === 'ml',
        ranking_source: current.ranking_source,
        ranking_fallback_reason: current.ranking_fallback_reason,
        selection: current.selection,
        total_papers_considered: null,
      });
    } catch (error) {
      console.error('Failed to load the current feed:', error);
    }
  };

  // What's Hot has its own tab now, so fetch it when that tab is first opened
  // rather than on every dashboard load. Kept in state after the first visit.
  useEffect(() => {
    if (authChecked && isAuthenticated && activeTab === 'hot' && !whatsHot && !isLoadingHot) {
      loadWhatsHot();
    }
  }, [authChecked, isAuthenticated, activeTab]);

  useEffect(() => {
    return () => {
      feedPollRunRef.current += 1;
      feedGenerationActiveRef.current = false;
      if (feedPollRef.current) clearTimeout(feedPollRef.current);
    };
  }, []);

  const loadStats = async () => {
    try {
      const [statsResult, modelResult] = await Promise.allSettled([
        interactionsApi.getStats(), interactionsApi.getModelStatus(),
      ]);
      if (statsResult.status === 'fulfilled') setStats(statsResult.value);
      if (modelResult.status === 'fulfilled') setModelStatus(modelResult.value);
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
    if (feedGenerationActiveRef.current) return;
    feedGenerationActiveRef.current = true;
    if (feedPollRef.current) clearTimeout(feedPollRef.current);
    const run = ++feedPollRunRef.current;
    setIsGenerating(true);
    setFeedProgress(forceRefresh ? 'Drawing a new batch...' : 'Loading your feed...');
    setFeed(null);
    setRefineResult(null);
    const deadline = Date.now() + 5 * 60_000;
    try {
      const { job_id } = await feedApi.generate({
        time_window_days: timeWindow,
        focus_areas: user?.focus_areas,
        use_ml: true,
        mode: feedMode,
        force_refresh: forceRefresh,
      });

      let failures = 0;
      const poll = async () => {
        if (run !== feedPollRunRef.current) return;
        if (Date.now() >= deadline) {
          toast.error('Feed generation is taking too long. Please try again.');
          setIsGenerating(false);
          feedGenerationActiveRef.current = false;
          return;
        }
        try {
          const status: FeedJobStatus = await feedApi.getJobStatus(job_id);
          if (run !== feedPollRunRef.current) return;
          failures = 0;

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
            feedPollRef.current = null;
            // No count — the server decides how large a feed is, so this does
            // not need updating if that size changes.
            // getCurrent rather than getPapers: a reused feed keeps its real generation time.
            const current = await feedApi.getCurrent();
            const papers = current.papers;
            if (run !== feedPollRunRef.current) return;
            setFeed({
              papers,
              generated_at: current.generated_at || new Date().toISOString(),
              time_window_days: timeWindow,
              focus_areas: user?.focus_areas || [],
              used_ml_ranking: status.used_ml_ranking ?? false,
              ranking_source: status.ranking_source,
              ranking_fallback_reason: status.ranking_fallback_reason,
              selection: current.selection ?? status.selection,
              total_papers_considered: status.papers_count ?? papers.length,
            });
            toast.success(status.reused ? "Showing today's feed" : 'Feed ready!');
            loadStats();
            setIsGenerating(false);
            feedGenerationActiveRef.current = false;
          } else if (status.status === 'error' || status.status === 'not_found') {
            feedPollRef.current = null;
            toast.error(status.message || 'Feed generation failed');
            setIsGenerating(false);
            feedGenerationActiveRef.current = false;
          } else {
            feedPollRef.current = setTimeout(poll, 2500);
          }
        } catch (error) {
          if (run !== feedPollRunRef.current) return;
          failures += 1;
          if (failures >= 3 || Date.now() >= deadline) {
            feedPollRef.current = null;
            toast.error('Could not check feed progress. Please try again.');
            setIsGenerating(false);
            feedGenerationActiveRef.current = false;
          } else {
            feedPollRef.current = setTimeout(poll, 2500);
          }
        }
      };
      feedPollRef.current = setTimeout(poll, 2500);
    } catch (error) {
      toast.error(errorMessage(error, 'Failed to start feed generation'));
      setIsGenerating(false);
      feedGenerationActiveRef.current = false;
    }
  };

  // V4: Evaluator-Optimizer pass — re-scores the current feed and swaps out
  // weak papers without re-running the whole pipeline.
  const handleRefine = async () => {
    setIsRefining(true);
    try {
      const result = await feedApi.refine();
      setFeed((prev) => {
        if (!prev) return prev;
        const changed = result.papers.some((paper, i) => paper.arxiv_id !== prev.papers[i]?.arxiv_id);
        return changed
          ? { ...prev, papers: result.papers, used_ml_ranking: false,
              ranking_source: 'refined', ranking_fallback_reason: null }
          : { ...prev, papers: result.papers };
      });
      setRefineResult({ score: result.score, rounds: result.rounds, passed: result.passed });
      toast.success(result.message || 'Feed refined!');
    } catch (error) {
      toast.error(errorMessage(error, 'Refinement failed'));
    } finally {
      setIsRefining(false);
    }
  };

  // A saved or edited profile is a request for a feed built from it; the server rebuilds
  // today's feed when the profile version differs from the one it was built with.
  const handleProfileSaved = (saved: ResearchProfile, regenerate = true) => {
    setProfile(saved);
    fetchProfile();
    if (regenerate) generateFeed(false);
  };

  const handlePaperFeedback = async (text: string, paperId: number) => {
    try {
      const result = await profileApi.feedback(text, paperId);
      setProfile(result.profile);
      toast.success(`${result.note} Generate Feed to see the change.`);
      return true;
    } catch (error) {
      toast.error(errorMessage(error, 'Could not apply that feedback'));
      return false;
    }
  };

  const handleInteraction = async (itemId: number, interactionType: 'viewed' | 'saved' | 'dismissed') => {
    try {
      await interactionsApi.create({
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

  if (!authChecked || isLoading || !isAuthenticated) {
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

      {user?.is_demo && (
        <div className="bg-violet-600 text-white">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-2.5 flex flex-wrap items-center justify-between gap-2 text-sm">
            <p>
              <span className="font-semibold">You are exploring the shared demo account.</span>{' '}
              <span className="text-violet-100">AI actions are limited each day, and saves are shared with other visitors.</span>
            </p>
            <button type="button" onClick={registerFromDemo}
              className="rounded-lg bg-white px-3 py-1 text-xs font-semibold text-violet-700 hover:bg-violet-50">
              Create your own account
            </button>
          </div>
        </div>
      )}

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Stats Bar — feedback counts only mean something next to the feed they train */}
        {activeTab === 'feed' && <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 mb-6">
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
            {modelStatus?.is_trained ? (
              <span className="inline-flex items-center gap-1.5 text-sm font-semibold text-emerald-700">
                <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                Model trained on {modelStatus.training_sample_count} labeled papers{feed?.ranking_source === 'refined' ? ' · this feed was refined' : feed && !feed.used_ml_ranking ? ' · this feed used heuristic ranking' : ''}
              </span>
            ) : stats?.ready_for_training ? (
              <span className="text-xs text-amber-700">Enough paper feedback to attempt training; {feed ? 'choose New batch.' : 'generate a feed.'}</span>
            ) : (
              <span className="text-xs text-gray-400 leading-relaxed">
                <span className="font-semibold text-gray-600">{stats?.interactions_until_training ?? 50}</span> more interactions to enable ML ranking
              </span>
            )}
          </div>
        </div>}

        {/* Tabs */}
        <div className="flex border-b border-gray-200 mb-6 -mx-0 overflow-x-auto scrollbar-none">
          {(
            [
              { id: 'feed', label: 'Daily Feed' },
              { id: 'hot', label: "What's Hot", Icon: Flame },
              { id: 'saved', label: 'Saved' },
              { id: 'chat', label: 'Agent Chat', Icon: Bot },
              { id: 'practice', label: 'Coding Agent', Icon: Code2 },
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
              {id === 'practice' && practiceAlerts > 0 && (
                <span title="Runs waiting for your review" className="ml-1 rounded-full bg-violet-600 px-1.5 text-[10px] font-semibold text-white">{practiceAlerts}</span>
              )}
              {activeTab === id && (
                <span className="absolute bottom-0 left-0 right-0 h-0.5 bg-primary-600 rounded-t-full" />
              )}
            </button>
          ))}
        </div>

        {activeTab === 'feed' && (
          <>
            {/* The research profile: onboarding first, then an editable card */}
            {needsOnboarding && (
              <ProfileOnboarding isDemo={user?.is_demo} onSaved={(saved) => handleProfileSaved(saved)}
                onSkip={feed || isGenerating ? undefined : () => generateFeed(false)} />
            )}
            {profile && profileReady && (
              <ProfileCard key={profileEditKey} profile={profile} isDemo={user?.is_demo} startEditing={profileEditKey > 0}
                onSaved={handleProfileSaved} />
            )}

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
              openEditorSignal={feedEditorSignal}
              hasResearchProfile={profileReady}
            />

            {/* Rating seed papers: the older onboarding, kept for users without a research profile */}
            {!profileReady && coldStart && coldStart.seeds.length > 0 && (
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
                          onAskAgent={(ctx) => openChat({ kind: 'paper', ...ctx, origin: 'Daily Feed' })}
                          onPractice={() => openPractice({ kind: 'paper', source: paper.arxiv_id, title: paper.title, description: paper.abstract, url: paper.arxiv_url, origin: 'Daily Feed' })}
                          onFeedback={profileReady && !user?.is_demo ? handlePaperFeedback : undefined}
                        />
                      ))}
                    </div>
                  </div>
                )}

                {/* Meta info */}
                <div className="flex items-center justify-center gap-2 pt-2">
                  <span className={`inline-flex items-center gap-1.5 text-xs font-medium px-3 py-1.5 rounded-full ${feed.used_ml_ranking ? 'bg-emerald-50 text-emerald-700' : 'bg-gray-100 text-gray-500'}`}>
                    {feed.ranking_source === 'refined' ? '✨ Refined feed' : feed.used_ml_ranking ? '🧠 Personalized ML ranking' : '📊 Heuristic ranking'}
                  </span>
                  {feed.selection === 'llm_judge' && feed.ranking_source !== 'refined' && (
                    <span className="inline-flex items-center gap-1.5 text-xs font-medium px-3 py-1.5 rounded-full bg-indigo-50 text-indigo-700"
                      title="Each shortlisted paper was judged against your research profile">
                      🧭 Judged against your profile
                    </span>
                  )}
                  {feed.ranking_fallback_reason && <span className="text-xs text-amber-700" title={feed.ranking_fallback_reason}>ML fallback: {feed.ranking_fallback_reason}</span>}
                  {feed.total_papers_considered !== null && (
                    <span className="text-xs text-gray-400">{feed.total_papers_considered} papers considered</span>
                  )}
                  <span className="text-xs text-gray-400">Generated {formatDistanceToNow(new Date(feed.generated_at), { addSuffix: true })}</span>
                </div>
              </div>
            )}

            {!feed && !isGenerating && !needsOnboarding && (
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

        {activeTab === 'feed' && !isGenerating && (
          <ReadingPathPanel onSave={(id) => handleInteraction(id, 'saved')} />
        )}

        {activeTab === 'hot' && (
          <WhatsHotSection
            data={whatsHot}
            isLoading={isLoadingHot}
            isRefreshing={isRefreshingHot}
            onRefresh={refreshWhatsHot}
            onAskPaper={(paper) => openChat({ kind: 'paper', title: paper.title, abstract: paper.abstract || null, arxivId: arxivIdFromHotPaper(paper) || '', arxivUrl: paper.arxiv_url || paper.url, origin: "What's Hot" })}
            onPracticePaper={(paper) => { const id = arxivIdFromHotPaper(paper); if (id) openPractice({ kind: 'paper', source: id, title: paper.title, description: paper.abstract, url: paper.arxiv_url, origin: "What's Hot" }); }}
            onPracticeRepo={(repo) => openPractice({ kind: 'repo', source: repo.url, title: repo.full_name, description: repo.description, url: repo.url, origin: "What's Hot" })}
          />
        )}

        {activeTab === 'saved' && (
          <SavedItems
            onGoToFeed={() => setActiveTab('feed')}
            onAskAgent={(paper) => openChat({ kind: 'paper', title: paper.title, abstract: paper.abstract || null, arxivId: paper.arxiv_id,
              arxivUrl: paper.arxiv_url || `https://arxiv.org/abs/${paper.arxiv_id}`, origin: 'Saved' })}
            onPractice={(paper) => openPractice({ kind: 'paper', source: paper.arxiv_id, title: paper.title, description: paper.abstract,
              url: paper.arxiv_url, origin: 'Saved' })}
          />
        )}

        {activeTab === 'chat' && (
          <AgentChat
            chatContext={chatContext}
            onClearContext={() => setChatContext(null)}
            onPractice={(context) => openPractice({ kind: 'paper', source: context.arxivId, title: context.title,
              description: context.abstract || undefined, url: context.arxivUrl, origin: context.origin })}
          />
        )}

        {activeTab === 'practice' && <PracticeWorkspace context={practiceContext} onAskAgent={askFromPractice} />}

        {activeTab === 'settings' && (
          <UserSettings user={user} onUpdate={fetchProfile}
            onEditInterests={() => {
              setActiveTab('feed');
              if (profileReady) setProfileEditKey((n) => n + 1);
              else setFeedEditorSignal((n) => n + 1);
            }} />
        )}
      </main>
    </div>
  );
}
