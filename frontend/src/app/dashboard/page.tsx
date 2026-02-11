'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/lib/auth';
import { feedApi, interactionsApi, qaApi, authApi, Paper, Article, FeedResponse, InteractionStats } from '@/lib/api';
import toast from 'react-hot-toast';
import { BookOpen, FileText, Save, Eye, X, RefreshCw, ExternalLink, MessageCircle, Upload, Trash2, Search, Settings, ChevronDown, ChevronUp } from 'lucide-react';

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
  const [timeWindow, setTimeWindow] = useState(7);
  const [activeTab, setActiveTab] = useState<'feed' | 'saved' | 'qa' | 'settings'>('feed');

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
    }
  }, [isAuthenticated]);

  const loadStats = async () => {
    try {
      const data = await interactionsApi.getStats();
      setStats(data);
    } catch (error) {
      console.error('Failed to load stats:', error);
    }
  };

  const generateFeed = async () => {
    setIsGenerating(true);
    try {
      const data = await feedApi.generate({
        time_window_days: timeWindow,
        focus_areas: user?.focus_areas,
        use_ml: true,
      });
      setFeed(data);
      toast.success('Feed generated!');
      loadStats();
    } catch (error: any) {
      toast.error(error.response?.data?.detail || 'Failed to generate feed');
    } finally {
      setIsGenerating(false);
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
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex justify-between items-center">
          <h1 className="text-2xl font-bold text-gray-900">Learning Assistant</h1>
          <div className="flex items-center gap-4">
            <span className="text-sm text-gray-600">
              {user?.email}
            </span>
            <button
              onClick={logout}
              className="text-sm text-gray-500 hover:text-gray-700"
            >
              Sign out
            </button>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Stats Bar */}
        <div className="card mb-6">
          <div className="grid grid-cols-2 sm:grid-cols-5 gap-4 items-center">
            <div>
              <p className="text-sm text-gray-500">Total Interactions</p>
              <p className="text-2xl font-bold text-gray-900">{stats?.total || 0}</p>
            </div>
            <div>
              <p className="text-sm text-gray-500">Saved</p>
              <p className="text-2xl font-bold text-green-600">{stats?.saved || 0}</p>
            </div>
            <div>
              <p className="text-sm text-gray-500">Viewed</p>
              <p className="text-2xl font-bold text-blue-600">{stats?.viewed || 0}</p>
            </div>
            <div>
              <p className="text-sm text-gray-500">Dismissed</p>
              <p className="text-2xl font-bold text-red-500">{stats?.dismissed || 0}</p>
            </div>
            <div className="text-right">
              {stats?.ready_for_training ? (
                <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-green-100 text-green-800">
                  Model ready to train
                </span>
              ) : (
                <span className="text-sm text-gray-500">
                  {stats?.interactions_until_training || 50} more interactions until training
                </span>
              )}
            </div>
          </div>
        </div>

        {/* Tabs */}
        <div className="flex gap-4 mb-6">
          <button
            onClick={() => setActiveTab('feed')}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              activeTab === 'feed'
                ? 'bg-primary-600 text-white'
                : 'bg-white text-gray-700 hover:bg-gray-100'
            }`}
          >
            Daily Feed
          </button>
          <button
            onClick={() => setActiveTab('saved')}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              activeTab === 'saved'
                ? 'bg-primary-600 text-white'
                : 'bg-white text-gray-700 hover:bg-gray-100'
            }`}
          >
            Saved Items
          </button>
          <button
            onClick={() => setActiveTab('qa')}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              activeTab === 'qa'
                ? 'bg-primary-600 text-white'
                : 'bg-white text-gray-700 hover:bg-gray-100'
            }`}
          >
            Q&A Assistant
          </button>
          <button
            onClick={() => setActiveTab('settings')}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              activeTab === 'settings'
                ? 'bg-primary-600 text-white'
                : 'bg-white text-gray-700 hover:bg-gray-100'
            }`}
          >
            Settings
          </button>
        </div>

        {activeTab === 'feed' && (
          <>
            {/* Personalization + Feed Controls */}
            <FeedControls
              user={user}
              timeWindow={timeWindow}
              setTimeWindow={setTimeWindow}
              isGenerating={isGenerating}
              onGenerate={generateFeed}
              onProfileUpdate={fetchProfile}
            />

            {/* Feed Results */}
            {feed && (
              <div className="space-y-6">
                {/* Papers */}
                {feed.papers.length > 0 && (
                  <div>
                    <h2 className="text-xl font-semibold text-gray-900 mb-4 flex items-center gap-2">
                      <BookOpen className="w-5 h-5" />
                      Research Papers ({feed.papers.length})
                    </h2>
                    <div className="space-y-4">
                      {feed.papers.map((paper) => (
                        <PaperCard
                          key={paper.id}
                          paper={paper}
                          onInteraction={handleInteraction}
                        />
                      ))}
                    </div>
                  </div>
                )}

                {/* Articles */}
                {feed.articles.length > 0 && (
                  <div>
                    <h2 className="text-xl font-semibold text-gray-900 mb-4 flex items-center gap-2">
                      <FileText className="w-5 h-5" />
                      Tech Articles ({feed.articles.length})
                    </h2>
                    <div className="space-y-4">
                      {feed.articles.map((article) => (
                        <ArticleCard
                          key={article.id}
                          article={article}
                          onInteraction={handleInteraction}
                        />
                      ))}
                    </div>
                  </div>
                )}

                {/* Meta info */}
                <div className="text-sm text-gray-500 text-center">
                  {feed.used_ml_ranking ? (
                    <span>Using personalized ML ranking</span>
                  ) : (
                    <span>Using heuristic ranking (interact more to enable ML)</span>
                  )}
                  {' | '}
                  Considered {feed.total_papers_considered} papers, {feed.total_articles_considered} articles
                </div>
              </div>
            )}

            {!feed && !isGenerating && (
              <div className="text-center py-12 text-gray-500">
                Click &quot;Generate Feed&quot; to get personalized recommendations
              </div>
            )}
          </>
        )}

        {activeTab === 'saved' && <SavedItems />}

        {activeTab === 'qa' && <QAAssistant />}

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
  isGenerating,
  onGenerate,
  onProfileUpdate,
}: {
  user: any;
  timeWindow: number;
  setTimeWindow: (v: number) => void;
  isGenerating: boolean;
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
      {/* Top row: time window + generate */}
      <div className="flex flex-wrap items-center gap-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
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

        {/* Current focus areas preview */}
        <div className="flex-1 min-w-0">
          <label className="block text-sm font-medium text-gray-700 mb-1">Focus areas</label>
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
}: {
  paper: Paper;
  onInteraction: (type: 'paper' | 'article', id: number, interaction: 'viewed' | 'saved' | 'dismissed') => void;
}) {
  return (
    <div className="card">
      <div className="flex justify-between items-start gap-4">
        <div className="flex-1">
          <h3 className="font-semibold text-gray-900 mb-1">
            {paper.rank}. {paper.title}
          </h3>
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-gray-500 mb-3">
            <span>arXiv:{paper.arxiv_id}</span>
            <span>Score: {paper.relevance_score.toFixed(3)}</span>
            {paper.impact_score != null && (
              <span>Impact: {paper.impact_score.toFixed(2)}</span>
            )}
            {paper.citation_count > 0 && (
              <span>{paper.citation_count} citations</span>
            )}
          </div>
          {paper.summary && <FormattedSummary text={paper.summary} />}
        </div>
      </div>
      <div className="flex items-center gap-2 mt-4 pt-4 border-t">
        <a
          href={paper.arxiv_url}
          target="_blank"
          rel="noopener noreferrer"
          onClick={() => onInteraction('paper', paper.id, 'viewed')}
          className="btn-secondary text-sm flex items-center gap-1"
        >
          <ExternalLink className="w-4 h-4" />
          View
        </a>
        <button
          onClick={() => onInteraction('paper', paper.id, 'saved')}
          className="btn-secondary text-sm flex items-center gap-1"
        >
          <Save className="w-4 h-4" />
          Save
        </button>
        <button
          onClick={() => onInteraction('paper', paper.id, 'dismissed')}
          className="text-gray-400 hover:text-gray-600 p-2"
        >
          <X className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
}

/* ── Article Card ── */
function ArticleCard({
  article,
  onInteraction,
}: {
  article: Article;
  onInteraction: (type: 'paper' | 'article', id: number, interaction: 'viewed' | 'saved' | 'dismissed') => void;
}) {
  return (
    <div className="card">
      <div className="flex justify-between items-start gap-4">
        <div className="flex-1">
          <h3 className="font-semibold text-gray-900 mb-1">
            {article.rank}. {article.title}
          </h3>
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-gray-500 mb-3">
            <span>{article.source}</span>
            <span>{article.upvotes} upvotes</span>
            <span>Score: {article.relevance_score.toFixed(3)}</span>
          </div>
          {article.summary && <FormattedSummary text={article.summary} />}
        </div>
      </div>
      <div className="flex items-center gap-2 mt-4 pt-4 border-t">
        <a
          href={article.url}
          target="_blank"
          rel="noopener noreferrer"
          onClick={() => onInteraction('article', article.id, 'viewed')}
          className="btn-secondary text-sm flex items-center gap-1"
        >
          <ExternalLink className="w-4 h-4" />
          Read
        </a>
        <button
          onClick={() => onInteraction('article', article.id, 'saved')}
          className="btn-secondary text-sm flex items-center gap-1"
        >
          <Save className="w-4 h-4" />
          Save
        </button>
        <button
          onClick={() => onInteraction('article', article.id, 'dismissed')}
          className="text-gray-400 hover:text-gray-600 p-2"
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

/* ── Q&A Assistant ── */
function QAAssistant() {
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState<{ answer: string; citations: any[] } | null>(null);
  const [isAsking, setIsAsking] = useState(false);
  const [nContext, setNContext] = useState(5);
  const [uploadsOnly, setUploadsOnly] = useState(false);
  const [documents, setDocuments] = useState<any[]>([]);
  const [isUploading, setIsUploading] = useState(false);

  useEffect(() => {
    loadDocuments();
  }, []);

  const loadDocuments = async () => {
    try {
      const data = await qaApi.listDocuments();
      setDocuments(data.documents || data || []);
    } catch (error) {
      console.error('Failed to load documents:', error);
    }
  };

  const handleAsk = async () => {
    if (!question.trim()) return;
    setIsAsking(true);
    setAnswer(null);
    try {
      const data = await qaApi.ask({
        question,
        n_context: nContext,
        filter_type: uploadsOnly ? 'user_doc' : undefined,
      });
      setAnswer(data);
    } catch (error: any) {
      toast.error(error.response?.data?.detail || 'Failed to get answer');
    } finally {
      setIsAsking(false);
    }
  };

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;
    setIsUploading(true);
    let uploaded = 0;
    for (let i = 0; i < files.length; i++) {
      try {
        await qaApi.uploadDocument(files[i], files[i].name);
        uploaded++;
      } catch (error: any) {
        toast.error(`Failed to upload ${files[i].name}: ${error.response?.data?.detail || 'Unknown error'}`);
      }
    }
    if (uploaded > 0) {
      toast.success(`Uploaded ${uploaded} document(s)`);
      loadDocuments();
    }
    setIsUploading(false);
    e.target.value = '';
  };

  const handleDelete = async (docId: number, title: string) => {
    try {
      await qaApi.deleteDocument(docId);
      toast.success(`Deleted "${title}"`);
      loadDocuments();
    } catch (error) {
      toast.error('Failed to delete document');
    }
  };

  return (
    <div className="space-y-6">
      {/* Document Upload */}
      <div className="card">
        <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
          <Upload className="w-5 h-5" />
          Upload Study Materials
        </h2>
        <p className="text-sm text-gray-500 mb-4">
          Upload .txt, .md, or .pdf files to build your knowledge base
        </p>
        <label className="inline-flex items-center gap-2 btn-secondary text-sm cursor-pointer">
          <Upload className="w-4 h-4" />
          {isUploading ? 'Uploading...' : 'Choose Files'}
          <input
            type="file"
            accept=".txt,.md,.pdf"
            multiple
            onChange={handleUpload}
            disabled={isUploading}
            className="hidden"
          />
        </label>

        {/* Document List */}
        {documents.length > 0 && (
          <div className="mt-4">
            <h3 className="text-sm font-medium text-gray-700 mb-2">
              Your Documents ({documents.length})
            </h3>
            <div className="space-y-2">
              {documents.map((doc: any) => (
                <div
                  key={doc.id}
                  className="flex items-center justify-between bg-gray-50 rounded-lg px-3 py-2"
                >
                  <div>
                    <span className="text-sm font-medium text-gray-900">{doc.title}</span>
                    <span className="text-xs text-gray-500 ml-2">
                      {doc.chunk_count ? `${doc.chunk_count} chunks` : ''}
                    </span>
                  </div>
                  <button
                    onClick={() => handleDelete(doc.id, doc.title)}
                    className="text-red-400 hover:text-red-600 p-1"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Question Input */}
      <div className="card">
        <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
          <MessageCircle className="w-5 h-5" />
          Ask a Question
        </h2>
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="e.g., Explain the attention mechanism in transformers"
          rows={3}
          className="input-field w-full mb-4"
        />
        <div className="flex flex-wrap items-center gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Context documents
            </label>
            <select
              value={nContext}
              onChange={(e) => setNContext(Number(e.target.value))}
              className="input-field w-24"
            >
              {[3, 5, 7, 10].map((n) => (
                <option key={n} value={n}>{n}</option>
              ))}
            </select>
          </div>
          <label className="flex items-center gap-2 text-sm text-gray-700 cursor-pointer mt-5">
            <input
              type="checkbox"
              checked={uploadsOnly}
              onChange={(e) => setUploadsOnly(e.target.checked)}
              className="rounded border-gray-300"
            />
            Use only my uploaded documents
          </label>
          <div className="flex-1" />
          <button
            onClick={handleAsk}
            disabled={isAsking || !question.trim()}
            className="btn-primary flex items-center gap-2 disabled:opacity-50 mt-5"
          >
            {isAsking ? (
              <>
                <RefreshCw className="w-4 h-4 animate-spin" />
                Thinking...
              </>
            ) : (
              <>
                <Search className="w-4 h-4" />
                Ask Question
              </>
            )}
          </button>
        </div>
      </div>

      {/* Answer */}
      {answer && (
        <div className="card">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Answer</h2>
          <div className="prose prose-sm max-w-none text-gray-700 whitespace-pre-wrap">
            {answer.answer}
          </div>

          {/* Citations */}
          {answer.citations && answer.citations.length > 0 && (
            <div className="mt-6 pt-4 border-t">
              <h3 className="text-sm font-semibold text-gray-900 mb-3">Sources</h3>
              <div className="space-y-2">
                {answer.citations.map((citation: any, i: number) => (
                  <div key={i} className="bg-gray-50 rounded-lg px-3 py-2">
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-medium bg-gray-200 rounded px-1.5 py-0.5">
                        {citation.type === 'paper' ? 'Paper' :
                         citation.type === 'article' ? 'Article' : 'Document'}
                      </span>
                      <span className="text-sm font-medium text-gray-900">
                        {citation.title}
                      </span>
                    </div>
                    {citation.url && (
                      <a
                        href={citation.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-xs text-primary-600 hover:text-primary-700 mt-1 inline-block"
                      >
                        Open source &rarr;
                      </a>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
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
