'use client';

import { useEffect, useState } from 'react';
import { authApi, errorMessage } from '@/lib/api';
import toast from 'react-hot-toast';
import { Save, RefreshCw, Settings, ChevronDown, ChevronUp } from 'lucide-react';
import { AVAILABLE_AREAS, EXAMPLE_INTERESTS } from './shared';

export function FeedControls({
  user,
  timeWindow,
  setTimeWindow,
  feedMode,
  setFeedMode,
  isGenerating,
  feedProgress,
  onGenerate,
  onProfileUpdate,
  openEditorSignal = 0,
  hasResearchProfile = false,
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
  /** Bumped by "Edit on Daily Feed" in Settings to open the editor. */
  openEditorSignal?: number;
  /** The research profile card replaces the focus-area editor as the place to personalize. */
  hasResearchProfile?: boolean;
}) {
  const [expanded, setExpanded] = useState(openEditorSignal > 0);
  useEffect(() => { if (openEditorSignal > 0) setExpanded(true); }, [openEditorSignal]);
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
    } catch (error) {
      toast.error(errorMessage(error, 'Failed to save preferences'));
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

        {/* Current focus areas preview (only without a research profile, which supersedes it) */}
        {hasResearchProfile ? <div className="flex-1 min-w-0" /> : <div className="flex-1 min-w-0">
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
        </div>}

        <div className="flex items-end gap-2">
          {!hasResearchProfile && <button
            onClick={() => setExpanded(!expanded)}
            className="btn-secondary text-sm flex items-center gap-1"
          >
            <Settings className="w-4 h-4" />
            {expanded ? 'Hide' : 'Edit'}
            {expanded ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
          </button>}
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
      {expanded && !hasResearchProfile && (
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
          {user?.is_demo && <p className="text-xs text-gray-500">The shared demo profile is fixed; your own account can change it.</p>}
          <button
            onClick={handleSavePreferences}
            disabled={isSaving || user?.is_demo}
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
