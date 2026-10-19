'use client';

import { useState } from 'react';
import toast from 'react-hot-toast';
import { Check, ChevronDown, ChevronUp, Pencil, Pin, Search, Send, Sparkles, X } from 'lucide-react';
import { errorMessage, PAPER_TYPES, PaperOption, PaperType, ProfileFields, profileApi, ResearchProfile, SeedPaper } from '@/lib/api';

const EMPTY: ProfileFields = { goal: '', level: '', known: [], interests: [], preferences: [], avoid_types: [], seed_papers: [], summary: '' };
const LEVELS: ProfileFields['level'][] = ['beginner', 'intermediate', 'advanced'];

function fieldsOf(profile: ResearchProfile | null): ProfileFields {
  if (!profile) return EMPTY;
  const { goal, level, known, interests, preferences, avoid_types, seed_papers, summary } = profile;
  return { goal, level, known, interests, preferences, avoid_types, seed_papers, summary };
}

/** A list of short phrases: type and press Enter to add, × to remove. */
function ChipInput({ label, hint, values, onChange, placeholder }: {
  label: string; hint?: string; values: string[]; onChange: (v: string[]) => void; placeholder: string;
}) {
  const [draft, setDraft] = useState('');
  const add = () => {
    const value = draft.trim();
    if (value && !values.some((v) => v.toLowerCase() === value.toLowerCase())) onChange([...values, value]);
    setDraft('');
  };
  return (
    <div>
      <label className="section-label mb-1.5">{label}</label>
      {hint && <p className="text-xs text-gray-500 mb-2">{hint}</p>}
      <div className="flex flex-wrap gap-1.5 mb-2">
        {values.map((v) => (
          <span key={v} className="inline-flex items-center gap-1 rounded-full bg-primary-50 text-primary-700 px-2.5 py-0.5 text-xs font-medium">
            {v}
            <button type="button" aria-label={`Remove ${v}`} onClick={() => onChange(values.filter((x) => x !== v))}
              className="text-primary-400 hover:text-primary-700"><X className="w-3 h-3" /></button>
          </span>
        ))}
      </div>
      <input value={draft} onChange={(e) => setDraft(e.target.value)} placeholder={placeholder}
        onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); add(); } }} onBlur={add}
        className="input-field text-sm py-2" />
    </div>
  );
}

/** Pin papers the user already found useful; the citation-graph channel recommends from them. */
function SeedPaperPicker({ value, onChange, suggestions = [] }: {
  value: SeedPaper[]; onChange: (v: SeedPaper[]) => void; suggestions?: PaperOption[];
}) {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<PaperOption[]>([]);
  const [searching, setSearching] = useState(false);
  const pinned = new Set(value.map((p) => p.id));
  const search = async () => {
    if (query.trim().length < 3) return;
    setSearching(true);
    try {
      setResults(await profileApi.searchPapers(query.trim()));
    } catch (error) {
      toast.error(errorMessage(error, 'Search failed'));
    } finally {
      setSearching(false);
    }
  };
  const pin = (p: PaperOption) => {
    if (!pinned.has(p.id) && value.length < 10) onChange([...value, { id: p.id, title: p.title, year: p.year ?? null }]);
  };
  const options = [...suggestions.filter((s) => !results.some((r) => r.id === s.id)), ...results];
  return (
    <div>
      <label className="section-label mb-1.5">Papers you found useful</label>
      <p className="text-xs text-gray-500 mb-2">
        Recommendations follow the citation graph from these, so pin the papers closest to what you are working on.
      </p>
      {value.length > 0 && (
        <ul className="space-y-1 mb-2">
          {value.map((p) => (
            <li key={p.id} className="flex items-center gap-2 text-sm text-gray-800">
              <Pin className="w-3.5 h-3.5 text-primary-500 shrink-0" />
              <span className="truncate">{p.title}{p.year ? ` (${p.year})` : ''}</span>
              <button type="button" aria-label="Unpin" onClick={() => onChange(value.filter((x) => x.id !== p.id))}
                className="ml-auto text-gray-300 hover:text-rose-500"><X className="w-3.5 h-3.5" /></button>
            </li>
          ))}
        </ul>
      )}
      <div className="flex gap-2">
        <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search by title or topic"
          onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); search(); } }} className="input-field text-sm py-2" />
        <button type="button" onClick={search} disabled={searching} className="btn-secondary btn-sm">
          <Search className={`w-3.5 h-3.5 ${searching ? 'animate-pulse' : ''}`} /> Search
        </button>
      </div>
      {options.length > 0 && (
        <ul className="mt-2 divide-y divide-gray-100 rounded-xl border border-gray-100">
          {options.slice(0, 10).map((p) => (
            <li key={p.id} className="flex items-center gap-2 px-3 py-2 text-sm">
              <a href={p.url} target="_blank" rel="noopener noreferrer" className="truncate text-gray-800 hover:text-primary-600">
                {p.title}{p.year ? ` (${p.year})` : ''}
              </a>
              <span className="shrink-0 text-[10px] uppercase tracking-wide text-gray-400">
                {p.source === 'library' ? 'library' : 'Semantic Scholar'}
              </span>
              <button type="button" onClick={() => pin(p)} disabled={pinned.has(p.id)}
                className="ml-auto btn-link text-xs shrink-0">{pinned.has(p.id) ? 'Pinned' : 'Pin'}</button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export function ProfileEditor({ initial, suggestions, onSave, onCancel, saving }: {
  initial: ProfileFields; suggestions?: PaperOption[]; saving: boolean;
  onSave: (fields: ProfileFields) => void; onCancel?: () => void;
}) {
  const [f, setF] = useState<ProfileFields>(initial);
  const set = <K extends keyof ProfileFields>(key: K, value: ProfileFields[K]) => setF((prev) => ({ ...prev, [key]: value }));
  const toggleAvoid = (t: PaperType) =>
    set('avoid_types', f.avoid_types.includes(t) ? f.avoid_types.filter((x) => x !== t) : [...f.avoid_types, t]);
  return (
    <div className="space-y-5">
      <div>
        <label className="section-label mb-1.5">What are you working toward?</label>
        <input value={f.goal} onChange={(e) => set('goal', e.target.value)} className="input-field text-sm py-2"
          placeholder="e.g. implement a diffusion-based text-to-speech model for my thesis" />
      </div>
      <div>
        <label className="section-label mb-1.5">Level in this area</label>
        <div className="segmented">
          {LEVELS.map((l) => (
            <button key={l} type="button" onClick={() => set('level', f.level === l ? '' : l)}
              className={`segmented-item capitalize ${f.level === l ? 'segmented-item-active' : ''}`}>{l}</button>
          ))}
        </div>
      </div>
      <ChipInput label="Topics" values={f.interests} onChange={(v) => set('interests', v)}
        placeholder="Add a topic and press Enter, e.g. score-based generative models" />
      <ChipInput label="Already understand" hint="Recommendations skip papers that only teach these."
        values={f.known} onChange={(v) => set('known', v)} placeholder="e.g. transformers, VAEs" />
      <ChipInput label="Preferences" values={f.preferences} onChange={(v) => set('preferences', v)}
        placeholder="e.g. papers with code, empirical results over theory" />
      <div>
        <label className="section-label mb-1.5">Paper types to skip</label>
        <div className="flex flex-wrap gap-1.5">
          {PAPER_TYPES.filter((t) => t !== 'other').map((t) => (
            <button key={t} type="button" onClick={() => toggleAvoid(t)}
              className={`px-3 py-1 rounded-full text-xs font-medium capitalize transition-colors ${
                f.avoid_types.includes(t) ? 'bg-rose-100 text-rose-700' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'}`}>
              {t}
            </button>
          ))}
        </div>
      </div>
      <SeedPaperPicker value={f.seed_papers} onChange={(v) => set('seed_papers', v)} suggestions={suggestions} />
      <div>
        <label className="section-label mb-1.5">In your words (optional)</label>
        <textarea value={f.summary} onChange={(e) => set('summary', e.target.value)} rows={2} className="input-field text-sm"
          placeholder="Anything else that should shape what you read" />
      </div>
      <div className="flex gap-2">
        <button type="button" onClick={() => onSave(f)} disabled={saving || (!f.goal.trim() && f.interests.length === 0)}
          className="btn-primary text-sm">
          <Check className="w-4 h-4" /> {saving ? 'Saving…' : 'Save and build my feed'}
        </button>
        {onCancel && <button type="button" onClick={onCancel} className="btn-secondary text-sm">Cancel</button>}
      </div>
    </div>
  );
}

/** First visit: describe yourself in a sentence or two, confirm the drafted profile. */
export function ProfileOnboarding({ onSaved, onSkip, isDemo }: {
  onSaved: (p: ResearchProfile) => void; onSkip?: () => void; isDemo?: boolean;
}) {
  const [description, setDescription] = useState('');
  const [drafting, setDrafting] = useState(false);
  // byModel: the fields came from "Draft my profile" rather than "Fill it in myself".
  const [draft, setDraft] = useState<{ fields: ProfileFields; suggestions: PaperOption[]; byModel: boolean } | null>(null);
  const [saving, setSaving] = useState(false);

  const makeDraft = async () => {
    setDrafting(true);
    try {
      const result = await profileApi.draft(description.trim());
      setDraft({ fields: { ...EMPTY, ...result.draft }, suggestions: result.suggested_papers, byModel: true });
    } catch (error) {
      toast.error(errorMessage(error, 'Could not draft a profile. You can fill it in yourself.'));
    } finally {
      setDrafting(false);
    }
  };
  const save = async (fields: ProfileFields) => {
    setSaving(true);
    try {
      onSaved(await profileApi.save(fields));
    } catch (error) {
      toast.error(errorMessage(error, 'Could not save your profile'));
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="card mb-6 border-primary-100">
      <div className="flex items-center gap-2 mb-1">
        <Sparkles className="w-4 h-4 text-primary-600" />
        <h2 className="text-base font-bold text-gray-900">Tell ResearchMate what you are working on</h2>
      </div>
      <p className="text-sm text-gray-500 mb-4">
        Your feed is chosen and explained against this profile. You can edit every field, now and later.
      </p>
      {isDemo ? (
        <p className="notice notice-info">The shared demo account cannot save a profile. Create your own account to set one up.</p>
      ) : !draft ? (
        <div className="space-y-3">
          <textarea value={description} onChange={(e) => setDescription(e.target.value)} rows={3} className="input-field text-sm"
            placeholder="e.g. I'm a master's student who knows transformers well. I want to implement diffusion models for speech, and I prefer papers with code." />
          <div className="flex flex-wrap gap-2">
            <button type="button" onClick={makeDraft} disabled={drafting || description.trim().length < 10} className="btn-primary text-sm">
              <Sparkles className={`w-4 h-4 ${drafting ? 'animate-pulse' : ''}`} /> {drafting ? 'Drafting…' : 'Draft my profile'}
            </button>
            <button type="button" onClick={() => setDraft({ fields: EMPTY, suggestions: [], byModel: false })} className="btn-secondary text-sm">
              Fill it in myself
            </button>
            {onSkip && (
              <button type="button" onClick={onSkip} className="btn-link text-sm ml-auto">Skip, show a general feed</button>
            )}
          </div>
        </div>
      ) : (
        <>
          {draft.byModel && <p className="notice notice-accent mb-4">Here is what we understood. Correct anything that is off, then save.</p>}
          <ProfileEditor initial={draft.fields} suggestions={draft.suggestions} saving={saving} onSave={save}
            onCancel={() => setDraft(null)} />
        </>
      )}
    </div>
  );
}

/** The saved profile, editable in place, with a box for plain-language feedback. */
export function ProfileCard({ profile, isDemo, onSaved, startEditing = false }: {
  profile: ResearchProfile; isDemo?: boolean; onSaved: (p: ResearchProfile, regenerate: boolean) => void; startEditing?: boolean;
}) {
  const [editing, setEditing] = useState(startEditing);
  const [saving, setSaving] = useState(false);
  const [open, setOpen] = useState(false);
  const [feedback, setFeedback] = useState('');
  const [sending, setSending] = useState(false);

  const save = async (fields: ProfileFields) => {
    setSaving(true);
    try {
      onSaved(await profileApi.save(fields), true);
      setEditing(false);
    } catch (error) {
      toast.error(errorMessage(error, 'Could not save your profile'));
    } finally {
      setSaving(false);
    }
  };
  const sendFeedback = async () => {
    setSending(true);
    try {
      const result = await profileApi.feedback(feedback.trim());
      toast.success(result.note || 'Profile updated');
      setFeedback('');
      onSaved(result.profile, false);
    } catch (error) {
      toast.error(errorMessage(error, 'Could not apply that feedback'));
    } finally {
      setSending(false);
    }
  };

  if (editing) {
    return (
      <div className="card mb-6">
        <h2 className="text-base font-bold text-gray-900 mb-4">Edit your research profile</h2>
        <ProfileEditor initial={fieldsOf(profile)} saving={saving} onSave={save} onCancel={() => setEditing(false)} />
      </div>
    );
  }
  return (
    <div className="card mb-6">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="section-label mb-1">Your research profile</p>
          <p className="text-sm font-semibold text-gray-900">{profile.goal || 'No goal stated yet'}</p>
          <div className="flex flex-wrap gap-1.5 mt-2">
            {profile.level && <span className="rounded-full bg-gray-100 px-2.5 py-0.5 text-xs font-medium capitalize text-gray-600">{profile.level}</span>}
            {profile.interests.map((i) => (
              <span key={i} className="rounded-full bg-primary-50 px-2.5 py-0.5 text-xs font-medium text-primary-700">{i}</span>
            ))}
          </div>
        </div>
        <div className="flex shrink-0 gap-2">
          <button type="button" onClick={() => setOpen(!open)} className="btn-secondary btn-sm">
            {open ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />} Details
          </button>
          {!isDemo && (
            <button type="button" onClick={() => setEditing(true)} className="btn-secondary btn-sm">
              <Pencil className="w-3.5 h-3.5" /> Edit
            </button>
          )}
        </div>
      </div>
      {open && (
        <div className="mt-4 grid gap-3 sm:grid-cols-2 text-sm">
          <Detail label="Already understand" items={profile.known} />
          <Detail label="Preferences" items={profile.preferences} />
          <Detail label="Skipping" items={profile.avoid_types} />
          <Detail label="Pinned papers" items={profile.seed_papers.map((p) => p.title)} />
          <Detail label="Implemented in the Coding Agent" items={profile.studied.map((s) => s.title)}
            empty="Papers you implement in the Coding Agent appear here and count as known ground." />
          <Detail label="Feedback applied" items={profile.feedback.slice(-3).map((f) => `“${f.text}” → ${f.note}`)} />
        </div>
      )}
      {!isDemo && (
        <div className="mt-4 flex gap-2">
          <input value={feedback} onChange={(e) => setFeedback(e.target.value)} className="input-field text-sm py-2"
            placeholder="Tell us what is off, e.g. “too theoretical” or “more on evaluation”"
            onKeyDown={(e) => { if (e.key === 'Enter' && feedback.trim().length >= 3) sendFeedback(); }} />
          <button type="button" onClick={sendFeedback} disabled={sending || feedback.trim().length < 3} className="btn-secondary btn-sm shrink-0">
            <Send className={`w-3.5 h-3.5 ${sending ? 'animate-pulse' : ''}`} /> {sending ? 'Applying…' : 'Apply'}
          </button>
        </div>
      )}
    </div>
  );
}

function Detail({ label, items, empty = 'None' }: { label: string; items: string[]; empty?: string }) {
  return (
    <div>
      <p className="section-label mb-1">{label}</p>
      {items.length ? (
        <ul className="space-y-0.5 text-gray-700">{items.map((i) => <li key={i} className="truncate">{i}</li>)}</ul>
      ) : <p className="text-xs text-gray-400">{empty}</p>}
    </div>
  );
}
