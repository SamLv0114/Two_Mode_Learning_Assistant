'use client';

import { useState } from 'react';
import { authApi, errorMessage, User } from '@/lib/api';
import toast from 'react-hot-toast';
import { Save, RefreshCw, Settings, SlidersHorizontal } from 'lucide-react';

/**
 * Account settings. Focus areas and interests are edited in one place only — the Daily Feed's
 * Edit panel, next to the feed they change — so this page summarises them and links there.
 */
export function UserSettings({ user, onUpdate, onEditInterests }: {
  user: User | null; onUpdate: () => void; onEditInterests: () => void;
}) {
  const [fullName, setFullName] = useState(user?.full_name || '');
  const [isSaving, setIsSaving] = useState(false);

  const handleSave = async () => {
    setIsSaving(true);
    try {
      await authApi.updateProfile({ full_name: fullName || undefined });
      toast.success('Profile saved');
      onUpdate();
    } catch (error) {
      toast.error(errorMessage(error, 'Failed to save profile'));
    } finally {
      setIsSaving(false);
    }
  };

  const areas = user?.focus_areas || [];
  const interests = user?.interests || [];
  return (
    <div className="space-y-6">
      <div className="card">
        <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
          <Settings className="w-5 h-5" />
          Profile
        </h2>
        <div className="space-y-4">
          <div>
            <label htmlFor="settings-email" className="block text-sm font-medium text-gray-700 mb-1">Email</label>
            <input id="settings-email" type="text" value={user?.email || ''} disabled
              className="input-field w-full bg-gray-100 cursor-not-allowed" />
          </div>
          <div>
            <label htmlFor="settings-name" className="block text-sm font-medium text-gray-700 mb-1">Full name</label>
            <input id="settings-name" type="text" value={fullName} onChange={(e) => setFullName(e.target.value)}
              placeholder="Your name" className="input-field w-full" disabled={user?.is_demo} />
          </div>
          {user?.is_demo && <p className="notice notice-info text-xs">The shared demo account cannot be edited. Create your own account to set a profile.</p>}
          <button onClick={handleSave} disabled={isSaving || user?.is_demo} className="btn-primary flex items-center gap-2 disabled:opacity-50">
            {isSaving ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Save className="w-4 h-4" />}
            {isSaving ? 'Saving...' : 'Save profile'}
          </button>
        </div>
      </div>

      <div className="card">
        <h2 className="text-lg font-semibold text-gray-900 mb-1 flex items-center gap-2">
          <SlidersHorizontal className="w-5 h-5" />
          Research profile
        </h2>
        <p className="text-sm text-gray-500 mb-4">What your feed and the agents use to personalise results.</p>
        <p className="section-label">Focus areas</p>
        <div className="mt-1 mb-3 flex flex-wrap gap-1.5">
          {areas.length ? areas.map((area) => (
            <span key={area} className="px-2.5 py-0.5 rounded-full text-xs font-medium bg-primary-100 text-primary-700">{area}</span>
          )) : <span className="text-sm text-gray-400">None selected</span>}
        </div>
        <p className="section-label">Interests</p>
        <ul className="mt-1 mb-4 list-disc pl-5 text-sm text-gray-700">
          {interests.length ? interests.map((interest) => <li key={interest}>{interest}</li>)
            : <li className="list-none -ml-5 text-gray-400">None yet</li>}
        </ul>
        <button type="button" onClick={onEditInterests} className="btn-secondary btn-sm">Edit on Daily Feed</button>
      </div>
    </div>
  );
}
