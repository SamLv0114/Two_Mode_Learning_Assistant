'use client';

import { ColdStartSeed, ColdStartResponse } from '@/lib/api';
import { Save, X } from 'lucide-react';

export function ColdStartSection({
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
