// Plain-language names for the Coding Agent's internal states, shown instead of raw codes.

export type Tone = 'gray' | 'sky' | 'violet' | 'emerald' | 'amber' | 'rose';

export const TONE_CLASSES: Record<Tone, string> = {
  gray: 'bg-gray-100 text-gray-700',
  sky: 'bg-sky-100 text-sky-800',
  violet: 'bg-violet-100 text-violet-800',
  emerald: 'bg-emerald-100 text-emerald-800',
  amber: 'bg-amber-100 text-amber-800',
  rose: 'bg-rose-100 text-rose-800',
};

const STAGES: Record<string, string> = {
  queued: 'queued', starting: 'starting', resuming: 'resuming', plan_confirmed: 'starting with your plan',
  evidence: 'reading the paper', gather: 'looking for missing details', confirm: 'waiting for you',
  generate: 'writing the code', review: 'checking the code against the paper', fix: 'revising the draft',
  execute: 'running the program', repair: 'fixing a failed run', finish: 'wrapping up',
  prepare: 'downloading the repository', explain: 'writing the overview',
};

export const STOP_REASONS: Record<string, { label: string; tone: Tone }> = {
  generated_unexecuted: { label: 'Draft ready · not run', tone: 'sky' },
  success: { label: 'Ran and passed its self-check', tone: 'emerald' },
  max_attempts: { label: 'Did not pass after 5 runs', tone: 'amber' },
  repeated_failure: { label: 'Stopped: the same error repeated', tone: 'amber' },
  method_detail_insufficient: { label: 'Not enough method detail in the paper', tone: 'amber' },
  draft_validation_failed: { label: 'Draft has an unresolved code problem', tone: 'amber' },
};

export function runStatus(run: { kind: string; status: string; stage: string; stop_reason?: string }): { label: string; tone: Tone } {
  switch (run.status) {
    case 'queued': return { label: 'Queued', tone: 'gray' };
    case 'running': return { label: `Working · ${STAGES[run.stage] || run.stage.replace(/_/g, ' ')}`, tone: 'sky' };
    case 'waiting': return { label: 'Needs your review', tone: 'violet' };
    case 'failed': return { label: 'Failed', tone: 'rose' };
    case 'cancelled': return { label: 'Cancelled', tone: 'gray' };
    case 'completed':
      return run.kind === 'repo' ? { label: 'Overview ready', tone: 'emerald' }
        : STOP_REASONS[run.stop_reason || ''] || { label: 'Finished', tone: 'emerald' };
    default: return { label: run.status, tone: 'gray' };
  }
}

export function StatusPill({ label, tone }: { label: string; tone: Tone }) {
  return <span className={`inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-xs font-medium ${TONE_CLASSES[tone]}`}>
    {(tone === 'sky' || tone === 'violet') && <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-current" />}{label}</span>;
}
