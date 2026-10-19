'use client';

import { useEffect, useState } from 'react';
import { publicApi, PublicStats } from './api';

/** Live corpus size and demo availability for unauthenticated pages; null until loaded or if unreachable. */
export function usePublicStats(): PublicStats | null {
  const [stats, setStats] = useState<PublicStats | null>(null);
  useEffect(() => {
    let active = true;
    publicApi.stats().then((value) => { if (active) setStats(value); }).catch(() => { /* render without numbers */ });
    return () => { active = false; };
  }, []);
  return stats;
}

/** 44774 -> "44k+"; small counts stay exact. */
export function roundedCount(value: number): string {
  return value >= 1000 ? `${Math.floor(value / 1000)}k+` : String(value);
}
