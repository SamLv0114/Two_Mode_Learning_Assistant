'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import toast from 'react-hot-toast';
import { useAuth } from '@/lib/auth';
import { errorMessage } from '@/lib/api';

/** Signs in to the shared demo account and opens the dashboard. */
export function TryDemoButton({ className, children = 'Try the demo — no sign-up' }: {
  className: string; children?: React.ReactNode;
}) {
  const { loginAsDemo } = useAuth();
  const router = useRouter();
  const [busy, setBusy] = useState(false);
  const start = async () => {
    setBusy(true);
    try {
      await loginAsDemo();
      router.push('/dashboard');
    } catch (error) {
      toast.error(errorMessage(error, 'The demo is not available right now'));
      setBusy(false);
    }
  };
  return <button type="button" onClick={start} disabled={busy} className={className}>{busy ? 'Opening the demo…' : children}</button>;
}
