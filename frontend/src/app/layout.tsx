import type { Metadata } from 'next';
import 'katex/dist/katex.min.css';
import './globals.css';
import { Providers } from './providers';
import { Toaster } from 'react-hot-toast';

const DESCRIPTION = 'Personalised ML paper recommendations, research agents that cite their sources, '
  + 'and a coding agent that turns papers and repositories into code you can trace back to the source.';

export const metadata: Metadata = {
  metadataBase: new URL(process.env.NEXT_PUBLIC_SITE_URL || 'https://researchmate.site'),
  title: 'ResearchMate',
  description: DESCRIPTION,
  openGraph: { title: 'ResearchMate', description: DESCRIPTION, siteName: 'ResearchMate', type: 'website' },
  twitter: { card: 'summary_large_image', title: 'ResearchMate', description: DESCRIPTION },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="font-sans">
        <Providers>
          {children}
          <Toaster position="bottom-right" />
        </Providers>
      </body>
    </html>
  );
}
