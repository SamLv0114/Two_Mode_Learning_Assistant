import { ImageResponse } from 'next/og';

export const runtime = 'edge';
export const alt = 'ResearchMate — read ML research, then build it yourself';
export const size = { width: 1200, height: 630 };
export const contentType = 'image/png';

/** Link preview for shared URLs (LinkedIn, Slack, messaging apps). */
export default function OpenGraphImage() {
  return new ImageResponse(
    (
      <div style={{
        width: '100%', height: '100%', display: 'flex', flexDirection: 'column', justifyContent: 'center',
        padding: '80px', background: 'linear-gradient(135deg, #0284c7 0%, #4f46e5 55%, #7c3aed 100%)', color: 'white',
      }}>
        <div style={{ fontSize: 34, opacity: 0.85, marginBottom: 28 }}>🔬 ResearchMate</div>
        <div style={{ fontSize: 76, fontWeight: 800, lineHeight: 1.1 }}>Read ML research.</div>
        <div style={{ fontSize: 76, fontWeight: 800, lineHeight: 1.1, opacity: 0.8 }}>Then build it yourself.</div>
        <div style={{ fontSize: 30, marginTop: 40, opacity: 0.85 }}>
          Personalised papers · cited research agents · paper-to-code with verifiable sources
        </div>
      </div>
    ),
    size,
  );
}
