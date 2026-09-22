// Pathways v2 — Home C · Workbench (2-up asymmetric)
// Primary action card (big) + region/peer activity (smaller). Recent cases below.
// Keeps the grid but quieter than v1 — no stats strip, no 3-up forum cards.

function HomeWorkbench() {
  return (
    <div className="v2">
      <V2Topnav active="cases" />
      <div className="body">
        <main style={{ maxWidth: 1080, margin: '0 auto', padding: '52px 32px 80px' }}>

          {/* Greeting */}
          <div style={{ marginBottom: 36 }}>
            <div className="eyebrow" style={{ marginBottom: 10 }}>Wednesday · May 7</div>
            <div style={{ display: 'flex', alignItems: 'baseline', gap: 12, flexWrap: 'wrap' }}>
              <h1 className="title-hero" style={{ fontSize: 32 }}>Good morning, Rita.</h1>
              <span style={{ fontSize: 18, color: 'var(--ink-3)', fontWeight: 400 }}>
                three saved cases waiting.
              </span>
            </div>
          </div>

          {/* Two-up: action + region */}
          <div style={{ display: 'grid', gridTemplateColumns: '1.5fr 1fr', gap: 16, marginBottom: 56 }}>

            {/* Action */}
            <button style={{
              all: 'unset', cursor: 'pointer',
              background: 'var(--ink)', color: 'var(--paper)',
              borderRadius: 'var(--r-lg)',
              padding: 28,
              display: 'flex', flexDirection: 'column', justifyContent: 'space-between',
              minHeight: 220,
              transition: 'transform 0.15s, box-shadow 0.15s',
              boxShadow: 'var(--shadow-sm)',
            }}>
              <div>
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10.5, textTransform: 'uppercase', letterSpacing: '0.08em', opacity: 0.55, marginBottom: 14 }}>
                  Start here
                </div>
                <h2 style={{ fontSize: 30, lineHeight: 1.1, letterSpacing: '-0.025em', fontWeight: 600, maxWidth: 460, marginBottom: 12 }}>
                  Find the right service for a patient.
                </h2>
                <p style={{ fontSize: 14.5, opacity: 0.75, maxWidth: 460, lineHeight: 1.5, fontWeight: 400 }}>
                  Sketch the case in plain language. We rank what fits.
                </p>
              </div>
              <div className="row" style={{ gap: 14, marginTop: 24 }}>
                <span style={{
                  display: 'inline-flex', alignItems: 'center', gap: 8,
                  padding: '9px 16px',
                  background: 'var(--accent)', color: 'white',
                  borderRadius: 'var(--r-sm)', fontWeight: 500, fontSize: 13.5,
                }}>
                  Start a case <V2Ico name="arrow-right" size={14} />
                </span>
                <span style={{ fontSize: 12, opacity: 0.5, fontFamily: 'var(--font-mono)' }}>
                  <span className="kbd" style={{ background: 'rgb(255 255 255 / 0.08)', borderColor: 'rgb(255 255 255 / 0.16)', color: 'rgb(255 255 255 / 0.7)' }}>N</span>
                  <span style={{ marginLeft: 8 }}>or paste a referral note</span>
                </span>
              </div>
            </button>

            {/* Region update */}
            <div className="card" style={{ display: 'flex', flexDirection: 'column', justifyContent: 'space-between', padding: 22, minHeight: 220 }}>
              <div>
                <div className="row" style={{ gap: 8, marginBottom: 12 }}>
                  <V2Ico name="sparkle" size={13} stroke={1.7} style={{ color: 'var(--accent)' }}/>
                  <span className="eyebrow" style={{ color: 'var(--accent-ink)' }}>Region update · 1h ago</span>
                </div>
                <p style={{ fontSize: 14, lineHeight: 1.55 }}>
                  <strong>Across Health</strong> walk-in MH hours changed Monday.
                </p>
                <p className="ital" style={{ fontSize: 13.5, color: 'var(--ink-3)', marginTop: 6 }}>
                  Three of your saved cases reference this resource.
                </p>
              </div>
              <div className="row" style={{ gap: 8 }}>
                <button className="btn sm">Review impact</button>
                <button className="btn ghost sm">Dismiss</button>
              </div>
            </div>
          </div>

          {/* Recent — clean table */}
          <section>
            <div className="between" style={{ marginBottom: 16 }}>
              <h3 className="title-2">Recent cases</h3>
              <div className="row" style={{ gap: 12 }}>
                <span className="muted" style={{ fontSize: 12.5 }}>14 total</span>
                <a className="muted" style={{ fontSize: 12.5 }}>View all →</a>
              </div>
            </div>
            <div className="card" style={{ padding: 0 }}>
              {V2_CASES.map((c, i) => (
                <a key={c.id} className="row" style={{
                  padding: '14px 18px',
                  borderBottom: i < V2_CASES.length - 1 ? '1px solid var(--stroke)' : 0,
                  display: 'grid',
                  gridTemplateColumns: '1.4fr 1fr auto auto auto',
                  gap: 16,
                  cursor: 'pointer',
                  fontSize: 13.5,
                }}>
                  <div className="stack-2">
                    <div style={{ fontWeight: 500 }}>{c.title}</div>
                    <div className="muted" style={{ fontSize: 12 }}>{c.summary}</div>
                  </div>
                  <div className="row" style={{ gap: 4, flexWrap: 'wrap' }}>
                    {c.tags.map(t => <span key={t} className="chip sm">{t}</span>)}
                  </div>
                  <span className="muted" style={{ fontSize: 12 }}>{c.region}</span>
                  <span className="row" style={{ gap: 5, fontSize: 12 }}>
                    <span className={`dot ${c.status === 'referred' ? 'good' : c.status === 'closed' ? 'muted' : 'accent'}`} />
                    <span className="muted" style={{ minWidth: 50 }}>{c.status}</span>
                  </span>
                  <span className="mono dim" style={{ fontSize: 11, minWidth: 64, textAlign: 'right' }}>{c.when}</span>
                </a>
              ))}
            </div>
          </section>
        </main>
      </div>
    </div>
  );
}

window.HomeWorkbench = HomeWorkbench;
