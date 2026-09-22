// Pathways v2 — Home B · Editorial brief
// Italic-serif greeting, single primary CTA card centered. Recent cases as an editorial list.

function HomeEditorial() {
  return (
    <div className="v2">
      <V2Topnav active="cases" />
      <div className="body">
        <main style={{ maxWidth: 880, margin: '0 auto', padding: '72px 32px 80px' }}>

          {/* Editorial header */}
          <div style={{ marginBottom: 48 }}>
            <div className="eyebrow" style={{ marginBottom: 22 }}>Today · Wednesday, May 7</div>
            <h1 style={{ fontSize: 56, lineHeight: 1.05, letterSpacing: '-0.025em', fontWeight: 600, fontFamily: 'var(--font-ui)', color: 'var(--ink)', marginBottom: 14 }}>
              Good morning, Rita.
            </h1>
            <p style={{ fontSize: 16, color: 'var(--ink-3)', maxWidth: 520, lineHeight: 1.55 }}>
              You have <span style={{ color: 'var(--ink)' }}>three saved cases</span> waiting and a peer flagged a wait-time change in your region.
            </p>
          </div>

          {/* Single primary CTA */}
          <div style={{
            border: '1px solid var(--stroke)',
            borderRadius: 'var(--r-lg)',
            background: 'var(--paper)',
            padding: 28,
            marginBottom: 56,
            display: 'grid',
            gridTemplateColumns: '1fr auto',
            gap: 32,
            alignItems: 'center',
          }}>
            <div>
              <div className="eyebrow" style={{ marginBottom: 10, color: 'var(--accent-ink)' }}>Start here</div>
              <h2 className="title-1" style={{ fontSize: 24, marginBottom: 8 }}>
                Find services for a patient
              </h2>
              <p className="ital" style={{ fontSize: 16, color: 'var(--ink-3)', lineHeight: 1.5, maxWidth: 460 }}>
                Sketch the case in plain language — we extract needs, eligibility, and constraints, then rank resources in your region.
              </p>
            </div>
            <button className="btn accent lg" style={{ padding: '12px 18px', fontSize: 14 }}>
              Start a new case <V2Ico name="arrow-right" size={14} />
            </button>
          </div>

          {/* Recent — editorial table */}
          <section>
            <div className="between" style={{ marginBottom: 18, paddingBottom: 8, borderBottom: '1px solid var(--ink)' }}>
              <h3 className="title-2" style={{ fontSize: 14, letterSpacing: '0.04em', textTransform: 'uppercase', fontWeight: 600 }}>
                Recent cases
              </h3>
              <a className="muted" style={{ fontSize: 12, fontFamily: 'var(--font-mono)', letterSpacing: '0.05em' }}>VIEW ALL →</a>
            </div>
            <div>
              {V2_CASES.map((c, i) => (
                <a key={c.id} style={{
                  display: 'grid', gridTemplateColumns: '40px 1fr auto auto',
                  gap: 20, padding: '18px 0',
                  borderBottom: i < V2_CASES.length - 1 ? '1px solid var(--stroke)' : 0,
                  alignItems: 'center',
                }}>
                  <span className="ital" style={{ fontSize: 22, color: 'var(--ink-4)' }}>
                    {String(i+1).padStart(2, '0')}
                  </span>
                  <div>
                    <div style={{ fontSize: 14.5, color: 'var(--ink)', marginBottom: 2 }}>{c.title}</div>
                    <div className="ital" style={{ fontSize: 13, color: 'var(--ink-3)' }}>{c.summary}</div>
                  </div>
                  <span className="row" style={{ gap: 6, fontSize: 12 }}>
                    <span className={`dot ${c.status === 'referred' ? 'good' : c.status === 'closed' ? 'muted' : 'accent'}`} />
                    <span className="muted" style={{ minWidth: 60 }}>{c.status}</span>
                  </span>
                  <span className="ital muted" style={{ fontSize: 13, minWidth: 80, textAlign: 'right' }}>{c.when}</span>
                </a>
              ))}
            </div>
          </section>

          {/* Region note — quiet pull-quote at bottom */}
          <aside style={{ marginTop: 56, paddingLeft: 18, borderLeft: '2px solid var(--accent)' }}>
            <div className="eyebrow" style={{ marginBottom: 8, color: 'var(--accent-ink)' }}>Region update · 1h ago</div>
            <p className="ital" style={{ fontSize: 17, color: 'var(--ink)', lineHeight: 1.5, marginBottom: 8 }}>
              "Across Health intake is now Monday and Wednesday only — confirmed by phone this morning."
            </p>
            <p className="muted" style={{ fontSize: 12.5 }}>M. Patel, RN · East Toronto · 4 peer confirmations</p>
          </aside>
        </main>
      </div>
    </div>
  );
}

window.HomeEditorial = HomeEditorial;
