// Pathways v2 — Home C v3 (Workbench, workflow-aware)
// Search-first. Your Asks + Verify nearby surface as the two side cards.
// Dashboards (recent cases) appear below once you have any.

function HomeWorkbenchV3() {
  return (
    <div className="v2">
      <V2Topnav active="cases" />
      <div className="body">
        <main style={{ maxWidth: 1080, margin: '0 auto', padding: '44px 32px 80px' }}>

          {/* Greeting */}
          <div style={{ marginBottom: 28 }}>
            <div className="eyebrow" style={{ marginBottom: 10 }}>Wednesday · May 7</div>
            <div style={{ display: 'flex', alignItems: 'baseline', gap: 12, flexWrap: 'wrap' }}>
              <h1 className="title-hero" style={{ fontSize: 32 }}>Good morning, Rita.</h1>
              <span style={{ fontSize: 18, color: 'var(--ink-3)' }}>
                3 saved cases · 2 Asks running.
              </span>
            </div>
          </div>

          {/* Search — the front door */}
          <div style={{
            background: 'var(--ink)', color: 'var(--paper)',
            borderRadius: 'var(--r-lg)', padding: 24, marginBottom: 16,
            boxShadow: 'var(--shadow-sm)',
          }}>
            <div className="row" style={{ gap: 8, marginBottom: 12, opacity: 0.55 }}>
              <V2Ico name="search" size={13} stroke={1.8} />
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10.5, textTransform: 'uppercase', letterSpacing: '0.08em' }}>
                Find a service · or post an Ask
              </span>
            </div>
            <div style={{
              fontSize: 22, fontWeight: 500, letterSpacing: '-0.018em',
              color: 'rgb(255 255 255 / 0.95)', lineHeight: 1.3,
              minHeight: 60, paddingBottom: 14,
              borderBottom: '1px solid rgb(255 255 255 / 0.12)',
            }}>
              Same-day MH walk-in for an adult without OHIP, East Toronto<span style={{ opacity: 0.6, marginLeft: 1, animation: 'none' }}>│</span>
            </div>
            <div className="row" style={{ marginTop: 14, gap: 14, justifyContent: 'space-between' }}>
              <span style={{ fontSize: 11.5, opacity: 0.55, fontFamily: 'var(--font-mono)' }}>
                Plain language works. Paste a referral note, or attach a photo.
              </span>
              <div className="row" style={{ gap: 8 }}>
                <button className="btn sm" style={{ background: 'rgb(255 255 255 / 0.08)', borderColor: 'rgb(255 255 255 / 0.16)', color: 'rgb(255 255 255 / 0.9)' }}>
                  <V2Ico name="plus" size={12}/> Tip
                </button>
                <span style={{ display: 'inline-flex', alignItems: 'center', gap: 8, padding: '7px 14px',
                  background: 'var(--accent)', color: 'white',
                  borderRadius: 'var(--r-sm)', fontWeight: 500, fontSize: 13 }}>
                  Search <V2Ico name="arrow-right" size={13} />
                </span>
              </div>
            </div>
          </div>

          {/* Two-up: Your Asks · Verify nearby */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 44 }}>

            {/* Your Asks */}
            <div className="card" style={{ padding: 18, display: 'flex', flexDirection: 'column', gap: 12 }}>
              <div className="between">
                <div className="row" style={{ gap: 8 }}>
                  <V2Ico name="corner-arrow" size={13} stroke={1.7} style={{ color: 'var(--accent)' }}/>
                  <span className="eyebrow" style={{ color: 'var(--accent-ink)' }}>Your Asks · 2 running</span>
                </div>
                <a className="muted" style={{ fontSize: 11.5 }}>All →</a>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                <div style={{ paddingBottom: 10, borderBottom: '1px solid var(--stroke)' }}>
                  <div style={{ fontSize: 13, color: 'var(--ink)', lineHeight: 1.45, marginBottom: 5 }}>
                    Same-day MH walk-in for an adolescent without OHIP — any options not on the index?
                  </div>
                  <div className="row" style={{ gap: 8, fontSize: 11, color: 'var(--ink-3)' }}>
                    <span style={{ color: 'var(--accent-ink)', fontWeight: 500 }}>● 1 new reply</span>
                    <span>·</span>
                    <span>3 replies · 7 watching</span>
                    <span>·</span>
                    <span className="mono">6d left</span>
                  </div>
                </div>
                <div>
                  <div style={{ fontSize: 13, color: 'var(--ink)', lineHeight: 1.45, marginBottom: 5 }}>
                    Cantonese-speaking caregiver respite, weekend hours, will pay out-of-pocket
                  </div>
                  <div className="row" style={{ gap: 8, fontSize: 11, color: 'var(--ink-3)' }}>
                    <span>0 replies · 2 watching</span>
                    <span>·</span>
                    <span className="mono">5d left</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Verify nearby */}
            <div className="card" style={{ padding: 18, display: 'flex', flexDirection: 'column', gap: 12 }}>
              <div className="between">
                <div className="row" style={{ gap: 8 }}>
                  <V2Ico name="check" size={13} stroke={2} style={{ color: 'var(--good)' }}/>
                  <span className="eyebrow">Verify nearby · East York</span>
                </div>
                <a className="muted" style={{ fontSize: 11.5 }}>Show 5 →</a>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                {[
                  { name: 'Across Health · MH Walk-in', field: 'Referral path', flag: '2× flagged stale', tone: 'crit' },
                  { name: 'Eastside Community MH', field: 'Hours', flag: 'AI-extracted, never confirmed', tone: 'ai' },
                  { name: 'Family Services East', field: 'Phone', flag: '210d since confirmed', tone: 'warn' },
                ].map((r, i) => (
                  <div key={i} style={{ paddingBottom: i < 2 ? 10 : 0, borderBottom: i < 2 ? '1px solid var(--stroke)' : 0 }}>
                    <div style={{ fontSize: 13, color: 'var(--ink)', marginBottom: 3, fontWeight: 500 }}>{r.name}</div>
                    <div className="row" style={{ gap: 8, fontSize: 11, color: 'var(--ink-3)' }}>
                      <span>{r.field}</span>
                      <span>·</span>
                      <span style={{ color: r.tone === 'crit' ? '#b91c1c' : r.tone === 'warn' ? '#a16207' : 'var(--accent-ink)', fontWeight: 500 }}>
                        {r.flag}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Recent cases — still here, but quieter */}
          <section>
            <div className="between" style={{ marginBottom: 14 }}>
              <h3 className="title-2">Recent cases</h3>
              <div className="row" style={{ gap: 12 }}>
                <span className="muted" style={{ fontSize: 12.5 }}>14 total</span>
                <a className="muted" style={{ fontSize: 12.5 }}>View all →</a>
              </div>
            </div>
            <div className="card" style={{ padding: 0 }}>
              {V2_CASES.map((c, i) => (
                <a key={c.id} className="row" style={{
                  padding: '13px 18px',
                  borderBottom: i < V2_CASES.length - 1 ? '1px solid var(--stroke)' : 0,
                  display: 'grid',
                  gridTemplateColumns: '1.4fr 1fr auto auto auto',
                  gap: 16, cursor: 'pointer', fontSize: 13.5,
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

window.HomeWorkbenchV3 = HomeWorkbenchV3;
