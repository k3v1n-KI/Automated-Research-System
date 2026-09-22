// Pathways v2 — Home A · Command-first (Linear/Raycast-inspired)
// One focal point: the command input. Recent cases below as a simple list.

function HomeCommand() {
  return (
    <div className="v2">
      <V2Topnav active="cases" />
      <div className="body">
        <main style={{ maxWidth: 720, margin: '0 auto', padding: '88px 32px 80px' }}>

          {/* Tag line + greeting */}
          <div style={{ textAlign: 'left', marginBottom: 36 }}>
            <div className="eyebrow" style={{ marginBottom: 14 }}>Wednesday · May 7</div>
            <h1 className="title-hero" style={{ fontSize: 36, marginBottom: 10 }}>
              Find the right service for a patient.
            </h1>
            <p className="ital" style={{ fontSize: 19, color: 'var(--ink-3)', lineHeight: 1.4 }}>
              Three saved cases waiting. One region update.
            </p>
          </div>

          {/* Command input */}
          <div style={{
            border: '1px solid var(--stroke)',
            borderRadius: 'var(--r-md)',
            background: 'var(--paper)',
            boxShadow: 'var(--shadow-sm)',
            overflow: 'hidden',
            marginBottom: 24,
          }}>
            <div className="row" style={{
              padding: '14px 16px',
              gap: 12,
              borderBottom: '1px solid var(--stroke)',
            }}>
              <V2Ico name="sparkle" size={16} stroke={1.7} style={{ color: 'var(--accent)' }} />
              <span style={{ fontSize: 14.5, color: 'var(--ink-3)' }}>
                Describe a case, or search for a service…
              </span>
              <span className="nav-spacer" />
              <span className="kbd">⌘</span><span className="kbd">K</span>
            </div>
            {/* Suggested actions */}
            <div className="col" style={{ padding: 6, gap: 0 }}>
              {[
                { ico: 'plus', label: 'Start a new case', meta: 'plain language + AI parse', shortcut: 'N' },
                { ico: 'corner-arrow', label: 'Continue last draft', meta: 'Senior · post-discharge · home support', shortcut: '↵' },
                { ico: 'book', label: 'Browse directory', meta: '412 verified resources in East TO OHT', shortcut: 'D' },
              ].map((it, i) => (
                <div key={i} className="row" style={{
                  padding: '10px 12px',
                  borderRadius: 'var(--r-sm)',
                  gap: 12,
                  background: i === 0 ? 'var(--paper-2)' : 'transparent',
                  cursor: 'pointer',
                }}>
                  <V2Ico name={it.ico} size={14} stroke={1.6} style={{ color: 'var(--ink-3)' }} />
                  <span style={{ fontSize: 13.5, color: 'var(--ink)', fontWeight: 500, minWidth: 160 }}>{it.label}</span>
                  <span className="muted" style={{ fontSize: 12, flex: 1 }}>{it.meta}</span>
                  <span className="kbd">{it.shortcut}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Inline region note */}
          <div style={{
            display: 'flex', alignItems: 'flex-start', gap: 10,
            padding: '12px 14px', marginBottom: 56,
            background: 'var(--accent-tint)',
            border: '1px solid color-mix(in srgb, var(--accent) 14%, var(--stroke))',
            borderRadius: 'var(--r-md)',
          }}>
            <V2Ico name="sparkle" size={13} stroke={1.7} style={{ color: 'var(--accent)', flexShrink: 0, marginTop: 2 }}/>
            <div style={{ flex: 1, fontSize: 13, lineHeight: 1.55 }}>
              <strong>Across Health</strong> changed walk-in MH hours Monday. <span className="ital muted">Three of your saved cases reference this resource.</span>
            </div>
            <a style={{ fontSize: 12.5, color: 'var(--accent-ink)', fontWeight: 500, whiteSpace: 'nowrap' }}>Review →</a>
          </div>

          {/* Recent cases — minimal list */}
          <section>
            <div className="between" style={{ marginBottom: 14 }}>
              <h3 className="title-2">Recent cases</h3>
              <a className="muted" style={{ fontSize: 12.5 }}>View all 14 →</a>
            </div>
            <div>
              {V2_CASES.map((c) => (
                <a key={c.id} className="row-link" style={{ gridTemplateColumns: '1fr auto auto' }}>
                  <div>
                    <div className="row-link-title" style={{ fontSize: 13.5, fontWeight: 500, color: 'var(--ink)', transition: 'color 0.12s' }}>{c.title}</div>
                    <div className="muted" style={{ fontSize: 12, marginTop: 2 }}>{c.summary}</div>
                  </div>
                  <span className="row" style={{ gap: 5 }}>
                    <span className={`dot ${c.status === 'referred' ? 'good' : c.status === 'closed' ? 'muted' : 'accent'}`} />
                    <span className="muted" style={{ fontSize: 12, minWidth: 56 }}>{c.status}</span>
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

window.HomeCommand = HomeCommand;
