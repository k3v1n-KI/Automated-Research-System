// Pathways v2 — Search Results
// Left: ranked resource rows with aggregate verification + most-flagged surfaced
// Right rail: "Not finding it?" → Post as Ask · plus the active region Asks

function SearchResults() {
  return (
    <div className="v2">
      <V2Topnav active="cases" />
      <div className="body">
        <main style={{ maxWidth: 1180, margin: '0 auto', padding: '28px 32px 80px' }}>

          {/* Search recap */}
          <div style={{ marginBottom: 18 }}>
            <div className="eyebrow" style={{ marginBottom: 8 }}>Search</div>
            <div style={{
              display: 'flex', alignItems: 'center', gap: 12,
              background: 'var(--paper-2)', border: '1px solid var(--stroke)',
              borderRadius: 'var(--r-md)', padding: '12px 16px',
            }}>
              <V2Ico name="search" size={14} style={{ color: 'var(--ink-3)', flexShrink: 0 }}/>
              <span style={{ fontSize: 15, color: 'var(--ink)', flex: 1 }}>
                Same-day MH walk-in for an adult without OHIP, East Toronto
              </span>
              <span className="row" style={{ gap: 4 }}>
                <span className="chip sm">MH</span>
                <span className="chip sm">walk-in</span>
                <span className="chip sm">uninsured</span>
                <span className="chip sm">East Toronto</span>
              </span>
              <a className="muted" style={{ fontSize: 12 }}>Edit</a>
            </div>
            <div className="row" style={{ marginTop: 10, gap: 10, fontSize: 12, color: 'var(--ink-3)' }}>
              <span className="mono">4 resources · 1 Ask answers this · ranked by fit + verification freshness</span>
            </div>
          </div>

          {/* 2-col: results + Asks rail */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 320px', gap: 24, alignItems: 'start' }}>

            {/* Results */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              {SEARCH_RESULTS.map((r, i) => (
                <ResultRow key={r.id} r={r} focused={i === 0}/>
              ))}

              {/* The decisive empty-state-ish prompt at the bottom */}
              <div style={{
                marginTop: 8,
                border: '1px dashed var(--stroke-2)', borderRadius: 'var(--r-md)',
                padding: '18px 20px', background: 'var(--accent-tint)',
                display: 'flex', gap: 16, alignItems: 'center', justifyContent: 'space-between',
              }}>
                <div style={{ flex: 1 }}>
                  <div className="row" style={{ gap: 7, marginBottom: 4 }}>
                    <V2Ico name="corner-arrow" size={13} stroke={1.8} style={{ color: 'var(--accent)' }}/>
                    <span className="eyebrow" style={{ color: 'var(--accent-ink)' }}>Not finding it?</span>
                  </div>
                  <div style={{ fontSize: 14, color: 'var(--ink)', lineHeight: 1.45 }}>
                    Post this search as an <strong>Ask</strong>. Other AOHT members in East Toronto see it for 7 days
                    and can point to resources we don't have indexed yet.
                  </div>
                </div>
                <div className="row" style={{ gap: 8, flexShrink: 0 }}>
                  <button className="btn sm">Add what you know</button>
                  <button className="btn accent sm">Post as Ask →</button>
                </div>
              </div>
            </div>

            {/* Right rail — region Asks */}
            <aside style={{ display: 'flex', flexDirection: 'column', gap: 12, position: 'sticky', top: 12 }}>
              <div className="card" style={{ padding: 16 }}>
                <div className="between" style={{ marginBottom: 10 }}>
                  <span className="eyebrow">Open Asks · East Toronto</span>
                  <a className="muted" style={{ fontSize: 11 }}>All →</a>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                  {ASKS.slice(0, 3).map(a => (
                    <div key={a.id} style={{
                      paddingBottom: 10,
                      borderBottom: '1px solid var(--stroke)',
                    }}>
                      <div style={{ fontSize: 12.5, color: 'var(--ink)', lineHeight: 1.45, marginBottom: 5 }}>
                        {a.text}
                      </div>
                      <div className="row" style={{ gap: 7, fontSize: 10.5, color: 'var(--ink-3)' }}>
                        <span style={{ fontWeight: 500 }}>{a.who}</span>
                        <span>·</span>
                        <span>{a.region}</span>
                        <span>·</span>
                        <span className="mono">{a.replies} replies</span>
                      </div>
                    </div>
                  ))}
                  <a className="row" style={{ gap: 6, fontSize: 12, color: 'var(--accent-ink)', fontWeight: 500, marginTop: 2 }}>
                    <V2Ico name="plus" size={12} stroke={2}/> Post a new Ask
                  </a>
                </div>
              </div>

              <div className="card muted" style={{ padding: 14, background: 'var(--paper-2)' }}>
                <div className="eyebrow" style={{ marginBottom: 6 }}>How ranking works</div>
                <p style={{ fontSize: 12, color: 'var(--ink-3)', lineHeight: 1.5, margin: 0 }}>
                  AI scores fit against your case. We boost resources with recent member verification
                  and penalize ones with active stale flags. <a style={{ color: 'var(--accent-ink)', fontWeight: 500 }}>Learn more →</a>
                </p>
              </div>
            </aside>
          </div>
        </main>
      </div>
    </div>
  );
}

function ResultRow({ r, focused }) {
  const m = matchTier(r.score);
  return (
    <a className="card" style={{
      padding: 18, display: 'grid', gridTemplateColumns: '52px 1fr auto', gap: 16,
      borderColor: focused ? 'var(--stroke-2)' : 'var(--stroke)',
      cursor: 'pointer', textAlign: 'left',
    }}>
      {/* Match tier */}
      <div style={{
        background: focused ? 'var(--ink)' : 'var(--paper-2)',
        color: focused ? 'var(--paper)' : 'var(--ink-2)',
        borderRadius: 'var(--r-sm)', padding: '8px 0',
        display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', gap: 5,
      }}>
        <MatchMeter tier={m.tier} dark={focused}/>
        <div className="mono" style={{ fontSize: 9, opacity: 0.7 }}>{m.tier}/5</div>
      </div>

      {/* Body */}
      <div>
        <div className="row" style={{ gap: 10, marginBottom: 6 }}>
          <div style={{ fontSize: 15, fontWeight: 600, color: 'var(--ink)', letterSpacing: '-0.015em' }}>{r.name}</div>
          <span className="mono" style={{ fontSize: 11, color: 'var(--ink-3)' }}>{r.neighborhood}</span>
        </div>
        <div style={{ fontSize: 13, color: 'var(--ink-2)', lineHeight: 1.5, marginBottom: 10, maxWidth: 620 }}>
          {r.blurb}
        </div>
        <div className="row" style={{ gap: 12, flexWrap: 'wrap' }}>
          <VerifyAggregate verified={r.verified} total={r.total} flagged={r.flagged} compact/>
          <span style={{ color: 'var(--ink-4)' }}>·</span>
          <span className="row" style={{ gap: 4, flexWrap: 'wrap' }}>
            {r.chips.map(c => <span key={c} className="chip sm">{c}</span>)}
          </span>
        </div>
        {r.flagged > 0 && (
          <div style={{
            marginTop: 10, paddingTop: 10, borderTop: '1px dashed var(--stroke)',
            display: 'flex', gap: 8, alignItems: 'center', fontSize: 11.5, color: 'var(--ink-3)',
          }}>
            <span style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--crit)' }}/>
            <span><strong style={{ color: '#b91c1c' }}>Referral path</strong> flagged stale 2× — "piloting fax referrals from primary care"</span>
          </div>
        )}
      </div>

      {/* Right meta */}
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 8, minWidth: 120 }}>
        <span className="mono" style={{ fontSize: 10.5, color: 'var(--ink-3)' }}>{m.label}</span>
        <V2Ico name="arrow-right" size={14} style={{ color: 'var(--ink-3)' }}/>
      </div>
    </a>
  );
}

window.SearchResults = SearchResults;
