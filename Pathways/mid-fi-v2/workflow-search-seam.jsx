// Pathways v2 — Seamless input · search-or-Ask
// The same input box. As you type, AI parses → constraint chips,
// live top matches appear below, and the primary CTA flips based on match quality.
//
// Three artboards on one canvas:
//   SearchSeam · strong-match state (Open is primary, Ask is secondary)
//   SearchSeam · weak-match state   (Ask is primary, results are de-emphasized)
//   SearchSeamMechanism             (explainer card: how the CTA flip works)

function SearchSeamMechanism() {
  const rules = [
    { range: 'fit ≥ 70 · no flag', cta: 'Open top match', flip: 'Ask is a quiet link below', tone: 'good' },
    { range: 'fit 40–69 · or flagged', cta: 'See all results', flip: '"Post as Ask" appears as a peer button', tone: 'warn' },
    { range: 'fit < 40 · or 0 results', cta: 'Post as Ask', flip: 'Weak matches still shown as "close but no" rail', tone: 'crit' },
  ];
  return (
    <div className="v2" style={{ padding: 28, height: '100%', background: 'var(--paper)', overflow: 'auto' }}>
      <div className="eyebrow" style={{ marginBottom: 8 }}>Mechanism · search ↔ Ask</div>
      <h2 style={{ fontSize: 22, fontWeight: 600, letterSpacing: '-0.022em', marginBottom: 4 }}>
        One input. The system chooses what to do with it.
      </h2>
      <p style={{ fontSize: 13, color: 'var(--ink-3)', lineHeight: 1.5, marginBottom: 18, maxWidth: 600 }}>
        Members describe a need in plain language. We don't ask them to pick Search vs. Ask up front — the
        primary action surfaces based on whether matches exist. The other option is always one tap away.
      </p>

      <div style={{ display: 'grid', gridTemplateColumns: '160px 1fr 1fr', gap: 12, marginBottom: 12, alignItems: 'center' }}>
        <span className="eyebrow">Match quality</span>
        <span className="eyebrow">Primary CTA</span>
        <span className="eyebrow">What else shows</span>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        {rules.map((r, i) => (
          <div key={i} style={{
            display: 'grid', gridTemplateColumns: '160px 1fr 1fr', gap: 12,
            padding: '14px 16px', border: '1px solid var(--stroke)', borderRadius: 'var(--r-md)',
            background: 'var(--paper)', alignItems: 'center',
          }}>
            <div className="row" style={{ gap: 8 }}>
              <span style={{ width: 6, height: 6, borderRadius: '50%',
                background: r.tone === 'good' ? 'var(--good)' : r.tone === 'warn' ? 'var(--warn)' : 'var(--crit)' }}/>
              <span className="mono" style={{ fontSize: 11.5, color: 'var(--ink-2)' }}>{r.range}</span>
            </div>
            <div style={{ fontSize: 13, color: 'var(--ink)', fontWeight: 500, letterSpacing: '-0.008em' }}>
              {r.cta}
            </div>
            <div style={{ fontSize: 12, color: 'var(--ink-3)', lineHeight: 1.5 }}>
              {r.flip}
            </div>
          </div>
        ))}
      </div>

      <div style={{
        marginTop: 14, padding: '12px 14px', borderRadius: 'var(--r-md)',
        background: 'var(--paper-2)', border: '1px solid var(--stroke)',
      }}>
        <div className="eyebrow" style={{ marginBottom: 6 }}>Keyboard shortcuts everywhere</div>
        <div className="row" style={{ gap: 12, flexWrap: 'wrap', fontSize: 12, color: 'var(--ink-2)' }}>
          <span className="row" style={{ gap: 6 }}><span className="kbd">↵</span> primary action</span>
          <span className="row" style={{ gap: 6 }}><span className="kbd">⌘</span><span className="kbd">↵</span> see all results</span>
          <span className="row" style={{ gap: 6 }}><span className="kbd">⌘</span><span className="kbd">P</span> post as Ask</span>
          <span className="row" style={{ gap: 6 }}><span className="kbd">↑</span><span className="kbd">↓</span> navigate matches</span>
        </div>
      </div>
    </div>
  );
}

// ────────────────────────────────────────────────
// The live search state — parameterized by match quality
// ────────────────────────────────────────────────
function SearchSeam({ mode = 'strong' }) {
  const isStrong = mode === 'strong';
  const isWeak   = mode === 'weak';

  // Query + extracted chips
  const query = isStrong
    ? 'Same-day MH walk-in for an adult without OHIP, East Toronto'
    : '78yo Cantonese senior, weekend caregiver respite, dementia — East York';
  const chips = isStrong
    ? ['MH', 'walk-in', 'uninsured', 'East Toronto']
    : ['senior · dementia', 'caregiver respite', 'weekend', 'Cantonese', 'East York'];

  // Results — strong has Across Health at 92, weak has best at 38
  const results = isStrong ? SEARCH_RESULTS.slice(0, 3) : [
    { id: 'r-1', name: 'Eastside Caregiver Support Network', neighborhood: 'East York',
      blurb: 'Weekday peer groups for dementia caregivers. No weekend respite — listed by mistake?',
      chips: ['caregiver', 'support-group'], verified: 3, total: 8, flagged: 0,
      score: 38, scoreLabel: 'Partial · no respite' },
    { id: 'r-2', name: 'Toronto Dementia Helpline', neighborhood: 'Province-wide',
      blurb: 'Phone information line. Useful for navigation, not a respite resource itself.',
      chips: ['info', 'helpline'], verified: 4, total: 6, flagged: 0,
      score: 34, scoreLabel: 'Weak · phone only' },
  ];

  return (
    <div className="v2">
      <V2Topnav active="cases" />
      <div className="body">
        <main style={{ maxWidth: 880, margin: '0 auto', padding: '40px 32px 80px' }}>

          {/* Live search surface — the dark card from home, expanded */}
          <div style={{
            background: 'var(--ink)', color: 'var(--paper)',
            borderRadius: 'var(--r-lg)', padding: '20px 22px 22px',
            boxShadow: 'var(--shadow-md)',
            marginBottom: 14,
          }}>
            {/* Top label */}
            <div className="row" style={{ gap: 8, marginBottom: 12, opacity: 0.55 }}>
              <V2Ico name="search" size={13} stroke={1.8}/>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10.5, textTransform: 'uppercase', letterSpacing: '0.08em' }}>
                Find a service · or post an Ask
              </span>
              <span style={{ flex: 1 }}/>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, opacity: 0.65 }}>
                AI parsed your case · tap a chip to refine
              </span>
            </div>

            {/* Typed query */}
            <div style={{
              fontSize: 20, fontWeight: 500, letterSpacing: '-0.014em',
              color: 'rgb(255 255 255 / 0.95)', lineHeight: 1.35,
              marginBottom: 12,
            }}>
              {query}<span style={{ opacity: 0.55, marginLeft: 1 }}>│</span>
            </div>

            {/* Extracted chips */}
            <div className="row" style={{ gap: 6, flexWrap: 'wrap', marginBottom: 14 }}>
              {chips.map(c => (
                <span key={c} style={{
                  padding: '3px 9px', fontSize: 11.5,
                  background: 'rgb(255 255 255 / 0.08)', border: '1px solid rgb(255 255 255 / 0.16)',
                  color: 'rgb(255 255 255 / 0.85)', borderRadius: 999, fontWeight: 500,
                }}>{c}</span>
              ))}
              <span style={{
                padding: '3px 9px', fontSize: 11.5,
                background: 'transparent', border: '1px dashed rgb(255 255 255 / 0.22)',
                color: 'rgb(255 255 255 / 0.55)', borderRadius: 999,
              }}>+ refine</span>
            </div>

            {/* Live matches — inline */}
            <div style={{
              background: 'rgb(255 255 255 / 0.04)',
              border: '1px solid rgb(255 255 255 / 0.08)',
              borderRadius: 'var(--r-sm)',
              overflow: 'hidden',
            }}>
              <div style={{
                padding: '10px 14px', borderBottom: '1px solid rgb(255 255 255 / 0.08)',
                display: 'flex', justifyContent: 'space-between', alignItems: 'center',
              }}>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10.5, textTransform: 'uppercase',
                  letterSpacing: '0.08em', color: isStrong ? '#86efac' : '#fbbf24' }}>
                  {isStrong ? 'Strong matches' : 'Best matches we have · weak'}
                </span>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10.5, color: 'rgb(255 255 255 / 0.55)' }}>
                  {isStrong ? '↑↓ navigate · ↵ open' : 'AI confidence below 40 — consider posting as Ask'}
                </span>
              </div>

              {results.map((r, i) => (
                <div key={r.id} style={{
                  padding: '12px 14px',
                  borderBottom: i < results.length - 1 ? '1px solid rgb(255 255 255 / 0.06)' : 0,
                  display: 'grid', gridTemplateColumns: '38px 1fr auto',
                  gap: 12, alignItems: 'center',
                  background: i === 0 && isStrong ? 'rgb(124 58 237 / 0.15)' : 'transparent',
                  cursor: 'pointer',
                }}>
                  <div style={{
                    width: 34, height: 34, borderRadius: 'var(--r-sm)',
                    background: i === 0 && isStrong ? 'var(--accent)' : 'rgb(255 255 255 / 0.06)',
                    color: i === 0 && isStrong ? 'white' : 'rgb(255 255 255 / 0.85)',
                    display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
                  }}>
                    <span style={{ fontSize: 13, fontWeight: 600, letterSpacing: '-0.02em', lineHeight: 1 }}>{r.score}</span>
                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: 8, opacity: 0.7, marginTop: 1 }}>fit</span>
                  </div>
                  <div>
                    <div style={{ fontSize: 13.5, color: 'rgb(255 255 255 / 0.95)', fontWeight: 500, marginBottom: 2, letterSpacing: '-0.008em' }}>
                      {r.name}
                    </div>
                    <div style={{ fontSize: 11.5, color: 'rgb(255 255 255 / 0.55)', lineHeight: 1.4 }}>
                      {r.neighborhood} · {r.verified}/{r.total} verified{r.flagged ? ` · ${r.flagged} flagged` : ''}
                    </div>
                  </div>
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10.5, color: 'rgb(255 255 255 / 0.4)' }}>
                    {i === 0 && isStrong ? <span className="kbd" style={{ background: 'rgb(255 255 255 / 0.1)', borderColor: 'rgb(255 255 255 / 0.18)', color: 'rgb(255 255 255 / 0.85)' }}>↵</span> : ''}
                  </span>
                </div>
              ))}
            </div>

            {/* Dual CTA — primary flips on mode */}
            <div className="row" style={{ marginTop: 14, gap: 10, justifyContent: 'space-between' }}>
              <span style={{ fontSize: 11.5, opacity: 0.55, fontFamily: 'var(--font-mono)' }}>
                {isStrong ? '3 strong matches in your region · 14 partial' : 'No close matches · 2 partial worth a look'}
              </span>
              <div className="row" style={{ gap: 8 }}>
                {isStrong ? (
                  <>
                    <a style={{
                      fontSize: 12, color: 'rgb(255 255 255 / 0.7)', fontWeight: 500,
                      padding: '7px 12px', cursor: 'pointer',
                    }}>
                      Post as Ask instead
                    </a>
                    <a style={{
                      display: 'inline-flex', alignItems: 'center', gap: 8, padding: '7px 14px',
                      background: 'var(--paper)', color: 'var(--ink)',
                      borderRadius: 'var(--r-sm)', fontWeight: 500, fontSize: 13,
                    }}>
                      See all 17 results <span className="kbd" style={{ background: 'var(--paper-2)' }}>⌘↵</span>
                    </a>
                  </>
                ) : (
                  <>
                    <a style={{
                      fontSize: 12, color: 'rgb(255 255 255 / 0.7)', fontWeight: 500,
                      padding: '7px 12px', cursor: 'pointer',
                    }}>
                      See partial matches anyway
                    </a>
                    <a style={{
                      display: 'inline-flex', alignItems: 'center', gap: 8, padding: '7px 14px',
                      background: 'var(--accent)', color: 'white',
                      borderRadius: 'var(--r-sm)', fontWeight: 500, fontSize: 13,
                    }}>
                      Post as Ask <V2Ico name="arrow-right" size={13}/>
                    </a>
                  </>
                )}
              </div>
            </div>
          </div>

          {/* Below the search — preview of what posting an Ask looks like, in the weak case;
              or quick context in the strong case */}
          {isWeak && (
            <div className="card" style={{
              padding: 18, background: 'var(--accent-tint)',
              borderColor: 'color-mix(in srgb, var(--accent) 22%, transparent)',
            }}>
              <div className="row" style={{ gap: 7, marginBottom: 8 }}>
                <V2Ico name="corner-arrow" size={13} stroke={1.7} style={{ color: 'var(--accent)' }}/>
                <span className="eyebrow" style={{ color: 'var(--accent-ink)' }}>If you Post as Ask · preview</span>
              </div>
              <p style={{ fontSize: 14, color: 'var(--ink)', lineHeight: 1.45, marginBottom: 10 }}>
                78yo Cantonese senior, weekend caregiver respite, dementia — East York
              </p>
              <div className="row" style={{ gap: 12, flexWrap: 'wrap', fontSize: 11.5, color: 'var(--ink-3)' }}>
                <span>Posted to <strong style={{ color: 'var(--ink-2)', fontWeight: 500 }}>East Toronto OHT · 312 members</strong></span>
                <span>·</span>
                <span>Visible for <strong style={{ color: 'var(--ink-2)', fontWeight: 500 }}>7 days</strong></span>
                <span>·</span>
                <span>Posts as <strong style={{ color: 'var(--ink-2)', fontWeight: 500 }}>Rita · AOHT</strong> (toggle anonymous before sending)</span>
              </div>
            </div>
          )}

          {isStrong && (
            <div className="card muted" style={{ padding: 14, background: 'var(--paper-2)' }}>
              <div className="row" style={{ gap: 8, fontSize: 12, color: 'var(--ink-3)', lineHeight: 1.5 }}>
                <V2Ico name="sparkle" size={12} style={{ color: 'var(--accent)', flexShrink: 0, marginTop: 2 }}/>
                <span>
                  You'll see partial matches and an option to <strong style={{ color: 'var(--ink-2)', fontWeight: 500 }}>Post as Ask</strong> on
                  the next page too — nothing here is one-way.
                </span>
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}

function SearchSeamStrong() { return <SearchSeam mode="strong"/>; }
function SearchSeamWeak()   { return <SearchSeam mode="weak"/>; }

Object.assign(window, { SearchSeam, SearchSeamStrong, SearchSeamWeak, SearchSeamMechanism });
