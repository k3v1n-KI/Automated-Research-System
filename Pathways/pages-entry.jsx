// Patient-needs entry — 3 variations.

// A · Free-text first (paste from notes), AI parses below in real time.
function EntryA() {
  return (
    <AppFrame active="Find for patient">
      <div style={{ maxWidth: 880, margin: '0 auto' }}>
        <div className="meta" style={{ marginBottom: 4 }}>Step 1 of 2 · No patient names — just structured needs</div>
        <h1 className="title-1" style={{ marginBottom: 16 }}>Tell me about the case</h1>

        <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: 18 }}>
          <div className="col">
            <div className="card" style={{ padding: 0 }}>
              <textarea
                className="field"
                style={{ minHeight: 220, border: 0, fontSize: 14, padding: 14 }}
                defaultValue={"76yo Mandarin-speaking, lives alone in Scarborough, recovering from hip surgery, no family nearby, OHIP only, mobility limited, daughter visits weekly. Needs PSW + meal support."}
              />
              <div className="row" style={{ padding: '8px 12px', borderTop: '1px solid var(--stroke)', background: 'var(--paper-2)' }}>
                <button className="btn sm">🎙 Voice</button>
                <button className="btn sm">＋ Tags</button>
                <span className="nav-spacer" />
                <span className="meta">412 chars</span>
              </div>
            </div>
            <ArtboardNote>↑ Free text · paste-from-notes friendly · voice secondary</ArtboardNote>
          </div>

          <AIZone style={{ padding: 14 }}>
            <div className="eyebrow" style={{ color: 'var(--accent-ink)', marginBottom: 8 }}>What I picked up</div>
            <div className="col" style={{ gap: 12 }}>
              {[
                ['Age', '76', 'good'],
                ['Language', 'Mandarin', 'good'],
                ['Location', 'Scarborough · East Toronto OHT', 'good'],
                ['Insurance', 'OHIP only', 'good'],
                ['Needs', 'PSW · meal support · post-op recovery', 'good'],
                ['Mobility', 'limited', 'good'],
                ['Caregiver', 'daughter, weekly only', 'warn'],
              ].map(([k, v, t]) => (
                <div key={k} className="between" style={{ fontSize: 12 }}>
                  <span className="meta">{k}</span>
                  <span className="row">
                    <span style={{ fontWeight: 500 }}>{v}</span>
                    <span className="chip-x">✕</span>
                  </span>
                </div>
              ))}
            </div>
            <div style={{ borderTop: '1px dashed var(--accent)', margin: '14px -14px 0' }} />
            <div style={{ padding: '10px 0 0' }}>
              <div className="meta" style={{ marginBottom: 6 }}>I might also want to know:</div>
              <div className="row" style={{ flexWrap: 'wrap', gap: 6 }}>
                <span className="chip outline">＋ urgency?</span>
                <span className="chip outline">＋ insurance for meds?</span>
              </div>
            </div>
          </AIZone>
        </div>

        <div className="row" style={{ marginTop: 22, justifyContent: 'flex-end' }}>
          <button className="btn ghost">Save as template</button>
          <button className="btn primary">Find resources →</button>
        </div>
      </div>
    </AppFrame>
  );
}

// B · Tag-only fast path
function EntryB() {
  const groups = {
    'Demographic': ['65+', '18–64', 'Pediatric', 'Mandarin', 'Tagalog', 'Tigrinya', 'Indigenous-led', 'Newcomer'],
    'Need': ['PSW', 'Nursing', 'Mental health', 'Meal support', 'Transport', 'Housing', 'Drug coverage', 'Palliative'],
    'Constraint': ['No OHIP', 'Limited mobility', 'No transport', 'Low income', 'No caregiver', 'Trans-affirming', '24/7'],
  };
  return (
    <AppFrame active="Find for patient">
      <div style={{ maxWidth: 880, margin: '0 auto' }}>
        <div className="meta" style={{ marginBottom: 4 }}>Step 1 of 2 · Tap tags to build the case</div>
        <h1 className="title-1" style={{ marginBottom: 16 }}>Quick-tag the case</h1>

        <div className="card accent" style={{ padding: 12, marginBottom: 16 }}>
          <div className="meta" style={{ marginBottom: 8 }}>your case so far</div>
          <div className="row" style={{ flexWrap: 'wrap', gap: 6 }}>
            {['65+', 'Mandarin', 'Scarborough', 'OHIP only', 'PSW', 'Meal support', 'Limited mobility'].map(t => (
              <span key={t} className="chip accent">{t}<span className="chip-x">✕</span></span>
            ))}
            <span className="chip outline">＋ add more</span>
          </div>
        </div>

        {Object.entries(groups).map(([g, tags]) => (
          <div key={g} style={{ marginBottom: 14 }}>
            <div className="eyebrow" style={{ marginBottom: 8 }}>{g}</div>
            <div className="row" style={{ flexWrap: 'wrap', gap: 6 }}>
              {tags.map(t => (
                <span key={t} className="chip">＋ {t}</span>
              ))}
            </div>
          </div>
        ))}

        <div className="row" style={{ marginTop: 22, justifyContent: 'space-between' }}>
          <button className="btn ghost">Switch to free text →</button>
          <button className="btn primary">Find resources →</button>
        </div>
      </div>
    </AppFrame>
  );
}

// C · Hybrid — text + tag scaffolding side by side, AI mediates
function EntryC() {
  return (
    <AppFrame active="Find for patient">
      <div style={{ maxWidth: 920, margin: '0 auto' }}>
        <div className="meta" style={{ marginBottom: 4 }}>Step 1 of 2 · Mix text and tags however you like</div>
        <h1 className="title-1" style={{ marginBottom: 16 }}>Build the case</h1>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          <div className="card" style={{ padding: 0 }}>
            <div className="meta" style={{ padding: '8px 12px', borderBottom: '1px solid var(--stroke)', background: 'var(--paper-2)' }}>describe in your words</div>
            <textarea
              className="field"
              style={{ minHeight: 200, border: 0, fontSize: 13, padding: 12 }}
              defaultValue="Hip surgery recovery, lives alone, daughter weekly only…"
            />
          </div>

          <div className="card muted" style={{ padding: 12 }}>
            <div className="meta" style={{ marginBottom: 8 }}>or pick from tags</div>
            <div className="col" style={{ gap: 10 }}>
              {[
                ['Age', ['65+', '18–64', 'Pediatric'], 0],
                ['Language', ['Mandarin', 'Tagalog', 'Tigrinya', 'EN/FR'], 0],
                ['Need', ['PSW', 'Nursing', 'Mental health', 'Meal'], 0],
                ['Constraint', ['No OHIP', 'No transport', 'Low income'], 0],
              ].map(([g, opts]) => (
                <div key={g}>
                  <div className="meta" style={{ marginBottom: 4 }}>{g}</div>
                  <div className="row" style={{ flexWrap: 'wrap', gap: 4 }}>
                    {opts.map(o => <span key={o} className="chip">＋ {o}</span>)}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        <AIZone style={{ marginTop: 16, padding: 12 }}>
          <div className="between" style={{ marginBottom: 8 }}>
            <span className="eyebrow" style={{ color: 'var(--accent-ink)' }}>combined case</span>
            <span className="meta">edit any value inline</span>
          </div>
          <div className="row" style={{ flexWrap: 'wrap', gap: 6 }}>
            {['76yo', 'Mandarin', 'Scarborough', 'East Toronto OHT', 'OHIP only', 'PSW', 'Meal support', 'Limited mobility', 'Hip post-op', 'No nearby caregiver'].map(t => (
              <span key={t} className="chip accent">{t}<span className="chip-x">✕</span></span>
            ))}
          </div>
        </AIZone>

        <div className="row" style={{ marginTop: 22, justifyContent: 'flex-end' }}>
          <button className="btn ghost">🎙 Voice (phase 2)</button>
          <button className="btn primary">Find resources →</button>
        </div>
      </div>
    </AppFrame>
  );
}

window.EntryA = EntryA; window.EntryB = EntryB; window.EntryC = EntryC;
