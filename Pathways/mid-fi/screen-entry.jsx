// Pathways mid-fi — Entry C · hybrid (free text + AI-extracted tags)
// Left: case sketch (textarea). Right: AI extracts needs/eligibility/constraints.
// Tags are editable. Bottom strip: location + primary action.

function ScreenEntry() {
  const r = useRouter();
  const [text, setText] = useState(SAMPLE_TEXT);
  const [parsed, setParsed] = useState(SAMPLE_PARSED);
  const [reparsing, setReparsing] = useState(false);

  // Simulate "live parse" debounce on edit
  useEffect(() => {
    if (text === SAMPLE_TEXT) return;
    setReparsing(true);
    const t = setTimeout(() => setReparsing(false), 700);
    return () => clearTimeout(t);
  }, [text]);

  const removeTag = (group, idx) => {
    setParsed(p => ({ ...p, [group]: p[group].filter((_, i) => i !== idx) }));
  };

  return (
    <div className="page-enter">
      <Topnav active="cases" />

      {/* Sub-bar with breadcrumb + case meta */}
      <div style={{
        borderBottom: '1px solid var(--stroke)',
        background: 'var(--paper)',
        padding: '12px 32px',
        position: 'sticky', top: 'var(--topnav-h)', zIndex: 9,
      }}>
        <div style={{ maxWidth: 1280, margin: '0 auto' }} className="between">
          <div className="row" style={{ gap: 10 }}>
            <button className="btn ghost sm" onClick={() => r.go({ name: 'home' })}>
              <Ico name="arrow-left" size={13} /> Cases
            </button>
            <span style={{ color: 'var(--ink-4)' }}>/</span>
            <span style={{ fontSize: 13, fontWeight: 500 }}>New case</span>
            <span className="chip sm">draft</span>
            <span className="mono dim" style={{ fontSize: 11 }}>auto-saved · just now</span>
          </div>
          <div className="row" style={{ gap: 8 }}>
            <button className="btn ghost sm">Discard</button>
            <button className="btn sm">Save draft</button>
          </div>
        </div>
      </div>

      <main style={{ maxWidth: 1280, margin: '0 auto', padding: '28px 32px 120px' }}>

        <div style={{ marginBottom: 22 }}>
          <h1 className="title-1" style={{ fontSize: 24 }}>Sketch the case</h1>
          <p className="muted" style={{ marginTop: 4, fontSize: 13.5 }}>
            Plain-language is fine — Pathways extracts what matters. <span className="dim">No patient identifiers.</span>
          </p>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1.15fr 1fr', gap: 20 }}>

          {/* Left — case sketch */}
          <section className="stack-16">
            <div className="card" style={{ padding: 0 }}>
              <div className="between" style={{ padding: '12px 16px', borderBottom: '1px solid var(--stroke)' }}>
                <span className="eyebrow">Case sketch</span>
                <span className="mono dim" style={{ fontSize: 11 }}>{text.length} chars</span>
              </div>
              <textarea
                className="field"
                value={text}
                onChange={e => setText(e.target.value)}
                style={{
                  border: 0, borderRadius: 0, padding: 18,
                  fontSize: 14, minHeight: 280, lineHeight: 1.65,
                  fontFamily: 'var(--font-ui)',
                }}
              />
              <div className="between" style={{ padding: '10px 14px', borderTop: '1px solid var(--stroke)', background: 'var(--paper-2)' }}>
                <div className="row" style={{ gap: 10, fontSize: 11.5, color: 'var(--ink-3)' }}>
                  <span className="row" style={{ gap: 4 }}>
                    <Ico name="shield" size={12} /> No identifiers detected
                  </span>
                  <span>·</span>
                  <span>Voice input available</span>
                </div>
                <div className="row" style={{ gap: 4 }}>
                  <button className="btn ghost sm">Templates</button>
                  <button className="btn ghost sm">Paste referral</button>
                </div>
              </div>
            </div>

            {/* Quick-add tag bar */}
            <div className="card" style={{ padding: 14 }}>
              <div className="row" style={{ gap: 10 }}>
                <span className="eyebrow">Add manually</span>
                <span className="dim" style={{ fontSize: 11.5 }}>if AI missed something</span>
              </div>
              <div className="row" style={{ marginTop: 10, gap: 6, flexWrap: 'wrap' }}>
                {['adult', 'senior', 'youth', 'uninsured', 'newcomer', 'francophone', 'wheelchair', 'crisis-now', 'overnight'].map(t => (
                  <button key={t} className="chip outline" style={{ cursor: 'pointer' }}>+ {t}</button>
                ))}
                <span className="chip outline dim" style={{ cursor: 'pointer' }}>more…</span>
              </div>
            </div>
          </section>

          {/* Right — AI-extracted */}
          <aside className="ai-zone" style={{ padding: 18 }}>
            <div className="between" style={{ marginBottom: 14 }}>
              <AITag>Extracted from your sketch</AITag>
              <span className="row mono" style={{ fontSize: 10.5, color: 'var(--accent-ink)', gap: 5 }}>
                {reparsing ? (
                  <><span style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--accent)', animation: 'pulse 1s ease-in-out infinite' }} /> reparsing…</>
                ) : (
                  <><Ico name="check" size={12} /> ready</>
                )}
              </span>
            </div>

            {GROUPS.map(g => (
              <div key={g.key} style={{ marginBottom: 14 }}>
                <div className="row" style={{ marginBottom: 7, gap: 6 }}>
                  <span className="eyebrow">{g.label}</span>
                  <span className="dim mono" style={{ fontSize: 10.5 }}>{(parsed[g.key] || []).length}</span>
                </div>
                <div className="row" style={{ gap: 5, flexWrap: 'wrap' }}>
                  {(parsed[g.key] || []).map((tag, i) => (
                    <span key={tag.label + i} className="chip" style={{
                      background: 'var(--paper)',
                      borderColor: 'color-mix(in oklch, var(--accent) 22%, var(--stroke))',
                      paddingRight: 4,
                    }}>
                      {tag.label}
                      {tag.conf && tag.conf < 0.85 && (
                        <span className="dim mono" style={{ fontSize: 10, marginLeft: 3 }}>?</span>
                      )}
                      <span className="chip-x" onClick={() => removeTag(g.key, i)} title="Remove">
                        <Ico name="x" size={11} />
                      </span>
                    </span>
                  ))}
                  <button className="chip outline dim" style={{ cursor: 'pointer' }}>
                    <Ico name="plus" size={10} stroke={2} />
                  </button>
                </div>
              </div>
            ))}

            <div className="divider" style={{ margin: '14px 0' }} />

            {/* Reasoning preview */}
            <div className="stack-6">
              <span className="eyebrow">Why these tags</span>
              <p style={{ fontSize: 12.5, lineHeight: 1.55, color: 'var(--ink-2)' }}>
                "<span style={{ color: 'var(--ink)' }}>78-year-old</span>" → <strong>senior</strong>; "<span style={{ color: 'var(--ink)' }}>lives alone</span>" + "<span style={{ color: 'var(--ink)' }}>post-discharge</span>" → <strong>home support</strong>; "<span style={{ color: 'var(--ink)' }}>Cantonese</span>" → <strong>language match</strong>. <span className="dim">Two low-confidence tags marked with "?" — please confirm.</span>
              </p>
              <a style={{ fontSize: 12, color: 'var(--accent-ink)', cursor: 'pointer', fontWeight: 500 }}>Show full extraction →</a>
            </div>
          </aside>
        </div>

        {/* Location row */}
        <div className="card" style={{ padding: 16, marginTop: 20 }}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr 1fr', gap: 18 }}>
            <div className="stack-6">
              <span className="label">Region</span>
              <div className="row" style={{ gap: 6 }}>
                <span className="chip accent">East Toronto OHT</span>
                <button className="btn ghost sm">change</button>
              </div>
            </div>
            <div className="stack-6">
              <span className="label">Postal (optional)</span>
              <input className="field" placeholder="M4K" style={{ width: 120 }} />
            </div>
            <div className="stack-6">
              <span className="label">Travel limit</span>
              <select className="field" defaultValue="20" style={{ width: 130 }}>
                <option value="10">≤ 10 km</option>
                <option value="20">≤ 20 km</option>
                <option value="40">≤ 40 km</option>
                <option value="any">any</option>
              </select>
            </div>
            <div className="stack-6">
              <span className="label">Urgency</span>
              <div className="row" style={{ gap: 4 }}>
                {['Routine', 'This week', 'Crisis'].map((u, i) => (
                  <button key={u} className={`chip ${i === 1 ? 'accent' : ''}`} style={{ cursor: 'pointer' }}>{u}</button>
                ))}
              </div>
            </div>
          </div>
        </div>

      </main>

      {/* Sticky CTA bar */}
      <div style={{
        position: 'sticky', bottom: 0,
        background: 'color-mix(in oklch, var(--paper) 85%, transparent)',
        backdropFilter: 'blur(8px)',
        borderTop: '1px solid var(--stroke)',
        padding: '14px 32px',
      }}>
        <div style={{ maxWidth: 1280, margin: '0 auto' }} className="between">
          <div className="row" style={{ gap: 12, fontSize: 12.5 }}>
            <span className="muted">Ready to search</span>
            <span style={{ color: 'var(--ink-4)' }}>·</span>
            <span className="muted"><strong style={{ color: 'var(--ink)' }}>{Object.values(parsed).flat().length}</strong> tags extracted</span>
            <span style={{ color: 'var(--ink-4)' }}>·</span>
            <span className="muted">East Toronto OHT · ≤ 20 km</span>
          </div>
          <div className="row" style={{ gap: 10 }}>
            <button className="btn">Save & continue later</button>
            <button className="btn accent lg" onClick={() => r.go({ name: 'results' })}>
              Find resources <Ico name="arrow-right" size={14} />
            </button>
          </div>
        </div>
      </div>

      <style>{`@keyframes pulse { 0%,100% { opacity: 1 } 50% { opacity: 0.3 } }`}</style>
    </div>
  );
}

const SAMPLE_TEXT = `78-year-old, post-discharge from East General after CHF episode. Lives alone in 2nd-floor walk-up, daughter 90 minutes away. Limited English — Cantonese first language. Needs home support visits, meal program, and a way to monitor weight daily. ODSP, no private benefits.`;

const SAMPLE_PARSED = {
  needs: [
    { label: 'home support', conf: 0.96 },
    { label: 'meal program', conf: 0.92 },
    { label: 'remote monitoring', conf: 0.74 },
    { label: 'medication review', conf: 0.68 },
  ],
  population: [
    { label: 'senior', conf: 0.99 },
    { label: 'post-discharge', conf: 0.95 },
    { label: 'lives-alone', conf: 0.93 },
  ],
  eligibility: [
    { label: 'ODSP', conf: 0.91 },
    { label: 'no private benefits', conf: 0.88 },
  ],
  preferences: [
    { label: 'Cantonese', conf: 0.97 },
    { label: 'home visit', conf: 0.84 },
  ],
};

const GROUPS = [
  { key: 'needs', label: 'Needs' },
  { key: 'population', label: 'Population' },
  { key: 'eligibility', label: 'Eligibility & coverage' },
  { key: 'preferences', label: 'Language & preferences' },
];

window.ScreenEntry = ScreenEntry;
