// Hi-fi case resolve — Rita closes case c-240 and records where she referred.
// Free-text outside the index becomes a new resource candidate.

function CaseResolve() {
  const [referredTo, setReferredTo] = useState('Algoma Family Services · Walk-in Counselling');
  const [outcome, setOutcome] = useState('referred');
  const [notes, setNotes] = useState('Walked in with grandparent Tuesday at 11:15 am — seen within 25 min. Follow-up offered for next week.');
  const [verifyHours, setVerifyHours] = useState(true);
  const [submitted, setSubmitted] = useState(false);
  const [showSuggest, setShowSuggest] = useState(false);

  const suggestions = SEARCH_RESULTS.filter(r =>
    r.name.toLowerCase().includes(referredTo.toLowerCase()) ||
    referredTo === ''
  );

  const isInIndex = SEARCH_RESULTS.some(r => r.name === referredTo);

  const submit = () => {
    setSubmitted(true);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  if (submitted) {
    return (
      <div className="v2">
        <Topnav active="cases" />
        <div className="body page-in">
          <main style={{ maxWidth: 720, margin: '0 auto', padding: '88px 32px 80px' }}>
            <div style={{
              width: 56, height: 56, borderRadius: '50%',
              background: '#ecfdf5', border: '1px solid #bbf7d0',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              marginBottom: 22,
            }}>
              <Ico name="check" size={26} stroke={2.2} style={{ color: '#15803d' }}/>
            </div>
            <h1 style={{ fontSize: 30, letterSpacing: '-0.025em', marginBottom: 8 }}>Case closed.</h1>
            <p style={{ fontSize: 16, color: 'var(--ink-3)', lineHeight: 1.55, marginBottom: 28 }}>
              Thanks, Rita. Your contributions just made the next AOHT member's search a little sharper.
            </p>
            <div className="stack-12">
              <ContribLine glyph="✓"
                primary="Confirmed Hours for Algoma Family Services Walk-in"
                secondary="That field is now Verified · just now — promoted in search ranking." />
              {!isInIndex && (
                <ContribLine glyph="⬡"
                  primary={`Seeded new resource candidate: "${referredTo}"`}
                  secondary="AI will enrich the fields overnight; AOHT members will verify in passing." />
              )}
              <ContribLine glyph="↳"
                primary="Resolved your Ask · a-103"
                secondary="Other watchers will see how this case landed." />
            </div>
            <div className="row" style={{ gap: 10, marginTop: 36 }}>
              <a className="btn accent" href="../Pathways Hi-Fi.html">Back to Cases</a>
              <a className="btn" href="resource.html">Open the resource</a>
              <a className="btn ghost" href="asks.html">See your Ask</a>
            </div>
          </main>
        </div>
      </div>
    );
  }

  return (
    <div className="v2">
      <Topnav active="cases" />
      <div className="body page-in">
        <main style={{ maxWidth: 880, margin: '0 auto', padding: '28px 32px 80px' }}>

          <div className="row" style={{ gap: 6, marginBottom: 14, fontSize: 12.5, color: 'var(--ink-3)' }}>
            <a href="../Pathways Hi-Fi.html" style={{ color: 'inherit' }}>Cases</a>
            <Ico name="chevron-right" size={11}/>
            <a href="search.html" style={{ color: 'inherit' }}>c-240 · Adolescent, self-harm risk</a>
            <Ico name="chevron-right" size={11}/>
            <span style={{ color: 'var(--ink)' }}>Close case</span>
          </div>

          <PhaseRail current="close" />

          <div style={{ marginBottom: 22 }}>
            <div className="eyebrow" style={{ marginBottom: 8 }}>Close case c-240</div>
            <h1 style={{ fontSize: 28, letterSpacing: '-0.022em', marginBottom: 6 }}>Where did you refer them?</h1>
            <p style={{ fontSize: 14, color: 'var(--ink-3)', lineHeight: 1.55, maxWidth: 600 }}>
              Doing your work seeds the index. Free-text outside the index becomes a candidate other AOHT members can verify.
            </p>
          </div>

          {/* Outcome */}
          <div className="card" data-tour="resolve-outcome" style={{ padding: 22, marginBottom: 14 }}>
            <div className="eyebrow" style={{ marginBottom: 10 }}>Outcome</div>
            <div className="row" style={{ gap: 8, flexWrap: 'wrap' }}>
              {[
                { id: 'referred', label: 'Referred to a service' },
                { id: 'self',     label: 'Patient self-managed' },
                { id: 'no-fit',   label: 'Nothing was a fit' },
                { id: 'declined', label: 'Patient declined' },
              ].map(o => (
                <button key={o.id} onClick={() => setOutcome(o.id)}
                  className="chip" style={{
                    border: 'none', cursor: 'pointer', padding: '6px 14px',
                    background: outcome === o.id ? 'var(--ink)' : 'var(--paper-2)',
                    color: outcome === o.id ? 'var(--paper)' : 'var(--ink-2)',
                    borderRadius: 999, fontSize: 12.5,
                  }}>{o.label}</button>
              ))}
            </div>
          </div>

          {/* Referred to */}
          {outcome === 'referred' && (
            <div className="card" style={{ padding: 22, marginBottom: 14 }}>
              <label className="eyebrow" style={{ marginBottom: 10, display: 'block' }}>Referred to</label>
              <div style={{ position: 'relative' }}>
                <input
                  className="field"
                  value={referredTo}
                  onChange={e => { setReferredTo(e.target.value); setShowSuggest(true); }}
                  onFocus={() => setShowSuggest(true)}
                  onBlur={() => setTimeout(() => setShowSuggest(false), 150)}
                  placeholder="Start typing a service name…"
                  style={{ fontSize: 14, padding: '11px 40px 11px 13px' }}
                />
                <div style={{ position: 'absolute', right: 8, top: '50%', transform: 'translateY(-50%)' }}>
                  <MicMount value={referredTo} setValue={(v) => { setReferredTo(v); setShowSuggest(true); }}
                    placement="inline"
                    samples={['Algoma Family Services — Wellington site']} />
                </div>
                {showSuggest && referredTo && suggestions.length > 0 && (
                  <div className="card" style={{
                    position: 'absolute', top: 'calc(100% + 4px)', left: 0, right: 0,
                    padding: 6, boxShadow: 'var(--shadow-md)', zIndex: 5,
                  }}>
                    {suggestions.slice(0, 4).map(s => (
                      <div key={s.id}
                        onMouseDown={() => { setReferredTo(s.name); setShowSuggest(false); }}
                        style={{
                          padding: '8px 10px', borderRadius: 'var(--r-sm)', cursor: 'pointer',
                          fontSize: 13,
                        }}
                        onMouseEnter={e => e.currentTarget.style.background = 'var(--paper-2)'}
                        onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
                      >
                        <div style={{ fontWeight: 500 }}>{s.name}</div>
                        <div className="mono" style={{ fontSize: 10.5, color: 'var(--ink-3)' }}>{s.neighborhood}</div>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* In-index banner */}
              {referredTo && !isInIndex && (
                <div style={{
                  marginTop: 12, padding: 12,
                  background: 'var(--accent-tint)', border: '1px dashed var(--stroke-2)', borderRadius: 'var(--r-sm)',
                  display: 'flex', alignItems: 'center', gap: 10,
                }}>
                  <Ico name="sparkle" size={14} style={{ color: 'var(--accent)' }}/>
                  <div style={{ flex: 1, fontSize: 12.5, color: 'var(--ink-2)', lineHeight: 1.45 }}>
                    <strong>Not in the index yet.</strong> We'll add "{referredTo}" as a candidate. AI will pull a phone / address / hours overnight, and AOHT members verify in passing.
                  </div>
                </div>
              )}
              {referredTo && isInIndex && (
                <div style={{
                  marginTop: 12, padding: 12,
                  background: '#ecfdf5', border: '1px solid #bbf7d0', borderRadius: 'var(--r-sm)',
                  display: 'flex', alignItems: 'center', gap: 10,
                }}>
                  <Ico name="check" size={13} stroke={2.2} style={{ color: '#15803d' }}/>
                  <div style={{ flex: 1, fontSize: 12.5, color: 'var(--ink-2)', lineHeight: 1.45 }}>
                    Found in your index. <a href="resource.html" className="ilink">Open the resource</a>.
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Notes */}
          <div className="card" style={{ padding: 22, marginBottom: 14 }}>
            <label className="eyebrow" style={{ marginBottom: 10, display: 'block' }}>What happened (one line)</label>
            <div style={{ position: 'relative' }}>
              <textarea
                className="field"
                rows={3}
                value={notes}
                onChange={e => setNotes(e.target.value)}
                placeholder="Two sentences max — what worked, what didn't. Other AOHT members read this when they hit a similar case."
                style={{ resize: 'vertical', fontSize: 13.5, lineHeight: 1.55, paddingRight: 40 }}
              />
              <MicMount value={notes} setValue={setNotes} samples={[
                'Phoned ahead, intake worker called back same day, family declined transport but agreed to virtual.',
                'Walked in at 10am, seen by 11:15. No OHIP not an issue.',
              ]} />
            </div>
            <div className="row" style={{ marginTop: 8, gap: 6, fontSize: 11, color: 'var(--ink-3)' }}>
              <Ico name="people" size={11}/>
              <span>Visible to AOHT members · attributed to <strong style={{ color: 'var(--ink-2)', fontWeight: 500 }}>Rita · AOHT</strong></span>
            </div>
          </div>

          {/* Drive-by verification */}
          {outcome === 'referred' && isInIndex && (
            <div className="card" data-tour="resolve-driveby" style={{ padding: 22, marginBottom: 14, background: 'var(--paper-2)' }}>
              <div className="eyebrow" style={{ marginBottom: 10 }}>While you're here…</div>
              <label className="row" style={{ gap: 10, alignItems: 'flex-start', cursor: 'pointer' }}>
                <input type="checkbox" checked={verifyHours}
                  onChange={e => setVerifyHours(e.target.checked)}
                  style={{ marginTop: 3 }} />
                <div>
                  <div style={{ fontSize: 13, color: 'var(--ink)', marginBottom: 2 }}>
                    Confirm <strong>Hours</strong> — "Mon · Wed · Fri — 9 am to 4 pm"
                  </div>
                  <div className="mono" style={{ fontSize: 10.5, color: 'var(--ink-3)' }}>
                    Confirmed 2d ago by Devon · we'll mark it fresh again.
                  </div>
                </div>
              </label>
            </div>
          )}

          {/* Submit */}
          <div className="between" style={{ marginTop: 22 }}>
            <a href="../Pathways Hi-Fi.html" className="btn ghost">← Cancel</a>
            <div className="row" style={{ gap: 10 }}>
              <button className="btn" onClick={submit}>Close without sharing</button>
              <button className="btn accent" onClick={submit}>Close case + share →</button>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}

function ContribLine({ glyph, primary, secondary }) {
  return (
    <div className="card" style={{ padding: 14, display: 'grid', gridTemplateColumns: '28px 1fr', gap: 12, alignItems: 'flex-start' }}>
      <span style={{
        width: 28, height: 28, borderRadius: '50%',
        background: 'var(--accent-soft)', color: 'var(--accent-ink)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        fontSize: 13, fontWeight: 600,
      }}>{glyph}</span>
      <div>
        <div style={{ fontSize: 13.5, fontWeight: 500, marginBottom: 2 }}>{primary}</div>
        <div style={{ fontSize: 12, color: 'var(--ink-3)', lineHeight: 1.5 }}>{secondary}</div>
      </div>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById('root')).render(<CaseResolve />);
