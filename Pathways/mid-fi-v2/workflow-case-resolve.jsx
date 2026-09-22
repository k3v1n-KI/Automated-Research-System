// Pathways v2 — Case resolution → resource capture
// The moment a case closes. Free-text "where did you refer them?"
// becomes a new resource candidate.

function CaseResolve() {
  return (
    <div className="v2">
      <V2Topnav active="cases" />
      <div className="body">
        <main style={{ maxWidth: 1080, margin: '0 auto', padding: '28px 32px 80px' }}>

          {/* Breadcrumb */}
          <div className="row" style={{ gap: 8, marginBottom: 14, fontSize: 12, color: 'var(--ink-3)' }}>
            <a>Cases</a><V2Ico name="chevron-right" size={11}/>
            <a className="mono">c-241 · Senior, post-discharge</a><V2Ico name="chevron-right" size={11}/>
            <span style={{ color: 'var(--ink-2)' }}>Close case</span>
          </div>

          {/* Header */}
          <div style={{ marginBottom: 22 }}>
            <div className="eyebrow" style={{ marginBottom: 8 }}>Closing case · East York</div>
            <h1 style={{ fontSize: 26, fontWeight: 600, letterSpacing: '-0.02em', lineHeight: 1.2, marginBottom: 6 }}>
              Senior, post-discharge, lives alone.
            </h1>
            <p style={{ fontSize: 13, color: 'var(--ink-3)', lineHeight: 1.5 }}>
              Home support · meals · Cantonese · ODSP — opened 4 days ago.
            </p>
          </div>

          {/* Two-up: form on left, capture preview on right */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 380px', gap: 28, alignItems: 'start' }}>

            <section style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>

              {/* Outcome */}
              <div className="card" style={{ padding: 20 }}>
                <div className="eyebrow" style={{ marginBottom: 12 }}>Outcome</div>
                <div className="row" style={{ gap: 8, flexWrap: 'wrap' }}>
                  <span className="chip accent" style={{ padding: '6px 12px', fontSize: 12.5 }}>● Referred</span>
                  <span className="chip" style={{ padding: '6px 12px', fontSize: 12.5 }}>Self-resolved</span>
                  <span className="chip" style={{ padding: '6px 12px', fontSize: 12.5 }}>No good option</span>
                  <span className="chip" style={{ padding: '6px 12px', fontSize: 12.5 }}>Patient declined</span>
                </div>
              </div>

              {/* Where did you refer them */}
              <div className="card" style={{ padding: 20 }}>
                <div className="between" style={{ marginBottom: 10 }}>
                  <span className="eyebrow">Where did you refer them?</span>
                  <span className="mono" style={{ fontSize: 11, color: 'var(--ink-3)' }}>add up to 3 · drag to reorder</span>
                </div>

                {/* Referral 1 — matched against the index */}
                <div style={{
                  padding: 14, border: '1px solid var(--stroke)', borderRadius: 'var(--r-sm)',
                  marginBottom: 10,
                }}>
                  <div className="row" style={{ gap: 10, marginBottom: 8 }}>
                    <V2Ico name="pin" size={13} style={{ color: 'var(--accent)' }}/>
                    <span style={{ fontSize: 13.5, color: 'var(--ink)', fontWeight: 500 }}>
                      East York Home Care Connect
                    </span>
                    <VerifyAggregate verified={6} total={8} flagged={0} compact/>
                    <span style={{ flex: 1 }}/>
                    <a className="muted" style={{ fontSize: 11 }}>Open</a>
                  </div>
                  <textarea className="field" rows={2} placeholder="One line about this referral · what worked, what to watch for"
                    defaultValue="Took the case Tuesday. Coordinator Wendy speaks Cantonese — ask for her."
                    style={{ resize: 'none', fontSize: 13, lineHeight: 1.5 }}/>
                </div>

                {/* Referral 2 — free text, becomes a new resource candidate */}
                <div style={{
                  padding: 14, border: '1px solid color-mix(in srgb, var(--accent) 28%, transparent)',
                  borderRadius: 'var(--r-sm)', background: 'var(--accent-tint)',
                  marginBottom: 10,
                }}>
                  <div className="row" style={{ gap: 10, marginBottom: 8 }}>
                    <V2Ico name="sparkle" size={13} style={{ color: 'var(--accent)' }}/>
                    <input className="field" type="text" defaultValue="Riverdale Meals Co-op"
                      style={{ flex: 1, fontSize: 13.5, fontWeight: 500, padding: '6px 10px', border: 'none', background: 'transparent' }}/>
                  </div>
                  <div style={{ fontSize: 12, color: 'var(--accent-ink)', lineHeight: 1.5, marginBottom: 8 }}>
                    Not in the index. We'll create a candidate resource from your note — other
                    members can verify it from there.
                  </div>
                  <textarea className="field" rows={2}
                    defaultValue="Volunteer-run meal delivery, Tuesdays + Fridays. Coordinator: 416-555-0188. They took on a senior on ODSP without a hitch — no paperwork."
                    style={{ resize: 'none', fontSize: 13, lineHeight: 1.5 }}/>
                </div>

                <a className="row" style={{ gap: 7, fontSize: 13, color: 'var(--accent-ink)', fontWeight: 500, marginTop: 6 }}>
                  <V2Ico name="plus" size={12} stroke={2}/> Add another
                </a>
              </div>

              {/* Notes for next time */}
              <div className="card" style={{ padding: 20 }}>
                <div className="eyebrow" style={{ marginBottom: 10 }}>One-line note · for the next member in your shoes</div>
                <textarea className="field" rows={3}
                  defaultValue="Home Care Connect intake closes at 4pm. If you call after, the answering service still books for next-day — say it's an OHIP referral and they'll triage same week."
                  style={{ resize: 'none', fontSize: 13, lineHeight: 1.55 }}/>
                <div className="row" style={{ marginTop: 10, gap: 12, fontSize: 11.5, color: 'var(--ink-3)' }}>
                  <span>Attaches to <strong style={{ color: 'var(--ink-2)', fontWeight: 500 }}>East York Home Care Connect</strong> as a verified note.</span>
                </div>
              </div>

              {/* Submit */}
              <div className="between" style={{ paddingTop: 10 }}>
                <label className="row" style={{ gap: 8, fontSize: 12.5, color: 'var(--ink-2)', cursor: 'pointer' }}>
                  <span style={{
                    width: 30, height: 18, background: 'var(--paper-3)', borderRadius: 999,
                    position: 'relative', border: '1px solid var(--stroke-2)',
                  }}>
                    <span style={{ position: 'absolute', top: 1, left: 1, width: 14, height: 14, borderRadius: '50%', background: 'var(--paper)', border: '1px solid var(--stroke-2)' }}/>
                  </span>
                  <span>Submit anonymously</span>
                  <span style={{ color: 'var(--ink-4)' }}>
                    (off — your name shows on the new resource as "Added by Rita · AOHT")
                  </span>
                </label>
                <div className="row" style={{ gap: 8 }}>
                  <button className="btn">Save draft</button>
                  <button className="btn accent">Close case · add 1 resource</button>
                </div>
              </div>
            </section>

            {/* Right preview rail — what becomes of your free-text referral */}
            <aside style={{ position: 'sticky', top: 12, display: 'flex', flexDirection: 'column', gap: 14 }}>
              <div className="card" style={{ padding: 16 }}>
                <div className="eyebrow" style={{ marginBottom: 10 }}>Preview · new resource</div>
                <div style={{ marginBottom: 12 }}>
                  <div style={{ fontSize: 14, fontWeight: 600, color: 'var(--ink)', letterSpacing: '-0.012em', marginBottom: 4 }}>
                    Riverdale Meals Co-op
                  </div>
                  <div className="row" style={{ gap: 6, marginBottom: 8 }}>
                    <VerifyAggregate verified={2} total={6} flagged={0} compact/>
                  </div>
                  <div style={{ fontSize: 12, color: 'var(--ink-3)', lineHeight: 1.5 }}>
                    Volunteer meal delivery · Tue & Fri.
                  </div>
                </div>

                <div style={{ display: 'flex', flexDirection: 'column', gap: 8, borderTop: '1px solid var(--stroke)', paddingTop: 12 }}>
                  <PreviewField label="Name"      value="Riverdale Meals Co-op"      state="verified-fresh"/>
                  <PreviewField label="Phone"     value="(416) 555-0188"             state="verified-fresh"/>
                  <PreviewField label="Schedule"  value="Tuesdays + Fridays"         state="verified-fresh"/>
                  <PreviewField label="Eligibility" value="No paperwork · senior on ODSP welcomed" state="verified-fresh" small/>
                  <PreviewField label="Address"   value="needs filling in"           state="unknown"/>
                  <PreviewField label="Hours"     value="needs filling in"           state="unknown"/>
                </div>

                <div style={{
                  marginTop: 12, padding: '10px 12px', borderRadius: 'var(--r-sm)',
                  background: 'var(--paper-2)', border: '1px solid var(--stroke)',
                  fontSize: 11.5, color: 'var(--ink-3)', lineHeight: 1.5,
                }}>
                  We'll try to enrich the rest from a web look-up after you close. Anything
                  AI fills in shows up as <span className="mono" style={{ fontSize: 10, color: 'var(--accent-ink)' }}>⬡ AI extracted</span>, never <span className="mono" style={{ fontSize: 10, color: '#15803d' }}>✓ Verified</span>.
                </div>
              </div>

              <div className="card muted" style={{ padding: 14, background: 'var(--paper-2)' }}>
                <div className="eyebrow" style={{ marginBottom: 6 }}>Why this matters</div>
                <p style={{ fontSize: 11.5, color: 'var(--ink-3)', lineHeight: 1.55, margin: 0 }}>
                  Volunteer-run, off-the-radar resources are the ones AI can't find. By naming
                  yours when you close a case, you save the next member a phone call — and the
                  resource keeps getting confirmed every time it's referred to.
                </p>
              </div>
            </aside>

          </div>
        </main>
      </div>
    </div>
  );
}

function PreviewField({ label, value, state, small }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '88px 1fr auto', gap: 10, alignItems: 'flex-start' }}>
      <div style={{
        fontSize: 10, color: 'var(--ink-3)', fontFamily: 'var(--font-mono)',
        textTransform: 'uppercase', letterSpacing: '0.04em', paddingTop: 3,
      }}>{label}</div>
      <div style={{ fontSize: 12, color: state === 'unknown' ? 'var(--ink-4)' : 'var(--ink-2)', lineHeight: 1.45, fontStyle: state === 'unknown' ? 'italic' : 'normal' }}>
        {value}
      </div>
      <VerifyBadge state={state} compact/>
    </div>
  );
}

window.CaseResolve = CaseResolve;
