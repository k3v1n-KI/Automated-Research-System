// Pathways v2 — Resource Detail
// Field-level verification state. Inline ✓ / ⚠ per field.
// Chain-of-custody sidebar with origin Ask + history.
// Exported twice: ResourceDetail (clean) and ResourceDetailWithFlag (sheet open).

function ResourceDetail({ flagOpen = false, flagField = 'referral' }) {
  return (
    <div className="v2" style={{ position: 'relative' }}>
      <V2Topnav active="cases" />
      <div className="body">
        <main style={{ maxWidth: 1180, margin: '0 auto', padding: '28px 32px 80px' }}>

          {/* Breadcrumb */}
          <div className="row" style={{ gap: 8, marginBottom: 14, fontSize: 12, color: 'var(--ink-3)' }}>
            <a>Cases</a><V2Ico name="chevron-right" size={11}/>
            <a>Search · MH walk-in</a><V2Ico name="chevron-right" size={11}/>
            <span style={{ color: 'var(--ink-2)' }}>Across Health · MH Walk-in</span>
          </div>

          {/* Header */}
          <div style={{ marginBottom: 22 }}>
            <div className="row" style={{ gap: 10, marginBottom: 6 }}>
              <span className="eyebrow">Resource · East York</span>
              <VerifyAggregate verified={5} total={8} flagged={1} compact/>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 24 }}>
              <div>
                <h1 style={{ fontSize: 28, fontWeight: 600, letterSpacing: '-0.022em', lineHeight: 1.15, marginBottom: 6 }}>
                  {RESOURCE.name}
                </h1>
                <p style={{ fontSize: 14, color: 'var(--ink-2)', lineHeight: 1.5, maxWidth: 640 }}>
                  {RESOURCE.blurb}
                </p>
              </div>
              <div className="row" style={{ gap: 8, flexShrink: 0 }}>
                <button className="btn">Save to case</button>
                <button className="btn">Share</button>
                <button className="btn accent">Refer →</button>
              </div>
            </div>
          </div>

          {/* 2-col: fields + chain of custody */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 320px', gap: 28, alignItems: 'start' }}>

            {/* Fields */}
            <section className="card" style={{ padding: 0 }}>
              <div style={{ padding: '14px 18px', borderBottom: '1px solid var(--stroke)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span className="eyebrow">Fields · tap ✓ or ⚠ on any row</span>
                <span className="mono" style={{ fontSize: 11, color: 'var(--ink-3)' }}>5 / 8 verified · 1 flagged · 2 AI-only</span>
              </div>
              {RESOURCE.fields.map((f, i) => (
                <FieldRow key={f.key} f={f} highlight={flagOpen && f.key === flagField}/>
              ))}
            </section>

            {/* Chain of custody */}
            <aside style={{ display: 'flex', flexDirection: 'column', gap: 12, position: 'sticky', top: 12 }}>
              <div className="card" style={{ padding: 16 }}>
                <div className="eyebrow" style={{ marginBottom: 10 }}>Chain of custody</div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                  {RESOURCE.history.map((h, i) => <HistoryItem key={i} h={h}/>)}
                </div>
              </div>

              <div className="card muted" style={{ padding: 14, background: 'var(--paper-2)' }}>
                <div className="eyebrow" style={{ marginBottom: 6 }}>Names + anonymity</div>
                <p style={{ fontSize: 11.5, color: 'var(--ink-3)', lineHeight: 1.5, margin: 0 }}>
                  Your flags show your name by default — it carries more weight when colleagues see who confirmed.
                  Flip <em style={{ fontStyle: 'normal', color: 'var(--ink-2)', fontWeight: 500 }}>Submit anonymously</em> on any single flag,
                  or change the default in settings.
                </p>
              </div>
            </aside>
          </div>
        </main>
      </div>

      {/* Flag sheet overlay */}
      {flagOpen && <FlagSheet field={RESOURCE.fields.find(f => f.key === flagField)}/>}
    </div>
  );
}

function FieldRow({ f, highlight }) {
  return (
    <div style={{
      padding: '14px 18px',
      borderBottom: '1px solid var(--stroke)',
      display: 'grid', gridTemplateColumns: '140px 1fr auto', gap: 16,
      background: highlight ? 'var(--accent-tint)' : 'transparent',
      transition: 'background 0.15s',
    }}>
      {/* Label */}
      <div>
        <div style={{ fontSize: 12, color: 'var(--ink-3)', fontWeight: 500, marginBottom: 4, textTransform: 'uppercase', letterSpacing: '0.04em' }}>
          {f.label}
        </div>
        <VerifyBadge state={f.state} count={f.count} days={f.days} source={f.source} compact/>
      </div>

      {/* Value + attribution */}
      <div>
        <div style={{ fontSize: 14, color: 'var(--ink)', lineHeight: 1.5, marginBottom: 6 }}>
          {f.value}
        </div>
        {f.state === 'flagged-stale' && f.flagNote && (
          <div style={{
            marginTop: 6, marginBottom: 6, padding: '8px 10px',
            background: '#fef2f2', border: '1px solid #fecaca', borderRadius: 'var(--r-sm)',
          }}>
            <div className="row" style={{ gap: 6, fontSize: 11, color: '#b91c1c', marginBottom: 2, fontWeight: 500 }}>
              <V2Ico name="flag" size={10} stroke={2}/>
              <span>Flagged stale · {f.flaggedDays}d ago · {f.lastBy}</span>
            </div>
            <div style={{ fontSize: 12.5, color: 'var(--ink-2)', lineHeight: 1.4, fontStyle: 'italic' }}>
              {f.flagNote}
            </div>
          </div>
        )}
        <AttribLine
          who={f.state === 'ai-only' ? 'AI extracted' : f.state === 'flagged-stale' ? `${f.lastBy} flagged this` : `Confirmed by ${f.lastBy}`}
          when={f.days != null ? `${f.days}d ago` : null}
          source={f.source}
        />
      </div>

      {/* Per-row actions */}
      <div className="row" style={{ gap: 6, alignSelf: 'flex-start' }}>
        <button className="btn sm" title="Still accurate">
          <V2Ico name="check" size={11} stroke={2.2} style={{ color: 'var(--good)' }}/>
          <span>Confirm</span>
        </button>
        <button className="btn sm" title="Flag a problem">
          <V2Ico name="flag" size={11} stroke={1.9} style={{ color: '#b91c1c' }}/>
          <span>Flag</span>
        </button>
      </div>
    </div>
  );
}

function HistoryItem({ h }) {
  const kindMeta = {
    'flag-issue':  { glyph: '⚠', color: '#b91c1c', verb: 'flagged' },
    'confirm':     { glyph: '✓', color: '#15803d', verb: 'confirmed' },
    'ai-seed':     { glyph: '⬡', color: 'var(--accent-ink)', verb: 'seeded' },
    'ask-origin':  { glyph: '↳', color: 'var(--ink-3)', verb: 'originated' },
  }[h.kind] || { glyph: '·', color: 'var(--ink-3)' };
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '18px 1fr', gap: 8 }}>
      <span style={{ fontSize: 13, color: kindMeta.color, fontWeight: 600, lineHeight: 1.4, textAlign: 'center' }}>
        {kindMeta.glyph}
      </span>
      <div>
        <div style={{ fontSize: 12, color: 'var(--ink)', lineHeight: 1.45 }}>
          <strong style={{ fontWeight: 500 }}>{h.who}</strong>
          {h.field ? <> {kindMeta.verb} <span style={{ color: 'var(--ink-3)' }}>{h.field}</span></> : <> {kindMeta.verb}</>}
        </div>
        {h.note && (
          <div style={{ fontSize: 11.5, color: 'var(--ink-3)', lineHeight: 1.45, marginTop: 2 }}>
            {h.note}
          </div>
        )}
        <div className="mono" style={{ fontSize: 10, color: 'var(--ink-4)', marginTop: 2 }}>{h.when}</div>
      </div>
    </div>
  );
}

function FlagSheet({ field }) {
  return (
    <>
      <div style={{
        position: 'absolute', inset: 0, top: 0,
        background: 'rgb(24 24 27 / 0.32)',
        pointerEvents: 'none', zIndex: 5,
      }}/>
      <div style={{
        position: 'absolute', left: '50%', bottom: 0, transform: 'translateX(-50%)',
        width: 'min(520px, calc(100% - 32px))',
        background: 'var(--paper)',
        borderTopLeftRadius: 'var(--r-lg)', borderTopRightRadius: 'var(--r-lg)',
        border: '1px solid var(--stroke)', borderBottom: 0,
        boxShadow: 'var(--shadow-lg)', padding: 22,
        zIndex: 6,
      }}>
        {/* Drag affordance */}
        <div style={{ width: 36, height: 4, background: 'var(--stroke-2)', borderRadius: 2, margin: '0 auto 14px' }}/>

        <div className="between" style={{ marginBottom: 14 }}>
          <div>
            <div className="eyebrow" style={{ marginBottom: 4 }}>Flag · Referral path</div>
            <div style={{ fontSize: 13, color: 'var(--ink-2)', lineHeight: 1.45 }}>
              Current value: <span style={{ color: 'var(--ink)' }}>{field?.value}</span>
            </div>
          </div>
          <button className="btn ghost sm" style={{ alignSelf: 'flex-start' }}>✕</button>
        </div>

        {/* Reason chips */}
        <div style={{ marginBottom: 14 }}>
          <div className="eyebrow" style={{ marginBottom: 8 }}>What's wrong</div>
          <div className="row" style={{ gap: 6, flexWrap: 'wrap' }}>
            <span className="chip" style={{ background: '#fef2f2', borderColor: '#fecaca', color: '#b91c1c' }}>
              ⚠ Stale (info changed)
            </span>
            <span className="chip">Wrong (was never accurate)</span>
            <span className="chip">Resource closed</span>
            <span className="chip">Unsure — needs a 2nd opinion</span>
          </div>
        </div>

        {/* Note */}
        <div style={{ marginBottom: 14 }}>
          <div className="eyebrow" style={{ marginBottom: 8 }}>One line · what should it say?</div>
          <textarea className="field" rows={2} defaultValue='Piloting fax referrals from primary care — confirmed on a call 5/12'
            style={{ resize: 'none', fontSize: 13, lineHeight: 1.5, padding: 10 }}/>
          <div className="row" style={{ marginTop: 8, gap: 10, fontSize: 11, color: 'var(--ink-3)' }}>
            <a className="row" style={{ gap: 5, color: 'var(--accent-ink)' }}>
              <V2Ico name="plus" size={11} stroke={2}/> Attach source
            </a>
            <span>·</span>
            <span>a phone-call note, a screenshot, a link</span>
          </div>
        </div>

        {/* Anonymity + submit */}
        <div className="between" style={{
          paddingTop: 12, borderTop: '1px solid var(--stroke)',
        }}>
          <label className="row" style={{ gap: 8, fontSize: 12.5, color: 'var(--ink-2)', cursor: 'pointer' }}>
            <span style={{
              width: 30, height: 18, background: 'var(--paper-3)', borderRadius: 999,
              position: 'relative', border: '1px solid var(--stroke-2)', flexShrink: 0,
            }}>
              <span style={{
                position: 'absolute', top: 1, left: 1,
                width: 14, height: 14, borderRadius: '50%', background: 'var(--paper)',
                border: '1px solid var(--stroke-2)',
              }}/>
            </span>
            <span>Submit anonymously</span>
            <span style={{ color: 'var(--ink-4)' }}>(off — shows as <strong style={{ fontWeight: 500, color: 'var(--ink-2)' }}>Rita · AOHT</strong>)</span>
          </label>
          <button className="btn accent">Submit flag</button>
        </div>
      </div>
    </>
  );
}

function ResourceDetailWithFlag() {
  return <ResourceDetail flagOpen flagField="referral"/>;
}

Object.assign(window, { ResourceDetail, ResourceDetailWithFlag });
