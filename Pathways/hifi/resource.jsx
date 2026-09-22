// Hi-fi resource detail — Algoma Family Services Walk-in
// Interactive: confirm / flag any field, see your action recorded in the chain of custody.

function AttribLine({ who, when, source }) {
  return (
    <span className="mono" style={{ fontSize: 10.5, color: 'var(--ink-3)' }}>
      {who}{when ? ` · ${when}` : ''}{source ? ` · ${source}` : ''}
    </span>
  );
}

function Resource() {
  // Track local edits to fields and prepend history items.
  const [fields, setFields] = useState(() => RESOURCE.fields.map(f => ({ ...f })));
  const [history, setHistory] = useState(() => RESOURCE.history.slice());
  const [flagOpen, setFlagOpen] = useState(null); // field key or null
  const [toast, setToast] = useState(null);

  const showToast = (msg) => {
    setToast(msg);
    setTimeout(() => setToast(null), 2400);
  };

  const confirmField = (key) => {
    setFields(fs => fs.map(f => f.key === key
      ? { ...f, state: 'verified-fresh', days: 0, confirms: (f.confirms || 0) + 1, lastBy: 'Rita · AOHT', source: 'just now' }
      : f));
    setHistory(h => [{ kind: 'confirm', who: 'Rita · AOHT', when: 'just now',
      field: RESOURCE.fields.find(f => f.key === key)?.label }, ...h]);
    showToast('Confirmed · thanks, that boosts this resource in search.');
  };

  const submitFlag = (key, note, anon) => {
    const who = anon ? 'AOHT member' : 'Rita · AOHT';
    setFields(fs => fs.map(f => f.key === key
      ? { ...f, state: 'flagged-stale', count: (f.count || 0) + 1,
          lastBy: who, flagNote: `"${note}"`, flaggedDays: 0 }
      : f));
    setHistory(h => [{ kind: 'flag-issue', who, when: 'just now',
      field: RESOURCE.fields.find(f => f.key === key)?.label, note }, ...h]);
    setFlagOpen(null);
    showToast(anon ? 'Flag submitted · posted anonymously.' : 'Flag submitted · visible to AOHT members.');
  };

  const verifiedCount = fields.filter(f => f.state === 'verified-fresh' || f.state === 'verified-aging').length;
  const flaggedCount  = fields.filter(f => f.state === 'flagged-stale').length;
  const aiCount       = fields.filter(f => f.state === 'ai-only').length;

  return (
    <div className="v2">
      <Topnav active="cases" />
      <div className="body page-in">
        <main style={{ maxWidth: 1180, margin: '0 auto', padding: '28px 32px 80px' }}>

          {/* Breadcrumb */}
          <div className="row" style={{ gap: 6, marginBottom: 14, fontSize: 12.5, color: 'var(--ink-3)' }}>
            <a href="../Pathways Hi-Fi.html" style={{ color: 'inherit' }}>Cases</a>
            <Ico name="chevron-right" size={11}/>
            <a href="search.html" style={{ color: 'inherit' }}>Search · MH walk-in</a>
            <Ico name="chevron-right" size={11}/>
            <span style={{ color: 'var(--ink)' }}>{RESOURCE.name}</span>
          </div>

          <PhaseRail current="verify" />

          {/* Header */}
          <div style={{ marginBottom: 22 }}>
            <div className="row" style={{ gap: 10, marginBottom: 6 }}>
              <span className="eyebrow">Resource · Sault Ste. Marie</span>
              <VerifyAggregate verified={verifiedCount} total={fields.length} flagged={flaggedCount} compact/>
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
                <button className="btn" onClick={() => showToast('Saved to case c-240.')}>Save to case</button>
                <button className="btn" onClick={() => showToast('Link copied to clipboard.')}>Share</button>
                <a className="btn accent" href="case-resolve.html">Refer →</a>
              </div>
            </div>
          </div>

          {/* 2-col: fields + chain of custody */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 320px', gap: 28, alignItems: 'start' }}>

            <section className="card" style={{ padding: 0 }}>
              <div style={{ padding: '14px 18px', borderBottom: '1px solid var(--stroke)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span className="eyebrow">Fields · tap ✓ or ⚠ on any row</span>
                <span className="mono" style={{ fontSize: 11, color: 'var(--ink-3)' }}>
                  {verifiedCount} / {fields.length} verified · {flaggedCount} flagged · {aiCount} AI-only
                </span>
              </div>
              {fields.map((f) => (
                <FieldRow key={f.key} f={f}
                  onConfirm={() => confirmField(f.key)}
                  onFlag={() => setFlagOpen(f.key)}/>
              ))}
            </section>

            <aside style={{ display: 'flex', flexDirection: 'column', gap: 12, position: 'sticky', top: 12 }}>
              <div className="card" style={{ padding: 16 }}>
                <div className="eyebrow" style={{ marginBottom: 10 }}>Chain of custody</div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                  {history.map((h, i) => <HistoryItem key={i} h={h}/>)}
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

      {flagOpen && (
        <FlagSheet
          field={fields.find(f => f.key === flagOpen)}
          onClose={() => setFlagOpen(null)}
          onSubmit={(note, anon) => submitFlag(flagOpen, note, anon)}
        />
      )}

      {toast && (
        <div className="toast">
          <span className="dot" style={{ width: 7, height: 7, borderRadius: '50%' }}/>
          {toast}
        </div>
      )}
    </div>
  );
}

function FieldRow({ f, onConfirm, onFlag }) {
  return (
    <div data-tour={f.key === 'hours' ? 'field-first' : f.key === 'referral' ? 'field-flagged' : undefined} style={{
      padding: '14px 18px',
      borderBottom: '1px solid var(--stroke)',
      display: 'grid', gridTemplateColumns: '140px 1fr auto', gap: 16,
      transition: 'background 0.15s',
    }}>
      <div>
        <div style={{ fontSize: 12, color: 'var(--ink-3)', fontWeight: 500, marginBottom: 4, textTransform: 'uppercase', letterSpacing: '0.04em' }}>
          {f.label}
        </div>
        <VerifyBadge state={f.state} count={f.count} days={f.days} source={f.source} compact/>
      </div>

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
              <Ico name="flag" size={10} stroke={2}/>
              <span>Flagged stale · {f.flaggedDays === 0 ? 'just now' : `${f.flaggedDays}d ago`} · {f.lastBy}</span>
            </div>
            <div style={{ fontSize: 12.5, color: 'var(--ink-2)', lineHeight: 1.4, fontStyle: 'italic' }}>
              {f.flagNote}
            </div>
          </div>
        )}
        <AttribLine
          who={f.state === 'ai-only' ? 'AI extracted' : f.state === 'flagged-stale' ? `${f.lastBy} flagged this` : `Confirmed by ${f.lastBy}`}
          when={f.days != null ? (f.days === 0 ? 'just now' : `${f.days}d ago`) : null}
          source={f.source}
        />
      </div>

      <div className="row" style={{ gap: 6, alignSelf: 'flex-start' }}>
        <button className="btn sm" data-tour={f.key === 'hours' ? 'field-confirm' : undefined} onClick={onConfirm} title="Still accurate">
          <Ico name="check" size={11} stroke={2.2} style={{ color: 'var(--good)' }}/>
          <span>Confirm</span>
        </button>
        <button className="btn sm" onClick={onFlag} title="Flag a problem">
          <Ico name="flag" size={11} stroke={1.9} style={{ color: '#b91c1c' }}/>
          <span>Flag</span>
        </button>
      </div>
    </div>
  );
}

function HistoryItem({ h }) {
  const meta = {
    'flag-issue':  { glyph: '⚠', color: '#b91c1c', verb: 'flagged' },
    'confirm':     { glyph: '✓', color: '#15803d', verb: 'confirmed' },
    'ai-seed':     { glyph: '⬡', color: 'var(--accent-ink)', verb: 'seeded' },
    'ask-origin':  { glyph: '↳', color: 'var(--ink-3)', verb: 'originated' },
  }[h.kind] || { glyph: '·', color: 'var(--ink-3)' };
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '18px 1fr', gap: 8 }}>
      <span style={{ fontSize: 13, color: meta.color, fontWeight: 600, lineHeight: 1.4, textAlign: 'center' }}>
        {meta.glyph}
      </span>
      <div>
        <div style={{ fontSize: 12, color: 'var(--ink)', lineHeight: 1.45 }}>
          <strong style={{ fontWeight: 500 }}>{h.who}</strong>
          {h.field ? <> {meta.verb} <span style={{ color: 'var(--ink-3)' }}>{h.field}</span></> : <> {meta.verb}</>}
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

function FlagSheet({ field, onClose, onSubmit }) {
  const [reason, setReason] = useState('stale');
  const [note, setNote] = useState('They piloted phone-ahead booking from primary care — confirmed today.');
  const [anon, setAnon] = useState(false);
  const reasons = [
    { id: 'stale',  label: '⚠ Stale (info changed)' },
    { id: 'wrong',  label: 'Wrong (was never accurate)' },
    { id: 'closed', label: 'Resource closed' },
    { id: 'unsure', label: 'Unsure — needs a 2nd opinion' },
  ];
  return (
    <div className="sheet-overlay" onClick={onClose}>
      <div className="sheet" onClick={e => e.stopPropagation()} style={{ width: 'min(560px, 92vw)' }}>
        <div className="between" style={{ marginBottom: 14 }}>
          <div>
            <div className="eyebrow" style={{ marginBottom: 4 }}>Flag · {field?.label}</div>
            <div style={{ fontSize: 13, color: 'var(--ink-2)', lineHeight: 1.45 }}>
              Current value: <span style={{ color: 'var(--ink)' }}>{field?.value}</span>
            </div>
          </div>
          <button className="btn ghost sm" onClick={onClose} aria-label="Close"><Ico name="x" size={13}/></button>
        </div>

        <div style={{ marginBottom: 14 }}>
          <div className="eyebrow" style={{ marginBottom: 8 }}>What's wrong</div>
          <div className="row" style={{ gap: 6, flexWrap: 'wrap' }}>
            {reasons.map(r => (
              <button key={r.id} onClick={() => setReason(r.id)} className="chip"
                style={{
                  cursor: 'pointer', border: 'none',
                  background: reason === r.id ? '#fef2f2' : 'var(--paper-2)',
                  borderColor: reason === r.id ? '#fecaca' : 'var(--stroke)',
                  color: reason === r.id ? '#b91c1c' : 'var(--ink-2)',
                }}>
                {r.label}
              </button>
            ))}
          </div>
        </div>

        <div style={{ marginBottom: 14 }}>
          <div className="eyebrow" style={{ marginBottom: 8 }}>One line · what should it say?</div>
          <div style={{ position: 'relative' }}>
            <textarea className="field" rows={2} value={note} onChange={e => setNote(e.target.value)}
              style={{ resize: 'none', fontSize: 13, lineHeight: 1.5, padding: '10px 40px 10px 10px' }}/>
            <MicMount value={note} setValue={setNote} samples={[
              'Hours wrong — confirmed by phone they close at 4pm Wed, not 6pm.',
              'Referral path now: phone-ahead from primary care, no fax needed.',
            ]} />
          </div>
          <div className="row" style={{ marginTop: 8, gap: 10, fontSize: 11, color: 'var(--ink-3)' }}>
            <a className="row ilink" style={{ gap: 5 }}>
              <Ico name="plus" size={11} stroke={2}/> Attach source
            </a>
            <span>·</span>
            <span>a phone-call note, a screenshot, a link</span>
          </div>
        </div>

        <div className="between" style={{ paddingTop: 12, borderTop: '1px solid var(--stroke)' }}>
          <label className="row" style={{ gap: 8, fontSize: 12.5, color: 'var(--ink-2)', cursor: 'pointer' }}
                 onClick={() => setAnon(a => !a)}>
            <span style={{
              width: 30, height: 18, background: anon ? 'var(--accent)' : 'var(--paper-3)', borderRadius: 999,
              position: 'relative', border: '1px solid var(--stroke-2)', flexShrink: 0,
              transition: 'background 0.18s',
            }}>
              <span style={{
                position: 'absolute', top: 1, left: anon ? 13 : 1,
                width: 14, height: 14, borderRadius: '50%', background: 'var(--paper)',
                border: '1px solid var(--stroke-2)', transition: 'left 0.18s',
              }}/>
            </span>
            <span>Submit anonymously</span>
            <span style={{ color: 'var(--ink-4)' }}>
              ({anon ? 'on — posts as AOHT member' : 'off — shows as Rita · AOHT'})
            </span>
          </label>
          <button className="btn accent" onClick={() => onSubmit(note, anon)}>Submit flag</button>
        </div>
      </div>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById('root')).render(<Resource />);
