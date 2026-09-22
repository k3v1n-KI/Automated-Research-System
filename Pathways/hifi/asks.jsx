// Hi-fi Asks feed — Algoma OHT
// Your Ask is open with 3 replies; you can reply, attach a resource, or mark resolved.

function Asks() {
  const [selected, setSelected] = useState('a-103'); // your Ask
  const [replies, setReplies] = useState(ASK_REPLIES);
  const [draft, setDraft] = useState('');
  const [resolved, setResolved] = useState(false);
  const [toast, setToast] = useState(null);

  const showToast = (msg) => { setToast(msg); setTimeout(() => setToast(null), 2200); };

  const ask = ASKS.find(a => a.id === selected);

  const postReply = () => {
    if (!draft.trim()) return;
    setReplies(r => [...r, {
      who: 'Rita · AOHT', when: 'just now', text: draft.trim(),
      attaches: null,
    }]);
    setDraft('');
    showToast('Reply posted to Algoma OHT.');
  };

  return (
    <div className="v2">
      <Topnav active="asks" />
      <div className="body page-in">
        <main style={{ maxWidth: 1180, margin: '0 auto', padding: '28px 32px 80px' }}>

          {/* Header */}
          <div style={{ marginBottom: 22 }}>
            <div className="row" style={{ gap: 6, marginBottom: 14, fontSize: 12.5, color: 'var(--ink-3)' }}>
              <a href="../Pathways Hi-Fi.html" style={{ color: 'inherit' }}>Cases</a>
              <Ico name="chevron-right" size={11}/>
              <span style={{ color: 'var(--ink)' }}>Asks</span>
            </div>
            <PhaseRail current="ask" />
            <div className="between">
              <div>
                <h1 style={{ fontSize: 28, fontWeight: 600, letterSpacing: '-0.022em', lineHeight: 1.15, marginBottom: 6 }}>
                  Asks · Algoma OHT
                </h1>
                <p style={{ fontSize: 14, color: 'var(--ink-2)', maxWidth: 640, lineHeight: 1.5 }}>
                  When the index doesn't have it, AOHT members ask each other. Rural communities get a 12-day window;
                  the City gets 7 days.
                </p>
              </div>
              <button className="btn accent" onClick={() => showToast('Use the search bar on Cases to draft an Ask.')}>
                <Ico name="plus" size={13} stroke={2}/> New Ask
              </button>
            </div>
          </div>

          {/* 2-col: feed + detail */}
          <div style={{ display: 'grid', gridTemplateColumns: '360px 1fr', gap: 24, alignItems: 'start' }}>

            {/* Feed */}
            <div className="card" style={{ padding: 0 }}>
              <div style={{ padding: '12px 14px', borderBottom: '1px solid var(--stroke)' }}>
                <div className="row" style={{ gap: 6 }}>
                  <span className="chip accent">Open · {ASKS.length}</span>
                  <span className="chip">Yours</span>
                  <span className="chip">Resolved</span>
                </div>
              </div>
              {ASKS.map((a, i) => (
                <a key={a.id} onClick={() => setSelected(a.id)} style={{
                  display: 'block', padding: '14px 16px',
                  borderBottom: i < ASKS.length - 1 ? '1px solid var(--stroke)' : 0,
                  background: selected === a.id ? 'var(--accent-tint)' : 'transparent',
                  cursor: 'pointer',
                  borderLeft: selected === a.id ? '3px solid var(--accent)' : '3px solid transparent',
                }}>
                  <div className="row" style={{ gap: 6, marginBottom: 5, flexWrap: 'wrap' }}>
                    {a.yours && <span className="chip sm accent">Yours</span>}
                    {a.hasNewReply && <span className="chip sm" style={{ background: '#ecfdf5', borderColor: '#bbf7d0', color: '#15803d' }}>● New reply</span>}
                    {a.tags.slice(0, 2).map(t => <span key={t} className="chip sm">{t}</span>)}
                  </div>
                  <div style={{ fontSize: 13, color: 'var(--ink)', lineHeight: 1.5, marginBottom: 6 }}>
                    {a.text}
                  </div>
                  <div className="row" style={{ gap: 6, fontSize: 10.5, color: 'var(--ink-3)' }}>
                    <span style={{ fontWeight: 500 }}>{a.who}</span>
                    <span>·</span>
                    <span>{a.region}</span>
                    <span>·</span>
                    <span className="mono">{a.replies} replies</span>
                    <span>·</span>
                    <span className="mono" style={{ color: 'var(--ink-4)' }}>{a.expires} left</span>
                  </div>
                </a>
              ))}
            </div>

            {/* Detail */}
            <section style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
              {/* Ask body */}
              <div className="card" style={{ padding: 22 }}>
                <div className="row" style={{ gap: 8, marginBottom: 8 }}>
                  {ask.yours && <span className="chip sm accent">Your Ask</span>}
                  <span className="eyebrow">{ask.region} · posted {ask.when} · expires in {ask.expires}</span>
                </div>
                <p style={{ fontSize: 17, lineHeight: 1.45, color: 'var(--ink)', letterSpacing: '-0.012em', marginBottom: 12 }}>
                  {ask.text}
                </p>
                <div className="row" style={{ gap: 6, flexWrap: 'wrap', marginBottom: 14 }}>
                  {ask.tags.map(t => <span key={t} className="chip sm">{t}</span>)}
                </div>
                <div className="between" style={{ paddingTop: 12, borderTop: '1px solid var(--stroke)' }}>
                  <div className="row" style={{ gap: 16, fontSize: 12, color: 'var(--ink-3)' }}>
                    <span className="row" style={{ gap: 5 }}>
                      <Ico name="people" size={12}/> {ask.watching} watching
                    </span>
                    <span className="row" style={{ gap: 5 }}>
                      <Ico name="clock" size={12}/> {ask.replies} replies
                    </span>
                  </div>
                  <div className="row" style={{ gap: 8 }}>
                    <button className="btn sm" onClick={() => showToast('Ask edited.')}>
                      <Ico name="edit" size={11}/> Edit
                    </button>
                    <button className="btn sm" onClick={() => { setResolved(true); showToast('Marked as resolved.'); }}>
                      <Ico name="check" size={11} stroke={2.2}/> {resolved ? 'Resolved' : 'Mark resolved'}
                    </button>
                  </div>
                </div>
              </div>

              {/* Replies */}
              <div data-tour="ask-replies">
                <div className="eyebrow" style={{ marginBottom: 10 }}>Replies · {replies.length}</div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                  {replies.map((r, i) => <ReplyCard key={i} r={r}/>)}
                </div>
              </div>

              {/* Compose */}
              <div className="card" style={{ padding: 18 }}>
                <div className="eyebrow" style={{ marginBottom: 8 }}>Your reply</div>
                <div style={{ position: 'relative', marginBottom: 12 }}>
                  <textarea
                    className="field" rows={3}
                    value={draft}
                    onChange={e => setDraft(e.target.value)}
                    placeholder="What you'd suggest — a known resource, a person to call, or new info to add to the index."
                    style={{ resize: 'vertical', fontSize: 13.5, lineHeight: 1.55, paddingRight: 40 }}
                  />
                  <MicMount value={draft} setValue={setDraft} samples={[
                    'Algoma Family Services takes phone-ahead bookings from primary care for adolescent MH walk-ins — I confirmed last Tuesday.',
                    'Try the Indigenous Friendship Centre on Wellington — they do same-day intake without OHIP.',
                  ]} />
                </div>
                <div className="between">
                  <div className="row" style={{ gap: 8 }}>
                    <button className="btn sm" onClick={() => showToast('Resource picker opened.')}>
                      <Ico name="book" size={11}/> Attach a resource
                    </button>
                    <a className="btn sm" href="case-resolve.html">
                      <Ico name="plus" size={11}/> Add a new resource
                    </a>
                  </div>
                  <button className="btn accent" onClick={postReply} disabled={!draft.trim()}
                    style={{ opacity: draft.trim() ? 1 : 0.5 }}>
                    <Ico name="send" size={12}/> Post reply
                  </button>
                </div>
              </div>
            </section>
          </div>
        </main>
      </div>

      {toast && (
        <div className="toast">
          <span className="dot" style={{ width: 7, height: 7, borderRadius: '50%' }}/>
          {toast}
        </div>
      )}
    </div>
  );
}

function ReplyCard({ r }) {
  return (
    <div className="card" style={{ padding: 16 }}>
      <div className="between" style={{ marginBottom: 8 }}>
        <div className="row" style={{ gap: 8 }}>
          <span className="avatar" style={{ width: 22, height: 22, fontSize: 9.5 }}>
            {r.who.split('·')[0].trim().split(' ').map(s => s[0]).slice(0, 2).join('').toUpperCase()}
          </span>
          <span style={{ fontSize: 12.5, fontWeight: 500 }}>{r.who}</span>
          <span className="mono" style={{ fontSize: 10.5, color: 'var(--ink-3)' }}>· {r.when}</span>
        </div>
      </div>
      <p style={{ fontSize: 13.5, lineHeight: 1.55, color: 'var(--ink)', marginBottom: 12 }}>{r.text}</p>
      {r.attaches?.kind === 'resource' && (
        <a href="resource.html" className="card tap" style={{ padding: 12, display: 'flex', alignItems: 'center', gap: 10, textDecoration: 'none', color: 'inherit' }}>
          <Ico name="book" size={14} style={{ color: 'var(--accent)' }}/>
          <div style={{ flex: 1 }}>
            <div style={{ fontSize: 12.5, fontWeight: 500 }}>{r.attaches.name}</div>
            <div className="mono" style={{ fontSize: 10.5, color: 'var(--ink-3)' }}>
              {r.attaches.verified}/{r.attaches.total} fields verified
            </div>
          </div>
          <Ico name="arrow-right" size={12} style={{ color: 'var(--ink-3)' }}/>
        </a>
      )}
      {r.attaches?.kind === 'new' && (
        <div style={{
          padding: 12, border: '1px dashed var(--stroke-2)', borderRadius: 'var(--r-sm)',
          background: 'var(--accent-tint)', display: 'flex', gap: 10, alignItems: 'center',
        }}>
          <Ico name="plus" size={14} style={{ color: 'var(--accent)' }}/>
          <div style={{ flex: 1 }}>
            <div style={{ fontSize: 12.5, fontWeight: 500 }}>{r.attaches.name}</div>
            <div className="mono" style={{ fontSize: 10.5, color: 'var(--accent-ink)' }}>
              {r.attaches.note}
            </div>
          </div>
          <a href="case-resolve.html" className="btn accent sm">Add to index</a>
        </div>
      )}
    </div>
  );
}

ReactDOM.createRoot(document.getElementById('root')).render(<Asks />);
