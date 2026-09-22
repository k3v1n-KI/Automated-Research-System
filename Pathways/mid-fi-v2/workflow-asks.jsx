// Pathways v2 — Asks feed (answer flow)
// Left: a single Ask thread (yours) with replies + answer composer.
// Right: feed of other Asks in your region you can help with.

function AsksFeed() {
  const yourAsk = ASKS.find(a => a.yours);
  return (
    <div className="v2">
      <V2Topnav active="cases" />
      <div className="body">
        <main style={{ maxWidth: 1180, margin: '0 auto', padding: '28px 32px 80px' }}>

          {/* Breadcrumb */}
          <div className="row" style={{ gap: 8, marginBottom: 14, fontSize: 12, color: 'var(--ink-3)' }}>
            <a>Cases</a><V2Ico name="chevron-right" size={11}/>
            <a>Your Asks</a><V2Ico name="chevron-right" size={11}/>
            <span style={{ color: 'var(--ink-2)' }}>Same-day MH walk-in…</span>
          </div>

          {/* Title */}
          <div style={{ marginBottom: 18 }}>
            <div className="row" style={{ gap: 10, marginBottom: 8 }}>
              <span className="eyebrow" style={{ color: 'var(--accent-ink)' }}>Your Ask · East Toronto OHT · public</span>
              <span className="chip sm">● 1 new reply</span>
              <span className="mono" style={{ fontSize: 11, color: 'var(--ink-3)' }}>posted 5h ago · 6d left</span>
            </div>
          </div>

          {/* 2-col: thread + region feed */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 340px', gap: 28, alignItems: 'start' }}>

            {/* Thread */}
            <section>
              {/* The Ask itself */}
              <div style={{
                padding: '20px 22px',
                background: 'var(--accent-tint)',
                border: '1px solid color-mix(in srgb, var(--accent) 18%, transparent)',
                borderRadius: 'var(--r-md)', marginBottom: 18,
              }}>
                <p style={{ fontSize: 18, color: 'var(--ink)', lineHeight: 1.45, marginBottom: 12, letterSpacing: '-0.012em' }}>
                  Same-day MH walk-in for an adolescent without OHIP — any options not on the index?
                </p>
                <div className="row" style={{ gap: 12, fontSize: 11.5, color: 'var(--ink-3)' }}>
                  <span><strong style={{ color: 'var(--ink-2)', fontWeight: 500 }}>Context:</strong> 16yo, brought by aunt, missed last 2 appts at clinic. No OHIP. Speaks Cantonese + some English.</span>
                </div>
                <div className="row" style={{ gap: 8, marginTop: 14, flexWrap: 'wrap' }}>
                  <span className="chip accent sm">youth</span>
                  <span className="chip accent sm">MH</span>
                  <span className="chip accent sm">uninsured</span>
                  <span className="chip accent sm">Cantonese</span>
                  <span style={{ flex: 1 }}/>
                  <span className="mono" style={{ fontSize: 10.5, color: 'var(--accent-ink)' }}>7 watching · 3 replies</span>
                </div>
              </div>

              {/* Replies */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: 14, marginBottom: 18 }}>
                {ASK_REPLIES.map((r, i) => <ReplyItem key={i} r={r} isNew={i === 0}/>)}
              </div>

              {/* Composer for your own reply (if it were someone else's Ask) — here flipped to "thank + close" since it's yours */}
              <div className="card" style={{ padding: 16, background: 'var(--paper-2)' }}>
                <div className="between" style={{ marginBottom: 10 }}>
                  <span className="eyebrow">Resolve this Ask</span>
                  <span className="mono" style={{ fontSize: 10.5, color: 'var(--ink-3)' }}>replies show your name by default — toggle anonymous per reply</span>
                </div>
                <div className="row" style={{ gap: 8, flexWrap: 'wrap' }}>
                  <button className="btn sm"><V2Ico name="check" size={11} stroke={2.2}/> Mark one of these as the answer</button>
                  <button className="btn sm">Reply with more context</button>
                  <button className="btn ghost sm">Close · still looking</button>
                </div>
              </div>
            </section>

            {/* Right rail — Asks I could help with */}
            <aside style={{ position: 'sticky', top: 12, display: 'flex', flexDirection: 'column', gap: 14 }}>
              <div className="card" style={{ padding: 16 }}>
                <div className="between" style={{ marginBottom: 12 }}>
                  <span className="eyebrow">Open Asks · your region</span>
                  <a className="muted" style={{ fontSize: 11 }}>Filters</a>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                  {ASKS.filter(a => !a.yours).map((a, i, arr) => (
                    <a key={a.id} style={{
                      paddingBottom: i < arr.length - 1 ? 12 : 0,
                      borderBottom: i < arr.length - 1 ? '1px solid var(--stroke)' : 0,
                      cursor: 'pointer', display: 'block',
                    }}>
                      <div style={{ fontSize: 13, color: 'var(--ink)', lineHeight: 1.45, marginBottom: 6 }}>
                        {a.text}
                      </div>
                      <div className="row" style={{ gap: 7, fontSize: 10.5, color: 'var(--ink-3)', flexWrap: 'wrap' }}>
                        <span style={{ fontWeight: 500 }}>{a.who}</span>
                        <span>·</span>
                        <span>{a.region}</span>
                        <span>·</span>
                        <span className="mono">{a.replies} replies</span>
                        <span>·</span>
                        <span className="mono" style={{ color: a.expires.includes('rural') ? 'var(--accent-ink)' : 'var(--ink-3)' }}>
                          {a.expires} left
                        </span>
                      </div>
                    </a>
                  ))}
                </div>
              </div>

              <div className="card muted" style={{ padding: 14, background: 'var(--paper-2)' }}>
                <div className="eyebrow" style={{ marginBottom: 6 }}>How Asks work</div>
                <p style={{ fontSize: 11.5, color: 'var(--ink-3)', lineHeight: 1.55, margin: 0 }}>
                  Anyone in your AOHT region sees your Ask for 7 days
                  (longer in rural / low-density areas, shorter if you set a deadline).
                  Replies can point to indexed resources, suggest new ones, or just share context.
                </p>
              </div>
            </aside>

          </div>
        </main>
      </div>
    </div>
  );
}

function ReplyItem({ r, isNew }) {
  return (
    <div style={{
      paddingLeft: 16, borderLeft: isNew ? '2px solid var(--accent)' : '2px solid var(--stroke)',
    }}>
      <div className="row" style={{ gap: 8, marginBottom: 6 }}>
        <span className="avatar" style={{ width: 22, height: 22, fontSize: 9 }}>
          {r.who.split(' ')[0].slice(0, 2).toUpperCase()}
        </span>
        <span style={{ fontSize: 12.5, color: 'var(--ink)', fontWeight: 500 }}>{r.who}</span>
        <span className="mono" style={{ fontSize: 10.5, color: 'var(--ink-3)' }}>· {r.when}</span>
        {isNew && <span style={{ fontSize: 10, color: 'var(--accent-ink)', fontWeight: 600, letterSpacing: '0.05em' }}>NEW</span>}
      </div>
      <p style={{ fontSize: 13, color: 'var(--ink-2)', lineHeight: 1.55, marginBottom: 10 }}>{r.text}</p>

      {/* Attached resource */}
      {r.attaches?.kind === 'resource' && (
        <a className="row" style={{
          gap: 10, padding: '10px 12px', background: 'var(--paper-2)',
          border: '1px solid var(--stroke)', borderRadius: 'var(--r-sm)',
          cursor: 'pointer',
        }}>
          <V2Ico name="pin" size={13} style={{ color: 'var(--accent)' }}/>
          <div style={{ flex: 1 }}>
            <div style={{ fontSize: 13, color: 'var(--ink)', fontWeight: 500, marginBottom: 2 }}>{r.attaches.name}</div>
            <VerifyAggregate verified={r.attaches.verified} total={r.attaches.total} flagged={0} compact/>
          </div>
          <V2Ico name="arrow-up-right" size={12} style={{ color: 'var(--ink-3)' }}/>
        </a>
      )}
      {r.attaches?.kind === 'new' && (
        <div className="row" style={{
          gap: 10, padding: '10px 12px', background: 'var(--accent-tint)',
          border: '1px dashed color-mix(in srgb, var(--accent) 28%, transparent)', borderRadius: 'var(--r-sm)',
        }}>
          <V2Ico name="sparkle" size={13} style={{ color: 'var(--accent)' }}/>
          <div style={{ flex: 1 }}>
            <div style={{ fontSize: 13, color: 'var(--ink)', fontWeight: 500, marginBottom: 2 }}>{r.attaches.name}</div>
            <span style={{ fontSize: 11, color: 'var(--ink-3)' }}>{r.attaches.note}</span>
          </div>
          <button className="btn sm">Add to index</button>
        </div>
      )}
    </div>
  );
}

window.AsksFeed = AsksFeed;
