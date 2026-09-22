// Pathways v2 — Browse + answer someone else's Ask
// Two artboards:
//   AsksBrowse — scrolling other members' Asks (the contribution surface)
//   AskAnswer  — opening one to reply (attach known resource OR draft a new one)

function AsksBrowse() {
  return (
    <div className="v2">
      <V2Topnav active="cases" />
      <div className="body">
        <main style={{ maxWidth: 1180, margin: '0 auto', padding: '28px 32px 80px' }}>

          {/* Header */}
          <div style={{ marginBottom: 18 }}>
            <div className="row" style={{ gap: 10, marginBottom: 8 }}>
              <span className="eyebrow">Asks · East Toronto OHT</span>
              <span className="mono" style={{ fontSize: 11, color: 'var(--ink-3)' }}>
                14 open · 3 in your saved tags
              </span>
            </div>
            <h1 style={{ fontSize: 26, fontWeight: 600, letterSpacing: '-0.02em', lineHeight: 1.2, marginBottom: 6 }}>
              What colleagues are looking for.
            </h1>
            <p style={{ fontSize: 13, color: 'var(--ink-3)', lineHeight: 1.5, maxWidth: 640 }}>
              Answering an Ask is the fastest way to seed the index — and to help a specific colleague this week.
              Replies show your name by default.
            </p>
          </div>

          {/* Filters */}
          <div className="row" style={{ gap: 8, marginBottom: 16, flexWrap: 'wrap' }}>
            <button className="btn sm" style={{ borderColor: 'var(--ink)', background: 'var(--ink)', color: 'var(--paper)' }}>
              All · 14
            </button>
            <button className="btn sm">Your tags · 3</button>
            <button className="btn sm">Unanswered · 6</button>
            <button className="btn sm">Closing soon · 2</button>
            <span style={{ flex: 1 }}/>
            <button className="btn accent sm"><V2Ico name="plus" size={11} stroke={2}/> Post an Ask</button>
          </div>

          {/* 2-col: feed · context rail */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 320px', gap: 24, alignItems: 'start' }}>

            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              {ASKS_BROWSE.map((a, i) => <AskCard key={a.id} a={a} focused={i === 0}/>)}
            </div>

            {/* Right rail — your contribution context */}
            <aside style={{ position: 'sticky', top: 12, display: 'flex', flexDirection: 'column', gap: 14 }}>
              <div className="card" style={{ padding: 16 }}>
                <div className="eyebrow" style={{ marginBottom: 10 }}>Your saved tags</div>
                <div className="row" style={{ gap: 6, flexWrap: 'wrap', marginBottom: 12 }}>
                  {['senior', 'home-care', 'Cantonese', 'East York', 'ODSP'].map(t => (
                    <span key={t} className="chip sm">{t}</span>
                  ))}
                </div>
                <p style={{ fontSize: 11.5, color: 'var(--ink-3)', lineHeight: 1.5, margin: 0 }}>
                  Asks matching your tags get a <span style={{ color: 'var(--accent-ink)', fontWeight: 500 }}>↳ for you</span> mark.
                  Edit tags in settings.
                </p>
              </div>

              <div className="card muted" style={{ padding: 14, background: 'var(--paper-2)' }}>
                <div className="eyebrow" style={{ marginBottom: 6 }}>Reply etiquette</div>
                <p style={{ fontSize: 11.5, color: 'var(--ink-3)', lineHeight: 1.55, margin: 0 }}>
                  Point to a resource you know, or describe one we don't have indexed. A phone number + a
                  one-line caveat is enough — the asker will call.
                </p>
              </div>
            </aside>
          </div>

        </main>
      </div>
    </div>
  );
}

function AskCard({ a, focused }) {
  return (
    <a className="card" style={{
      padding: 18, cursor: 'pointer',
      borderColor: focused ? 'var(--stroke-2)' : 'var(--stroke)',
      background: focused ? 'var(--paper)' : 'var(--paper)',
    }}>
      <div className="row" style={{ gap: 8, marginBottom: 10, flexWrap: 'wrap' }}>
        <span className="avatar" style={{ width: 22, height: 22, fontSize: 9 }}>
          {a.who.split(' ')[0].slice(0, 2).toUpperCase()}
        </span>
        <span style={{ fontSize: 12.5, color: 'var(--ink)', fontWeight: 500 }}>{a.who}</span>
        <span className="mono" style={{ fontSize: 11, color: 'var(--ink-3)' }}>· {a.region} · {a.when}</span>
        {a.matched && (
          <span className="chip accent sm">↳ for you</span>
        )}
        <span style={{ flex: 1 }}/>
        <span className="mono" style={{ fontSize: 10.5, color: a.expires.includes('rural') || a.expires.includes('hour') ? 'var(--accent-ink)' : 'var(--ink-3)' }}>
          {a.expires} left
        </span>
      </div>
      <div style={{ fontSize: 15, color: 'var(--ink)', lineHeight: 1.45, marginBottom: 10, letterSpacing: '-0.008em' }}>
        {a.text}
      </div>
      {a.context && (
        <div style={{ fontSize: 12.5, color: 'var(--ink-3)', lineHeight: 1.5, marginBottom: 12, paddingLeft: 10, borderLeft: '2px solid var(--stroke)' }}>
          <strong style={{ fontWeight: 500, color: 'var(--ink-2)' }}>Context:</strong> {a.context}
        </div>
      )}
      <div className="row" style={{ gap: 8, flexWrap: 'wrap' }}>
        {a.tags.map(t => <span key={t} className="chip sm">{t}</span>)}
        <span style={{ flex: 1 }}/>
        <span className="mono" style={{ fontSize: 10.5, color: 'var(--ink-3)' }}>
          {a.replies} {a.replies === 1 ? 'reply' : 'replies'} · {a.watching} watching
        </span>
        <button className="btn sm"><V2Ico name="check" size={11} stroke={2}/> I know one</button>
        <button className="btn ghost sm">Watch</button>
      </div>
    </a>
  );
}

function AskAnswer() {
  return (
    <div className="v2">
      <V2Topnav active="cases" />
      <div className="body">
        <main style={{ maxWidth: 920, margin: '0 auto', padding: '28px 32px 80px' }}>

          <div className="row" style={{ gap: 8, marginBottom: 14, fontSize: 12, color: 'var(--ink-3)' }}>
            <a>Cases</a><V2Ico name="chevron-right" size={11}/>
            <a>Asks · East Toronto</a><V2Ico name="chevron-right" size={11}/>
            <span style={{ color: 'var(--ink-2)' }}>Low-cost dental for seniors…</span>
          </div>

          {/* The Ask */}
          <div style={{
            padding: '22px 24px',
            background: 'var(--accent-tint)',
            border: '1px solid color-mix(in srgb, var(--accent) 18%, transparent)',
            borderRadius: 'var(--r-md)', marginBottom: 18,
          }}>
            <div className="row" style={{ gap: 8, marginBottom: 12 }}>
              <span className="avatar" style={{ width: 22, height: 22, fontSize: 9 }}>KH</span>
              <span style={{ fontSize: 12.5, color: 'var(--ink)', fontWeight: 500 }}>Khalil · AOHT</span>
              <span className="mono" style={{ fontSize: 11, color: 'var(--ink-3)' }}>· East York · 2h ago</span>
              <span className="chip accent sm">↳ matches your tags</span>
              <span style={{ flex: 1 }}/>
              <span className="mono" style={{ fontSize: 10.5, color: 'var(--accent-ink)' }}>6d left · 4 watching</span>
            </div>
            <p style={{ fontSize: 18, color: 'var(--ink)', lineHeight: 1.45, marginBottom: 10, letterSpacing: '-0.012em' }}>
              Low-cost dental for seniors, Cantonese-speaking, will travel to Scarborough?
            </p>
            <div style={{ fontSize: 12.5, color: 'var(--ink-3)', lineHeight: 1.55, paddingLeft: 10, borderLeft: '2px solid color-mix(in srgb, var(--accent) 24%, transparent)' }}>
              <strong style={{ fontWeight: 500, color: 'var(--ink-2)' }}>Context:</strong> 78yo, ODSP, refers to herself in Cantonese. Daughter can drive on weekends only.
            </div>
            <div className="row" style={{ gap: 6, marginTop: 12, flexWrap: 'wrap' }}>
              <span className="chip accent sm">senior</span>
              <span className="chip accent sm">dental</span>
              <span className="chip accent sm">language</span>
              <span className="chip accent sm">low-cost</span>
            </div>
          </div>

          {/* Existing replies (1 in this case) */}
          <div style={{ marginBottom: 18 }}>
            <div className="eyebrow" style={{ marginBottom: 12 }}>Replies · 1</div>
            <div style={{ paddingLeft: 16, borderLeft: '2px solid var(--stroke)' }}>
              <div className="row" style={{ gap: 8, marginBottom: 6 }}>
                <span className="avatar" style={{ width: 22, height: 22, fontSize: 9 }}>AS</span>
                <span style={{ fontSize: 12.5, color: 'var(--ink)', fontWeight: 500 }}>Anh-Sang · AOHT</span>
                <span className="mono" style={{ fontSize: 10.5, color: 'var(--ink-3)' }}>· 1h ago</span>
              </div>
              <p style={{ fontSize: 13, color: 'var(--ink-2)', lineHeight: 1.55, marginBottom: 10 }}>
                Don't have Cantonese, but Toronto Public Health's senior dental program (free) has translators on weekends.
                Worth a call.
              </p>
              <a className="row" style={{
                gap: 10, padding: '10px 12px', background: 'var(--paper-2)',
                border: '1px solid var(--stroke)', borderRadius: 'var(--r-sm)',
                cursor: 'pointer', textDecoration: 'none',
              }}>
                <V2Ico name="pin" size={13} style={{ color: 'var(--accent)' }}/>
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: 13, color: 'var(--ink)', fontWeight: 500, marginBottom: 2 }}>
                    TPH Senior Dental · Scarborough Centre
                  </div>
                  <VerifyAggregate verified={6} total={8} flagged={0} compact/>
                </div>
                <V2Ico name="arrow-up-right" size={12} style={{ color: 'var(--ink-3)' }}/>
              </a>
            </div>
          </div>

          {/* Compose your reply */}
          <div className="card" style={{ padding: 20 }}>
            <div className="between" style={{ marginBottom: 12 }}>
              <span className="eyebrow" style={{ color: 'var(--accent-ink)' }}>Your reply</span>
              <span className="mono" style={{ fontSize: 10.5, color: 'var(--ink-3)' }}>posts as Rita · AOHT · toggle anonymous below</span>
            </div>

            <textarea className="field" rows={3}
              defaultValue="Cedar Health has Cantonese-speaking volunteers running a Saturday dental clinic for seniors on ODSP. Coordinator is Susan — she screens by phone first."
              style={{ resize: 'none', fontSize: 13.5, lineHeight: 1.55, padding: 12, marginBottom: 12 }}/>

            {/* Attach a resource */}
            <div className="eyebrow" style={{ marginBottom: 8 }}>Attach</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginBottom: 14 }}>
              <button className="row" style={{
                gap: 10, padding: '10px 12px', border: '1px solid var(--stroke)', borderRadius: 'var(--r-sm)',
                background: 'var(--paper)', textAlign: 'left', cursor: 'pointer', width: '100%',
              }}>
                <V2Ico name="search" size={13} style={{ color: 'var(--ink-3)' }}/>
                <span style={{ flex: 1, fontSize: 13, color: 'var(--ink-3)' }}>
                  Search the index for a resource to attach…
                </span>
                <span className="kbd">⌘K</span>
              </button>

              {/* A pending new-resource the user is drafting */}
              <div style={{
                padding: 12, border: '1px solid color-mix(in srgb, var(--accent) 28%, transparent)',
                borderRadius: 'var(--r-sm)', background: 'var(--accent-tint)',
              }}>
                <div className="row" style={{ gap: 8, marginBottom: 8 }}>
                  <V2Ico name="sparkle" size={12} style={{ color: 'var(--accent)' }}/>
                  <span className="eyebrow" style={{ color: 'var(--accent-ink)' }}>Or add a resource we don't have</span>
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginBottom: 8 }}>
                  <input className="field" placeholder="Name" defaultValue="Cedar Health · Sat Senior Dental"
                    style={{ fontSize: 13, padding: '8px 10px' }}/>
                  <input className="field" placeholder="Phone" defaultValue="(416) 555-0270"
                    style={{ fontSize: 13, padding: '8px 10px', fontFamily: 'var(--font-mono)' }}/>
                </div>
                <textarea className="field" rows={2} placeholder="One line — what to say when you call · or what they should know"
                  defaultValue="Saturdays only · ask for Susan · she screens by phone first · ODSP welcomed"
                  style={{ resize: 'none', fontSize: 13, lineHeight: 1.5 }}/>
                <div className="row" style={{ marginTop: 8, gap: 8, fontSize: 11, color: 'var(--ink-3)' }}>
                  <span>AI will enrich the rest later · members can verify.</span>
                </div>
              </div>
            </div>

            {/* Anonymity + submit */}
            <div className="between" style={{ paddingTop: 12, borderTop: '1px solid var(--stroke)' }}>
              <label className="row" style={{ gap: 8, fontSize: 12.5, color: 'var(--ink-2)', cursor: 'pointer' }}>
                <span style={{
                  width: 30, height: 18, background: 'var(--paper-3)', borderRadius: 999,
                  position: 'relative', border: '1px solid var(--stroke-2)',
                }}>
                  <span style={{ position: 'absolute', top: 1, left: 1, width: 14, height: 14, borderRadius: '50%', background: 'var(--paper)', border: '1px solid var(--stroke-2)' }}/>
                </span>
                <span>Reply anonymously</span>
                <span style={{ color: 'var(--ink-4)' }}>(off — shows as <strong style={{ color: 'var(--ink-2)', fontWeight: 500 }}>Rita · AOHT</strong>)</span>
              </label>
              <div className="row" style={{ gap: 8 }}>
                <button className="btn">Save draft</button>
                <button className="btn accent">Send reply · add 1 resource</button>
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}

// Browse-feed sample
const ASKS_BROWSE = [
  { id: 'a-104', who: 'Khalil · AOHT', region: 'East York', when: '2h ago',
    text: 'Low-cost dental for seniors, Cantonese-speaking, will travel to Scarborough?',
    context: '78yo, ODSP, refers to herself in Cantonese. Daughter can drive on weekends only.',
    expires: '6 days', replies: 1, watching: 4, tags: ['senior', 'dental', 'language', 'Cantonese'],
    matched: true },
  { id: 'a-101', who: 'Amelie · AOHT', region: 'Scarborough', when: '1d ago',
    text: 'Postpartum support group in French — any peer-led groups still meeting?',
    context: 'New mum, isolated, French is her stronger language. Has OHIP.',
    expires: '5 days', replies: 2, watching: 3, tags: ['perinatal', 'language'] },
  { id: 'a-100', who: 'Devon · AOHT', region: 'East York', when: '1d ago',
    text: 'Weekend respite for caregiver of someone living with dementia — Cantonese-speaking ideal.',
    context: 'Caregiver burnout, considering crisis intake. Trying to avoid that.',
    expires: '12 hours · urgent', replies: 0, watching: 9, tags: ['caregiver', 'dementia', 'Cantonese', 'East York'],
    matched: true },
  { id: 'a-098', who: 'Priya · AOHT', region: 'Riverdale', when: '2d ago',
    text: 'Continuing methadone + shelter coordination — looking for a single contact who handles both.',
    context: '32M, returning from incarceration, no fixed address. Has been on methadone before.',
    expires: '12 days · rural extended', replies: 0, watching: 6, tags: ['housing', 'sud'] },
  { id: 'a-097', who: 'Maria · AOHT', region: 'East York', when: '3d ago',
    text: 'After-hours pediatric urgent care that takes uninsured. Anyone got one not on the index?',
    context: 'Toddler, recurrent ear infections, family has no OHIP yet.',
    expires: '4 days', replies: 4, watching: 11, tags: ['pediatric', 'uninsured', 'East York'],
    matched: true },
  { id: 'a-095', who: 'Anh-Sang · AOHT', region: 'Scarborough', when: '4d ago',
    text: 'Free legal aid for housing — specifically tenants facing renoviction.',
    expires: '3 days', replies: 3, watching: 8, tags: ['legal', 'housing'] },
];

Object.assign(window, { AsksBrowse, AskAnswer });
