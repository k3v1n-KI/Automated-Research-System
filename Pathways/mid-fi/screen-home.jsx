// Pathways mid-fi — Home A · two-path split
// Primary action (start case) on the left, browse directory on the right.
// Below: recent cases, peer activity from forum, lightweight stats.

function ScreenHome() {
  const r = useRouter();
  return (
    <div className="page-enter">
      <Topnav active="cases" />
      <main style={{ maxWidth: 1280, margin: '0 auto', padding: '40px 32px 72px' }}>

        {/* Hero greeting */}
        <div style={{ marginBottom: 32 }}>
          <div className="eyebrow" style={{ marginBottom: 10 }}>Wednesday · May 7</div>
          <h1 className="title-display" style={{ fontSize: 44 }}>
            Good morning, Rita.
          </h1>
          <p className="muted" style={{ marginTop: 10, fontSize: 15, maxWidth: 640 }}>
            You have <span style={{ color: 'var(--ink)', fontWeight: 500 }}>3 saved cases</span> waiting and a peer flagged a wait-time change in your region.
          </p>
        </div>

        {/* Two-path split */}
        <div style={{ display: 'grid', gridTemplateColumns: '1.35fr 1fr', gap: 16, marginBottom: 36 }}>

          {/* Primary path — start a new case */}
          <button
            onClick={() => r.go({ name: 'entry' })}
            className="card lift"
            style={{
              padding: 28, textAlign: 'left', cursor: 'pointer',
              background: 'var(--ink)', color: 'var(--paper)', borderColor: 'var(--ink)',
              display: 'flex', flexDirection: 'column', justifyContent: 'space-between',
              minHeight: 260, transition: 'transform 0.15s, box-shadow 0.15s',
              boxShadow: 'var(--shadow-md)',
            }}
            onMouseEnter={e => e.currentTarget.style.transform = 'translateY(-1px)'}
            onMouseLeave={e => e.currentTarget.style.transform = ''}
          >
            <div>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.08em', opacity: 0.55, marginBottom: 14 }}>
                01 · Start here
              </div>
              <h2 className="title-display" style={{ fontSize: 38, color: 'var(--paper)', maxWidth: 560 }}>
                Find the right resource for a patient.
              </h2>
              <p style={{ marginTop: 14, fontSize: 14, opacity: 0.72, maxWidth: 480, lineHeight: 1.55 }}>
                Sketch the case in plain language — Pathways extracts needs, eligibility and constraints, then ranks resources in your region.
              </p>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginTop: 28 }}>
              <span style={{
                display: 'inline-flex', alignItems: 'center', gap: 8,
                padding: '10px 16px',
                background: 'var(--paper)', color: 'var(--ink)',
                borderRadius: 'var(--r-sm)', fontWeight: 500, fontSize: 14,
              }}>
                Start a new case <Ico name="arrow-right" size={15} />
              </span>
              <span style={{ fontSize: 12, opacity: 0.55, fontFamily: 'var(--font-mono)' }}>
                <span className="kbd" style={{ background: 'rgb(255 255 255 / 0.08)', borderColor: 'rgb(255 255 255 / 0.16)', color: 'rgb(255 255 255 / 0.7)' }}>N</span>
                <span style={{ marginLeft: 6 }}>or paste a referral note</span>
              </span>
            </div>
          </button>

          {/* Secondary path — browse directory */}
          <div className="card" style={{ padding: 24, display: 'flex', flexDirection: 'column', justifyContent: 'space-between', minHeight: 260 }}>
            <div>
              <div className="eyebrow" style={{ marginBottom: 14 }}>02 · Or browse</div>
              <h2 className="title-1" style={{ fontSize: 22 }}>Directory</h2>
              <p className="muted" style={{ marginTop: 8, fontSize: 13.5, lineHeight: 1.55 }}>
                412 verified resources across East Toronto OHT. Search by service, language, or eligibility.
              </p>
            </div>
            <div style={{ marginTop: 16 }}>
              <div className="row" style={{
                border: '1px solid var(--stroke-2)', borderRadius: 'var(--r-sm)',
                padding: '8px 10px', background: 'var(--paper-2)', gap: 8,
              }}>
                <Ico name="search" size={14} />
                <span className="muted" style={{ fontSize: 13 }}>Search resources…</span>
                <span className="nav-spacer" />
                <span className="kbd">/</span>
              </div>
              <div className="row" style={{ flexWrap: 'wrap', gap: 6, marginTop: 12 }}>
                {['Mental health', 'Housing', 'Senior care', 'Pediatrics', 'Substance use', 'Newcomer'].map(t => (
                  <span key={t} className="chip">{t}</span>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Recent + Peer activity row */}
        <div style={{ display: 'grid', gridTemplateColumns: '1.35fr 1fr', gap: 16, marginBottom: 36 }}>

          {/* Recent cases */}
          <section>
            <div className="between" style={{ marginBottom: 14 }}>
              <h3 className="title-2">Recent cases</h3>
              <a className="muted" style={{ fontSize: 12.5, cursor: 'pointer' }}>View all 14 →</a>
            </div>
            <div className="card" style={{ padding: 0 }}>
              {RECENT_CASES.map((c, i) => (
                <div
                  key={c.id}
                  onClick={() => r.go({ name: 'results', caseId: c.id })}
                  className="row-tap"
                  style={{
                    display: 'grid',
                    gridTemplateColumns: '1fr auto auto auto',
                    alignItems: 'center',
                    gap: 16, padding: '14px 18px',
                    borderBottom: i < RECENT_CASES.length - 1 ? '1px solid var(--stroke)' : 'none',
                    cursor: 'pointer',
                  }}
                >
                  <div className="stack-2">
                    <div style={{ fontSize: 13.5, fontWeight: 500 }}>{c.label}</div>
                    <div className="muted" style={{ fontSize: 12 }}>{c.summary}</div>
                  </div>
                  <div className="row" style={{ gap: 4 }}>
                    {c.tags.map(t => <span key={t} className="chip sm">{t}</span>)}
                  </div>
                  <span className="badge" style={{ background: c.statusBg, color: c.statusFg }}>{c.status}</span>
                  <span className="mono" style={{ fontSize: 11, color: 'var(--ink-3)', minWidth: 64, textAlign: 'right' }}>{c.when}</span>
                </div>
              ))}
            </div>
          </section>

          {/* Peer activity / region */}
          <aside className="stack-12">
            <div className="ai-zone" style={{ padding: 16 }}>
              <AITag>Region update</AITag>
              <p style={{ marginTop: 10, fontSize: 13, lineHeight: 1.55 }}>
                <strong>Across Health</strong> walk-in mental-health intake hours changed Monday. <span className="muted">3 of your saved cases reference this resource.</span>
              </p>
              <div className="row" style={{ marginTop: 12, gap: 6 }}>
                <button className="btn sm">Review impact</button>
                <button className="btn ghost sm">Dismiss</button>
              </div>
            </div>

            <div className="card" style={{ padding: 16 }}>
              <div className="between" style={{ marginBottom: 10 }}>
                <h4 className="title-3">From the forum</h4>
                <a className="muted" style={{ fontSize: 12, cursor: 'pointer' }}>Open feed →</a>
              </div>
              <div className="stack-12">
                {FORUM_BLIPS.map(b => (
                  <div key={b.id} className="stack-4">
                    <div className="row" style={{ gap: 6, fontSize: 11.5 }}>
                      <span className="mono" style={{ color: 'var(--ink-3)' }}>{b.region}</span>
                      <span style={{ color: 'var(--ink-4)' }}>·</span>
                      <span className="mono" style={{ color: 'var(--ink-3)' }}>{b.when}</span>
                    </div>
                    <div style={{ fontSize: 13, lineHeight: 1.5 }}>{b.text}</div>
                    <div className="row" style={{ gap: 10, fontSize: 11.5, color: 'var(--ink-3)' }}>
                      <span>{b.author}</span>
                      <span>·</span>
                      <span>★ {b.confirms}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </aside>
        </div>

        {/* Stats strip */}
        <section className="card muted" style={{ padding: 22, display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 24, alignItems: 'end' }}>
          {STATS.map(s => (
            <div key={s.label} className="stack-4">
              <div className="eyebrow">{s.label}</div>
              <div className="stat-num sm">{s.value}</div>
              <div className="muted" style={{ fontSize: 11.5 }}>{s.note}</div>
            </div>
          ))}
        </section>
      </main>
    </div>
  );
}

const RECENT_CASES = [
  { id: 'c-241', label: 'Senior, post-discharge, lives alone', summary: 'Needs home support + meals · ESL Cantonese · East York', tags: ['senior', 'home-care'], status: 'Saved', statusBg: 'var(--paper-3)', statusFg: 'var(--ink-2)', when: '2h ago' },
  { id: 'c-240', label: 'Adolescent · self-harm risk · uninsured', summary: 'Needs walk-in MH within 48h · no OHIP · Scarborough', tags: ['mental-health', 'youth', 'uninsured'], status: 'Referred', statusBg: 'color-mix(in oklch, var(--good) 14%, var(--paper))', statusFg: 'oklch(0.36 0.10 160)', when: 'yesterday' },
  { id: 'c-238', label: 'New mom · postpartum support', summary: 'Wants peer group + lactation · French preferred', tags: ['perinatal', 'french'], status: 'Saved', statusBg: 'var(--paper-3)', statusFg: 'var(--ink-2)', when: '2d ago' },
  { id: 'c-235', label: 'Adult · housing-insecure · methadone program', summary: 'Continuing care + shelter coordination', tags: ['housing', 'sud'], status: 'Closed', statusBg: 'var(--paper-3)', statusFg: 'var(--ink-3)', when: '4d ago' },
];

const FORUM_BLIPS = [
  { id: 1, region: 'East Toronto', when: '1h', text: 'Across Health intake now Mon/Wed only — confirmed 9:15 AM call.', author: 'M. Patel, RN', confirms: 4 },
  { id: 2, region: 'Scarborough', when: '5h', text: 'TAIBU Community Health is taking new uninsured patients again.', author: 'D. Singh, SW', confirms: 7 },
  { id: 3, region: 'North York', when: '1d', text: 'Hong Fook waitlist down to 3 weeks for adult intake.', author: 'L. Wong, NP', confirms: 2 },
];

const STATS = [
  { label: 'In your region', value: '412', note: 'verified resources' },
  { label: 'Last 7 days', value: '23', note: 'updates from peers' },
  { label: 'Avg. time-to-refer', value: '94s', note: 'from case start' },
  { label: 'Coverage', value: '88%', note: 'of OHT services represented' },
];

window.ScreenHome = ScreenHome;
