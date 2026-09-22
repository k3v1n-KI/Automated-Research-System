// Resource detail · Forum feed · Forum thread

function ResourceDetail() {
  return (
    <AppFrame active="Directory">
      <div style={{ maxWidth: 880, margin: '0 auto' }}>
        <div className="meta" style={{ marginBottom: 4 }}>← back to results</div>
        <div className="between" style={{ marginBottom: 6 }}>
          <h1 className="title-1">Yee Hong Centre — Scarborough</h1>
          <div className="row">
            <span className="badge good">verified 3d ago</span>
            <button className="btn sm">Save</button>
            <button className="btn sm">PDF</button>
            <button className="btn sm">SMS to patient</button>
            <button className="btn primary sm">Copy link</button>
          </div>
        </div>
        <div className="row" style={{ flexWrap: 'wrap', gap: 6, marginBottom: 16 }}>
          {['PSW', 'Home care', 'Mandarin', 'Cantonese', 'OHIP', 'Trans-affirming', 'Indigenous-allied'].map(t => <span key={t} className="chip">{t}</span>)}
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1.4fr 1fr', gap: 16 }}>
          <div className="col">
            <div className="card">
              <div className="title-3" style={{ marginBottom: 8 }}>At a glance</div>
              <table className="tbl">
                <tbody>
                  {[
                    ['Hours', '24/7 intake line · service Mon–Sun 7am–9pm'],
                    ['Address', '5 Crown Princess Crt, Scarborough'],
                    ['Phone', '416-xxx-xxxx'],
                    ['Eligibility', 'OHIP-eligible · 65+'],
                    ['Cost', 'No cost (publicly funded)'],
                    ['Wait', '≈ 2 weeks (last reported by 3 nurses)'],
                    ['Languages', 'Mandarin · Cantonese · English · Tagalog'],
                    ['Cultural fit', 'Mandarin/Cantonese-led · culturally-specific menus'],
                    ['Referral', 'Self-refer or via fax form (printable)'],
                  ].map(([k, v]) => (
                    <tr key={k}><td className="meta" style={{ width: 110 }}>{k}</td><td>{v}</td></tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="card">
              <div className="between" style={{ marginBottom: 8 }}>
                <div className="title-3">Recent forum mentions</div>
                <a className="meta" style={{ color: 'var(--accent-ink)' }}>see all 12 →</a>
              </div>
              <div className="col" style={{ gap: 10, fontSize: 12 }}>
                <div><span className="badge good">tip</span> <b>Sarah K.</b> · 2d — "Intake line answers in Mandarin even at 11pm."</div>
                <div><span className="badge warn">waitlist</span> <b>Anwar T.</b> · 5d — "Wait pushed to ~2wks since last month."</div>
                <div><span className="badge">question</span> <b>Priya R.</b> · 1w — "Anyone tried their meal program?"</div>
              </div>
            </div>

            <AIZone>
              <div className="eyebrow" style={{ color: 'var(--accent-ink)', marginBottom: 6 }}>data sources</div>
              <ul style={{ margin: 0, paddingLeft: 18, fontSize: 12, color: 'var(--ink-2)' }}>
                <li>Official site (yeehong.com) — last fetched 3d ago</li>
                <li>Forum closure report by Sarah K. — 2d</li>
                <li>OHT directory — auto-synced</li>
                <li>Wait time: avg of 3 forum reports in past 14d</li>
              </ul>
            </AIZone>
          </div>

          <div className="col">
            <div className="stripe" style={{ height: 180 }}>map · 5 Crown Princess</div>
            <div className="card muted">
              <div className="title-3" style={{ marginBottom: 6 }}>Peer reviews <span className="meta">· 14</span></div>
              <div className="row" style={{ marginBottom: 6 }}>
                <span style={{ fontSize: 22, fontWeight: 600 }}>4.6</span>
                <span className="meta">★★★★★ avg from 14 verified RNs</span>
              </div>
              <div className="meta">"Reliable Mandarin coverage" · "Daughters appreciate menu options"</div>
            </div>
            <div className="card">
              <div className="title-3" style={{ marginBottom: 6 }}>Used in {'  '}your saved cases</div>
              <div className="meta">3 of your cases referred here · 2 marked "used ✓"</div>
            </div>
          </div>
        </div>
      </div>
    </AppFrame>
  );
}

function ForumFeed() {
  const posts = [
    ['closure', 'crit', 'Parkdale Walk-In closed Wed–Fri this week', 'Priya R.', 'East Toronto OHT', '2h', '14 views · 3 replies'],
    ['question', '', 'Anyone used Albion Hub Tigrinya counselling? New as of last week.', 'Anwar T.', 'Mid-West Toronto OHT', '5h', '22 views · 6 replies'],
    ['tip', 'good', 'For PSW waitlists: cross-list with Spectrum Home + Carefirst', 'Sarah K.', 'East Toronto OHT', '1d', '88 views · 12 replies'],
    ['waitlist', 'warn', 'St. Joseph\'s CCAC PSW waitlist now ~6 weeks (was 3)', 'Mei L.', 'Hamilton OHT', '1d', '40 views · 4 replies'],
    ['question', '', 'Mandarin-speaking grief counsellor anywhere in GTA?', 'Jamie M.', 'Mid-West Toronto OHT', '2d', '67 views · 9 replies · ⚠ flagged shortage'],
  ];
  return (
    <AppFrame active="Forum">
      <div style={{ display: 'grid', gridTemplateColumns: '180px 1fr 240px', gap: 18 }}>
        <aside>
          <div className="eyebrow" style={{ marginBottom: 8 }}>filter</div>
          <div className="col" style={{ gap: 6 }}>
            {['All Ontario', '★ My OHT', 'East Toronto', 'Mid-West Toronto', 'Hamilton', 'North'].map((r, i) => (
              <div key={r} className="row" style={{ padding: '4px 8px', borderRadius: 4, background: i === 1 ? 'var(--paper-3)' : 'transparent', fontSize: 12 }}>
                <span style={{ fontSize: 11, color: i === 1 ? 'var(--ink)' : 'var(--ink-2)' }}>{r}</span>
              </div>
            ))}
          </div>
          <div className="eyebrow" style={{ margin: '14px 0 8px' }}>type · auto-tagged</div>
          <div className="col" style={{ gap: 4 }}>
            {[['closure', 12], ['waitlist', 28], ['tip', 44], ['question', 67], ['new program', 9]].map(([t, n]) => (
              <div key={t} className="between" style={{ fontSize: 11 }}><span>{t}</span><span className="meta">{n}</span></div>
            ))}
          </div>
        </aside>

        <div>
          <div className="between" style={{ marginBottom: 12 }}>
            <h1 className="title-1">Forum</h1>
            <button className="btn primary">＋ New post</button>
          </div>
          <AIZone style={{ marginBottom: 14, padding: 10 }}>
            <div className="eyebrow" style={{ color: 'var(--accent-ink)', marginBottom: 4 }}>trending in your OHT</div>
            <div style={{ fontSize: 12 }}>3 closures reported this week · "PSW waitlist" mentioned in 11 posts · Mandarin grief support unmet (9 nurses asking)</div>
          </AIZone>

          <div className="col" style={{ gap: 0 }}>
            {posts.map(([type, tone, title, who, oht, when, meta], i) => (
              <div key={i} style={{ padding: '12px 4px', borderBottom: '1px solid var(--stroke)' }}>
                <div className="row" style={{ gap: 8, marginBottom: 4 }}>
                  <span className={'badge' + (tone ? ' ' + tone : '')}>{type}</span>
                  <span className="meta">{oht} · {when}</span>
                </div>
                <div className="title-3" style={{ marginBottom: 3 }}>{title}</div>
                <div className="meta">{who} <span className="badge verified" style={{ marginLeft: 4, fontSize: 9 }}>RN ✓</span> · {meta}</div>
              </div>
            ))}
          </div>
        </div>

        <aside>
          <div className="card muted" style={{ padding: 10 }}>
            <div className="title-3" style={{ marginBottom: 4, fontSize: 12 }}>Top contributors · OHT</div>
            <div className="col" style={{ gap: 6, fontSize: 11 }}>
              {['Sarah K. — 88 helpful', 'Priya R. — 64', 'Anwar T. — 51'].map(s => <div key={s}>{s}</div>)}
            </div>
          </div>
          <div className="card" style={{ padding: 10, marginTop: 10 }}>
            <div className="title-3" style={{ fontSize: 12, marginBottom: 4 }}>How AI uses posts</div>
            <p className="meta" style={{ lineHeight: 1.5 }}>Closures &amp; waitlists are auto-extracted into the directory after a senior nurse reviews. Trends feed the shortage page.</p>
          </div>
        </aside>
      </div>
    </AppFrame>
  );
}

function ForumThread() {
  return (
    <AppFrame active="Forum">
      <div style={{ maxWidth: 760, margin: '0 auto' }}>
        <div className="meta" style={{ marginBottom: 4 }}>← Forum · East Toronto OHT</div>
        <div className="row" style={{ gap: 6, marginBottom: 8 }}>
          <span className="badge crit">closure</span>
          <span className="meta">2h · 14 views · 3 replies</span>
        </div>
        <h1 className="title-1" style={{ marginBottom: 14 }}>Parkdale Walk-In closed Wed–Fri this week</h1>

        <AIZone style={{ marginBottom: 14 }}>
          <div className="eyebrow" style={{ color: 'var(--accent-ink)', marginBottom: 4 }}>auto-extracted update — pending review</div>
          <div className="between">
            <div style={{ fontSize: 12 }}><b>Parkdale Walk-In</b> · status: <b>temporarily closed</b> · dates: <b>May 5–7</b></div>
            <div className="row"><button className="btn sm">Edit</button><button className="btn primary sm">Apply to directory</button></div>
          </div>
        </AIZone>

        <div className="card">
          <div className="row" style={{ gap: 8, marginBottom: 8 }}>
            <div className="avatar">PR</div>
            <div>
              <div className="row" style={{ gap: 6 }}>
                <span className="title-3" style={{ fontSize: 12 }}>Priya R.</span>
                <span className="badge verified">RN ✓</span>
                <span className="meta">East Toronto OHT · 2h</span>
              </div>
            </div>
          </div>
          <p style={{ fontSize: 13, lineHeight: 1.6 }}>Just spoke with their reception. Confirmed closure Wed–Fri (May 5–7) for staff training. Patients are being directed to St. Joseph's UCC and Toronto East General. They'll be back to normal hours Monday May 10.</p>
          <div className="row" style={{ marginTop: 10, gap: 12, fontSize: 11 }}>
            <span className="meta">👍 helpful · 12</span>
            <span className="meta">↩ reply</span>
            <span className="meta">⚠ flag</span>
          </div>
        </div>

        <div style={{ marginLeft: 24, marginTop: 12 }} className="col">
          {[
            ['Sarah K.', '1h', 'Confirmed via my Wed visit — sign on door says May 5–7. Thanks for posting!'],
            ['Anwar T.', '45m', 'Should we update the directory entry? Tagging an admin.'],
          ].map(([n, w, b]) => (
            <div key={n} className="card muted" style={{ padding: 10 }}>
              <div className="row" style={{ gap: 6, marginBottom: 4 }}>
                <div className="avatar" style={{ width: 22, height: 22 }}>{n[0]}</div>
                <span className="title-3" style={{ fontSize: 12 }}>{n}</span>
                <span className="badge verified">RN ✓</span>
                <span className="meta">{w}</span>
              </div>
              <p style={{ fontSize: 12 }}>{b}</p>
            </div>
          ))}
        </div>
      </div>
    </AppFrame>
  );
}

window.ResourceDetail = ResourceDetail;
window.ForumFeed = ForumFeed;
window.ForumThread = ForumThread;
