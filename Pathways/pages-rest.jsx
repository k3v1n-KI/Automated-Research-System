// Shortage public · Shortage admin · Oversight (3 tabs) · Onboarding · Submission · Notifications · Saved · About

function ShortagePublic() {
  return (
    <AppFrame active="Shortage">
      <div style={{ maxWidth: 880, margin: '0 auto' }}>
        <div className="eyebrow" style={{ marginBottom: 6 }}>Stat of the week · Apr 27 – May 3</div>
        <h1 className="title-1" style={{ marginBottom: 10 }}>Where Ontario's care system fell short this week</h1>
        <p className="muted-text" style={{ marginBottom: 18 }}>Aggregated from 412 nurse searches and 88 forum reports. Public summary; detail available to admins.</p>

        <div className="card accent" style={{ padding: 22, marginBottom: 18 }}>
          <div className="between" style={{ alignItems: 'flex-start' }}>
            <div>
              <div className="eyebrow" style={{ color: 'var(--accent-ink)', marginBottom: 6 }}>this week's signal</div>
              <div style={{ fontSize: 36, fontWeight: 600, letterSpacing: '-0.02em', lineHeight: 1.1 }}>
                <span style={{ color: 'var(--accent-ink)' }}>0</span> Mandarin grief counsellors found
              </div>
              <p className="muted-text" style={{ marginTop: 6, fontSize: 13 }}>across <b>14 nurse searches</b> in Mid-West &amp; East Toronto OHTs</p>
            </div>
            <div className="col" style={{ gap: 6 }}>
              <button className="btn">📥 Download as graphic</button>
              <button className="btn">✉ Letter template</button>
              <button className="btn ghost sm">copy stat</button>
            </div>
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12, marginBottom: 18 }}>
          {[
            ['Searches with no good match', '147', '+22 vs last wk'],
            ['Avg PSW wait reported', '5.2 wks', '+1.4'],
            ['Resources marked closed/closing', '11', 'across 6 OHTs'],
          ].map(([k, v, d]) => (
            <div key={k} className="card" style={{ padding: 14 }}>
              <div className="meta">{k}</div>
              <div style={{ fontSize: 26, fontWeight: 600, letterSpacing: '-0.02em' }}>{v}</div>
              <div className="meta" style={{ color: 'var(--accent-ink)' }}>{d}</div>
            </div>
          ))}
        </div>

        <div className="card">
          <div className="title-3" style={{ marginBottom: 10 }}>Top unmet needs · Ontario</div>
          <div className="col" style={{ gap: 8 }}>
            {[
              ['Mandarin grief counselling', 92, 'crit'],
              ['Trans-affirming primary care · North', 78, 'crit'],
              ['Tagalog PSW · Etobicoke', 64, 'warn'],
              ['Indigenous-led mental health · Sudbury', 58, 'warn'],
              ['Free dental · uninsured adults', 51, 'warn'],
            ].map(([k, v, t]) => (
              <div key={k}>
                <div className="between" style={{ fontSize: 12 }}>
                  <span>{k}</span>
                  <span className="meta">{v} unmet searches</span>
                </div>
                <div className={'score-bar ' + (t === 'crit' ? '' : 'warn')}><i style={{ width: v + '%' }} /></div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </AppFrame>
  );
}

function ShortageAdmin() {
  return (
    <AppFrame active="Shortage">
      <div className="between" style={{ marginBottom: 14 }}>
        <div>
          <div className="eyebrow">Admin detail · Mid-West Toronto OHT</div>
          <h1 className="title-1">Shortage map</h1>
        </div>
        <div className="row">
          <span className="chip">last 7d</span>
          <span className="chip accent">last 30d ▾</span>
          <button className="btn">Export monthly report</button>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1.4fr 1fr', gap: 16 }}>
        <div className="col">
          <div className="card" style={{ padding: 0 }}>
            <div className="stripe" style={{ height: 280, borderRadius: '6px 6px 0 0', borderBottom: 'none' }}>heatmap · GTA · need-vs-supply by service type</div>
            <div className="row" style={{ padding: 10, gap: 6, flexWrap: 'wrap' }}>
              {['mental health', 'PSW', 'dental', 'palliative', 'pediatric', 'newcomer', 'trans-affirming'].map(t => <span key={t} className="chip">{t}</span>)}
            </div>
          </div>

          <div className="card">
            <div className="title-3" style={{ marginBottom: 8 }}>Signals feeding the shortage view</div>
            <table className="tbl">
              <thead><tr><th>Signal</th><th>Source</th><th>Weight</th><th>Last 7d</th></tr></thead>
              <tbody>
                {[
                  ['Search with no good match', 'AI', '40%', '147'],
                  ['Recommendation rejected ("none fit")', 'User', '20%', '38'],
                  ['Forum waitlist reports', 'Forum', '20%', '28'],
                  ['Geographic gap (no resource within Xkm)', 'AI', '10%', '12 OHTs'],
                  ['Demographic gap (e.g. no Mandarin)', 'AI', '10%', '9'],
                ].map(r => <tr key={r[0]}>{r.map((c, i) => <td key={i} className={i > 1 ? 'num' : ''}>{c}</td>)}</tr>)}
              </tbody>
            </table>
          </div>
        </div>

        <div className="col">
          <AIZone>
            <div className="eyebrow" style={{ color: 'var(--accent-ink)', marginBottom: 4 }}>flagged for advocacy</div>
            <div style={{ fontSize: 12 }}><b>3 patterns</b> have crossed escalation thresholds this month. Recommend submitting to OHT planning meeting.</div>
            <button className="btn sm" style={{ marginTop: 8 }}>Generate brief →</button>
          </AIZone>
          <div className="card muted">
            <div className="title-3" style={{ marginBottom: 6, fontSize: 12 }}>Trend · Mandarin grief counsellors</div>
            <div className="stripe" style={{ height: 100 }}>line chart · 12wk</div>
            <div className="meta" style={{ marginTop: 6 }}>0 supply · 92 unmet searches · trending ↑</div>
          </div>
        </div>
      </div>
    </AppFrame>
  );
}

function Oversight() {
  const sb = [
    { items: ['Moderation queue', 'Directory', 'Data quality'] },
    { label: 'Filters', items: ['All Ontario', 'My OHT', 'Flagged', 'Stale (>90d)'] },
  ];
  const queue = [
    ['Albion Hub Tigrinya counselling', 'Forum extract · Anwar T.', 'new program', 'high'],
    ['Parkdale Walk-In · closed May 5–7', 'Forum extract · Priya R.', 'closure', 'high'],
    ['St. Joseph\'s CCAC · wait 6wk', 'Forum extract · Mei L.', 'waitlist update', 'med'],
    ['Health for All Mississauga', 'Web fetch · health.gov.on.ca', 'new resource', 'med'],
    ['Yee Hong meal program · Mandarin menu', 'Web fetch · yeehong.com', 'metadata enrich', 'low'],
  ];
  return (
    <AdminFrame active="Directory" sidebarItems={sb} sidebarActive="Moderation queue">
      <div className="between" style={{ marginBottom: 14 }}>
        <div>
          <h1 className="title-1">Moderation queue</h1>
          <span className="meta">5 pending · auto-extracted by AI from forum + web · awaiting senior nurse review</span>
        </div>
        <div className="row">
          <button className="btn">Bulk approve safe</button>
          <button className="btn primary">＋ Add resource manually</button>
        </div>
      </div>

      <div className="card" style={{ padding: 0, marginBottom: 16, overflow: 'hidden' }}>
        <table className="tbl">
          <thead><tr>
            <th style={{ width: 28 }}><input type="checkbox" /></th>
            <th>Proposed change</th>
            <th>Source</th>
            <th>Type</th>
            <th>AI confidence</th>
            <th>Priority</th>
            <th></th>
          </tr></thead>
          <tbody>
            {queue.map(([t, src, type, pri], i) => (
              <tr key={i}>
                <td><input type="checkbox" /></td>
                <td><div className="title-3" style={{ fontSize: 12 }}>{t}</div></td>
                <td className="meta">{src}</td>
                <td><span className="badge">{type}</span></td>
                <td><div className="score-bar good" style={{ width: 60 }}><i style={{ width: (60 + i * 8) + '%' }} /></div></td>
                <td><span className={'badge ' + (pri === 'high' ? 'crit' : pri === 'med' ? 'warn' : '')}>{pri}</span></td>
                <td><div className="row" style={{ gap: 4 }}><button className="btn sm">Edit</button><button className="btn primary sm">Approve</button><button className="btn ghost sm">✕</button></div></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14 }}>
        <div className="card">
          <div className="title-3" style={{ marginBottom: 8 }}>Directory · 2,431 entries</div>
          <div className="row" style={{ flexWrap: 'wrap', gap: 4, marginBottom: 8 }}>
            {['mental health (412)', 'home care (388)', 'meals (104)', 'shelter (76)', 'palliative (51)', '+18 more'].map(t => <span key={t} className="chip">{t}</span>)}
          </div>
          <Note>full browse + filter view; sortable; click any row to edit.</Note>
        </div>
        <div className="card">
          <div className="title-3" style={{ marginBottom: 8 }}>Data quality</div>
          <div className="col" style={{ gap: 6, fontSize: 12 }}>
            <div className="between"><span>Stale (>90d unverified)</span><span className="badge warn">142</span></div>
            <div className="between"><span>Missing language tags</span><span className="badge warn">88</span></div>
            <div className="between"><span>Missing hours</span><span className="badge crit">31</span></div>
            <div className="between"><span>Conflicting reports</span><span className="badge crit">9</span></div>
          </div>
        </div>
      </div>
    </AdminFrame>
  );
}

function Onboarding() {
  return (
    <div className="app" style={{ background: 'var(--paper-2)' }}>
      <div className="topnav"><div className="logo"><span className="logo-dot" />Pathways</div><div className="nav-spacer" /><span className="meta">Step 2 of 4</span></div>
      <div className="body" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <div style={{ maxWidth: 460 }}>
          <div className="eyebrow" style={{ marginBottom: 8 }}>Welcome</div>
          <h1 className="title-1" style={{ marginBottom: 8 }}>Where do you mostly work?</h1>
          <p className="muted-text" style={{ marginBottom: 18 }}>We'll prioritize resources in your Ontario Health Team. You can switch any time from the nav.</p>

          <div className="col" style={{ gap: 10 }}>
            <div className="card">
              <div className="meta" style={{ marginBottom: 4 }}>option 1</div>
              <input className="field" placeholder="Postal code (e.g. M6K 1J5)" />
              <div className="meta" style={{ marginTop: 4 }}>we map to your OHT silently</div>
            </div>
            <div className="card">
              <div className="meta" style={{ marginBottom: 4 }}>option 2</div>
              <select className="field"><option>Pick your OHT…</option><option>Mid-West Toronto OHT</option><option>East Toronto OHT</option></select>
            </div>
            <div className="card">
              <div className="meta" style={{ marginBottom: 4 }}>option 3</div>
              <div className="stripe" style={{ height: 130 }}>map of Ontario · click your area</div>
            </div>
          </div>

          <div className="row" style={{ marginTop: 18, justifyContent: 'space-between' }}>
            <button className="btn ghost">← back</button>
            <button className="btn primary">Continue →</button>
          </div>
          <Note style={{ marginTop: 14 }}><b>steps:</b> verify (employer email) → set OHT → pick familiar resource types → tour</Note>
        </div>
      </div>
    </div>
  );
}

function Submission() {
  return (
    <AppFrame active="Directory">
      <div style={{ maxWidth: 720, margin: '0 auto' }}>
        <h1 className="title-1" style={{ marginBottom: 4 }}>Add a resource</h1>
        <p className="muted-text" style={{ marginBottom: 18 }}>Goes to the moderation queue. AI fills in what it can from the website you provide.</p>

        <div className="card">
          <div className="col" style={{ gap: 12 }}>
            <div>
              <label className="label">Website or phone (optional, AI will pre-fill)</label>
              <div className="row"><input className="field" defaultValue="https://albionhub.org" style={{ flex: 1 }} /><button className="btn">✦ Pre-fill</button></div>
            </div>
            <AIZone style={{ padding: 10 }}>
              <div className="eyebrow" style={{ color: 'var(--accent-ink)', marginBottom: 6 }}>AI pulled from the site — review &amp; edit</div>
              <div className="col" style={{ gap: 8 }}>
                <div><label className="label">Name</label><input className="field" defaultValue="Albion Neighbourhood Hub" /></div>
                <div className="row" style={{ gap: 8 }}><div style={{ flex: 1 }}><label className="label">Category</label><select className="field"><option>Mental health · counselling</option></select></div><div style={{ flex: 1 }}><label className="label">OHT</label><select className="field"><option>Mid-West Toronto</option></select></div></div>
                <div><label className="label">Languages</label><div className="row" style={{ flexWrap: 'wrap', gap: 4 }}>{['Tigrinya', 'Amharic', 'EN', 'AR'].map(l => <span key={l} className="chip accent">{l} ✕</span>)}<span className="chip outline">＋ add</span></div></div>
                <div className="row" style={{ gap: 8 }}><div style={{ flex: 1 }}><label className="label">Hours</label><input className="field" defaultValue="Mon, Wed 10–6" /></div><div style={{ flex: 1 }}><label className="label">Cost</label><input className="field" defaultValue="Sliding scale" /></div></div>
                <div><label className="label">Cultural safety tags</label><div className="row" style={{ flexWrap: 'wrap', gap: 4 }}>{['Newcomer-focused', 'Trauma-informed', '2SLGBTQ+ affirming'].map(l => <span key={l} className="chip">＋ {l}</span>)}</div></div>
              </div>
            </AIZone>
            <div className="row" style={{ justifyContent: 'flex-end' }}><button className="btn ghost">Save draft</button><button className="btn primary">Submit for review →</button></div>
          </div>
        </div>
      </div>
    </AppFrame>
  );
}

function Notifications() {
  return (
    <AppFrame active="Home">
      <div style={{ maxWidth: 720, margin: '0 auto' }}>
        <h1 className="title-1" style={{ marginBottom: 14 }}>Notifications</h1>
        <div className="row" style={{ marginBottom: 12, gap: 6 }}>
          {['All', 'Resource changes', 'Forum replies', 'Cases', 'System'].map((t, i) => <span key={t} className={'chip' + (i === 0 ? ' accent' : '')}>{t}</span>)}
        </div>
        <div className="card" style={{ padding: 0 }}>
          {[
            ['🔄', 'Yee Hong waitlist updated to ~2 wks', 'affects your case C-1042', '15m', true],
            ['💬', 'Sarah K. replied to your post about Mandarin grief support', 'forum thread', '1h', true],
            ['⚠', 'Parkdale Walk-In closed May 5–7', 'affects 3 of your saved cases', '2h', true],
            ['✓', 'Your submission "Albion Hub" was approved', 'now live in directory', '1d', false],
            ['📊', 'Weekly shortage stat ready to share', 'Mandarin grief, 0 supply', '2d', false],
          ].map(([ic, t, sub, w, unread], i) => (
            <div key={i} style={{ padding: 12, borderBottom: '1px solid var(--stroke)', background: unread ? 'var(--accent-soft)' : 'transparent' }}>
              <div className="row" style={{ alignItems: 'flex-start' }}>
                <span style={{ width: 22, fontSize: 14 }}>{ic}</span>
                <div style={{ flex: 1 }}>
                  <div className="title-3" style={{ fontSize: 12 }}>{t}</div>
                  <div className="meta">{sub}</div>
                </div>
                <span className="meta">{w}</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </AppFrame>
  );
}

function SavedCases() {
  return (
    <AppFrame active="Saved">
      <div className="between" style={{ marginBottom: 14 }}>
        <div>
          <h1 className="title-1">Saved cases</h1>
          <span className="meta">23 total · 18 open · 5 closed · no patient names stored</span>
        </div>
        <button className="btn primary">＋ New case</button>
      </div>

      <div className="card" style={{ padding: 0 }}>
        <table className="tbl">
          <thead><tr><th>ID</th><th>Label</th><th>Tags</th><th>Recommended</th><th>Status</th><th>Updated</th><th></th></tr></thead>
          <tbody>
            {[
              ['C-1042', 'Mrs. K — home care', 'Mandarin · 65+ · Scarborough', 'Yee Hong (used ✓)', 'open', '2h'],
              ['C-1039', 'mental health newcomer', 'Tigrinya · no OHIP', 'Albion Hub', 'open', '1d'],
              ['C-1031', 'postpartum support', 'Tagalog · Etobicoke', 'Spectrum (used ✓)', 'closed', '3d'],
              ['C-1024', 'palliative in-home', 'Tagalog · 80+', 'Hospice Toronto', 'follow-up', '5d'],
              ['C-0998', 'trans-affirming GP', 'North · 18-24', '— no good match', 'shortage filed', '12d'],
            ].map(r => (
              <tr key={r[0]}>
                <td className="num">{r[0]}</td>
                <td>{r[1]} <span className="meta">(label only — not patient name)</span></td>
                <td className="meta">{r[2]}</td>
                <td>{r[3]}</td>
                <td><span className={'badge' + (r[4].includes('shortage') ? ' crit' : r[4] === 'closed' ? ' good' : '')}>{r[4]}</span></td>
                <td className="meta">{r[5]}</td>
                <td><button className="btn sm">→</button></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <Note style={{ marginTop: 12 }}><b>privacy:</b> labels are nurse-typed shorthand, never auto-populated from notes. Cases auto-purge after 180d unless pinned.</Note>
    </AppFrame>
  );
}

function About() {
  return (
    <AppFrame active="Home">
      <div style={{ maxWidth: 680, margin: '0 auto', fontFamily: 'var(--font-serif)' }}>
        <div className="eyebrow" style={{ fontFamily: 'var(--font-mono)', marginBottom: 12 }}>Mission</div>
        <h1 style={{ fontSize: 30, fontWeight: 600, lineHeight: 1.2, letterSpacing: '-0.02em', marginBottom: 18 }}>Your shared notebook, smarter — for the nurses Ontario quietly relies on.</h1>
        <p style={{ fontSize: 15, lineHeight: 1.65, marginBottom: 14, color: 'var(--ink-2)' }}>Pathways exists because the best healthcare resource directory in Ontario today is the dog-eared notebook in a referral nurse's pocket. We want to honour that knowledge — make it easier to share, easier to find, and visible enough that decision-makers can see where the gaps really are.</p>
        <p style={{ fontSize: 15, lineHeight: 1.65, marginBottom: 14, color: 'var(--ink-2)' }}>We never store patient identifiers. AI helps you go faster — it never replaces your judgement. Every claim shows its source.</p>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12, margin: '24px 0', fontFamily: 'var(--font-ui)' }}>
          {[['1.', 'Honour the knowledge', 'Verified RNs hold the pen — AI assists, humans decide.'], ['2.', 'Privacy by default', 'No patient names. Ever.'], ['3.', 'Make shortage visible', 'Every unmet search becomes evidence.']].map(([n, t, b]) => (
            <div key={n} className="card">
              <div className="eyebrow" style={{ marginBottom: 6 }}>{n}</div>
              <div className="title-3">{t}</div>
              <p className="meta" style={{ marginTop: 6, lineHeight: 1.5 }}>{b}</p>
            </div>
          ))}
        </div>

        <div style={{ fontFamily: 'var(--font-ui)' }}>
          <div className="eyebrow" style={{ marginBottom: 8 }}>Acknowledgement</div>
          <p className="muted-text" style={{ fontSize: 12, lineHeight: 1.6 }}>We work on the traditional territories of many Indigenous nations across what is now Ontario. Indigenous-led services in Pathways are tagged and prioritized when culturally appropriate; we do not direct patients to non-Indigenous services in their place.</p>
        </div>
      </div>
    </AppFrame>
  );
}

window.ShortagePublic = ShortagePublic;
window.ShortageAdmin = ShortageAdmin;
window.Oversight = Oversight;
window.Onboarding = Onboarding;
window.Submission = Submission;
window.Notifications = Notifications;
window.SavedCases = SavedCases;
window.About = About;
