// Home / Dashboard — 3 variations
// All share AppFrame chrome; bodies differ.

// A · Two-path split (literal "Find for a patient" / "Browse the directory")
function HomeA() {
  return (
    <AppFrame active="Home">
      <div style={{ maxWidth: 880, margin: '0 auto' }}>
        <div className="eyebrow" style={{ marginBottom: 8 }}>Tuesday · 2:14 pm</div>
        <h1 className="title-1" style={{ marginBottom: 4 }}>Hi Jamie. What's the case?</h1>
        <p className="muted-text" style={{ marginBottom: 28 }}>Two ways to start, depending on whether you have a specific patient in mind.</p>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 28 }}>
          <div className="card accent" style={{ padding: 22 }}>
            <div className="eyebrow" style={{ color: 'var(--accent-ink)', marginBottom: 10 }}>For a patient</div>
            <h2 className="title-2" style={{ marginBottom: 6 }}>Find for a patient</h2>
            <p className="muted-text" style={{ fontSize: 12, marginBottom: 16 }}>Paste your notes or pick a few tags. AI ranks resources across fit, distance, cost.</p>
            <button className="btn primary">Start a case →</button>
          </div>
          <div className="card" style={{ padding: 22 }}>
            <div className="eyebrow" style={{ marginBottom: 10 }}>Just looking</div>
            <h2 className="title-2" style={{ marginBottom: 6 }}>Browse the directory</h2>
            <p className="muted-text" style={{ fontSize: 12, marginBottom: 16 }}>Search by category, region, language, or keyword.</p>
            <button className="btn">Open directory →</button>
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          <div className="card muted">
            <div className="between" style={{ marginBottom: 10 }}>
              <span className="title-3">Recent in your OHT</span>
              <a className="meta" style={{ color: 'var(--accent-ink)' }}>see all →</a>
            </div>
            <div className="col" style={{ gap: 10 }}>
              <div className="row" style={{ gap: 8, alignItems: 'flex-start' }}>
                <span className="badge crit">closed</span>
                <span style={{ fontSize: 12 }}>Parkdale Walk-In is closed Wed–Fri this week</span>
              </div>
              <div className="row" style={{ gap: 8, alignItems: 'flex-start' }}>
                <span className="badge warn">waitlist</span>
                <span style={{ fontSize: 12 }}>St. Joseph's CCAC: PSW waitlist now ~6 weeks</span>
              </div>
              <div className="row" style={{ gap: 8, alignItems: 'flex-start' }}>
                <span className="badge good">new</span>
                <span style={{ fontSize: 12 }}>Tigrinya-language counselling at Albion Hub (Mon)</span>
              </div>
            </div>
          </div>

          <div className="card muted">
            <div className="between" style={{ marginBottom: 10 }}>
              <span className="title-3">Your recent cases</span>
              <a className="meta" style={{ color: 'var(--accent-ink)' }}>saved (3) →</a>
            </div>
            <div className="col" style={{ gap: 8, fontSize: 12 }}>
              {[
                ['Case #C-1042', 'Home care, Mandarin, 65+', '2h ago'],
                ['Case #C-1039', 'Mental health, no OHIP', 'yesterday'],
                ['Case #C-1031', 'Postpartum support, Etobicoke', 'Mon'],
              ].map(([id, sum, when]) => (
                <div key={id} className="between" style={{ padding: '4px 0', borderBottom: '1px dashed var(--stroke)' }}>
                  <div>
                    <div className="meta">{id}</div>
                    <div>{sum}</div>
                  </div>
                  <span className="meta">{when}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </AppFrame>
  );
}

// B · Search-first (big input is the hero)
function HomeB() {
  return (
    <AppFrame active="Home">
      <div style={{ maxWidth: 760, margin: '40px auto 0' }}>
        <h1 className="title-1" style={{ marginBottom: 4, textAlign: 'center' }}>Who are you helping today?</h1>
        <p className="muted-text" style={{ textAlign: 'center', marginBottom: 24 }}>Type a few words, paste your notes, or just tap tags.</p>

        <div className="ai-zone" style={{ padding: 0 }}>
          <textarea
            className="field"
            style={{ minHeight: 120, border: 0, background: 'transparent', fontSize: 14, padding: 14 }}
            placeholder="e.g. 76yo Mandarin-speaking, lives alone in Scarborough, recovering from hip surgery, no family nearby, OHIP only…"
            defaultValue=""
          />
          <div className="row" style={{ padding: '8px 12px', borderTop: '1px dashed var(--accent)', background: 'var(--paper)' }}>
            <button className="btn sm">⌘ Voice</button>
            <button className="btn sm">＋ Tags</button>
            <span className="meta" style={{ flex: 1, textAlign: 'center' }}>no patient identifiers — just structured needs</span>
            <button className="btn primary sm">Find resources →</button>
          </div>
        </div>

        <div className="row" style={{ flexWrap: 'wrap', gap: 6, marginTop: 14, justifyContent: 'center' }}>
          {['PSW', 'Mandarin', 'No OHIP', 'Mental health', '65+', 'Etobicoke', 'Trans-affirming', 'Indigenous-led', 'Newcomer', '24/7'].map(t => (
            <span key={t} className="chip">＋ {t}</span>
          ))}
        </div>

        <div style={{ marginTop: 32, display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12 }}>
          {[
            ['Browse directory', '2,400+ Ontario resources'],
            ['Forum', '14 new posts in your OHT'],
            ['Shortage', 'Stat of the week'],
          ].map(([t, s]) => (
            <div key={t} className="card" style={{ padding: 12 }}>
              <div className="title-3">{t}</div>
              <div className="meta" style={{ marginTop: 4 }}>{s}</div>
            </div>
          ))}
        </div>
      </div>
    </AppFrame>
  );
}

// C · Dashboard-rich (signal-dense for power users / quiet days)
function HomeC() {
  return (
    <AppFrame active="Home">
      <div className="between" style={{ marginBottom: 18 }}>
        <div>
          <h1 className="title-1">Today in your OHT</h1>
          <span className="meta">Mid-West Toronto · 12 nurses active · 3 new updates</span>
        </div>
        <div className="row">
          <button className="btn">Browse directory</button>
          <button className="btn primary">＋ Find for a patient</button>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1.4fr 1fr', gap: 16 }}>
        <div className="col">
          <AIZone>
            <div className="eyebrow" style={{ color: 'var(--accent-ink)', marginBottom: 6 }}>For your shifts</div>
            <div className="title-3" style={{ marginBottom: 6 }}>3 things that may affect your patients today</div>
            <ul style={{ margin: 0, paddingLeft: 18, fontSize: 12, lineHeight: 1.7, color: 'var(--ink-2)' }}>
              <li><b style={{ color: 'var(--ink)' }}>Parkdale Walk-In</b> closed Wed–Fri — 8 of your saved cases routed there</li>
              <li><b style={{ color: 'var(--ink)' }}>PSW waitlist at St. Joseph's</b> jumped from 3 → 6 weeks (3 forum reports)</li>
              <li>New <b style={{ color: 'var(--ink)' }}>Tigrinya counselling</b> at Albion Hub — fits 2 of your prior cases</li>
            </ul>
          </AIZone>

          <div className="card">
            <div className="between" style={{ marginBottom: 10 }}>
              <span className="title-3">Your saved cases</span>
              <a className="meta" style={{ color: 'var(--accent-ink)' }}>all 23 →</a>
            </div>
            <table className="tbl">
              <thead><tr><th>ID</th><th>Summary</th><th>Status</th><th>Updated</th></tr></thead>
              <tbody>
                {[
                  ['C-1042', 'Home care · Mandarin · 65+', 'open', '2h'],
                  ['C-1039', 'Mental health · no OHIP', 'open', '1d'],
                  ['C-1031', 'Postpartum support · Etobicoke', 'closed ✓', '3d'],
                  ['C-1024', 'Palliative · Tagalog · in-home', 'follow-up', '5d'],
                ].map(([id, s, st, u]) => (
                  <tr key={id}>
                    <td className="num">{id}</td>
                    <td>{s}</td>
                    <td><span className="badge">{st}</span></td>
                    <td className="num">{u}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="col">
          <div className="card muted">
            <div className="title-3" style={{ marginBottom: 8 }}>Forum — your OHT</div>
            <div className="col" style={{ gap: 10, fontSize: 12 }}>
              <div>
                <div className="row" style={{ gap: 6 }}>
                  <span className="badge crit">closure</span>
                  <span className="meta">2h · Priya R.</span>
                </div>
                <div style={{ marginTop: 3 }}>Parkdale Walk-In closed Wed–Fri this week</div>
              </div>
              <div>
                <div className="row" style={{ gap: 6 }}>
                  <span className="badge">question</span>
                  <span className="meta">5h · Anwar T.</span>
                </div>
                <div style={{ marginTop: 3 }}>Anyone used the new Albion Hub Tigrinya program?</div>
              </div>
              <div>
                <div className="row" style={{ gap: 6 }}>
                  <span className="badge good">tip</span>
                  <span className="meta">1d · Sarah K.</span>
                </div>
                <div style={{ marginTop: 3 }}>For PSW waitlists, try cross-listing with Spectrum Home</div>
              </div>
            </div>
            <button className="btn sm" style={{ marginTop: 12, width: '100%' }}>Open forum →</button>
          </div>

          <div className="card">
            <div className="title-3" style={{ marginBottom: 6 }}>Shortage — stat of the week</div>
            <div className="ai-zone" style={{ padding: 14 }}>
              <div className="title-1" style={{ fontSize: 26, color: 'var(--accent-ink)' }}>0</div>
              <div style={{ fontSize: 12, color: 'var(--ink-2)' }}>Mandarin-speaking grief counsellors found in your OHT this month — across <b>14 nurse searches</b>.</div>
              <div className="meta" style={{ marginTop: 8 }}>tap to share as graphic →</div>
            </div>
          </div>
        </div>
      </div>
    </AppFrame>
  );
}

window.HomeA = HomeA;
window.HomeB = HomeB;
window.HomeC = HomeC;
