// Results — 3 variations.

// Mock dataset (consistent across variations).
const RES = [
  { name: 'Yee Hong Centre — Scarborough', cat: 'PSW · home care', dist: '3.2 km', wait: '2 wks', cost: 'Free (OHIP)', langs: ['Mandarin', 'Cantonese', 'EN'], hours: '24/7', verified: '3d ago', score: 96, tags: ['Cantonese-speaking PSWs', '24/7 line'] },
  { name: 'Spectrum Home Health', cat: 'PSW · home care', dist: '5.8 km', wait: '~1 wk', cost: '$$ private', langs: ['EN', 'Mandarin'], hours: 'Mon–Sat', verified: '1d ago', score: 88, tags: ['Mandarin coordinator'] },
  { name: 'Mandarin Meals on Wheels', cat: 'meal support', dist: '4.1 km', wait: 'next day', cost: 'Sliding scale', langs: ['Mandarin', 'EN'], hours: 'Mon–Fri', verified: '12d', score: 91, tags: ['culturally-specific menu'] },
  { name: 'Toronto CCAC East', cat: 'home care', dist: '6.0 km', wait: '6 wks', cost: 'Free (OHIP)', langs: ['EN'], hours: 'Mon–Fri', verified: '6h ago', score: 72, tags: [] },
  { name: 'Carefirst Seniors', cat: 'PSW · seniors', dist: '7.4 km', wait: '3 wks', cost: 'Free + private', langs: ['Mandarin', 'Cantonese'], hours: 'Mon–Sat', verified: '2d', score: 85, tags: ['Mandarin-led'] },
];

function CaseStrip() {
  return (
    <div className="card accent" style={{ padding: 10, marginBottom: 14 }}>
      <div className="between">
        <div className="row" style={{ flexWrap: 'wrap', gap: 6 }}>
          <span className="meta">case:</span>
          {['76yo', 'Mandarin', 'Scarborough', 'OHIP', 'PSW', 'Meal', 'Mobility ↓'].map(t => (
            <span key={t} className="chip accent">{t}</span>
          ))}
        </div>
        <button className="btn sm">edit case ✎</button>
      </div>
    </div>
  );
}

// A · Triaged tiers — 1 best fit + alternates by dimension
function ResultsA() {
  return (
    <AppFrame active="Find for patient">
      <CaseStrip />
      <div className="between" style={{ marginBottom: 12 }}>
        <h1 className="title-2">3 best matches · alternates ranked by lens →</h1>
        <div className="row">
          <span className="chip">all dimensions</span>
          <span className="chip">closest</span>
          <span className="chip">cheapest</span>
          <span className="chip">soonest</span>
          <span className="chip">cultural fit</span>
        </div>
      </div>

      <AIZone style={{ marginBottom: 16 }}>
        <div className="eyebrow" style={{ color: 'var(--accent-ink)', marginBottom: 6 }}>why these · top 3 best fit</div>
        <div style={{ fontSize: 12, color: 'var(--ink-2)' }}>
          Filtered to Mandarin-speaking PSW + meal support within East Toronto OHT, OHIP-eligible, mobility-aware. Ranked by overall fit. <a style={{ color: 'var(--accent-ink)' }}>see reasoning →</a>
        </div>
      </AIZone>

      <div className="col" style={{ gap: 10, marginBottom: 22 }}>
        {RES.slice(0, 3).map((r, i) => (
          <div key={r.name} className="card" style={{ padding: 14 }}>
            <div className="between" style={{ marginBottom: 6 }}>
              <div className="row" style={{ gap: 8 }}>
                <span className="badge acc">#{i + 1} best fit</span>
                <span className="title-3">{r.name}</span>
                <span className="meta">{r.cat}</span>
              </div>
              <div className="row">
                <span className="meta">verified {r.verified}</span>
                <span className="badge good">{r.score}% match</span>
              </div>
            </div>
            <div className="row" style={{ flexWrap: 'wrap', gap: 12, fontSize: 11.5, color: 'var(--ink-2)', marginBottom: 8 }}>
              <span>📍 {r.dist}</span><span>⏱ wait {r.wait}</span><span>💲 {r.cost}</span>
              <span>🗣 {r.langs.join(' · ')}</span><span>🕒 {r.hours}</span>
            </div>
            <div className="ai-zone" style={{ padding: 8, fontSize: 11.5 }}>
              <b>matches because</b> Mandarin-speaking PSWs on staff, OHIP-covered, 3.2 km from patient postal, current waitlist 2 weeks · <a style={{ color: 'var(--accent-ink)' }}>3 sources</a>
            </div>
            <div className="row" style={{ marginTop: 10, justifyContent: 'flex-end', gap: 6 }}>
              <button className="btn sm">Save</button>
              <button className="btn sm">PDF for patient</button>
              <button className="btn primary sm">Open →</button>
            </div>
          </div>
        ))}
      </div>

      <div className="eyebrow" style={{ marginBottom: 8 }}>or, ranked by other lenses</div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 10 }}>
        {[
          ['📍 Closest', RES[0]],
          ['💲 Cheapest', RES[2]],
          ['⏱ Soonest', RES[1]],
        ].map(([t, r]) => (
          <div key={t} className="card muted" style={{ padding: 10 }}>
            <div className="meta" style={{ marginBottom: 4 }}>{t}</div>
            <div className="title-3" style={{ fontSize: 12 }}>{r.name}</div>
            <div className="meta" style={{ marginTop: 4 }}>{r.dist} · {r.wait} · {r.cost}</div>
          </div>
        ))}
      </div>
    </AppFrame>
  );
}

// B · Comparison table (the column-per-dimension you called for)
function ResultsB() {
  const cols = ['Match', 'Distance', 'Wait', 'Cost', 'Language', 'Hours', 'Verified'];
  return (
    <AppFrame active="Find for patient">
      <CaseStrip />
      <div className="between" style={{ marginBottom: 10 }}>
        <h1 className="title-2">5 matches — compare</h1>
        <div className="row">
          <span className="meta">sort:</span>
          <span className="chip accent">Best fit ▾</span>
          <button className="btn sm">＋ column</button>
          <button className="btn sm">Export CSV</button>
        </div>
      </div>
      <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
        <table className="tbl">
          <thead>
            <tr>
              <th style={{ width: 220 }}>Resource</th>
              {cols.map(c => <th key={c}>{c}</th>)}
              <th></th>
            </tr>
          </thead>
          <tbody>
            {RES.map((r, i) => (
              <tr key={r.name}>
                <td>
                  <div className="title-3" style={{ fontSize: 12 }}>{r.name}</div>
                  <div className="meta">{r.cat}</div>
                </td>
                <td><div className="score-bar good" style={{ width: 60 }}><i style={{ width: r.score + '%' }} /></div><div className="meta">{r.score}%</div></td>
                <td className="num">{r.dist}</td>
                <td className="num">{r.wait}</td>
                <td>{r.cost}</td>
                <td>{r.langs.join(', ')}</td>
                <td>{r.hours}</td>
                <td className="meta">{r.verified}</td>
                <td><button className="btn sm">→</button></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="row" style={{ marginTop: 10, gap: 8 }}>
        <Note><b>note:</b> column headers click-to-sort. table view is the densest, best for desktop power users.</Note>
      </div>
    </AppFrame>
  );
}

// C · Map + list split
function ResultsC() {
  return (
    <AppFrame active="Find for patient" padding={false}>
      <div style={{ padding: '14px 22px' }}>
        <CaseStrip />
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '380px 1fr', borderTop: '1px solid var(--stroke)', flex: 1, minHeight: 480 }}>
        <div className="scroll-y" style={{ borderRight: '1px solid var(--stroke)' }}>
          <div className="row" style={{ padding: '10px 14px', gap: 6, borderBottom: '1px solid var(--stroke)', background: 'var(--paper-2)' }}>
            <span className="chip accent">Best fit</span>
            <span className="chip">Closest</span>
            <span className="chip">Cheapest</span>
            <span className="chip">Soonest</span>
          </div>
          {RES.map((r, i) => (
            <div key={r.name} style={{ padding: 12, borderBottom: '1px solid var(--stroke)', cursor: 'pointer', background: i === 0 ? 'var(--accent-soft)' : 'transparent' }}>
              <div className="between" style={{ marginBottom: 4 }}>
                <span className="title-3" style={{ fontSize: 12 }}>{i + 1}. {r.name}</span>
                <span className="badge good">{r.score}%</span>
              </div>
              <div className="meta">{r.dist} · {r.wait} · {r.cost}</div>
              <div className="meta" style={{ color: 'var(--accent-ink)', marginTop: 4 }}>🗣 {r.langs.join(' · ')}</div>
            </div>
          ))}
        </div>
        <div style={{ position: 'relative' }}>
          <div className="stripe" style={{ position: 'absolute', inset: 0, borderRadius: 0, fontSize: 11 }}>map · scarborough · 5 pins</div>
          {[[120, 90], [220, 140], [180, 220], [310, 160], [260, 250]].map(([x, y], i) => (
            <div key={i} style={{
              position: 'absolute', left: x, top: y, transform: 'translate(-50%,-50%)',
              width: 22, height: 22, borderRadius: '50% 50% 50% 0', background: i === 0 ? 'var(--accent)' : 'var(--paper)',
              border: '2px solid var(--accent)', rotate: '-45deg',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontSize: 9, fontWeight: 700, color: i === 0 ? 'white' : 'var(--accent-ink)',
            }}><span style={{ rotate: '45deg' }}>{i + 1}</span></div>
          ))}
        </div>
      </div>
    </AppFrame>
  );
}

window.ResultsA = ResultsA; window.ResultsB = ResultsB; window.ResultsC = ResultsC;
