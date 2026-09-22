// Shortage Admin v2 — for OHT planners
// Two screens: overview dashboard + single-signal drill-down

function ShortageAdminV2() {
  return (
    <AppFrame active="Shortage">
      {/* Header strip — frame for OHT planner */}
      <div className="between" style={{ marginBottom: 14 }}>
        <div>
          <div className="eyebrow">Mid-West Toronto OHT · Admin · Apr 2026</div>
          <h1 className="title-1">Shortage</h1>
          <span className="meta">12 active findings · 3 emerging this week · last sync 12 min ago</span>
        </div>
        <div className="row">
          <span className="chip">last 7d</span>
          <span className="chip accent">last 30d ▾</span>
          <span className="chip">last 12mo</span>
          <span className="region-pill" style={{ marginLeft: 6 }}>vs <b>Ontario avg ▾</b></span>
          <button className="btn">⟱ Brief</button>
          <button className="btn">⎙ Slides</button>
        </div>
      </div>

      {/* KPI strip — daily glance */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 10, marginBottom: 14 }}>
        {[
          ['Active findings', '12', '+3 vs last mo', 'crit', '▲'],
          ['Searches with no match', '147', '+22 vs last mo', 'warn', '▲'],
          ['Avg PSW wait', '5.2 wks', '+1.4 wks', 'warn', '▲'],
          ['Resolved this month', '2', 'celebrated ✓', 'good', '✓'],
        ].map(([k, v, d, t, ic]) => (
          <div key={k} className="card" style={{ padding: 12 }}>
            <div className="meta">{k}</div>
            <div className="row" style={{ alignItems: 'baseline', gap: 6 }}>
              <span style={{ fontSize: 24, fontWeight: 600, letterSpacing: '-0.02em' }}>{v}</span>
              <span className={'badge ' + t}>{ic} {d}</span>
            </div>
          </div>
        ))}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1.45fr 1fr', gap: 14 }}>
        {/* Findings table — the work surface */}
        <div className="card" style={{ padding: 0 }}>
          <div className="between" style={{ padding: '10px 12px', borderBottom: '1px solid var(--stroke)' }}>
            <div className="title-3">Findings · what's unmet here</div>
            <div className="row">
              <span className="chip">all</span>
              <span className="chip accent">emerging (3)</span>
              <span className="chip">acknowledged (5)</span>
              <span className="chip">addressing (2)</span>
              <span className="chip">resolved (2)</span>
            </div>
          </div>
          <table className="tbl">
            <thead><tr>
              <th>Need</th><th>Signal</th><th>Trend</th><th>Confidence</th><th>Status</th><th>vs ON</th><th></th>
            </tr></thead>
            <tbody>
              {[
                ['Mandarin grief counselling', '92 unmet · 14 nurses', 'up', '88%', 'emerging', '8× higher', 'crit'],
                ['Trans-affirming GP · north of OHT', '51 unmet · 9 nurses', 'up', '74%', 'emerging', '3× higher', 'crit'],
                ['Tagalog PSW · Etobicoke', '64 unmet · 11 nurses', 'flat', '81%', 'acknowledged', '2× higher', 'warn'],
                ['Newcomer dental · uninsured', '38 unmet · 7 nurses', 'up', '69%', 'acknowledged', '1.5×', 'warn'],
                ['Indigenous-led mental health', '58 unmet · 6 nurses', 'flat', '62%', 'addressing', '— no avg', ''],
                ['Free legal advocacy · seniors', '12 unmet · 4 nurses', 'down', '44%', 'data check', 'low n', ''],
                ['Postpartum night nursing', 'resolved · 2 new providers', 'down', '92%', 'resolved ✓', '—', 'good'],
              ].map(([k, sig, tr, c, st, vs, tone], i) => (
                <tr key={i} style={{ cursor: 'pointer' }}>
                  <td><div className="title-3" style={{ fontSize: 12 }}>{k}</div></td>
                  <td className="meta">{sig}</td>
                  <td><span style={{ color: tr === 'up' ? 'var(--crit)' : tr === 'down' ? 'var(--good)' : 'var(--ink-3)' }}>{tr === 'up' ? '↗' : tr === 'down' ? '↘' : '→'}</span></td>
                  <td><div className="row" style={{ gap: 4 }}><div className="score-bar" style={{ width: 38 }}><i style={{ width: c }} /></div><span className="meta">{c}</span></div></td>
                  <td><span className={'badge' + (tone ? ' ' + tone : '')}>{st}</span></td>
                  <td className="meta">{vs}</td>
                  <td><button className="btn sm">→</button></td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="row" style={{ padding: 10, gap: 6, borderTop: '1px solid var(--stroke)', background: 'var(--paper-2)' }}>
            <span className="meta">selection:</span>
            <button className="btn sm">tag…</button>
            <button className="btn sm">add note</button>
            <button className="btn sm">subscribe alerts</button>
            <span className="nav-spacer" />
            <button className="btn sm">CSV</button>
          </div>
        </div>

        {/* Right column — comparisons, alerts, evidence quality */}
        <div className="col">
          <AIZone>
            <div className="eyebrow" style={{ color: 'var(--accent-ink)', marginBottom: 6 }}>this week's signal</div>
            <div className="title-3" style={{ marginBottom: 4 }}>Mandarin grief counselling has crossed escalation threshold</div>
            <p className="meta" style={{ marginBottom: 8, lineHeight: 1.5 }}>92 unmet searches in 30d (8× Ontario avg). Confidence 88% from 14 distinct nurses. Recommend tagging <b style={{ color: 'var(--ink)' }}>acknowledged</b> &amp; bringing to OHT planning meeting.</p>
            <div className="row"><button className="btn primary sm">Generate brief →</button><button className="btn sm">Open finding</button></div>
          </AIZone>

          <div className="card">
            <div className="title-3" style={{ marginBottom: 8 }}>Compare · this OHT vs…</div>
            <div className="col" style={{ gap: 10 }}>
              {[
                ['Ontario average', '+62% findings', 'crit'],
                ['Last month', '+3 findings', 'warn'],
                ['Similar OHTs (peer set of 6)', '+24% findings', 'warn'],
              ].map(([k, v, t]) => (
                <div key={k}>
                  <div className="between" style={{ fontSize: 12 }}><span>{k}</span><span className={'badge ' + t}>{v}</span></div>
                  <div className="stripe" style={{ height: 22, marginTop: 4 }}>bar comparison</div>
                </div>
              ))}
              <div className="meta" style={{ paddingTop: 4 }}>peer set: similar by population, density, % newcomers — <a style={{ color: 'var(--accent-ink)' }}>view list</a></div>
            </div>
          </div>

          <div className="card muted">
            <div className="between" style={{ marginBottom: 6 }}>
              <span className="title-3">Evidence quality</span>
              <span className="meta">how solid is this picture?</span>
            </div>
            <div className="col" style={{ gap: 6, fontSize: 12 }}>
              <div className="between"><span>Findings with low sample (n&lt;5)</span><span className="badge">2</span></div>
              <div className="between"><span>Stale signals (no report 6wk+)</span><span className="badge warn">1</span></div>
              <div className="between"><span>Possible double-counts</span><span className="badge">3</span></div>
              <div className="between"><span>Findings with no supply data</span><span className="badge crit">4</span></div>
            </div>
            <Note style={{ marginTop: 8 }}><b>honest unknown:</b> 4 findings have no public supply data. We say "unknown" rather than guess.</Note>
          </div>

          <div className="card">
            <div className="title-3" style={{ marginBottom: 8 }}>Alerts</div>
            <div className="col" style={{ gap: 6, fontSize: 12 }}>
              <div className="between"><span>Threshold: any need crosses 50 unmet/mo</span><span className="badge good">on</span></div>
              <div className="between"><span>Trend: +50% growth in 30d</span><span className="badge good">on</span></div>
              <div className="between"><span>New finding detected</span><span className="badge good">on</span></div>
              <div className="between"><span>Weekly digest (Mondays)</span><span className="badge">off</span></div>
            </div>
            <button className="btn sm" style={{ marginTop: 8 }}>configure</button>
          </div>
        </div>
      </div>
    </AppFrame>
  );
}

function ShortageDrilldown() {
  return (
    <AppFrame active="Shortage">
      <div className="meta" style={{ marginBottom: 4 }}>← Shortage · Mid-West Toronto OHT</div>
      <div className="between" style={{ marginBottom: 6, alignItems: 'flex-start' }}>
        <div>
          <div className="row" style={{ gap: 8, marginBottom: 4 }}>
            <span className="badge crit">emerging</span>
            <span className="meta">finding #SH-204 · first detected 3 weeks ago</span>
          </div>
          <h1 className="title-1">Mandarin-speaking grief counselling</h1>
          <span className="meta">demographic: 65+ recent loss · language: Mandarin · category: mental health</span>
        </div>
        <div className="row">
          <button className="btn">⟱ Advocacy brief (PDF)</button>
          <button className="btn">⎙ Slide</button>
          <button className="btn">＋ Note</button>
          <button className="btn primary">Tag status ▾</button>
        </div>
      </div>

      {/* Lifecycle stepper */}
      <div className="card muted" style={{ padding: 12, marginBottom: 14 }}>
        <div className="row" style={{ gap: 0 }}>
          {[
            ['Emerging', true, 'apr 12'],
            ['Acknowledged', false, '—'],
            ['Being addressed', false, '—'],
            ['Resolved', false, '—'],
          ].map(([s, on, d], i) => (
            <Fragment key={s}>
              <div style={{ flex: 1, textAlign: 'center' }}>
                <div style={{
                  width: 24, height: 24, borderRadius: '50%',
                  background: on ? 'var(--accent)' : 'var(--paper)',
                  border: '2px solid ' + (on ? 'var(--accent)' : 'var(--stroke-2)'),
                  margin: '0 auto 4px',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  color: on ? 'white' : 'var(--ink-3)', fontSize: 11, fontWeight: 600,
                }}>{i + 1}</div>
                <div className="title-3" style={{ fontSize: 11, color: on ? 'var(--ink)' : 'var(--ink-3)' }}>{s}</div>
                <div className="meta">{d}</div>
              </div>
              {i < 3 && <div style={{ flex: 1, height: 2, background: 'var(--stroke-2)', alignSelf: 'flex-start', marginTop: 13 }} />}
            </Fragment>
          ))}
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1.4fr 1fr', gap: 14 }}>
        <div className="col">
          {/* Headline + AI summary */}
          <AIZone style={{ padding: 14 }}>
            <div className="eyebrow" style={{ color: 'var(--accent-ink)', marginBottom: 6 }}>headline · update weekly</div>
            <div style={{ fontSize: 26, fontWeight: 600, letterSpacing: '-0.02em', lineHeight: 1.15 }}>
              <span style={{ color: 'var(--accent-ink)' }}>0</span> Mandarin grief counsellors found across <b>14 nurse searches</b> in 30 days
            </div>
            <p className="muted-text" style={{ marginTop: 6, fontSize: 12 }}>Confidence <b>88%</b> · sample 14 distinct verified RNs · 8× higher than Ontario average.</p>
          </AIZone>

          {/* Time series */}
          <div className="card">
            <div className="between" style={{ marginBottom: 6 }}>
              <div className="title-3">When did this start trending?</div>
              <div className="row"><span className="chip">searches</span><span className="chip">forum reports</span><span className="chip accent">both</span></div>
            </div>
            <div className="stripe" style={{ height: 140 }}>line chart · 12wk · stacked: searches + forum mentions</div>
            <div className="meta" style={{ marginTop: 6 }}>annotation: spike on Apr 12 coincides with closure of Mt. Sinai Mandarin counselling line.</div>
          </div>

          {/* Evidence — searches behind the signal */}
          <div className="card" style={{ padding: 0 }}>
            <div className="between" style={{ padding: '10px 12px', borderBottom: '1px solid var(--stroke)' }}>
              <div className="title-3">Searches with no good match · 14 cases</div>
              <span className="meta">anonymized · case tags only · no patient names</span>
            </div>
            <table className="tbl">
              <thead><tr><th>Date</th><th>Case tags</th><th>Nurse</th><th>What AI showed</th><th>Why rejected</th></tr></thead>
              <tbody>
                {[
                  ['Apr 28', 'Mandarin · 78yo · widowed · grief · OHIP · Scarborough', 'Jamie M.', '3 EN counsellors', '"language mismatch"'],
                  ['Apr 26', 'Mandarin · 71yo · spousal loss · low income · Etobicoke', 'Priya R.', '2 EN, 1 Cantonese', '"not Mandarin"'],
                  ['Apr 24', 'Mandarin/Cantonese · 82yo · grief support group', 'Anwar T.', '0 results', '"no match"'],
                  ['Apr 22', 'Mandarin · 65yo · recent widow · Markham', 'Sarah K.', '1 private $$$', '"out of budget"'],
                  ['Apr 20', 'Mandarin · 70yo · adult-child loss · Mid-West Toronto', 'Mei L.', '0 results', '"no match"'],
                ].map((r, i) => (
                  <tr key={i}>{r.map((c, j) => <td key={j} className={j === 0 ? 'num' : ''}>{c}</td>)}</tr>
                ))}
                <tr><td colSpan={5} className="meta" style={{ textAlign: 'center', padding: 8 }}>+ 9 more · expand all</td></tr>
              </tbody>
            </table>
          </div>

          {/* Forum posts contributing */}
          <div className="card">
            <div className="title-3" style={{ marginBottom: 8 }}>Forum posts that fed this finding · 6</div>
            <div className="col" style={{ gap: 8, fontSize: 12 }}>
              <div><span className="badge">question</span> <b>Jamie M.</b> · 2d — "Mandarin-speaking grief counsellor anywhere in GTA?" <span className="meta">9 replies · all "no luck"</span></div>
              <div><span className="badge crit">closure</span> <b>Mei L.</b> · 12d — "Mt. Sinai's Mandarin counselling line discontinued" <span className="meta">14 replies</span></div>
              <div><span className="badge">question</span> <b>Sarah K.</b> · 2w — "Anyone with Cantonese grief group? Mandarin would also work." <span className="meta">3 replies</span></div>
              <div className="meta">+ 3 more →</div>
            </div>
          </div>
        </div>

        <div className="col">
          {/* Compare panel */}
          <div className="card">
            <div className="title-3" style={{ marginBottom: 10 }}>Compare</div>
            <div className="col" style={{ gap: 12 }}>
              <div>
                <div className="meta" style={{ marginBottom: 4 }}>This OHT vs Ontario avg</div>
                <div className="row" style={{ alignItems: 'baseline', gap: 6 }}>
                  <span style={{ fontSize: 22, fontWeight: 600 }}>92</span>
                  <span className="meta">unmet here · vs</span>
                  <span style={{ fontSize: 14 }}>11</span>
                  <span className="meta">avg</span>
                </div>
                <div className="score-bar" style={{ marginTop: 6 }}><i style={{ width: '88%', background: 'var(--crit)' }} /></div>
              </div>
              <div>
                <div className="meta" style={{ marginBottom: 4 }}>vs last month (here)</div>
                <div className="row" style={{ alignItems: 'baseline', gap: 6 }}>
                  <span style={{ fontSize: 22, fontWeight: 600 }}>+34</span>
                  <span className="meta">growth</span>
                </div>
                <div className="stripe" style={{ height: 32, marginTop: 4 }}>spark · last 6 mo</div>
              </div>
              <div>
                <div className="meta" style={{ marginBottom: 4 }}>Similar OHTs (peer 6)</div>
                <div className="meta">2 of 6 also flagged this · 4 have at least 1 provider</div>
              </div>
            </div>
          </div>

          {/* Demographics */}
          <div className="card muted">
            <div className="title-3" style={{ marginBottom: 8 }}>Who's affected</div>
            <div className="col" style={{ gap: 6, fontSize: 11.5 }}>
              {[
                ['Age 65+', 92],
                ['Mandarin first language', 100],
                ['OHIP-only', 86],
                ['Lower-income postal codes', 71],
                ['Newcomer (<5yr Canada)', 43],
                ['Recent widow/widower', 64],
              ].map(([k, p]) => (
                <div key={k}>
                  <div className="between"><span>{k}</span><span className="meta">{p}%</span></div>
                  <div className="score-bar"><i style={{ width: p + '%' }} /></div>
                </div>
              ))}
            </div>
          </div>

          {/* Supply panel — honest about unknown */}
          <div className="card">
            <div className="title-3" style={{ marginBottom: 8 }}>Supply we know about</div>
            <div className="col" style={{ gap: 8, fontSize: 12 }}>
              <div><span className="badge">directory</span> 0 Mandarin grief counsellors registered in this OHT</div>
              <div><span className="badge">forum</span> 1 closure reported (Mt. Sinai line · Apr 12)</div>
              <div><span className="badge warn">unknown</span> Private therapists not always listed publicly</div>
              <div><span className="badge">StatsCan</span> Mandarin first-language pop. in OHT: <b>~38,000</b></div>
              <div className="meta" style={{ paddingTop: 4 }}>ratio: 0 supply : ~38,000 potential need · honest unknown for private supply</div>
            </div>
          </div>

          {/* Actions panel */}
          <div className="card accent">
            <div className="title-3" style={{ marginBottom: 8 }}>Take action</div>
            <div className="col" style={{ gap: 6 }}>
              <button className="btn">⟱ Advocacy brief PDF (1-pg)</button>
              <button className="btn">⎙ Slide for OHT meeting</button>
              <button className="btn">💬 Open forum post · ask peers</button>
              <button className="btn">🔔 Subscribe to changes</button>
              <button className="btn">⚑ Mark as data quality issue</button>
              <button className="btn">✓ Tag as resolved</button>
            </div>
          </div>

          {/* Admin notes */}
          <div className="card">
            <div className="title-3" style={{ marginBottom: 6 }}>Admin notes <span className="meta">· visible to OHT admins only</span></div>
            <div className="col" style={{ gap: 8, fontSize: 12 }}>
              <div>
                <div className="meta">Erin C. · OHT planner · 2d</div>
                <div>"Bringing to May 14 planning. Pre-meeting: contact Yee Hong about expanding."</div>
              </div>
              <textarea className="field" placeholder="Add a note for other admins…" style={{ minHeight: 50, fontSize: 12 }} />
            </div>
          </div>
        </div>
      </div>
    </AppFrame>
  );
}

window.ShortageAdminV2 = ShortageAdminV2;
window.ShortageDrilldown = ShortageDrilldown;
