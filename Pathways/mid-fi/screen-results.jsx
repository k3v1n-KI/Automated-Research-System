// Pathways mid-fi — Results C · map + ranked list
// Left: filters · Center: ranked resource list · Right: schematic map + region context.

function ScreenResults() {
  const r = useRouter();
  const [view, setView] = useState('split'); // split | list | map
  const [hovered, setHovered] = useState(null);
  const [sort, setSort] = useState('fit');

  return (
    <div className="page-enter">
      <Topnav active="cases" />

      {/* Sub-bar with case summary */}
      <div style={{
        borderBottom: '1px solid var(--stroke)',
        background: 'var(--paper)',
        padding: '12px 32px',
        position: 'sticky', top: 'var(--topnav-h)', zIndex: 9,
      }}>
        <div style={{ maxWidth: 1480, margin: '0 auto' }} className="between">
          <div className="row" style={{ gap: 10, flexWrap: 'wrap' }}>
            <button className="btn ghost sm" onClick={() => r.go({ name: 'entry' })}>
              <Ico name="arrow-left" size={13} /> Edit case
            </button>
            <span style={{ color: 'var(--ink-4)' }}>/</span>
            <span style={{ fontSize: 13, fontWeight: 500 }}>Senior · post-discharge · home support</span>
            <span className="row" style={{ gap: 4 }}>
              <span className="chip sm">senior</span>
              <span className="chip sm">home support</span>
              <span className="chip sm">Cantonese</span>
              <span className="chip sm">ODSP</span>
              <span className="chip sm dim">+4 more</span>
            </span>
          </div>
          <div className="row" style={{ gap: 8 }}>
            <button className="btn ghost sm"><Ico name="bookmark" size={13} /> Save case</button>
            <button className="btn sm"><Ico name="share" size={13} /> Share</button>
          </div>
        </div>
      </div>

      <div style={{ display: 'flex', maxWidth: 1480, margin: '0 auto', minHeight: 'calc(100vh - var(--topnav-h) - 50px)' }}>

        {/* Filters sidebar */}
        <aside className="sidebar" style={{ width: 220, padding: '20px 14px' }}>
          <div className="between" style={{ marginBottom: 12 }}>
            <span className="title-3">Filters</span>
            <a className="dim" style={{ fontSize: 11.5, cursor: 'pointer' }}>reset</a>
          </div>

          <div className="group-label">Service type</div>
          {[
            ['Home & community care', 12],
            ['Mental health', 8],
            ['Meal program', 5],
            ['Senior day program', 4],
            ['Transportation', 3],
          ].map(([l, n]) => (
            <label key={l} className="opt">
              <input type="checkbox" defaultChecked={l === 'Home & community care' || l === 'Meal program'} />
              <span>{l}</span><span className="count">{n}</span>
            </label>
          ))}

          <div className="group-label">Languages</div>
          {[['English', 32], ['Cantonese', 6], ['Mandarin', 9], ['French', 4]].map(([l, n]) => (
            <label key={l} className="opt">
              <input type="checkbox" defaultChecked={l === 'Cantonese'} />
              <span>{l}</span><span className="count">{n}</span>
            </label>
          ))}

          <div className="group-label">Eligibility</div>
          {[['Accepts ODSP', 18], ['No OHIP required', 7], ['Sliding scale', 11]].map(([l, n]) => (
            <label key={l} className="opt">
              <input type="checkbox" defaultChecked={l === 'Accepts ODSP'} />
              <span>{l}</span><span className="count">{n}</span>
            </label>
          ))}

          <div className="group-label">Wait time</div>
          <div className="col" style={{ gap: 4 }}>
            {['Same day', '< 1 week', '< 1 month', 'Any'].map((l, i) => (
              <label key={l} className="opt">
                <input type="radio" name="wait" defaultChecked={i === 1} />
                <span>{l}</span>
              </label>
            ))}
          </div>

          <div className="group-label">Distance</div>
          <input type="range" min="1" max="40" defaultValue="20" style={{ width: '100%', accentColor: 'var(--accent)' }} />
          <div className="between mono dim" style={{ fontSize: 10.5, marginTop: 2 }}>
            <span>1 km</span><span>20 km</span><span>40+</span>
          </div>
        </aside>

        {/* Center — list */}
        <section style={{ flex: 1, padding: '20px 22px', minWidth: 0, display: view === 'map' ? 'none' : 'block' }}>
          <div className="between" style={{ marginBottom: 14 }}>
            <div className="row" style={{ gap: 8 }}>
              <h2 className="title-1" style={{ fontSize: 22 }}>{RESOURCES.length} resources</h2>
              <span className="muted" style={{ fontSize: 13 }}>matching your case in East Toronto OHT</span>
            </div>
            <div className="row" style={{ gap: 6 }}>
              <div className="row" style={{ gap: 0, border: '1px solid var(--stroke-2)', borderRadius: 'var(--r-sm)', padding: 2 }}>
                {[['split', 'list'], ['list', 'list'], ['map', 'map']].map(([v, ic]) => (
                  <button key={v} className={`btn ghost sm ${view === v ? '' : ''}`}
                    onClick={() => setView(v)}
                    style={{
                      padding: '4px 10px',
                      background: view === v ? 'var(--paper-3)' : 'transparent',
                      color: view === v ? 'var(--ink)' : 'var(--ink-3)',
                    }}>
                    <Ico name={ic} size={13} /> <span style={{ textTransform: 'capitalize' }}>{v}</span>
                  </button>
                ))}
              </div>
              <select className="field" value={sort} onChange={e => setSort(e.target.value)}
                style={{ width: 'auto', padding: '6px 10px', fontSize: 12 }}>
                <option value="fit">Sort: best fit</option>
                <option value="dist">Sort: distance</option>
                <option value="wait">Sort: wait time</option>
                <option value="verified">Sort: recently verified</option>
              </select>
            </div>
          </div>

          {/* AI ranking explainer */}
          <div className="ai-zone" style={{ padding: '11px 14px', marginBottom: 16 }}>
            <div className="row" style={{ gap: 10, alignItems: 'flex-start' }}>
              <Ico name="sparkle" size={14} stroke={1.8} style={{ color: 'var(--accent)', marginTop: 2 }} />
              <div className="stack-2" style={{ flex: 1 }}>
                <div style={{ fontSize: 12.5, lineHeight: 1.5 }}>
                  Ranked by <strong>service match</strong>, then <strong>language</strong>, then <strong>distance</strong>. Two top results offer Cantonese-speaking staff.
                </div>
                <div style={{ fontSize: 11, color: 'var(--accent-ink)', cursor: 'pointer', fontWeight: 500 }}>
                  Adjust ranking weights →
                </div>
              </div>
            </div>
          </div>

          <div className="stack-12">
            {RESOURCES.map(res => (
              <ResourceRow
                key={res.id}
                res={res}
                onHover={setHovered}
                hovered={hovered === res.id}
                onOpen={() => r.go({ name: 'resource', id: res.id })}
              />
            ))}
          </div>
        </section>

        {/* Right — schematic map */}
        {view !== 'list' && (
          <aside style={{
            width: view === 'map' ? '100%' : 460,
            flexShrink: 0,
            borderLeft: view === 'map' ? 0 : '1px solid var(--stroke)',
            background: 'var(--paper-2)',
            padding: 18,
            position: 'sticky', top: 'calc(var(--topnav-h) + 50px)',
            alignSelf: 'flex-start',
            height: 'calc(100vh - var(--topnav-h) - 50px)',
            display: 'flex', flexDirection: 'column', gap: 12,
          }}>
            <div className="between">
              <div className="stack-2">
                <span className="eyebrow">East Toronto OHT</span>
                <div style={{ fontSize: 13.5, fontWeight: 500 }}>{RESOURCES.length} pinned · 6 within 5 km</div>
              </div>
              <button className="btn ghost sm">Expand region</button>
            </div>
            <SchematicMap resources={RESOURCES} hovered={hovered} onHover={setHovered}
              onOpen={(id) => r.go({ name: 'resource', id })} />
            <div className="row mono dim" style={{ fontSize: 10.5, gap: 10 }}>
              <span>● fit ≥ 4</span>
              <span>○ fit 2–3</span>
              <span>· transit hub</span>
              <span style={{ marginLeft: 'auto' }}>schematic · not to scale</span>
            </div>
          </aside>
        )}
      </div>
    </div>
  );
}

function ResourceRow({ res, onHover, hovered, onOpen }) {
  return (
    <div
      onMouseEnter={() => onHover(res.id)}
      onMouseLeave={() => onHover(null)}
      onClick={onOpen}
      className="row-tap"
      style={{
        display: 'grid',
        gridTemplateColumns: 'auto 1fr auto',
        gap: 16,
        padding: 16,
        border: '1px solid',
        borderColor: hovered ? 'var(--stroke-3)' : 'var(--stroke)',
        borderRadius: 'var(--r-md)',
        background: 'var(--paper)',
        cursor: 'pointer',
        boxShadow: hovered ? 'var(--shadow-md)' : 'none',
        transition: 'border-color 0.12s, box-shadow 0.12s',
      }}
    >
      <RAvatar name={res.name} />
      <div className="stack-6" style={{ minWidth: 0 }}>
        <div className="row" style={{ gap: 8, flexWrap: 'wrap' }}>
          <span style={{ fontSize: 14.5, fontWeight: 600 }}>{res.name}</span>
          {res.verified && <span className="badge dark"><Ico name="check" size={10} stroke={2.5} /> verified</span>}
          {res.peer > 0 && <span className="badge"><Ico name="star" size={10} /> {res.peer}</span>}
        </div>
        <div className="muted" style={{ fontSize: 12.5 }}>{res.type} · {res.neighborhood} · {res.distance} km</div>
        <div className="row" style={{ gap: 4, flexWrap: 'wrap' }}>
          {res.tags.map(t => <span key={t} className="chip sm">{t}</span>)}
        </div>
        <div className="row" style={{ gap: 12, marginTop: 4, fontSize: 11.5 }}>
          <span className="row" style={{ gap: 4, color: 'var(--ink-2)' }}>
            <Ico name="clock" size={12} /> {res.wait}
          </span>
          <span className="row" style={{ gap: 4, color: 'var(--ink-2)' }}>
            <Ico name="globe" size={12} /> {res.languages.join(', ')}
          </span>
          <span className="row" style={{ gap: 4, color: 'var(--ink-3)' }}>
            <Ico name="verify" size={12} /> verified {res.verifiedAt}
          </span>
        </div>
      </div>
      <div className="stack-6" style={{ alignItems: 'flex-end', justifyContent: 'space-between' }}>
        <div className="stack-4" style={{ alignItems: 'flex-end' }}>
          <span className="eyebrow">Fit</span>
          <FitBars value={res.fit} />
          <span className="mono dim" style={{ fontSize: 10.5 }}>{res.fitNote}</span>
        </div>
        <Ico name="chevron-right" size={16} />
      </div>
    </div>
  );
}

// ── Schematic map (abstract Toronto-ish neighborhoods) ─────────
function SchematicMap({ resources, hovered, onHover, onOpen }) {
  return (
    <div className="map-canvas" style={{ flex: 1, position: 'relative', minHeight: 360 }}>
      {/* abstract neighborhood polygons */}
      <svg viewBox="0 0 600 460" preserveAspectRatio="none" style={{ position: 'absolute', inset: 0, width: '100%', height: '100%' }}>
        {/* lake / water bottom */}
        <path d="M0 400 L600 400 L600 460 L0 460 Z" fill="var(--paper-3)" opacity="0.6" />
        {/* main road grid suggestion */}
        <line x1="0" y1="180" x2="600" y2="180" stroke="var(--stroke-2)" strokeWidth="2" opacity="0.55"/>
        <line x1="0" y1="280" x2="600" y2="280" stroke="var(--stroke-2)" strokeWidth="2" opacity="0.55"/>
        <line x1="220" y1="0" x2="220" y2="400" stroke="var(--stroke-2)" strokeWidth="2" opacity="0.55"/>
        <line x1="380" y1="0" x2="380" y2="400" stroke="var(--stroke-2)" strokeWidth="2" opacity="0.55"/>
        {/* park blob */}
        <path d="M260 60 q40 -10 70 0 q15 30 0 60 q-40 20 -70 -10 z" fill="color-mix(in oklch, var(--good) 18%, var(--paper))" opacity="0.6"/>
        <text x="295" y="100" fontSize="9" fill="var(--ink-3)" fontFamily="JetBrains Mono">park</text>
        {/* region label */}
        <text x="20" y="30" fontSize="10" fill="var(--ink-3)" fontFamily="JetBrains Mono" letterSpacing="1">EAST TORONTO OHT</text>
        <text x="20" y="395" fontSize="9" fill="var(--ink-3)" fontFamily="JetBrains Mono">lake ontario</text>
        <text x="80" y="170" fontSize="10" fill="var(--ink-2)" fontFamily="JetBrains Mono">RIVERDALE</text>
        <text x="280" y="170" fontSize="10" fill="var(--ink-2)" fontFamily="JetBrains Mono">EAST YORK</text>
        <text x="450" y="170" fontSize="10" fill="var(--ink-2)" fontFamily="JetBrains Mono">SCARBOROUGH W.</text>
        <text x="80" y="300" fontSize="10" fill="var(--ink-2)" fontFamily="JetBrains Mono">LESLIEVILLE</text>
        <text x="320" y="370" fontSize="10" fill="var(--ink-2)" fontFamily="JetBrains Mono">THE BEACH</text>
        {/* user pin (the case) */}
        <g>
          <circle cx="300" cy="240" r="10" fill="var(--ink)" opacity="0.12"/>
          <circle cx="300" cy="240" r="5" fill="var(--ink)"/>
          <text x="312" y="244" fontSize="10" fill="var(--ink)" fontWeight="600" fontFamily="IBM Plex Sans">case · M4K</text>
        </g>
      </svg>

      {/* resource pins */}
      {resources.map((res) => {
        const isHovered = hovered === res.id;
        const isStrong = res.fit >= 4;
        return (
          <button
            key={res.id}
            onMouseEnter={() => onHover(res.id)}
            onMouseLeave={() => onHover(null)}
            onClick={() => onOpen(res.id)}
            style={{
              position: 'absolute',
              left: `${res.x}%`, top: `${res.y}%`,
              transform: 'translate(-50%, -100%)',
              background: 'transparent', border: 0, padding: 0,
              cursor: 'pointer',
              zIndex: isHovered ? 5 : 1,
            }}
          >
            <div style={{
              display: 'flex', alignItems: 'center', gap: 4,
              padding: '4px 8px 4px 4px',
              background: isHovered ? 'var(--ink)' : 'var(--paper)',
              color: isHovered ? 'var(--paper)' : 'var(--ink)',
              border: `1px solid ${isHovered ? 'var(--ink)' : 'var(--stroke-2)'}`,
              borderRadius: 999,
              boxShadow: isHovered ? 'var(--shadow-md)' : 'var(--shadow-sm)',
              fontSize: 11.5, fontWeight: 500,
              transition: 'all 0.12s',
            }}>
              <span style={{
                width: 14, height: 14, borderRadius: '50%',
                background: isStrong ? 'var(--accent)' : 'transparent',
                border: isStrong ? 0 : `2px solid var(--accent)`,
                display: 'inline-block',
              }} />
              <span style={{ whiteSpace: 'nowrap' }}>{res.shortName}</span>
            </div>
            <div style={{
              width: 0, height: 0, margin: '0 auto',
              borderLeft: '5px solid transparent', borderRight: '5px solid transparent',
              borderTop: `6px solid ${isHovered ? 'var(--ink)' : 'var(--stroke-2)'}`,
            }} />
          </button>
        );
      })}
    </div>
  );
}

const RESOURCES = [
  { id: 'r1', name: 'Yee Hong Centre — Markham', shortName: 'Yee Hong', type: 'Senior community care', neighborhood: 'Scarborough', distance: 6.4, tags: ['home-care', 'meals', 'cantonese-staff'], languages: ['Cantonese', 'Mandarin', 'English'], wait: 'intake within 5 days', verified: true, verifiedAt: '3d ago', peer: 12, fit: 5, fitNote: '5/5 services match', x: 70, y: 38 },
  { id: 'r2', name: 'East Toronto Health Partners — Home First', shortName: 'ETHP Home First', type: 'Transitional home support', neighborhood: 'East York', distance: 2.1, tags: ['home-care', 'post-discharge', 'medication'], languages: ['English', 'Cantonese (on request)'], wait: 'same week', verified: true, verifiedAt: '1d ago', peer: 8, fit: 5, fitNote: 'language tentative', x: 50, y: 52 },
  { id: 'r3', name: 'Meals on Wheels — Toronto East', shortName: 'MoW East', type: 'Meal delivery', neighborhood: 'Riverdale', distance: 3.7, tags: ['meals', 'low-sodium', 'culturally-tailored'], languages: ['English'], wait: '2–3 days', verified: true, verifiedAt: '2w ago', peer: 5, fit: 4, fitNote: '4/5 — diet match', x: 30, y: 58 },
  { id: 'r4', name: 'Hong Fook Mental Health — Senior Wellness', shortName: 'Hong Fook Snr', type: 'Senior peer & wellness', neighborhood: 'East York', distance: 4.0, tags: ['social', 'cantonese-staff', 'walk-in'], languages: ['Cantonese', 'Mandarin', 'Korean'], wait: '1–2 weeks', verified: true, verifiedAt: '6d ago', peer: 4, fit: 4, fitNote: 'strong language fit', x: 56, y: 36 },
  { id: 'r5', name: 'SE Health — Community Nursing East', shortName: 'SE Health', type: 'Home nursing', neighborhood: 'Beaches', distance: 5.2, tags: ['rn-visits', 'wound-care', 'monitoring'], languages: ['English', 'French'], wait: '7–10 days', verified: true, verifiedAt: '4d ago', peer: 2, fit: 3, fitNote: 'no Cantonese', x: 60, y: 78 },
  { id: 'r6', name: "VHA Home HealthCare", shortName: 'VHA', type: 'Personal support workers', neighborhood: 'Leslieville', distance: 4.5, tags: ['psw', 'odsp-billed'], languages: ['English', 'Tagalog'], wait: '~2 weeks', verified: true, verifiedAt: '5d ago', peer: 3, fit: 3, fitNote: 'no Cantonese · ODSP ✓', x: 38, y: 64 },
];

window.ScreenResults = ScreenResults;
