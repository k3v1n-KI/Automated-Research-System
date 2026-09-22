// Pathways mid-fi — Resource detail
// Hero, tabs (about / eligibility / contact / evidence / peer notes),
// right rail with quick facts, last-verified, mini-map.

function ScreenResource({ id }) {
  const r = useRouter();
  const [tab, setTab] = useState('about');
  // For demo, hard-code to Yee Hong (r1) regardless of id
  const res = RES_DETAIL;

  return (
    <div className="page-enter">
      <Topnav active="cases" />

      {/* Sub-bar */}
      <div style={{
        borderBottom: '1px solid var(--stroke)',
        background: 'var(--paper)',
        padding: '12px 32px',
        position: 'sticky', top: 'var(--topnav-h)', zIndex: 9,
      }}>
        <div style={{ maxWidth: 1280, margin: '0 auto' }} className="between">
          <div className="row" style={{ gap: 10 }}>
            <button className="btn ghost sm" onClick={() => r.go({ name: 'results' })}>
              <Ico name="arrow-left" size={13} /> Back to results
            </button>
            <span style={{ color: 'var(--ink-4)' }}>/</span>
            <span className="muted" style={{ fontSize: 12.5 }}>Resource {res.code}</span>
          </div>
          <div className="row" style={{ gap: 8 }}>
            <button className="btn ghost sm"><Ico name="bookmark" size={13} /> Save</button>
            <button className="btn ghost sm"><Ico name="flag" size={13} /> Flag issue</button>
            <button className="btn sm"><Ico name="share" size={13} /> Share</button>
          </div>
        </div>
      </div>

      <main style={{ maxWidth: 1280, margin: '0 auto', padding: '32px 32px 80px' }}>

        {/* Hero */}
        <section style={{ display: 'grid', gridTemplateColumns: '1fr auto', gap: 32, marginBottom: 28 }}>
          <div className="row" style={{ gap: 22, alignItems: 'flex-start' }}>
            <RAvatar name={res.name} size="lg" />
            <div className="stack-8" style={{ flex: 1 }}>
              <div className="row" style={{ gap: 8, flexWrap: 'wrap' }}>
                <span className="badge dark"><Ico name="check" size={10} stroke={2.5} /> verified</span>
                <span className="badge good dot">accepting referrals</span>
                <span className="badge acc">strong fit for this case</span>
              </div>
              <h1 className="title-display" style={{ fontSize: 40 }}>{res.name}</h1>
              <p className="muted" style={{ fontSize: 14.5, maxWidth: 640 }}>
                {res.tagline}
              </p>
              <div className="row" style={{ gap: 14, fontSize: 12.5, color: 'var(--ink-2)' }}>
                <span className="row" style={{ gap: 5 }}><Ico name="building" size={13} /> {res.type}</span>
                <span>·</span>
                <span className="row" style={{ gap: 5 }}><Ico name="pin" size={13} /> {res.neighborhood} · {res.distance} km</span>
                <span>·</span>
                <span className="row" style={{ gap: 5 }}><Ico name="globe" size={13} /> {res.languages.join(' · ')}</span>
              </div>
            </div>
          </div>

          {/* CTA cluster */}
          <div className="card" style={{ padding: 18, minWidth: 260, alignSelf: 'flex-start' }}>
            <div className="stack-12">
              <div>
                <span className="eyebrow">Refer this patient</span>
                <p className="muted" style={{ fontSize: 12, marginTop: 4 }}>Generates a referral packet from your case sketch.</p>
              </div>
              <button className="btn accent block lg">
                Start referral <Ico name="arrow-right" size={14} />
              </button>
              <button className="btn block">
                <Ico name="phone" size={13} /> Call intake · {res.phone}
              </button>
              <div className="row dim" style={{ fontSize: 11, gap: 6, justifyContent: 'center' }}>
                <Ico name="clock" size={11} />
                Intake: Mon–Fri 9–4
              </div>
            </div>
          </div>
        </section>

        {/* Tabs + content */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 320px', gap: 32 }}>
          <div>
            <nav className="tabs">
              {TABS.map(t => (
                <a key={t.id} className={tab === t.id ? 'active' : ''} onClick={() => setTab(t.id)}>{t.label}</a>
              ))}
            </nav>

            <article style={{ padding: '24px 0' }}>
              {tab === 'about' && <AboutTab res={res} />}
              {tab === 'eligibility' && <EligibilityTab res={res} />}
              {tab === 'contact' && <ContactTab res={res} />}
              {tab === 'evidence' && <EvidenceTab res={res} />}
              {tab === 'peer' && <PeerTab res={res} />}
            </article>
          </div>

          {/* Right rail */}
          <aside className="stack-16">
            <div className="card">
              <div className="eyebrow" style={{ marginBottom: 10 }}>Quick facts</div>
              <dl style={{ margin: 0 }}>
                <div className="dl-row"><dt>Wait time</dt><dd>{res.wait}</dd></div>
                <div className="dl-row"><dt>Service area</dt><dd>{res.serviceArea}</dd></div>
                <div className="dl-row"><dt>Cost</dt><dd>{res.cost}</dd></div>
                <div className="dl-row"><dt>Coverage</dt><dd>{res.coverage}</dd></div>
                <div className="dl-row"><dt>Funded by</dt><dd>{res.funder}</dd></div>
                <div className="dl-row"><dt>Region</dt><dd>{res.region}</dd></div>
              </dl>
            </div>

            <div className="card">
              <div className="between" style={{ marginBottom: 10 }}>
                <span className="eyebrow">Verification</span>
                <span className="badge good dot">current</span>
              </div>
              <div className="stack-8" style={{ fontSize: 12.5 }}>
                <div>
                  <div className="muted">Last verified</div>
                  <div><strong>{res.verifiedAt}</strong> by Pathways admin</div>
                </div>
                <div>
                  <div className="muted">Confirmed by peers</div>
                  <div className="row" style={{ gap: 6, marginTop: 4 }}>
                    <div className="row" style={{ marginLeft: 0 }}>
                      {[0,1,2].map(i => (
                        <span key={i} className="avatar" style={{ width: 22, height: 22, fontSize: 9, marginLeft: i ? -6 : 0, background: 'var(--paper-3)' }}>
                          {['MP','DS','LW'][i]}
                        </span>
                      ))}
                    </div>
                    <span style={{ fontSize: 12 }}><strong>12</strong> coordinators</span>
                  </div>
                </div>
                <a className="src">view verification log</a>
              </div>
            </div>

            <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
              <div className="map-canvas" style={{ height: 160, position: 'relative', borderRadius: 0, border: 0 }}>
                <svg viewBox="0 0 300 160" style={{ position: 'absolute', inset: 0, width: '100%', height: '100%' }}>
                  <line x1="0" y1="80" x2="300" y2="80" stroke="var(--stroke-2)" strokeWidth="1.5" opacity="0.6"/>
                  <line x1="150" y1="0" x2="150" y2="160" stroke="var(--stroke-2)" strokeWidth="1.5" opacity="0.6"/>
                  <circle cx="195" cy="55" r="6" fill="var(--accent)"/>
                  <circle cx="195" cy="55" r="14" fill="var(--accent)" opacity="0.18"/>
                  <text x="208" y="59" fontSize="9.5" fill="var(--ink)" fontFamily="IBM Plex Sans" fontWeight="500">Yee Hong</text>
                  <circle cx="120" cy="100" r="4" fill="var(--ink)"/>
                  <text x="80" y="118" fontSize="9" fill="var(--ink-3)" fontFamily="JetBrains Mono">case · 6.4 km</text>
                </svg>
              </div>
              <div className="between" style={{ padding: '10px 12px' }}>
                <span className="muted" style={{ fontSize: 12 }}>{res.address}</span>
                <a className="src">directions</a>
              </div>
            </div>
          </aside>
        </div>
      </main>
    </div>
  );
}

// ── Tabs ───────────────────────────────────────────────
const TABS = [
  { id: 'about', label: 'About' },
  { id: 'eligibility', label: 'Eligibility' },
  { id: 'contact', label: 'Contact & intake' },
  { id: 'evidence', label: 'Evidence & sources' },
  { id: 'peer', label: 'Peer notes' },
];

function AboutTab({ res }) {
  return (
    <div className="stack-24">
      <div className="ai-zone" style={{ padding: 16 }}>
        <AITag>Plain-language summary</AITag>
        <p style={{ marginTop: 10, fontSize: 14, lineHeight: 1.65, color: 'var(--ink)' }}>
          {res.aiSummary}
        </p>
        <div className="row" style={{ marginTop: 12, gap: 6 }}>
          <button className="btn ghost sm">Translate to French</button>
          <button className="btn ghost sm">Plainer language</button>
          <span className="dim mono" style={{ fontSize: 10.5, marginLeft: 'auto' }}>last regenerated 2d ago</span>
        </div>
      </div>

      <div>
        <h3 className="title-2" style={{ marginBottom: 8 }}>Services offered</h3>
        <ul style={{ margin: 0, padding: 0, listStyle: 'none', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
          {res.services.map(s => (
            <li key={s} className="row" style={{ gap: 8, fontSize: 13.5 }}>
              <Ico name="check-circle" size={15} style={{ color: 'var(--accent)' }} />
              <span>{s}</span>
            </li>
          ))}
        </ul>
      </div>

      <div>
        <h3 className="title-2" style={{ marginBottom: 8 }}>Catchment & access</h3>
        <p style={{ fontSize: 13.5, lineHeight: 1.65, color: 'var(--ink-2)' }}>
          {res.catchment}
        </p>
      </div>

      <div>
        <h3 className="title-2" style={{ marginBottom: 8 }}>Referral process</h3>
        <ol style={{ margin: 0, paddingLeft: 0, listStyle: 'none', counterReset: 'step' }}>
          {res.process.map((p, i) => (
            <li key={i} className="row" style={{ alignItems: 'flex-start', gap: 12, padding: '10px 0', borderBottom: i < res.process.length - 1 ? '1px solid var(--stroke)' : 0 }}>
              <span style={{
                width: 24, height: 24, borderRadius: '50%', background: 'var(--paper-3)',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: 11, fontWeight: 600, color: 'var(--ink-2)', flexShrink: 0,
                fontFamily: 'var(--font-mono)',
              }}>{i + 1}</span>
              <span style={{ fontSize: 13.5, lineHeight: 1.55 }}>{p}</span>
            </li>
          ))}
        </ol>
      </div>
    </div>
  );
}

function EligibilityTab({ res }) {
  return (
    <div className="stack-16">
      <div className="ai-zone" style={{ padding: 14 }}>
        <AITag>Match against your case</AITag>
        <div className="stack-8" style={{ marginTop: 10 }}>
          {res.eligMatch.map((m, i) => (
            <div key={i} className="row" style={{ gap: 10, fontSize: 13 }}>
              <Ico name={m.ok ? 'check-circle' : 'x'} size={16}
                style={{ color: m.ok ? 'var(--good)' : 'var(--crit)', flexShrink: 0 }} />
              <span style={{ flex: 1 }}>{m.text}</span>
              <span className="dim mono" style={{ fontSize: 10.5 }}>{m.src}</span>
            </div>
          ))}
        </div>
      </div>
      <div>
        <h3 className="title-2" style={{ marginBottom: 8 }}>Eligibility criteria</h3>
        <ul style={{ margin: 0, paddingLeft: 18, fontSize: 13.5, lineHeight: 1.7, color: 'var(--ink-2)' }}>
          {res.eligibility.map(e => <li key={e}>{e}</li>)}
        </ul>
      </div>
    </div>
  );
}

function ContactTab({ res }) {
  return (
    <div className="stack-16">
      <div className="card" style={{ padding: 16 }}>
        <div className="stack-12">
          <div className="row" style={{ gap: 12 }}>
            <Ico name="phone" size={16} style={{ color: 'var(--ink-2)' }} />
            <div className="stack-2" style={{ flex: 1 }}>
              <div style={{ fontWeight: 500 }}>{res.phone}</div>
              <div className="muted" style={{ fontSize: 12 }}>Intake line · Mon–Fri 9–4</div>
            </div>
            <button className="btn sm">Call</button>
          </div>
          <div className="divider" />
          <div className="row" style={{ gap: 12 }}>
            <Ico name="globe" size={16} style={{ color: 'var(--ink-2)' }} />
            <div className="stack-2" style={{ flex: 1 }}>
              <div style={{ fontWeight: 500 }}>{res.website}</div>
              <div className="muted" style={{ fontSize: 12 }}>Online intake form available in EN/中文</div>
            </div>
            <button className="btn sm">Open</button>
          </div>
        </div>
      </div>
      <div>
        <h3 className="title-2" style={{ marginBottom: 8 }}>Hours of operation</h3>
        <table style={{ width: '100%', fontSize: 13, borderCollapse: 'collapse' }}>
          <tbody>
            {res.hours.map(([d, h]) => (
              <tr key={d} style={{ borderBottom: '1px solid var(--stroke)' }}>
                <td style={{ padding: '9px 0', color: 'var(--ink-2)', width: 140 }}>{d}</td>
                <td style={{ padding: '9px 0' }}>{h}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function EvidenceTab({ res }) {
  return (
    <div className="stack-16">
      <p className="muted" style={{ fontSize: 13.5 }}>
        Every claim on this page is sourced. Click any source to see the original.
      </p>
      {res.sources.map((s, i) => (
        <div key={i} className="card" style={{ padding: 14 }}>
          <div className="between" style={{ marginBottom: 6 }}>
            <span style={{ fontWeight: 500, fontSize: 13.5 }}>{s.claim}</span>
            <span className="badge">{s.kind}</span>
          </div>
          <div className="row dim" style={{ fontSize: 11.5, gap: 10 }}>
            <a className="src">{s.url}</a>
            <span>verified {s.verifiedAt}</span>
          </div>
        </div>
      ))}
    </div>
  );
}

function PeerTab({ res }) {
  return (
    <div className="stack-12">
      <div className="row between" style={{ marginBottom: 4 }}>
        <p className="muted" style={{ fontSize: 13 }}>Notes from coordinators who've referred to this resource.</p>
        <button className="btn sm"><Ico name="plus" size={12} stroke={2}/> Add note</button>
      </div>
      {res.peerNotes.map(n => (
        <div key={n.id} className="card" style={{ padding: 14 }}>
          <div className="between" style={{ marginBottom: 6 }}>
            <div className="row" style={{ gap: 8 }}>
              <span className="avatar" style={{ width: 26, height: 26, fontSize: 10 }}>{n.initials}</span>
              <span style={{ fontSize: 13, fontWeight: 500 }}>{n.author}</span>
              <span className="badge dark"><Ico name="check" size={9} stroke={2.5}/>verified</span>
              <span className="dim mono" style={{ fontSize: 11 }}>{n.role}</span>
            </div>
            <span className="dim mono" style={{ fontSize: 11 }}>{n.when}</span>
          </div>
          <p style={{ fontSize: 13.5, lineHeight: 1.6, marginTop: 6 }}>{n.text}</p>
          <div className="row" style={{ marginTop: 10, gap: 14, fontSize: 11.5, color: 'var(--ink-3)' }}>
            <span className="row" style={{ gap: 4 }}><Ico name="star" size={12}/> {n.confirms} confirm</span>
            <span className="row" style={{ gap: 4 }}>↩ reply</span>
          </div>
        </div>
      ))}
    </div>
  );
}

// ── Sample data ───────────────────────────────────────
const RES_DETAIL = {
  code: 'r-1042',
  name: 'Yee Hong Centre — Markham',
  type: 'Senior community care',
  neighborhood: 'Scarborough',
  distance: 6.4,
  languages: ['Cantonese', 'Mandarin', 'English'],
  tagline: 'Culturally-tailored home & community care for Asian seniors. Cantonese-speaking PSWs, congregate dining, and same-week intake.',
  aiSummary: 'Yee Hong runs a home support program that pairs seniors with Cantonese- or Mandarin-speaking PSWs for personal care, light housekeeping, and meal delivery. They handle ODSP billing and accept referrals directly from coordinators (no MD signature needed). Typical intake is 3–5 days. They do not provide skilled nursing — for daily weight checks you would need to pair this with a home-nursing service.',
  services: ['Cantonese-speaking PSWs', 'Meal delivery (low-sodium, cultural)', 'Adult day program', 'Caregiver respite', 'Transportation to appointments', 'Social & wellness groups'],
  catchment: 'Serves all of East Toronto OHT and parts of Scarborough North. Travel to client home included; clients can also attend the Markham day program (transit subsidies available).',
  process: [
    'Coordinator submits Pathways referral — Yee Hong intake replies within one business day.',
    'Phone screen with caregiver/client (Cantonese available) to confirm needs and ODSP coverage.',
    'In-home assessment scheduled within 3–5 days; PSW match made same week.',
    'Pathways auto-shares case outcome with you weekly until closed.',
  ],
  wait: 'Intake within 5 days',
  serviceArea: 'East TO + Scarb. N.',
  cost: 'Free with ODSP / OHIP+',
  coverage: 'ODSP · CCAC · self-pay sliding scale',
  funder: 'Ontario Health · United Way',
  region: 'East Toronto OHT',
  verifiedAt: '3 days ago',
  phone: '(416) 555-0144',
  website: 'yeehong.com/intake',
  address: '2311 McNicoll Ave · Scarborough',
  hours: [['Monday', '9:00 – 4:00'], ['Tuesday', '9:00 – 4:00'], ['Wednesday', '9:00 – 6:00'], ['Thursday', '9:00 – 4:00'], ['Friday', '9:00 – 4:00'], ['Saturday', 'Voicemail only'], ['Sunday', 'Closed']],
  eligMatch: [
    { ok: true, text: 'Senior (78) — within target population (60+)', src: 'criteria §1' },
    { ok: true, text: 'ODSP coverage accepted — no out-of-pocket cost', src: 'funder docs' },
    { ok: true, text: 'Cantonese first language — language match confirmed', src: 'staff roster' },
    { ok: true, text: 'East Toronto OHT — within catchment', src: 'service area' },
    { ok: false, text: 'Daily skilled nursing — not offered (pair with SE Health)', src: 'service list' },
  ],
  eligibility: ['Age 60 or older (some exceptions for caregivers)', 'Ontario resident with valid OHIP, ODSP, or willingness to self-pay', 'Lives within East Toronto OHT or Scarborough North', 'Independent in some ADLs or has caregiver support available'],
  sources: [
    { claim: 'Cantonese-speaking PSW availability', kind: 'staff roster', url: 'yeehong.com/team', verifiedAt: '3d ago' },
    { claim: 'ODSP billing accepted', kind: 'funder doc', url: 'ontario.ca/HCCSA-2023', verifiedAt: '14d ago' },
    { claim: 'Intake 3–5 days', kind: 'peer-confirmed', url: '4 confirmations', verifiedAt: 'last week' },
    { claim: 'No skilled nursing', kind: 'service list', url: 'yeehong.com/services', verifiedAt: '3d ago' },
  ],
  peerNotes: [
    { id: 1, author: 'Mira Patel', initials: 'MP', role: 'RN · East Toronto', when: '2 days ago', text: 'Referred a Cantonese-speaking 81-year-old post-stroke. Intake was actually 4 days, PSW was a great match. Heads up — they will not coordinate medication; pair with home nursing if that\'s in scope.', confirms: 5 },
    { id: 2, author: 'Daniel Singh', initials: 'DS', role: 'SW · Scarborough', when: '1 week ago', text: 'Used the day program for caregiver respite — Wednesday is the busiest day, Thursday quieter and easier to get a same-day spot.', confirms: 2 },
    { id: 3, author: 'Linda Wong', initials: 'LW', role: 'NP · East York', when: '3 weeks ago', text: 'For ODSP-only clients, billing was straightforward. For mixed coverage, ask intake to flag the file — they handle it but it adds 1–2 days to start.', confirms: 7 },
  ],
};

window.ScreenResource = ScreenResource;
