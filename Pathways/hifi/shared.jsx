// Pathways Hi-Fi — shared atoms, icons, topnav, and Algoma OHT data
// Note: all phone numbers, hours, and verification logs are illustrative.

const { useState, useEffect, useRef, useMemo } = React;

// ── Icons ──────────────────────────────────────────────────
function Ico({ name, size = 14, stroke = 1.6, ...rest }) {
  const common = { width: size, height: size, viewBox: '0 0 24 24', fill: 'none',
    stroke: 'currentColor', strokeWidth: stroke, strokeLinecap: 'round', strokeLinejoin: 'round', ...rest };
  switch (name) {
    case 'arrow-right': return (<svg {...common}><path d="M5 12h14M13 6l6 6-6 6"/></svg>);
    case 'arrow-up-right': return (<svg {...common}><path d="M7 17 17 7M9 7h8v8"/></svg>);
    case 'arrow-left':  return (<svg {...common}><path d="M19 12H5M11 18l-6-6 6-6"/></svg>);
    case 'plus':        return (<svg {...common}><path d="M12 5v14M5 12h14"/></svg>);
    case 'search':      return (<svg {...common}><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></svg>);
    case 'sparkle':     return (<svg {...common}><path d="M12 3l1.8 5L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.7L12 3z"/></svg>);
    case 'cmd':         return (<svg {...common}><path d="M9 6a3 3 0 1 0 0 6h6a3 3 0 1 0 0-6v6m0 0a3 3 0 1 0 0 6V12m-6 0a3 3 0 1 0 0 6V12"/></svg>);
    case 'clock':       return (<svg {...common}><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></svg>);
    case 'check':       return (<svg {...common}><path d="m5 12 5 5L20 7"/></svg>);
    case 'chevron-right': return (<svg {...common}><path d="m9 6 6 6-6 6"/></svg>);
    case 'chevron-down':  return (<svg {...common}><path d="m6 9 6 6 6-6"/></svg>);
    case 'chevron-left':  return (<svg {...common}><path d="m15 6-6 6 6 6"/></svg>);
    case 'pin':         return (<svg {...common}><path d="M12 21s7-7.5 7-12a7 7 0 1 0-14 0c0 4.5 7 12 7 12z"/><circle cx="12" cy="9" r="2.5"/></svg>);
    case 'globe':       return (<svg {...common}><circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3a14 14 0 0 1 0 18M12 3a14 14 0 0 0 0 18"/></svg>);
    case 'bell':        return (<svg {...common}><path d="M6 16V11a6 6 0 0 1 12 0v5l2 3H4l2-3z"/><path d="M10 21h4"/></svg>);
    case 'star':        return (<svg {...common}><path d="m12 3 2.7 5.6 6.3.9-4.5 4.4 1 6.1L12 17.3 6.5 20l1-6.1L3 9.5l6.3-.9L12 3z"/></svg>);
    case 'book':        return (<svg {...common}><path d="M4 5a2 2 0 0 1 2-2h12v18H6a2 2 0 0 1-2-2V5z"/><path d="M4 5v14M8 7h6M8 11h6"/></svg>);
    case 'people':      return (<svg {...common}><circle cx="9" cy="8" r="3.5"/><path d="M3 20a6 6 0 0 1 12 0M16 11a3 3 0 1 0 0-6M21 20a5 5 0 0 0-5-5"/></svg>);
    case 'flag':        return (<svg {...common}><path d="M5 21V4M5 4h11l-2 4 2 4H5"/></svg>);
    case 'corner-arrow': return (<svg {...common}><path d="M5 9l4-4 4 4M9 5v10a4 4 0 0 0 4 4h6"/></svg>);
    case 'phone':       return (<svg {...common}><path d="M5 4h3l2 5-2 1a11 11 0 0 0 6 6l1-2 5 2v3a2 2 0 0 1-2 2A16 16 0 0 1 3 6a2 2 0 0 1 2-2z"/></svg>);
    case 'x':           return (<svg {...common}><path d="M6 6l12 12M18 6 6 18"/></svg>);
    case 'edit':        return (<svg {...common}><path d="M4 20h4l10-10-4-4L4 16v4z"/><path d="m14 6 4 4"/></svg>);
    case 'send':        return (<svg {...common}><path d="M4 12 21 4l-3 17-5-7-9-2z"/></svg>);
    case 'mic':         return (<svg {...common}><rect x="9" y="3" width="6" height="11" rx="3"/><path d="M5 11a7 7 0 0 0 14 0M12 18v3"/></svg>);
    case 'mic-off':     return (<svg {...common}><path d="M3 3l18 18"/><path d="M9 9v2a3 3 0 0 0 5 2"/><path d="M15 11V6a3 3 0 0 0-5.5-1.7"/><path d="M5 11a7 7 0 0 0 11 5M19 11a7 7 0 0 1-.3 2"/><path d="M12 18v3"/></svg>);
    case 'tag':         return (<svg {...common}><path d="M3 3h8l10 10-8 8L3 11V3z"/><circle cx="7.5" cy="7.5" r="1.2" fill="currentColor"/></svg>);
    default: return null;
  }
}

// ── Top nav ───────────────────────────────────────────────
function Topnav({ active = 'cases' }) {
  const items = [
    { id: 'cases',     label: 'Cases',     href: '../Pathways Hi-Fi.html' },
    { id: 'directory', label: 'Directory', href: 'resource.html' },
    { id: 'asks',      label: 'Asks',      href: 'asks.html' },
    { id: 'forum',     label: 'Forum',     href: '#' },
  ];
  return (
    <header className="topnav">
      <a href="../Pathways Hi-Fi.html" className="brand">
        <span className="brand-mark" />
        Pathways
      </a>
      <nav className="nav-links">
        {items.map(it => (
          <a key={it.id} href={it.href} className={active === it.id ? 'active' : ''}>{it.label}</a>
        ))}
      </nav>
      <span className="nav-spacer" />
      <span className="region-pill" title="Algoma Ontario Health Team">
        <span className="dot" />
        <span style={{ color: 'var(--ink)', fontWeight: 500 }}>Algoma OHT</span>
        <Ico name="chevron-down" size={11} />
      </span>
      <button className="btn ghost" style={{ padding: '6px 8px' }} aria-label="Notifications">
        <Ico name="bell" size={14}/>
      </button>
      <span className="avatar" title="Rita LaFlamme">RL</span>
    </header>
  );
}

// Topnav for top-level pages (different relative paths)
function TopnavRoot({ active = 'cases' }) {
  const items = [
    { id: 'cases',     label: 'Cases',     href: 'Pathways Hi-Fi.html' },
    { id: 'directory', label: 'Directory', href: 'hifi/resource.html' },
    { id: 'asks',      label: 'Asks',      href: 'hifi/asks.html' },
    { id: 'forum',     label: 'Forum',     href: '#' },
  ];
  return (
    <header className="topnav">
      <a href="Pathways Hi-Fi.html" className="brand">
        <span className="brand-mark" />
        Pathways
      </a>
      <nav className="nav-links">
        {items.map(it => (
          <a key={it.id} href={it.href} className={active === it.id ? 'active' : ''}>{it.label}</a>
        ))}
      </nav>
      <span className="nav-spacer" />
      <span className="region-pill" title="Algoma Ontario Health Team">
        <span className="dot" />
        <span style={{ color: 'var(--ink)', fontWeight: 500 }}>Algoma OHT</span>
        <Ico name="chevron-down" size={11} />
      </span>
      <button className="btn ghost" style={{ padding: '6px 8px' }} aria-label="Notifications">
        <Ico name="bell" size={14}/>
      </button>
      <span className="avatar" title="Rita LaFlamme">RL</span>
    </header>
  );
}

// ── Dictation mic — overlay for inputs/textareas ─────────
// Renders a small mic button positioned absolutely; clicking it streams
// `sample` text (or one of `samples`) into the bound value. The wrapper
// must be position:relative.
function MicMount({ value, setValue, sample, samples, placement = 'br', dark = false }) {
  const [rec, setRec] = useState(false);
  const baseRef = useRef('');
  const sampleRef = useRef(sample || (samples && samples[0]) || '');

  const start = () => {
    if (samples && samples.length) {
      // pick the first sample not already substantially present
      const v = (value || '').toLowerCase();
      sampleRef.current = samples.find(s => !v.includes(s.toLowerCase().slice(0, 16))) || samples[0];
    }
    baseRef.current = value || '';
    setRec(true);
  };
  const stop = () => setRec(false);

  useEffect(() => {
    if (!rec) return;
    const target = (baseRef.current ? baseRef.current.replace(/\s+$/, '') + ' ' : '') + sampleRef.current;
    let i = baseRef.current.length;
    const id = setInterval(() => {
      i += 2;
      if (i >= target.length) {
        setValue(target);
        setRec(false);
        clearInterval(id);
        return;
      }
      setValue(target.slice(0, i));
    }, 55);
    return () => clearInterval(id);
  }, [rec]);

  const pos = {
    br: { right: 8,  bottom: 8 },
    tr: { right: 8,  top: 8 },
    inline: null,
  }[placement];

  return (
    <button
      type="button"
      onClick={() => (rec ? stop() : start())}
      className={'mic-mount' + (rec ? ' on' : '') + (dark ? ' dark' : '')}
      style={pos ? { position: 'absolute', ...pos } : undefined}
      title={rec ? 'Stop dictation' : 'Dictate'}
      aria-label={rec ? 'Stop dictation' : 'Dictate'}
    >
      {rec ? (
        <>
          <span className="rec-dot" />
          <span className="wave"><span/><span/><span/><span/><span/></span>
        </>
      ) : (
        <Ico name="mic" size={12}/>
      )}
    </button>
  );
}

// ── Algoma cases data ─────────────────────────────────────
const CASES = [
  { id: 'c-241', title: 'Senior, post-discharge, lives alone',
    summary: 'Home support · meals · ODB · East-end Sault',
    tags: ['senior', 'home-care'], status: 'open', when: '2h ago', region: 'Sault Ste. Marie' },
  { id: 'c-240', title: 'Adolescent, self-harm risk, uninsured',
    summary: 'Same-day MH · no OHIP · transport from Garden River',
    tags: ['youth', 'mental-health'], status: 'referred', when: 'yesterday', region: 'Garden River FN' },
  { id: 'c-238', title: 'New parent · postpartum support',
    summary: 'Peer group + lactation · French preferred',
    tags: ['perinatal', 'francophone'], status: 'open', when: '2d ago', region: 'Sault Ste. Marie' },
  { id: 'c-235', title: 'Adult · housing-insecure · OAT',
    summary: 'Methadone continuity + shelter coordination',
    tags: ['housing', 'sud'], status: 'open', when: '3d ago', region: 'Blind River' },
  { id: 'c-232', title: 'Couple in Wawa · primary-care attachment',
    summary: 'Both unattached after retirement of FP · diabetes',
    tags: ['primary-care', 'rural'], status: 'closed', when: '5d ago', region: 'Wawa' },
];

// ── Match tier (signal-strength) ──────────────────────────
function matchTier(score) {
  if (score >= 90) return { tier: 5, label: 'Strong match' };
  if (score >= 75) return { tier: 4, label: 'Good match' };
  if (score >= 60) return { tier: 3, label: 'Fair match' };
  if (score >= 40) return { tier: 2, label: 'Partial match' };
  return { tier: 1, label: 'Weak match' };
}

function MatchMeter({ tier, dark = false, size = 'md' }) {
  const dims = size === 'sm'
    ? { gap: 1.5, w: 2.5, h: [4, 6, 8, 10, 12] }
    : { gap: 2,   w: 3.5, h: [5, 8, 11, 14, 17] };
  const onColor  = dark ? '#fafaf7' : '#29261b';
  const offColor = dark ? 'rgba(250,250,247,0.22)' : 'rgba(41,38,27,0.16)';
  return (
    <div style={{ display: 'flex', alignItems: 'flex-end', gap: dims.gap }}>
      {[1,2,3,4,5].map(i => (
        <span key={i} style={{
          width: dims.w, height: dims.h[i-1], borderRadius: 1,
          background: i <= tier ? onColor : offColor,
        }}/>
      ))}
    </div>
  );
}

// ── Verification badge ────────────────────────────────────
function VerifyBadge({ state, count, days, source, compact = false }) {
  const variants = {
    'verified-fresh': { label: 'Verified', tone: 'good', glyph: '✓' },
    'verified-aging': { label: 'Verified', tone: 'warn', glyph: '✓' },
    'ai-only':        { label: 'AI extracted', tone: 'ai', glyph: '⬡' },
    'flagged-stale':  { label: 'Flagged stale', tone: 'crit', glyph: '⚠' },
    'unknown':        { label: 'Unknown', tone: 'mute', glyph: '—' },
  };
  const v = variants[state] || variants.unknown;
  const toneStyles = {
    good: { color: '#15803d', bg: '#ecfdf5', border: '#bbf7d0' },
    warn: { color: '#a16207', bg: '#fefce8', border: '#fde68a' },
    ai:   { color: 'var(--accent-ink)', bg: 'var(--accent-soft)', border: 'color-mix(in srgb, var(--accent) 18%, transparent)' },
    crit: { color: '#b91c1c', bg: '#fef2f2', border: '#fecaca' },
    mute: { color: 'var(--ink-3)', bg: 'var(--paper-2)', border: 'var(--stroke)' },
  };
  const ts = toneStyles[v.tone];
  let suffix = '';
  if (state === 'verified-fresh' && days != null) suffix = ` · ${days}d ago`;
  else if (state === 'verified-aging' && days != null) suffix = ` · ${days}d old`;
  else if (state === 'flagged-stale' && count) suffix = count > 1 ? ` · ${count}×` : '';
  else if (state === 'ai-only' && source) suffix = compact ? '' : ` · ${source}`;
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: 5,
      padding: compact ? '1px 6px' : '2px 8px',
      borderRadius: 999,
      background: ts.bg, border: `1px solid ${ts.border}`,
      color: ts.color, fontSize: compact ? 10.5 : 11, fontWeight: 500,
      letterSpacing: '-0.005em', lineHeight: 1.3,
      fontFamily: 'var(--font-mono)',
    }}>
      <span style={{ fontFamily: 'var(--font-ui)' }}>{v.glyph}</span>
      <span style={{ fontFamily: 'var(--font-ui)' }}>{v.label}{suffix}</span>
    </span>
  );
}

function VerifyAggregate({ verified, total, flagged, compact }) {
  const pct = verified / total;
  const tone = flagged > 0 ? 'crit' : pct >= 0.7 ? 'good' : pct >= 0.3 ? 'warn' : 'ai';
  const toneColors = {
    good: { dot: 'var(--good)', text: '#15803d' },
    warn: { dot: 'var(--warn)', text: '#a16207' },
    crit: { dot: 'var(--crit)', text: '#b91c1c' },
    ai:   { dot: 'var(--accent)', text: 'var(--accent-ink)' },
  };
  const c = toneColors[tone];
  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6, fontSize: compact ? 11 : 12, color: 'var(--ink-3)' }}>
      <span style={{ width: 6, height: 6, borderRadius: '50%', background: c.dot }}/>
      <span style={{ color: c.text, fontWeight: 500, fontFamily: 'var(--font-mono)' }}>{verified}/{total}</span>
      <span>fields verified</span>
      {flagged > 0 && (
        <>
          <span style={{ color: 'var(--ink-4)' }}>·</span>
          <span style={{ color: '#b91c1c', fontWeight: 500 }}>{flagged} flagged</span>
        </>
      )}
    </span>
  );
}

// ── Algoma-flavored search results (query: same-day MH walk-in, no OHIP, Sault Ste. Marie) ──
const SEARCH_RESULTS = [
  { id: 'r-afs-walkin', name: 'Algoma Family Services · Walk-in Counselling',
    neighborhood: 'Sault Ste. Marie · downtown',
    blurb: 'Same-day mental-health assessment + single-session counselling. No appointment, no OHIP required. All ages.',
    chips: ['MH', 'walk-in', 'all ages', 'no OHIP'], verified: 5, total: 8, flagged: 1,
    score: 92 },
  { id: 'r-cmha-crisis', name: 'CMHA Algoma · Mobile Crisis Response',
    neighborhood: 'Sault Ste. Marie · regional',
    blurb: '24/7 crisis line. Mobile team responds across the City + Garden River. Bridge to community MH services.',
    chips: ['MH', 'crisis', '24/7', 'mobile'], verified: 6, total: 8, flagged: 0,
    score: 84 },
  { id: 'r-ghc-sda', name: 'Group Health Centre · Same-Day Access',
    neighborhood: 'Sault Ste. Marie',
    blurb: 'Same-day MH counselling for rostered GHC patients. Phone-first triage.',
    chips: ['MH', 'rostered only'], verified: 4, total: 8, flagged: 0,
    score: 58 },
  { id: 'r-connex', name: 'ConnexOntario · Virtual MH (province-wide)',
    neighborhood: 'Online',
    blurb: 'Province-wide phone navigation for MH + addictions. Long wait for specialist callback in evenings.',
    chips: ['MH', 'virtual'], verified: 2, total: 8, flagged: 2,
    score: 41 },
];

// ── Asks ──────────────────────────────────────────────────
const ASKS = [
  { id: 'a-104', who: 'Khalil · AOHT', region: 'Sault Ste. Marie', when: '2h ago',
    text: 'Low-cost dental for an elder on ODSP, Ojibwe-speaking interpreter available?',
    expires: '6 days', replies: 1, watching: 4, tags: ['senior', 'dental', 'language'] },
  { id: 'a-103', who: 'You', region: 'Garden River FN', when: '5h ago',
    text: 'Same-day MH walk-in for an adolescent without OHIP — any Algoma options not yet in the index?',
    expires: '6 days', replies: 3, watching: 7, tags: ['youth', 'MH', 'uninsured'], yours: true,
    hasNewReply: true },
  { id: 'a-101', who: 'Amélie · AOHT', region: 'Sault Ste. Marie', when: '1d ago',
    text: 'Postpartum peer group en français — any groups still meeting weekly?',
    expires: '5 days', replies: 2, watching: 3, tags: ['perinatal', 'francophone'] },
  { id: 'a-098', who: 'Priya · AOHT', region: 'Blind River', when: '2d ago',
    text: 'Methadone continuity + shelter coordination — looking for a single contact who handles both on the North Shore.',
    expires: '12 days · rural extended', replies: 0, watching: 6, tags: ['housing', 'sud'] },
  { id: 'a-095', who: 'Daniel · AOHT', region: 'Wawa', when: '3d ago',
    text: 'Anyone with a working number for the diabetes educator covering Wawa / White River? Last two numbers we had don\'t answer.',
    expires: '11 days · rural extended', replies: 1, watching: 5, tags: ['diabetes', 'primary-care'] },
];

const ASK_REPLIES = [
  { who: 'Marielle · AOHT', when: '1h ago',
    text: 'Algoma Family Services took a 16-year-old without OHIP last month — walked in at 11am, seen same day. Worth a call before sending.',
    attaches: { kind: 'resource', name: 'Algoma Family Services · Walk-in Counselling', id: 'r-afs-walkin', verified: 5, total: 8 } },
  { who: 'Devon · AOHT', when: '3h ago',
    text: 'CMHA Algoma\'s mobile crisis team can come to a school or home if there\'s an active risk piece. Less of a wait than calling around.',
    attaches: { kind: 'resource', name: 'CMHA Algoma · Mobile Crisis Response', id: 'r-cmha-crisis', verified: 6, total: 8 } },
  { who: 'AOHT member', when: '4h ago',
    text: 'There\'s a new youth drop-in starting at the Indian Friendship Centre — I don\'t have details yet. Posting anonymously, still verifying internally.',
    attaches: { kind: 'new', name: 'Indian Friendship Centre · Youth Drop-in (pilot)', note: 'Mentioned, not yet in index' } },
];

// ── Resource detail (Algoma Family Services walk-in) ──────
const RESOURCE = {
  id: 'r-afs-walkin',
  name: 'Algoma Family Services · Walk-in Counselling',
  org: 'Algoma Family Services',
  neighborhood: 'Sault Ste. Marie · downtown',
  blurb: 'Same-day mental health assessment + single-session counselling. No appointment, no OHIP required. Children, youth, adults.',
  fields: [
    { key: 'hours',       label: 'Hours',
      value: 'Mon · Wed · Fri — 9 am to 4 pm',
      state: 'verified-fresh', days: 2, confirms: 4, flags: 0,
      lastBy: 'Devon · AOHT', source: 'from a call' },
    { key: 'phone',       label: 'Phone',
      value: '(705) 555-0144 · option 2 for walk-in clinic',
      state: 'verified-fresh', days: 12, confirms: 7, flags: 0,
      lastBy: 'Marielle · AOHT', source: 'from a call' },
    { key: 'address',     label: 'Address',
      value: '101 McNabb St · Sault Ste. Marie, ON P6B 1Y4',
      state: 'verified-aging', days: 142, confirms: 2, flags: 0,
      lastBy: 'Sam · AOHT', source: 'in person' },
    { key: 'eligibility', label: 'Eligibility',
      value: 'Children, youth, adults · no OHIP required · uninsured welcome · English, French, Ojibwe interpretation available',
      state: 'ai-only', source: 'algomafamilyservices.org',
      lastBy: null },
    { key: 'fees',        label: 'Fees',
      value: 'No fee for walk-in · sliding scale for follow-up sessions',
      state: 'ai-only', source: 'algomafamilyservices.org',
      lastBy: null },
    { key: 'referral',    label: 'Referral path',
      value: 'Drop-in only · self-referrals · no formal referral required',
      state: 'flagged-stale', count: 2,
      lastBy: 'Khalil · AOHT', flagNote: '"They piloted phone-ahead booking from primary care this spring — confirmed 5/10 with intake clinician"',
      flaggedDays: 3 },
    { key: 'wait',        label: 'Typical wait',
      value: '20–45 min on Mon · longer on Fri afternoons',
      state: 'verified-fresh', days: 5, confirms: 3, flags: 0,
      lastBy: 'AOHT member', source: 'observed · anonymous' },
    { key: 'access',      label: 'Accessibility',
      value: 'Wheelchair accessible · scent-free building · transit on Wellington line',
      state: 'verified-aging', days: 210, confirms: 1, flags: 0,
      lastBy: 'Priya · AOHT', source: 'in person' },
  ],
  history: [
    { kind: 'flag-issue', who: 'Khalil · AOHT', when: '3d ago', field: 'Referral path',
      note: 'piloting phone-ahead booking from primary care' },
    { kind: 'confirm',    who: 'Devon · AOHT',  when: '2d ago', field: 'Hours' },
    { kind: 'confirm',    who: 'AOHT member',   when: '5d ago', field: 'Typical wait',
      note: 'submitted anonymously' },
    { kind: 'confirm',    who: 'Marielle · AOHT', when: '12d ago', field: 'Phone' },
    { kind: 'ai-seed',    who: 'AI',            when: '3 mo ago', field: 'All fields',
      note: 'seeded from algomafamilyservices.org' },
    { kind: 'ask-origin', who: 'Priya · AOHT',  when: '4 mo ago', field: null,
      note: 'asked: "Same-day MH walk-in for uninsured adult in Sault Ste. Marie?"' },
  ],
};

// ── The four moves of Pathways ────────────────────────────
// The product is one flywheel: Find → Verify → Ask → Close, and every
// Close feeds the index back into the next Find. These power the home-page
// loop, the per-page phase rail, and the guided tour's tracker.
const MOVES = [
  { key: 'find',   n: 1, label: 'Find',   tag: 'Search',   icon: 'search', file: 'search.html',
    blurb: 'Sketch the case in a sentence. Pathways ranks every Algoma service by fit — and by how fresh its facts are.' },
  { key: 'verify', n: 2, label: 'Verify', tag: 'Resource', icon: 'book', file: 'resource.html',
    blurb: 'Confirm or flag each field. Fresh facts climb in search; stale ones quietly drop until someone re-checks.' },
  { key: 'ask',    n: 3, label: 'Ask',    tag: 'Network',  icon: 'people', file: 'asks.html',
    blurb: 'Nothing fits? Ask the network. Replies often attach a service that was never indexed — seeding it for the next navigator.' },
  { key: 'close',  n: 4, label: 'Close',  tag: 'Resolve',  icon: 'check', file: 'case-resolve.html',
    blurb: 'Record where you referred. Your note seeds the directory, re-confirms what you used, and resolves your Ask.' },
];
const MOVE_LOOPBACK = 'Every case you close sharpens the next navigator’s search.';

function moveHref(file, root) { return root ? 'hifi/' + file : file; }

// Slim, persistent phase rail for the inner pages — always shows which of the
// four moves this page is, and lets reviewers step through the loop.
function PhaseRail({ current, root = false }) {
  const idx = MOVES.findIndex(m => m.key === current);
  return (
    <div className="phase-rail" role="navigation" aria-label="Pathways loop">
      <span className="phase-rail-label">The loop</span>
      <div className="phase-rail-track">
        {MOVES.map((m, i) => {
          const state = i < idx ? 'done' : i === idx ? 'active' : 'todo';
          return (
            <React.Fragment key={m.key}>
              <a href={moveHref(m.file, root)} className={'phase-step ' + state}
                 aria-current={state === 'active' ? 'step' : undefined}>
                <span className="pn">{state === 'done' ? '✓' : m.n}</span>
                <span className="pl">{m.label}</span>
              </a>
              {i < MOVES.length - 1 && <span className="phase-sep" aria-hidden="true">→</span>}
            </React.Fragment>
          );
        })}
        <span className="phase-loop-back" title={MOVE_LOOPBACK} aria-hidden="true">
          <span className="loop-glyph">↻</span> feeds&nbsp;Find
        </span>
      </div>
    </div>
  );
}

// Big home-page version — the demo spine. Four numbered stages laid out as a
// cycle, each clickable into that part of the prototype, with the feedback
// lane returning from Close to Find.
function PhaseLoop({ root = false }) {
  return (
    <div className="phase-loop">
      <div className="phase-loop-stages">
        {MOVES.map((m, i) => (
          <React.Fragment key={m.key}>
            <a href={moveHref(m.file, root)} className="loop-card">
              <div className="loop-card-head">
                <span className="loop-num">{m.n}</span>
                <span className="loop-move">{m.label}</span>
                <span className="loop-tag">{m.tag}</span>
              </div>
              <p className="loop-blurb">{m.blurb}</p>
              <span className="loop-open">
                Open <Ico name="arrow-right" size={12} />
              </span>
            </a>
            {i < MOVES.length - 1 && (
              <span className="loop-arrow" aria-hidden="true">
                <Ico name="arrow-right" size={15} />
              </span>
            )}
          </React.Fragment>
        ))}
      </div>
      <div className="phase-loop-return" aria-hidden="true">
        <span className="return-line" />
        <span className="return-note">
          <span className="loop-glyph">↻</span> {MOVE_LOOPBACK}
        </span>
        <span className="return-line" />
      </div>
    </div>
  );
}

Object.assign(window, {
  Ico, Topnav, TopnavRoot, MicMount,
  CASES, SEARCH_RESULTS, ASKS, ASK_REPLIES, RESOURCE,
  matchTier, MatchMeter,
  VerifyBadge, VerifyAggregate,
  MOVES, MOVE_LOOPBACK, moveHref, PhaseRail, PhaseLoop,
});
