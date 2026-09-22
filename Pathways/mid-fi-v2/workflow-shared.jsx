// Pathways v2 — workflow shared: verification model + sample data + atoms

// ── Verification state model ─────────────────────────────
// A field on a resource can be in one of these states.
// State is computed from the flag log (confirms vs. issues) + age.
//
//   verified-fresh  → ✓ recent confirms, no recent issues
//   verified-aging  → ✓ confirmed but ≥ decay window
//   ai-only         → ⬡ AI-extracted, never member-confirmed
//   flagged-stale   → ⚠ one or more recent "stale/wrong" flags
//   unknown         → — no data
//
// Decay window varies by field type:
//   phone: 180d · hours: 30d · eligibility: 90d · address: 365d
// ──────────────────────────────────────────────────────────

function VerifyBadge({ state, count, days, source, compact = false }) {
  const variants = {
    'verified-fresh': { icon: 'check',  label: 'Verified', tone: 'good',    glyph: '✓' },
    'verified-aging': { icon: 'clock',  label: 'Verified', tone: 'warn',    glyph: '✓' },
    'ai-only':        { icon: 'sparkle',label: 'AI extracted', tone: 'ai',  glyph: '⬡' },
    'flagged-stale':  { icon: 'flag',   label: 'Flagged stale',tone: 'crit',glyph: '⚠' },
    'unknown':        { icon: null,     label: 'Unknown', tone: 'mute',     glyph: '—' },
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

// Aggregate verification read for a resource (used in search rows)
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

// Tiny attribution line: "Verified by AOHT member · May 12 · from a call"
function AttribLine({ who, when, source }) {
  return (
    <span className="mono" style={{ fontSize: 10.5, color: 'var(--ink-3)' }}>
      {who}{when ? ` · ${when}` : ''}{source ? ` · ${source}` : ''}
    </span>
  );
}

// ── Sample data ──────────────────────────────────────────

const RESOURCE = {
  id: 'r-acrosshealth-mh',
  name: 'Across Health · Mental Health Walk-in',
  org: 'Across Health Centre',
  neighborhood: 'East York',
  blurb: 'Same-day mental health assessment + brief intervention. No appointment, no OHIP required.',
  fields: [
    { key: 'hours',       label: 'Hours',
      value: 'Tue & Wed · 1–6 pm',
      state: 'verified-fresh', days: 2, confirms: 4, flags: 0,
      lastBy: 'Devon · AOHT', source: 'from a call' },
    { key: 'phone',       label: 'Phone',
      value: '(416) 555-0144 · ext. 2 for intake',
      state: 'verified-fresh', days: 12, confirms: 7, flags: 0,
      lastBy: 'Maria · AOHT', source: 'from a call' },
    { key: 'address',     label: 'Address',
      value: '847 Mortimer Ave, East York, M4J 2G7',
      state: 'verified-aging', days: 142, confirms: 2, flags: 0,
      lastBy: 'Sam · AOHT', source: 'in person' },
    { key: 'eligibility', label: 'Eligibility',
      value: 'Adults 18+ · no OHIP required · uninsured welcome · Cantonese / English / French',
      state: 'ai-only', source: 'acrosshealth.ca/services',
      lastBy: null },
    { key: 'fees',        label: 'Fees',
      value: 'No fee · sliding scale for follow-up',
      state: 'ai-only', source: 'acrosshealth.ca/fees',
      lastBy: null },
    { key: 'referral',    label: 'Referral path',
      value: 'Drop-in only · referrals not accepted',
      state: 'flagged-stale', count: 2,
      lastBy: 'Khalil · AOHT', flagNote: '"They\'re piloting fax referrals from primary care — confirmed 5/10"',
      flaggedDays: 3 },
    { key: 'wait',        label: 'Typical wait',
      value: '20–45 min on Tue · longer on Wed',
      state: 'verified-fresh', days: 5, confirms: 3, flags: 0,
      lastBy: 'AOHT member', source: 'observed · anonymous' },
    { key: 'access',      label: 'Accessibility',
      value: 'Wheelchair accessible · scent-free building',
      state: 'verified-aging', days: 210, confirms: 1, flags: 0,
      lastBy: 'Priya · AOHT', source: 'in person' },
  ],
  history: [
    { kind: 'flag-issue', who: 'Khalil · AOHT', when: '3d ago', field: 'Referral path', note: 'piloting fax referrals from primary care' },
    { kind: 'confirm',    who: 'Devon · AOHT',  when: '2d ago', field: 'Hours' },
    { kind: 'confirm',    who: 'AOHT member',   when: '5d ago', field: 'Typical wait', note: 'submitted anonymously' },
    { kind: 'confirm',    who: 'Maria · AOHT',  when: '12d ago',field: 'Phone' },
    { kind: 'ai-seed',    who: 'AI',            when: '3mo ago',field: 'All fields',  note: 'seeded from acrosshealth.ca' },
    { kind: 'ask-origin', who: 'Priya · AOHT',  when: '4mo ago',field: null,
      note: 'asked: "Same-day MH walk-in for uninsured adult in East Toronto?"' },
  ],
};

// Match tiers — 1 (weakest) to 5 (strongest). Score is kept for ranking,
// but the UI only ever shows tier + label.
function matchTier(score) {
  if (score >= 90) return { tier: 5, label: 'Strong match' };
  if (score >= 75) return { tier: 4, label: 'Good match' };
  if (score >= 60) return { tier: 3, label: 'Fair match' };
  if (score >= 40) return { tier: 2, label: 'Partial match' };
  return { tier: 1, label: 'Weak match' };
}
window.matchTier = matchTier;

// 5-bar tier indicator (signal-strength glyph). Filled bars = tier.
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
window.MatchMeter = MatchMeter;

// Search results sample
const SEARCH_RESULTS = [
  { id: 'r-acrosshealth-mh', name: 'Across Health · Mental Health Walk-in', neighborhood: 'East York',
    blurb: 'Same-day mental health assessment + brief intervention. No appointment, no OHIP required.',
    chips: ['MH', 'walk-in', 'uninsured', 'Cantonese'], verified: 5, total: 8, flagged: 1,
    score: 92, scoreLabel: 'Strong match' },
  { id: 'r-stmichaels-mh', name: 'St. Michael\'s · Hope Outreach Clinic', neighborhood: 'Downtown East',
    blurb: 'Drop-in mental health for adults experiencing homelessness or instability. OHIP not required.',
    chips: ['MH', 'walk-in', 'shelter-friendly'], verified: 6, total: 8, flagged: 0,
    score: 84, scoreLabel: 'Good match' },
  { id: 'r-eastside-cmh', name: 'Eastside Community MH', neighborhood: 'Riverdale',
    blurb: 'Counselling intake — wait 4–6 weeks. Accepts uninsured on sliding scale.',
    chips: ['MH', 'long-wait'], verified: 2, total: 8, flagged: 0,
    score: 58, scoreLabel: 'Partial match · long wait' },
  { id: 'r-onlinemh', name: 'OHT Virtual MH (province-wide)', neighborhood: 'Online',
    blurb: 'Same-day phone or video assessment. Province-wide. Eligibility unverified for uninsured.',
    chips: ['MH', 'virtual'], verified: 1, total: 8, flagged: 2,
    score: 41, scoreLabel: 'Weak match · needs verifying' },
];

// Asks feed sample
const ASKS = [
  { id: 'a-104', who: 'Khalil · AOHT', region: 'East York', when: '2h ago',
    text: 'Low-cost dental for seniors, Cantonese-speaking, will travel to Scarborough?',
    expires: '6 days', replies: 1, watching: 4, tags: ['senior', 'dental', 'language'] },
  { id: 'a-103', who: 'You', region: 'East York', when: '5h ago',
    text: 'Same-day MH walk-in for an adolescent without OHIP — any options not on the index?',
    expires: '6 days', replies: 3, watching: 7, tags: ['youth', 'MH', 'uninsured'], yours: true,
    hasNewReply: true },
  { id: 'a-101', who: 'Amelie · AOHT', region: 'Scarborough', when: '1d ago',
    text: 'Postpartum support group in French — any peer-led groups still meeting?',
    expires: '5 days', replies: 2, watching: 3, tags: ['perinatal', 'language'] },
  { id: 'a-098', who: 'Priya · AOHT', region: 'Riverdale', when: '2d ago',
    text: 'Continuing methadone + shelter coordination — looking for a single contact who handles both.',
    expires: '12 days · rural extended', replies: 0, watching: 6, tags: ['housing', 'sud'] },
];

const ASK_REPLIES = [
  { who: 'Maria · AOHT', when: '1h ago', text: 'Tried Roncesvalles Youth Hub last month for a similar case — they took a 16yo without OHIP. Worth a call.',
    attaches: { kind: 'resource', name: 'Roncesvalles Youth Hub', verified: 4, total: 8 } },
  { who: 'Devon · AOHT', when: '3h ago', text: 'Across Health has done it case-by-case. Not advertised but ask for the intake clinician directly.',
    attaches: { kind: 'resource', name: 'Across Health · MH Walk-in', verified: 5, total: 8 } },
  { who: 'AOHT member', when: '4h ago', text: 'There\'s a new pilot at Michael Garron — I don\'t have details. Anyone? (posting anonymously, still verifying internally)',
    attaches: { kind: 'new', name: 'Michael Garron MH pilot', note: 'Mentioned, not yet in index' } },
];

Object.assign(window, {
  VerifyBadge, VerifyAggregate, AttribLine,
  RESOURCE, SEARCH_RESULTS, ASKS, ASK_REPLIES,
});
