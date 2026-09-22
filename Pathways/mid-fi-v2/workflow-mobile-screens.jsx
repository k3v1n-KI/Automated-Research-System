// Pathways v2 — Mobile companion screens
// Three iOS artboards: search results · resource detail (thumb-reach verify) · answer-an-Ask.

// Tiny atoms locally — iOS-flavored padding + type
const MOB = {
  ink: '#18181b', ink2: '#3f3f46', ink3: '#71717a', ink4: '#a1a1aa',
  paper: '#fff', paper2: '#fafafa', paper3: '#f4f4f5',
  stroke: '#e4e4e7', stroke2: '#d4d4d8',
  accent: '#7c3aed', accentInk: '#5b21b6', accentTint: '#faf6ff',
  good: '#15803d', goodBg: '#ecfdf5',
  warn: '#a16207', warnBg: '#fefce8',
  crit: '#b91c1c', critBg: '#fef2f2',
  mono: 'JetBrains Mono, ui-monospace, Menlo, monospace',
  ui: 'Inter, -apple-system, system-ui',
};

function MobileTopBar({ title, back = true }) {
  return (
    <div style={{
      marginTop: 56, padding: '10px 16px 12px',
      borderBottom: `1px solid ${MOB.stroke}`,
      display: 'flex', alignItems: 'center', gap: 10,
      background: MOB.paper,
    }}>
      {back && (
        <span style={{ display: 'inline-flex', alignItems: 'center', color: MOB.accent, gap: 3, fontSize: 14 }}>
          <svg width="11" height="14" viewBox="0 0 11 14" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round"><path d="M8 1L2 7l6 6"/></svg>
          Back
        </span>
      )}
      <span style={{ flex: 1, textAlign: 'center', fontSize: 14, fontWeight: 600, color: MOB.ink, letterSpacing: '-0.012em' }}>
        {title}
      </span>
      <span style={{ width: 40 }}/>
    </div>
  );
}

// ───────────────────────────────────────────────
// Mobile · Search results
// ───────────────────────────────────────────────
function MobileSearch() {
  return (
    <IOSDevice width={390} height={844}>
      <MobileTopBar title="Search" back={false}/>

      {/* Search bar */}
      <div style={{ padding: '12px 16px', background: MOB.paper, borderBottom: `1px solid ${MOB.stroke}` }}>
        <div style={{
          background: MOB.paper2, border: `1px solid ${MOB.stroke}`,
          borderRadius: 10, padding: '10px 12px', display: 'flex', alignItems: 'center', gap: 8,
          fontFamily: MOB.ui,
        }}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke={MOB.ink3} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></svg>
          <span style={{ fontSize: 13, color: MOB.ink, flex: 1 }}>Same-day MH walk-in · no OHIP</span>
          <span style={{ fontFamily: MOB.mono, fontSize: 10.5, color: MOB.ink3 }}>4 results</span>
        </div>
        <div style={{ display: 'flex', gap: 6, marginTop: 8, overflowX: 'auto' }}>
          {['MH', 'walk-in', 'uninsured', 'East Toronto'].map(t => (
            <span key={t} style={{
              padding: '3px 9px', fontSize: 11, background: MOB.paper3,
              border: `1px solid ${MOB.stroke}`, borderRadius: 999, color: MOB.ink2, fontFamily: MOB.ui, whiteSpace: 'nowrap',
            }}>{t}</span>
          ))}
        </div>
      </div>

      {/* Results */}
      <div style={{ padding: 12, background: MOB.paper2, display: 'flex', flexDirection: 'column', gap: 10, fontFamily: MOB.ui }}>
        {SEARCH_RESULTS.slice(0, 3).map((r, i) => {
          const m = matchTier(r.score);
          return (
          <div key={r.id} style={{
            background: MOB.paper, border: `1px solid ${MOB.stroke}`, borderRadius: 12,
            padding: 14,
          }}>
            <div style={{ display: 'flex', alignItems: 'flex-start', gap: 10, marginBottom: 8 }}>
              <div style={{
                width: 38, height: 38, borderRadius: 8,
                background: i === 0 ? MOB.ink : MOB.paper2,
                color: i === 0 ? MOB.paper : MOB.ink2,
                display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 3,
                flexShrink: 0,
              }}>
                <MatchMeter tier={m.tier} dark={i === 0} size="sm"/>
                <span style={{ fontFamily: MOB.mono, fontSize: 7.5, opacity: 0.7 }}>{m.tier}/5</span>
              </div>
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ fontSize: 14, fontWeight: 600, color: MOB.ink, letterSpacing: '-0.012em', marginBottom: 2 }}>
                  {r.name}
                </div>
                <div style={{ fontFamily: MOB.mono, fontSize: 10.5, color: MOB.ink3 }}>{r.neighborhood} · {m.label}</div>
              </div>
            </div>
            <div style={{ fontSize: 12.5, color: MOB.ink2, lineHeight: 1.5, marginBottom: 10 }}>
              {r.blurb}
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 11, color: MOB.ink3 }}>
              <span style={{ width: 6, height: 6, borderRadius: '50%',
                background: r.flagged > 0 ? MOB.crit : (r.verified / r.total) >= 0.7 ? MOB.good : MOB.accent }}/>
              <span style={{ fontFamily: MOB.mono, color: r.flagged > 0 ? MOB.crit : (r.verified / r.total) >= 0.7 ? MOB.good : MOB.accentInk, fontWeight: 500 }}>
                {r.verified}/{r.total} verified
              </span>
              {r.flagged > 0 && <span style={{ color: MOB.crit, fontWeight: 500 }}>· {r.flagged} flagged</span>}
            </div>
          </div>
          );
        })}

        {/* Post as Ask */}
        <div style={{
          background: MOB.accentTint, border: `1px dashed ${MOB.stroke2}`,
          borderRadius: 12, padding: 14,
        }}>
          <div style={{ fontSize: 11, color: MOB.accentInk, fontFamily: MOB.mono, textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 6 }}>
            Not finding it?
          </div>
          <div style={{ fontSize: 13, color: MOB.ink, lineHeight: 1.45, marginBottom: 12 }}>
            Post as an <strong>Ask</strong>. Colleagues in East Toronto see it for 7 days.
          </div>
          <button style={{
            background: MOB.accent, color: MOB.paper, border: 'none', padding: '10px 14px',
            borderRadius: 8, fontSize: 13, fontWeight: 500, fontFamily: MOB.ui, width: '100%',
          }}>Post as Ask →</button>
        </div>
      </div>
    </IOSDevice>
  );
}

// ───────────────────────────────────────────────
// Mobile · Resource detail (thumb-reach verify)
// ───────────────────────────────────────────────
function MobileResource() {
  const fields = RESOURCE.fields.slice(0, 5);
  return (
    <IOSDevice width={390} height={844}>
      <MobileTopBar title="Resource"/>

      {/* Header */}
      <div style={{ padding: '14px 16px 16px', background: MOB.paper, borderBottom: `1px solid ${MOB.stroke}`, fontFamily: MOB.ui }}>
        <div style={{ fontFamily: MOB.mono, fontSize: 10, color: MOB.ink3, textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 6 }}>
          Resource · East York
        </div>
        <h1 style={{ fontSize: 19, fontWeight: 600, letterSpacing: '-0.016em', lineHeight: 1.2, color: MOB.ink, margin: 0, marginBottom: 8 }}>
          {RESOURCE.name}
        </h1>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11, color: MOB.ink3 }}>
          <span style={{ width: 6, height: 6, borderRadius: '50%', background: MOB.crit }}/>
          <span style={{ fontFamily: MOB.mono, color: MOB.crit, fontWeight: 500 }}>5/8 verified</span>
          <span>·</span>
          <span style={{ color: MOB.crit, fontWeight: 500 }}>1 flagged</span>
        </div>
      </div>

      {/* Quick actions */}
      <div style={{ padding: '10px 16px', background: MOB.paper, borderBottom: `1px solid ${MOB.stroke}`, display: 'flex', gap: 8, fontFamily: MOB.ui }}>
        <button style={{ flex: 1, padding: '9px 0', background: MOB.paper, border: `1px solid ${MOB.stroke2}`, borderRadius: 8, fontSize: 12.5, color: MOB.ink, fontWeight: 500, fontFamily: MOB.ui }}>
          Save
        </button>
        <button style={{ flex: 1, padding: '9px 0', background: MOB.paper, border: `1px solid ${MOB.stroke2}`, borderRadius: 8, fontSize: 12.5, color: MOB.ink, fontWeight: 500, fontFamily: MOB.ui }}>
          Call
        </button>
        <button style={{ flex: 1.4, padding: '9px 0', background: MOB.accent, border: `1px solid ${MOB.accent}`, borderRadius: 8, fontSize: 12.5, color: MOB.paper, fontWeight: 500, fontFamily: MOB.ui }}>
          Refer →
        </button>
      </div>

      {/* Fields */}
      <div style={{ padding: '8px 12px 16px', background: MOB.paper2, fontFamily: MOB.ui }}>
        {fields.map((f, i) => <MobileFieldCard key={f.key} f={f}/>)}

        {/* Hint */}
        <div style={{
          marginTop: 10, padding: 12, background: MOB.paper, border: `1px solid ${MOB.stroke}`,
          borderRadius: 10, fontSize: 11.5, color: MOB.ink3, lineHeight: 1.5,
        }}>
          One tap per field — your name shows on confirmations by default.
          Slide left on any field to flag, or tap <strong style={{ color: MOB.crit, fontWeight: 500 }}>Flag</strong>.
        </div>
      </div>
    </IOSDevice>
  );
}

function MobileFieldCard({ f }) {
  const tone = f.state === 'verified-fresh' ? { bg: MOB.goodBg, color: MOB.good, label: 'Verified', sub: `${f.days}d ago` }
            : f.state === 'verified-aging' ? { bg: MOB.warnBg, color: MOB.warn, label: 'Verified', sub: `${f.days}d old` }
            : f.state === 'ai-only' ? { bg: MOB.accentTint, color: MOB.accentInk, label: 'AI extracted', sub: 'never confirmed' }
            : { bg: MOB.critBg, color: MOB.crit, label: 'Flagged stale', sub: f.count ? `${f.count}×` : '' };
  return (
    <div style={{
      background: MOB.paper, border: `1px solid ${MOB.stroke}`, borderRadius: 10,
      padding: 12, marginBottom: 8,
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 10, marginBottom: 6 }}>
        <div style={{ fontFamily: MOB.mono, fontSize: 10, color: MOB.ink3, textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 500 }}>
          {f.label}
        </div>
        <div style={{
          padding: '2px 7px', borderRadius: 999, background: tone.bg, color: tone.color,
          fontSize: 10.5, fontWeight: 500, display: 'inline-flex', alignItems: 'center', gap: 4, whiteSpace: 'nowrap',
        }}>
          <span>{f.state === 'flagged-stale' ? '⚠' : f.state === 'ai-only' ? '⬡' : '✓'}</span>
          <span>{tone.label}{tone.sub ? ` · ${tone.sub}` : ''}</span>
        </div>
      </div>
      <div style={{ fontSize: 13.5, color: MOB.ink, lineHeight: 1.45, marginBottom: 10 }}>
        {f.value}
      </div>
      <div style={{ display: 'flex', gap: 8 }}>
        <button style={{
          flex: 1, padding: '8px 0', background: MOB.paper, border: `1px solid ${MOB.stroke2}`,
          borderRadius: 8, fontSize: 12.5, color: MOB.ink, fontWeight: 500, fontFamily: MOB.ui,
          display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6,
        }}>
          <span style={{ color: MOB.good, fontSize: 13 }}>✓</span>
          Confirm
        </button>
        <button style={{
          flex: 1, padding: '8px 0', background: MOB.paper, border: `1px solid ${MOB.stroke2}`,
          borderRadius: 8, fontSize: 12.5, color: MOB.ink, fontWeight: 500, fontFamily: MOB.ui,
          display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6,
        }}>
          <span style={{ color: MOB.crit, fontSize: 13 }}>⚠</span>
          Flag
        </button>
      </div>
    </div>
  );
}

// ───────────────────────────────────────────────
// Mobile · Answer an Ask
// ───────────────────────────────────────────────
function MobileAskAnswer() {
  return (
    <IOSDevice width={390} height={844} keyboard>
      <MobileTopBar title="Reply to Ask"/>

      <div style={{ padding: '14px 16px', background: MOB.paper, borderBottom: `1px solid ${MOB.stroke}`, fontFamily: MOB.ui }}>
        {/* The Ask */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
          <span style={{
            width: 22, height: 22, borderRadius: '50%', background: MOB.paper3,
            border: `1px solid ${MOB.stroke}`, display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
            fontSize: 9, fontWeight: 600, color: MOB.ink2,
          }}>KH</span>
          <span style={{ fontSize: 12.5, color: MOB.ink, fontWeight: 500 }}>Khalil · AOHT</span>
          <span style={{ fontFamily: MOB.mono, fontSize: 10.5, color: MOB.ink3 }}>· 2h ago</span>
          <span style={{ flex: 1 }}/>
          <span style={{ fontFamily: MOB.mono, fontSize: 10, color: MOB.accentInk }}>6d left</span>
        </div>
        <p style={{ fontSize: 14, color: MOB.ink, lineHeight: 1.45, margin: 0, marginBottom: 10, letterSpacing: '-0.008em' }}>
          Low-cost dental for seniors, Cantonese-speaking, will travel to Scarborough?
        </p>
        <div style={{ display: 'flex', gap: 5, flexWrap: 'wrap' }}>
          {['senior', 'dental', 'Cantonese'].map(t => (
            <span key={t} style={{ padding: '2px 7px', fontSize: 10.5, background: MOB.accentTint,
              border: `1px solid color-mix(in srgb, ${MOB.accent} 18%, transparent)`,
              color: MOB.accentInk, borderRadius: 999, fontWeight: 500 }}>{t}</span>
          ))}
        </div>
      </div>

      {/* Reply composer */}
      <div style={{ padding: 14, background: MOB.paper2, fontFamily: MOB.ui }}>
        <div style={{ fontFamily: MOB.mono, fontSize: 10, color: MOB.ink3, textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 8 }}>
          Your reply
        </div>
        <div style={{
          background: MOB.paper, border: `1px solid ${MOB.accent}`,
          boxShadow: `0 0 0 3px color-mix(in srgb, ${MOB.accent} 12%, transparent)`,
          borderRadius: 10, padding: 12, marginBottom: 10,
        }}>
          <div style={{ fontSize: 13, color: MOB.ink, lineHeight: 1.5, minHeight: 56 }}>
            Cedar Health has a Cantonese volunteer dental clinic on Saturdays. Ask for Susan │
          </div>
        </div>

        {/* Attach picker */}
        <div style={{ fontFamily: MOB.mono, fontSize: 10, color: MOB.ink3, textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 6 }}>
          Attach
        </div>
        <div style={{ display: 'flex', gap: 6, marginBottom: 12, flexWrap: 'wrap' }}>
          {[
            { icon: '⚲', label: 'Search resource' },
            { icon: '+', label: 'Add new' },
            { icon: '◉', label: 'Voice note' },
            { icon: '◎', label: 'Photo' },
          ].map(b => (
            <button key={b.label} style={{
              flex: '1 1 auto', minWidth: 0, padding: '8px 10px',
              background: MOB.paper, border: `1px solid ${MOB.stroke}`, borderRadius: 8,
              fontSize: 11.5, color: MOB.ink, fontWeight: 500, fontFamily: MOB.ui,
              display: 'flex', alignItems: 'center', gap: 6, whiteSpace: 'nowrap',
            }}>
              <span style={{ color: MOB.ink3 }}>{b.icon}</span>
              {b.label}
            </button>
          ))}
        </div>

        {/* Anonymity + submit */}
        <div style={{
          display: 'flex', justifyContent: 'space-between', alignItems: 'center',
          fontSize: 11.5, color: MOB.ink2, marginBottom: 10,
        }}>
          <span>Reply as <strong style={{ fontWeight: 500 }}>Rita · AOHT</strong></span>
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6, color: MOB.ink3 }}>
            <span style={{ width: 26, height: 16, background: MOB.paper3, borderRadius: 999, position: 'relative', border: `1px solid ${MOB.stroke2}` }}>
              <span style={{ position: 'absolute', top: 1, left: 1, width: 12, height: 12, borderRadius: '50%', background: MOB.paper, border: `1px solid ${MOB.stroke2}` }}/>
            </span>
            <span>Anonymous</span>
          </span>
        </div>

        <button style={{
          width: '100%', padding: '12px 0', background: MOB.accent, color: MOB.paper,
          border: 'none', borderRadius: 10, fontSize: 14, fontWeight: 500, fontFamily: MOB.ui,
        }}>Send reply →</button>
      </div>
    </IOSDevice>
  );
}

Object.assign(window, { MobileSearch, MobileResource, MobileAskAnswer });
