// Pathways v2 — Mobile paste-anything sheet (iOS share intent)
// Imagine: user is in Facebook, sees a post about a new clinic, taps Share → Pathways.
// Lands here. AI structures the share into a resource candidate. User confirms 2 things and submits.

function MobilePaste() {
  return (
    <IOSDevice width={390} height={844} dark={false}>
      {/* Header strip indicating share-sheet origin */}
      <div style={{
        marginTop: 56, padding: '8px 16px',
        background: 'var(--accent-tint)', borderBottom: '1px solid color-mix(in srgb, var(--accent) 18%, transparent)',
        display: 'flex', alignItems: 'center', gap: 8,
        fontFamily: 'Inter, system-ui',
      }}>
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#7c3aed" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
          <path d="M4 12v8a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-8"/><path d="M16 6l-4-4-4 4"/><path d="M12 2v13"/>
        </svg>
        <span style={{ fontSize: 11.5, color: '#5b21b6', fontWeight: 500 }}>
          Shared from Facebook · 11:42 AM
        </span>
      </div>

      <div style={{
        padding: '20px 18px 24px',
        fontFamily: 'Inter, -apple-system, system-ui',
        color: '#18181b',
      }}>

        {/* Header */}
        <div style={{ marginBottom: 14 }}>
          <div style={{ fontFamily: 'JetBrains Mono', fontSize: 10, color: '#71717a',
            textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 4 }}>
            Got a tip?
          </div>
          <h1 style={{ fontSize: 22, fontWeight: 600, letterSpacing: '-0.022em', lineHeight: 1.2, margin: 0 }}>
            We'll structure this for you.
          </h1>
        </div>

        {/* Shared content preview */}
        <div style={{
          background: '#fafafa', border: '1px solid #e4e4e7', borderRadius: 10,
          padding: 12, marginBottom: 14,
        }}>
          <div style={{ fontSize: 10.5, color: '#71717a', fontFamily: 'JetBrains Mono', marginBottom: 6 }}>
            FACEBOOK POST · East Toronto Health Workers (private group)
          </div>
          <div style={{ fontSize: 13, color: '#3f3f46', lineHeight: 1.5 }}>
            "Heads up — Michael Garron just opened a same-day youth MH walk-in.
            Mon–Thu 9–4. They take uninsured. Call 416-555-0212 first to confirm
            they're open that day. Pilot through August."
          </div>
          <div style={{ fontSize: 10.5, color: '#a1a1aa', marginTop: 8, fontFamily: 'JetBrains Mono' }}>
            posted by @lina.huang · 2h ago
          </div>
        </div>

        {/* AI extraction */}
        <div style={{ marginBottom: 14 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 10 }}>
            <svg width="11" height="11" viewBox="0 0 24 24" fill="#7c3aed">
              <path d="M12 3l1.8 5L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.7L12 3z"/>
            </svg>
            <span style={{ fontSize: 11, fontFamily: 'JetBrains Mono', textTransform: 'uppercase',
              letterSpacing: '0.06em', color: '#5b21b6', fontWeight: 500 }}>
              AI extracted · tap to fix
            </span>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            <ExtractRow label="Name"        value="Michael Garron · Youth MH Walk-in"/>
            <ExtractRow label="Hours"       value="Mon–Thu · 9 am – 4 pm"/>
            <ExtractRow label="Phone"       value="(416) 555-0212"/>
            <ExtractRow label="Eligibility" value="Youth · uninsured welcome · no OHIP required"/>
            <ExtractRow label="Notes"       value="Pilot through August · call before to confirm"/>
            <ExtractRow label="Neighborhood" value="East York · Toronto" tone="muted"/>
          </div>
        </div>

        {/* Verify the 2 things that matter */}
        <div style={{
          background: '#fef7ff00', border: '1px solid #e4e4e7', borderRadius: 10,
          padding: '12px 14px', marginBottom: 14,
          backgroundColor: '#faf6ff',
        }}>
          <div style={{ fontSize: 12, color: '#5b21b6', fontWeight: 500, marginBottom: 6 }}>
            Before we save · quick check
          </div>
          <div style={{ fontSize: 13, color: '#3f3f46', lineHeight: 1.5 }}>
            <strong style={{ fontWeight: 500, color: '#18181b' }}>Name + phone</strong> look right?
            Everything else can be flagged later.
          </div>
        </div>

        {/* Visibility */}
        <div style={{
          display: 'flex', justifyContent: 'space-between', alignItems: 'center',
          padding: '12px 0', borderTop: '1px solid #e4e4e7',
          fontSize: 12, color: '#52525b',
        }}>
          <span>Submit as <strong style={{ fontWeight: 500 }}>Rita Okonkwo · AOHT</strong></span>
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6, color: '#71717a' }}>
            <span style={{ width: 26, height: 16, background: '#e4e4e7', borderRadius: 999, position: 'relative' }}>
              <span style={{ position: 'absolute', top: 1, left: 1, width: 12, height: 12, borderRadius: '50%', background: '#fff', border: '1px solid #d4d4d8' }}/>
            </span>
            <span>Anonymous</span>
          </span>
        </div>

        {/* Buttons */}
        <div style={{ display: 'flex', gap: 8, marginTop: 8 }}>
          <button style={{
            flex: 1, padding: '12px 16px', background: '#fff', border: '1px solid #d4d4d8',
            borderRadius: 8, fontSize: 14, fontWeight: 500, color: '#18181b', fontFamily: 'inherit',
          }}>
            Looks wrong
          </button>
          <button style={{
            flex: 2, padding: '12px 16px', background: '#7c3aed', border: '1px solid #7c3aed',
            borderRadius: 8, fontSize: 14, fontWeight: 500, color: '#fff', fontFamily: 'inherit',
          }}>
            Add to index →
          </button>
        </div>
      </div>
    </IOSDevice>
  );
}

function ExtractRow({ label, value, tone }) {
  return (
    <div style={{
      display: 'grid', gridTemplateColumns: '92px 1fr', gap: 10, alignItems: 'flex-start',
      paddingBottom: 8, borderBottom: '1px dashed #e4e4e7',
    }}>
      <div style={{
        fontSize: 10.5, color: '#71717a', fontFamily: 'JetBrains Mono',
        textTransform: 'uppercase', letterSpacing: '0.04em', paddingTop: 2,
      }}>{label}</div>
      <div style={{ fontSize: 13, color: tone === 'muted' ? '#71717a' : '#18181b', lineHeight: 1.45 }}>{value}</div>
    </div>
  );
}

window.MobilePaste = MobilePaste;
