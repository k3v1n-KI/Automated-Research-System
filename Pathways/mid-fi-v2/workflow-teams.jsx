// Pathways v2 — Teams setup
// Two-paths-to-a-team UI: auto-suggest from domain · or join with a code.

function TeamsSetup() {
  return (
    <div className="v2" style={{ padding: 40, background: 'var(--paper)' }}>
      <div className="eyebrow" style={{ marginBottom: 10 }}>Settings · Teams</div>
      <h2 style={{ fontSize: 24, fontWeight: 600, letterSpacing: '-0.022em', marginBottom: 4 }}>
        Work with colleagues.
      </h2>
      <p style={{ fontSize: 13, color: 'var(--ink-3)', lineHeight: 1.5, marginBottom: 22, maxWidth: 560 }}>
        Teams are a layer over your individual account. Your contributions stay yours. A team gives you a
        shared resource library, a private feed of your colleagues' saves, and the option to post Asks
        to your team only.
      </p>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14, maxWidth: 720 }}>

        {/* Auto-suggested team */}
        <div className="card" style={{
          padding: 18, borderColor: 'color-mix(in srgb, var(--accent) 28%, transparent)',
          background: 'var(--accent-tint)',
        }}>
          <div className="row" style={{ gap: 6, marginBottom: 10 }}>
            <V2Ico name="people" size={12} stroke={1.8} style={{ color: 'var(--accent)' }}/>
            <span className="eyebrow" style={{ color: 'var(--accent-ink)' }}>Detected · cedarhealth.ca</span>
          </div>
          <h3 style={{ fontSize: 16, fontWeight: 600, letterSpacing: '-0.014em', marginBottom: 6 }}>
            3 colleagues from Cedar Health are here.
          </h3>
          <p style={{ fontSize: 12.5, color: 'var(--ink-2)', lineHeight: 1.5, marginBottom: 14 }}>
            Form a team to share a saved-resource library and see what they refer to. You can change
            the team name later.
          </p>
          <div className="row" style={{ gap: 8, marginBottom: 12 }}>
            {['MR', 'AS', 'KL'].map(i => (
              <span key={i} className="avatar" style={{ background: 'var(--paper)', borderColor: 'var(--stroke)' }}>{i}</span>
            ))}
            <span className="mono" style={{ fontSize: 11, color: 'var(--ink-3)' }}>+ you</span>
          </div>
          <div className="row" style={{ gap: 8 }}>
            <button className="btn accent sm">Form team</button>
            <button className="btn ghost sm">Not now</button>
          </div>
        </div>

        {/* Join with code */}
        <div className="card" style={{ padding: 18 }}>
          <div className="row" style={{ gap: 6, marginBottom: 10 }}>
            <V2Ico name="corner-arrow" size={12} stroke={1.8} style={{ color: 'var(--ink-3)' }}/>
            <span className="eyebrow">Join with a code</span>
          </div>
          <h3 style={{ fontSize: 16, fontWeight: 600, letterSpacing: '-0.014em', marginBottom: 6 }}>
            For cross-org teams.
          </h3>
          <p style={{ fontSize: 12.5, color: 'var(--ink-2)', lineHeight: 1.5, marginBottom: 14 }}>
            A working group or task force. One member creates a team, shares the code in chat. Others
            paste it here.
          </p>
          <div className="row" style={{ gap: 8, marginBottom: 12 }}>
            {['7', 'M', 'X', '·', 'K', '4', 'Q'].map((c, i) => (
              <div key={i} style={{
                width: 30, height: 36, border: '1px solid var(--stroke-2)', borderRadius: 'var(--r-sm)',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontFamily: 'var(--font-mono)', fontSize: 16, fontWeight: 500,
                color: c === '·' ? 'var(--ink-4)' : 'var(--ink)',
                background: c === '·' ? 'transparent' : 'var(--paper-2)',
                borderColor: c === '·' ? 'transparent' : 'var(--stroke-2)',
              }}>{c}</div>
            ))}
          </div>
          <div className="row" style={{ gap: 8 }}>
            <button className="btn sm">Join team</button>
            <a className="muted" style={{ fontSize: 12 }}>or create a new team →</a>
          </div>
        </div>
      </div>

      {/* Footnote */}
      <div style={{ marginTop: 22, paddingTop: 16, borderTop: '1px solid var(--stroke)', maxWidth: 720 }}>
        <div className="eyebrow" style={{ marginBottom: 6 }}>What teams change</div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 14, fontSize: 11.5, color: 'var(--ink-2)', lineHeight: 1.5 }}>
          <div>
            <strong style={{ fontWeight: 500, color: 'var(--ink)' }}>Shared library.</strong> Saves and verified
            resources show up in your teammates' feeds.
          </div>
          <div>
            <strong style={{ fontWeight: 500, color: 'var(--ink)' }}>Team Asks.</strong> Post to your team only —
            for cases you can't share publicly.
          </div>
          <div>
            <strong style={{ fontWeight: 500, color: 'var(--ink)' }}>Your name, by default.</strong> Contributions show
            who you are — weight comes from accountability. Anonymous is a per-action toggle.
          </div>
        </div>
      </div>
    </div>
  );
}

window.TeamsSetup = TeamsSetup;
