// Pathways v2 — Onboarding
// Three moments: Sign-in (AOHT email gate) · Welcome (with team-detection) · Cold-start home (day 1).

function OnboardSignIn() {
  return (
    <div className="v2" style={{ background: 'var(--paper-2)', minHeight: '100%' }}>
      <header style={{
        height: 'var(--topnav-h)', padding: '0 28px', display: 'flex', alignItems: 'center',
        borderBottom: '1px solid var(--stroke)', background: 'var(--paper)',
      }}>
        <span className="brand">
          <span className="brand-mark" />
          Pathways
        </span>
      </header>

      <main style={{ maxWidth: 460, margin: '0 auto', padding: '88px 32px' }}>
        <div className="eyebrow" style={{ marginBottom: 12 }}>Sign in</div>
        <h1 style={{ fontSize: 30, fontWeight: 600, letterSpacing: '-0.022em', lineHeight: 1.15, marginBottom: 10 }}>
          Pathways is for AOHT members.
        </h1>
        <p style={{ fontSize: 14, color: 'var(--ink-2)', lineHeight: 1.55, marginBottom: 28 }}>
          Use your AOHT-affiliated work email. We'll send a magic link — no password to remember.
          Your contributions show your name by default; you can flip any one to anonymous.
        </p>

        <div className="card" style={{ padding: 18 }}>
          <label className="eyebrow" style={{ display: 'block', marginBottom: 8 }}>Work email</label>
          <input className="field" type="email" defaultValue="rita.okonkwo@cedarhealth.ca"
            style={{ fontSize: 14, padding: '11px 12px' }}/>
          <div className="row" style={{ gap: 8, marginTop: 6, fontSize: 11, color: 'var(--ink-3)' }}>
            <V2Ico name="check" size={11} stroke={2.2} style={{ color: 'var(--good)' }}/>
            <span><strong style={{ color: 'var(--good)', fontWeight: 500 }}>cedarhealth.ca</strong> is an AOHT-recognized domain</span>
          </div>

          <button className="btn accent" style={{ width: '100%', marginTop: 16, padding: '11px 16px' }}>
            Send magic link →
          </button>

          <div style={{
            marginTop: 18, paddingTop: 14, borderTop: '1px solid var(--stroke)',
            display: 'flex', flexDirection: 'column', gap: 8,
            fontSize: 12, color: 'var(--ink-3)', lineHeight: 1.5,
          }}>
            <div className="row" style={{ gap: 8 }}>
              <V2Ico name="people" size={12} style={{ color: 'var(--ink-4)', flexShrink: 0 }}/>
              <span>Email not from an AOHT org? <a style={{ color: 'var(--accent-ink)', fontWeight: 500 }}>Request access →</a></span>
            </div>
            <div className="row" style={{ gap: 8 }}>
              <V2Ico name="corner-arrow" size={12} style={{ color: 'var(--ink-4)', flexShrink: 0 }}/>
              <span>Joining a team? Paste a team code on the next screen.</span>
            </div>
          </div>
        </div>

        <p style={{ fontSize: 11, color: 'var(--ink-4)', marginTop: 18, lineHeight: 1.5 }}>
          Pathways stores resource info, not patient data.
          AOHT members reviewing this resource see your name (or "AOHT member" if you opt anonymous);
          the public never does.
        </p>
      </main>
    </div>
  );
}

function OnboardWelcome() {
  return (
    <div className="v2">
      <V2Topnav active="cases" />
      <main style={{ maxWidth: 880, margin: '0 auto', padding: '60px 32px 80px' }}>

        <div style={{ marginBottom: 32 }}>
          <div className="eyebrow" style={{ marginBottom: 12 }}>Welcome to Pathways</div>
          <h1 style={{ fontSize: 36, fontWeight: 600, letterSpacing: '-0.025em', lineHeight: 1.1, marginBottom: 12 }}>
            You're in, Rita.
          </h1>
          <p style={{ fontSize: 16, color: 'var(--ink-2)', lineHeight: 1.5, maxWidth: 620 }}>
            Here's how Pathways works, in one paragraph. <strong>Search</strong> for services in plain language.
            If nothing fits, post it as an <strong>Ask</strong> — other AOHT members in your region
            see it for 7 days. Every resource shows what's verified vs. what's just AI-extracted, and
            confirming or flagging takes one tap. Your name is on what you contribute, by default.
          </p>
        </div>

        {/* Team detection — inline at first-run */}
        <div className="card" style={{
          padding: 22, marginBottom: 18,
          background: 'var(--accent-tint)',
          borderColor: 'color-mix(in srgb, var(--accent) 28%, transparent)',
        }}>
          <div className="row" style={{ gap: 7, marginBottom: 8 }}>
            <V2Ico name="people" size={13} stroke={1.8} style={{ color: 'var(--accent)' }}/>
            <span className="eyebrow" style={{ color: 'var(--accent-ink)' }}>Detected · cedarhealth.ca</span>
          </div>
          <h2 style={{ fontSize: 18, fontWeight: 600, letterSpacing: '-0.014em', marginBottom: 6 }}>
            3 colleagues from Cedar Health are already here.
          </h2>
          <p style={{ fontSize: 13, color: 'var(--ink-2)', lineHeight: 1.5, marginBottom: 14, maxWidth: 560 }}>
            Want to form a team? You'll share a saved-resource library and see what they refer to.
            Skip this if you'd rather work solo — you can join later from settings.
          </p>
          <div className="row" style={{ gap: 10, marginBottom: 0 }}>
            <div className="row" style={{ gap: 6 }}>
              {['MR', 'AS', 'KL'].map(i => (
                <span key={i} className="avatar" style={{ background: 'var(--paper)', borderColor: 'var(--stroke)' }}>{i}</span>
              ))}
              <span className="mono" style={{ fontSize: 11, color: 'var(--ink-3)' }}>+ you</span>
            </div>
            <span style={{ flex: 1 }}/>
            <button className="btn ghost sm">Skip for now</button>
            <button className="btn sm">Have a team code →</button>
            <button className="btn accent sm">Form team</button>
          </div>
        </div>

        {/* First action prompt */}
        <div style={{
          background: 'var(--ink)', color: 'var(--paper)',
          borderRadius: 'var(--r-lg)', padding: 28, marginBottom: 14,
          boxShadow: 'var(--shadow-sm)',
        }}>
          <div className="row" style={{ gap: 8, marginBottom: 10, opacity: 0.55 }}>
            <V2Ico name="search" size={13} stroke={1.8}/>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10.5, textTransform: 'uppercase', letterSpacing: '0.08em' }}>
              Try your first search
            </span>
          </div>
          <h2 style={{ fontSize: 22, fontWeight: 500, letterSpacing: '-0.014em', lineHeight: 1.3, marginBottom: 14, color: 'rgb(255 255 255 / 0.95)' }}>
            Describe a case you'd refer today — even one you already know the answer to.
          </h2>
          <div style={{
            fontSize: 14, color: 'rgb(255 255 255 / 0.5)', fontStyle: 'italic',
            paddingBottom: 14, borderBottom: '1px solid rgb(255 255 255 / 0.12)',
          }}>
            e.g. "Same-day MH walk-in for an adult without OHIP, East Toronto"
          </div>
          <div className="row" style={{ marginTop: 12, justifyContent: 'space-between' }}>
            <span style={{ fontSize: 11.5, opacity: 0.55, fontFamily: 'var(--font-mono)' }}>
              We seed with AI. Your verifications make it real.
            </span>
            <span style={{ display: 'inline-flex', alignItems: 'center', gap: 8, padding: '7px 14px',
              background: 'var(--accent)', color: 'white', borderRadius: 'var(--r-sm)',
              fontWeight: 500, fontSize: 13 }}>
              Try a search <V2Ico name="arrow-right" size={13}/>
            </span>
          </div>
        </div>

        {/* Secondary nudges */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
          <a className="card tap" style={{ padding: 16 }}>
            <div className="row" style={{ gap: 7, marginBottom: 7 }}>
              <V2Ico name="corner-arrow" size={12} stroke={1.8} style={{ color: 'var(--ink-3)' }}/>
              <span className="eyebrow">Or browse open Asks</span>
            </div>
            <div style={{ fontSize: 13, color: 'var(--ink-2)', lineHeight: 1.5 }}>
              See what colleagues are looking for in East Toronto right now. Answering one is the
              fastest way to help.
            </div>
          </a>
          <a className="card tap" style={{ padding: 16 }}>
            <div className="row" style={{ gap: 7, marginBottom: 7 }}>
              <V2Ico name="plus" size={12} stroke={2} style={{ color: 'var(--ink-3)' }}/>
              <span className="eyebrow">Got a tip already?</span>
            </div>
            <div style={{ fontSize: 13, color: 'var(--ink-2)', lineHeight: 1.5 }}>
              Paste a link, dictate a voice note, or share from another app. We'll structure it.
            </div>
          </a>
        </div>
      </main>
    </div>
  );
}

function OnboardColdHome() {
  return (
    <div className="v2">
      <V2Topnav active="cases" />
      <div className="body">
        <main style={{ maxWidth: 1080, margin: '0 auto', padding: '44px 32px 80px' }}>

          {/* Greeting */}
          <div style={{ marginBottom: 28 }}>
            <div className="eyebrow" style={{ marginBottom: 10 }}>Wednesday · May 7 · day 1</div>
            <div style={{ display: 'flex', alignItems: 'baseline', gap: 12, flexWrap: 'wrap' }}>
              <h1 className="title-hero" style={{ fontSize: 32 }}>Good morning, Rita.</h1>
              <span style={{ fontSize: 18, color: 'var(--ink-3)' }}>
                Nothing saved yet.
              </span>
            </div>
          </div>

          {/* Search — same dark card, calmer placeholder */}
          <div style={{
            background: 'var(--ink)', color: 'var(--paper)',
            borderRadius: 'var(--r-lg)', padding: 24, marginBottom: 16,
            boxShadow: 'var(--shadow-sm)',
          }}>
            <div className="row" style={{ gap: 8, marginBottom: 12, opacity: 0.55 }}>
              <V2Ico name="search" size={13} stroke={1.8} />
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10.5, textTransform: 'uppercase', letterSpacing: '0.08em' }}>
                Find a service · or post an Ask
              </span>
            </div>
            <div style={{
              fontSize: 22, fontWeight: 500, letterSpacing: '-0.018em',
              color: 'rgb(255 255 255 / 0.4)', lineHeight: 1.3,
              minHeight: 60, paddingBottom: 14, fontStyle: 'italic',
              borderBottom: '1px solid rgb(255 255 255 / 0.12)',
            }}>
              Describe a case in plain language…
            </div>
            <div className="row" style={{ marginTop: 14, gap: 14, justifyContent: 'space-between' }}>
              <span style={{ fontSize: 11.5, opacity: 0.55, fontFamily: 'var(--font-mono)' }}>
                4,200 resources in your region · 1,180 verified by AOHT members in the last 90 days
              </span>
              <span style={{ display: 'inline-flex', alignItems: 'center', gap: 8, padding: '7px 14px',
                background: 'var(--accent)', color: 'white', borderRadius: 'var(--r-sm)',
                fontWeight: 500, fontSize: 13 }}>
                Search <V2Ico name="arrow-right" size={13}/>
              </span>
            </div>
          </div>

          {/* Two-up: Open Asks (read-only nudge) · Verify nearby (seeded by AI) */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 32 }}>

            <div className="card" style={{ padding: 18, display: 'flex', flexDirection: 'column', gap: 12 }}>
              <div className="between">
                <div className="row" style={{ gap: 8 }}>
                  <V2Ico name="corner-arrow" size={13} stroke={1.7} style={{ color: 'var(--accent)' }}/>
                  <span className="eyebrow" style={{ color: 'var(--accent-ink)' }}>Open Asks · East Toronto</span>
                </div>
                <a className="muted" style={{ fontSize: 11.5 }}>All →</a>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                {ASKS.filter(a => !a.yours).slice(0, 3).map((a, i, arr) => (
                  <div key={a.id} style={{
                    paddingBottom: i < arr.length - 1 ? 10 : 0,
                    borderBottom: i < arr.length - 1 ? '1px solid var(--stroke)' : 0,
                  }}>
                    <div style={{ fontSize: 13, color: 'var(--ink)', lineHeight: 1.45, marginBottom: 5 }}>
                      {a.text}
                    </div>
                    <div className="row" style={{ gap: 7, fontSize: 10.5, color: 'var(--ink-3)' }}>
                      <span style={{ fontWeight: 500 }}>{a.who}</span><span>·</span>
                      <span>{a.region}</span><span>·</span>
                      <span className="mono">{a.replies} replies</span>
                    </div>
                  </div>
                ))}
              </div>
              <div style={{ paddingTop: 4, marginTop: 4 }}>
                <span style={{ fontSize: 11.5, color: 'var(--ink-3)', lineHeight: 1.5 }}>
                  Answering one is the fastest way to help a colleague — and to seed your saved library.
                </span>
              </div>
            </div>

            <div className="card" style={{ padding: 18, display: 'flex', flexDirection: 'column', gap: 12 }}>
              <div className="between">
                <div className="row" style={{ gap: 8 }}>
                  <V2Ico name="check" size={13} stroke={2} style={{ color: 'var(--good)' }}/>
                  <span className="eyebrow">Help verify · East York</span>
                </div>
                <a className="muted" style={{ fontSize: 11.5 }}>Show 5 →</a>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                {[
                  { name: 'Across Health · MH Walk-in', field: 'Referral path', flag: '2× flagged stale', tone: 'crit' },
                  { name: 'Eastside Community MH', field: 'Hours', flag: 'AI-extracted, never confirmed', tone: 'ai' },
                  { name: 'Family Services East', field: 'Phone', flag: '210d since confirmed', tone: 'warn' },
                ].map((r, i, arr) => (
                  <div key={i} style={{ paddingBottom: i < arr.length - 1 ? 10 : 0, borderBottom: i < arr.length - 1 ? '1px solid var(--stroke)' : 0 }}>
                    <div style={{ fontSize: 13, color: 'var(--ink)', marginBottom: 3, fontWeight: 500 }}>{r.name}</div>
                    <div className="row" style={{ gap: 8, fontSize: 11, color: 'var(--ink-3)' }}>
                      <span>{r.field}</span><span>·</span>
                      <span style={{ color: r.tone === 'crit' ? '#b91c1c' : r.tone === 'warn' ? '#a16207' : 'var(--accent-ink)', fontWeight: 500 }}>
                        {r.flag}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
              <div style={{ paddingTop: 4, marginTop: 4 }}>
                <span style={{ fontSize: 11.5, color: 'var(--ink-3)', lineHeight: 1.5 }}>
                  These were AI-seeded. A 10-second call to confirm makes a real difference for the next member.
                </span>
              </div>
            </div>
          </div>

          {/* Recent cases — empty state */}
          <section>
            <div className="between" style={{ marginBottom: 14 }}>
              <h3 className="title-2">Recent cases</h3>
              <span className="muted" style={{ fontSize: 12.5 }}>none yet</span>
            </div>
            <div className="card" style={{
              padding: '28px 24px', display: 'flex', justifyContent: 'space-between',
              alignItems: 'center', background: 'var(--paper-2)',
            }}>
              <div>
                <div style={{ fontSize: 14, color: 'var(--ink-2)', marginBottom: 4 }}>Your case list shows up here once you save your first search.</div>
                <div style={{ fontSize: 12, color: 'var(--ink-3)' }}>Or attach a case to a referral note you paste into search.</div>
              </div>
              <button className="btn sm">Start a case →</button>
            </div>
          </section>
        </main>
      </div>
    </div>
  );
}

Object.assign(window, { OnboardSignIn, OnboardWelcome, OnboardColdHome });
