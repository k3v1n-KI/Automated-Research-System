// Pathways — first-run guided tour, restructured into FOUR runnable chapters.
// The product is one flywheel of four moves:
//   ① Find   — Home hero  →  ranked Search results
//   ② Verify — Resource field trail (confirm) + flag
//   ③ Ask    — Asks replies that seed the index
//   ④ Close  — Case-resolve: record the referral, which feeds the index back
//
// Each move is its own chapter. The first run auto-plays only Find; at the end
// of every chapter the navigator chooses "Continue" or "Done for now", so the
// tour never runs four moves back-to-back unless they want it to. A launcher
// pill opens a chapter menu to (re)play any single move on demand.
//
// State lives in localStorage so chapters survive real page-to-page navigation.

(function () {
  const { useState, useEffect, useLayoutEffect, useRef, useCallback } = React;

  // ── Where are we? ─────────────────────────────────────────
  const FILE = (location.pathname.split('/').pop() || '').toLowerCase();
  const IN_HIFI = location.pathname.toLowerCase().includes('/hifi/');
  function detectPage() {
    if (FILE.includes('case-resolve')) return 'case-resolve';
    if (FILE.includes('search'))   return 'search';
    if (FILE.includes('resource')) return 'resource';
    if (FILE.includes('asks'))     return 'asks';
    return 'home'; // Pathways Hi-Fi.html (or root)
  }
  const PAGE = detectPage();

  const URLS = IN_HIFI
    ? { home: '../Pathways Hi-Fi.html', search: 'search.html', resource: 'resource.html', asks: 'asks.html', 'case-resolve': 'case-resolve.html' }
    : { home: 'Pathways Hi-Fi.html', search: 'hifi/search.html', resource: 'hifi/resource.html', asks: 'hifi/asks.html', 'case-resolve': 'hifi/case-resolve.html' };

  const SAMPLE = 'Same-day MH walk-in for an adolescent, no OHIP, Sault Ste. Marie';

  const LS_ACTIVE   = 'pw_tour_active_v1';
  const LS_STEP     = 'pw_tour_step_v1';
  const LS_DONE     = 'pw_tour_done_v1';      // user has engaged at least once
  const LS_CHAPTERS = 'pw_tour_chapters_v2';  // comma list of completed phase keys

  // ── The four moves (labels mirror the shared MOVES) ────────
  const PHASES = (typeof MOVES !== 'undefined' ? MOVES : [
    { key: 'find', label: 'Find' }, { key: 'verify', label: 'Verify' },
    { key: 'ask', label: 'Ask' }, { key: 'close', label: 'Close' },
  ]).map(m => ({ key: m.key, label: m.label }));

  // ── The script — one linear array, tagged by chapter ───────
  const STEPS = [
    // 0 · intro
    {
      page: 'home', sel: null, place: 'center', phase: null, intro: true,
      kicker: 'Welcome',
      title: 'Welcome to Pathways, Rita.',
      body: 'Finding the right service is one loop of four moves — <strong>Find · Verify · Ask · Close</strong> — and every case you close makes the next search sharper. We’ll walk one move at a time; stop whenever you like.',
      primary: 'Start with Find', skip: true,
    },

    // ── Chapter 1 · FIND (start index 1) ──
    {
      page: 'home', sel: '[data-tour="find-hero"]', place: 'auto', phase: 'find',
      kicker: 'Move 1 · Find',
      title: 'Start with a sentence.',
      body: 'Sketch the situation, pick tags, or just talk. Pathways reads it and ranks every service across Algoma by how well it fits the patient in front of you.',
      demo: { label: 'Hear it dictate', sel: '[data-tour="hero-dictate"]', busyMs: 2600, busyLabel: 'Listening…', doneLabel: 'Transcribed' },
      primary: 'See the matches', back: true,
    },
    {
      page: 'search', sel: '[data-tour="top-result"]', place: 'auto', phase: 'find',
      kicker: 'Move 1 · Find',
      title: 'Ranked by fit — and by trust.',
      body: 'The strongest match sits on top. Each result also shows how many fields fellow navigators have <strong>verified</strong>, so you can refer with confidence — not guesswork.',
      primary: 'Finish Find', back: true,
    },
    // 3 · interstitial Find → Verify (sits on the search page)
    {
      page: 'search', place: 'center', phase: 'find', interstitial: true,
      completes: 'find', next: 'verify',
      title: 'Find — done.',
      body: 'You sketched a case and got matches ranked by fit and freshness. Next move: <strong>Verify</strong> — making sure the facts you’re about to act on are still true.',
    },

    // ── Chapter 2 · VERIFY (start index 4) ──
    {
      page: 'resource', sel: '[data-tour="field-first"]', place: 'auto', phase: 'verify',
      kicker: 'Move 2 · Verify',
      title: 'Every fact carries a trail.',
      body: 'Hours, phone, eligibility — each field shows <strong>who</strong> confirmed it and <strong>when</strong>. Know it’s still true? Confirm it in one tap and it climbs in search.',
      demo: { label: 'Confirm this field', sel: '[data-tour="field-confirm"]', busyMs: 700, doneLabel: 'Confirmed by you' },
      primary: 'Next', back: true,
    },
    {
      page: 'resource', sel: '[data-tour="field-flagged"]', place: 'auto', phase: 'verify',
      kicker: 'Move 2 · Verify',
      title: 'Something changed? Flag it.',
      body: 'A stale flag warns the whole network and quietly drops the listing in search until someone re-checks it. Nothing goes out of date in silence.',
      primary: 'Finish Verify', back: true,
    },
    // 6 · interstitial Verify → Ask (sits on the resource page)
    {
      page: 'resource', place: 'center', phase: 'verify', interstitial: true,
      completes: 'verify', next: 'ask',
      title: 'Verify — done.',
      body: 'Confirms lift a listing; flags drop it until someone re-checks. Next move: <strong>Ask</strong> — what to do when nothing in the index quite fits.',
    },

    // ── Chapter 3 · ASK (start index 7) ──
    {
      page: 'asks', sel: '[data-tour="ask-replies"]', place: 'auto', phase: 'ask',
      kicker: 'Move 3 · Ask',
      title: 'Nothing fits? Ask the network.',
      body: 'Post an Ask and AOHT members reply — often attaching a service that was never indexed. Their answer seeds the directory for the next navigator who needs it.',
      primary: 'Finish Ask', back: true,
    },
    // 8 · interstitial Ask → Close (sits on the asks page)
    {
      page: 'asks', place: 'center', phase: 'ask', interstitial: true,
      completes: 'ask', next: 'close',
      title: 'Ask — done.',
      body: 'Members reply, often attaching a resource that was never indexed. Last move: <strong>Close</strong> — recording the outcome so your work feeds back into the index.',
    },

    // ── Chapter 4 · CLOSE (start index 9) ──
    {
      page: 'case-resolve', sel: '[data-tour="resolve-outcome"]', place: 'auto', phase: 'close',
      kicker: 'Move 4 · Close',
      title: 'Close the loop.',
      body: 'Record where you referred and what happened. Type a service that isn’t indexed yet and it becomes a <strong>candidate</strong> the network can verify.',
      primary: 'Next', back: true,
    },
    {
      page: 'case-resolve', sel: '[data-tour="resolve-driveby"]', place: 'auto', phase: 'close',
      kicker: 'Move 4 · Close',
      title: 'One close, three contributions.',
      body: 'Closing here re-confirms the field you used, seeds any new resource, and resolves your Ask — <strong>sharpening the next navigator’s Find.</strong>',
      primary: 'Finish', back: true,
    },
    // 11 · finale (sits on the case-resolve page)
    {
      page: 'case-resolve', place: 'center', phase: 'close', finale: true, completes: 'close',
      kicker: 'That’s the loop',
      title: 'Find → Verify → Ask → Close.',
      body: 'Everything you confirm and close makes the next search faster. Replay any single move anytime from the tour button, bottom-left.',
      primary: 'Start using Pathways',
    },
  ];

  // Chapter metadata for the launcher menu — start step + one-liner.
  const CHAPTERS = [
    { key: 'find',   start: 1,  sub: 'Sketch a case, read the ranked matches.' },
    { key: 'verify', start: 4,  sub: 'Confirm and flag fields on a resource.' },
    { key: 'ask',    start: 7,  sub: 'See how network replies seed the index.' },
    { key: 'close',  start: 9,  sub: 'Record a referral; watch the loop feed back.' },
  ];
  const labelFor = (key) => (PHASES.find(p => p.key === key) || {}).label || key;

  // ── localStorage chapter helpers ──────────────────────────
  function readChapters() {
    return (localStorage.getItem(LS_CHAPTERS) || '').split(',').filter(Boolean);
  }
  function writeChapters(arr) {
    localStorage.setItem(LS_CHAPTERS, Array.from(new Set(arr)).join(','));
  }

  // ── Small DOM helpers ─────────────────────────────────────
  function findEl(sel, cb, tries = 24) {
    const el = sel && document.querySelector(sel);
    if (el) { cb(el); return; }
    if (tries <= 0) { cb(null); return; }
    setTimeout(() => findEl(sel, cb, tries - 1), 120);
  }

  function ensureVisible(el) {
    if (!el) return;
    const r = el.getBoundingClientRect();
    const vh = window.innerHeight;
    if (r.top < 96 || r.bottom > vh - 220) {
      const target = window.scrollY + r.top - Math.max(110, (vh - r.height) / 2 - 40);
      window.scrollTo({ top: Math.max(0, target), behavior: 'smooth' });
    }
  }

  // ── Tour component ────────────────────────────────────────
  function Tour() {
    const [active, setActive] = useState(() => localStorage.getItem(LS_ACTIVE) === '1');
    const [step, setStep] = useState(() => {
      const n = parseInt(localStorage.getItem(LS_STEP) || '0', 10);
      return isNaN(n) ? 0 : n;
    });
    const [chapters, setChapters] = useState(readChapters);
    const [menuOpen, setMenuOpen] = useState(false);
    const [rect, setRect] = useState(null);
    const [pos, setPos] = useState({ left: -9999, top: -9999, ready: false });
    const [demoState, setDemoState] = useState('idle'); // idle | busy | done
    const cardRef = useRef(null);
    const targetRef = useRef(null);

    const S = STEPS[step] || STEPS[0];
    const onThisPage = active && S && S.page === PAGE;
    const mismatched = active && S && S.page !== PAGE;
    const doneCount = chapters.length;

    // First-run auto-start (Home only) — only Find chapter plays automatically.
    useEffect(() => {
      if (PAGE !== 'home') return;
      if (localStorage.getItem(LS_DONE) === '1') return;
      if (localStorage.getItem(LS_ACTIVE) === '1') return;
      if (readChapters().length) return;
      const t = setTimeout(() => start(0), 650);
      return () => clearTimeout(t);
    }, []);

    const persist = (a, s) => {
      localStorage.setItem(LS_ACTIVE, a ? '1' : '0');
      localStorage.setItem(LS_STEP, String(s));
      if (a) localStorage.setItem(LS_DONE, '1');
    };

    const start = (s = 0) => { setStep(s); setActive(true); persist(true, s); };

    const close = () => {
      setActive(false);
      localStorage.setItem(LS_ACTIVE, '0');
      localStorage.setItem(LS_DONE, '1');
    };

    const navTo = (page, s) => {
      persist(true, s);
      let url = URLS[page];
      if (page === 'search') url += '?q=' + encodeURIComponent(SAMPLE);
      window.location.href = url;
    };

    // Jump to an arbitrary step (navigating pages if needed).
    const goToStep = (idx) => {
      const next = STEPS[idx];
      if (!next) return;
      setMenuOpen(false);
      setActive(true);
      if (next.page === PAGE) { setStep(idx); persist(true, idx); }
      else navTo(next.page, idx);
    };

    const go = (delta) => {
      const ni = step + delta;
      if (ni >= STEPS.length) { finishAll(); return; }
      if (ni < 0) return;
      goToStep(ni);
    };

    const finishAll = () => {
      writeChapters([...readChapters(), 'close']);
      setChapters(readChapters());
      close();
      if (PAGE !== 'home') window.location.href = URLS.home;
    };

    // Mark a chapter complete whenever we land on its interstitial/finale.
    useEffect(() => {
      if (!active || !S || !S.completes) return;
      const cur = readChapters();
      if (!cur.includes(S.completes)) {
        const upd = [...cur, S.completes];
        writeChapters(upd);
        setChapters(upd);
      }
    }, [step, active]);

    // Reset demo state whenever the step changes
    useEffect(() => { setDemoState('idle'); }, [step, active]);

    // Acquire the target element + keep its rect in sync
    useEffect(() => {
      if (!onThisPage || !S.sel) { setRect(null); targetRef.current = null; return; }
      let alive = true;
      let raf = 0;
      findEl(S.sel, (el) => {
        if (!alive) return;
        targetRef.current = el;
        if (!el) { setRect(null); return; }
        ensureVisible(el);
        const sync = () => {
          if (!alive || !targetRef.current) return;
          const r = targetRef.current.getBoundingClientRect();
          setRect({ left: r.left, top: r.top, width: r.width, height: r.height });
          raf = requestAnimationFrame(sync);
        };
        setTimeout(sync, 360);
      });
      return () => { alive = false; cancelAnimationFrame(raf); };
    }, [step, active, onThisPage, S && S.sel]);

    // Position the card relative to the spotlight (or center it)
    useLayoutEffect(() => {
      if (!onThisPage) return;
      const card = cardRef.current;
      if (!card) return;
      const cw = card.offsetWidth, ch = card.offsetHeight;
      const vw = window.innerWidth, vh = window.innerHeight;
      const M = 16, GAP = 16;

      if (S.place === 'center' || !rect) {
        setPos({ left: Math.round((vw - cw) / 2), top: Math.round((vh - ch) / 2), ready: true });
        return;
      }
      let left, top;
      const belowTop = rect.top + rect.height + GAP;
      const aboveTop = rect.top - ch - GAP;
      if (belowTop + ch <= vh - M) {
        top = belowTop;
        left = rect.left + rect.width / 2 - cw / 2;
      } else if (aboveTop >= M) {
        top = aboveTop;
        left = rect.left + rect.width / 2 - cw / 2;
      } else if (rect.left + rect.width + GAP + cw <= vw - M) {
        left = rect.left + rect.width + GAP;
        top = rect.top + rect.height / 2 - ch / 2;
      } else if (rect.left - GAP - cw >= M) {
        left = rect.left - GAP - cw;
        top = rect.top + rect.height / 2 - ch / 2;
      } else {
        left = (vw - cw) / 2; top = vh - ch - M;
      }
      left = Math.max(M, Math.min(left, vw - cw - M));
      top = Math.max(M, Math.min(top, vh - ch - M));
      setPos({ left: Math.round(left), top: Math.round(top), ready: true });
    }, [rect, step, active, onThisPage, S && S.place]);

    // Esc closes
    useEffect(() => {
      if (!active) return;
      const onKey = (e) => { if (e.key === 'Escape') close(); };
      window.addEventListener('keydown', onKey);
      return () => window.removeEventListener('keydown', onKey);
    }, [active]);

    const runDemo = useCallback(() => {
      if (!S.demo || demoState !== 'idle') return;
      const el = document.querySelector(S.demo.sel);
      if (el) el.click();
      setDemoState('busy');
      setTimeout(() => setDemoState('done'), S.demo.busyMs || 1200);
    }, [S, demoState]);

    // ── Launcher (dormant) — pill + chapter menu ─────────────
    const Launcher = () => {
      const pillLabel = doneCount === 0 ? 'Take the tour'
        : doneCount >= PHASES.length ? 'Replay tour'
        : `Resume tour · ${doneCount}/${PHASES.length}`;
      return (
        <>
          {menuOpen && (
            <div className="pw-menu" role="menu" aria-label="Guided tour chapters">
              <div className="pw-menu-head">
                <span className="t">Walk a move</span>
                <span className="c">{doneCount}/{PHASES.length} done</span>
              </div>
              {CHAPTERS.map(ch => {
                const done = chapters.includes(ch.key);
                const i = PHASES.findIndex(p => p.key === ch.key);
                return (
                  <button key={ch.key} className={'pw-chapter' + (done ? ' done' : '')}
                          role="menuitem" onClick={() => { if (PAGE === 'home') start(ch.start); else goToStep(ch.start); }}>
                    <span className="ci">{done ? '✓' : i + 1}</span>
                    <span>
                      <span className="cl">{labelFor(ch.key)}</span>
                      <span className="cs" style={{ display: 'block' }}>{ch.sub}</span>
                    </span>
                    <span className="cgo">
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                           strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M5 12h14M13 6l6 6-6 6"/>
                      </svg>
                    </span>
                  </button>
                );
              })}
            </div>
          )}
          <button className="pw-replay" onClick={() => setMenuOpen(o => !o)} aria-expanded={menuOpen}>
            <svg className="pw-spark" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                 strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 3l1.8 5L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.7L12 3z"/>
            </svg>
            {pillLabel}
          </button>
        </>
      );
    };

    // ── Render ──────────────────────────────────────────────
    if (!active) {
      // Suppress the launcher on the very first home visit (auto-start handles it).
      if (PAGE === 'home' && localStorage.getItem(LS_DONE) !== '1' && readChapters().length === 0) return null;
      return <Launcher />;
    }

    // Active but on the wrong page (e.g. manual nav) — offer a resume nudge.
    if (mismatched) {
      return (
        <button className="pw-replay" onClick={() => navTo(S.page, step)}>
          <svg className="pw-spark" viewBox="0 0 24 24" fill="none" stroke="currentColor"
               strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
            <path d="M5 12h14M13 6l6 6-6 6"/>
          </svg>
          Resume tour
        </button>
      );
    }

    const centered = S.place === 'center' || !rect;
    const isStop = S.interstitial || S.finale;

    return (
      <div className="pw-tour-root">
        <div className={'pw-catch' + (centered ? ' dim' : '')} onClick={() => { /* swallow */ }} />

        {!centered && rect && (
          <div className="pw-spot" style={{
            left: rect.left - 6, top: rect.top - 6,
            width: rect.width + 12, height: rect.height + 12,
          }} />
        )}

        <div
          ref={cardRef}
          className={'pw-card' + (centered ? ' center' : '')}
          style={{ left: pos.left, top: pos.top, visibility: pos.ready ? 'visible' : 'hidden' }}
        >
          {/* Phase tracker — the four moves */}
          <div className="pw-phases" aria-hidden="true">
            {PHASES.map((p) => {
              const state = chapters.includes(p.key) ? 'done' : (p.key === S.phase ? 'active' : '');
              const num = PHASES.findIndex(x => x.key === p.key) + 1;
              return (
                <span key={p.key} className={'pw-phase ' + state}>
                  <span className="pw-pnum">{state === 'done' ? '✓' : num}</span>
                  <span className="pw-plabel">{p.label}</span>
                </span>
              );
            })}
          </div>

          {S.kicker && <div className="pw-kicker">{S.kicker}</div>}
          <h3 className="pw-title">{S.title}</h3>
          <p className="pw-body" dangerouslySetInnerHTML={{ __html: S.body }} />

          {S.demo && (
            <button
              className={'pw-demo' + (demoState === 'done' ? ' done' : '')}
              disabled={demoState !== 'idle'}
              onClick={runDemo}
            >
              {demoState === 'idle' && (
                <>
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                       strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M8 5v14l11-7z"/>
                  </svg>
                  {S.demo.label}
                </>
              )}
              {demoState === 'busy' && (
                <>
                  <span style={{ width: 7, height: 7, borderRadius: '50%', background: 'var(--crit)',
                                 display: 'inline-block', animation: 'pw-pulse 1.2s infinite' }} />
                  {S.demo.busyLabel || 'Working…'}
                </>
              )}
              {demoState === 'done' && (
                <>
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                       strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
                    <path d="m5 12 5 5L20 7"/>
                  </svg>
                  {S.demo.doneLabel || 'Done'}
                </>
              )}
            </button>
          )}

          <div className="pw-foot">
            <div className="pw-dots">
              {STEPS.map((_, i) => <span key={i} className={'pw-dot' + (i === step ? ' on' : '')} />)}
            </div>
            <div className="pw-actions">
              {isStop ? (
                S.finale ? (
                  <button className="pw-btn primary" onClick={finishAll}>{S.primary}</button>
                ) : (
                  <>
                    <button className="pw-btn ghost" onClick={close}>Done for now</button>
                    <button className="pw-btn primary" onClick={() => go(1)}>
                      Continue · {labelFor(S.next)}
                      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                           strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M5 12h14M13 6l6 6-6 6"/>
                      </svg>
                    </button>
                  </>
                )
              ) : (
                <>
                  {S.skip
                    ? <button className="pw-skip" onClick={close}>Skip tour</button>
                    : (S.back && <button className="pw-btn ghost" onClick={() => go(-1)}>Back</button>)}
                  <button className="pw-btn primary" onClick={() => go(1)}>
                    {S.primary}
                    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                         strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M5 12h14M13 6l6 6-6 6"/>
                    </svg>
                  </button>
                </>
              )}
            </div>
          </div>
        </div>
      </div>
    );
  }

  // ── Mount on its own root ─────────────────────────────────
  function mount() {
    if (document.getElementById('pw-tour-root')) return;
    const host = document.createElement('div');
    host.id = 'pw-tour-root';
    document.body.appendChild(host);
    ReactDOM.createRoot(host).render(<Tour />);
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', mount);
  } else {
    mount();
  }
})();
