// Hi-fi home — workbench-style for Rita LaFlamme, Algoma OHT
// Adds tag-based and voice (dictation) input alongside the default free-text sketch.

const SAMPLE_DICTATION = 'Same-day MH walk-in for an adolescent, no OHIP, Sault Ste. Marie';

// Tag composer — grouped chip suggestions for the "Tags" mode.
const TAG_GROUPS = [
  { key: 'need',  label: 'Need',
    options: ['MH walk-in', 'Same-day', 'Primary care', 'Home support', 'Meal program', 'Housing', 'Harm reduction', 'Addictions', 'Dental'] },
  { key: 'pop',   label: 'Population',
    options: ['Adolescent', 'Youth', 'Adult', 'Senior', 'Indigenous', 'Newcomer', 'Francophone'] },
  { key: 'cov',   label: 'Coverage',
    options: ['OHIP', 'No OHIP', 'IFHP', 'ODSP', 'No private benefits'] },
  { key: 'loc',   label: 'Location',
    options: ['Sault Ste. Marie', 'Garden River', 'Wawa', 'Elliot Lake', 'Blind River'] },
];

// Lightweight "AI extraction" preview for sketch mode.
function extractFromSketch(text) {
  const q = (text || '').toLowerCase();
  const out = [];
  if (/(mh|mental.?health|self.?harm|depress|anx)/.test(q)) out.push('MH');
  if (/walk.?in|drop.?in|same.?day/.test(q))                out.push('walk-in');
  if (/adolesc|teen|youth|child/.test(q))                   out.push('youth');
  if (/senior|elder|\b7\d|\b8\d|\b9\d/.test(q))             out.push('senior');
  if (/no ohip|uninsured|without ohip/.test(q))             out.push('no OHIP');
  if (/odsp/.test(q))                                       out.push('ODSP');
  if (/post.?discharge|discharged/.test(q))                 out.push('post-discharge');
  if (/cantonese|mandarin|french|francoph/.test(q))         out.push('language');
  const loc = q.match(/garden river|sault ste\. marie|sault|wawa|elliot lake|blind river/);
  if (loc) {
    const map = {
      'garden river': 'Garden River', 'sault ste. marie': 'Sault Ste. Marie',
      'sault': 'Sault Ste. Marie', 'wawa': 'Wawa',
      'elliot lake': 'Elliot Lake', 'blind river': 'Blind River',
    };
    out.push(map[loc[0]]);
  }
  return out;
}

function Home() {
  const [showCmdK, setShowCmdK] = useState(false);
  const [query, setQuery] = useState('');

  // Inline tag-chip popover anchored to the Tags button — augments the same
  // query the user is already typing in, rather than yanking them into a
  // second text input inside a modal.
  const [tagOpen, setTagOpen] = useState(false);
  const tagAnchorRef = useRef(null);

  useEffect(() => {
    if (!tagOpen) return;
    const onDown = (e) => {
      if (tagAnchorRef.current && !tagAnchorRef.current.contains(e.target)) {
        setTagOpen(false);
      }
    };
    const onKey = (e) => { if (e.key === 'Escape') setTagOpen(false); };
    document.addEventListener('mousedown', onDown);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onDown);
      document.removeEventListener('keydown', onKey);
    };
  }, [tagOpen]);

  const escapeRe = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const tagInQuery = (label) => {
    if (!query) return false;
    const re = new RegExp('(^|,\\s*|\\s)' + escapeRe(label) + '(?=$|,|\\s)', 'i');
    return re.test(query);
  };
  const toggleTagInQuery = (label) => {
    setQuery(prev => {
      const p = prev || '';
      const re = new RegExp('(^|,\\s*|\\s+)' + escapeRe(label) + '(?=$|,|\\s)', 'i');
      if (re.test(p)) {
        return p.replace(re, (m, lead) => lead.startsWith(',') ? '' : lead)
                .replace(/\s*,\s*,/g, ', ')
                .replace(/^[,\s]+|[,\s]+$/g, '')
                .replace(/\s+/g, ' ');
      }
      const trimmed = p.replace(/\s+$/, '');
      const sep = !trimmed ? '' : (/[,]$/.test(trimmed) ? ' ' : ', ');
      return trimmed + sep + label;
    });
  };
  const activeTagCount = TAG_GROUPS.reduce(
    (n, g) => n + g.options.filter(tagInQuery).length, 0
  );
  const clearAllTags = () => {
    setQuery(prev => {
      let p = prev || '';
      for (const g of TAG_GROUPS) for (const o of g.options) {
        const re = new RegExp('(^|,\\s*|\\s+)' + escapeRe(o) + '(?=$|,|\\s)', 'gi');
        p = p.replace(re, (m, lead) => lead.startsWith(',') ? '' : lead);
      }
      return p.replace(/\s*,\s*,/g, ', ')
              .replace(/^[,\s]+|[,\s]+$/g, '')
              .replace(/\s+/g, ' ');
    });
  };

  // Inline voice dictation on the dark hero card.
  const [heroRec, setHeroRec] = useState(false);
  useEffect(() => {
    if (!heroRec) return;
    let i = query.length;
    const start = query;
    const id = setInterval(() => {
      i += 2;
      if (i > SAMPLE_DICTATION.length) { setHeroRec(false); clearInterval(id); return; }
      setQuery(start + SAMPLE_DICTATION.slice(start.length, i));
    }, 55);
    return () => clearInterval(id);
  }, [heroRec]);

  // ⌘K / Ctrl-K / N to open the new-case sheet
  useEffect(() => {
    const onKey = (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault(); setShowCmdK(true);
      }
      if (e.key === 'Escape') setShowCmdK(false);
      if (e.key.toLowerCase() === 'n' && !e.metaKey && !e.ctrlKey
          && !['INPUT','TEXTAREA'].includes(document.activeElement?.tagName)) {
        setShowCmdK(true);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  const submitToSearch = (q) => {
    const final = q || 'Same-day MH walk-in for an adolescent, no OHIP, Sault Ste. Marie';
    window.location.href = `hifi/search.html?q=${encodeURIComponent(final)}`;
  };

  return (
    <div className="v2">
      <TopnavRoot active="cases" />
      <div className="body page-in">
        <main style={{ maxWidth: 1080, margin: '0 auto', padding: '52px 32px 80px' }}>

          {/* Greeting */}
          <div style={{ marginBottom: 36 }}>
            <div className="eyebrow" style={{ marginBottom: 10 }}>Wednesday · May 13 · Sault Ste. Marie</div>
            <div style={{ display: 'flex', alignItems: 'baseline', gap: 12, flexWrap: 'wrap' }}>
              <h1 className="title-hero" style={{ fontSize: 32 }}>Good morning, Rita.</h1>
              <span style={{ fontSize: 18, color: 'var(--ink-3)', fontWeight: 400 }}>
                three saved cases waiting.
              </span>
            </div>
          </div>

          {/* Two-up: action + region update */}
          <div style={{ display: 'grid', gridTemplateColumns: '1.5fr 1fr', gap: 16, marginBottom: 56 }}>

            {/* Action: open search */}
            <form data-tour="find-hero" onSubmit={e => { e.preventDefault(); submitToSearch(query); }} style={{
              background: 'var(--ink)', color: 'var(--paper)',
              borderRadius: 'var(--r-lg)',
              padding: 28,
              display: 'flex', flexDirection: 'column', justifyContent: 'space-between',
              minHeight: 290,
              boxShadow: 'var(--shadow-sm)', border: 'none',
            }}>
              <div>
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10.5, textTransform: 'uppercase', letterSpacing: '0.08em', opacity: 0.55, marginBottom: 14 }}>
                  Start here
                </div>
                <h2 style={{ fontSize: 30, lineHeight: 1.1, letterSpacing: '-0.025em', fontWeight: 600, maxWidth: 460, marginBottom: 12, color: 'var(--paper)' }}>
                  Find the right service for a patient.
                </h2>
                <p style={{ fontSize: 14.5, opacity: 0.75, maxWidth: 460, lineHeight: 1.5 }}>
                  Sketch it, dictate it, or pick tags. We rank what fits across Algoma.
                </p>
              </div>

              <div style={{ marginTop: 24 }}>
                <div style={{
                  background: 'rgb(255 255 255 / 0.08)',
                  border: '1px solid ' + (heroRec ? 'rgb(252 165 165 / 0.4)' : 'rgb(255 255 255 / 0.14)'),
                  borderRadius: 'var(--r-md)',
                  transition: 'border-color 0.15s',
                }}>
                  {/* Multi-line input area */}
                  <div style={{
                    display: 'flex', gap: 10,
                    padding: '11px 14px 6px',
                    alignItems: 'flex-start',
                    minWidth: 0,
                  }}>
                    <Ico name="search" size={14} style={{ color: 'rgb(255 255 255 / 0.5)', marginTop: 4, flexShrink: 0 }}/>

                    {heroRec ? (
                      <div style={{
                        flex: 1, minWidth: 0,
                        fontSize: 14, lineHeight: 1.5, color: 'var(--paper)',
                        minHeight: 60, maxHeight: 84, overflowY: 'auto',
                        wordBreak: 'break-word', whiteSpace: 'pre-wrap',
                      }}>
                        {query || <span style={{ color: 'rgb(255 255 255 / 0.5)', fontStyle: 'italic' }}>Listening… speak naturally.</span>}
                        <span style={{
                          display: 'inline-block', width: 1, height: '1em',
                          background: 'currentColor', verticalAlign: '-2px',
                          marginLeft: 2, opacity: 0.7,
                          animation: 'caret-blink 1s steps(1) infinite',
                        }}/>
                      </div>
                    ) : (
                      <textarea
                        value={query}
                        onChange={e => setQuery(e.target.value)}
                        onKeyDown={e => {
                          if (e.key === 'Enter' && !e.shiftKey) {
                            e.preventDefault();
                            submitToSearch(query);
                          }
                        }}
                        rows={3}
                        placeholder={'e.g. same-day MH walk-in, adolescent,\nno OHIP, Sault Ste. Marie…'}
                        style={{
                          flex: 1, minWidth: 0,
                          background: 'transparent', border: 'none', outline: 'none',
                          color: 'var(--paper)', fontFamily: 'inherit', fontSize: 14,
                          lineHeight: 1.5, resize: 'none', padding: 0,
                        }}
                      />
                    )}
                  </div>

                  {/* Toolbar */}
                  <div style={{
                    display: 'flex', alignItems: 'center', gap: 6,
                    padding: '8px 10px 9px',
                    borderTop: '1px solid rgb(255 255 255 / 0.08)',
                  }}>
                    <button
                      type="button"
                      data-tour="hero-dictate"
                      className={'mic-ghost' + (heroRec ? ' on' : '')}
                      onClick={() => setHeroRec(r => !r)}
                      title={heroRec ? 'Stop dictation' : 'Dictate'}
                      aria-label="Dictate"
                      style={{ width: 'auto', padding: '4px 9px', gap: 6, fontSize: 12 }}
                    >
                      {heroRec ? (
                        <>
                          <span className="rec-dot"/>
                          <span className="wave" style={{ color: '#fca5a5' }}>
                            <span/><span/><span/><span/><span/>
                          </span>
                          <span style={{ color: '#fecaca' }}>Listening…</span>
                        </>
                      ) : (
                        <><Ico name="mic" size={13}/> <span>Dictate</span></>
                      )}
                    </button>

                    <span ref={tagAnchorRef} className="tag-pop-anchor">
                      <button type="button"
                        onClick={() => setTagOpen(o => !o)}
                        className={'mic-ghost' + (tagOpen ? ' on' : '')}
                        title="Pick tags"
                        aria-expanded={tagOpen}
                        style={{ width: 'auto', padding: '4px 9px', gap: 6, fontSize: 12 }}
                      >
                        <Ico name="tag" size={13}/> <span>Tags</span>
                        {activeTagCount > 0 && (
                          <span style={{
                            display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
                            minWidth: 16, height: 16, padding: '0 5px',
                            borderRadius: 999, fontFamily: 'var(--font-mono)',
                            fontSize: 10, fontWeight: 600,
                            background: 'var(--accent)', color: 'white',
                          }}>{activeTagCount}</span>
                        )}
                      </button>

                      {tagOpen && (
                        <div className="tag-popover" role="dialog" aria-label="Pick tags">
                          <div className="tag-popover-head">
                            <span className="eyebrow">Pick tags</span>
                            <span className="count">{activeTagCount} selected</span>
                          </div>
                          <div className="tag-popover-body">
                            {TAG_GROUPS.map(g => (
                              <div className="tag-group" key={g.key}>
                                <div className="tag-group-label">
                                  <span className="eyebrow">{g.label}</span>
                                </div>
                                <div className="row" style={{ gap: 5, flexWrap: 'wrap' }}>
                                  {g.options.map(o => (
                                    <button type="button" key={o}
                                      className={'tag-pill' + (tagInQuery(o) ? ' on' : '')}
                                      onClick={() => toggleTagInQuery(o)}>
                                      {o}
                                    </button>
                                  ))}
                                </div>
                              </div>
                            ))}
                          </div>
                          <div className="tag-popover-foot">
                            <span>Tags are added to your search.</span>
                            <span className="row" style={{ gap: 10 }}>
                              {activeTagCount > 0 && (
                                <button type="button" className="clear" onClick={clearAllTags}>
                                  Clear
                                </button>
                              )}
                              <button type="button" className="clear" onClick={() => setTagOpen(false)}>
                                Done
                              </button>
                            </span>
                          </div>
                        </div>
                      )}
                    </span>

                    <span style={{ flex: 1 }}/>

                    <span className="row" style={{ gap: 6, fontSize: 11, color: 'rgb(255 255 255 / 0.4)', fontFamily: 'var(--font-mono)' }}>
                      <span className="kbd" style={{ background: 'rgb(255 255 255 / 0.08)', borderColor: 'rgb(255 255 255 / 0.16)', color: 'rgb(255 255 255 / 0.6)' }}>↵</span>
                      <span>to search</span>
                    </span>

                    <button type="submit" style={{
                      all: 'unset', display: 'inline-flex', alignItems: 'center', gap: 7,
                      padding: '6px 12px', background: 'var(--accent)', color: 'white',
                      borderRadius: 'var(--r-sm)', fontWeight: 500, fontSize: 13, cursor: 'pointer',
                    }}>
                      Search <Ico name="arrow-right" size={13} />
                    </button>
                  </div>
                </div>

                <div className="row" style={{ gap: 8, marginTop: 12, fontSize: 12, opacity: 0.55, fontFamily: 'var(--font-mono)' }}>
                  <span className="kbd" style={{ background: 'rgb(255 255 255 / 0.08)', borderColor: 'rgb(255 255 255 / 0.16)', color: 'rgb(255 255 255 / 0.7)' }}>N</span>
                  <span>new case</span>
                  <span style={{ opacity: 0.4 }}>·</span>
                  <span className="kbd" style={{ background: 'rgb(255 255 255 / 0.08)', borderColor: 'rgb(255 255 255 / 0.16)', color: 'rgb(255 255 255 / 0.7)' }}>⌘K</span>
                  <span>expanded composer</span>
                </div>
              </div>
            </form>

            {/* Region update */}
            <a href="hifi/resource.html" className="card" style={{
              display: 'flex', flexDirection: 'column', justifyContent: 'space-between',
              padding: 22, minHeight: 290, textDecoration: 'none', color: 'inherit',
            }}>
              <div>
                <div className="row" style={{ gap: 8, marginBottom: 12 }}>
                  <Ico name="sparkle" size={13} stroke={1.7} style={{ color: 'var(--accent)' }}/>
                  <span className="eyebrow" style={{ color: 'var(--accent-ink)' }}>Region update · 1h ago</span>
                </div>
                <p style={{ fontSize: 14, lineHeight: 1.55 }}>
                  <strong>Algoma Family Services</strong> may now accept phone-ahead bookings from primary care.
                </p>
                <p className="ital" style={{ fontSize: 13.5, color: 'var(--ink-3)', marginTop: 6 }}>
                  Two of your saved cases reference this resource. Verify before referring.
                </p>
              </div>
              <div className="row" style={{ gap: 8 }}>
                <span className="btn sm">Review impact</span>
                <span className="btn ghost sm">Dismiss</span>
              </div>
            </a>
          </div>

          {/* Recent cases */}
          <section>
            <div className="between" style={{ marginBottom: 16 }}>
              <h3 className="title-2">Recent cases</h3>
              <div className="row" style={{ gap: 12 }}>
                <span className="muted" style={{ fontSize: 12.5 }}>{CASES.length} this week</span>
                <a className="ilink" style={{ fontSize: 12.5 }}>View all →</a>
              </div>
            </div>
            <div className="card" style={{ padding: 0 }}>
              {CASES.map((c, i) => (
                <a key={c.id}
                   href={c.id === 'c-240' ? 'hifi/search.html?q=' + encodeURIComponent('Same-day MH walk-in for an adolescent, no OHIP, Sault Ste. Marie') : '#'}
                   className="row" style={{
                  padding: '14px 18px',
                  borderBottom: i < CASES.length - 1 ? '1px solid var(--stroke)' : 0,
                  display: 'grid',
                  gridTemplateColumns: '1.4fr 1fr auto auto auto',
                  gap: 16,
                  fontSize: 13.5, color: 'inherit', textDecoration: 'none',
                }}>
                  <div className="stack-2">
                    <div style={{ fontWeight: 500 }}>{c.title}</div>
                    <div className="muted" style={{ fontSize: 12 }}>{c.summary}</div>
                  </div>
                  <div className="row" style={{ gap: 4, flexWrap: 'wrap' }}>
                    {c.tags.map(t => <span key={t} className="chip sm">{t}</span>)}
                  </div>
                  <span className="muted" style={{ fontSize: 12, minWidth: 130 }}>{c.region}</span>
                  <span className="row" style={{ gap: 5, fontSize: 12 }}>
                    <span className={`dot ${c.status === 'referred' ? 'good' : c.status === 'closed' ? 'muted' : 'accent'}`} />
                    <span className="muted" style={{ minWidth: 56 }}>{c.status}</span>
                  </span>
                  <span className="mono dim" style={{ fontSize: 11, minWidth: 64, textAlign: 'right' }}>{c.when}</span>
                </a>
              ))}
            </div>
          </section>

          {/* The Pathways loop — demo spine */}
          <section style={{ marginTop: 48 }}>
            <div className="between" style={{ marginBottom: 6 }}>
              <h3 className="title-2">How Pathways works · four moves</h3>
              <span className="muted mono" style={{ fontSize: 11.5 }}>jump in anywhere ↓</span>
            </div>
            <p className="muted" style={{ fontSize: 13, marginBottom: 20, maxWidth: 680 }}>
              One loop keeps the directory alive: find a fit, verify what you use, ask the network when nothing fits,
              and close the case so your work seeds the next search.
            </p>
            <PhaseLoop root />
          </section>
        </main>
      </div>

      {showCmdK && (
        <NewCaseSheet
          initial={query}
          onClose={() => setShowCmdK(false)}
          onSubmit={(q) => { setShowCmdK(false); submitToSearch(q); }}
        />
      )}
    </div>
  );
}

// ── New-case sheet — three input modalities: Sketch (free text), Tags, Voice ──
function NewCaseSheet({ initial, onClose, onSubmit }) {
  const [mode, setMode] = useState('sketch');       // 'sketch' | 'tags'
  const [text, setText] = useState(initial || '');
  const [picked, setPicked] = useState([]);          // selected tag labels
  const [custom, setCustom] = useState('');
  const [rec, setRec] = useState(false);
  const dictScrollRef = useRef(null);

  // Keep dictation pane scrolled to the latest word
  useEffect(() => {
    if (rec && dictScrollRef.current) {
      dictScrollRef.current.scrollTop = dictScrollRef.current.scrollHeight;
    }
  }, [text, rec]);

  // Simulated dictation streams into whichever mode is active.
  useEffect(() => {
    if (!rec) return;
    let i = 0;
    const words = SAMPLE_DICTATION.split(/(\s+)/);
    if (mode === 'sketch') {
      const start = text;
      const id = setInterval(() => {
        i += 2;
        if (i > SAMPLE_DICTATION.length) { setRec(false); clearInterval(id); return; }
        setText(start + (start && !start.endsWith(' ') ? ' ' : '') + SAMPLE_DICTATION.slice(0, i));
      }, 55);
      return () => clearInterval(id);
    } else {
      // tags mode — drop in tag picks one-by-one
      const sequence = ['Same-day', 'MH walk-in', 'Adolescent', 'No OHIP', 'Sault Ste. Marie'];
      let k = 0;
      const id = setInterval(() => {
        if (k >= sequence.length) { setRec(false); clearInterval(id); return; }
        const t = sequence[k++];
        setPicked(p => p.includes(t) ? p : [...p, t]);
      }, 380);
      return () => clearInterval(id);
    }
  }, [rec, mode]);

  const togglePick = (t) => setPicked(p => p.includes(t) ? p.filter(x => x !== t) : [...p, t]);

  const tagQuery = useMemo(
    () => [...picked, custom.trim()].filter(Boolean).join(', '),
    [picked, custom]
  );
  const extracted = useMemo(() => extractFromSketch(text), [text]);
  const canSubmit = mode === 'sketch' ? text.trim().length > 0 : tagQuery.length > 0;

  const handleSubmit = (e) => {
    e?.preventDefault?.();
    if (!canSubmit) return;
    onSubmit(mode === 'sketch' ? text : tagQuery);
  };

  return (
    <div className="sheet-overlay" onClick={onClose}>
      <div className="sheet" onClick={e => e.stopPropagation()} style={{ width: 'min(620px, 94vw)' }}>
        {/* Header */}
        <div className="between" style={{ marginBottom: 14, alignItems: 'flex-start' }}>
          <div>
            <div className="eyebrow" style={{ marginBottom: 5 }}>New case</div>
            <h3 className="title-1">Describe the case</h3>
          </div>
          <div className="seg" role="tablist" aria-label="Input mode">
            <button type="button" role="tab"
              className={mode === 'sketch' ? 'on' : ''} onClick={() => setMode('sketch')}>
              <Ico name="edit" size={11}/> Sketch
            </button>
            <button type="button" role="tab"
              className={mode === 'tags' ? 'on' : ''} onClick={() => setMode('tags')}>
              <Ico name="tag" size={11}/> Tags
            </button>
          </div>
        </div>

        <form onSubmit={handleSubmit}>
          {/* Body */}
          {mode === 'sketch' ? (
            <div>
              {rec ? (
                <div className="dictation" style={{
                  height: 108, alignItems: 'flex-start',
                  padding: '12px 14px', gap: 12, overflow: 'hidden',
                }}>
                  <div className="row" style={{ gap: 8, flexShrink: 0, paddingTop: 2 }}>
                    <span className="rec-dot" />
                    <span className="wave"><span/><span/><span/><span/><span/></span>
                  </div>
                  <div ref={dictScrollRef} style={{
                    flex: 1, minWidth: 0, alignSelf: 'stretch',
                    fontSize: 13.5, color: 'var(--ink)', lineHeight: 1.55,
                    overflowY: 'auto', overflowX: 'hidden',
                  }}>
                    {text || <span className="dim ital">Listening… speak naturally. We'll transcribe and tag.</span>}
                    <span style={{ display: 'inline-block', width: 1, height: '1em', verticalAlign: '-2px', background: 'currentColor', marginLeft: 2, animation: 'caret-blink 1s steps(1) infinite' }} />
                  </div>
                </div>
              ) : (
                <textarea
                  autoFocus
                  className="field"
                  rows={4}
                  value={text}
                  onChange={e => setText(e.target.value)}
                  placeholder="e.g. 14-year-old, recent self-harm disclosure at school, family declined hospital, no OHIP, Garden River — needs same-day support…"
                  style={{ fontSize: 14, lineHeight: 1.5 }}
                />
              )}

              {/* AI extraction preview — only when there's enough to read */}
              {text.trim().length > 12 && !rec && (
                <div className="row" style={{ gap: 8, marginTop: 10, flexWrap: 'wrap', alignItems: 'center' }}>
                  <span className="eyebrow" style={{ color: 'var(--accent-ink)' }}>
                    <Ico name="sparkle" size={10} style={{ marginRight: 4, verticalAlign: '-1px' }}/>AI sees
                  </span>
                  {extracted.length === 0 ? (
                    <span className="dim" style={{ fontSize: 11.5 }}>nothing yet — keep typing</span>
                  ) : (
                    extracted.map(c => <span key={c} className="chip sm accent">{c}</span>)
                  )}
                  {extracted.length > 0 && (
                    <a className="ilink" style={{ fontSize: 11.5, marginLeft: 'auto' }} onClick={() => { setPicked(extracted); setMode('tags'); }}>
                      Edit as tags →
                    </a>
                  )}
                </div>
              )}
            </div>
          ) : (
            <div>
              {/* Selected preview */}
              <div className="case-preview" style={{ marginBottom: 12 }}>
                {tagQuery
                  ? (
                    <span className="row" style={{ gap: 5, flexWrap: 'wrap' }}>
                      {picked.map(t => (
                        <span key={t} className="chip accent sm" style={{ paddingRight: 4 }}>
                          {t}
                          <button type="button" onClick={() => togglePick(t)} style={{ all: 'unset', cursor: 'pointer', marginLeft: 3, opacity: 0.6 }} aria-label={`Remove ${t}`}>
                            <Ico name="x" size={10}/>
                          </button>
                        </span>
                      ))}
                      {custom.trim() && <span className="chip sm">{custom.trim()}</span>}
                    </span>
                  )
                  : <span className="ph">Pick chips below to build the case — or dictate.</span>}
              </div>

              {/* Groups */}
              <div style={{ maxHeight: 240, overflowY: 'auto', paddingRight: 4, marginRight: -4 }}>
                {TAG_GROUPS.map(g => (
                  <div className="tag-group" key={g.key}>
                    <div className="tag-group-label">
                      <span className="eyebrow">{g.label}</span>
                    </div>
                    <div className="row" style={{ gap: 5, flexWrap: 'wrap' }}>
                      {g.options.map(o => (
                        <button type="button" key={o}
                          className={'tag-pill' + (picked.includes(o) ? ' on' : '')}
                          onClick={() => togglePick(o)}>
                          {o}
                        </button>
                      ))}
                    </div>
                  </div>
                ))}
              </div>

              {/* Custom tag */}
              <div className="row" style={{ gap: 8, marginTop: 4 }}>
                <input
                  className="field"
                  value={custom}
                  onChange={e => setCustom(e.target.value)}
                  placeholder="Add anything else — e.g. transport from Garden River"
                  style={{ fontSize: 13 }}
                  onKeyDown={e => {
                    if (e.key === 'Enter' && custom.trim()) {
                      e.preventDefault();
                      togglePick(custom.trim()); setCustom('');
                    }
                  }}
                />
              </div>

              {/* Dictation strip when recording in Tags mode */}
              {rec && (
                <div className="dictation" style={{ marginTop: 10 }}>
                  <span className="rec-dot" />
                  <span className="wave"><span/><span/><span/><span/><span/></span>
                  <span style={{ fontSize: 12.5 }}>Listening… new tags will appear above.</span>
                </div>
              )}
            </div>
          )}

          {/* Footer */}
          <div className="row" style={{ justifyContent: 'space-between', gap: 8, marginTop: 16 }}>
            <button type="button"
              className={'mic-btn' + (rec ? ' on' : '')}
              onClick={() => setRec(r => !r)}
              title="Toggle voice dictation">
              {rec ? (<><span className="rec-dot"/> Stop</>) : (<><Ico name="mic" size={12}/> Dictate</>)}
            </button>
            <div className="row" style={{ gap: 8 }}>
              <button type="button" className="btn" onClick={onClose}>Cancel</button>
              <button type="submit" className="btn accent" disabled={!canSubmit}
                style={{ opacity: canSubmit ? 1 : 0.5, cursor: canSubmit ? 'pointer' : 'not-allowed' }}>
                Find services →
              </button>
            </div>
          </div>
        </form>
      </div>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById('root')).render(<Home />);
