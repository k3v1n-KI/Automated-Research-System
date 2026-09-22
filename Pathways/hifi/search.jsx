// Hi-fi search results — Algoma OHT
// Query is read from ?q= so the home page's search bar feeds it.

function Search() {
  const params = new URLSearchParams(window.location.search);
  const initialQuery = params.get('q') || 'Same-day MH walk-in for an adolescent, no OHIP, Sault Ste. Marie';
  const [query, setQuery] = useState(initialQuery);
  const [editing, setEditing] = useState(false);
  const [postedAsk, setPostedAsk] = useState(false);

  // Pull token chips out of the query (rough)
  const chips = useMemo(() => {
    const tokens = [];
    const q = query.toLowerCase();
    if (/(mh|mental.?health|self.?harm|depress|anx)/.test(q)) tokens.push('MH');
    if (/walk.?in|drop.?in|same.?day/.test(q)) tokens.push('walk-in');
    if (/no ohip|uninsured|without ohip/.test(q)) tokens.push('uninsured');
    if (/adolesc|teen|youth|child/.test(q)) tokens.push('youth');
    const locMatch = q.match(/garden river|sault ste\. marie|sault|wawa|elliot lake|blind river/);
    if (locMatch) {
      const k = locMatch[0].toLowerCase();
      const map = {
        'garden river':      'Garden River',
        'sault ste. marie':  'Sault Ste. Marie',
        'sault':             'Sault Ste. Marie',
        'wawa':              'Wawa',
        'elliot lake':       'Elliot Lake',
        'blind river':       'Blind River',
      };
      tokens.push(map[k]);
    }
    return tokens;
  }, [query]);

  return (
    <div className="v2">
      <Topnav active="cases" />
      <div className="body page-in">
        <main style={{ maxWidth: 1180, margin: '0 auto', padding: '28px 32px 80px' }}>

          {/* Breadcrumb */}
          <div className="row" style={{ gap: 6, marginBottom: 14, fontSize: 12.5, color: 'var(--ink-3)' }}>
            <a href="../Pathways Hi-Fi.html" style={{ color: 'inherit' }}>Cases</a>
            <Ico name="chevron-right" size={11} />
            <span style={{ color: 'var(--ink)' }}>Search</span>
          </div>

          <PhaseRail current="find" />

          {/* Search recap */}
          <div style={{ marginBottom: 18 }}>
            <div className="eyebrow" style={{ marginBottom: 8 }}>Search</div>
            <div style={{
              display: 'flex', alignItems: 'center', gap: 12,
              background: 'var(--paper-2)', border: '1px solid var(--stroke)',
              borderRadius: 'var(--r-md)', padding: '12px 16px',
            }}>
              <Ico name="search" size={14} style={{ color: 'var(--ink-3)', flexShrink: 0 }}/>
              {editing ? (
                <div style={{ flex: 1, position: 'relative', display: 'flex', alignItems: 'center', gap: 6 }}>
                  <input
                    autoFocus value={query}
                    onChange={e => setQuery(e.target.value)}
                    onBlur={() => setTimeout(() => setEditing(false), 120)}
                    onKeyDown={e => e.key === 'Enter' && setEditing(false)}
                    style={{
                      flex: 1, fontSize: 15, color: 'var(--ink)', border: 'none',
                      background: 'transparent', outline: 'none', fontFamily: 'inherit',
                    }}
                  />
                  <MicMount value={query} setValue={setQuery} placement="inline"
                    samples={['Add: needs francophone provider — and after-school hours.']} />
                </div>
              ) : (
                <span style={{ fontSize: 15, color: 'var(--ink)', flex: 1 }}>{query}</span>
              )}
              <span className="row" style={{ gap: 4 }}>
                {chips.map(c => <span key={c} className="chip sm">{c}</span>)}
              </span>
              <a className="ilink" style={{ fontSize: 12 }} onClick={() => setEditing(v => !v)}>
                {editing ? 'Done' : 'Edit'}
              </a>
            </div>
            <div className="row" style={{ marginTop: 10, gap: 10, fontSize: 12, color: 'var(--ink-3)' }}>
              <span className="mono">{SEARCH_RESULTS.length} resources · 1 Ask answers this · ranked by fit + verification freshness</span>
            </div>
          </div>

          {/* 2-col: results + Asks rail */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 320px', gap: 24, alignItems: 'start' }}>

            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              {SEARCH_RESULTS.map((r, i) => (
                <ResultRow key={r.id} r={r} focused={i === 0}/>
              ))}

              {/* Not finding it? prompt */}
              <div style={{
                marginTop: 8,
                border: '1px dashed var(--stroke-2)', borderRadius: 'var(--r-md)',
                padding: '18px 20px', background: 'var(--accent-tint)',
                display: 'flex', gap: 16, alignItems: 'center', justifyContent: 'space-between',
              }}>
                <div style={{ flex: 1 }}>
                  <div className="row" style={{ gap: 7, marginBottom: 4 }}>
                    <Ico name="corner-arrow" size={13} stroke={1.8} style={{ color: 'var(--accent)' }}/>
                    <span className="eyebrow" style={{ color: 'var(--accent-ink)' }}>Not finding it?</span>
                  </div>
                  <div style={{ fontSize: 14, color: 'var(--ink)', lineHeight: 1.45 }}>
                    Post this search as an <strong>Ask</strong>. Other AOHT members across Algoma see it for 7 days
                    (12 in rural communities) and can point to resources we don't have indexed yet.
                  </div>
                </div>
                <div className="row" style={{ gap: 8, flexShrink: 0 }}>
                  <a className="btn sm" href="case-resolve.html">Add what you know</a>
                  <button className="btn accent sm" onClick={() => setPostedAsk(true)}>Post as Ask →</button>
                </div>
              </div>
            </div>

            {/* Right rail */}
            <aside style={{ display: 'flex', flexDirection: 'column', gap: 12, position: 'sticky', top: 12 }}>
              <div className="card" style={{ padding: 16 }}>
                <div className="between" style={{ marginBottom: 10 }}>
                  <span className="eyebrow">Open Asks · Algoma</span>
                  <a className="ilink" href="asks.html" style={{ fontSize: 11 }}>All →</a>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                  {ASKS.slice(0, 3).map(a => (
                    <a key={a.id} href="asks.html" style={{
                      paddingBottom: 10, borderBottom: '1px solid var(--stroke)',
                      textDecoration: 'none', color: 'inherit', display: 'block',
                    }}>
                      <div style={{ fontSize: 12.5, color: 'var(--ink)', lineHeight: 1.45, marginBottom: 5 }}>
                        {a.text}
                      </div>
                      <div className="row" style={{ gap: 7, fontSize: 10.5, color: 'var(--ink-3)' }}>
                        <span style={{ fontWeight: 500 }}>{a.who}</span>
                        <span>·</span>
                        <span>{a.region}</span>
                        <span>·</span>
                        <span className="mono">{a.replies} replies</span>
                      </div>
                    </a>
                  ))}
                  <a className="row" href="asks.html" style={{ gap: 6, fontSize: 12, color: 'var(--accent-ink)', fontWeight: 500, marginTop: 2, textDecoration: 'none' }}>
                    <Ico name="plus" size={12} stroke={2}/> Post a new Ask
                  </a>
                </div>
              </div>

              <div className="card muted" style={{ padding: 14, background: 'var(--paper-2)' }}>
                <div className="eyebrow" style={{ marginBottom: 6 }}>How ranking works</div>
                <p style={{ fontSize: 12, color: 'var(--ink-3)', lineHeight: 1.5, margin: 0 }}>
                  AI scores fit against your case. We boost resources with recent member verification
                  and penalize ones with active stale flags. <a className="ilink">Learn more →</a>
                </p>
              </div>
            </aside>
          </div>
        </main>
      </div>

      {postedAsk && (
        <div className="toast">
          <span className="dot" style={{ width: 7, height: 7, borderRadius: '50%' }}/>
          Posted to Algoma OHT · 7 days · <a href="asks.html" style={{ color: 'white', textDecoration: 'underline' }}>View your Ask</a>
        </div>
      )}
    </div>
  );
}

function ResultRow({ r, focused }) {
  const m = matchTier(r.score);
  return (
    <a href="resource.html" data-tour={focused ? 'top-result' : undefined} className="card tap" style={{
      padding: 18, display: 'grid', gridTemplateColumns: '52px 1fr auto', gap: 16,
      borderColor: focused ? 'var(--stroke-2)' : 'var(--stroke)',
      textDecoration: 'none', color: 'inherit',
    }}>
      <div style={{
        background: focused ? 'var(--ink)' : 'var(--paper-2)',
        color: focused ? 'var(--paper)' : 'var(--ink-2)',
        borderRadius: 'var(--r-sm)', padding: '8px 0',
        display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', gap: 5,
      }}>
        <MatchMeter tier={m.tier} dark={focused}/>
        <div className="mono" style={{ fontSize: 9, opacity: 0.7 }}>{m.tier}/5</div>
      </div>

      <div>
        <div className="row" style={{ gap: 10, marginBottom: 6 }}>
          <div style={{ fontSize: 15, fontWeight: 600, color: 'var(--ink)', letterSpacing: '-0.015em' }}>{r.name}</div>
          <span className="mono" style={{ fontSize: 11, color: 'var(--ink-3)' }}>{r.neighborhood}</span>
        </div>
        <div style={{ fontSize: 13, color: 'var(--ink-2)', lineHeight: 1.5, marginBottom: 10, maxWidth: 620 }}>
          {r.blurb}
        </div>
        <div className="row" style={{ gap: 12, flexWrap: 'wrap' }}>
          <VerifyAggregate verified={r.verified} total={r.total} flagged={r.flagged} compact/>
          <span style={{ color: 'var(--ink-4)' }}>·</span>
          <span className="row" style={{ gap: 4, flexWrap: 'wrap' }}>
            {r.chips.map(c => <span key={c} className="chip sm">{c}</span>)}
          </span>
        </div>
        {r.flagged > 0 && (
          <div style={{
            marginTop: 10, paddingTop: 10, borderTop: '1px dashed var(--stroke)',
            display: 'flex', gap: 8, alignItems: 'center', fontSize: 11.5, color: 'var(--ink-3)',
          }}>
            <span style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--crit)' }}/>
            <span><strong style={{ color: '#b91c1c' }}>Referral path</strong> flagged 2× — "piloting phone-ahead booking from primary care"</span>
          </div>
        )}
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 8, minWidth: 120 }}>
        <span className="mono" style={{ fontSize: 10.5, color: 'var(--ink-3)' }}>{m.label}</span>
        <Ico name="arrow-right" size={14} style={{ color: 'var(--ink-3)' }}/>
      </div>
    </a>
  );
}

ReactDOM.createRoot(document.getElementById('root')).render(<Search />);
