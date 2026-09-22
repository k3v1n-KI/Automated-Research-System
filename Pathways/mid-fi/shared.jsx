// Pathways mid-fi — shared atoms, icons, topnav, brand mark
// Exposed on window for cross-script access.

const { useState, useMemo, useEffect, useRef, createContext, useContext } = React;

// ── Router (tiny, in-memory) ───────────────────────────
const RouterCtx = createContext(null);
function useRouter() { return useContext(RouterCtx); }

function RouterProvider({ children, initial = { name: 'home' } }) {
  const [route, setRoute] = useState(initial);
  const [history, setHistory] = useState([initial]);
  const go = (next) => {
    setRoute(next);
    setHistory(h => [...h, next]);
    window.scrollTo({ top: 0, behavior: 'instant' });
  };
  const back = () => {
    if (history.length <= 1) return;
    const next = history[history.length - 2];
    setRoute(next);
    setHistory(h => h.slice(0, -1));
  };
  return (
    <RouterCtx.Provider value={{ route, go, back, canBack: history.length > 1 }}>
      {children}
    </RouterCtx.Provider>
  );
}

// ── Icons (minimal stroke set) ─────────────────────────
function Ico({ name, size = 16, stroke = 1.6, ...rest }) {
  const sz = size;
  const common = { width: sz, height: sz, viewBox: '0 0 24 24', fill: 'none',
    stroke: 'currentColor', strokeWidth: stroke, strokeLinecap: 'round', strokeLinejoin: 'round', ...rest };
  switch (name) {
    case 'arrow-right': return (<svg {...common}><path d="M5 12h14M13 6l6 6-6 6"/></svg>);
    case 'arrow-left':  return (<svg {...common}><path d="M19 12H5M11 18l-6-6 6-6"/></svg>);
    case 'plus':        return (<svg {...common}><path d="M12 5v14M5 12h14"/></svg>);
    case 'search':      return (<svg {...common}><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></svg>);
    case 'sparkle':     return (<svg {...common}><path d="M12 3l2.2 5.8L20 11l-5.8 2.2L12 19l-2.2-5.8L4 11l5.8-2.2L12 3z"/></svg>);
    case 'pin':         return (<svg {...common}><path d="M12 21s7-7.5 7-12a7 7 0 1 0-14 0c0 4.5 7 12 7 12z"/><circle cx="12" cy="9" r="2.5"/></svg>);
    case 'phone':       return (<svg {...common}><path d="M5 4h4l2 5-3 2a12 12 0 0 0 5 5l2-3 5 2v4a2 2 0 0 1-2 2A16 16 0 0 1 3 6a2 2 0 0 1 2-2z"/></svg>);
    case 'globe':       return (<svg {...common}><circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3a14 14 0 0 1 0 18M12 3a14 14 0 0 0 0 18"/></svg>);
    case 'clock':       return (<svg {...common}><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></svg>);
    case 'check':       return (<svg {...common}><path d="m5 12 5 5L20 7"/></svg>);
    case 'check-circle':return (<svg {...common}><circle cx="12" cy="12" r="9"/><path d="m8 12 3 3 5-6"/></svg>);
    case 'x':           return (<svg {...common}><path d="M6 6l12 12M18 6 6 18"/></svg>);
    case 'star':        return (<svg {...common}><path d="m12 3 2.7 5.6 6.3.9-4.5 4.4 1 6.1L12 17.3 6.5 20l1-6.1L3 9.5l6.3-.9L12 3z"/></svg>);
    case 'flag':        return (<svg {...common}><path d="M5 21V4M5 4h11l-2 4 2 4H5"/></svg>);
    case 'shield':      return (<svg {...common}><path d="M12 3 4 6v6c0 5 4 8 8 9 4-1 8-4 8-9V6l-8-3z"/></svg>);
    case 'list':        return (<svg {...common}><path d="M8 6h13M8 12h13M8 18h13M3.5 6h.01M3.5 12h.01M3.5 18h.01"/></svg>);
    case 'map':         return (<svg {...common}><path d="m3 6 6-2 6 2 6-2v14l-6 2-6-2-6 2V6zM9 4v14M15 6v14"/></svg>);
    case 'filter':      return (<svg {...common}><path d="M3 5h18M6 12h12M10 19h4"/></svg>);
    case 'bookmark':    return (<svg {...common}><path d="M6 4h12v17l-6-4-6 4V4z"/></svg>);
    case 'share':       return (<svg {...common}><path d="M4 12v7a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-7M16 6l-4-4-4 4M12 2v13"/></svg>);
    case 'chevron-down':return (<svg {...common}><path d="m6 9 6 6 6-6"/></svg>);
    case 'chevron-right':return (<svg {...common}><path d="m9 6 6 6-6 6"/></svg>);
    case 'circle':      return (<svg {...common}><circle cx="12" cy="12" r="9"/></svg>);
    case 'people':      return (<svg {...common}><circle cx="9" cy="8" r="3.5"/><path d="M3 20a6 6 0 0 1 12 0M16 11a3 3 0 1 0 0-6M21 20a5 5 0 0 0-5-5"/></svg>);
    case 'building':    return (<svg {...common}><path d="M4 21V5a2 2 0 0 1 2-2h8a2 2 0 0 1 2 2v16M16 9h2a2 2 0 0 1 2 2v10M4 21h16M8 7h0M8 11h0M8 15h0M12 7h0M12 11h0M12 15h0"/></svg>);
    case 'note':        return (<svg {...common}><path d="M5 4h11l3 3v13a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V5a1 1 0 0 1 1-1zM8 10h8M8 14h6"/></svg>);
    case 'verify':      return (<svg {...common}><path d="m12 3 2.5 1.7L17 4l1.4 2.4 2.6.9-.5 2.7L22 12l-1.5 2-.5 2.7-2.6.9L17 20l-2.5-.7L12 21l-2.5-1.7L7 20l-1.4-2.4-2.6-.9.5-2.7L2 12l1.5-2 .5-2.7 2.6-.9L7 4l2.5.7L12 3z"/><path d="m9 12 2.2 2L15 10"/></svg>);
    default: return null;
  }
}

// ── Brand mark ─────────────────────────────────────────
function BrandMark() {
  return (
    <span className="brand-mark" aria-hidden>
      <svg viewBox="0 0 24 24" fill="none">
        <circle cx="12" cy="12" r="11" fill="var(--accent)" opacity="0.10"/>
        <path d="M5 17c2-3 4-3 7 0s5 3 7 0" stroke="var(--accent)" strokeWidth="1.8" strokeLinecap="round" fill="none"/>
        <circle cx="12" cy="12" r="2.4" fill="var(--accent)"/>
      </svg>
    </span>
  );
}

// ── Topnav ─────────────────────────────────────────────
function Topnav({ active = 'cases' }) {
  const r = useRouter();
  const items = [
    { id: 'cases', label: 'Cases', go: () => r.go({ name: 'home' }) },
    { id: 'directory', label: 'Directory' },
    { id: 'forum', label: 'Forum' },
    { id: 'shortage', label: 'Shortage' },
    { id: 'submit', label: 'Submit' },
  ];
  return (
    <header className="topnav">
      <a className="brand" onClick={() => r.go({ name: 'home' })} style={{ cursor: 'pointer' }}>
        <BrandMark />
        Pathways
      </a>
      <nav className="nav-links">
        {items.map(it => (
          <a key={it.id} className={active === it.id ? 'active' : ''} onClick={it.go}>{it.label}</a>
        ))}
      </nav>
      <span className="nav-spacer" />
      <span className="region-pill" title="Active Ontario Health Team region">
        <span className="dot" />
        <span style={{ color: 'var(--ink)', fontWeight: 500 }}>East Toronto OHT</span>
        <Ico name="chevron-down" size={12} />
      </span>
      <button className="btn ghost btn-icon" title="Notifications">
        <Ico name="circle" size={14} stroke={2} />
      </button>
      <span className="avatar" title="R. Okafor, RN">RO</span>
    </header>
  );
}

// ── Resource avatar (coloured initials) ────────────────
const AVATAR_HUES = [165, 220, 320, 50, 280, 180, 130, 25];
function rAvatarColor(seed) {
  let s = 0;
  for (let i = 0; i < seed.length; i++) s = (s * 31 + seed.charCodeAt(i)) | 0;
  const h = AVATAR_HUES[Math.abs(s) % AVATAR_HUES.length];
  return {
    background: `oklch(0.42 0.085 ${h})`,
  };
}
function RAvatar({ name, size = 'md' }) {
  const initials = name.split(/\s+/).slice(0, 2).map(w => w[0]).join('').toUpperCase();
  return <span className={`r-avatar ${size === 'lg' ? 'lg' : size === 'sm' ? 'sm' : ''}`} style={rAvatarColor(name)}>{initials}</span>;
}

// ── Fit indicator (5 bars) ─────────────────────────────
function FitBars({ value = 4, max = 5 }) {
  return (
    <span className="fit-bars" title={`Fit ${value}/${max}`}>
      {Array.from({ length: max }, (_, i) => (
        <i key={i} className={i < value ? 'on' : ''} />
      ))}
    </span>
  );
}

// ── AI tag ─────────────────────────────────────────────
function AITag({ children = 'AI suggestion' }) {
  return <span className="ai-tag">{children}</span>;
}

Object.assign(window, { RouterProvider, useRouter, Ico, BrandMark, Topnav, RAvatar, FitBars, AITag });
