// Pathways v2 — shared atoms (topnav, icons)

const { useState, useEffect, useRef, useMemo } = React;

function V2Ico({ name, size = 14, stroke = 1.6, ...rest }) {
  const common = { width: size, height: size, viewBox: '0 0 24 24', fill: 'none',
    stroke: 'currentColor', strokeWidth: stroke, strokeLinecap: 'round', strokeLinejoin: 'round', ...rest };
  switch (name) {
    case 'arrow-right': return (<svg {...common}><path d="M5 12h14M13 6l6 6-6 6"/></svg>);
    case 'arrow-up-right': return (<svg {...common}><path d="M7 17 17 7M9 7h8v8"/></svg>);
    case 'plus':        return (<svg {...common}><path d="M12 5v14M5 12h14"/></svg>);
    case 'search':      return (<svg {...common}><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></svg>);
    case 'sparkle':     return (<svg {...common} viewBox="0 0 24 24"><path d="M12 3l1.8 5L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.7L12 3z"/></svg>);
    case 'cmd':         return (<svg {...common}><path d="M9 6a3 3 0 1 0 0 6h6a3 3 0 1 0 0-6v6m0 0a3 3 0 1 0 0 6V12m-6 0a3 3 0 1 0 0 6V12"/></svg>);
    case 'clock':       return (<svg {...common}><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></svg>);
    case 'check':       return (<svg {...common}><path d="m5 12 5 5L20 7"/></svg>);
    case 'chevron-right': return (<svg {...common}><path d="m9 6 6 6-6 6"/></svg>);
    case 'chevron-down':  return (<svg {...common}><path d="m6 9 6 6 6-6"/></svg>);
    case 'pin':         return (<svg {...common}><path d="M12 21s7-7.5 7-12a7 7 0 1 0-14 0c0 4.5 7 12 7 12z"/><circle cx="12" cy="9" r="2.5"/></svg>);
    case 'globe':       return (<svg {...common}><circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3a14 14 0 0 1 0 18M12 3a14 14 0 0 0 0 18"/></svg>);
    case 'bell':        return (<svg {...common}><path d="M6 16V11a6 6 0 0 1 12 0v5l2 3H4l2-3z"/><path d="M10 21h4"/></svg>);
    case 'star':        return (<svg {...common}><path d="m12 3 2.7 5.6 6.3.9-4.5 4.4 1 6.1L12 17.3 6.5 20l1-6.1L3 9.5l6.3-.9L12 3z"/></svg>);
    case 'book':        return (<svg {...common}><path d="M4 5a2 2 0 0 1 2-2h12v18H6a2 2 0 0 1-2-2V5z"/><path d="M4 5v14M8 7h6M8 11h6"/></svg>);
    case 'people':      return (<svg {...common}><circle cx="9" cy="8" r="3.5"/><path d="M3 20a6 6 0 0 1 12 0M16 11a3 3 0 1 0 0-6M21 20a5 5 0 0 0-5-5"/></svg>);
    case 'flag':        return (<svg {...common}><path d="M5 21V4M5 4h11l-2 4 2 4H5"/></svg>);
    case 'corner-arrow': return (<svg {...common}><path d="M5 9l4-4 4 4M9 5v10a4 4 0 0 0 4 4h6"/></svg>);
    default: return null;
  }
}

function V2Topnav({ active = 'cases' }) {
  const items = [
    { id: 'cases', label: 'Cases' },
    { id: 'directory', label: 'Directory' },
    { id: 'forum', label: 'Forum' },
    { id: 'shortage', label: 'Shortage' },
  ];
  return (
    <header className="topnav">
      <span className="brand">
        <span className="brand-mark" />
        Pathways
      </span>
      <nav className="nav-links">
        {items.map(it => (
          <a key={it.id} className={active === it.id ? 'active' : ''}>{it.label}</a>
        ))}
      </nav>
      <span className="nav-spacer" />
      <span className="region-pill">
        <span className="dot" />
        <span style={{ color: 'var(--ink)', fontWeight: 500 }}>East Toronto OHT</span>
        <V2Ico name="chevron-down" size={11} />
      </span>
      <button className="btn ghost" style={{ padding: '6px 8px' }}><V2Ico name="bell" size={14}/></button>
      <span className="avatar">RO</span>
    </header>
  );
}

// Shared cases data
const V2_CASES = [
  { id: 'c-241', title: 'Senior, post-discharge, lives alone', summary: 'Home support · meals · Cantonese · ODSP', tags: ['senior', 'home-care'], status: 'open', when: '2h ago', region: 'East York' },
  { id: 'c-240', title: 'Adolescent, self-harm risk, uninsured', summary: 'Walk-in MH < 48h · no OHIP', tags: ['youth', 'mental-health'], status: 'referred', when: 'yesterday', region: 'Scarborough' },
  { id: 'c-238', title: 'New parent · postpartum support', summary: 'Peer group + lactation · French preferred', tags: ['perinatal'], status: 'open', when: '2d ago', region: 'East York' },
  { id: 'c-235', title: 'Adult · housing-insecure · methadone', summary: 'Continuing care + shelter coordination', tags: ['housing', 'sud'], status: 'closed', when: '4d ago', region: 'Riverdale' },
];

Object.assign(window, { V2Ico, V2Topnav, V2_CASES });
