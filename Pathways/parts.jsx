// Shared wireframe building blocks for Pathways.
// Exported to window for cross-script use.
const { useState, useMemo, Fragment } = React;

// ── Top navigation ────────────────────────────────────────
function TopNav({ active, region = "Mid-West Toronto OHT" }) {
  const links = ["Home", "Find for patient", "Directory", "Forum", "Shortage", "Saved"];
  return (
    <div className="topnav">
      <div className="logo">
        <span className="logo-dot" />
        Pathways
      </div>
      <div className="nav-links">
        {links.map(l => (
          <a key={l} className={active === l ? "active" : ""} href="#">{l}</a>
        ))}
      </div>
      <div className="nav-spacer" />
      <span className="region-pill" title="Click to switch OHT">
        <span className="dot" />
        {region}
        <span className="meta" style={{ opacity: 0.5 }}>▾</span>
      </span>
      <span className="badge verified" style={{ fontSize: 9 }}>RN ✓</span>
      <div className="avatar">JM</div>
    </div>
  );
}

// ── Frame: app chrome around content (no sidebar) ────────
function AppFrame({ active, region, children, padding = true }) {
  return (
    <div className="app">
      <TopNav active={active} region={region} />
      <div className="body" style={padding ? {} : { padding: 0 }}>
        {children}
      </div>
    </div>
  );
}

// ── Frame with sidebar (admin / oversight) ───────────────
function AdminFrame({ active, sidebarItems, sidebarActive, children }) {
  return (
    <div className="app">
      <TopNav active={active} />
      <div style={{ display: "flex", flex: 1, overflow: "hidden" }}>
        <aside className="sidebar">
          {sidebarItems.map((g, i) => (
            <div key={i}>
              {g.label && <div className="group-label">{g.label}</div>}
              {g.items.map(it => (
                <div
                  key={it}
                  className={"item" + (sidebarActive === it ? " active" : "")}
                >
                  <span style={{ width: 4, height: 4, borderRadius: 2, background: 'var(--ink-3)' }} />
                  {it}
                </div>
              ))}
            </div>
          ))}
        </aside>
        <div className="body">{children}</div>
      </div>
    </div>
  );
}

// ── Generic placeholder primitives ───────────────────────
const Stripe = ({ label, h = 80, w }) => (
  <div className="stripe" style={{ height: h, width: w }}>{label}</div>
);
const AIZone = ({ children, style }) => (
  <div className="ai-zone" style={{ padding: 12, ...style }}>{children}</div>
);

// ── Score row ────────────────────────────────────────────
function ScoreRow({ label, pct, tone }) {
  return (
    <div className="row" style={{ gap: 8 }}>
      <span className="meta" style={{ width: 80, flexShrink: 0 }}>{label}</span>
      <div className="score-bar" style={{ flex: 1 }}>
        <i style={{ width: pct + '%', background: tone === 'warn' ? 'var(--warn)' : tone === 'crit' ? 'var(--crit)' : 'var(--accent)' }} />
      </div>
      <span className="meta" style={{ width: 28, textAlign: 'right' }}>{pct}</span>
    </div>
  );
}

// ── Annotation callout (sticky-note for designer notes) ──
function Note({ children }) {
  return <div className="note">{children}</div>;
}

// ── Designer-note ribbon shown above each artboard ───────
function ArtboardNote({ children }) {
  return (
    <div style={{
      fontFamily: 'var(--font-mono)', fontSize: 10,
      color: 'var(--ink-3)', padding: '6px 4px 8px',
      lineHeight: 1.5,
    }}>{children}</div>
  );
}

Object.assign(window, {
  TopNav, AppFrame, AdminFrame, Stripe, AIZone, ScoreRow, Note, ArtboardNote
});
