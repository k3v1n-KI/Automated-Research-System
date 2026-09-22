// Pathways mid-fi — App shell + router + tweaks panel

function App() {
  return (
    <RouterProvider initial={{ name: 'home' }}>
      <Shell />
    </RouterProvider>
  );
}

function Shell() {
  const { route } = useRouter();
  const [t, setTweak] = useTweaks(/*EDITMODE-BEGIN*/{
    "accent": "forest",
    "density": "medium",
    "aiTinted": true
  }/*EDITMODE-END*/);

  // Apply tweaks as body classes
  useEffect(() => {
    const cls = [];
    if (t.accent === 'slate') cls.push('accent-slate');
    if (t.accent === 'aubergine') cls.push('accent-aubergine');
    if (t.accent === 'clay') cls.push('accent-clay');
    if (t.density === 'compact') cls.push('density-compact');
    if (!t.aiTinted) cls.push('ai-flat');
    document.body.className = cls.join(' ');
  }, [t.accent, t.density, t.aiTinted]);

  let screen;
  switch (route.name) {
    case 'entry': screen = <ScreenEntry />; break;
    case 'results': screen = <ScreenResults />; break;
    case 'resource': screen = <ScreenResource id={route.id} />; break;
    case 'home':
    default: screen = <ScreenHome />; break;
  }

  return (
    <>
      <div className="app">{screen}</div>
      <TweaksPanel title="Tweaks">
        <TweakSection label="Accent">
          <TweakColor value={t.accent} onChange={(v) => setTweak('accent', v)}
            options={[
              { value: 'forest', color: 'oklch(0.42 0.085 165)' },
              { value: 'slate', color: 'oklch(0.42 0.075 250)' },
              { value: 'aubergine', color: 'oklch(0.40 0.085 320)' },
              { value: 'clay', color: 'oklch(0.55 0.105 50)' },
            ]} />
        </TweakSection>
        <TweakSection label="Density">
          <TweakRadio label="Spacing" value={t.density} onChange={(v) => setTweak('density', v)}
            options={[
              { value: 'medium', label: 'Medium' },
              { value: 'compact', label: 'Compact' },
            ]} />
        </TweakSection>
        <TweakSection label="AI zones">
          <TweakToggle label="Tinted background" value={t.aiTinted} onChange={(v) => setTweak('aiTinted', v)} />
        </TweakSection>
      </TweaksPanel>
    </>
  );
}

ReactDOM.createRoot(document.getElementById('root')).render(<App />);
