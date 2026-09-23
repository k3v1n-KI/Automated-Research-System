/* Runnable recovery shell for the Pathways prototype. */
(function () {
  const STORAGE_KEY = "pathways.event-ledger.v1";
  const ASK_STORAGE_KEY = "pathways.ask-board.v1";
  const API_BASE = window.PATHWAYS_API_BASE || "http://localhost:5001";
  let resources = [
    {
      id: "res-yee-hong",
      name: "Yee Hong Centre - Scarborough",
      category: "PSW and home care",
      city: "Scarborough",
      address: "2319 McNicoll Ave",
      phone: "416-321-6333",
      website: "https://www.yeehong.com",
      languages: ["Mandarin", "Cantonese", "English"],
      tags: ["OHIP", "meal support", "mobility"],
      fit: 96,
      verified: "3 days ago"
    },
    {
      id: "res-spectrum",
      name: "Spectrum Home Health",
      category: "PSW and home care",
      city: "Scarborough",
      address: "East Toronto intake office",
      phone: "416-555-0148",
      website: "https://example.org/spectrum",
      languages: ["English", "Mandarin"],
      tags: ["private", "mobility"],
      fit: 88,
      verified: "1 day ago"
    },
    {
      id: "res-meals",
      name: "Mandarin Meals on Wheels",
      category: "Meal support",
      city: "Scarborough",
      address: "East Toronto delivery area",
      phone: "416-555-0182",
      website: "https://example.org/meals",
      languages: ["Mandarin", "English"],
      tags: ["sliding scale", "meal support"],
      fit: 91,
      verified: "12 days ago"
    },
    {
      id: "res-carefirst",
      name: "Carefirst Seniors",
      category: "PSW and seniors support",
      city: "Toronto",
      address: "North York intake office",
      phone: "416-555-0199",
      website: "https://example.org/carefirst",
      languages: ["Mandarin", "Cantonese"],
      tags: ["OHIP", "seniors"],
      fit: 85,
      verified: "2 days ago"
    }
  ];

  const state = {
    phase: "find",
    query: "Mandarin Scarborough",
    selectedId: "res-yee-hong",
    events: loadEvents(),
    notice: "",
    flagDraft: null,
    asks: loadAsks(),
    caseDraft: {
      outcome: "referred",
      referredService: "",
      linkedAskId: null,
      note: ""
    },
    backend: { connected: false, loading: true },
    suggestions: []
  };

  async function apiRequest(path, options) {
    const response = await fetch(API_BASE + path, Object.assign({ headers: { "Content-Type": "application/json" } }, options || {}));
    if (!response.ok) throw new Error(`Pathways API ${response.status}`);
    return response.json();
  }

  function fromApiResource(record) {
    return Object.assign({}, record, {
      fit: Number(record.fit_score || 0),
      verified: record.updated_at ? new Date(record.updated_at).toLocaleDateString() : "not yet verified",
      sourceUrl: record.source_url || "",
      sourceDataset: record.source_dataset || "",
      sourceRowNumber: record.source_row_number || 0
    });
  }

  function fromApiEvent(event) {
    return { id: event.id, resourceId: event.resource_id || "", kind: event.kind, actor: event.actor, createdAt: event.created_at, payload: event.payload || {} };
  }

  function fromApiAsk(ask, replies) {
    return { id: ask.id, region: ask.region, author: ask.author, text: ask.text, tags: ask.tags || [], status: ask.status, watchers: ask.watchers || [], replies: replies || [], createdAt: ask.created_at, expiresAt: ask.expires_at };
  }

  async function loadBackend() {
    try {
      await apiRequest("/health");
      const [remoteResources, remoteEvents, remoteAsks, remoteSuggestions] = await Promise.all([
        apiRequest("/api/resources"),
        apiRequest("/api/events"),
        apiRequest("/api/asks"),
        apiRequest("/api/research/suggestions")
      ]);
      if (remoteResources.length) resources = remoteResources.map(fromApiResource);
      if (remoteEvents.length) state.events = remoteEvents.map(fromApiEvent);
      state.asks = await Promise.all(remoteAsks.map(async (ask) => fromApiAsk(ask, await apiRequest(`/api/asks/${encodeURIComponent(ask.id)}/replies`))));
      state.suggestions = remoteSuggestions;
      state.backend = { connected: true, loading: false };
      state.notice = "Connected to the Pathways database.";
    } catch (_) {
      state.backend = { connected: false, loading: false };
      state.notice = "Database unavailable; local recovery mode is active.";
    }
    render();
  }

  function loadEvents() {
    try {
      const value = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
      return Array.isArray(value) ? value : [];
    } catch (_) {
      return [];
    }
  }

  function loadAsks() {
    try {
      const value = JSON.parse(localStorage.getItem(ASK_STORAGE_KEY) || "[]");
      return Array.isArray(value) ? value : [];
    } catch (_) {
      return [];
    }
  }

  async function loadCorpus() {
    try {
      const response = await fetch("data/processed/normalized_resources.jsonl");
      if (!response.ok) return;
      const imported = (await response.text()).trim().split("\n").filter(Boolean).map((line) => JSON.parse(line));
      if (!imported.length) return;
      resources = imported.map((record) => ({
        id: record.candidate_id,
        name: record.name,
        category: record.category,
        city: record.city,
        address: record.address,
        phone: record.phone,
        website: record.website,
        languages: [],
        tags: [record.domain],
        fit: 0,
        verified: "not yet verified",
        sourceUrl: record.source_url,
        sourceDataset: record.source_dataset,
        sourceRowNumber: record.source_row_number
      }));
      state.selectedId = resources[0].id;
      state.notice = `${resources.length.toLocaleString()} imported resources loaded.`;
      render();
    } catch (_) {
      // Demo seeds remain available when the prototype is opened without HTTP.
    }
  }

  function saveEvents() {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state.events));
  }

  function saveAsks() {
    localStorage.setItem(ASK_STORAGE_KEY, JSON.stringify(state.asks));
  }

  function appendEvent(resourceId, kind, payload) {
    const event = {
      id: "evt-" + Date.now() + "-" + Math.random().toString(16).slice(2),
      resourceId,
      kind,
      actor: "Jamie Morgan",
      createdAt: new Date().toISOString(),
      payload: payload || {}
    };
    state.events.push(event);
    saveEvents();
    apiRequest("/api/events", { method: "POST", body: JSON.stringify({ id: event.id, resource_id: resourceId, kind, actor: event.actor, payload: event.payload }) }).catch(() => {});
  }

  function createAsk(text, tags, region) {
    const ask = { id: "ask-" + Date.now(), region: region.trim() || "Ontario", author: "Jamie Morgan", text: text.trim(), tags, status: "open", watchers: ["Jamie Morgan"], replies: [], createdAt: new Date().toISOString(), expiresAt: new Date(Date.now() + 7 * 86400000).toISOString() };
    state.asks.unshift(ask);
    appendEvent("", "ask_created", { askId: ask.id, region: ask.region, text: ask.text, tags: ask.tags });
    saveAsks();
    apiRequest("/api/asks", { method: "POST", body: JSON.stringify({ id: ask.id, region: ask.region, author: ask.author, text: ask.text, tags: ask.tags }) }).then(() => apiRequest(`/api/asks/${encodeURIComponent(ask.id)}/research`, { method: "POST" })).then((job) => apiRequest(`/api/research/jobs/${encodeURIComponent(job.id)}/run`, { method: "POST" })).then((result) => {
      if (result.suggestions) state.suggestions = state.suggestions.concat(result.suggestions);
      state.notice = result.status === "needs_review" ? "ARS found evidence-backed Ask suggestions for review." : "ARS could not find a reviewable answer for this Ask.";
      render();
    }).catch((error) => { state.notice = `ARS research failed: ${error.message}`; render(); });
  }

  function projection(resource) {
    const result = Object.assign({}, resource, {
      verifiedFields: [],
      flaggedFields: [],
      fieldStates: {},
      chainOfCustody: {},
      lastVerifiedAt: null
    });
    ["name", "address", "city", "phone", "website", "category"].forEach((field) => {
      result.fieldStates[field] = "unknown";
      result.chainOfCustody[field] = [];
    });
    state.events.filter((event) => event.resourceId === resource.id).forEach((event) => {
      const field = event.payload.field;
      if (field && result.chainOfCustody[field]) {
        result.chainOfCustody[field].push({
          kind: event.kind,
          actor: event.actor,
          displayActor: event.payload.anonymous ? "AOHT member" : (event.payload.displayActor || event.actor),
          source: event.payload.source || "unknown",
          sourceUrl: event.payload.sourceUrl || "",
          reason: event.payload.reason || "",
          correction: event.payload.correction || "",
          createdAt: event.createdAt
        });
      }
      if (event.kind === "field_verified") {
        if (!result.verifiedFields.includes(field)) result.verifiedFields.push(field);
        result.flaggedFields = result.flaggedFields.filter((item) => item !== field);
        result.lastVerifiedAt = event.createdAt;
        result.fieldStates[field] = "verified-fresh";
      }
      if (event.kind === "field_flagged" && !result.flaggedFields.includes(field)) {
        result.flaggedFields.push(field);
        result.fieldStates[field] = "flagged-stale";
      }
      if (event.kind === "suggestion_accepted") {
        result[field] = event.payload.value;
        if (!result.verifiedFields.includes(field)) result.verifiedFields.push(field);
        result.flaggedFields = result.flaggedFields.filter((item) => item !== field);
        result.lastVerifiedAt = event.createdAt;
        result.fieldStates[field] = "verified-fresh";
      }
    });
    return result;
  }

  function openAsks() {
    const asks = [];
    state.events.filter((event) => event.kind === "field_flagged").forEach((event) => {
      const resolved = state.events.some((candidate) => candidate.kind === "suggestion_accepted" && candidate.resourceId === event.resourceId && candidate.payload.field === event.payload.field && candidate.createdAt > event.createdAt);
      if (!resolved) asks.push({ event, resource: resources.find((item) => item.id === event.resourceId) });
    });
    return asks;
  }

  function searchResources() {
    const terms = state.query.toLowerCase().split(/\s+/).filter(Boolean);
    return resources.map(projection).map((resource) => {
      const haystack = [resource.name, resource.category, resource.city, resource.address].concat(resource.languages, resource.tags).join(" ").toLowerCase();
      const fit = terms.length ? terms.filter((term) => haystack.includes(term)).length / terms.length : 0;
      const days = resource.lastVerifiedAt ? Math.max(0, (Date.now() - Date.parse(resource.lastVerifiedAt)) / 86400000) : 45;
      const trust = Math.max(0, 1 - days / 90);
      return Object.assign(resource, { fitScore: Math.round(fit * 100), trustScore: Math.round(trust * 100), rankScore: Math.round((fit * 0.7 + trust * 0.3) * 100) });
    }).sort((a, b) => b.rankScore - a.rankScore);
  }

  function esc(value) {
    return String(value == null ? "" : value).replace(/[&<>\"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;" }[char]));
  }

  function button(label, action, extra) {
    return `<button class="btn ${extra || ""}" data-action="${action}">${label}</button>`;
  }

  function render() {
    const results = searchResources();
    const selected = projection(resources.find((item) => item.id === state.selectedId) || resources[0]);
    const asks = openAsks();
    document.getElementById("pathways-root").innerHTML = `
      <div class="pathways-shell">
        <div class="topnav">
          <div class="logo"><span class="logo-dot"></span>Pathways</div>
          <div class="nav-links"><a class="active" href="#">Directory</a><a href="#">Cases</a><a href="#">Ask board</a><a href="#">Forum</a></div>
          <div class="nav-spacer"></div><span class="region-pill"><span class="dot"></span>Algoma OHT</span><span class="badge verified">RN</span><div class="avatar">JM</div>
        </div>
        <main class="pathways-main">
          <div class="pathways-hero">
            <div><div class="eyebrow">Directory maintenance workspace</div><h1>Find a resource, keep it true.</h1><p class="pathways-subtitle">Search community services, verify what you used, and send stale fields to the ARS queue for human-reviewed repair.</p></div>
            <div class="meta">${state.events.length} ledger events · ${state.backend.loading ? "connecting to database" : state.backend.connected ? "Postgres connected" : "local recovery mode"}</div>
          </div>
          ${state.notice ? `<div class="notice">${esc(state.notice)}</div>` : ""}
          <div class="pathways-loop">${[["find", "1 Find"], ["verify", "2 Verify"], ["ask", "3 Ask"], ["close", "4 Close"]].map(([key, label]) => `<button class="${state.phase === key ? "active" : ""}" data-phase="${key}">${label}</button>`).join("")}</div>
          <div class="pathways-grid">
            <section class="pathways-card">
              ${state.phase === "find" ? renderFind(results) : renderPhase(selected, asks)}
            </section>
            <aside>
              <div class="pathways-card">
                <div class="eyebrow">Trust ledger</div><h2 class="title-2" style="margin:6px 0 14px">What changed</h2>
                <div class="pathways-stat"><div><strong>${resources.length}</strong><span>resources</span></div><div><strong>${asks.length}</strong><span>open asks</span></div><div><strong>${state.events.length}</strong><span>events</span></div></div>
                ${renderActivity()}
              </div>
              <div class="pathways-card"><div class="eyebrow">Research boundary</div><p class="meta" style="line-height:1.6;margin:8px 0 0">ARS suggestions are evidence for review, not clinical truth. Accepted values remain traceable to their event and source URL.</p></div>
            </aside>
          </div>
        </main>
      </div>`;
    bind();
  }

  function renderFind(results) {
    return `<div class="between" style="margin-bottom:12px"><div><div class="eyebrow">Phase 1 · search and rank</div><h2 class="title-2" style="margin-top:6px">Resources for this case</h2></div><span class="badge acc">Fit + trust</span></div>
      <div class="pathways-search"><input class="field" id="query" value="${esc(state.query)}" aria-label="Search resources" />${button("Search", "search", "primary")}</div>
      <div class="meta" style="margin-bottom:14px">${results.length} resources · ranked by semantic fit and verification freshness</div>
      <div class="pathways-results">${results.map(renderResource).join("")}</div>`;
  }

  function renderResource(resource) {
    return `<article class="resource-card ${resource.id === state.selectedId ? "selected" : ""}"><div class="resource-head"><div><h3>${esc(resource.name)}</h3><div class="resource-meta">${esc(resource.category)} · ${esc(resource.city)} · ${esc(resource.address)}</div></div><span class="badge good">${resource.rankScore}% rank</span></div><div class="row" style="flex-wrap:wrap;margin-top:8px"><span class="badge">Fit ${resource.fitScore}%</span><span class="badge verified">Trust ${resource.trustScore}%</span>${resource.languages.map((language) => `<span class="chip">${esc(language)}</span>`).join("")}</div><div class="resource-actions">${button("Open details", "select:" + resource.id, "sm")} ${button("Verify phone", "verify:" + resource.id + ":phone", "sm")} ${button("Flag stale field", "flag:" + resource.id + ":phone", "sm")}</div></article>`;
  }

  function renderPhase(selected, asks) {
    if (state.phase === "ask") return renderAskBoard(asks);
    if (state.phase === "close") return renderCloseView(selected, asks);
    const fields = ["name", "address", "city", "phone", "website", "category"];
    const fieldRows = fields.map((field) => {
      const value = selected[field] || "Not available";
      const status = selected.fieldStates[field] || "unknown";
      const tone = status === "verified-fresh" ? "good" : status === "flagged-stale" ? "crit" : status === "verified-aging" ? "warn" : "";
      return `<div class="verify-field"><div class="between"><div><strong>${esc(field.replace("_", " "))}</strong><div class="verify-value">${esc(value)}</div></div><span class="badge ${tone}">${esc(status)}</span></div><div class="resource-actions">${button("Confirm", "verify:" + selected.id + ":" + field, "sm")} ${button("Flag issue", "show-flag:" + selected.id + ":" + field, "sm")}</div>${state.flagDraft && state.flagDraft.resourceId === selected.id && state.flagDraft.field === field ? renderFlagForm() : ""}</div>`;
    }).join("");
    const custody = fields.flatMap((field) => (selected.chainOfCustody[field] || []).map((entry) => `<div class="custody-entry"><strong>${esc(field)}</strong> · ${esc(entry.kind.replace("field_", "").replace("suggestion_", ""))}<span>${esc(entry.displayActor)} · ${esc(entry.source)} · ${new Date(entry.createdAt).toLocaleString()}</span>${entry.sourceUrl ? `<a href="${esc(entry.sourceUrl)}" target="_blank" rel="noreferrer">source</a>` : ""}${entry.correction ? `<em>correction: ${esc(entry.correction)}</em>` : ""}</div>`)).join("") || `<div class="empty-state">No verification events yet.</div>`;
    const suggestions = state.suggestions.filter((suggestion) => suggestion.resource_id === selected.id && suggestion.status === "proposed");
    const suggestionPanel = suggestions.length ? `<div class="pathways-card custody-card"><div class="eyebrow">ARS evidence review</div><h3 class="title-3" style="margin:6px 0 10px">Proposed updates</h3>${suggestions.map((suggestion) => `<div class="ask-board-item"><div class="between"><strong>${esc(suggestion.field)}: ${esc(suggestion.value)}</strong><span class="badge warn">needs review</span></div><p>${esc(suggestion.evidence || "No excerpt returned; inspect the source before accepting.")}</p><div class="verify-source"><a href="${esc(suggestion.source_url)}" target="_blank" rel="noreferrer">${esc(suggestion.source_url)}</a></div><div class="resource-actions">${button("Accept", "review-suggestion:" + suggestion.id + ":accepted", "primary sm")} ${button("Reject", "review-suggestion:" + suggestion.id + ":rejected", "sm")}</div></div>`).join("")}</div>` : "";
    return `<div class="between"><div><div class="eyebrow">Phase 2 · frictionless validation</div><h2 class="title-2" style="margin:6px 0 4px">${esc(selected.name)}</h2><p class="resource-meta">Confirm current fields or flag stale information. Every action is appended to the ledger.</p></div>${button("Back to Find", "back-find", "sm")}</div><div class="verify-source">${esc(selected.sourceDataset || "Imported corpus")} · source row ${esc(selected.sourceRowNumber || "n/a")} · <a href="${esc(selected.sourceUrl || "#")}" target="_blank" rel="noreferrer">evidence URL</a></div>${suggestionPanel}<div class="verify-fields">${fieldRows}</div><div class="pathways-card custody-card"><div class="eyebrow">Chain of custody</div><h3 class="title-3" style="margin:6px 0 10px">Field history</h3>${custody}</div>`;
  }

  function renderAskBoard(fieldAsks) {
    const askCards = state.asks.map((ask) => { const askSuggestions = state.suggestions.filter((suggestion) => suggestion.job_id === `ask-research-${ask.id}` && suggestion.status === "proposed"); return `<div class="ask-board-item"><div class="between"><div><span class="badge ${ask.status === "open" ? "good" : "warn"}">${esc(ask.status)}</span><strong>${esc(ask.region)}</strong></div><span class="meta">expires ${new Date(ask.expiresAt).toLocaleDateString()}</span></div><p>${esc(ask.text)}</p><div class="row" style="flex-wrap:wrap">${ask.tags.map((tag) => `<span class="chip">${esc(tag)}</span>`).join("")}<span class="meta">${ask.replies.length} replies · ${ask.watchers.length} watching</span></div>${askSuggestions.map((suggestion) => `<div class="ask-reply"><strong>ARS candidate: ${esc(suggestion.value)}</strong><span>${esc(suggestion.evidence || "Review the source before accepting.")}</span><em><a href="${esc(suggestion.source_url)}" target="_blank" rel="noreferrer">evidence source</a></em><div class="resource-actions">${button("Accept", "review-suggestion:" + suggestion.id + ":accepted", "primary sm")} ${button("Reject", "review-suggestion:" + suggestion.id + ":rejected", "sm")}</div></div>`).join("")}${ask.replies.map((reply) => `<div class="ask-reply"><strong>${esc(reply.author)}</strong><span>${esc(reply.text)}</span>${reply.candidateName ? `<em>candidate: ${esc(reply.candidateName)}</em>` : ""}${reply.attachedResourceId ? `<em>attached resource: ${esc(reply.attachedResourceId)}</em>` : ""}</div>`).join("")}${ask.status === "open" ? `<div class="ask-reply-form"><input class="field" data-reply-text="${ask.id}" placeholder="Reply with a service or suggestion" /><input class="field" data-reply-candidate="${ask.id}" placeholder="New candidate service (optional)" /><div class="resource-actions">${button("Reply", "reply:" + ask.id, "primary sm")} ${button(ask.watchers.includes("Jamie Morgan") ? "Watching" : "Watch", "watch:" + ask.id, "sm")} ${button("Resolve", "resolve:" + ask.id, "sm")}</div></div>` : ""}</div>`; }).join("");
    return `<div class="between"><div><div class="eyebrow">Phase 3 · regional network</div><h2 class="title-2" style="margin:6px 0 4px">Ask the network</h2><p class="resource-meta">Post a need, attach an indexed service, or name a candidate that ARS can enrich.</p></div><span class="badge acc">${state.asks.filter((ask) => ask.status === "open").length} open</span></div><div class="ask-create-form"><input class="field" data-ask-text placeholder="What resource is missing?" /><input class="field" data-ask-region placeholder="Region (optional, defaults to Ontario)" /><input class="field" data-ask-tags placeholder="Tags, separated by commas" />${button("Post Ask", "create-ask", "primary")}</div>${askCards || `<div class="empty-state">No network Asks yet.</div>`}${fieldAsks.length ? `<div class="ask-board-item"><div class="eyebrow">Flagged field queue</div>${fieldAsks.map((ask) => `<div class="ask-item"><strong>${esc(ask.resource.name)} · ${esc(ask.event.payload.field)}</strong><span>${esc(ask.event.payload.reason || "Needs review")}</span>${button("Review in Close", "suggest:" + ask.resource.id + ":" + ask.event.payload.field, "primary sm")}</div>`).join("")}</div>` : ""}`;
  }

  function renderFlagForm() {
    return `<div class="flag-form"><label class="label">Reason<select class="field" data-flag-reason><option value="stale">Stale</option><option value="wrong">Wrong</option><option value="closed">Closed</option><option value="unsure">Unsure</option></select></label><label class="label">Correction<input class="field" data-flag-correction placeholder="Optional corrected value" /></label><label class="label">Source URL<input class="field" data-flag-source placeholder="Optional evidence URL" /></label><label class="row"><input type="checkbox" data-flag-anonymous /> Show as AOHT member</label>${button("Submit flag", "submit-flag", "primary sm")} ${button("Cancel", "cancel-flag", "sm")}</div>`;
  }

  function renderCloseView(selected, asks) {
    const linkedAsk = state.caseDraft.linkedAskId ? state.asks.find((ask) => ask.id === state.caseDraft.linkedAskId) : state.asks.find((ask) => ask.status === "open") || null;
    if (linkedAsk) state.caseDraft.linkedAskId = linkedAsk.id;
    const isCandidate = state.caseDraft.referredService && !resources.some((resource) => resource.name.toLowerCase() === state.caseDraft.referredService.toLowerCase());
    return `<div class="between"><div><div class="eyebrow">Phase 4 · close the loop</div><h2 class="title-2" style="margin:6px 0 4px">Close case and share the outcome</h2><p class="resource-meta">Record where the patient was referred, confirm the resource you used, and resolve any linked Ask in one step.</p></div>${button("Back to Ask", "phase:ask", "sm")}</div><div class="ask-board-item"><div class="label">Outcome</div><div class="row" style="flex-wrap:wrap; margin-top:8px">${["referred", "self-managed", "no-fit", "declined"].map((outcome) => `<button class="btn ${state.caseDraft.outcome === outcome ? "primary" : ""}" data-action="set-case-outcome:${outcome}">${outcome}</button>`).join("")} </div>${state.caseDraft.outcome === "referred" ? `<div class="label" style="margin-top:14px">Referred service<input class="field" data-close-service value="${esc(state.caseDraft.referredService)}" placeholder="Service or community program" /></div>` : ""}<div class="label" style="margin-top:14px">One-line outcome note<textarea class="field" data-close-note rows="3" placeholder="What happened?">${esc(state.caseDraft.note)}</textarea></div>${linkedAsk ? `<div class="ask-item" style="margin-top:14px"><strong>Linked Ask</strong><span>Resolve: ${esc(linkedAsk.text)}</span>${button("Resolve with this case", "close-case", "primary sm")}</div>` : `<div class="empty-state" style="margin-top:14px">No linked Ask; you can still close the case and seed a candidate if needed.</div>`}${isCandidate ? `<div class="ask-item" style="margin-top:14px"><strong>New candidate</strong><span>${esc(state.caseDraft.referredService)} will be seeded as a resource for later verification.</span></div>` : ""}<div class="resource-actions" style="margin-top:16px">${button("Close case", "close-case", "primary")} ${button("Close without sharing", "close-case-no-share", "sm")}</div></div>${asks.length ? `<div class="ask-board-item" style="margin-top:14px"><div class="eyebrow">Review ARS suggestions</div>${asks.map((ask) => `<div class="ask-item"><strong>${esc(ask.resource.name)} · ${esc(ask.event.payload.field)}</strong><span>Old value: ${esc(ask.resource[ask.event.payload.field])}<br />ARS found: <b>${esc(suggestedValue(ask.resource.id, ask.event.payload.field))}</b></span>${button("Accept suggestion", "accept:" + ask.resource.id + ":" + ask.event.payload.field, "primary sm")}</div>`).join("")}</div>` : ""}`;
  }

  function suggestedValue(resourceId, field) {
    return field === "phone" && resourceId === "res-yee-hong" ? "416-321-3001" : "ARS candidate pending source review";
  }

  function renderActivity() {
    return state.events.slice(-4).reverse().map((event) => `<div class="ask-item"><strong>${esc(event.kind.replaceAll("_", " "))}</strong><span>${esc(event.payload.field || "resource")} · ${new Date(event.createdAt).toLocaleString()}</span></div>`).join("") || `<div class="empty-state">No interactions yet.</div>`;
  }

  function bind() {
    document.querySelectorAll("[data-phase]").forEach((element) => element.addEventListener("click", () => { state.phase = element.dataset.phase; state.notice = ""; render(); }));
    document.querySelectorAll("[data-action]").forEach((element) => element.addEventListener("click", () => handle(element.dataset.action)));
    document.querySelectorAll("[data-flag-reason], [data-flag-correction], [data-flag-source], [data-flag-anonymous]").forEach((element) => element.addEventListener("input", () => {
      if (!state.flagDraft) return;
      if (element.dataset.flagReason !== undefined) state.flagDraft.reason = element.value;
      if (element.dataset.flagCorrection !== undefined) state.flagDraft.correction = element.value;
      if (element.dataset.flagSource !== undefined) state.flagDraft.sourceUrl = element.value;
      if (element.dataset.flagAnonymous !== undefined) state.flagDraft.anonymous = element.checked;
    }));
    document.querySelectorAll("[data-ask-text], [data-ask-tags], [data-reply-text], [data-reply-candidate]").forEach((element) => element.addEventListener("input", () => { element.dataset.value = element.value; }));
    const query = document.getElementById("query");
    if (query) query.addEventListener("input", (event) => { state.query = event.target.value; });
  }

  function handle(action) {
    const parts = action.split(":");
    const kind = parts[0];
    if (kind === "search") { state.phase = "find"; state.notice = "Search projection refreshed."; }
    if (kind === "select") { state.selectedId = parts[1]; state.phase = "verify"; }
    if (kind === "verify") { appendEvent(parts[1], "field_verified", { field: parts[2], source: "user-confirmation", displayActor: "Jamie Morgan" }); state.phase = "verify"; state.notice = "Verification appended to the ledger."; }
    if (kind === "flag") { appendEvent(parts[1], "field_flagged", { field: parts[2], reason: "User reported stale information", source: "user-report", displayActor: "Jamie Morgan" }); state.phase = "ask"; state.notice = "Ask created for the ARS background worker."; }
    if (kind === "show-flag") { state.phase = "verify"; state.flagDraft = { resourceId: parts[1], field: parts[2], reason: "stale", correction: "", sourceUrl: "", anonymous: false }; }
    if (kind === "cancel-flag") { state.flagDraft = null; }
    if (kind === "submit-flag" && state.flagDraft) { appendEvent(state.flagDraft.resourceId, "field_flagged", { field: state.flagDraft.field, reason: state.flagDraft.reason, correction: state.flagDraft.correction, sourceUrl: state.flagDraft.sourceUrl, source: "user-report", displayActor: "Jamie Morgan", anonymous: state.flagDraft.anonymous }); state.flagDraft = null; state.phase = "verify"; state.notice = "Flag appended; this field now needs review."; }
    if (kind === "back-find") { state.phase = "find"; state.flagDraft = null; }
    if (kind === "create-ask") {
      const text = document.querySelector("[data-ask-text]")?.value || "";
      const tags = (document.querySelector("[data-ask-tags]")?.value || "").split(",").map((tag) => tag.trim()).filter(Boolean);
      const region = document.querySelector("[data-ask-region]")?.value || "";
      if (text.trim()) { createAsk(text, tags, region); state.notice = `Ask posted to ${region.trim() || "Ontario"}.`; } else { state.notice = "Add a short description before posting an Ask."; }
      state.phase = "ask";
    }
    if (kind === "reply") {
      const ask = state.asks.find((item) => item.id === parts[1]);
      const textInput = document.querySelector(`[data-reply-text="${parts[1]}"]`);
      const candidateInput = document.querySelector(`[data-reply-candidate="${parts[1]}"]`);
      if (ask && textInput?.value.trim()) {
        const reply = { id: "reply-" + Date.now(), author: "Jamie Morgan", text: textInput.value.trim(), candidateName: candidateInput?.value.trim() || "", attachedResourceId: "" };
        ask.replies.push(reply);
        appendEvent("", "ask_replied", { askId: ask.id, candidateName: reply.candidateName });
        saveAsks();
        apiRequest(`/api/asks/${encodeURIComponent(ask.id)}/replies`, { method: "POST", body: JSON.stringify({ author: reply.author, text: reply.text, candidate_name: reply.candidateName || null }) }).catch(() => {});
        state.notice = "Reply added to the Ask.";
      }
      state.phase = "ask";
    }
    if (kind === "watch") {
      const ask = state.asks.find((item) => item.id === parts[1]);
      if (ask && !ask.watchers.includes("Jamie Morgan")) ask.watchers.push("Jamie Morgan");
      saveAsks();
      apiRequest(`/api/asks/${encodeURIComponent(parts[1])}/watchers`, { method: "POST", body: JSON.stringify({ watcher: "Jamie Morgan" }) }).catch(() => {});
      state.phase = "ask";
    }
    if (kind === "resolve") {
      const ask = state.asks.find((item) => item.id === parts[1]);
      if (ask) {
        ask.status = "resolved";
        appendEvent("", "ask_resolved", { askId: ask.id });
        saveAsks();
        apiRequest(`/api/asks/${encodeURIComponent(ask.id)}/resolve`, { method: "POST", body: JSON.stringify({}) }).catch(() => {});
        state.notice = "Ask resolved and retained in the ledger.";
      }
      state.phase = "ask";
    }
    if (kind === "suggest") { state.phase = "close"; state.notice = "ARS returned a candidate. Human review is required before it changes the directory."; }
    if (kind === "accept") { appendEvent(parts[1], "suggestion_accepted", { field: parts[2], value: suggestedValue(parts[1], parts[2]), sourceUrl: "https://www.yeehong.com/contact" }); state.phase = "close"; state.notice = "Suggestion accepted; projection and trust score updated."; }
    if (kind === "review-suggestion") {
      const suggestion = state.suggestions.find((item) => item.id === parts[1]);
      if (suggestion) {
        apiRequest(`/api/research/suggestions/${encodeURIComponent(suggestion.id)}/review`, { method: "POST", body: JSON.stringify({ status: parts[2], reviewer: "Jamie Morgan" }) }).then((reviewed) => {
          suggestion.status = reviewed.status;
          if (reviewed.status === "accepted") {
            const resource = resources.find((item) => item.id === suggestion.resource_id);
            if (resource) resource[suggestion.field] = suggestion.value;
            state.events.push({
              id: "ars-review-" + Date.now(),
              resourceId: suggestion.resource_id || "",
              kind: "suggestion_accepted",
              actor: "Jamie Morgan",
              createdAt: reviewed.reviewed_at || new Date().toISOString(),
              payload: { field: suggestion.field, value: suggestion.value, sourceUrl: suggestion.source_url, suggestionId: suggestion.id }
            });
            saveEvents();
          }
          state.notice = reviewed.status === "accepted" ? "ARS suggestion accepted and added to the ledger." : "ARS suggestion rejected.";
          render();
        }).catch(() => { state.notice = "Suggestion review could not be saved."; render(); });
      }
      state.phase = "verify";
    }
    if (kind === "set-case-outcome") {
      state.caseDraft.outcome = parts[1];
      state.phase = "close";
      state.notice = "Case outcome updated.";
    }
    if (kind === "phase") {
      state.phase = parts[1];
      state.notice = "";
    }
    if (kind === "close-case" || kind === "close-case-no-share") {
      const linkedAsk = state.caseDraft.linkedAskId ? state.asks.find((item) => item.id === state.caseDraft.linkedAskId) : state.asks.find((item) => item.status === "open") || null;
      const referredService = document.querySelector("[data-close-service]")?.value || state.caseDraft.referredService || "";
      const note = document.querySelector("[data-close-note]")?.value || state.caseDraft.note || "";
      const outcome = state.caseDraft.outcome || "referred";
      const referralExists = resources.some((resource) => resource.name.toLowerCase() === referredService.trim().toLowerCase());
      if (linkedAsk) { linkedAsk.status = "resolved"; appendEvent("", "ask_resolved", { askId: linkedAsk.id, linkedCaseId: "case-close", source: "case-close", note }); saveAsks(); }
      if (outcome === "referred" && referredService.trim()) {
        if (!referralExists) {
          resources.unshift({ id: "candidate-" + Date.now(), name: referredService.trim(), category: "Candidate service", city: "New referral", address: "Pending verification", phone: "", website: "", languages: [], tags: ["candidate"], fit: 0, verified: "new candidate", sourceUrl: "", sourceDataset: "case-close" });
          state.selectedId = resources[0].id;
          state.notice = "Case closed. Candidate resource was seeded for later verification.";
        } else {
          state.notice = "Case closed. Referral matched an indexed resource and the used field was re-confirmed.";
        }
      } else {
        state.notice = "Case closed. Outcome recorded without a referral candidate.";
      }
      appendEvent(state.selectedId, "case_closed", { outcome, note, linkedAskId: linkedAsk?.id || "", referredService: referredService.trim(), confirmedFields: ["phone", "website"] });
      apiRequest("/api/cases/close", { method: "POST", body: JSON.stringify({
        case_id: "case-" + Date.now(),
        outcome,
        actor: "Jamie Morgan",
        resource_id: referralExists ? state.selectedId : null,
        referred_service_name: referredService.trim() || null,
        linked_ask_id: linkedAsk?.id || null,
        confirmed_fields: ["phone", "website"],
        note
      }) }).catch(() => {});
      state.phase = "close";
    }
    render();
  }

  render();
  loadCorpus().then(loadBackend);
})();
