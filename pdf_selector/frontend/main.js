// PDF viewer — vanilla JS using PDF.js, with native scroll across all pages.
//
// Behaviour:
//   1. On first PDF load, render every page (canvas + text layer + annotation
//      layer) inside a scrollable wrap.  Pages stay mounted from then on so
//      scrolling is buttery-smooth.
//   2. An IntersectionObserver watches each page; whichever page covers the
//      most of the viewport is the "current" page.  When it changes we emit
//      a {kind:"page"} event back to Python so the ref-list on the left
//      filters to that page.  Debounced 200 ms so a fast scroll doesn't
//      flood Streamlit with reruns.
//   3. When only annotations change (e.g. user adopts an orphan ref), we
//      do NOT re-render canvases.  We just clear and re-paint annotation
//      layers — visually instant.
//   4. Text-layer painting is manual (PDF.js v4 TextLayer class was
//      flaky); spans are transparent but selectable.

const PDFJS_VERSION = "4.0.379";

async function loadPdfJs() {
  if (window._pdfjs) return window._pdfjs;
  const mod = await import(
    `https://cdn.jsdelivr.net/npm/pdfjs-dist@${PDFJS_VERSION}/build/pdf.min.mjs`
  );
  mod.GlobalWorkerOptions.workerSrc =
    `https://cdn.jsdelivr.net/npm/pdfjs-dist@${PDFJS_VERSION}/build/pdf.worker.min.mjs`;
  window._pdfjs = mod;
  return mod;
}

const state = {
  pdfDoc: null,
  pdfHash: null,
  annotations: [],
  pageNum: 1,
  totalPages: 0,
  pageElements: [],          // { el, annotLayer, viewport }
  pageObserver: null,
  pendingScrollToPage: null,
  selectionCounter: 0,
};

function b64ToBytes(b64) {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

function quickHash(bytes) {
  const len = bytes.length;
  let h = len * 2654435761;
  const step = Math.max(1, Math.floor(len / 256));
  for (let i = 0; i < len; i += step) {
    h = (h ^ bytes[i]) * 16777619;
    h |= 0;
  }
  return `${len}:${h >>> 0}`;
}

function selectionPayload() {
  const sel = window.getSelection();
  if (!sel || sel.isCollapsed) return null;
  const text = sel.toString().trim();
  if (text.length < 5) return null;

  let node = sel.anchorNode;
  while (node && (!node.dataset || !node.dataset.pageNum)) {
    node = node.parentNode;
  }
  const page = node ? parseInt(node.dataset.pageNum, 10) : state.pageNum;

  const rects = Array.from(sel.getRangeAt(0).getClientRects())
    .map(r => [r.left, r.top, r.right, r.bottom]);

  state.selectionCounter += 1;
  return {
    kind: "selection",
    text: text,
    page: page,
    rects: rects,
    id: `sel-${Date.now()}-${state.selectionCounter}`,
  };
}

function emitPageChange() {
  // Only emit page-change events back to Python if the host has
  // explicitly opted in.  Each emission costs a full Streamlit rerun
  // (the entire left panel re-renders), so by default we stay silent
  // and let the user scroll without consequences.
  if (!state.emitPageChanges) return;
  Streamlit.setComponentValue({
    kind: "page",
    page: state.pageNum,
    id: `page-${Date.now()}-${state.pageNum}`,
  });
}

function paintTextLayer(textContent, viewport, container, pdfjs) {
  const items = textContent.items || [];
  for (const item of items) {
    if (!item.str || item.str === "") continue;
    const tx = pdfjs.Util.transform(viewport.transform, item.transform);
    const fontHeight = Math.hypot(tx[2], tx[3]);
    if (fontHeight < 0.5) continue;
    const angle = Math.atan2(tx[1], tx[0]);

    const span = document.createElement("span");
    span.textContent = item.str;
    span.style.position    = "absolute";
    span.style.left        = `${tx[4]}px`;
    span.style.top         = `${tx[5] - fontHeight}px`;
    span.style.fontSize    = `${fontHeight}px`;
    span.style.fontFamily  = "sans-serif";
    span.style.transformOrigin = "0% 0%";
    if (Math.abs(angle) > 1e-6) {
      span.style.transform = `rotate(${angle}rad)`;
    }
    container.appendChild(span);
  }
}

// Convert a Python-side annotation (in PDF coords) to a CSS rect on the
// rendered page.  Used both by drawAnnotation (to position the overlay)
// and by the page-level click handler (to hit-test mouse coords).
function annotViewportRect(annot, viewport) {
  const [pdfX0, pdfY0, pdfX1, pdfY1] = annot.rect;
  const [vx0, vy0] = viewport.convertToViewportPoint(pdfX0, pdfY0);
  const [vx1, vy1] = viewport.convertToViewportPoint(pdfX1, pdfY1);
  return {
    left: Math.min(vx0, vx1),
    top:  Math.min(vy0, vy1),
    w:    Math.abs(vx1 - vx0),
    h:    Math.abs(vy1 - vy0),
  };
}

function drawAnnotation(layer, annot, viewport) {
  const r = annotViewportRect(annot, viewport);
  const el = document.createElement("div");
  el.style.position = "absolute";
  el.style.left   = `${r.left}px`;
  el.style.top    = `${r.top}px`;
  el.style.width  = `${r.w}px`;
  el.style.height = `${r.h}px`;
  // ALWAYS non-interactive — the layer is below the text layer, and
  // clicks are handled at the page level via hit-testing the
  // annotation rectangles.  This way text selection through the
  // colored highlight still works normally.
  el.style.pointerEvents = "none";
  if (annot.kind === "outline") {
    el.style.border  = `1.5px dashed ${annot.color || "#88a"}`;
    el.style.opacity = annot.opacity || 0.85;
  } else {
    el.style.background   = annot.color || "rgba(255, 215, 50, 0.3)";
    el.style.opacity      = annot.opacity || 0.30;
    el.style.mixBlendMode = "multiply";
  }
  if (annot.label) el.title = annot.label;
  if (typeof annot.ref_index === "number") {
    // Indicate clickability via cursor on hover — the click event itself
    // is handled at the page level.
    el.style.cursor = "pointer";
  }
  layer.appendChild(el);
}

function repaintAnnotationsOnly() {
  // Clear and redraw annotation overlays for every page.  PDF.js doesn't
  // render PDF annotation objects on its canvas, so we paint matched-ref
  // highlights here as DOM overlay divs.
  for (const pi of state.pageElements) {
    pi.annotLayer.innerHTML = "";
    for (const a of (state.annotations || [])) {
      if (a.page === pi.pageNum) drawAnnotation(pi.annotLayer, a, pi.viewport);
    }
  }
}

async function renderAllPages() {
  const pdfjs = await loadPdfJs();
  const pages = document.getElementById("pages");
  pages.innerHTML = "";
  state.pageElements = [];

  if (!state.pdfDoc) return;
  state.totalPages = state.pdfDoc.numPages;

  const wrap = document.getElementById("pages-wrap");
  const targetWidth = Math.max(360, Math.min(wrap.clientWidth - 28, 980));

  // Render pages sequentially.  ~50 ms per page is typical; for a 20-page
  // paper that's ~1 s upfront, then zero per-scroll cost.
  for (let n = 1; n <= state.totalPages; n++) {
    const page = await state.pdfDoc.getPage(n);
    const baseVp = page.getViewport({ scale: 1 });
    const scale = targetWidth / baseVp.width;
    const viewport = page.getViewport({ scale });

    const pageEl = document.createElement("div");
    pageEl.className = "pdf-page";
    pageEl.dataset.pageNum = n;
    pageEl.style.position   = "relative";
    pageEl.style.margin     = "12px auto";
    pageEl.style.width      = `${viewport.width}px`;
    pageEl.style.height     = `${viewport.height}px`;
    pageEl.style.background = "white";
    pageEl.style.boxShadow  = "0 2px 12px rgba(0,0,0,0.15)";

    const canvas = document.createElement("canvas");
    canvas.width  = viewport.width;
    canvas.height = viewport.height;
    canvas.style.position = "absolute";
    canvas.style.left = "0";
    canvas.style.top  = "0";
    pageEl.appendChild(canvas);
    await page.render({ canvasContext: canvas.getContext("2d"), viewport }).promise;

    // ORDER MATTERS:
    //   canvas (visual)  →  annotLayer (visible but non-interactive)
    //                     →  textLayer (on top — captures selection)
    // The annotation layer never captures pointer events; everything
    // passes through to the text layer.  To still get "click annotation
    // → focus ref" linking, we attach a page-level CLICK handler that
    // hit-tests the click coordinate against the page's annotation
    // rectangles — `click` only fires on a real click (no drag), so
    // text selection still works normally.
    const annotLayer = document.createElement("div");
    annotLayer.style.position = "absolute";
    annotLayer.style.left = "0";
    annotLayer.style.top  = "0";
    annotLayer.style.width  = `${viewport.width}px`;
    annotLayer.style.height = `${viewport.height}px`;
    annotLayer.style.pointerEvents = "none";
    pageEl.appendChild(annotLayer);

    const textLayer = document.createElement("div");
    textLayer.className = "text-layer";
    textLayer.style.position = "absolute";
    textLayer.style.left = "0";
    textLayer.style.top  = "0";
    textLayer.style.width  = `${viewport.width}px`;
    textLayer.style.height = `${viewport.height}px`;
    pageEl.appendChild(textLayer);

    const textContent = await page.getTextContent();
    paintTextLayer(textContent, viewport, textLayer, pdfjs);

    // Click hit-test against the page's annotation rectangles.
    // `click` only fires for a real (non-drag) click — drag selections
    // do NOT trigger it, so this co-exists with text selection.
    pageEl.addEventListener("click", (e) => {
      // Compute click position in page-local CSS pixels
      const pageRect = pageEl.getBoundingClientRect();
      const x = e.clientX - pageRect.left;
      const y = e.clientY - pageRect.top;
      // Find a clickable annotation that contains this point
      for (const a of (state.annotations || [])) {
        if (a.page !== n || typeof a.ref_index !== "number") continue;
        const r = annotViewportRect(a, viewport);
        if (x >= r.left && x <= r.left + r.w && y >= r.top && y <= r.top + r.h) {
          Streamlit.setComponentValue({
            kind: "ref_click",
            ref_index: a.ref_index,
            id: `ref-${Date.now()}-${a.ref_index}`,
          });
          e.stopPropagation();
          return;
        }
      }
    });

    pages.appendChild(pageEl);
    state.pageElements.push({
      el: pageEl, annotLayer, viewport, pageNum: n,
    });
  }

  // Initial annotation paint
  repaintAnnotationsOnly();

  // Scroll to requested page (if any) — passed in via state.pendingScrollToPage
  if (state.pendingScrollToPage) {
    const idx = state.pendingScrollToPage - 1;
    if (state.pageElements[idx]) {
      // small delay to let layout settle
      requestAnimationFrame(() =>
        state.pageElements[idx].el.scrollIntoView({ block: "start", behavior: "auto" })
      );
    }
    state.pendingScrollToPage = null;
  }

  setupPageObserver();
  updatePageBar();
  Streamlit.setFrameHeight();
}

function setupPageObserver() {
  // Stable, deterministic page detection: on scroll, find the page whose
  // CENTER is closest to the viewport's center.  IntersectionObserver was
  // flickering between adjacent pages because their ratios are both >0
  // when the user is in-between — a max-ratio tiebreak isn't smooth.
  const wrap = document.getElementById("pages-wrap");
  if (state._scrollHandler) wrap.removeEventListener("scroll", state._scrollHandler);

  function detectVisiblePage() {
    if (!state.pageElements.length) return;
    const wrapRect = wrap.getBoundingClientRect();
    const targetY = wrapRect.top + wrapRect.height / 2;   // viewport vertical centre

    let best = state.pageElements[0];
    let bestDist = Infinity;
    for (const pi of state.pageElements) {
      const r = pi.el.getBoundingClientRect();
      const centre = r.top + r.height / 2;
      const dist = Math.abs(centre - targetY);
      if (dist < bestDist) {
        bestDist = dist;
        best = pi;
      }
    }

    if (best.pageNum !== state.pageNum) {
      state.pageNum = best.pageNum;
      updatePageBar();
      emitPageChange();
    }
  }

  // Debounced scroll handler — wait for the user to pause before
  // emitting page-change to Python.  rAF for smoothness, setTimeout for
  // settle.
  let scrollRafId = null;
  let scrollSettleTimer = null;
  state._scrollHandler = () => {
    if (scrollRafId) cancelAnimationFrame(scrollRafId);
    if (scrollSettleTimer) clearTimeout(scrollSettleTimer);
    scrollSettleTimer = setTimeout(() => {
      scrollRafId = requestAnimationFrame(detectVisiblePage);
    }, 180);
  };
  wrap.addEventListener("scroll", state._scrollHandler, { passive: true });

  // Detect once on setup
  detectVisiblePage();
}

function updatePageBar() {
  const label = document.getElementById("page-label");
  if (label) label.textContent = `${state.pageNum} / ${state.totalPages}`;
}

function scrollToPage(n) {
  if (n < 1 || n > state.totalPages) return;
  const pi = state.pageElements[n - 1];
  if (pi) pi.el.scrollIntoView({ block: "start", behavior: "smooth" });
}

async function loadPdf(pdfBytes, annotations, scrollToTarget) {
  const newHash = quickHash(pdfBytes);
  const annotsChanged =
    JSON.stringify(state.annotations || []) !== JSON.stringify(annotations || []);
  state.annotations = annotations || [];

  // Fast path: same PDF — just update annotations (and maybe scroll).
  if (state.pdfDoc && state.pdfHash === newHash) {
    if (annotsChanged) repaintAnnotationsOnly();
    if (scrollToTarget && scrollToTarget !== state.pageNum) {
      scrollToPage(scrollToTarget);
    }
    return;
  }

  // Slow path: new PDF.
  const pdfjs = await loadPdfJs();
  state.pdfHash         = newHash;
  state.pdfDoc          = await pdfjs.getDocument({ data: pdfBytes }).promise;
  state.totalPages      = state.pdfDoc.numPages;
  state.pendingScrollToPage = scrollToTarget || null;
  state.pageNum         = scrollToTarget || 1;
  await renderAllPages();
}

function setupNavButtons() {
  const prev = document.getElementById("page-prev");
  const next = document.getElementById("page-next");
  if (prev) prev.onclick = () => scrollToPage(state.pageNum - 1);
  if (next) next.onclick = () => scrollToPage(state.pageNum + 1);
}

function setupSelectionWatcher() {
  const addBtn = document.getElementById("add-btn");
  const statusText = document.getElementById("status-text");
  let lastSelection = null;

  // Selection state shown in the toolbar.  The Add button is always
  // visible but only ENABLED when there's a real selection — clarifies
  // to the user that the button exists and what it does, vs floating
  // appear-on-selection which can be hidden/missed.
  function updateState() {
    const payload = selectionPayload();
    if (payload) {
      lastSelection = payload;
      addBtn.disabled = false;
      const preview = payload.text.length > 90 ? payload.text.slice(0, 90) + "…" : payload.text;
      statusText.innerHTML = `Selected: <b>${preview.replace(/&/g, "&amp;").replace(/</g, "&lt;")}</b>`;
    } else {
      addBtn.disabled = true;
      // Don't clear lastSelection on every mousemove — only when we know
      // there's no active selection at this moment.
      lastSelection = null;
      statusText.textContent = "Tip: drag-select an unhighlighted reference in the PDF, then click the button.";
    }
  }

  // Native browser selection lifecycle.  selectionchange fires whenever
  // the user changes the selection (drag, click-collapse, etc.).
  document.addEventListener("selectionchange", () => {
    // Debounce: settle for a brief moment before reading the selection
    clearTimeout(state._selectionTimer);
    state._selectionTimer = setTimeout(updateState, 50);
  });

  // Capture on mousedown so we don't lose the selection to focus changes
  addBtn.addEventListener("mousedown", (e) => {
    e.preventDefault();   // keep the selection alive
    const payload = selectionPayload();
    if (payload) lastSelection = payload;
  });

  addBtn.addEventListener("click", (e) => {
    e.preventDefault();
    // Last-ditch: re-read selection in case state went stale
    const fresh = selectionPayload();
    const payload = fresh || lastSelection;
    if (!payload) {
      statusText.textContent = "Couldn't read your selection — try selecting again.";
      return;
    }
    Streamlit.setComponentValue(payload);
    statusText.innerHTML = `<b>Sent to backend.</b>  Processing…`;
    addBtn.disabled = true;
    lastSelection = null;
    window.getSelection().removeAllRanges();
  });
}

let _lastRenderKey = null;
function onRender(event) {
  const args = event.detail.args || {};
  // Always update the emit flag — it controls JS-side behaviour for
  // page-change emissions and doesn't itself require a re-render.
  state.emitPageChanges = !!args.emit_page_changes;

  // Skip the render entirely if nothing meaningful changed.  Streamlit
  // can re-fire RENDER_EVENT after harmless reruns (e.g. page-change
  // round-trip); we don't want to interrupt the user's scroll.
  const key = JSON.stringify({
    p: (args.pdf_b64 || "").length,        // PDF identity proxy
    a: args.annotations || [],             // overlay rectangles
    s: args.scroll_to_page || null,        // explicit scroll target
    e: !!args.emit_page_changes,           // sync mode
  });
  if (key === _lastRenderKey) return;
  _lastRenderKey = key;

  const pdfBytes = b64ToBytes(args.pdf_b64 || "");
  const annotations = args.annotations || [];
  const scrollTarget = args.scroll_to_page || null;
  loadPdf(pdfBytes, annotations, scrollTarget).catch((err) => {
    document.getElementById("pages").innerHTML =
      `<pre style="color:red; padding:1em;">PDF render error: ${err.message}\n${err.stack}</pre>`;
    Streamlit.setFrameHeight();
  });
}

Streamlit.events.addEventListener(Streamlit.RENDER_EVENT, onRender);
Streamlit.setComponentReady();
setupNavButtons();
setupSelectionWatcher();
Streamlit.setFrameHeight(900);
