// ============================================================
// XAUUSD AI Terminal — app.js v5  (Dual-Timeframe + Tiers)
// ============================================================

const POLL_MS       = 60_000;
const REFRESH_SEC   = 900;   // 15 min server cycle

let _data           = null;
let _newsAll        = [];
let _currentFilter  = 'ALL';
let _prevScalp      = null;
let _prevSwing      = null;
let _cdRemaining    = REFRESH_SEC;
let _cdInterval     = null;
let _pollTimer      = null;
let _timeframe      = 'NY';
let _isRefreshingNow= false;
let _staleTriggered = false;

// ── FORMAT ────────────────────────────────────────────────────
const fmt  = v => new Intl.NumberFormat('en-US',{style:'currency',currency:'USD',minimumFractionDigits:2}).format(v);
const pct  = v => `${(v*100).toFixed(1)}%`;
const mmss = s => `${String(Math.floor(s/60)).padStart(2,'0')}:${String(s%60).padStart(2,'0')}`;

// ── PAGE SWITCHING ────────────────────────────────────────────
function switchPage(name) {
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.getElementById(`page-${name}`)?.classList.add('active');
    document.querySelector(`[data-page="${name}"]`)?.classList.add('active');
    if (name === 'news'     && _data) renderFullNews(_newsAll);
    if (name === 'calendar') loadCalendar();
    if (name === 'history')  loadHistory();
}

// ── COUNTDOWN RING ────────────────────────────────────────────
function updateCountdown(s) {
    const arc  = document.getElementById('countdown-arc');
    const lbl  = document.getElementById('countdown-label');
    if (!arc || !lbl) return;
    const circ = 2 * Math.PI * 15.9;
    arc.style.strokeDasharray  = `${circ}`;
    arc.style.strokeDashoffset = `${circ * (1 - s / REFRESH_SEC)}`;
    lbl.textContent = mmss(s);
}
function startCountdown() {
    clearInterval(_cdInterval);
    _cdRemaining = REFRESH_SEC;
    _cdInterval  = setInterval(() => { _cdRemaining = Math.max(0, _cdRemaining - 1); updateCountdown(_cdRemaining); }, 1000);
}

// ── TICKER ────────────────────────────────────────────────────
function buildTicker(news) {
    const scroll = document.getElementById('ticker-scroll');
    if (!scroll || !news?.length) return;
    const SC = {BULLISH:'var(--green)', BEARISH:'var(--red)', NEUTRAL:'var(--text-3)'};
    let html = '';
    news.forEach(n => {
        const col = SC[n.sentiment] || 'var(--text-2)';
        const tag = n.sentiment==='BULLISH'?'▲': n.sentiment==='BEARISH'?'▼':'—';
        html += `<span class="tick-item">
            <span class="tick-tag" style="color:${col}">${tag}</span>
            <a href="${n.url||'#'}" target="_blank" rel="noopener" class="tick-link">${n.headline}</a>
            <span class="tick-src">${n.source}</span>
        </span><span class="tick-sep">·</span>`;
    });
    const newHTML = html + html;
    if (scroll.innerHTML === newHTML) return;
    scroll.innerHTML = newHTML;
    scroll.style.animation = 'none';
    scroll.offsetHeight;
    const dur = Math.max(30, Math.floor(scroll.scrollWidth / 2 / 60));
    scroll.style.animation = `tickerScroll ${dur}s linear infinite`;
}

// ── DUAL-TIMEFRAME SIGNAL CARDS ───────────────────────────────
function renderDualSignals(data) {
    const scalp = data.scalp || {};
    const swing = data.swing || {};
    const scalpSig = scalp.signal || 'NEUTRAL';
    const swingSig = swing.signal || 'NEUTRAL';
    const icons = {LONG:'▲', SHORT:'▼', NEUTRAL:'◈'};

    // ── Scalp Card ──
    const cardScalp = document.getElementById('card-scalp');
    cardScalp.className = `card card-tf card-scalp sig-${scalpSig}`;
    document.getElementById('scalp-icon').textContent = icons[scalpSig] || '◈';
    document.getElementById('scalp-signal').textContent = scalpSig;

    // Strength badge
    const scalpStr = scalp.strength || 'NONE';
    const scalpStrEl = document.getElementById('scalp-strength');
    scalpStrEl.textContent = scalpStr === 'NONE' ? 'NO SIGNAL' : `● ${scalpStr}`;
    scalpStrEl.className = `strength-badge str-${scalpStr}`;

    // SL/TP
    if (scalpSig !== 'NEUTRAL') {
        document.getElementById('scalp-sl').textContent = fmt(scalp.stop_loss || 0);
        document.getElementById('scalp-tp').textContent = fmt(scalp.take_profit || 0);
    } else {
        document.getElementById('scalp-sl').textContent = '⏸ —';
        document.getElementById('scalp-tp').textContent = '⏸ —';
    }

    // ── Swing Card ──
    const cardSwing = document.getElementById('card-swing');
    cardSwing.className = `card card-tf card-swing sig-${swingSig}`;
    document.getElementById('swing-icon').textContent = icons[swingSig] || '◈';
    document.getElementById('swing-signal').textContent = swingSig;

    const swingStr = swing.strength || 'NONE';
    const swingStrEl = document.getElementById('swing-strength');
    swingStrEl.textContent = swingStr === 'NONE' ? 'NO SIGNAL' : `● ${swingStr}`;
    swingStrEl.className = `strength-badge str-${swingStr}`;

    if (swingSig !== 'NEUTRAL') {
        document.getElementById('swing-sl').textContent = fmt(swing.stop_loss || 0);
        document.getElementById('swing-tp').textContent = fmt(swing.take_profit || 0);
    } else {
        document.getElementById('swing-sl').textContent = '⏸ —';
        document.getElementById('swing-tp').textContent = '⏸ —';
    }

    // ── Top bar pills ──
    const pillScalp = document.getElementById('pill-scalp');
    const pillSwing = document.getElementById('pill-swing');
    if (pillScalp) {
        pillScalp.textContent = `⚡${scalpSig}`;
        pillScalp.className = `signal-pill pill-mini pill-${scalpSig}`;
    }
    if (pillSwing) {
        pillSwing.textContent = `🎯${swingSig}`;
        pillSwing.className = `signal-pill pill-mini pill-${swingSig}`;
    }

    _prevScalp = scalpSig;
    _prevSwing = swingSig;
}

// ── PROBABILITY CARD ──────────────────────────────────────────
function renderProbCard(data) {
    const probUp = data.prediction?.probability_up ?? 0.5;
    document.getElementById('prob-up-val').textContent = pct(probUp);
    setTimeout(() => {
        document.getElementById('prob-bar').style.width   = `${(probUp*100).toFixed(1)}%`;
        document.getElementById('conf-needle').style.left = `${(probUp*100).toFixed(1)}%`;
    }, 80);

    // VADER
    const badge = document.getElementById('vader-badge');
    if (badge) {
        const lbl = data.live_vader_label || '—';
        const col = {BULLISH:'var(--green)', BEARISH:'var(--red)', NEUTRAL:'var(--text-3)'}[lbl] || 'var(--text-2)';
        badge.textContent = `${lbl}  (${data.live_vader_sentiment >= 0 ? '+' : ''}${(data.live_vader_sentiment||0).toFixed(3)})`;
        badge.style.color = col;
    }

    // Model votes
    const v = data.model_votes;
    if (v) {
        document.getElementById('vote-cat').textContent = pct(v.catboost);
        document.getElementById('vote-xgb').textContent = pct(v.xgboost);
        document.getElementById('vote-lgb').textContent = pct(v.lightgbm);
        ['cat','xgb','lgb'].forEach(id => {
            const el  = document.getElementById(`vote-${id}`);
            const val = v[{cat:'catboost',xgb:'xgboost',lgb:'lightgbm'}[id]];
            el.className = val>=0.6?'vote-val text-green':val<=0.4?'vote-val text-red':'vote-val text-muted';
        });
    }

    // Consensus
    const consensusEl = document.getElementById('consensus-badge');
    if (consensusEl) {
        if (data.consensus_ok === true) {
            consensusEl.textContent = '\u2713 CONSENSUS';
            consensusEl.className = 'consensus-badge consensus-ok';
        } else if (data.consensus_ok === false) {
            consensusEl.textContent = '\u2717 DIVERGED';
            consensusEl.className = 'consensus-badge consensus-fail';
        } else {
            consensusEl.textContent = '\u25ca NEUTRAL';
            consensusEl.className = 'consensus-badge consensus-neutral';
        }
    }
}

// ── SMART TIMING ──────────────────────────────────────────────
function renderSmartTiming(data) {
    const timing = data.smart_timing || {};

    // Session label
    const sessionLabel = document.getElementById('timing-session-label');
    if (sessionLabel) {
        sessionLabel.textContent = timing.session_label || 'Unknown session';
    }

    // Next signal window
    const nextText = document.getElementById('timing-next-text');
    if (nextText) {
        nextText.textContent = timing.next_signal_window || 'Monitoring...';
    }

    // Proximity bar (only show when NEUTRAL)
    const sig = data.prediction?.signal || 'NEUTRAL';
    const proxBlock = document.getElementById('proximity-block');
    if (proxBlock) {
        if (sig === 'NEUTRAL' && timing.proximity_pct > 0) {
            proxBlock.style.display = 'block';
            document.getElementById('proximity-pct').textContent = `${timing.proximity_pct.toFixed(0)}%`;
            document.getElementById('proximity-fill').style.width = `${Math.min(100, timing.proximity_pct)}%`;
            const hint = document.getElementById('proximity-hint');
            if (hint) {
                hint.textContent = `Approaching ${timing.nearest_threshold || '—'} threshold`;
            }
        } else {
            proxBlock.style.display = 'none';
        }
    }

    // Flip conditions
    const flipBlock = document.getElementById('flip-conditions');
    const flipList = document.getElementById('flip-list');
    if (flipBlock && flipList) {
        if (timing.flip_conditions && timing.flip_conditions.length > 0) {
            flipBlock.style.display = 'block';
            flipList.innerHTML = timing.flip_conditions.map(c => `<li>${c}</li>`).join('');
        } else {
            flipBlock.style.display = 'none';
        }
    }
}

// ── EXECUTION + LEVELS ────────────────────────────────────────
function renderExecution(data) {
    const rm  = data.risk_management || {};
    const lv  = data.intraday_levels || {};
    document.getElementById('top-date').textContent  = data.target_date || data.date || '\u2014';
    document.getElementById('entry-val').textContent = fmt(rm.entry_price||0);
    document.getElementById('atr-val').textContent   = fmt(rm.atr_14||0);
    document.getElementById('r2-val').textContent = fmt(lv.r2||0);
    document.getElementById('r1-val').textContent = fmt(lv.r1||0);
    document.getElementById('pp-val').textContent = fmt(lv.pp||0);
    document.getElementById('s1-val').textContent = fmt(lv.s1||0);
    document.getElementById('s2-val').textContent = fmt(lv.s2||0);
}

// ── AI ANALYSIS PAGE ──────────────────────────────────────────
function renderAnalysis(data) {
    const n = data.narrative;
    if (n) {
        document.getElementById('narrative-block').innerHTML = `
            <p class="narrative-reasoning">${n.reasoning}</p>
            <div class="narrative-risk">${n.risk_note}</div>`;
    }
    const sl = document.getElementById('shap-list');
    if (sl && data.shap_drivers?.length) {
        sl.innerHTML = '';
        data.shap_drivers.forEach(d => {
            const up  = d.direction==='UP';
            const div = document.createElement('div');
            div.className = `shap-item ${up?'up':'down'}`;
            div.innerHTML = `
                <div class="shap-icon">${up?'▲':'▼'}</div>
                <div class="shap-body">
                    <div class="shap-text">${d.text}</div>
                    <div class="shap-feat">${d.feature} &nbsp;|&nbsp; impact: ${d.impact>=0?'+':''}${d.impact.toFixed(4)}</div>
                </div>
                <div class="shap-arrow ${up?'up':'down'}">${up?'BULL':'BEAR'}</div>`;
            sl.appendChild(div);
        });
    }
    const v = data.model_votes || {};
    const probUp = data.prediction?.probability_up ?? 0.5;
    [['cat','catboost'],['xgb','xgboost'],['lgb','lightgbm']].forEach(([id,key]) => {
        const val = v[key] || 0;
        const bar = document.getElementById(`bar-${id}`);
        const lbl = document.getElementById(`dval-${id}`);
        if (bar) { bar.style.width = `${(val*100).toFixed(1)}%`; bar.style.background = val>=0.6?'var(--green)':val<=0.4?'var(--red)':'var(--amber)'; }
        if (lbl) { lbl.textContent = pct(val); }
    });
    const mb = document.getElementById('bar-meta');
    const ml = document.getElementById('dval-meta');
    if (mb) { mb.style.width = `${(probUp*100).toFixed(1)}%`; }
    if (ml) { ml.textContent = pct(probUp); }
}

// ── NEWS PAGE ─────────────────────────────────────────────────
function renderFullNews(news) {
    const list = document.getElementById('full-news-list');
    if (!list) return;
    let filtered;
    if (_currentFilter === 'ALL') {
        filtered = news;
    } else if (_currentFilter === 'GEO') {
        filtered = news.filter(n => n.category === 'WAR_MILITARY');
    } else if (_currentFilter === 'FED') {
        filtered = news.filter(n => n.category === 'FED_POLICY');
    } else {
        filtered = news.filter(n => n.sentiment === _currentFilter);
    }
    const SC = {BULLISH:'var(--green)', BEARISH:'var(--red)', NEUTRAL:'var(--text-3)'};
    const CC = {
        WAR_MILITARY: '#e74c3c',
        FED_POLICY:   '#f39c12',
        INFLATION:    '#27ae60',
        DOLLAR_FX:    '#3498db',
        CRISIS:       '#e67e22',
        ENERGY:       '#8e44ad',
        GOLD_MARKET:  '#f5c842',
        OTHER:        'var(--text-3)',
    };
    if (!filtered.length) {
        list.innerHTML = `<p style="color:var(--text-3);padding:20px;">No ${_currentFilter.toLowerCase()} headlines found.</p>`;
        return;
    }
    list.innerHTML = filtered.map(n => {
        const col  = SC[n.sentiment] || 'var(--text-3)';
        const tag  = n.sentiment==='BULLISH'?'▲ BULLISH':n.sentiment==='BEARISH'?'▼ BEARISH':'— NEUTRAL';
        const dt   = n.datetime ? n.datetime.slice(0,16) : '';
        const cat  = n.category  || 'OTHER';
        const icon = n.cat_icon  || '📰';
        const catCol = CC[cat] || 'var(--text-3)';
        return `<a href="${n.url||'#'}" target="_blank" rel="noopener" class="news-item-full">
            <div class="nf-left">
                <div class="nf-headline">
                    <span class="nf-cat-badge" style="background:${catCol}22;color:${catCol};border:1px solid ${catCol}44;" title="${cat}">${icon}</span>
                    ${n.headline}
                </div>
                <div class="nf-meta">
                    <span class="nf-source">${n.source}</span>
                    ${dt ? `<span class="nf-time">${dt}</span>` : ''}
                    <span class="nf-cat-label" style="color:${catCol}">${cat.replace('_',' ')}</span>
                </div>
            </div>
            <div class="nf-sentiment" style="color:${col}">${tag}</div>
        </a>`;
    }).join('');
}

function filterNews(sentiment, btn) {
    _currentFilter = sentiment;
    document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    renderFullNews(_newsAll);
}

// ── CALENDAR PAGE ─────────────────────────────────────────────
async function loadCalendar() {
    const list = document.getElementById('full-calendar-list');
    if (!list || list.dataset.loaded) return;
    list.innerHTML = '<div class="cal-loading">Loading events...</div>';
    try {
        const res  = await fetch('/api/macro-calendar');
        const data = await res.json();
        if (!data.events?.length) { list.innerHTML = '<p style="color:var(--text-3)">No events in next 30 days.</p>'; return; }
        const IMP = {HIGH:'var(--red)', MEDIUM:'var(--amber)', LOW:'var(--blue)'};
        list.innerHTML = data.events.map(ev => {
            const col = IMP[ev.impact]||'#fff';
            let badge = '';
            if (ev.is_today)   badge = '<span class="cal-today-badge">TODAY</span>';
            else if (ev.is_tomorrow) badge = '<span class="cal-tmrw-badge">TOMORROW</span>';
            else               badge = `<span class="cal-days-badge">in ${ev.days_until}d</span>`;
            return `<div class="cal-item-full impact-${ev.impact.toLowerCase()}">
                <div class="calf-left">
                    <div class="calf-header">
                        <span class="cal-dot" style="background:${col}"></span>
                        <span class="calf-event">${ev.event}</span>
                        ${badge}
                    </div>
                    <div class="calf-desc">${ev.description}</div>
                </div>
                <div class="calf-right">
                    <div class="calf-date">${ev.date}</div>
                    <div class="calf-impact" style="color:${col}">${ev.impact}</div>
                </div>
            </div>`;
        }).join('');
        list.dataset.loaded = '1';
    } catch(e) { list.innerHTML = '<p style="color:var(--red)">Failed to load calendar.</p>'; }
}

// ── MAIN RENDER ───────────────────────────────────────────────
function renderAll(data, news) {
    _data    = data;
    _newsAll = news || data.live_news || [];

    document.getElementById('loader').style.display = 'none';
    document.getElementById('page-tabs').style.display = 'flex';
    
    const activePage = document.querySelector('.page.active');
    if (!activePage) {
        document.getElementById('page-dashboard').classList.add('active');
    }

    // Stale banner
    const hd = data.header || {};
    const staleBanner = document.getElementById('stale-banner');
    const staleMsg    = document.getElementById('stale-msg');
    if (hd.is_stale && staleBanner && !_isRefreshingNow && !_staleTriggered) {
        staleBanner.style.display = 'flex';
        staleMsg.textContent = `Data from ${hd.inference_date} (${hd.data_age_days}d old). Auto-refreshing system...`;
        _staleTriggered = true;
        triggerRefresh();
    } else if (staleBanner && !hd.is_stale) {
        staleBanner.style.display = 'none';
        _staleTriggered = false;
    }

    if (hd.refreshing_daily && !hd.is_stale && !_isRefreshingNow) {
        if (staleBanner) {
            staleBanner.style.display = 'flex';
            staleMsg.textContent = `System is currently running a background daily update...`;
            const refreshBtn = document.querySelector('.refresh-btn');
            if (refreshBtn) { refreshBtn.disabled = true; refreshBtn.textContent = '⏳ Updating...'; }
        }
        _isRefreshingNow = true;
        pollRefreshCompletion();
    }

    const upd = document.getElementById('tab-update-label');
    if (upd && data.last_refresh) upd.textContent = `Updated ${data.last_refresh}`;

    const nb = document.getElementById('news-count-badge');
    if (nb) nb.textContent = _newsAll.length || '';

    // ── Render all sections ──
    renderDualSignals(data);
    renderProbCard(data);
    renderSmartTiming(data);
    renderExecution(data);
    renderAnalysis(data);
    buildTicker(_newsAll);
    updateRiskAlerts(data.risk_management?.entry_price);

    const ntl = document.getElementById('news-total-label');
    if (ntl) ntl.textContent = `${_newsAll.length} headlines`;

    const newsPage = document.getElementById('page-news');
    if (newsPage && newsPage.classList.contains('active')) {
        renderFullNews(_newsAll);
    }
}

// ── LIGHTWEIGHT POLL ──────────────────────────────────────────
async function pollSignal() {
    try {
        const [res, hRes] = await Promise.all([
            fetch('/api/predict'),
            fetch('/api/health')
        ]);
        const data = await res.json();
        const hd = await hRes.json();

        if (data.status === 'success') {
            data.header = hd;
            const dot = document.querySelector('.upd-dot');
            if (dot) dot.classList.remove('offline');
            renderAll(data);
        } else {
            console.warn('API error in poll:', data.message);
            const upd = document.getElementById('tab-update-label');
            if (upd) upd.textContent = 'API Error';
            const dot = document.querySelector('.upd-dot');
            if (dot) dot.classList.add('offline');
        }
        startCountdown();
    } catch(e) {
        console.warn('Poll error', e);
        const upd = document.getElementById('tab-update-label');
        if (upd) upd.textContent = 'Connection lost (Retrying...)';
        const dot = document.querySelector('.upd-dot');
        if (dot) dot.classList.add('offline');
    }
}

// ── INITIAL FULL LOAD ─────────────────────────────────────────
async function initialLoad() {
    try {
        const [pr, nr, hr] = await Promise.all([
            fetch('/api/predict'),
            fetch('/api/live-news'),
            fetch('/api/health')
        ]);
        const data = await pr.json();
        const news = await nr.json();
        const hd   = await hr.json();

        if (data.status !== 'success') {
            document.getElementById('loader').innerHTML =
                `<div class="loader-inner"><p style="color:var(--red)">⚠ API Error: ${data.message}</p></div>`;
            return;
        }

        data.header = hd;
        renderAll(data, news.news || []);
        startCountdown();
        _pollTimer = setInterval(pollSignal, POLL_MS);

    } catch(err) {
        console.error(err);
        document.getElementById('loader').innerHTML =
            `<div class="loader-inner"><p style="color:var(--red)">⚠ Cannot connect to API.<br>
            <small>Make sure step11_api_server.py is running.</small></p></div>`;
        const dot = document.querySelector('.upd-dot');
        if (dot) dot.classList.add('offline');
        const upd = document.getElementById('tab-update-label');
        if (upd) upd.textContent = 'Offline';
    }
}

// ── MANUAL REFRESH ────────────────────────────────────────────
async function triggerRefresh() {
    if (_isRefreshingNow) return;
    _isRefreshingNow = true;
    const btn = document.querySelector('.refresh-btn');
    const msg = document.getElementById('stale-msg');
    btn.disabled = true;
    btn.textContent = '⏳ Starting refresh…';
    msg.textContent = 'Initializing data pipeline…';
    try {
        const res  = await fetch('/api/refresh', {method:'POST'});
        const data = await res.json();
        if (data.status === 'refresh_started' || data.status === 'success') {
            btn.textContent = '⏳ Refreshing… (2–5 min)';
            msg.textContent = 'Pipeline running in background…';
            pollRefreshCompletion();
        } else {
            btn.disabled = false;
            btn.textContent = '↻ Retry';
            msg.textContent = 'Failed: ' + (data.message || 'Check server terminal.');
        }
    } catch(e) {
        btn.disabled = false;
        btn.textContent = '↻ Retry';
        msg.textContent = 'Network error.';
    }
}

function pollRefreshCompletion() {
    const btn = document.querySelector('.refresh-btn');
    const msg = document.getElementById('stale-msg');
    const pollInterval = setInterval(async () => {
        try {
            const hRes = await fetch('/api/health');
            const hData = await hRes.json();
            if (!hData.refreshing_daily) {
                clearInterval(pollInterval);
                if (btn) btn.textContent = '✓ Update Complete';
                if (msg) msg.textContent = 'Data is now up to date! Reloading dashboard...';
                setTimeout(() => location.reload(), 1500);
            }
        } catch (e) {
            console.warn('Poll error during refresh', e);
        }
    }, 5000);
}

// ── TIMEFRAME CLOCK & SESSIONS ────────────────────────────────
function updateClock() {
    const elTime = document.getElementById('tf-time');
    const elLabel = document.getElementById('tf-label');
    if (!elTime || !elLabel) return;
    const now = new Date();
    if (_timeframe === 'NY') {
        elLabel.textContent = 'NY';
        elTime.textContent = now.toLocaleTimeString('en-US', { timeZone: 'America/New_York', hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
    } else {
        elLabel.textContent = 'LOCAL';
        elTime.textContent = now.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
    }
    updateMarketSessions();
}

function toggleTimeframe() {
    _timeframe = _timeframe === 'NY' ? 'LOCAL' : 'NY';
    updateClock();
    const widget = document.getElementById('timeframe-widget');
    if (widget) { widget.classList.add('flash'); setTimeout(() => widget.classList.remove('flash'), 300); }
}

function updateMarketSessions() {
    const now = new Date();
    const utcHours = now.getUTCHours();
    const sessions = [
        { id: 'sess-asian', start: 0, end: 9 },
        { id: 'sess-london', start: 8, end: 16 },
        { id: 'sess-ny', start: 13, end: 21 }
    ];
    sessions.forEach(s => {
        const item = document.getElementById(s.id);
        if (!item) return;
        const dot = item.querySelector('.sess-dot');
        const status = item.querySelector('.sess-status');
        let isOpen = utcHours >= s.start && utcHours < s.end;
        if (isOpen) {
            dot.className = 'sess-dot open';
            status.className = 'sess-status open';
            status.textContent = 'OPEN';
        } else {
            dot.className = 'sess-dot closed';
            status.className = 'sess-status closed';
            status.textContent = 'CLOSED';
        }
    });
}

// ── LIVE RISK MONITOR & ALERTS ────────────────────────────────
async function updateRiskAlerts(livePrice) {
    const elState = document.getElementById('monitor-state');
    const elAlert = document.getElementById('monitor-alert');
    const elStatus = document.getElementById('monitor-status');
    const elIcon = document.getElementById('monitor-icon');
    if (!elState || !elAlert || !elStatus) return;

    if (!_data) {
        elStatus.className = 'monitor-status';
        elState.className = 'monitor-state';
        elState.textContent = 'STANDBY';
        elAlert.textContent = 'Awaiting initial market connection...';
        return;
    }

    const sig = _data.prediction?.signal || 'NEUTRAL';
    if (sig === 'NEUTRAL') {
        elStatus.className = 'monitor-status';
        elState.className = 'monitor-state';
        if (elIcon) elIcon.textContent = '🛡️';
        elState.textContent = 'STANDBY';

        // Show what scalp/swing say when primary is NEUTRAL
        const scalpSig = _data.scalp?.signal || 'NEUTRAL';
        const swingSig = _data.swing?.signal || 'NEUTRAL';
        if (scalpSig === 'NEUTRAL' && swingSig === 'NEUTRAL') {
            elAlert.textContent = 'No active trade signal on either timeframe. Staying flat is recommended. Check Smart Timing for when to expect the next signal.';
        } else {
            elAlert.textContent = `Scalp: ${scalpSig} | Swing: ${swingSig} — Monitor active timeframes for execution.`;
        }
        return;
    }

    const rm = _data.risk_management || {};
    const latestClose = rm.latest_close || rm.entry_price || 0;
    const currentPrice = livePrice || rm.entry_price || 0;
    
    if (latestClose === 0 || currentPrice === 0) {
        elStatus.className = 'monitor-status';
        elState.className = 'monitor-state';
        elState.textContent = 'STANDBY';
        elAlert.textContent = 'Standing by for live price feed.';
        return;
    }

    const priceChangePct = (currentPrice - latestClose) / latestClose;
    
    let hasHighImpactToday = false;
    let macroEventName = '';
    try {
        const calRes = await fetch('/api/macro-calendar');
        const calData = await calRes.json();
        if (calData.status === 'success' && calData.events?.length) {
            const todayEvent = calData.events.find(e => e.days_until <= 0 && e.impact === 'HIGH');
            if (todayEvent) { hasHighImpactToday = true; macroEventName = todayEvent.event; }
        }
    } catch (e) { console.warn('Failed to load macro calendar for alerts', e); }

    let action = 'HOLD';
    let reason = 'Position is within normal parameters. Risk levels are stable.';
    let icon = '🛡️';
    let statusClass = 'monitor-status';
    let stateClass = 'monitor-state';

    if (sig === 'LONG') {
        if (priceChangePct <= -0.01) {
            action = 'CLOSE'; reason = `🚨 CLOSE: Gold dropped significantly (-${(Math.abs(priceChangePct)*100).toFixed(2)}%). Exit immediately.`; icon = '🚨'; statusClass = 'monitor-status alert-close'; stateClass = 'monitor-state alert-close';
        } else if (priceChangePct <= -0.005) {
            action = 'CLOSE'; reason = `🚨 CLOSE: Gold declined ${(Math.abs(priceChangePct)*100).toFixed(2)}%${hasHighImpactToday ? ` + macro event (${macroEventName})` : ''}. Exit position.`; icon = '🚨'; statusClass = 'monitor-status alert-close'; stateClass = 'monitor-state alert-close';
        } else if (priceChangePct >= 0.005) {
            action = 'BE'; reason = `⚠️ MOVE SL TO BE: Gold +${(priceChangePct*100).toFixed(2)}%. Lock in break-even.`; icon = '⚠️'; statusClass = 'monitor-status alert-be'; stateClass = 'monitor-state alert-be';
        } else if (hasHighImpactToday) {
            action = 'BE'; reason = `⚠️ MOVE SL TO BE: Macro event (${macroEventName}) today. Secure break-even.`; icon = '⚠️'; statusClass = 'monitor-status alert-be'; stateClass = 'monitor-state alert-be';
        }
    } else if (sig === 'SHORT') {
        if (priceChangePct >= 0.01) {
            action = 'CLOSE'; reason = `🚨 CLOSE: Gold rallied +${(priceChangePct*100).toFixed(2)}%. Exit immediately.`; icon = '🚨'; statusClass = 'monitor-status alert-close'; stateClass = 'monitor-state alert-close';
        } else if (priceChangePct >= 0.005) {
            action = 'CLOSE'; reason = `🚨 CLOSE: Gold advanced ${(priceChangePct*100).toFixed(2)}%${hasHighImpactToday ? ` + macro event (${macroEventName})` : ''}. Exit position.`; icon = '🚨'; statusClass = 'monitor-status alert-close'; stateClass = 'monitor-state alert-close';
        } else if (priceChangePct <= -0.005) {
            action = 'BE'; reason = `⚠️ MOVE SL TO BE: Position in profit +${(Math.abs(priceChangePct)*100).toFixed(2)}%. Secure break-even.`; icon = '⚠️'; statusClass = 'monitor-status alert-be'; stateClass = 'monitor-state alert-be';
        } else if (hasHighImpactToday) {
            action = 'BE'; reason = `⚠️ MOVE SL TO BE: Macro event (${macroEventName}) today. Secure break-even.`; icon = '⚠️'; statusClass = 'monitor-status alert-be'; stateClass = 'monitor-state alert-be';
        }
    }

    elStatus.className = statusClass;
    elState.className = stateClass;
    if (elIcon) elIcon.textContent = icon;
    elState.textContent = action;
    elAlert.textContent = reason;
}

// ── HISTORY TAB SWITCH ────────────────────────────────────
function switchHistoryTab(tf) {
    document.querySelectorAll('.htab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.history-panel').forEach(p => p.classList.remove('active'));
    document.getElementById(`htab-${tf}`)?.classList.add('active');
    document.getElementById(`history-panel-${tf}`)?.classList.add('active');
}

// ── HISTORY ──────────────────────────────────────────────
let _historyLoaded = false;
async function loadHistory() {
    if (_historyLoaded) return;
    const scalpTbody = document.getElementById('history-tbody-scalp');
    const swingTbody = document.getElementById('history-tbody-swing');
    if (!scalpTbody || !swingTbody) return;

    const loadMsg = `<tr><td colspan="7" style="text-align:center;padding:20px;color:var(--text-2);">Fetching history...</td></tr>`;
    scalpTbody.innerHTML = loadMsg;
    swingTbody.innerHTML = loadMsg;

    try {
        const res  = await fetch('/api/history');
        const data = await res.json();
        if (data.status !== 'success') throw new Error(data.message);

        const fmt2 = v => v ? `$${parseFloat(v).toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2})}` : '—';

        function renderHistoryPanel(rows, tbodyId, statsId) {
            const tbody  = document.getElementById(tbodyId);
            const statEl = document.getElementById(statsId);
            if (!rows || !rows.length) {
                tbody.innerHTML = `<tr><td colspan="7" style="text-align:center;padding:20px;color:var(--text-2);">No history found.</td></tr>`;
                return;
            }
            const wins   = rows.filter(r => r.result === 'WIN');
            const longT  = rows.filter(r => r.signal === 'LONG');
            const shortT = rows.filter(r => r.signal === 'SHORT');
            const longW  = longT.filter(r => r.result === 'WIN');
            const shortW = shortT.filter(r => r.result === 'WIN');
            const wr     = rows.length ? (wins.length / rows.length * 100).toFixed(1) : '0.0';
            const lwrStr = longT.length  ? `${(longW.length  / longT.length  * 100).toFixed(1)}%` : 'N/A';
            const swrStr = shortT.length ? `${(shortW.length / shortT.length * 100).toFixed(1)}%` : 'N/A';
            const strongR = rows.filter(r => r.strength === 'STRONG');
            const strongW = strongR.filter(r => r.result === 'WIN');
            const strongWR = strongR.length ? `${(strongW.length/strongR.length*100).toFixed(1)}%` : 'N/A';

            if (statEl) {
                const wrColor = parseFloat(wr) >= 55 ? 'text-green' : parseFloat(wr) < 45 ? 'text-red' : 'text-amber';
                statEl.innerHTML = `
                    <div class="hstat-item"><div class="hstat-label">Win Rate</div><div class="hstat-val ${wrColor}">${wr}%</div></div>
                    <div class="hstat-item"><div class="hstat-label">Total Signals</div><div class="hstat-val">${rows.length}</div></div>
                    <div class="hstat-item"><div class="hstat-label">LONG W/R</div><div class="hstat-val text-green">${lwrStr}</div></div>
                    <div class="hstat-item"><div class="hstat-label">SHORT W/R</div><div class="hstat-val text-red">${swrStr}</div></div>
                    <div class="hstat-item"><div class="hstat-label">STRONG W/R</div><div class="hstat-val text-amber">${strongWR}</div></div>`;
            }

            const STR_COLORS = { STRONG:'var(--green)', MODERATE:'var(--amber)', WEAK:'var(--text-3)' };
            tbody.innerHTML = rows.map(row => {
                const resClass = row.result==='WIN' ? 'res-win' : 'res-loss';
                const sigClass = row.signal==='LONG' ? 'text-green' : 'text-red';
                const strCol   = STR_COLORS[row.strength] || 'var(--text-3)';
                return `<tr>
                    <td>${row.date}</td>
                    <td class="${sigClass}"><b>${row.signal}</b></td>
                    <td style="color:${strCol};font-size:0.75rem;">${row.strength||'—'}</td>
                    <td>${row.probability}%</td>
                    <td style="color:var(--red);font-family:var(--mono);font-size:0.78rem;">${fmt2(row.stop_loss)}</td>
                    <td style="color:var(--green);font-family:var(--mono);font-size:0.78rem;">${fmt2(row.take_profit)}</td>
                    <td class="${resClass}">${row.result}</td>
                </tr>`;
            }).join('');
        }

        renderHistoryPanel(data.scalp, 'history-tbody-scalp', 'history-stats-scalp');
        renderHistoryPanel(data.swing, 'history-tbody-swing', 'history-stats-swing');
        _historyLoaded = true;

    } catch (e) {
        const errMsg = `<tr><td colspan="7" style="text-align:center;padding:20px;color:var(--red);">Error: ${e.message}</td></tr>`;
        document.getElementById('history-tbody-scalp').innerHTML = errMsg;
        document.getElementById('history-tbody-swing').innerHTML = errMsg;
    }
}


// ── BOOT ──────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('page-tabs').style.display = 'none';
    initialLoad();
    setInterval(updateClock, 1000);
    updateClock();
});
