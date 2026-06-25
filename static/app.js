// ============================================================
// XAUUSD AI Terminal — app.js v4  (multi-page)
// ============================================================

const POLL_MS       = 60_000;
const REFRESH_SEC   = 900;   // 15 min server cycle

let _data           = null;   // full /api/predict payload
let _newsAll        = [];     // all news with sentiment
let _currentFilter  = 'ALL';
let _prevSignal     = null;
let _cdRemaining    = REFRESH_SEC;
let _cdInterval     = null;
let _pollTimer      = null;
let _timeframe      = 'NY';

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
    // Lazy-render news/calendar on first visit
    if (name === 'news'     && _data) renderFullNews(_newsAll);
    if (name === 'calendar') loadCalendar();
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
    if (scroll.innerHTML === newHTML) return; // Keep continuous smooth scroll without jump restarts
    
    scroll.innerHTML = newHTML;
    scroll.style.animation = 'none';
    scroll.offsetHeight;
    const dur = Math.max(30, Math.floor(scroll.scrollWidth / 2 / 60));
    scroll.style.animation = `tickerScroll ${dur}s linear infinite`;
}

// ── SIGNAL CARD ───────────────────────────────────────────────
function renderSignalCard(data) {
    const sig = data.prediction?.signal || data.signal || 'NEUTRAL';
    const probUp = data.prediction?.probability_up ?? 0.5;

    // Flash if changed
    const card = document.getElementById('signal-card');
    if (_prevSignal && _prevSignal !== sig) {
        card.classList.add('signal-changed');
        setTimeout(() => card.classList.remove('signal-changed'), 1200);
    }
    _prevSignal = sig;

    card.className = `card card-signal signal-${sig}`;
    const icons = {LONG:'▲', SHORT:'▼', NEUTRAL:'◈'};
    document.getElementById('signal-icon').textContent  = icons[sig]||'◈';
    document.getElementById('signal-value').textContent = sig;
    document.getElementById('signal-sub-text').textContent = data.narrative?.summary || '';
    document.getElementById('prob-up-val').textContent  = pct(probUp);

    // Signal pill in top bar
    const pill = document.getElementById('signal-pill');
    if (pill) {
        pill.textContent  = sig;
        pill.className    = `signal-pill pill-${sig}`;
    }

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
}

// ── EXECUTION + LEVELS ────────────────────────────────────────
function renderExecution(data) {
    const rm  = data.risk_management || {};
    const lv  = data.intraday_levels || {};
    const sig = data.prediction?.signal || 'NEUTRAL';
    document.getElementById('ticker-price').textContent = fmt(rm.entry_price||0);
    document.getElementById('top-date').textContent     = data.target_date || data.date || '—';
    document.getElementById('entry-val').textContent    = fmt(rm.entry_price||0);
    document.getElementById('atr-val').textContent      = fmt(rm.atr_14||0);
    if (sig==='LONG'||sig==='SHORT') {
        document.getElementById('sl-val').textContent = fmt(rm.stop_loss||0);
        document.getElementById('tp-val').textContent = fmt(rm.take_profit||0);
    } else {
        document.getElementById('sl-val').textContent = '⏸ No Trade';
        document.getElementById('tp-val').textContent = '⏸ No Trade';
    }
    document.getElementById('r2-val').textContent = fmt(lv.r2||0);
    document.getElementById('r1-val').textContent = fmt(lv.r1||0);
    document.getElementById('pp-val').textContent = fmt(lv.pp||0);
    document.getElementById('s1-val').textContent = fmt(lv.s1||0);
    document.getElementById('s2-val').textContent = fmt(lv.s2||0);
}

// ── AI ANALYSIS PAGE ──────────────────────────────────────────
function renderAnalysis(data) {
    // Narrative
    const n = data.narrative;
    if (n) {
        document.getElementById('narrative-block').innerHTML = `
            <p class="narrative-reasoning">${n.reasoning}</p>
            <div class="narrative-risk">${n.risk_note}</div>`;
    }
    // SHAP
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
    // Model bars
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
    const filtered = _currentFilter === 'ALL' ? news : news.filter(n => n.sentiment === _currentFilter);
    const SC = {BULLISH:'var(--green)', BEARISH:'var(--red)', NEUTRAL:'var(--text-3)'};
    if (!filtered.length) {
        list.innerHTML = `<p style="color:var(--text-3);padding:20px;">No ${_currentFilter.toLowerCase()} headlines found.</p>`;
        return;
    }
    list.innerHTML = filtered.map(n => {
        const col = SC[n.sentiment]||'var(--text-3)';
        const tag = n.sentiment==='BULLISH'?'▲ BULLISH':n.sentiment==='BEARISH'?'▼ BEARISH':'— NEUTRAL';
        const dt  = n.datetime ? n.datetime.slice(0,16) : '';
        return `<a href="${n.url||'#'}" target="_blank" rel="noopener" class="news-item-full">
            <div class="nf-left">
                <div class="nf-headline">${n.headline}</div>
                <div class="nf-meta">
                    <span class="nf-source">${n.source}</span>
                    ${dt ? `<span class="nf-time">${dt}</span>` : ''}
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

    // Hide loader, show pages
    document.getElementById('loader').style.display = 'none';
    document.getElementById('page-tabs').style.display = 'flex';
    
    // Only activate page-dashboard if no page is currently active
    const activePage = document.querySelector('.page.active');
    if (!activePage) {
        document.getElementById('page-dashboard').classList.add('active');
    }

    // Stale banner
    const age = data.data_age_days ?? 0;
    const banner = document.getElementById('stale-banner');
    if (banner) {
        if (age > 1) {
            document.getElementById('stale-msg').textContent =
                `Data from ${data.date} (${age}d old). Click Refresh for today's forecast.`;
            banner.style.display = 'flex';
        } else {
            banner.style.display = 'none';
        }
    }

    // Update tab status
    const upd = document.getElementById('tab-update-label');
    if (upd && data.last_refresh) upd.textContent = `Updated ${data.last_refresh}`;

    // News badge
    const nb = document.getElementById('news-count-badge');
    if (nb) nb.textContent = _newsAll.length || '';

    renderSignalCard(data);
    renderExecution(data);
    renderAnalysis(data);
    buildTicker(_newsAll);
    updateRiskAlerts(data.risk_management?.entry_price);

    // News total label
    const ntl = document.getElementById('news-total-label');
    if (ntl) ntl.textContent = `${_newsAll.length} headlines`;

    // Live update news list page if active
    const newsPage = document.getElementById('page-news');
    if (newsPage && newsPage.classList.contains('active')) {
        renderFullNews(_newsAll);
    }
}

// ── LIGHTWEIGHT POLL ──────────────────────────────────────────
async function pollSignal() {
    try {
        const res = await fetch('/api/predict');
        const data = await res.json();

        if (data.status === 'success') {
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
        const [pr, nr] = await Promise.all([
            fetch('/api/predict'),
            fetch('/api/live-news'),
        ]);
        const data = await pr.json();
        const news = await nr.json();

        if (data.status !== 'success') {
            document.getElementById('loader').innerHTML =
                `<div class="loader-inner"><p style="color:var(--red)">⚠ API Error: ${data.message}</p></div>`;
            return;
        }

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
    const btn = document.querySelector('.refresh-btn');
    const msg = document.getElementById('stale-msg');
    btn.disabled = true;
    btn.textContent = '⏳ Refreshing… (2–5 min)';
    msg.textContent = 'Running data pipeline…';
    try {
        const res  = await fetch('/api/refresh', {method:'POST'});
        const data = await res.json();
        if (data.status === 'success') {
            btn.textContent = '✓ Reloading...';
            setTimeout(() => location.reload(), 1500);
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

// ── TIMEFRAME CLOCK & SESSIONS ────────────────────────────────
function updateClock() {
    const elTime = document.getElementById('tf-time');
    const elLabel = document.getElementById('tf-label');
    if (!elTime || !elLabel) return;

    const now = new Date();
    if (_timeframe === 'NY') {
        elLabel.textContent = 'NY';
        elTime.textContent = now.toLocaleTimeString('en-US', {
            timeZone: 'America/New_York',
            hour12: false,
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        });
    } else {
        elLabel.textContent = 'LOCAL';
        elTime.textContent = now.toLocaleTimeString('en-US', {
            hour12: false,
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        });
    }
    updateMarketSessions();
}

function toggleTimeframe() {
    _timeframe = _timeframe === 'NY' ? 'LOCAL' : 'NY';
    updateClock();
    const widget = document.getElementById('timeframe-widget');
    if (widget) {
        widget.classList.add('flash');
        setTimeout(() => widget.classList.remove('flash'), 300);
    }
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
        
        let isOpen = false;
        if (s.start <= s.end) {
            isOpen = utcHours >= s.start && utcHours < s.end;
        } else {
            isOpen = utcHours >= s.start || utcHours < s.end;
        }
        
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
        elAlert.textContent = 'No active trade signal. Staying flat is recommended.';
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
    
    // Fetch macro calendar for high impact events today
    let hasHighImpactToday = false;
    let macroEventName = '';
    try {
        const calRes = await fetch('/api/macro-calendar');
        const calData = await calRes.json();
        if (calData.status === 'success' && calData.events?.length) {
            const todayEvent = calData.events.find(e => e.days_until <= 0 && e.impact === 'HIGH');
            if (todayEvent) {
                hasHighImpactToday = true;
                macroEventName = todayEvent.event;
            }
        }
    } catch (e) {
        console.warn('Failed to load macro calendar for alerts', e);
    }

    let action = 'HOLD';
    let reason = 'Position is within normal parameters. Risk levels are stable.';
    let icon = '🛡️';
    let statusClass = 'monitor-status';
    let stateClass = 'monitor-state';

    if (sig === 'LONG') {
        if (priceChangePct <= -0.01) {
            action = 'CLOSE';
            reason = `🚨 CLOSE POSITION: Gold has dropped significantly (-${(Math.abs(priceChangePct)*100).toFixed(2)}%) below yesterday's close, indicating massive bearish intraday momentum. Exit immediately to protect capital.`;
            icon = '🚨';
            statusClass = 'monitor-status alert-close';
            stateClass = 'monitor-state alert-close';
        } else if (priceChangePct <= -0.005) {
            if (hasHighImpactToday) {
                action = 'CLOSE';
                reason = `🚨 CLOSE POSITION: High-impact macro news event (${macroEventName}) is scheduled for today, and the market is moving against our LONG position (-${(Math.abs(priceChangePct)*100).toFixed(2)}%). Exit before high volatility spikes.`;
                icon = '🚨';
                statusClass = 'monitor-status alert-close';
                stateClass = 'monitor-state alert-close';
            } else {
                action = 'CLOSE';
                reason = `🚨 CLOSE POSITION: Gold has declined by ${(Math.abs(priceChangePct)*100).toFixed(2)}% intraday. Technical indicators support a continued correction.`;
                icon = '🚨';
                statusClass = 'monitor-status alert-close';
                stateClass = 'monitor-state alert-close';
            }
        } else if (priceChangePct >= 0.005) {
            action = 'BE';
            reason = `⚠️ MOVE SL TO BE: Gold has risen to $${currentPrice.toLocaleString('en-US', {minimumFractionDigits: 2})} (+${(priceChangePct*100).toFixed(2)}% in profit). Lock in break-even stop loss to guarantee a risk-free position.`;
            icon = '⚠️';
            statusClass = 'monitor-status alert-be';
            stateClass = 'monitor-state alert-be';
        } else if (hasHighImpactToday) {
            action = 'BE';
            reason = `⚠️ MOVE SL TO BE: High-impact macro news (${macroEventName}) is scheduled for today. Move Stop Loss to break-even to hedge against immediate news-driven spread jumps.`;
            icon = '⚠️';
            statusClass = 'monitor-status alert-be';
            stateClass = 'monitor-state alert-be';
        }
    } else if (sig === 'SHORT') {
        if (priceChangePct >= 0.01) {
            action = 'CLOSE';
            reason = `🚨 CLOSE POSITION: Gold has rallied significantly (+${(priceChangePct*100).toFixed(2)}%) above yesterday's close, indicating strong bullish intraday momentum. Exit immediately.`;
            icon = '🚨';
            statusClass = 'monitor-status alert-close';
            stateClass = 'monitor-state alert-close';
        } else if (priceChangePct >= 0.005) {
            if (hasHighImpactToday) {
                action = 'CLOSE';
                reason = `🚨 CLOSE POSITION: High-impact macro news event (${macroEventName}) is scheduled for today, and the market is moving against our SHORT position (+${(priceChangePct*100).toFixed(2)}%). Exit to manage risk.`;
                icon = '🚨';
                statusClass = 'monitor-status alert-close';
                stateClass = 'monitor-state alert-close';
            } else {
                action = 'CLOSE';
                reason = `🚨 CLOSE POSITION: Gold has advanced by ${(priceChangePct*100).toFixed(2)}% against our SHORT target. Exit position.`;
                icon = '🚨';
                statusClass = 'monitor-status alert-close';
                stateClass = 'monitor-state alert-close';
            }
        } else if (priceChangePct <= -0.005) {
            action = 'BE';
            reason = `⚠️ MOVE SL TO BE: Position is in solid profit (+${(Math.abs(priceChangePct)*100).toFixed(2)}% short return). Move Stop Loss to entry (Break-Even) immediately.`;
            icon = '⚠️';
            statusClass = 'monitor-status alert-be';
            stateClass = 'monitor-state alert-be';
        } else if (hasHighImpactToday) {
            action = 'BE';
            reason = `⚠️ MOVE SL TO BE: High-impact macro news (${macroEventName}) is scheduled for today. Secure break-even stop loss before release time.`;
            icon = '⚠️';
            statusClass = 'monitor-status alert-be';
            stateClass = 'monitor-state alert-be';
        }
    }

    elStatus.className = statusClass;
    elState.className = stateClass;
    if (elIcon) elIcon.textContent = icon;
    elState.textContent = action;
    elAlert.textContent = reason;
}

// ── BOOT ──────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('page-tabs').style.display = 'none';
    initialLoad();
    setInterval(updateClock, 1000);
    updateClock();
});
