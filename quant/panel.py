"""iPad paneli: botun durumunu gösterir, duraklat / devam / hepsini kapat komutlarını alır.

Standart kütüphaneyle çalışır. Her istek QUANT_PANEL_TOKEN ister (Authorization: Bearer ...).
iPad'de ilk açılış: http://<adres>:8787/?t=<token>  → token tarayıcıda saklanır, adres çubuğundan silinir.
Safari → Paylaş → "Ana Ekrana Ekle" ile uygulama gibi açılır.
"""
import hmac
import json
import logging
import os
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

log = logging.getLogger("quant.panel")


def panel_token(eng):
    if eng.cfg.panel_token:
        return eng.cfg.panel_token
    path = os.path.join(eng.cfg.state_dir, "panel_token")
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        tok = secrets.token_urlsafe(24)
        with open(os.open(path, os.O_WRONLY | os.O_CREAT, 0o600), "w") as f:
            f.write(tok)
        return tok


def make_handler(eng, token):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json"):
            data = body.encode() if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype + "; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _authed(self):
            got = self.headers.get("Authorization", "").removeprefix("Bearer ").strip()
            return bool(got) and hmac.compare_digest(got, token)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                return self._send(200, PAGE, "text/html")
            if path == "/api/status":
                if not self._authed():
                    return self._send(401, '{"error":"token"}')
                return self._send(200, json.dumps(eng.status()))
            self._send(404, '{"error":"yok"}')

        def do_POST(self):
            if self.path != "/api/cmd":
                return self._send(404, '{"error":"yok"}')
            if not self._authed():
                return self._send(401, '{"error":"token"}')
            try:
                n = min(int(self.headers.get("Content-Length") or 0), 1000)
                cmd = json.loads(self.rfile.read(n) or b"{}").get("cmd")
                eng.command(cmd)
            except (ValueError, AttributeError):
                return self._send(400, '{"error":"geçersiz komut"}')
            log.info("panel komutu: %s", cmd)
            self._send(200, '{"ok":true}')

    return H


def serve(eng):
    token = panel_token(eng)
    srv = ThreadingHTTPServer((eng.cfg.panel_host, eng.cfg.panel_port), make_handler(eng, token))
    log.info("panel: http://%s:%s/?t=<token>  (token: %s dosyasında veya QUANT_PANEL_TOKEN)",
             eng.cfg.panel_host, eng.cfg.panel_port, os.path.join(eng.cfg.state_dir, "panel_token"))
    srv.serve_forever()


PAGE = r"""<!doctype html>
<html lang="tr"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="Quant">
<title>Quant Panel</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--fg:#14171c;--mut:#667085;--line:#e4e7ec;--up:#0f8a4b;--down:#c8322f;--acc:#2f5bea;--warn:#b7791f}
@media (prefers-color-scheme:dark){:root{--bg:#0e1116;--card:#171b22;--fg:#e7eaf0;--mut:#8b93a3;--line:#262c36;--up:#3ecf7f;--down:#ff6b64;--acc:#6f8fff;--warn:#e3a53a}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 -apple-system,BlinkMacSystemFont,"SF Pro Text",system-ui,sans-serif;
 padding:max(16px,env(safe-area-inset-top)) 16px 32px}
main{max-width:1000px;margin:0 auto;display:grid;gap:14px}
header{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
h1{font-size:20px;margin:0 auto 0 0}
.badge{font-size:12px;font-weight:700;padding:3px 9px;border-radius:99px;border:1px solid var(--line)}
.live{background:var(--down);color:#fff;border-color:transparent}.paper{color:var(--acc)}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px 16px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:14px}
.k small{color:var(--mut);display:block;font-size:12px}.k b{font-size:22px;font-variant-numeric:tabular-nums}
.up{color:var(--up)}.down{color:var(--down)}.mut{color:var(--mut)}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:14px}
th{text-align:left;color:var(--mut);font-weight:500;font-size:12px}th,td{padding:7px 6px;border-bottom:1px solid var(--line)}
.scroll{overflow-x:auto}
.btns{display:flex;gap:10px;flex-wrap:wrap}
button{font:inherit;font-weight:600;border:0;border-radius:10px;padding:11px 16px;background:var(--acc);color:#fff;cursor:pointer}
button.sec{background:transparent;color:var(--fg);border:1px solid var(--line)}button.danger{background:var(--down)}
svg{width:100%;height:120px;display:block}
.ev{font-size:13px;max-height:260px;overflow:auto}.ev div{padding:4px 0;border-bottom:1px solid var(--line)}
h2{font-size:14px;margin:0 0 8px;color:var(--mut);font-weight:600}
#login{display:none}input{font:inherit;padding:10px;border-radius:10px;border:1px solid var(--line);background:var(--bg);color:var(--fg);width:100%;margin:8px 0}
</style></head><body><main>
<header><h1>Quant</h1><span id="mode" class="badge">…</span><span id="st" class="badge">…</span></header>
<section id="login" class="card"><h2>Panel anahtarı</h2><input id="tok" type="password" placeholder="QUANT_PANEL_TOKEN">
<button onclick="setTok(document.getElementById('tok').value)">Giriş</button></section>
<section class="kpis">
 <div class="card k"><small>Sermaye (USDT)</small><b id="eq">–</b></div>
 <div class="card k"><small>Bugün</small><b id="day">–</b></div>
 <div class="card k"><small>Kapanan işlemler</small><b id="tr">–</b></div>
 <div class="card k"><small>Kazanma oranı</small><b id="wr">–</b></div>
</section>
<section class="card"><h2>Sermaye eğrisi</h2><svg id="chart" viewBox="0 0 600 120" preserveAspectRatio="none"></svg></section>
<section class="card"><h2>Açık pozisyonlar</h2><div class="scroll"><table><thead><tr><th>Sembol</th><th>Yön</th><th>Adet</th><th>Giriş</th><th>Fiyat</th><th>Stop</th><th>Açılış</th></tr></thead><tbody id="pos"></tbody></table></div></section>
<section class="card btns">
 <button class="sec" onclick="cmd('pause')">Duraklat</button>
 <button onclick="cmd('resume')">Devam</button>
 <button class="danger" onclick="if(confirm('Tüm pozisyonlar piyasa fiyatından kapatılsın ve bot duraklatılsın mı?'))cmd('close_all')">Hepsini kapat</button>
</section>
<section class="card"><h2>Son işlemler</h2><div class="scroll"><table><thead><tr><th>Kapanış</th><th>Sembol</th><th>Yön</th><th>Giriş → Çıkış</th><th>PnL</th><th>Neden</th></tr></thead><tbody id="trades"></tbody></table></div></section>
<section class="card"><h2>Olaylar</h2><div id="ev" class="ev"></div></section>
<p class="mut" id="cfg" style="font-size:12px"></p>
</main><script>
const q=new URLSearchParams(location.search);let T='';try{T=localStorage.getItem('qt')||''}catch(e){}
if(q.get('t')){T=q.get('t');try{localStorage.setItem('qt',T)}catch(e){}history.replaceState(null,'',location.pathname)}
function setTok(v){T=v.trim();try{localStorage.setItem('qt',T)}catch(e){}load()}
const $=id=>document.getElementById(id),f=(x,d=2)=>x==null?'–':Number(x).toLocaleString('tr-TR',{maximumFractionDigits:d,minimumFractionDigits:d});
const tm=t=>t?new Date(t*1000).toLocaleString('tr-TR',{day:'2-digit',month:'2-digit',hour:'2-digit',minute:'2-digit'}):'–';
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
async function load(){
 let r;try{r=await fetch('/api/status',{headers:{Authorization:'Bearer '+T}})}catch(e){$('st').textContent='bağlantı yok';return}
 if(r.status==401){$('login').style.display='block';return}$('login').style.display='none';
 const s=await r.json();
 $('mode').textContent=s.live?'CANLI':'PAPER';$('mode').className='badge '+(s.live?'live':'paper');
 $('st').textContent=s.halted?'günlük limit':s.paused?'duraklatıldı':'çalışıyor';
 $('eq').textContent=f(s.equity);
 const d=s.day_start_equity?s.equity-s.day_start_equity:0;$('day').textContent=(d>=0?'+':'')+f(d);$('day').className=d>=0?'up':'down';
 const tr=s.trades||[];$('tr').textContent=tr.length;$('wr').textContent=tr.length?Math.round(100*tr.filter(t=>t.pnl>0).length/tr.length)+'%':'–';
 const L=(s.equity_log||[]).slice(-600);if(L.length>1){const v=L.map(x=>x[1]),mn=Math.min(...v),mx=Math.max(...v),rg=mx-mn||1;
  const pts=v.map((y,i)=>`${(i/(v.length-1)*600).toFixed(1)},${(112-(y-mn)/rg*104).toFixed(1)}`).join(' ');
  $('chart').innerHTML=`<polyline fill="none" stroke="var(--acc)" stroke-width="2" vector-effect="non-scaling-stroke" points="${pts}"/>`}
 const P=Object.entries(s.positions||{});
 $('pos').innerHTML=P.length?P.map(([k,p])=>{const m=s.marks[k],u=m?(p.side=='long'?m-p.entry:p.entry-m):0;
  return `<tr><td>${esc(k)}</td><td class="${p.side=='long'?'up':'down'}">${p.side=='long'?'LONG':'SHORT'}</td><td>${p.vol}</td><td>${f(p.entry,4)}</td><td class="${u>=0?'up':'down'}">${f(m,4)}</td><td>${f(p.trail??p.stop,4)}</td><td>${tm(p.opened)}</td></tr>`}).join(''):'<tr><td colspan="7" class="mut">Açık pozisyon yok</td></tr>';
 $('trades').innerHTML=tr.slice(-30).reverse().map(t=>`<tr><td>${tm(t.closed)}</td><td>${esc(t.symbol)}</td><td>${t.side}</td><td>${f(t.entry,4)} → ${f(t.exit,4)}</td><td class="${t.pnl>=0?'up':'down'}">${f(t.pnl)}</td><td class="mut">${esc(t.reason)}</td></tr>`).join('')||'<tr><td colspan="6" class="mut">Henüz işlem yok</td></tr>';
 $('ev').innerHTML=(s.events||[]).slice().reverse().map(e=>`<div><span class="mut">${tm(e[0])}</span> ${esc(e[1])}</div>`).join('');
 const p=s.params;$('cfg').textContent=`${s.symbols.join(', ')} · ${s.interval} · ${s.leverage}x · risk %${s.risk_pct} · EMA ${p.fast}/${p.slow} trend ${p.trend} · stop ${p.stop_atr}×ATR, iz ${p.trail_atr}×ATR · günlük limit %${s.max_daily_loss_pct}`;
}
async function cmd(c){const r=await fetch('/api/cmd',{method:'POST',headers:{Authorization:'Bearer '+T,'Content-Type':'application/json'},body:JSON.stringify({cmd:c})});
 if(!r.ok)alert('Komut başarısız ('+r.status+')');setTimeout(load,1500)}
load();setInterval(load,10000);document.addEventListener('visibilitychange',()=>{if(!document.hidden)load()});
</script></body></html>
"""
