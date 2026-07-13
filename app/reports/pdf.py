"""
RailSafe HTML Report Generator
Real DB ma'lumotlaridan zamonaviy HTML hisobot yaratadi.
test_hsobot.py dizayniga mos — CSS kartalar, jadvallar, rang kodlari.
"""

import html
import logging
from datetime import datetime

from app.utils.language import t, LM

logger = logging.getLogger("RailSafe.reports")


def build_html_report(config_manager, stats_db,
                      date_from: str, date_to: str) -> str:
    """HTML string qaytaradi — PDF yoki preview uchun."""

    # HTML injection oldini olish uchun — config/DB dan kelgan har qanday
    # matnni interpolatsiyadan oldin escape qilamiz.
    def _e(s):
        return html.escape(str(s)) if s is not None else ""

    def _fmt(n):
        return f"{int(n):,}".replace(",", " ") if isinstance(n, (int, float)) else str(n)

    def _fmt_dur(secs):
        if not secs:
            return "—"
        m = int(secs) // 60
        s = int(secs) % 60
        return (f"{m} {t('unit.min')} {s:02d} {t('unit.sec')}"
                if m > 0 else f"{s} {t('unit.sec')}")

    def _fdate(d):
        try:
            return datetime.strptime(d, "%Y-%m-%d").strftime("%d.%m.%Y")
        except Exception:
            return d

    crossings = config_manager.get_crossings()
    total_cams = sum(len(cr.get("cameras", [])) for cr in crossings)

    grand_light = grand_heavy = grand_trains = 0
    cx_data = []

    for cr in crossings:
        cid = cr["id"]
        light, heavy = stats_db.get_date_range_total(cid, date_from, date_to)
        grand_light += light
        grand_heavy += heavy

        ts = stats_db.get_train_range_stats(cid, date_from, date_to)
        grand_trains += ts["count"]

        # Kamera satrlari
        cam_rows_html = ""
        for i, cam in enumerate(cr.get("cameras", [])):
            cn = cam.get("name", "?")
            ct = t("crossing.type.main") if cam.get("type") == "main" else t("crossing.type.additional")
            cl, ch = stats_db.get_date_range_camera(cid, cn, date_from, date_to)
            cam_rows_html += f"""
            <tr>
                <td>{i+1}</td>
                <td>{_e(cn)}</td>
                <td><span class="badge">{_e(ct)}</span></td>
                <td>{_fmt(cl)}</td>
                <td>{_fmt(ch)}</td>
                <td class="highlight">{_fmt(cl+ch)}</td>
            </tr>"""

        # Poyezd jadvali satrlari
        events = stats_db.get_train_events_range(cid, date_from, date_to)
        train_rows_html = ""
        for idx, ev in enumerate(events, 1):
            train_rows_html += f"""
            <tr>
                <td>{idx}</td>
                <td>{_e(ev.get('date',''))}</td>
                <td><strong>{_e(ev.get('start',''))}</strong></td>
                <td><strong>{_e(ev.get('end',''))}</strong></td>
                <td class="highlight">{_e(ev.get('duration_fmt',''))}</td>
            </tr>"""

        train_section = (f"""
        <div class="sub-title">🚂 {_e(t("rpt.train_pass_table"))}</div>
        <table>
            <thead><tr>
                <th>{_e(t("rpt.col_num"))}</th><th>{_e(t("rpt.col_date"))}</th><th>{_e(t("rpt.col_enter"))}</th><th>{_e(t("rpt.col_exit"))}</th><th>{_e(t("rpt.col_pass_time"))}</th>
            </tr></thead>
            <tbody>{train_rows_html}</tbody>
        </table>""") if events else f"""
        <div class='no-data'>{_e(t("rpt.no_train_data"))}</div>"""

        cx_data.append({
            "cr": cr, "light": light, "heavy": heavy, "total": light + heavy,
            "ts": ts, "cam_rows_html": cam_rows_html, "train_section": train_section,
        })

    # Kesishmalar qiyosiy jadvali satrlari
    cmp_rows = ""
    for i, d in enumerate(cx_data):
        cr = d["cr"]
        cmp_rows += f"""
        <tr>
            <td>{i+1}</td>
            <td><strong>{_e(cr.get('name',''))}</strong></td>
            <td>{_e(cr.get('location','—'))}</td>
            <td>{_fmt(d['light'])}</td>
            <td>{_fmt(d['heavy'])}</td>
            <td class="highlight">{_fmt(d['total'])}</td>
            <td>{len(cr.get('cameras',[]))}</td>
        </tr>"""

    # Har bir kesishma bo'limi
    sections_html = ""
    for d in cx_data:
        cr  = d["cr"]
        ts  = d["ts"]
        sections_html += f"""
        <div class="section">
            <div class="section-header">
                <div>
                    <div class="section-title">{_e(cr.get('name',''))}</div>
                    <div class="section-sub">📡 {_e(cr.get('location','—'))}</div>
                </div>
                <div class="stat-row">
                    <div class="mini-stat" style="border-color:#1d4ed8">
                        <div class="ms-val" style="color:#1d4ed8">{_fmt(d['total'])}</div>
                        <div class="ms-lbl">{_e(t("rpt.total"))}</div>
                    </div>
                    <div class="mini-stat" style="border-color:#10b981">
                        <div class="ms-val" style="color:#10b981">{_fmt(d['light'])}</div>
                        <div class="ms-lbl">{_e(t("rpt.light"))}</div>
                    </div>
                    <div class="mini-stat" style="border-color:#f59e0b">
                        <div class="ms-val" style="color:#f59e0b">{_fmt(d['heavy'])}</div>
                        <div class="ms-lbl">{_e(t("rpt.heavy"))}</div>
                    </div>
                </div>
            </div>

            <div class="sub-title">📷 {_e(t("rpt.cam_stats"))}</div>
            <table>
                <thead><tr>
                    <th>{_e(t("rpt.col_num"))}</th><th>{_e(t("rpt.col_camera"))}</th><th>{_e(t("rpt.col_type"))}</th>
                    <th>{_e(t("rpt.light"))}</th><th>{_e(t("rpt.heavy"))}</th><th>{_e(t("rpt.total"))}</th>
                </tr></thead>
                <tbody>{d['cam_rows_html']}</tbody>
            </table>

            <div class="sub-title" style="margin-top:18px">🚂 {_e(t("rpt.train_movement_stats"))}</div>
            <div class="stat-row" style="margin-bottom:14px">
                <div class="mini-stat" style="border-color:#7c3aed">
                    <div class="ms-val" style="color:#7c3aed">{ts['count']}</div>
                    <div class="ms-lbl">{_e(t("rpt.total_trains"))}</div>
                </div>
                <div class="mini-stat" style="border-color:#1d4ed8">
                    <div class="ms-val" style="color:#1d4ed8">{_fmt_dur(ts['avg']) if ts['count'] else '—'}</div>
                    <div class="ms-lbl">{_e(t("rpt.avg_time"))}</div>
                </div>
                <div class="mini-stat" style="border-color:#10b981">
                    <div class="ms-val" style="color:#10b981">{_fmt_dur(ts['min']) if ts['count'] else '—'}</div>
                    <div class="ms-lbl">{_e(t("rpt.min_time"))}</div>
                </div>
                <div class="mini-stat" style="border-color:#ef4444">
                    <div class="ms-val" style="color:#ef4444">{_fmt_dur(ts['max']) if ts['count'] else '—'}</div>
                    <div class="ms-lbl">{_e(t("rpt.max_time"))}</div>
                </div>
            </div>
            {d['train_section']}
        </div>"""

    period_str = f"{_fdate(date_from)} — {_fdate(date_to)}"
    created_str = datetime.now().strftime("%d.%m.%Y  %H:%M")

    return f"""<!DOCTYPE html>
<html lang="{_e(LM.current)}">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>{_e(t("rpt.doc_title"))}</title>
<style>
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ font-family:'Segoe UI',Arial,sans-serif; background:#f0f4f8; color:#1f2937; font-size:13px; }}

  .header {{
    background:#0d2b4e;
    color:#fff;
    -webkit-print-color-adjust:exact; print-color-adjust:exact;
  }}
  .header-top-bar {{
    background:#1a3d6b;
    padding:7px 36px;
    display:flex; justify-content:space-between; align-items:center;
    border-bottom:1px solid rgba(255,255,255,0.1);
    -webkit-print-color-adjust:exact; print-color-adjust:exact;
  }}
  .header-org-name {{
    font-size:11px; font-weight:700; letter-spacing:2.5px;
    text-transform:uppercase; color:#a8c8e8;
  }}
  .header-doc-num {{ font-size:10px; color:#7aaac8; letter-spacing:0.5px; }}
  .header-main {{
    padding:20px 36px 16px;
    display:flex; justify-content:space-between; align-items:flex-start; gap:16px;
  }}
  .header-left {{ flex:1; }}
  .header-system-badge {{
    display:inline-block;
    background:rgba(255,255,255,0.08);
    border:1px solid rgba(255,255,255,0.15);
    border-radius:4px;
    font-size:9px; letter-spacing:1.5px; text-transform:uppercase;
    color:#90b8d8; padding:3px 10px; margin-bottom:10px;
  }}
  .header-title {{
    font-size:19px; font-weight:800; letter-spacing:0.5px;
    line-height:1.3; color:#fff; text-transform:uppercase;
  }}
  .header-sub {{ font-size:12px; color:#90b8d8; margin-top:6px; }}
  .header-right {{
    text-align:right; min-width:200px;
    background:rgba(255,255,255,0.05);
    border:1px solid rgba(255,255,255,0.1);
    border-radius:6px; padding:12px 16px;
  }}
  .header-meta-row {{
    font-size:11px; color:#a8c8e8; line-height:2;
    display:flex; justify-content:space-between; gap:12px;
  }}
  .header-meta-label {{ color:#6a94b8; font-size:10px; text-transform:uppercase; letter-spacing:0.5px; }}
  .header-meta-val {{ font-weight:700; color:#fff; }}
  .header-bottom-bar {{
    background:#1558a0;
    padding:8px 36px;
    display:flex; gap:20px; align-items:center;
    border-top:1px solid rgba(255,255,255,0.1);
    -webkit-print-color-adjust:exact; print-color-adjust:exact;
  }}
  .hb-item {{
    font-size:11px; color:rgba(255,255,255,0.85);
    display:flex; align-items:center; gap:6px;
  }}
  .hb-item::before {{
    content:''; display:inline-block;
    width:6px; height:6px; border-radius:50%;
    background:#60a5fa; flex-shrink:0;
    -webkit-print-color-adjust:exact; print-color-adjust:exact;
  }}
  .hb-sep {{ color:rgba(255,255,255,0.2); font-size:16px; }}

  .content {{ max-width:960px; margin:0 auto; padding:16px 20px 40px; }}

  .sec-title {{
    display:flex; align-items:center; gap:10px;
    margin:24px 0 12px;
  }}
  .sec-title::before {{
    content:''; display:block; width:4px; height:20px;
    background:#1d4ed8; border-radius:2px;
    -webkit-print-color-adjust:exact; print-color-adjust:exact;
  }}
  .sec-title span {{ font-size:14px; font-weight:700; color:#1e3a5f; }}

  .cards {{ display:flex; gap:12px; flex-wrap:wrap; }}
  .card {{
    flex:1; min-width:120px; background:#fff;
    border:1px solid #e5e7eb; border-radius:10px;
    padding:16px 18px;
    -webkit-print-color-adjust:exact; print-color-adjust:exact;
  }}
  .card-val {{ font-size:26px; font-weight:800; color:#111827; }}
  .card-lbl {{ font-size:10px; color:#6b7280; font-weight:600; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:4px; }}

  .section {{
    background:#fff; border:1px solid #e5e7eb;
    border-radius:10px; padding:20px 22px; margin-bottom:16px;
  }}
  .section-header {{ display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:10px; margin-bottom:16px; }}
  .section-title {{ font-size:16px; font-weight:700; color:#1e3a5f; }}
  .section-sub {{ font-size:11px; color:#6b7280; margin-top:3px; }}
  .sub-title {{ font-size:12px; font-weight:700; color:#374151; margin-bottom:8px; }}

  .stat-row {{ display:flex; gap:10px; flex-wrap:wrap; }}
  .mini-stat {{
    flex:1; min-width:90px; background:#f8fafc;
    border:1px solid; border-radius:8px; padding:8px 12px;
    text-align:center;
    -webkit-print-color-adjust:exact; print-color-adjust:exact;
  }}
  .ms-val {{ font-size:17px; font-weight:800; }}
  .ms-lbl {{ font-size:10px; color:#6b7280; font-weight:600; margin-top:2px; }}

  table {{ width:100%; border-collapse:collapse; font-size:12px; }}
  thead tr {{ background:#f1f5f9; -webkit-print-color-adjust:exact; print-color-adjust:exact; }}
  th {{ padding:9px 12px; text-align:left; font-weight:700; color:#374151; border-bottom:2px solid #e5e7eb; white-space:nowrap; }}
  td {{ padding:8px 12px; border-bottom:1px solid #f3f4f6; color:#374151; }}
  tr:nth-child(even) td {{ background:#f9fafb; -webkit-print-color-adjust:exact; print-color-adjust:exact; }}
  .highlight {{ color:#1d4ed8; font-weight:700; }}
  .badge {{ background:#dbeafe; color:#1d4ed8; font-size:10px; padding:2px 8px; border-radius:20px; font-weight:600; }}
  .no-data {{ text-align:center; color:#9ca3af; padding:16px; background:#f9fafb; border-radius:8px; font-size:12px; }}
  .footer {{ display:none; }}

  @media print {{
    @page {{ size:A4; margin:12mm 10mm; }}
    body {{ background:#fff; font-size:11px; }}
    .content {{ max-width:100%; padding:0; }}
    .header-top-bar {{ padding:5px 18px; }}
    .header-main {{ padding:12px 18px 10px; }}
    .header-bottom-bar {{ padding:6px 18px; }}
    .header-title {{ font-size:15px; }}
    .header-right {{ padding:8px 10px; min-width:160px; }}
    .cards {{ gap:8px; }}
    .card {{ padding:10px 12px; min-width:80px; }}
    .card-val {{ font-size:18px; }}
    /* --- Sahifa uzilish nazorati --- */
    .section {{
      padding:12px 14px; margin-bottom:10px;
      break-inside:auto;
    }}
    .section-header {{
      margin-bottom:10px;
      break-inside:avoid;
      break-after:avoid;
    }}
    .stat-row {{
      gap:6px;
      break-inside:avoid;
    }}
    .sub-title {{ break-after:avoid; }}
    .cards {{ break-inside:avoid; }}
    .card {{ break-inside:avoid; }}
    table {{ break-inside:auto; }}
    tr {{ break-inside:avoid; page-break-inside:avoid; }}
    thead {{ display:table-header-group; }}
    .section-title {{ font-size:13px; }}
    .mini-stat {{ padding:6px 8px; min-width:70px; }}
    .ms-val {{ font-size:13px; }}
    th, td {{ padding:5px 8px; font-size:11px; }}
    .sec-title {{ margin:14px 0 8px; break-after:avoid; }}
    .content {{ padding-bottom:10px; }}
  }}
</style>
</head>
<body>
<div class="header">
  <div class="header-top-bar">
    <div class="header-org-name">{_e(t("rpt.org_name"))}</div>
    <div class="header-doc-num">{_e(t("rpt.system_ver"))}</div>
  </div>
  <div class="header-main">
    <div class="header-left">
      <div class="header-system-badge">{_e(t("rpt.auto_system"))}</div>
      <div class="header-title">{_e(t("rpt.main_title"))}</div>
      <div class="header-sub">{_e(t("rpt.subtitle"))}</div>
    </div>
    <div class="header-right">
      <div class="header-meta-row">
        <span class="header-meta-label">{_e(t("rpt.period"))}</span>
        <span class="header-meta-val">{_e(period_str)}</span>
      </div>
      <div class="header-meta-row">
        <span class="header-meta-label">{_e(t("rpt.created"))}</span>
        <span class="header-meta-val">{_e(created_str)}</span>
      </div>
      <div class="header-meta-row">
        <span class="header-meta-label">{_e(t("rpt.crossings"))}</span>
        <span class="header-meta-val">{_e(t("rpt.meta_count", n=len(crossings)))}</span>
      </div>
      <div class="header-meta-row">
        <span class="header-meta-label">{_e(t("rpt.cameras"))}</span>
        <span class="header-meta-val">{_e(t("rpt.meta_count", n=total_cams))}</span>
      </div>
    </div>
  </div>
  <div class="header-bottom-bar">
    <div class="hb-item">{_e(t("rpt.checkpoints", n=len(crossings)))}</div>
    <span class="hb-sep">|</span>
    <div class="hb-item">{_e(t("rpt.monitor_cameras", n=total_cams))}</div>
    <span class="hb-sep">|</span>
    <div class="hb-item">{_e(t("rpt.trains_recorded", n=grand_trains))}</div>
    <span class="hb-sep">|</span>
    <div class="hb-item">{_e(t("rpt.period_label", period=period_str))}</div>
  </div>
</div>

<div class="content">
  <div class="sec-title"><span>1. {_e(t("rpt.sec_summary"))}</span></div>
  <div class="cards">
    <div class="card" style="border-top:4px solid #1d4ed8">
      <div class="card-lbl">{_e(t("rpt.card_total_transport"))}</div>
      <div class="card-val">{_fmt(grand_light+grand_heavy)}</div>
    </div>
    <div class="card" style="border-top:4px solid #10b981">
      <div class="card-lbl">{_e(t("rpt.card_light_transport"))}</div>
      <div class="card-val">{_fmt(grand_light)}</div>
    </div>
    <div class="card" style="border-top:4px solid #f59e0b">
      <div class="card-lbl">{_e(t("rpt.card_heavy_transport"))}</div>
      <div class="card-val">{_fmt(grand_heavy)}</div>
    </div>
    <div class="card" style="border-top:4px solid #7c3aed">
      <div class="card-lbl">{_e(t("rpt.card_trains"))}</div>
      <div class="card-val">{_fmt(grand_trains)}</div>
    </div>
  </div>

  <div class="sec-title"><span>2. {_e(t("rpt.sec_compare"))}</span></div>
  <div class="section">
    <table>
      <thead><tr>
        <th>{_e(t("rpt.col_num"))}</th><th>{_e(t("rpt.col_crossing"))}</th><th>{_e(t("rpt.col_location"))}</th>
        <th>{_e(t("rpt.light"))}</th><th>{_e(t("rpt.heavy"))}</th><th>{_e(t("rpt.total"))}</th><th>{_e(t("rpt.col_cameras"))}</th>
      </tr></thead>
      <tbody>{cmp_rows}</tbody>
    </table>
  </div>

  <div class="sec-title"><span>3. {_e(t("rpt.sec_detail"))}</span></div>
  {sections_html}

  <div class="footer">
    — {_e(t("rpt.footer_end"))} —<br>
    {_e(t("rpt.footer_created", created=created_str))}
  </div>
</div>
</body>
</html>"""
