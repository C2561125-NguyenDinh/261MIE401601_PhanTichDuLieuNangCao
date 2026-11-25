"""Sinh báo cáo tiểu luận (.docx) TỰ ĐỘNG từ kết quả thực nghiệm trong results/.

Nội dung văn bản nằm trong report/content/*.md với cú pháp mở rộng:
  # / ## / ### / ####         tiêu đề cấp 1-4 (cấp 1 luôn sang trang mới)
  - mục                       gạch đầu dòng
  $$ công thức $$ (x.y)       công thức căn giữa, đánh số bên phải
  {{fig:tên|w=15}}            chèn hình results/figures/tên.png + chú thích tự đánh số
  {{tab:tên|rows=40|cols=A;B}} chèn bảng results/tables/tên.csv + chú thích tự đánh số
  [[fig:tên]] / [[tab:tên]]   tham chiếu chéo "Hình 4.2" / "Bảng 4.3"
  {{m:giai_đoạn.khóa.con|fmt}} giá trị từ results/metrics.json (fmt: .3f, pct1, int, raw)
  {{tv:bảng|Cột=giá trị;...|Cột lấy|fmt}} giá trị một ô trong bảng kết quả
  {{code:src/tệp.py}}          chèn mã nguồn (phụ lục)
  {{toc}} {{lof}} {{lot}}      mục lục, danh mục hình, danh mục bảng (trường Word)
  \\pagebreak                  ngắt trang
Mọi con số định lượng trong báo cáo đều được lấy qua {{m:...}} / {{tv:...}} / {{tab:...}}.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from utils import load_config  # noqa: E402

FONT = "Times New Roman"
STATIC_CAPTIONS = {
    "x_stage_timings": {"caption": "Thời gian chạy từng giai đoạn của pipeline trên máy thực nghiệm"},
    "s_plan": {"caption": "Kế hoạch thực hiện đồ án"},
    "s_tasks": {"caption": "Bảng phân công công việc"},
    "s_abbrev": {"caption": "Danh mục từ viết tắt"},
    "s_lit": {"caption": "Tổng hợp các nghiên cứu chính theo giai đoạn của chu trình và vai trò trong đề tài"},
    "s_raci": {"caption": "Ma trận phân công trách nhiệm RACI cho vận hành hệ thống"},
}


# ------------------------------------------------------------------ number formatting (vi-VN)
def vn_num(x, fmt=".4f"):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "–"
    if isinstance(x, bool):
        return "Có" if x else "Không"
    if fmt == "raw":
        return str(x)
    if isinstance(x, str):
        try:
            x = float(x)
        except ValueError:
            return x
    if fmt == "int":
        s = f"{int(round(x)):,}"
        return s.replace(",", ".")
    if fmt.startswith("pct"):
        d = int(fmt[3:] or 1)
        s = f"{100 * x:,.{d}f}"
        return s.replace(",", "X").replace(".", ",").replace("X", ".") + "%"
    if fmt.startswith(",") or fmt.startswith("."):
        s = format(x, "," + fmt.lstrip(",")) if fmt.startswith(",") else format(x, "," + fmt)
        return s.replace(",", "X").replace(".", ",").replace("X", ".")
    return format(x, fmt)


CONSTRAINT_VI = {"capacity": "Năng lực gửi", "budget": "Ngân sách"}


def _vi_constraint(v):
    if v in CONSTRAINT_VI:
        return CONSTRAINT_VI[v]
    m = re.fullmatch(r"(max_share|min_cov)\[(.+)\]", v)
    if m:
        grp = m.group(2).replace("Không rõ", "không rõ tuổi")
        return ("Trần tỷ trọng nhóm " if m.group(1) == "max_share" else "Sàn độ phủ nhóm ") + grp
    return v


def auto_fmt(v):
    if isinstance(v, bool):
        return "Có" if v else "Không"
    if isinstance(v, (int,)) and not isinstance(v, bool):
        return vn_num(v, "int")
    if isinstance(v, float):
        if pd.isna(v):
            return "–"
        if abs(v) < 5e-5 and v != 0:
            v = 0.0
        if v == 0:
            return "0"
        if float(v).is_integer() and abs(v) >= 1:
            return vn_num(v, "int")
        if abs(v) >= 100:
            return vn_num(v, ",.0f")
        if abs(v) >= 10:
            return vn_num(v, ".2f")
        return vn_num(v, ".4f")
    if isinstance(v, str) and v == "NOT_SOLVED":
        return "Không khả thi"
    if isinstance(v, str) and v in ("True", "False"):
        return "Có" if v == "True" else "Không"
    if isinstance(v, str):
        return _vi_constraint(v)
    return str(v)


class Report:
    def __init__(self):
        self.cfg = load_config()
        res = self.cfg["paths"]["results"]
        self.metrics = json.loads((res / "metrics.json").read_text(encoding="utf-8"))
        self.registry = json.loads((res / "registry.json").read_text(encoding="utf-8"))
        tp = res / "stage_timings.json"
        self.metrics["timings"] = json.loads(tp.read_text(encoding="utf-8")) if tp.exists() else {}
        self.fig_no, self.tab_no, self.eq_count = {}, {}, {}
        self.chapter = 0
        self.missing = []
        self.doc = Document()
        self._setup()

    # -------------------------------------------------------------- styles / page
    def _setup(self):
        d = self.doc
        for s in d.sections:
            s.page_height, s.page_width = Cm(29.7), Cm(21.0)
            s.top_margin, s.bottom_margin = Cm(2.0), Cm(2.0)
            s.left_margin, s.right_margin = Cm(3.0), Cm(2.0)
        st = d.styles["Normal"]
        st.font.name = FONT; st.font.size = Pt(13)
        st.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
        pf = st.paragraph_format
        pf.line_spacing = 1.5; pf.space_after = Pt(6); pf.space_before = Pt(0)
        sizes = {1: 16, 2: 14, 3: 13, 4: 13}
        for lvl in range(1, 5):
            hs = d.styles[f"Heading {lvl}"]
            hs.font.name = FONT; hs.font.size = Pt(sizes[lvl]); hs.font.bold = True
            hs.font.italic = lvl == 4
            hs.font.color.rgb = RGBColor(0, 0, 0)
            rpr = hs.element.get_or_add_rPr()
            rf = rpr.find(qn("w:rFonts"))
            if rf is None:
                rf = OxmlElement("w:rFonts"); rpr.append(rf)
            for a in ["w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"]:
                rf.set(qn(a), FONT)
            for a in ["w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"]:
                if rf.get(qn(a)) is not None:
                    del rf.attrib[qn(a)]
            hs.paragraph_format.space_before = Pt(12 if lvl > 1 else 0)
            hs.paragraph_format.space_after = Pt(6)
            hs.paragraph_format.line_spacing = 1.3
            hs.paragraph_format.keep_with_next = True
            if lvl == 1:
                hs.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap = d.styles["Caption"]
        cap.font.name = FONT; cap.font.size = Pt(12); cap.font.bold = True; cap.font.italic = False
        cap.font.color.rgb = RGBColor(0, 0, 0)
        cap.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        from docx.enum.style import WD_STYLE_TYPE
        for nm in ["Chu thich hinh", "Chu thich bang"]:
            stl = d.styles.add_style(nm, WD_STYLE_TYPE.PARAGRAPH)
            stl.base_style = cap
            stl.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
            stl.paragraph_format.space_after = Pt(4)
        self._page_numbers()

    def _page_numbers(self):
        footer = self.doc.sections[0].footer
        p = footer.paragraphs[0]; p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        self._field(p, "PAGE")

    def _field(self, p, instr):
        r = p.add_run()
        f1 = OxmlElement("w:fldChar"); f1.set(qn("w:fldCharType"), "begin")
        it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = instr
        f2 = OxmlElement("w:fldChar"); f2.set(qn("w:fldCharType"), "separate")
        t = OxmlElement("w:t"); t.text = "(Nhấn F9 trong Word để cập nhật)"
        f3 = OxmlElement("w:fldChar"); f3.set(qn("w:fldCharType"), "end")
        r._r.append(f1); r._r.append(it); r._r.append(f2); r._r.append(t); r._r.append(f3)

    # -------------------------------------------------------------- value resolvers
    def metric(self, path, fmt="auto"):
        cur = self.metrics
        for k in path.split("."):
            if isinstance(cur, list):
                cur = cur[int(k)]
            elif isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                self.missing.append(path)
                return f"[THIẾU:{path}]"
        return auto_fmt(cur) if fmt == "auto" else vn_num(cur, fmt)

    def table_value(self, name, filt, col, fmt="auto"):
        df = self.load_table(name)
        if df is None:
            self.missing.append(name)
            return f"[THIẾU:{name}]"
        for cond in [c for c in filt.split(";") if c]:
            k, v = cond.rsplit("=", 1)
            if k not in df.columns:
                self.missing.append(f"{name}: không có cột '{k}'")
                return f"[THIẾU:{name}:{k}]"
            df = df[df[k].astype(str) == v]
        if len(df) != 1:
            self.missing.append(f"{name}|{filt}|{col} -> {len(df)} dòng")
            return f"[THIẾU:{name}:{filt}]"
        if col not in df.columns:
            self.missing.append(f"{name}: không có cột '{col}'")
            return f"[THIẾU:{name}:{col}]"
        v = df.iloc[0][col]
        return auto_fmt(v) if fmt == "auto" else vn_num(v, fmt)

    def load_table(self, name):
        if name == "x_stage_timings":
            t = self.metrics.get("timings", {})
            return pd.DataFrame({"Giai đoạn": list(t.keys()), "Thời gian chạy (giây)": list(t.values())})
        sp = ROOT / "report" / "static" / f"{name}.csv"
        if sp.exists():
            return pd.read_csv(sp, encoding="utf-8")
        p = self.cfg["paths"]["tables"] / f"{name}.csv"
        if not p.exists():
            return None
        return pd.read_csv(p, encoding="utf-8-sig")

    def resolve_inline(self, text):
        def rep_m(mo):
            parts = mo.group(1).split("|")
            return self.metric(parts[0], parts[1] if len(parts) > 1 else "auto")

        def rep_tv(mo):
            parts = mo.group(1).split("|")
            return self.table_value(parts[0], parts[1], parts[2], parts[3] if len(parts) > 3 else "auto")

        def rep_ref(mo):
            kind, name = mo.group(1), mo.group(2)
            num = (self.fig_no if kind == "fig" else self.tab_no).get(name)
            if num is None:
                self.missing.append(f"ref:{name}")
                return f"[?{name}]"
            return ("Hình " if kind == "fig" else "Bảng ") + num

        text = re.sub(r"\{\{m:([^}]+)\}\}", rep_m, text)
        text = re.sub(r"\{\{tv:([^}]+)\}\}", rep_tv, text)
        text = re.sub(r"\[\[(fig|tab):([^\]]+)\]\]", rep_ref, text)
        return text

    # -------------------------------------------------------------- writers
    def add_runs(self, p, text):
        """Hỗ trợ **đậm** và *nghiêng*."""
        for tok in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*)", text):
            if not tok:
                continue
            if tok.startswith("**"):
                r = p.add_run(tok[2:-2]); r.bold = True
            elif tok.startswith("*"):
                r = p.add_run(tok[1:-1]); r.italic = True
            else:
                p.add_run(tok)

    def para(self, text, style=None, align=WD_ALIGN_PARAGRAPH.JUSTIFY, indent=True):
        p = self.doc.add_paragraph(style=style)
        p.alignment = align
        if getattr(self, "in_refs", False) and style is None:
            p.paragraph_format.left_indent = Cm(1.0)
            p.paragraph_format.first_line_indent = Cm(-1.0)
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        elif indent and style is None:
            p.paragraph_format.first_line_indent = Cm(1.0)
        self.add_runs(p, self.resolve_inline(text))
        return p

    # -------------------------------------------------------------- cover (theo mẫu bìa UEL)
    def _cover_image(self, date_text):
        """In dòng ngày tháng (chữ trắng) lên dải xanh của ảnh nền bìa."""
        from PIL import Image, ImageDraw, ImageFont
        src = ROOT / "report" / "static" / "uel_cover.jpg"
        out = ROOT / "report" / "output" / "_cover_bg.jpg"
        im = Image.open(src).convert("RGB"); W, H = im.size
        x = W // 2
        y0 = next(y for y in range(H // 2, H) if im.getpixel((x, y))[2] > 100 and im.getpixel((x, y))[0] < 60)
        y1 = y0
        while im.getpixel((x, y1 + 1))[2] > 100 and im.getpixel((x, y1 + 1))[0] < 60:
            y1 += 1
        font = None
        for fp in ("C:/Windows/Fonts/timesbd.ttf", "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf"):
            if Path(fp).exists():
                font = ImageFont.truetype(fp, int(round(11 / 72 * 2.54 * W / 21.0)))
                break
        d = ImageDraw.Draw(im)
        b = d.textbbox((0, 0), date_text, font=font)
        d.text(((W - (b[2] - b[0])) / 2 - b[0], (y0 + y1) / 2 - (b[3] - b[1]) / 2 - b[1]), date_text,
               font=font, fill=(255, 255, 255))
        im.save(out, quality=92)
        return out

    def cover(self, info):
        """Trang bìa theo mẫu UEL: ảnh nền toàn trang + các dòng chữ đặt đúng vị trí."""
        from docx.shared import Emu
        d = self.doc
        d.sections[0].different_first_page_header_footer = True
        img = self._cover_image(info["date"])
        p0 = d.add_paragraph(); p0.paragraph_format.space_after = Pt(0); p0.paragraph_format.first_line_indent = Cm(0)
        run = p0.add_run()
        inline = run.add_picture(str(img), width=Emu(7560310), height=Emu(10692130))
        # chuyển inline -> anchor nằm sau chữ, gốc tại góc trang
        il = inline._inline
        graphic = il.find(qn("a:graphic"))
        anc = OxmlElement("wp:anchor")
        for k, v in dict(distT="0", distB="0", distL="0", distR="0", simplePos="0", relativeHeight="251658240",
                         behindDoc="1", locked="1", layoutInCell="1", allowOverlap="1").items():
            anc.set(k, v)
        sp = OxmlElement("wp:simplePos"); sp.set("x", "0"); sp.set("y", "0"); anc.append(sp)
        for tag, rel in (("wp:positionH", "page"), ("wp:positionV", "page")):
            ph = OxmlElement(tag); ph.set("relativeFrom", rel)
            off = OxmlElement("wp:posOffset"); off.text = "0"; ph.append(off); anc.append(ph)
        ext = OxmlElement("wp:extent"); ext.set("cx", "7560310"); ext.set("cy", "10692130"); anc.append(ext)
        ee = OxmlElement("wp:effectExtent")
        for k in "ltrb":
            ee.set(k, "0")
        anc.append(ee)
        anc.append(OxmlElement("wp:wrapNone"))
        dp = OxmlElement("wp:docPr"); dp.set("id", "9001"); dp.set("name", "Cover background"); anc.append(dp)
        anc.append(OxmlElement("wp:cNvGraphicFramePr"))
        anc.append(graphic)
        il.getparent().replace(il, anc)

        def line(text, size, bold=False, align=WD_ALIGN_PARAGRAPH.CENTER, after=0, before=0, line=None,
                 right_ind=None, color=None, frame_y=None):
            p = d.add_paragraph()
            pf = p.paragraph_format
            pf.space_after = Pt(after); pf.space_before = Pt(before); pf.first_line_indent = Cm(0)
            pf.line_spacing = line if line else 1.0
            if right_ind is not None:
                pf.right_indent = Cm(right_ind)
            p.alignment = align
            if frame_y is not None:
                fr = OxmlElement("w:framePr")
                for k, v in (("w:w", "11906"), ("w:wrap", "around"), ("w:vAnchor", "page"),
                             ("w:hAnchor", "page"), ("w:y", str(frame_y))):
                    fr.set(qn(k), v)
                p._p.get_or_add_pPr().insert(0, fr)
            r = p.add_run(text); r.bold = bold; r.font.size = Pt(size)
            if color:
                r.font.color.rgb = RGBColor.from_string(color)
            return p

        line("KHOA HỆ THỐNG THÔNG TIN", 20, bold=True, before=26)
        line("", 13, after=140)
        line("TIỂU LUẬN CUỐI KỲ", 14, bold=True, after=14)
        line(info["title"].upper(), 16, bold=True, after=30, line=1.25)
        line("Môn học: " + info["course"], 16, bold=True, after=22)
        R = WD_ALIGN_PARAGRAPH.RIGHT
        line("GVHD:", 14, bold=True, align=R, line=1.2, right_ind=2.9)
        line(info["gvhd"], 14, align=R, after=14, line=1.2, right_ind=2.9)
        line("HVTH:", 14, bold=True, align=R, line=1.2, right_ind=2.9)
        line(info["hvth"], 14, align=R, line=1.2, right_ind=2.9)

    def heading(self, text, lvl):
        if lvl == 1:
            m = re.match(r"CHƯƠNG\s+(\d+)", text.upper())
            if m:
                self.chapter = int(m.group(1))
            elif text.upper().startswith("PHỤ LỤC"):
                self.chapter = "PL"
        if lvl == 1:
            self.in_refs = text.upper().startswith("TÀI LIỆU THAM KHẢO")
        h = self.doc.add_heading(self.resolve_inline(text), level=lvl)
        if lvl == 1:
            h.paragraph_format.page_break_before = True
        return h

    def _new_num(self, p):
        """Tạo một instance đánh số mới (bắt đầu lại từ 1) cho mỗi danh sách đánh số."""
        numbering = self.doc.part.numbering_part.numbering_definitions._numbering
        ppr = p.style.element.pPr
        base_num_id = int(ppr.numPr.numId.val)
        abs_id = None
        for num in numbering.findall(qn("w:num")):
            if int(num.get(qn("w:numId"))) == base_num_id:
                abs_id = num.find(qn("w:abstractNumId")).get(qn("w:val"))
        ids = [int(n.get(qn("w:numId"))) for n in numbering.findall(qn("w:num"))]
        new_id = max(ids) + 1
        num = OxmlElement("w:num"); num.set(qn("w:numId"), str(new_id))
        a = OxmlElement("w:abstractNumId"); a.set(qn("w:val"), abs_id); num.append(a)
        ov = OxmlElement("w:lvlOverride"); ov.set(qn("w:ilvl"), "0")
        so = OxmlElement("w:startOverride"); so.set(qn("w:val"), "1"); ov.append(so); num.append(ov)
        numbering.append(num)
        return new_id

    def bullet(self, text, numbered=False):
        p = self.doc.add_paragraph(style="List Number" if numbered else "List Bullet")
        if numbered:
            prev = p._p.getprevious()
            prev_is_num = prev is not None and getattr(self, "_last_num_p", None) is prev
            if not prev_is_num:
                self._cur_num_id = self._new_num(p)
            pPr = p._p.get_or_add_pPr()
            numPr = OxmlElement("w:numPr")
            il = OxmlElement("w:ilvl"); il.set(qn("w:val"), "0")
            ni = OxmlElement("w:numId"); ni.set(qn("w:val"), str(self._cur_num_id))
            numPr.append(il); numPr.append(ni); pPr.insert(1 if pPr.find(qn("w:pStyle")) is not None else 0, numPr)
            self._last_num_p = p._p
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.line_spacing = 1.5
        self.add_runs(p, self.resolve_inline(text))

    def equation(self, eq, num):
        p = self.doc.add_paragraph()
        sec = self.doc.sections[0]
        width = sec.page_width - sec.left_margin - sec.right_margin
        p.paragraph_format.tab_stops.add_tab_stop(int(width / 2), WD_TAB_ALIGNMENT.CENTER)
        p.paragraph_format.tab_stops.add_tab_stop(width, WD_TAB_ALIGNMENT.RIGHT)
        p.add_run("\t")
        r = p.add_run(self.resolve_inline(eq.strip())); r.italic = True; r.font.name = "Cambria Math"
        if num:
            p.add_run(f"\t({num})")

    def figure(self, name, width=15.5):
        meta = self.registry.get("figures", {}).get(name)
        path = self.cfg["paths"]["figures"] / f"{name}.png"
        if meta is None or not path.exists():
            self.missing.append(f"fig:{name}")
            self.para(f"[THIẾU HÌNH: {name}]")
            return
        p = self.doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.keep_with_next = True
        p.add_run().add_picture(str(path), width=Cm(width))
        c = self.doc.add_paragraph(style="Chu thich hinh")
        c.add_run(f"Hình {self.fig_no[name]}. ").bold = True
        r = c.add_run(meta["caption"]); r.bold = False; r.italic = True
        src = self.doc.add_paragraph(); src.alignment = WD_ALIGN_PARAGRAPH.CENTER
        rr = src.add_run("Nguồn: Kết quả thực nghiệm của tác giả" if not name.startswith("f00") else "Nguồn: Tác giả thiết kế")
        rr.font.size = Pt(11); rr.italic = True

    def table(self, name, rows=None, cols=None, fs=9.5, landscape=False):
        meta = self.registry.get("tables", {}).get(name) or STATIC_CAPTIONS.get(name)
        df = self.load_table(name)
        if meta is None or df is None:
            self.missing.append(f"tab:{name}")
            self.para(f"[THIẾU BẢNG: {name}]")
            return
        if cols:
            df = df[[c for c in cols if c in df.columns]]
        total = len(df)
        if rows:
            df = df.head(int(rows))
        c = self.doc.add_paragraph(style="Chu thich bang")
        c.paragraph_format.keep_with_next = True
        c.add_run(f"Bảng {self.tab_no[name]}. ").bold = True
        r = c.add_run(meta["caption"] + (f" ({len(df)}/{total} dòng đầu)" if rows and total > int(rows) else ""))
        r.bold = False; r.italic = True
        t = self.doc.add_table(rows=1, cols=len(df.columns))
        t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for j, col in enumerate(df.columns):
            cell = t.rows[0].cells[j]; cell.text = ""
            run = cell.paragraphs[0].add_run(str(col)); run.bold = True; run.font.size = Pt(fs)
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            run.font.color.rgb = RGBColor(0, 0, 0)
        for _, row in df.iterrows():
            cells = t.add_row().cells
            for j, col in enumerate(df.columns):
                v = row[col]
                txt = auto_fmt(v.item() if hasattr(v, "item") else v)
                cells[j].text = ""
                pp = cells[j].paragraphs[0]
                run = pp.add_run(txt); run.font.size = Pt(fs); run.font.color.rgb = RGBColor(0, 0, 0)
                pp.paragraph_format.line_spacing = 1.0; pp.paragraph_format.space_after = Pt(0)
                pp.alignment = WD_ALIGN_PARAGRAPH.RIGHT if isinstance(v, (int, float)) and not isinstance(v, bool) else WD_ALIGN_PARAGRAPH.LEFT
        self._repeat_header(t.rows[0])
        src = self.doc.add_paragraph(); src.alignment = WD_ALIGN_PARAGRAPH.CENTER
        rr = src.add_run("Nguồn: Tác giả thiết kế" if (name in ("t00_decision_contract", "t00b_stage_mapping", "t00c_metric_system", "t40_exception_playbook", "t49_retraining_policy", "t09_leakage_checklist") or name.startswith("s_")) else "Nguồn: Kết quả thực nghiệm của tác giả")
        rr.font.size = Pt(11); rr.italic = True

    def _shade(self, cell, color):
        tcPr = cell._tc.get_or_add_tcPr()
        shd = OxmlElement("w:shd"); shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), color)
        tcPr.append(shd)

    def _repeat_header(self, row):
        trPr = row._tr.get_or_add_trPr()
        el = OxmlElement("w:tblHeader"); el.set(qn("w:val"), "true"); trPr.append(el)

    def code(self, rel):
        path = ROOT / rel
        text = path.read_text(encoding="utf-8")
        p = self.doc.add_paragraph()
        p.add_run(f"Tệp: {rel}").bold = True
        for line in text.splitlines():
            q = self.doc.add_paragraph()
            q.paragraph_format.line_spacing = 1.0; q.paragraph_format.space_after = Pt(0)
            r = q.add_run(line if line else " "); r.font.name = "Consolas"; r.font.size = Pt(8)

    def field_list(self, kind):
        p = self.doc.add_paragraph()
        instr = {"toc": 'TOC \\o "1-3" \\h \\z \\u', "lof": 'TOC \\h \\z \\t "Chu thich hinh,1"', "lot": 'TOC \\h \\z \\t "Chu thich bang,1"'}[kind]
        self._field(p, instr)

    # -------------------------------------------------------------- pipeline
    def number_pass(self, lines):
        chap, fi, ti = "MĐ", {}, {}
        for ln in lines:
            if ln.startswith("# "):
                m = re.match(r"#\s+CHƯƠNG\s+(\d+)", ln.upper())
                if m:
                    chap = int(m.group(1))
                elif "PHỤ LỤC" in ln.upper():
                    chap = "PL"
            for kind, name in re.findall(r"\{\{(fig|tab):([^|}]+)", ln):
                store, cnt = (self.fig_no, fi) if kind == "fig" else (self.tab_no, ti)
                if name in store:
                    continue
                cnt[chap] = cnt.get(chap, 0) + 1
                store[name] = f"{chap}.{cnt[chap]}"

    def build(self, content_dir: Path, out: Path):
        files = sorted(content_dir.glob("*.md"))
        lines = []
        for f in files:
            lines += f.read_text(encoding="utf-8").splitlines() + [""]
        self.number_pass(lines)
        buf = []
        eq_no = {}

        def flush():
            if buf:
                self.para(" ".join(s.strip() for s in buf))
                buf.clear()

        for ln in lines:
            s = ln.rstrip()
            if not s.strip():
                flush(); continue
            if s.startswith("<!--"):
                continue
            mh = re.match(r"^(#{1,4})\s+(.*)", s)
            if mh:
                flush(); self.heading(mh.group(2), len(mh.group(1))); continue
            mcv = re.match(r"^\{\{cover\|(.+)\}\}$", s.strip())
            if mcv:
                flush(); self.cover(dict(kv.split("=", 1) for kv in mcv.group(1).split("|"))); continue
            mr = re.match(r"^>\|(.*)", s)
            if mr:
                flush(); self.para(mr.group(1).strip(), align=WD_ALIGN_PARAGRAPH.RIGHT, indent=False); continue
            if s.strip() == "\\pagebreak":
                flush(); self.doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE); continue
            if s.strip() in ("{{toc}}", "{{lof}}", "{{lot}}"):
                flush(); self.field_list(s.strip()[2:-2]); continue
            mf = re.match(r"^\{\{fig:([^|}]+)(?:\|w=([\d.]+))?\}\}$", s.strip())
            if mf:
                flush(); self.figure(mf.group(1), float(mf.group(2) or 15.5)); continue
            mt = re.match(r"^\{\{tab:([^|}]+)((?:\|[^}]*)?)\}\}$", s.strip())
            if mt:
                flush()
                opts = dict(o.split("=", 1) for o in mt.group(2).split("|") if "=" in o)
                self.table(mt.group(1), rows=opts.get("rows"), cols=opts["cols"].split(";") if "cols" in opts else None,
                           fs=float(opts.get("fs", 9.5)))
                continue
            mc = re.match(r"^\{\{code:([^}]+)\}\}$", s.strip())
            if mc:
                flush(); self.code(mc.group(1)); continue
            me = re.match(r"^\$\$(.+)\$\$\s*$", s.strip())
            if me:
                flush()
                eq_no[self.chapter] = eq_no.get(self.chapter, 0) + 1
                self.equation(me.group(1), f"{self.chapter}.{eq_no[self.chapter]}"); continue
            mb = re.match(r"^\s*[-•]\s+(.*)", s)
            if mb:
                flush(); self.bullet(mb.group(1)); continue
            mn = re.match(r"^\s*\d+\.\s+(.*)", s)
            if mn and not re.match(r"^\s*\d+\.\d", s):
                flush(); self.bullet(mn.group(1), numbered=True); continue
            mcen = re.match(r"^>>(.*)", s)
            if mcen:
                flush(); self.para(mcen.group(1).strip(), align=WD_ALIGN_PARAGRAPH.CENTER, indent=False); continue
            buf.append(s)
        flush()
        self.doc.save(out)
        return out


def main():
    rep = Report()
    out_dir = ROOT / "report" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = rep.build(ROOT / "report" / "content", out_dir / "Tieu_luan_PTDLNC_X5_Uplift.docx")
    print(f"Đã sinh báo cáo: {out}")
    if rep.missing:
        print(f"CẢNH BÁO: {len(rep.missing)} tham chiếu thiếu:")
        for x in sorted(set(rep.missing))[:50]:
            print("  -", x)


if __name__ == "__main__":
    main()
