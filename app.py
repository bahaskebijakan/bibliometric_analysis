import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from wordcloud import WordCloud, STOPWORDS
from collections import Counter
from itertools import combinations
import re, os, io, zipfile, tempfile

import bibtexparser
import rispy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from pyvis.network import Network

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Bahas Kebijakan — Bibliometric Analysis Suite",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL STYLE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Background */
.stApp { background: #0F1117; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #161B27 !important;
    border-right: 1px solid #1E2A40;
}

/* Hero title */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #60A5FA 0%, #A78BFA 50%, #34D399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 0.2rem;
}
.hero-sub {
    font-family: 'DM Sans', sans-serif;
    color: #64748B;
    font-size: 1rem;
    margin-bottom: 2rem;
}

/* KPI Cards */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 12px;
    margin: 1.5rem 0;
}
.kpi-card {
    background: linear-gradient(135deg, #161B27 0%, #1E2A40 100%);
    border: 1px solid #1E3A5F;
    border-radius: 14px;
    padding: 18px 12px;
    text-align: center;
    transition: transform 0.2s, border-color 0.2s;
}
.kpi-card:hover { transform: translateY(-2px); border-color: #3B82F6; }
.kpi-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.7rem;
    font-weight: 700;
    color: #F1F5F9;
}
.kpi-label {
    font-size: 0.72rem;
    color: #64748B;
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Section header */
.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: #F1F5F9;
    border-left: 3px solid #3B82F6;
    padding-left: 12px;
    margin: 2rem 0 1rem;
}

/* Status badges */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
}
.badge-subj  { background:#1E3A5F; color:#60A5FA; }
.badge-obj   { background:#14532D; color:#4ADE80; }
.badge-hyb   { background:#451A03; color:#FCD34D; }
.badge-ns    { background:#1E1B4B; color:#A5B4FC; }

/* Upload zone */
.upload-hint {
    background: #161B27;
    border: 1px dashed #1E3A5F;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    color: #475569;
    font-size: 0.9rem;
    margin-bottom: 1rem;
}

/* Download button */
.stDownloadButton > button {
    background: linear-gradient(135deg, #1D4ED8, #7C3AED) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.5rem !important;
    transition: opacity 0.2s !important;
}
.stDownloadButton > button:hover { opacity: 0.88 !important; }

/* Tab styling */
.stTabs [data-baseweb="tab"] {
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    color: #64748B;
}
.stTabs [aria-selected="true"] { color: #60A5FA !important; }

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
/* Insight boxes */
.insight-box {
    background: linear-gradient(135deg, #0F1B2D 0%, #0D1F33 100%);
    border: 1px solid #1E3A5F;
    border-left: 4px solid #3B82F6;
    border-radius: 0 10px 10px 0;
    padding: 18px 20px;
    margin-top: 1.5rem;
    line-height: 1.8;
}
.insight-box.green  { border-left-color: #22C55E; }
.insight-box.amber  { border-left-color: #F59E0B; }
.insight-box.purple { border-left-color: #A78BFA; }
.insight-box.red    { border-left-color: #EF4444; }
.insight-title {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 0.9rem;
    color: #60A5FA;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.insight-title.green  { color: #4ADE80; }
.insight-title.amber  { color: #FCD34D; }
.insight-title.purple { color: #C4B5FD; }
.insight-title.red    { color: #FCA5A5; }
.insight-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin-top: 12px;
}
.insight-card {
    background: #0F1117;
    border: 1px solid #1E2A40;
    border-radius: 8px;
    padding: 12px 14px;
    font-size: 0.82rem;
    color: #94A3B8;
    line-height: 1.6;
}
.insight-card b { color: #F1F5F9; }
.insight-tag {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.72rem;
    font-weight: 600;
    margin-right: 4px;
    margin-bottom: 4px;
}
.tag-blue   { background:#1E3A5F; color:#60A5FA; }
.tag-green  { background:#14532D; color:#4ADE80; }
.tag-amber  { background:#451A03; color:#FCD34D; }
.tag-purple { background:#2E1065; color:#C4B5FD; }
.tag-red    { background:#450A0A; color:#FCA5A5; }

</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
C = dict(blue='#3B82F6', green='#22C55E', red='#EF4444',
         amber='#F59E0B', purple='#A78BFA', slate='#64748B',
         navy='#1E3A5F', bg='#0F1117', card='#161B27')

DPI_EXPORT = 600

plt.rcParams.update({
    'figure.facecolor': '#161B27', 'axes.facecolor': '#161B27',
    'axes.edgecolor': '#1E2A40',   'text.color': '#CBD5E1',
    'axes.labelcolor': '#94A3B8',  'xtick.color': '#94A3B8',
    'ytick.color': '#94A3B8',      'axes.grid': True,
    'grid.color': '#1E2A40',       'grid.linewidth': 0.5,
    'font.family': 'DejaVu Sans',  'axes.titlesize': 12,
    'axes.titleweight': 'bold',    'axes.titlecolor': '#F1F5F9',
})

# ══════════════════════════════════════════════════════════════════════════════
# PARSERS  —  Auto-detects Scopus vs Dimensions BIB
# ══════════════════════════════════════════════════════════════════════════════

def _detect_bib_source(content: str) -> str:
    """Return 'scopus', 'dimensions', or 'generic'."""
    if re.search(r'^Scopus\s*\nEXPORT DATE', content, re.IGNORECASE | re.MULTILINE):
        return 'scopus'
    if re.search(r'@article\{pub\.\d+', content[:500]):
        return 'dimensions'
    # Fallback: check for Scopus-specific fields
    if 'author_keywords' in content[:3000]:
        return 'scopus'
    return 'generic'


def _parse_bib_entries(content: str):
    """Strip non-BIB header and parse all entries with bibtexparser."""
    raw = re.sub(r'^[^@]+(?=@)', '', content, count=1)  # strip Scopus header
    parser = bibtexparser.bparser.BibTexParser(common_strings=True)
    parser.ignore_nonstandard_types = False
    parser.homogenise_fields = False
    return bibtexparser.loads(raw, parser).entries


def parse_bib(content: str) -> pd.DataFrame:
    """Universal BIB parser — handles Scopus, Dimensions, and generic BibTeX."""
    source = _detect_bib_source(content)
    entries = _parse_bib_entries(content)
    rows = []

    for e in entries:
        title = e.get('title', '').replace('{', '').replace('}', '').strip()
        authors = e.get('author', '')
        doi = e.get('doi', '')
        abstract = e.get('abstract', '')
        journal = e.get('journal', e.get('booktitle', e.get('publisher', '')))

        # ── Year: prefer 'year'; fall back to 'date' (Dimensions: "2024-03-15") ──
        year_raw = e.get('year', '') or e.get('date', '')
        year_m = re.search(r'\b(19|20)\d{2}\b', str(year_raw))
        year = year_m.group(0) if year_m else ''

        # ── Document type ────────────────────────────────────────────────────────
        doc_type = e.get('type', e.get('ENTRYTYPE', 'Unknown'))

        # ── Source-specific field extraction ────────────────────────────────────
        if source == 'scopus':
            note  = e.get('note', '')
            m     = re.search(r'Cited by:\s*(\d+)', note)
            cited = int(m.group(1)) if m else np.nan
            if   'Gold Open Access' in note: oa = 'Gold OA'
            elif 'Hybrid Gold'      in note: oa = 'Hybrid OA'
            elif 'Green'            in note: oa = 'Green OA'
            elif 'Open Access'      in note: oa = 'Open Access'
            else:                            oa = 'Closed'
            auth_kw  = e.get('author_keywords', '')
            idx_kw   = e.get('keywords', '')
            affiliations = e.get('affiliations', e.get('affiliation', ''))

        elif source == 'dimensions':
            # Dimensions has NO citation count in BIB export — leave NaN
            cited = np.nan
            oa = 'Unknown'
            # 'keywords' field exists but is always empty in Dimensions BIB —
            # extract from abstract as best-effort fallback
            auth_kw  = e.get('keywords', '')   # usually empty
            idx_kw   = ''
            affiliations = e.get('affiliations', '')

        else:  # generic BibTeX
            note  = e.get('note', '')
            m     = re.search(r'Cited by:\s*(\d+)', note)
            cited = int(m.group(1)) if m else np.nan
            oa = 'Unknown'
            auth_kw  = e.get('author_keywords', e.get('keywords', ''))
            idx_kw   = ''
            affiliations = e.get('affiliations', e.get('affiliation', ''))

        rows.append({
            'title':           title,
            'authors':         authors,
            'year':            year,
            'journal':         journal,
            'doi':             doi,
            'abstract':        abstract,
            'author_keywords': auth_kw,
            'index_keywords':  idx_kw,
            'keywords':        '; '.join(filter(None, [auth_kw, idx_kw])),
            'cited_by':        cited,
            'oa_status':       oa,
            'document_type':   doc_type,
            'affiliations':    affiliations,
            '_source':         source,   # track origin for UI badge
        })

    df = pd.DataFrame(rows)
    df['_source'] = source
    return df


# Keep alias for backwards compatibility
parse_scopus_bib = parse_bib


def parse_ris(content: str) -> pd.DataFrame:
    import io as _io
    entries = rispy.load(_io.StringIO(content))
    MAP = {'title':['title','primary_title'],'authors':['authors','first_authors'],
           'year':['year','publication_year'],'journal':['journal_name','secondary_title'],
           'abstract':['abstract'],'author_keywords':['keywords'],
           'doi':['doi'],'cited_by':['cited_by'],'affiliations':['affiliations']}
    rows = []
    for e in entries:
        row = {}
        for canon, keys in MAP.items():
            for k in keys:
                v = e.get(k,'')
                if v: row[canon] = '; '.join(v) if isinstance(v,list) else str(v); break
            if canon not in row: row[canon] = np.nan
        row.setdefault('index_keywords','')
        row['keywords']   = row.get('author_keywords','')
        row['oa_status']  = 'Unknown'
        row['document_type'] = e.get('type_of_reference','Unknown')
        rows.append(row)
    return pd.DataFrame(rows)


def parse_csv(content: str) -> pd.DataFrame:
    import io as _io
    for enc in ['utf-8','latin-1','cp1252']:
        try:
            df = pd.read_csv(_io.StringIO(content)); break
        except: continue
    SMAP = {'Title':'title','Authors':'authors','Year':'year',
            'Source title':'journal','Cited by':'cited_by','DOI':'doi',
            'Abstract':'abstract','Author Keywords':'author_keywords',
            'Index Keywords':'index_keywords','Affiliations':'affiliations',
            'Document Type':'document_type','Open Access':'oa_status'}
    df = df.rename(columns={k:v for k,v in SMAP.items() if k in df.columns})
    for c in ['author_keywords','index_keywords','oa_status']:
        if c not in df.columns: df[c] = ''
    df['keywords'] = df.get('author_keywords','').fillna('') + '; ' + df.get('index_keywords','').fillna('')
    return df


def load_and_merge(uploaded_files) -> pd.DataFrame:
    frames = []
    source_log = []   # for UI badge display
    for f in uploaded_files:
        content = f.read().decode('utf-8', errors='replace')
        ext = os.path.splitext(f.name)[1].lower()
        if ext in ('.bib', '.bibtex'):
            parsed = parse_bib(content)
            src = _detect_bib_source(content)
            source_log.append((f.name, src, len(parsed)))
            frames.append(parsed)
        elif ext == '.ris':
            frames.append(parse_ris(content))
            source_log.append((f.name, 'ris', len(frames[-1])))
        elif ext == '.csv':
            frames.append(parse_csv(content))
            source_log.append((f.name, 'csv', len(frames[-1])))
    if not frames: return pd.DataFrame(), []
    df = pd.concat(frames, ignore_index=True)

    CANONICAL = ['title','authors','year','journal','keywords','author_keywords',
                 'index_keywords','abstract','cited_by','doi','affiliations',
                 'document_type','oa_status']
    for c in CANONICAL:
        if c not in df.columns: df[c] = np.nan

    df['year']     = pd.to_numeric(df['year'],     errors='coerce')
    df['cited_by'] = pd.to_numeric(df['cited_by'], errors='coerce')
    df = df.drop_duplicates(subset=['title'], keep='first').reset_index(drop=True)
    df = df[df.year.between(1900, pd.Timestamp.now().year)].copy()

    def parse_authors(s):
        if pd.isna(s) or not str(s).strip(): return []
        s = str(s)
        if ' and ' in s.lower():
            return [p.strip() for p in re.split(r'\s+and\s+', s, flags=re.IGNORECASE) if p.strip()]
        return [p.strip() for p in re.split(r'[;|]', s) if p.strip()]

    def split_kw(x):
        if pd.isna(x) or not str(x).strip(): return []
        return [k.strip().lower() for k in re.split(r'[;,]', str(x)) if len(k.strip()) > 2]

    df['author_list']    = df['authors'].apply(parse_authors)
    df['author_count']   = df['author_list'].apply(len)
    df['author_kw_list'] = df['author_keywords'].apply(split_kw)
    df['index_kw_list']  = df['index_keywords'].apply(split_kw)

    COUNTRIES = [
        'Indonesia','Malaysia','Singapore','Thailand','Philippines','Vietnam',
        'China','Japan','South Korea','India','Bangladesh','Australia',
        'United States','United Kingdom','Germany','France','Spain','Italy',
        'Netherlands','Sweden','Norway','Brazil','Canada','Mexico','Egypt',
        'South Africa','Nigeria','Kenya','Ethiopia','Saudi Arabia','UAE','Iran',
        'Turkey','Poland','Russia','Algeria','Morocco','Togo','Cyprus','Ghana'
    ]
    _cpat  = re.compile(r'\b(' + '|'.join(re.escape(c) for c in COUNTRIES) + r')\b', re.IGNORECASE)
    _cnorm = {'USA':'United States','UK':'United Kingdom'}
    df['country_list'] = df['affiliations'].apply(
        lambda x: list(dict.fromkeys(_cnorm.get(c.title(),c.title())
                       for c in _cpat.findall(str(x)))) if pd.notna(x) else [])
    return df, source_log



# ══════════════════════════════════════════════════════════════════════════════
# FIGURE HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def fig_to_bytes(fig, dpi=DPI_EXPORT):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.read()

def dl_btn(label, fig, fname):
    st.download_button(
        label=f"⬇️ {label} (600 DPI PNG)",
        data=fig_to_bytes(fig),
        file_name=fname,
        mime='image/png',
        use_container_width=True,
    )

def section(title):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='margin-bottom:0.15rem'>
        <span style='font-family:Syne,sans-serif;font-size:1.35rem;font-weight:800;
        background:linear-gradient(135deg,#60A5FA,#A78BFA);-webkit-background-clip:text;
        -webkit-text-fill-color:transparent;'>Bahas Kebijakan</span>
    </div>
    <div style='color:#475569;font-size:0.75rem;margin-bottom:0.15rem;font-style:italic'>
    Built by Bahas Kebijakan</div>
    <div style='color:#334155;font-size:0.72rem;margin-bottom:1.5rem'>
    Bibliometric & Scientometric Analysis Suite</div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload your data file",
        type=['bib','ris','csv'],
        accept_multiple_files=True,
        help="Scopus .bib, .ris, or .csv — upload multiple files at once to merge automatically"
    )

    st.markdown("---")
    st.markdown("<div style='color:#475569;font-size:0.78rem'>⚙️ Analysis Settings</div>",
                unsafe_allow_html=True)
    top_n     = st.slider("Top N items in charts", 5, 30, 15)
    n_topics  = st.slider("Number of LDA topics", 2, 10, 5)
    min_cooc  = st.slider("Min. keyword co-occurrence", 1, 10, 2)
    dpi_label = st.selectbox("Download quality", ["600 DPI (publication)", "300 DPI (presentation)", "150 DPI (web)"],
                             index=0)
    DPI = 600 if '600' in dpi_label else 300 if '300' in dpi_label else 150

    st.markdown("---")
    st.markdown("""
    <div style='color:#374151;font-size:0.73rem;line-height:1.7'>
    📌 <b style='color:#64748B'>Quick guide:</b><br>
    1. Export your data from Scopus as <b style='color:#60A5FA'>.bib</b><br>
    2. Upload the file above<br>
    3. Explore results across tabs<br>
    4. Download charts at 600 DPI<br><br>
    💡 Upload <b style='color:#60A5FA'>multiple .bib files</b> at once — they will be merged automatically
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='color:#1E3A5F;font-size:0.7rem;text-align:center;line-height:1.6'>
    © Bahas Kebijakan<br>
    <span style='color:#334155'>All rights reserved</span>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style='display:flex;align-items:baseline;gap:12px;flex-wrap:wrap'>
    <div class='hero-title'>Bahas Kebijakan</div>
    <div style='font-family:DM Sans,sans-serif;font-size:1rem;color:#3B82F6;font-weight:500;
    padding:3px 12px;background:#1E3A5F;border-radius:20px;white-space:nowrap'>
    Bibliometric Suite</div>
</div>
<div class='hero-sub'>World-class bibliometric & scientometric analysis · Built by Bahas Kebijakan · Optimised for Scopus .bib exports</div>
""", unsafe_allow_html=True)

if not uploaded:
    st.markdown("""
    <div class='upload-hint'>
    📂 Upload your <b>.bib</b> (Scopus), <b>.ris</b>, or <b>.csv</b> file in the sidebar to start.<br>
    Multiple files are supported — they will be merged automatically.
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div style='background:#161B27;border:1px solid #1E2A40;border-radius:12px;padding:20px'>
        <div style='font-size:1.5rem'>📊</div>
        <div style='font-family:Syne,sans-serif;font-weight:700;color:#F1F5F9;margin:8px 0 4px'>
        14+ Analyses</div>
        <div style='color:#64748B;font-size:0.85rem'>Trends, Bradford's Law, Lotka's Law,
        citation impact, LDA topic modelling, research fronts, networks</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style='background:#161B27;border:1px solid #1E2A40;border-radius:12px;padding:20px'>
        <div style='font-size:1.5rem'>🔗</div>
        <div style='font-family:Syne,sans-serif;font-weight:700;color:#F1F5F9;margin:8px 0 4px'>
        Network Analysis</div>
        <div style='color:#64748B;font-size:0.85rem'>Co-authorship & keyword co-occurrence
        networks — static 600 DPI + interactive HTML</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div style='background:#161B27;border:1px solid #1E2A40;border-radius:12px;padding:20px'>
        <div style='font-size:1.5rem'>⬇️</div>
        <div style='font-family:Syne,sans-serif;font-weight:700;color:#F1F5F9;margin:8px 0 4px'>
        Download 600 DPI</div>
        <div style='color:#64748B;font-size:0.85rem'>Publication-ready charts,
        interactive HTML networks, and CSV tables</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── How to Get Data guide (shown on landing page too) ────────────────────
    st.markdown("""
    <div class='section-header'>📥 How to Export Data from Scopus</div>
    """, unsafe_allow_html=True)

    step_col1, step_col2 = st.columns([1, 1])

    with step_col1:
        st.markdown("""
        <div style='background:#161B27;border:1px solid #1E2A40;border-radius:12px;padding:24px;line-height:2'>
        <div style='font-family:Syne,sans-serif;font-weight:700;color:#60A5FA;margin-bottom:12px;font-size:1rem'>
        Step-by-step Guide</div>

        <div style='display:flex;gap:12px;margin-bottom:10px'>
            <div style='background:#1E3A5F;color:#60A5FA;font-weight:700;font-family:Syne,sans-serif;
            min-width:26px;height:26px;border-radius:50%;display:flex;align-items:center;
            justify-content:center;font-size:0.8rem;margin-top:2px'>1</div>
            <div style='color:#CBD5E1;font-size:0.88rem'>Go to <b style='color:#F1F5F9'>scopus.com</b>
            and sign in using your <b style='color:#60A5FA'>institutional email</b>
            (university or research institution account)</div>
        </div>

        <div style='display:flex;gap:12px;margin-bottom:10px'>
            <div style='background:#1E3A5F;color:#60A5FA;font-weight:700;font-family:Syne,sans-serif;
            min-width:26px;height:26px;border-radius:50%;display:flex;align-items:center;
            justify-content:center;font-size:0.8rem;margin-top:2px'>2</div>
            <div style='color:#CBD5E1;font-size:0.88rem'>In the search bar, enter your
            <b style='color:#F1F5F9'>keyword(s)</b> — e.g. <i>"multi-criteria analysis"</i> or
            <i>"AHP weighting"</i>. Apply filters (year range, document type, subject area) as needed</div>
        </div>

        <div style='display:flex;gap:12px;margin-bottom:10px'>
            <div style='background:#1E3A5F;color:#60A5FA;font-weight:700;font-family:Syne,sans-serif;
            min-width:26px;height:26px;border-radius:50%;display:flex;align-items:center;
            justify-content:center;font-size:0.8rem;margin-top:2px'>3</div>
            <div style='color:#CBD5E1;font-size:0.88rem'>Tick the checkbox at the top of the results list,
            then click <b style='color:#F1F5F9'>"Select all [N] documents"</b>
            (not just the current page)</div>
        </div>

        <div style='display:flex;gap:12px;margin-bottom:10px'>
            <div style='background:#1E3A5F;color:#60A5FA;font-weight:700;font-family:Syne,sans-serif;
            min-width:26px;height:26px;border-radius:50%;display:flex;align-items:center;
            justify-content:center;font-size:0.8rem;margin-top:2px'>4</div>
            <div style='color:#CBD5E1;font-size:0.88rem'>Click the <b style='color:#F1F5F9'>Export</b>
            button → choose <b style='color:#60A5FA'>BibTeX</b> format</div>
        </div>

        <div style='display:flex;gap:12px;margin-bottom:10px'>
            <div style='background:#1E3A5F;color:#60A5FA;font-weight:700;font-family:Syne,sans-serif;
            min-width:26px;height:26px;border-radius:50%;display:flex;align-items:center;
            justify-content:center;font-size:0.8rem;margin-top:2px'>5</div>
            <div style='color:#CBD5E1;font-size:0.88rem'>In the export dialog, make sure to check:
            <b style='color:#F1F5F9'>Citation information</b>,
            <b style='color:#F1F5F9'>Bibliographical information</b>,
            <b style='color:#F1F5F9'>Abstract & keywords</b>, and
            <b style='color:#F1F5F9'>Other information</b> (includes citation count)</div>
        </div>

        <div style='display:flex;gap:12px'>
            <div style='background:#1E3A5F;color:#60A5FA;font-weight:700;font-family:Syne,sans-serif;
            min-width:26px;height:26px;border-radius:50%;display:flex;align-items:center;
            justify-content:center;font-size:0.8rem;margin-top:2px'>6</div>
            <div style='color:#CBD5E1;font-size:0.88rem'>Click <b style='color:#F1F5F9'>Export</b>.
            You will get a <b style='color:#60A5FA'>.bib file</b> — upload it directly to this tool</div>
        </div>
        </div>
        """, unsafe_allow_html=True)

    with step_col2:
        st.markdown("""
        <div style='background:#161B27;border:1px solid #1E2A40;border-radius:12px;padding:24px'>
        <div style='font-family:Syne,sans-serif;font-weight:700;color:#60A5FA;margin-bottom:16px;font-size:1rem'>
        Important Notes</div>

        <div style='background:#0F1117;border-left:3px solid #F59E0B;border-radius:0 8px 8px 0;
        padding:12px 16px;margin-bottom:12px'>
            <div style='color:#FCD34D;font-weight:600;font-size:0.82rem;margin-bottom:4px'>⚠️ Batch export limit</div>
            <div style='color:#94A3B8;font-size:0.82rem'>Scopus exports a maximum of
            <b style='color:#F1F5F9'>2,000 records per file</b>. If your results exceed 2,000,
            split by year range and upload all batches here — they will be merged automatically.</div>
        </div>

        <div style='background:#0F1117;border-left:3px solid #3B82F6;border-radius:0 8px 8px 0;
        padding:12px 16px;margin-bottom:12px'>
            <div style='color:#60A5FA;font-weight:600;font-size:0.82rem;margin-bottom:4px'>💡 Institutional access required</div>
            <div style='color:#94A3B8;font-size:0.82rem'>Scopus is a subscription database.
            You must log in with an <b style='color:#F1F5F9'>institutional email</b>
            (university or research body). Personal Gmail will not work.</div>
        </div>

        <div style='background:#0F1117;border-left:3px solid #22C55E;border-radius:0 8px 8px 0;
        padding:12px 16px;margin-bottom:12px'>
            <div style='color:#4ADE80;font-weight:600;font-size:0.82rem;margin-bottom:4px'>✅ Supported formats</div>
            <div style='color:#94A3B8;font-size:0.82rem'>
            <b style='color:#F1F5F9'>.bib</b> — Scopus BibTeX (recommended)<br>
            <b style='color:#F1F5F9'>.ris</b> — RIS format<br>
            <b style='color:#F1F5F9'>.csv</b> — Scopus or Web of Science CSV</div>
        </div>

        <div style='background:#0F1117;border-left:3px solid #A78BFA;border-radius:0 8px 8px 0;
        padding:12px 16px'>
            <div style='color:#C4B5FD;font-weight:600;font-size:0.82rem;margin-bottom:4px'>🔑 Required fields for full analysis</div>
            <div style='color:#94A3B8;font-size:0.82rem'>For best results, make sure to include
            <b style='color:#F1F5F9'>Abstract & keywords</b> when exporting — these are needed
            for keyword analysis, LDA topic modelling, and research front mapping.</div>
        </div>
        </div>
        """, unsafe_allow_html=True)

    st.stop()


# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner("⏳ Loading and processing data..."):
    df, source_log = load_and_merge(uploaded)

if df.empty or len(df) == 0:
    st.error("No data could be parsed. Please check your file format.")
    st.stop()

# ── Source badge banner ────────────────────────────────────────────────────────
src_badge_map = {
    'scopus':     ('<span style="background:#1E3A5F;color:#60A5FA;padding:2px 10px;border-radius:12px;font-size:0.75rem;font-weight:700">Scopus .bib</span>', '🔵'),
    'dimensions': ('<span style="background:#14532D;color:#4ADE80;padding:2px 10px;border-radius:12px;font-size:0.75rem;font-weight:700">Dimensions .bib</span>', '🟢'),
    'ris':        ('<span style="background:#451A03;color:#FCD34D;padding:2px 10px;border-radius:12px;font-size:0.75rem;font-weight:700">RIS</span>', '🟡'),
    'csv':        ('<span style="background:#2E1065;color:#C4B5FD;padding:2px 10px;border-radius:12px;font-size:0.75rem;font-weight:700">CSV</span>', '🟣'),
    'generic':    ('<span style="background:#1E2A40;color:#94A3B8;padding:2px 10px;border-radius:12px;font-size:0.75rem;font-weight:700">BibTeX</span>', '⚪'),
}
badge_html = ' &nbsp;'.join(
    f"{src_badge_map.get(src, src_badge_map['generic'])[0]} <span style='color:#64748B;font-size:0.78rem'>{fname} ({n} records)</span>"
    for fname, src, n in source_log
)
st.markdown(f"""
<div style='background:#161B27;border:1px solid #1E2A40;border-radius:10px;
padding:12px 16px;margin-bottom:1rem;display:flex;align-items:center;gap:16px;flex-wrap:wrap'>
<span style='color:#4ADE80;font-weight:700'>✅ {len(df):,} records loaded</span>
&nbsp;·&nbsp; {badge_html}
</div>""", unsafe_allow_html=True)

# ── Dimensions-specific note ──────────────────────────────────────────────────
sources_detected = [src for _, src, _ in source_log]
if 'dimensions' in sources_detected:
    st.info("""ℹ️ **Dimensions export detected.** Note: Dimensions BIB files do not include citation counts
    or Open Access status — those columns will show as N/A. Keywords are also not exported by Dimensions,
    so keyword analysis will be limited. For full analysis, combine with a Scopus export of the same query.""")

# ── Pre-compute common values ─────────────────────────────────────────────────
total_pubs     = len(df)
total_cites    = df.cited_by.sum()
avg_cites      = df.cited_by.mean()
median_cites   = df.cited_by.median()
sc             = sorted(df.cited_by.dropna(), reverse=True)
h_index        = sum(1 for i,c in enumerate(sc,1) if c >= i)
unique_authors = len({a for lst in df.author_list for a in lst})
unique_jnls    = df.journal.nunique()
collab_rate    = (df.author_count > 1).mean() * 100
oa_rate        = (df.oa_status != 'Closed').mean() * 100
year_span      = f"{int(df.year.min())}–{int(df.year.max())}"

auth_cites = {}
for _, row in df.iterrows():
    c = row.cited_by if pd.notna(row.cited_by) else 0
    for a in row.author_list: auth_cites[a] = auth_cites.get(a, 0) + c


# ── KPI Dashboard ─────────────────────────────────────────────────────────────
st.success(f"✅ **{total_pubs:,} records** loaded from {len(uploaded)} file(s)")

kpis = [
    ("📄", "Total Publications", f"{total_pubs:,}"),
    ("📅", "Year Span",          year_span),
    ("✍️", "Unique Authors",     f"{unique_authors:,}"),
    ("📰", "Unique Journals",    f"{unique_jnls:,}"),
    ("📊", "Total Citations",    f"{total_cites:,.0f}"),
    ("⭐", "Mean Cites/Paper",   f"{avg_cites:.1f}"),
    ("📈", "Median Citations",   f"{median_cites:.0f}"),
    ("🏆", "Corpus H-Index",     f"{h_index}"),
    ("🤝", "Collaboration Rate", f"{collab_rate:.1f}%"),
    ("🔓", "Open Access Rate",   f"{oa_rate:.1f}%"),
]

kpi_html = "<div class='kpi-grid'>"
for icon, label, val in kpis:
    kpi_html += f"""<div class='kpi-card'>
        <div class='kpi-value'>{val}</div>
        <div class='kpi-label'>{icon} {label}</div></div>"""
kpi_html += "</div>"
st.markdown(kpi_html, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs([
    "📈 Trends",
    "📰 Journals",
    "✍️ Authors",
    "🌍 Country & OA",
    "🔑 Keywords",
    "📊 Citations",
    "🔗 Networks",
    "🧠 Topics (LDA)",
    "🚀 Research Fronts",
    "⬇️ Export All",
    "📥 How to Get Data",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — TRENDS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    section("Publication Trends & Growth Analysis")

    yc = df.groupby('year').size().reset_index(name='papers')
    yc['cumulative'] = yc.papers.cumsum()
    yc['yoy_pct']    = yc.papers.pct_change() * 100
    yc['rolling3']   = yc.papers.rolling(3, center=True).mean()

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle('Publication Trend Analysis', fontsize=14, fontweight='bold', color='#F1F5F9')

    ax = axes[0,0]
    ax.bar(yc.year, yc.papers, color=C['blue'], alpha=0.7, label='Annual output')
    ax.plot(yc.year, yc.rolling3, color=C['red'], lw=2.5, label='3-yr rolling avg')
    ax.set_title('Annual Publication Volume'); ax.set_xlabel('Year'); ax.set_ylabel('Papers')
    ax.legend(facecolor='#1E2A40', labelcolor='#CBD5E1')

    ax = axes[0,1]
    ax.fill_between(yc.year, yc.cumulative, alpha=0.3, color=C['green'])
    ax.plot(yc.year, yc.cumulative, color=C['green'], lw=2)
    ax.set_title('Cumulative Growth'); ax.set_xlabel('Year'); ax.set_ylabel('Cumulative Papers')

    ax = axes[1,0]
    cc = [C['green'] if v>=0 else C['red'] for v in yc.yoy_pct.fillna(0)]
    ax.bar(yc.year, yc.yoy_pct.fillna(0), color=cc, alpha=0.8)
    ax.axhline(0, color='#94A3B8', lw=0.8)
    ax.set_title('Year-on-Year Growth (%)'); ax.set_xlabel('Year'); ax.set_ylabel('Growth %')

    ax = axes[1,1]
    if df.document_type.notna().sum() > 0:
        dt = df.document_type.value_counts().head(8)
        ax.pie(dt.values, labels=dt.index, autopct='%1.1f%%', startangle=140,
               colors=plt.cm.Set2.colors[:len(dt)],
               textprops={'color':'#CBD5E1'})
        ax.set_title('Document Type Mix')

    plt.tight_layout()
    st.pyplot(fig)
    dl_btn("Download Trend Chart", fig, "trends.png")

    st.markdown("""
    <div class='insight-box'>
    <div class='insight-title'>💡 How to Read & Use This</div>
    <div class='insight-row'>
        <div class='insight-card'>
            <b>Annual Volume + Rolling Average</b><br>
            The bar chart shows yearly output. The red line smooths short-term noise —
            if it trends upward, the field is growing. A flat or declining line suggests
            saturation or shifting interest.
        </div>
        <div class='insight-card'>
            <b>Cumulative Growth Curve</b><br>
            A steep S-curve indicates an emerging field accelerating quickly.
            A curve that has plateaued suggests a mature field with stable output —
            useful context for positioning a new study.
        </div>
        <div class='insight-card'>
            <b>Year-on-Year Growth (%)</b><br>
            Green bars = growth years; red bars = decline. A surge of &gt;30%/yr
            signals a hot topic. Consecutive red bars may indicate the field peaked —
            or that interest has moved to a sub-topic.
        </div>
        <div class='insight-card'>
            <b>Document Type Mix</b><br>
            A high share of <b>Review</b> papers means the field is mature enough
            to warrant synthesis. Dominance of <b>Articles</b> signals active empirical
            work. Conference papers indicate fast-moving applied research.
        </div>
    </div>
    <div style='margin-top:12px;font-size:0.82rem;color:#64748B'>
    <b style='color:#94A3B8'>Use this for:</b>
    <span class='insight-tag tag-blue'>Justifying research novelty</span>
    <span class='insight-tag tag-green'>Selecting publication timing</span>
    <span class='insight-tag tag-amber'>Grant narrative (growing field)</span>
    <span class='insight-tag tag-purple'>Thesis background section</span>
    </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — JOURNALS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    section("Journal Analysis & Bradford's Law")

    jc  = df.journal.value_counts().dropna()
    bdf = jc.reset_index(); bdf.columns = ['journal','count']
    bdf['rank']    = range(1, len(bdf)+1)
    bdf['cum_pct'] = bdf['count'].cumsum() / bdf['count'].sum() * 100
    z1 = (bdf.cum_pct - 33.3).abs().idxmin()
    z2 = (bdf.cum_pct - 66.7).abs().idxmin()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    top_j = jc.head(top_n)
    axes[0].barh(top_j.index[::-1], top_j.values[::-1],
                 color=plt.cm.Blues(np.linspace(0.4, 0.9, len(top_j)))[::-1])
    axes[0].set_title(f"Top {top_n} Journals"); axes[0].set_xlabel('Papers')

    axes[1].semilogx(bdf['rank'], bdf['cum_pct'], color=C['blue'], lw=2)
    axes[1].axhline(33.3, color=C['amber'], ls='--', lw=1, label=f"Zone 1: {z1+1} journals")
    axes[1].axhline(66.7, color=C['red'],   ls='--', lw=1, label=f"Zone 2: {z2-z1} journals")
    axes[1].fill_between(bdf['rank'][:z1+1],   bdf.cum_pct[:z1+1],   alpha=0.1, color=C['amber'])
    axes[1].fill_between(bdf['rank'][z1:z2+1], bdf.cum_pct[z1:z2+1], alpha=0.07, color=C['red'])
    axes[1].set_title("Bradford's Law"); axes[1].set_xlabel('Journal Rank (log)')
    axes[1].set_ylabel('Cumulative %')
    axes[1].legend(facecolor='#1E2A40', labelcolor='#CBD5E1')
    plt.tight_layout(); st.pyplot(fig)
    dl_btn("Download Journal Chart", fig, "journals.png")

    st.markdown("""
    <div class='insight-box green'>
    <div class='insight-title green'>💡 How to Read & Use This</div>
    <div class='insight-row'>
        <div class='insight-card'>
            <b>Top Journals Bar Chart</b><br>
            These are the most productive venues in your corpus. They are your
            <b>primary submission targets</b> — journals already engaged with your topic
            and whose editors/reviewers are familiar with the discourse.
        </div>
        <div class='insight-card'>
            <b>Bradford's Law — Zone 1 (Core)</b><br>
            A small number of journals producing ~33% of all papers. These are the
            <b>specialist journals</b> of this field — highest relevance and acceptance
            likelihood for similar work.
        </div>
        <div class='insight-card'>
            <b>Bradford's Law — Zone 2 & 3</b><br>
            Zone 2 = mainstream journals with occasional coverage.
            Zone 3 = peripheral venues — useful if targeting
            interdisciplinary or applied audiences, or for finding underexplored outlets.
        </div>
        <div class='insight-card'>
            <b>Strategic Use</b><br>
            Cross-reference Zone 1 journals with their <b>Impact Factor / CiteScore</b>
            on the Scopus journal page. Prioritise Q1 journals in Zone 1 for flagship
            publications; use Zone 2 for faster turnaround.
        </div>
    </div>
    <div style='margin-top:12px;font-size:0.82rem;color:#64748B'>
    <b style='color:#94A3B8'>Use this for:</b>
    <span class='insight-tag tag-green'>Choosing submission venue</span>
    <span class='insight-tag tag-blue'>Literature review scope</span>
    <span class='insight-tag tag-amber'>Systematic review protocol</span>
    <span class='insight-tag tag-purple'>Field mapping</span>
    </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style='background:#161B27;border:1px solid #1E2A40;border-radius:10px;padding:16px;margin-top:1rem'>
    <b style='color:#60A5FA'>Bradford's Law Results:</b><br>
    <span style='color:#94A3B8'>Zone 1 (Core): <b style='color:#FCD34D'>{z1+1} journals</b> → {bdf.cum_pct[z1]:.1f}% of all papers<br>
    Zone 2: <b style='color:#F87171'>{z2-z1} journals</b><br>
    Zone 3 (Tail): <b style='color:#94A3B8'>{len(bdf)-z2} journals</b></span></div>""",
    unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — AUTHORS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    section("Author Analytics & Lotka's Law")

    all_authors = [a for lst in df.author_list for a in lst if a]
    auth_df = pd.DataFrame(Counter(all_authors).items(), columns=['author','papers'])\
                .sort_values('papers', ascending=False).reset_index(drop=True)
    auth_df['total_cites']     = auth_df.author.map(auth_cites).fillna(0).astype(int)
    auth_df['cites_per_paper'] = (auth_df.total_cites / auth_df.papers).round(1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    axes[0].barh(auth_df.head(top_n).author[::-1], auth_df.head(top_n).papers[::-1], color=C['blue'])
    axes[0].set_title(f'Top {top_n} Authors — Publications'); axes[0].set_xlabel('Papers')

    top_ac = auth_df.nlargest(top_n,'total_cites')
    axes[1].barh(top_ac.author[::-1], top_ac.total_cites[::-1], color=C['green'])
    axes[1].set_title(f'Top {top_n} Authors — Citations'); axes[1].set_xlabel('Total Citations')

    lotka_obs = auth_df.papers.value_counts().sort_index()
    n1        = (auth_df.papers == 1).sum()
    lotka_exp = pd.Series({n: n1/(n**2) for n in lotka_obs.index})
    axes[2].scatter(lotka_obs.index, lotka_obs.values, color=C['blue'], label='Observed', s=45, zorder=3)
    axes[2].plot(lotka_exp.index, lotka_exp.values, color=C['red'], ls='--', lw=2, label="Lotka's Law")
    axes[2].set_xscale('log'); axes[2].set_yscale('log')
    axes[2].set_title("Lotka's Law"); axes[2].set_xlabel('Papers (log)'); axes[2].set_ylabel('Authors (log)')
    axes[2].legend(facecolor='#1E2A40', labelcolor='#CBD5E1')

    plt.tight_layout(); st.pyplot(fig)
    dl_btn("Download Author Chart", fig, "authors.png")

    st.markdown("""
    <div class='insight-box purple'>
    <div class='insight-title purple'>💡 How to Read & Use This</div>
    <div class='insight-row'>
        <div class='insight-card'>
            <b>Top Authors by Publications</b><br>
            These are the most prolific contributors. They are likely to be
            <b>key researchers to follow</b> — check their recent work, Google Scholar
            profile, and affiliated institutions for research directions.
        </div>
        <div class='insight-card'>
            <b>Top Authors by Citations</b><br>
            High citation count indicates <b>influence</b>, not just productivity.
            Authors with fewer papers but high citations are producing seminal work —
            prioritise reading their papers first.
        </div>
        <div class='insight-card'>
            <b>Lotka's Law</b><br>
            If observed data (blue dots) follows the 1/n² curve closely, authorship
            is normally distributed. Deviation upward at n=1 means many one-time
            contributors — a sign of a broad, interdisciplinary field.
        </div>
        <div class='insight-card'>
            <b>Strategic Use</b><br>
            Authors in both top-publications and top-citations lists are
            <b>field leaders</b>. Consider them as: potential collaborators,
            peer reviewers who will likely review your paper, or
            benchmarks for your own research contribution.
        </div>
    </div>
    <div style='margin-top:12px;font-size:0.82rem;color:#64748B'>
    <b style='color:#94A3B8'>Use this for:</b>
    <span class='insight-tag tag-purple'>Finding collaborators</span>
    <span class='insight-tag tag-blue'>Building your reading list</span>
    <span class='insight-tag tag-green'>Identifying peer reviewers</span>
    <span class='insight-tag tag-amber'>Citation gap analysis</span>
    </div>
    </div>
    """, unsafe_allow_html=True)

    st.dataframe(
        auth_df.head(20).rename(columns={'author':'Author','papers':'Papers',
                                          'total_cites':'Total Citations','cites_per_paper':'Cites/Paper'}),
        use_container_width=True, hide_index=True
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — COUNTRY & OA
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    section("Country Output & Open Access Analysis")

    all_countries  = [c for lst in df.country_list for c in lst]
    c_df = pd.DataFrame(Counter(all_countries).items(), columns=['country','papers'])\
             .sort_values('papers', ascending=False).reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    if len(c_df) > 0:
        top_c = c_df.head(top_n)
        axes[0].barh(top_c.country[::-1], top_c.papers[::-1],
                     color=plt.cm.viridis(np.linspace(0.2,0.85,len(top_c))))
        axes[0].set_title(f'Top {top_n} Countries'); axes[0].set_xlabel('Papers')
    else:
        axes[0].text(0.5,0.5,'No country data\n(affiliations not in export)',
                     ha='center', va='center', transform=axes[0].transAxes, color='#94A3B8')

    oa_counts = df.oa_status.value_counts()
    col_map   = {'Gold OA':C['amber'],'Hybrid OA':C['blue'],'Green OA':C['green'],
                 'Open Access':C['purple'],'Closed':C['slate'],'Unknown':C['red']}
    bar_cols  = [col_map.get(s, C['slate']) for s in oa_counts.index]
    axes[1].bar(oa_counts.index, oa_counts.values, color=bar_cols, alpha=0.85)
    axes[1].set_title('Open Access Status (from Scopus note field)')
    axes[1].set_xlabel('OA Category'); axes[1].set_ylabel('Papers')
    for i,(cat,val) in enumerate(oa_counts.items()):
        axes[1].text(i, val+0.2, str(val), ha='center', fontsize=9,
                     fontweight='bold', color='#F1F5F9')

    plt.tight_layout(); st.pyplot(fig)
    dl_btn("Download Country & OA Chart", fig, "country_oa.png")

    st.markdown("""
    <div class='insight-box amber'>
    <div class='insight-title amber'>💡 How to Read & Use This</div>
    <div class='insight-row'>
        <div class='insight-card'>
            <b>Country Output</b><br>
            Dominant countries reflect where research funding, institutions, and
            expertise are concentrated. A <b>heavily skewed distribution</b> toward
            a few countries signals potential geographic bias in the literature —
            worth noting in a systematic review's limitations section.
        </div>
        <div class='insight-card'>
            <b>Geographic Gaps</b><br>
            If your study region is absent or underrepresented, that is a
            <b>research gap you can explicitly claim</b>. For instance, "no study
            has applied this method in Southeast Asia" is a strong justification
            for a new paper.
        </div>
        <div class='insight-card'>
            <b>Open Access Status</b><br>
            <b>Gold OA</b> = freely available, published in OA journal.<br>
            <b>Hybrid OA</b> = OA article in a subscription journal (APC paid).<br>
            <b>Green OA</b> = author's manuscript deposited in a repository.<br>
            <b>Closed</b> = paywalled — check if your institution has access.
        </div>
        <div class='insight-card'>
            <b>Strategic Use</b><br>
            High OA rate = the community values open dissemination. If you plan to
            publish, consider whether an OA route aligns with funder mandates
            (e.g. EU Horizon, NIH). Green OA (self-archiving) is free and
            available for most journals.
        </div>
    </div>
    <div style='margin-top:12px;font-size:0.82rem;color:#64748B'>
    <b style='color:#94A3B8'>Use this for:</b>
    <span class='insight-tag tag-amber'>Identifying geographic research gaps</span>
    <span class='insight-tag tag-green'>OA publishing decisions</span>
    <span class='insight-tag tag-blue'>Systematic review bias assessment</span>
    </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — KEYWORDS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    section("Keyword Intelligence — Author Keywords vs Index Keywords")

    auth_kw_all  = [k for lst in df.author_kw_list for k in lst]
    idx_kw_all   = [k for lst in df.index_kw_list  for k in lst]
    auth_kw_freq = Counter(auth_kw_all)
    idx_kw_freq  = Counter(idx_kw_all)
    akw_df = pd.DataFrame(auth_kw_freq.items(), columns=['keyword','count'])\
               .sort_values('count', ascending=False).reset_index(drop=True)
    ikw_df = pd.DataFrame(idx_kw_freq.items(), columns=['keyword','count'])\
               .sort_values('count', ascending=False).reset_index(drop=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Keyword Analysis', fontsize=13, fontweight='bold', color='#F1F5F9')

    axes[0,0].barh(akw_df.head(top_n).keyword[::-1], akw_df.head(top_n)['count'][::-1], color=C['blue'])
    axes[0,0].set_title('Top Author Keywords'); axes[0,0].set_xlabel('Frequency')

    axes[0,1].barh(ikw_df.head(top_n).keyword[::-1], ikw_df.head(top_n)['count'][::-1], color=C['green'])
    axes[0,1].set_title('Top Scopus Index Keywords'); axes[0,1].set_xlabel('Frequency')

    if auth_kw_freq:
        wc = WordCloud(width=700, height=320, background_color='#161B27',
                       colormap='Blues', max_words=80).generate_from_frequencies(auth_kw_freq)
        axes[1,0].imshow(wc, interpolation='bilinear'); axes[1,0].axis('off')
        axes[1,0].set_title('Author Keyword Cloud')
    if idx_kw_freq:
        wc2 = WordCloud(width=700, height=320, background_color='#161B27',
                        colormap='Greens', max_words=80).generate_from_frequencies(idx_kw_freq)
        axes[1,1].imshow(wc2, interpolation='bilinear'); axes[1,1].axis('off')
        axes[1,1].set_title('Index Keyword Cloud')

    plt.tight_layout(); st.pyplot(fig)
    dl_btn("Download Keyword Chart", fig, "keywords.png")

    st.markdown("""
    <div class='insight-box'>
    <div class='insight-title'>💡 How to Read & Use This</div>
    <div class='insight-row'>
        <div class='insight-card'>
            <b>Author Keywords (Blue)</b><br>
            These are concepts authors <i>chose</i> to describe their work — they reflect
            the <b>intent and framing</b> of the research community. High-frequency author
            keywords are the terms your own paper should use to be discoverable by peers.
        </div>
        <div class='insight-card'>
            <b>Scopus Index Keywords (Green)</b><br>
            Controlled vocabulary assigned by Scopus indexers. More standardised and
            stable over time. Use these when designing <b>systematic review search strings</b>
            to ensure comprehensive coverage across databases.
        </div>
        <div class='insight-card'>
            <b>Word Clouds</b><br>
            Word size = frequency. Terms in the centre of the cloud are the
            <b>conceptual core</b> of this literature. Peripheral terms may represent
            emerging niches or interdisciplinary bridges worth exploring.
        </div>
        <div class='insight-card'>
            <b>Strategic Use — Two-layer Search</b><br>
            Build your database search string by combining <b>author keywords</b>
            (conceptual terms) AND <b>index keywords</b> (controlled vocab).
            Example: ("multi-criteria" OR "MCDM") AND ("weighting" OR "AHP")
        </div>
    </div>
    <div style='margin-top:12px;font-size:0.82rem;color:#64748B'>
    <b style='color:#94A3B8'>Use this for:</b>
    <span class='insight-tag tag-blue'>Choosing paper keywords</span>
    <span class='insight-tag tag-green'>Systematic review search strings</span>
    <span class='insight-tag tag-amber'>Identifying core concepts</span>
    <span class='insight-tag tag-purple'>Finding adjacent topics</span>
    </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — CITATIONS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    section("Citation & Impact Analysis")

    cit_df = df[df.cited_by.notna()].sort_values('cited_by', ascending=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    nz = cit_df[cit_df.cited_by > 0].cited_by
    if len(nz) > 0:
        axes[0].hist(np.log10(nz+1), bins=min(40, len(nz)//2+1),
                     color=C['blue'], edgecolor='#0F1117', alpha=0.85)
    axes[0].axvline(np.log10(cit_df.cited_by.mean()+1),   color=C['red'],   ls='--',
                    label=f"Mean={cit_df.cited_by.mean():.1f}")
    axes[0].axvline(np.log10(cit_df.cited_by.median()+1), color=C['green'], ls='--',
                    label=f"Median={cit_df.cited_by.median():.0f}")
    axes[0].set_title('Citation Distribution (log₁₀)'); axes[0].set_xlabel('log₁₀(Cites+1)')
    axes[0].legend(facecolor='#1E2A40', labelcolor='#CBD5E1')

    sc_s = np.sort(cit_df.cited_by.values)
    cp   = np.arange(1, len(sc_s)+1) / len(sc_s)
    cc_s = np.cumsum(sc_s) / sc_s.sum()
    axes[1].plot(cp*100, cc_s*100, color=C['blue'], lw=2)
    axes[1].plot([0,100],[0,100], color='#94A3B8', ls='--', lw=1, label='Perfect equality')
    axes[1].fill_between(cp*100, cc_s*100, cp*100, alpha=0.1, color=C['blue'])
    idx90 = np.searchsorted(cp, 0.90)
    axes[1].annotate(f"Top 10% papers\n→ {(1-cc_s[idx90])*100:.0f}% of cites",
                     xy=(90, cc_s[idx90]*100), xytext=(50,20),
                     arrowprops=dict(arrowstyle='->', color='#94A3B8'),
                     color=C['red'], fontsize=9)
    axes[1].set_title('Citation Lorenz Curve'); axes[1].set_xlabel('% Papers')
    axes[1].set_ylabel('Cumulative Cites %')
    axes[1].legend(facecolor='#1E2A40', labelcolor='#CBD5E1')

    cyt = df.groupby('year')['cited_by'].agg(['mean','sum'])
    ax2 = axes[2].twinx()
    axes[2].bar(cyt.index, cyt['sum'], color=C['blue'], alpha=0.4, label='Total cites')
    ax2.plot(cyt.index, cyt['mean'], color=C['red'], lw=2, marker='o', ms=4, label='Mean/paper')
    axes[2].set_title('Citations by Year'); axes[2].set_xlabel('Year')
    axes[2].set_ylabel('Total Cites', color=C['blue'])
    ax2.set_ylabel('Mean Cites/Paper', color=C['red'])

    plt.tight_layout(); st.pyplot(fig)
    dl_btn("Download Citation Chart", fig, "citations.png")

    st.markdown("""
    <div class='insight-box red'>
    <div class='insight-title red'>💡 How to Read & Use This</div>
    <div class='insight-row'>
        <div class='insight-card'>
            <b>Citation Distribution</b><br>
            Most fields follow a <b>power law</b> — a few papers get most citations.
            If the distribution is very steep, the field has clear canonical papers
            that every new paper must cite. A flatter distribution suggests
            more distributed influence.
        </div>
        <div class='insight-card'>
            <b>Lorenz Curve & Inequality</b><br>
            The further the curve bends from the diagonal, the more
            <b>citation inequality</b> exists. If the top 10% of papers account for
            &gt;80% of citations, the field has a small set of highly influential works —
            read those first.
        </div>
        <div class='insight-card'>
            <b>Citations by Year</b><br>
            Recent years often show low citation counts — papers need time to accumulate
            citations. A high mean for older years indicates <b>classic foundational papers</b>
            in that period. Look for years with sudden spikes — they often correspond
            to breakthrough publications.
        </div>
        <div class='insight-card'>
            <b>Top Cited Papers Table</b><br>
            These are your <b>must-read papers</b>. They represent the intellectual
            foundation of the field. In your own paper, citing them signals to reviewers
            that you are familiar with the core literature. Also check their
            reference lists for earlier seminal work.
        </div>
    </div>
    <div style='margin-top:12px;font-size:0.82rem;color:#64748B'>
    <b style='color:#94A3B8'>Use this for:</b>
    <span class='insight-tag tag-red'>Building your reading list</span>
    <span class='insight-tag tag-blue'>Identifying seminal papers</span>
    <span class='insight-tag tag-green'>Benchmarking impact</span>
    <span class='insight-tag tag-amber'>Understanding field maturity</span>
    </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-header' style='font-size:1rem'>🏆 Top Cited Papers</div>",
                unsafe_allow_html=True)
    tc = cit_df[['title','year','journal','cited_by']].head(15).copy()
    tc['title'] = tc['title'].str[:70] + '...'
    st.dataframe(tc.rename(columns={'title':'Title','year':'Year',
                                     'journal':'Journal','cited_by':'Citations'}),
                 use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — NETWORKS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    section("Collaboration Networks")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("<b style='color:#60A5FA'>Author Co-authorship Network</b>", unsafe_allow_html=True)
        G = nx.Graph()
        for _, row in df.iterrows():
            auths = row.author_list
            if len(auths) > 1:
                for a1,a2 in combinations(auths,2):
                    if G.has_edge(a1,a2): G[a1][a2]['weight'] += 1
                    else:                 G.add_edge(a1,a2, weight=1)

        deg_c = dict(G.degree())
        btw_c = nx.betweenness_centrality(G, normalized=True, weight='weight') if G.number_of_nodes() > 0 else {}

        top50 = sorted(deg_c, key=deg_c.get, reverse=True)[:min(50, len(deg_c))]
        SG    = G.subgraph(top50)

        fig2, ax = plt.subplots(figsize=(8, 8))
        if SG.number_of_nodes() > 0:
            pos  = nx.spring_layout(SG, k=0.6, seed=42)
            nsz  = [SG.degree(n)*80+50 for n in SG.nodes()]
            ncol = [btw_c.get(n,0) for n in SG.nodes()]
            ew   = [SG[u][v].get('weight',1)*0.4 for u,v in SG.edges()]
            nx.draw_networkx_edges(SG, pos, ax=ax, width=ew, edge_color='#1E2A40', alpha=0.6)
            sc2 = nx.draw_networkx_nodes(SG, pos, ax=ax, node_size=nsz,
                                          node_color=ncol, cmap='YlOrRd', alpha=0.9)
            nx.draw_networkx_labels(SG, pos, ax=ax, font_size=6, font_color='#CBD5E1')
            plt.colorbar(sc2, ax=ax, label='Betweenness', shrink=0.6)
        ax.set_title('Author Network (Top 50)', color='#F1F5F9'); ax.axis('off')
        plt.tight_layout(); st.pyplot(fig2)
        dl_btn("Download Network Chart", fig2, "author_network.png")

    with col_b:
        st.markdown("<b style='color:#60A5FA'>Keyword Co-occurrence Network</b>", unsafe_allow_html=True)
        G_kw = nx.Graph()
        for kw_list in df.author_kw_list:
            if len(kw_list) > 1:
                for k1,k2 in combinations(kw_list,2):
                    if G_kw.has_edge(k1,k2): G_kw[k1][k2]['weight'] += 1
                    else:                    G_kw.add_edge(k1,k2, weight=1)
        G_kw = nx.Graph([(u,v,d) for u,v,d in G_kw.edges(data=True) if d['weight'] >= min_cooc])

        fig3, ax3 = plt.subplots(figsize=(8,8))
        if G_kw.number_of_nodes() > 0:
            kw_deg = dict(G_kw.degree())
            pos_kw = nx.spring_layout(G_kw, k=0.5, seed=42)
            node_sz = [kw_deg[n]*60+30 for n in G_kw.nodes()]
            nx.draw_networkx_edges(G_kw, pos_kw, ax=ax3,
                                   edge_color='#1E2A40', alpha=0.5)
            nx.draw_networkx_nodes(G_kw, pos_kw, ax=ax3,
                                   node_size=node_sz, node_color=C['purple'], alpha=0.85)
            nx.draw_networkx_labels(G_kw, pos_kw, ax=ax3, font_size=6.5, font_color='#CBD5E1')
        ax3.set_title(f'Keyword Network (co-occur ≥{min_cooc})', color='#F1F5F9'); ax3.axis('off')
        plt.tight_layout(); st.pyplot(fig3)
        dl_btn("Download Keyword Network Chart", fig3, "keyword_network.png")

    st.markdown("""
    <div class='insight-box purple'>
    <div class='insight-title purple'>💡 How to Read & Use This</div>
    <div class='insight-row'>
        <div class='insight-card'>
            <b>Co-authorship Network</b><br>
            <b>Node size</b> = number of collaborators (degree).
            <b>Node colour</b> = betweenness centrality — darker nodes act as bridges
            between different research groups. Bridge authors are valuable collaborators
            as they connect otherwise disconnected communities.
        </div>
        <div class='insight-card'>
            <b>Isolated Clusters</b><br>
            Disconnected groups of nodes represent research silos — teams that publish
            on the same topic but never collaborate. This is a
            <b>collaboration opportunity</b> and a gap you can point to in a review paper.
        </div>
        <div class='insight-card'>
            <b>Keyword Co-occurrence Network</b><br>
            Two keywords linked by an edge appear together in the same paper.
            <b>Dense clusters</b> = coherent sub-themes.
            <b>Keywords bridging clusters</b> = integrative concepts that span
            multiple sub-fields — often the most impactful contribution space.
        </div>
        <div class='insight-card'>
            <b>Strategic Use</b><br>
            Use the keyword network to identify <b>concept combinations</b> that are
            underexplored (pairs of keywords that should connect but don't yet).
            Use the author network to find collaboration pathways to reach
            researchers outside your immediate circle.
        </div>
    </div>
    <div style='margin-top:12px;font-size:0.82rem;color:#64748B'>
    <b style='color:#94A3B8'>Use this for:</b>
    <span class='insight-tag tag-purple'>Finding collaborators</span>
    <span class='insight-tag tag-blue'>Identifying concept gaps</span>
    <span class='insight-tag tag-green'>Mapping sub-field structure</span>
    <span class='insight-tag tag-amber'>Spotting research silos</span>
    </div>
    </div>
    """, unsafe_allow_html=True)

    # Interactive PyVis
    st.markdown("---")
    if st.button("🕸️ Generate Interactive Network (HTML)", use_container_width=True):
        with st.spinner("Building interactive network..."):
            deg_thresh = max(1, int(np.percentile(list(deg_c.values()), 70))) if deg_c else 1
            SG2 = G.subgraph([n for n,d in deg_c.items() if d >= deg_thresh])
            nv  = Network(height='600px', width='100%', bgcolor='#0F1117',
                          font_color='#CBD5E1', notebook=False)
            nv.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=120)
            for node in SG2.nodes():
                d = SG2.degree(node); b = btw_c.get(node,0)
                nv.add_node(node, label=node, size=d*4+6,
                            color=f'rgba(59,130,246,{min(0.3+b*15,1):.2f})',
                            title=f'{node}\nDegree: {d}\nBetweenness: {b:.4f}')
            for u,v,d in SG2.edges(data=True):
                nv.add_edge(u,v,width=d.get('weight',1)*0.6)
            with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as tmp:
                nv.save_graph(tmp.name)
                with open(tmp.name, 'r') as f: html_str = f.read()
            st.download_button("⬇️ Download Interactive Network (HTML)",
                               data=html_str.encode(),
                               file_name="author_network_interactive.html",
                               mime='text/html', use_container_width=True)
            st.components.v1.html(html_str, height=620)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 8 — LDA
# ══════════════════════════════════════════════════════════════════════════════
with tabs[7]:
    section("LDA Topic Modelling from Abstracts")

    abstracts = df.abstract.dropna().astype(str)
    abstracts = abstracts[abstracts.str.len() > 80].reset_index(drop=True)

    if len(abstracts) < 5:
        st.warning("Terlalu sedikit abstrak (< 5). Upload lebih banyak data.")
    else:
        st.info(f"Fitting LDA dengan **{n_topics} topik** pada **{len(abstracts)}** abstrak...")
        vec   = CountVectorizer(max_df=0.9, min_df=2, max_features=1500,
                                stop_words='english', ngram_range=(1,2))
        try:
            dtm   = vec.fit_transform(abstracts)
            vocab = np.array(vec.get_feature_names_out())
            lda   = LatentDirichletAllocation(n_components=n_topics, max_iter=30,
                                               learning_method='online', random_state=42)
            lda.fit(dtm)
            topics = {i: vocab[comp.argsort()[::-1][:12]].tolist()
                      for i,comp in enumerate(lda.components_)}

            n_cols = min(3, n_topics)
            n_rows = (n_topics + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*5, n_rows*4))
            axes = axes.flatten() if n_topics > 1 else [axes]
            for i in range(n_topics):
                words  = topics[i]
                scores = sorted(lda.components_[i], reverse=True)[:12]
                norm   = np.array(scores) / np.sum(scores)
                axes[i].barh(words[::-1], norm[::-1], color=plt.cm.tab10(i/n_topics))
                axes[i].set_title(f'Topic {i+1}', fontweight='bold', color='#F1F5F9')
                axes[i].set_xlabel('Relative Weight')
            for j in range(n_topics, len(axes)): axes[j].set_visible(False)
            fig.suptitle(f'LDA Topic Model ({n_topics} Topics)', fontsize=13,
                         fontweight='bold', color='#F1F5F9')
            plt.tight_layout(); st.pyplot(fig)
            dl_btn("Download LDA Chart", fig, "lda_topics.png")

            st.markdown("""
            <div class='insight-box green'>
            <div class='insight-title green'>💡 How to Read & Use This</div>
            <div class='insight-row'>
                <div class='insight-card'>
                    <b>Each Topic Bar Chart</b><br>
                    Words with higher relative weight define that topic's theme.
                    Read the top 5–6 words together as a concept cluster and give it
                    a human-readable label — e.g. "Topic 2: GIS-based site selection".
                    That label is a <b>sub-theme</b> of your research field.
                </div>
                <div class='insight-card'>
                    <b>Topic Share (%)</b><br>
                    Shows how many abstracts are dominated by each topic.
                    A topic with &gt;40% share is the <b>mainstream strand</b>.
                    Topics with &lt;10% share are niche or emerging — potentially
                    higher novelty for a new paper.
                </div>
                <div class='insight-card'>
                    <b>Interpreting Overlap</b><br>
                    If two topics share many words, they may be the same sub-theme
                    split across the model. Try reducing the number of topics
                    in the sidebar slider to merge them into a cleaner picture.
                </div>
                <div class='insight-card'>
                    <b>Strategic Use</b><br>
                    Map your own paper's abstract onto these topics — which one does it
                    fit? If your work spans two topics that rarely co-occur, that is a
                    <b>novelty claim</b>: "We integrate [Topic A] with [Topic B],
                    an approach absent in prior literature."
                </div>
            </div>
            <div style='margin-top:12px;font-size:0.82rem;color:#64748B'>
            <b style='color:#94A3B8'>Use this for:</b>
            <span class='insight-tag tag-green'>Structuring literature review chapters</span>
            <span class='insight-tag tag-blue'>Identifying research sub-themes</span>
            <span class='insight-tag tag-amber'>Claiming novelty at intersection of topics</span>
            </div>
            </div>
            """, unsafe_allow_html=True)

            dom = Counter(lda.transform(dtm).argmax(axis=1))
            st.markdown("**Topic Share:**")
            for t in sorted(dom):
                pct = dom[t]/len(abstracts)*100
                st.markdown(f"- **Topic {t+1}** ({pct:.1f}%): {', '.join(topics[t][:5])}")
        except Exception as ex:
            st.error(f"LDA error: {ex}. Try reducing the number of topics or upload more data.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 9 — RESEARCH FRONTS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[8]:
    section("Research Front Map — Emerging vs Established Themes")

    kw_recs = [{'keyword':kw,'year':row.year}
               for _,row in df.iterrows() if pd.notna(row.year)
               for kw in row.author_kw_list]

    if len(kw_recs) < 5:
        st.warning("Tidak cukup data keyword-tahun. Pastikan abstrak dan keyword tersedia.")
    else:
        kw_temp  = pd.DataFrame(kw_recs)
        kw_stats = kw_temp.groupby('keyword').agg(
            count=('year','count'), mean_year=('year','mean')).reset_index()
        cutoff   = df.year.max() - 2
        recent   = kw_temp[kw_temp.year >= cutoff].groupby('keyword').size().rename('recent_n')
        kw_stats = kw_stats.merge(recent, on='keyword', how='left').fillna(0)
        kw_stats['recent_pct'] = kw_stats.recent_n / kw_stats['count']
        kw_stats = kw_stats[kw_stats['count'] >= 2].copy()

        if len(kw_stats) > 0:
            med_y = kw_stats.mean_year.median()
            med_r = kw_stats.recent_pct.median()

            def classify(r):
                if   r.mean_year>=med_y and r.recent_pct>=med_r: return 'Emerging Front'
                elif r.mean_year>=med_y and r.recent_pct< med_r: return 'New but Niche'
                elif r.mean_year< med_y and r.recent_pct>=med_r: return 'Classic Resurgent'
                else:                                             return 'Established'

            kw_stats['category'] = kw_stats.apply(classify, axis=1)
            cat_col = {'Emerging Front':C['red'],'New but Niche':C['amber'],
                       'Classic Resurgent':C['purple'],'Established':C['slate']}

            fig, ax = plt.subplots(figsize=(13, 8))
            for cat, grp in kw_stats.groupby('category'):
                ax.scatter(grp.mean_year, grp.recent_pct,
                           s=np.sqrt(grp['count'])*40, color=cat_col[cat],
                           label=cat, alpha=0.75, edgecolors='#0F1117', lw=0.5)

            top_em = kw_stats[kw_stats.category=='Emerging Front'].nlargest(12,'count')
            for _,r in top_em.iterrows():
                ax.annotate(r.keyword, (r.mean_year, r.recent_pct),
                            fontsize=7.5, color='#FCA5A5',
                            xytext=(4,4), textcoords='offset points')

            ax.axvline(med_y, color='#334155', ls='--', lw=1)
            ax.axhline(med_r, color='#334155', ls='--', lw=1)
            ax.set_xlabel('Mean Year'); ax.set_ylabel('Recent Activity (last 2yr %)')
            ax.set_title('Research Front Map · Bubble size = frequency', color='#F1F5F9')
            ax.legend(facecolor='#161B27', labelcolor='#CBD5E1', title='Category',
                      title_fontsize=9, loc='lower right')
            plt.tight_layout(); st.pyplot(fig)
            dl_btn("Download Research Front Map", fig, "research_fronts.png")

            st.markdown("""
            <div class='insight-box red'>
            <div class='insight-title red'>💡 How to Read & Use This</div>
            <div class='insight-row'>
                <div class='insight-card'>
                    <b>Emerging Front (top-right)</b><br>
                    Keywords that are <b>recent AND actively growing</b>. These are the
                    hottest topics right now. A paper contributing to an Emerging Front
                    has higher novelty value and is more likely to attract citations
                    quickly after publication.
                </div>
                <div class='insight-card'>
                    <b>Established (bottom-left)</b><br>
                    Keywords with long history and stable recent activity —
                    the foundational concepts. Your paper needs to
                    <b>cite this literature</b> to anchor itself, but contributing
                    here alone offers limited novelty.
                </div>
                <div class='insight-card'>
                    <b>Classic Resurgent (bottom-right)</b><br>
                    Older concepts gaining renewed interest. Often signals a
                    <b>methodological revival</b> — a classic method applied to
                    new domains. High opportunity for comparative or replication studies.
                </div>
                <div class='insight-card'>
                    <b>New but Niche (top-left)</b><br>
                    Recent concepts that haven't gained traction yet —
                    possibly <b>ahead of their time</b>, or concepts that didn't
                    resonate broadly. Worth monitoring but higher-risk to base a paper on.
                </div>
            </div>
            <div style='margin-top:12px;font-size:0.82rem;color:#64748B'>
            <b style='color:#94A3B8'>Use this for:</b>
            <span class='insight-tag tag-red'>Positioning paper contribution</span>
            <span class='insight-tag tag-blue'>Choosing research direction</span>
            <span class='insight-tag tag-green'>Writing the "gap" section</span>
            <span class='insight-tag tag-amber'>Forecasting where the field is heading</span>
            </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("**🚀 Top Emerging Research Fronts:**")
            for _,r in top_em.head(8).iterrows():
                st.markdown(f"- **{r.keyword}** — freq: {int(r['count'])}, mean year: {r.mean_year:.1f}, recent: {r.recent_pct*100:.0f}%")



# ══════════════════════════════════════════════════════════════════════════════
# TAB 11 — EXPORT ALL
# ══════════════════════════════════════════════════════════════════════════════
with tabs[9]:
    section("Export All Results")

    st.markdown("""
    <div style='background:#161B27;border:1px solid #1E2A40;border-radius:10px;
    padding:16px;margin-bottom:1rem;color:#94A3B8;font-size:0.88rem'>
    Download all analysis results in a single ZIP file — all tables as CSV,
    ready to open in Excel or import into your report.
    </div>""", unsafe_allow_html=True)

    if st.button("📦 Generate & Download All CSVs (ZIP)", use_container_width=True):
        with st.spinner("Preparing all files..."):
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as zf:

                # KPI summary
                kpi_df = pd.DataFrame({'Metric':['Total Publications','Year Range','Unique Authors',
                    'Unique Journals','Total Citations','Mean Cites','Median Cites','H-Index',
                    'Collaboration Rate','OA Rate'],
                    'Value':[total_pubs, year_span, unique_authors, unique_jnls,
                             f'{total_cites:.0f}', f'{avg_cites:.1f}', f'{median_cites:.0f}',
                             h_index, f'{collab_rate:.1f}%', f'{oa_rate:.1f}%']})
                zf.writestr('kpi_summary.csv', kpi_df.to_csv(index=False))

                # Authors
                auth_df2 = pd.DataFrame(Counter([a for lst in df.author_list for a in lst if a]).items(),
                                        columns=['author','papers'])\
                             .sort_values('papers', ascending=False)
                auth_df2['total_cites'] = auth_df2.author.map(auth_cites).fillna(0).astype(int)
                zf.writestr('author_rankings.csv', auth_df2.to_csv(index=False))

                # Journals
                zf.writestr('journal_rankings.csv',
                            jc.reset_index().rename(columns={'index':'journal',0:'papers'}).to_csv(index=False))

                # Keywords
                zf.writestr('author_keyword_freq.csv',
                            pd.DataFrame(Counter([k for lst in df.author_kw_list for k in lst]).items(),
                                         columns=['keyword','count']).sort_values('count',ascending=False)\
                              .to_csv(index=False))
                zf.writestr('index_keyword_freq.csv',
                            pd.DataFrame(Counter([k for lst in df.index_kw_list for k in lst]).items(),
                                         columns=['keyword','count']).sort_values('count',ascending=False)\
                              .to_csv(index=False))

                # Trends
                zf.writestr('publication_trends.csv',
                            df.groupby('year').size().reset_index(name='papers').to_csv(index=False))

                # Countries
                c_df2 = pd.DataFrame(Counter([c for lst in df.country_list for c in lst]).items(),
                                     columns=['country','papers']).sort_values('papers',ascending=False)
                zf.writestr('country_output.csv', c_df2.to_csv(index=False))

            zip_buf.seek(0)
            st.download_button(
                label="⬇️ Download ZIP (all CSVs)",
                data=zip_buf.read(),
                file_name="bibliometric_results.zip",
                mime="application/zip",
                use_container_width=True,
            )
        st.success("✅ All files ready!")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 11 — HOW TO GET DATA
# ══════════════════════════════════════════════════════════════════════════════
with tabs[10]:
    section("📥 How to Export Data — Scopus & Dimensions")

    st.markdown("""
    <div style='color:#64748B;font-size:0.9rem;margin-bottom:1.5rem'>
    This tool supports exports from <b style='color:#60A5FA'>Scopus</b> and
    <b style='color:#4ADE80'>Dimensions</b>. Click the tab below for your database.
    </div>
    """, unsafe_allow_html=True)

    guide_tab1, guide_tab2, guide_tab3 = st.tabs(["🔵 Scopus (Recommended)", "🟢 Dimensions (Free)", "📊 Comparison"])

    with guide_tab2:
        st.markdown("""
        <div style='background:#0D1F0D;border:1px solid #14532D;border-radius:10px;
        padding:14px 18px;margin-bottom:1rem;font-size:0.83rem;color:#4ADE80'>
        ✅ <b>Dimensions is free to access</b> — no institutional subscription required.
        Sign up at <b>app.dimensions.ai</b> with any email.
        </div>
        """, unsafe_allow_html=True)

        d_col1, d_col2 = st.columns([1, 1])
        with d_col1:
            st.markdown("""
            <div style='background:#161B27;border:1px solid #1E2A40;border-radius:12px;padding:22px;line-height:2'>
            <div style='font-family:Syne,sans-serif;font-weight:700;color:#4ADE80;margin-bottom:14px;font-size:1rem'>
            Dimensions Export Guide</div>

            <div style='display:flex;gap:12px;margin-bottom:12px'>
                <div style='background:#14532D;color:#4ADE80;font-weight:700;font-family:Syne,sans-serif;
                min-width:26px;height:26px;border-radius:50%;display:flex;align-items:center;
                justify-content:center;font-size:0.8rem;flex-shrink:0'>1</div>
                <div style='color:#CBD5E1;font-size:0.87rem'>Go to <b style='color:#F1F5F9'>app.dimensions.ai</b>
                and sign in (free account) or use your institutional login</div>
            </div>
            <div style='display:flex;gap:12px;margin-bottom:12px'>
                <div style='background:#14532D;color:#4ADE80;font-weight:700;font-family:Syne,sans-serif;
                min-width:26px;height:26px;border-radius:50%;display:flex;align-items:center;
                justify-content:center;font-size:0.8rem;flex-shrink:0'>2</div>
                <div style='color:#CBD5E1;font-size:0.87rem'>Click <b style='color:#F1F5F9'>Publications</b>
                in the top menu, then enter your search keyword(s) in the search bar</div>
            </div>
            <div style='display:flex;gap:12px;margin-bottom:12px'>
                <div style='background:#14532D;color:#4ADE80;font-weight:700;font-family:Syne,sans-serif;
                min-width:26px;height:26px;border-radius:50%;display:flex;align-items:center;
                justify-content:center;font-size:0.8rem;flex-shrink:0'>3</div>
                <div style='color:#CBD5E1;font-size:0.87rem'>Apply filters as needed
                (year range, publication type, research category, open access)</div>
            </div>
            <div style='display:flex;gap:12px;margin-bottom:12px'>
                <div style='background:#14532D;color:#4ADE80;font-weight:700;font-family:Syne,sans-serif;
                min-width:26px;height:26px;border-radius:50%;display:flex;align-items:center;
                justify-content:center;font-size:0.8rem;flex-shrink:0'>4</div>
                <div style='color:#CBD5E1;font-size:0.87rem'>Click the
                <b style='color:#F1F5F9'>Save / Export</b> button (top right of results) →
                select <b style='color:#4ADE80'>Export results</b></div>
            </div>
            <div style='display:flex;gap:12px;margin-bottom:12px'>
                <div style='background:#14532D;color:#4ADE80;font-weight:700;font-family:Syne,sans-serif;
                min-width:26px;height:26px;border-radius:50%;display:flex;align-items:center;
                justify-content:center;font-size:0.8rem;flex-shrink:0'>5</div>
                <div style='color:#CBD5E1;font-size:0.87rem'>Choose format:
                <b style='color:#4ADE80'>BibTeX</b> → select fields:
                <b style='color:#F1F5F9'>All available fields</b> (includes abstract)</div>
            </div>
            <div style='display:flex;gap:12px'>
                <div style='background:#14532D;color:#4ADE80;font-weight:700;font-family:Syne,sans-serif;
                min-width:26px;height:26px;border-radius:50%;display:flex;align-items:center;
                justify-content:center;font-size:0.8rem;flex-shrink:0'>6</div>
                <div style='color:#CBD5E1;font-size:0.87rem'>Click <b style='color:#F1F5F9'>Export</b> —
                you will get a <b style='color:#4ADE80'>.bib file</b>.
                Upload it directly to this tool. It will be auto-detected as a Dimensions export.</div>
            </div>
            </div>
            """, unsafe_allow_html=True)

        with d_col2:
            st.markdown("""
            <div style='background:#161B27;border:1px solid #1E2A40;border-radius:12px;padding:22px'>
            <div style='font-family:Syne,sans-serif;font-weight:700;color:#4ADE80;margin-bottom:14px;font-size:1rem'>
            What Dimensions BIB includes</div>

            <div style='background:#0F1117;border-left:3px solid #22C55E;border-radius:0 8px 8px 0;
            padding:11px 14px;margin-bottom:10px'>
                <div style='color:#4ADE80;font-weight:600;font-size:0.82rem;margin-bottom:4px'>✅ Available</div>
                <div style='color:#94A3B8;font-size:0.82rem;line-height:1.7'>
                Title · Authors · Year · Journal<br>
                DOI · Abstract · URL · Volume / Issue / Pages</div>
            </div>

            <div style='background:#0F1117;border-left:3px solid #EF4444;border-radius:0 8px 8px 0;
            padding:11px 14px;margin-bottom:10px'>
                <div style='color:#FCA5A5;font-weight:600;font-size:0.82rem;margin-bottom:4px'>❌ Not in Dimensions BIB</div>
                <div style='color:#94A3B8;font-size:0.82rem;line-height:1.7'>
                <b style='color:#F1F5F9'>Citation count</b> — not exported (will show N/A)<br>
                <b style='color:#F1F5F9'>Keywords</b> — field exists but is always empty<br>
                <b style='color:#F1F5F9'>Affiliations</b> — not in BIB format<br>
                <b style='color:#F1F5F9'>Open Access status</b> — not exported</div>
            </div>

            <div style='background:#0F1117;border-left:3px solid #F59E0B;border-radius:0 8px 8px 0;
            padding:11px 14px'>
                <div style='color:#FCD34D;font-weight:600;font-size:0.82rem;margin-bottom:4px'>💡 Export limit</div>
                <div style='color:#94A3B8;font-size:0.82rem;line-height:1.7'>
                Free Dimensions accounts: <b style='color:#F1F5F9'>500 records per export</b>.
                For larger sets, split by year and upload multiple files — they will be merged automatically.</div>
            </div>
            </div>
            """, unsafe_allow_html=True)

    with guide_tab3:
        st.markdown("""
        <div style='background:#161B27;border:1px solid #1E2A40;border-radius:12px;padding:22px'>
        <div style='font-family:Syne,sans-serif;font-weight:700;color:#F1F5F9;margin-bottom:16px;font-size:1rem'>
        Scopus vs Dimensions — Which should you use?</div>
        """, unsafe_allow_html=True)

        comp_data = {
            'Feature': ['Access', 'Records limit/export', 'Citation count in BIB', 'Keywords in BIB',
                        'Affiliations in BIB', 'OA status in BIB', 'Abstract', 'Coverage'],
            'Scopus 🔵': ['Institutional subscription required', '2,000 per file', '✅ Yes (in note field)',
                          '✅ Yes (author + index)', '✅ Yes', '✅ Yes (Gold/Hybrid/Green)',
                          '✅ Yes', 'Peer-reviewed journals & conferences'],
            'Dimensions 🟢': ['Free (with account)', '500 per file (free tier)', '❌ Not in BIB',
                               '❌ Always empty', '❌ Not in BIB', '❌ Not in BIB',
                               '✅ Yes', 'Broader: journals, books, datasets, patents'],
        }
        st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)
        st.markdown("""
        <div style='color:#64748B;font-size:0.82rem;margin-top:12px;line-height:1.7'>
        <b style='color:#94A3B8'>Recommendation:</b> Use <b style='color:#60A5FA'>Scopus</b> when you have institutional access
        — it provides richer metadata for citation and keyword analysis.
        Use <b style='color:#4ADE80'>Dimensions</b> as a free alternative or to supplement with broader coverage.
        You can <b style='color:#F1F5F9'>upload both together</b> — this tool merges them automatically.
        </div>
        </div>
        """, unsafe_allow_html=True)

    with guide_tab1:
        col_l, col_r = st.columns([1, 1])
    with col_l:
        st.markdown("""
        <div style='background:#161B27;border:1px solid #1E2A40;border-radius:12px;padding:24px;line-height:2'>
        <div style='font-family:Syne,sans-serif;font-weight:700;color:#60A5FA;margin-bottom:16px;font-size:1rem'>
        Step-by-step Export Guide</div>

        <div style='display:flex;gap:12px;margin-bottom:14px'>
            <div style='background:#1E3A5F;color:#60A5FA;font-weight:700;font-family:Syne,sans-serif;
            min-width:28px;height:28px;border-radius:50%;display:flex;align-items:center;
            justify-content:center;font-size:0.82rem;flex-shrink:0;margin-top:1px'>1</div>
            <div style='color:#CBD5E1;font-size:0.88rem'>Go to
            <b style='color:#F1F5F9'>scopus.com</b> and sign in with your
            <b style='color:#60A5FA'>institutional email</b>
            (university or research institution account required)</div>
        </div>

        <div style='display:flex;gap:12px;margin-bottom:14px'>
            <div style='background:#1E3A5F;color:#60A5FA;font-weight:700;font-family:Syne,sans-serif;
            min-width:28px;height:28px;border-radius:50%;display:flex;align-items:center;
            justify-content:center;font-size:0.82rem;flex-shrink:0;margin-top:1px'>2</div>
            <div style='color:#CBD5E1;font-size:0.88rem'>Type your search keyword(s) in the search bar —
            e.g. <i style='color:#94A3B8'>"multi-criteria analysis"</i>,
            <i style='color:#94A3B8'>"AHP weighting"</i>,
            <i style='color:#94A3B8'>"bibliometric"</i>.
            Apply filters for year range, document type, or subject area as needed</div>
        </div>

        <div style='display:flex;gap:12px;margin-bottom:14px'>
            <div style='background:#1E3A5F;color:#60A5FA;font-weight:700;font-family:Syne,sans-serif;
            min-width:28px;height:28px;border-radius:50%;display:flex;align-items:center;
            justify-content:center;font-size:0.82rem;flex-shrink:0;margin-top:1px'>3</div>
            <div style='color:#CBD5E1;font-size:0.88rem'>Tick the checkbox at the top of the results list →
            click <b style='color:#F1F5F9'>"Select all [N] documents"</b>
            (not just the ones on the current page)</div>
        </div>

        <div style='display:flex;gap:12px;margin-bottom:14px'>
            <div style='background:#1E3A5F;color:#60A5FA;font-weight:700;font-family:Syne,sans-serif;
            min-width:28px;height:28px;border-radius:50%;display:flex;align-items:center;
            justify-content:center;font-size:0.82rem;flex-shrink:0;margin-top:1px'>4</div>
            <div style='color:#CBD5E1;font-size:0.88rem'>Click the <b style='color:#F1F5F9'>Export</b>
            button → select format: <b style='color:#60A5FA'>BibTeX</b></div>
        </div>

        <div style='display:flex;gap:12px;margin-bottom:14px'>
            <div style='background:#1E3A5F;color:#60A5FA;font-weight:700;font-family:Syne,sans-serif;
            min-width:28px;height:28px;border-radius:50%;display:flex;align-items:center;
            justify-content:center;font-size:0.82rem;flex-shrink:0;margin-top:1px'>5</div>
            <div style='color:#CBD5E1;font-size:0.88rem'>In the export dialog, check all these fields:<br>
            <span style='color:#4ADE80'>✓</span> <b style='color:#F1F5F9'>Citation information</b><br>
            <span style='color:#4ADE80'>✓</span> <b style='color:#F1F5F9'>Bibliographical information</b><br>
            <span style='color:#4ADE80'>✓</span> <b style='color:#F1F5F9'>Abstract & keywords</b>
            <span style='color:#F59E0B;font-size:0.78rem'> ← important for keyword & topic analysis</span><br>
            <span style='color:#4ADE80'>✓</span> <b style='color:#F1F5F9'>Other information</b>
            <span style='color:#F59E0B;font-size:0.78rem'> ← includes citation count</span></div>
        </div>

        <div style='display:flex;gap:12px'>
            <div style='background:#1E3A5F;color:#60A5FA;font-weight:700;font-family:Syne,sans-serif;
            min-width:28px;height:28px;border-radius:50%;display:flex;align-items:center;
            justify-content:center;font-size:0.82rem;flex-shrink:0;margin-top:1px'>6</div>
            <div style='color:#CBD5E1;font-size:0.88rem'>Click <b style='color:#F1F5F9'>Export</b>.
            A <b style='color:#60A5FA'>.bib file</b> will download automatically →
            upload it in the sidebar to start your analysis</div>
        </div>
        </div>
        """, unsafe_allow_html=True)

    with col_r:
        st.markdown("""
        <div style='background:#161B27;border:1px solid #1E2A40;border-radius:12px;padding:24px;'>
        <div style='font-family:Syne,sans-serif;font-weight:700;color:#60A5FA;margin-bottom:16px;font-size:1rem'>
        Important Notes</div>

        <div style='background:#0F1117;border-left:3px solid #F59E0B;border-radius:0 8px 8px 0;
        padding:12px 16px;margin-bottom:12px'>
            <div style='color:#FCD34D;font-weight:600;font-size:0.82rem;margin-bottom:5px'>
            ⚠️  2,000 records per file limit</div>
            <div style='color:#94A3B8;font-size:0.82rem;line-height:1.6'>
            Scopus exports a maximum of <b style='color:#F1F5F9'>2,000 records per file</b>.
            If your search returns more, split the export by year range
            (e.g. 2010–2015, 2016–2020, 2021–2026), then upload all files here at once —
            they will be merged automatically.</div>
        </div>

        <div style='background:#0F1117;border-left:3px solid #3B82F6;border-radius:0 8px 8px 0;
        padding:12px 16px;margin-bottom:12px'>
            <div style='color:#60A5FA;font-weight:600;font-size:0.82rem;margin-bottom:5px'>
            💡  Institutional access required</div>
            <div style='color:#94A3B8;font-size:0.82rem;line-height:1.6'>
            Scopus is a subscription database. You must log in with an
            <b style='color:#F1F5F9'>institutional email</b> from a university or research body.
            Personal Gmail or Yahoo accounts will not have access.</div>
        </div>

        <div style='background:#0F1117;border-left:3px solid #22C55E;border-radius:0 8px 8px 0;
        padding:12px 16px;margin-bottom:12px'>
            <div style='color:#4ADE80;font-weight:600;font-size:0.82rem;margin-bottom:5px'>
            ✅  Supported file formats</div>
            <div style='color:#94A3B8;font-size:0.82rem;line-height:1.6'>
            <b style='color:#F1F5F9'>.bib</b> — Scopus BibTeX <span style='color:#60A5FA'>(recommended)</span><br>
            <b style='color:#F1F5F9'>.ris</b> — RIS format (Scopus / Mendeley)<br>
            <b style='color:#F1F5F9'>.csv</b> — Scopus CSV or Web of Science CSV</div>
        </div>

        <div style='background:#0F1117;border-left:3px solid #A78BFA;border-radius:0 8px 8px 0;
        padding:12px 16px'>
            <div style='color:#C4B5FD;font-weight:600;font-size:0.82rem;margin-bottom:5px'>
            🔑  Fields needed for full analysis</div>
            <div style='color:#94A3B8;font-size:0.82rem;line-height:1.6'>
            Always include <b style='color:#F1F5F9'>Abstract & keywords</b> when exporting.
            These fields are required for keyword frequency, word clouds,
            LDA topic modelling, and research front mapping.
            Without them, several analysis tabs will show limited results.</div>
        </div>
        </div>

        <div style='background:#161B27;border:1px solid #1E2A40;border-radius:12px;
        padding:20px;margin-top:12px;text-align:center'>
        <div style='color:#475569;font-size:0.78rem;margin-bottom:6px'>Built by</div>
        <div style='font-family:Syne,sans-serif;font-size:1.3rem;font-weight:800;
        background:linear-gradient(135deg,#60A5FA,#A78BFA);-webkit-background-clip:text;
        -webkit-text-fill-color:transparent;'>Bahas Kebijakan</div>
        <div style='color:#334155;font-size:0.75rem;margin-top:4px'>
        Bibliometric & Scientometric Analysis Suite</div>
        </div>
        """, unsafe_allow_html=True)
