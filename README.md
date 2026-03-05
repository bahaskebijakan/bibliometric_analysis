# BibloMetrix — Bibliometric Analysis Suite
### Streamlit App · Deploy sekali, pakai selamanya

---

## 🚀 Cara Deploy ke Streamlit Community Cloud (GRATIS)

### Step 1 — Upload ke GitHub

1. Buat akun di [github.com](https://github.com) (gratis)
2. Buat **New Repository** → nama: `bibliometric-app` → **Public** atau **Private**
3. Upload 2 file ini:
   - `app.py`
   - `requirements.txt`

### Step 2 — Deploy di Streamlit Cloud

1. Buka [share.streamlit.io](https://share.streamlit.io)
2. Login dengan akun GitHub kamu
3. Klik **"New app"**
4. Pilih:
   - Repository: `bibliometric-app`
   - Branch: `main`
   - Main file path: `app.py`
5. Klik **Deploy!**
6. Tunggu 2-3 menit → dapat URL seperti: `https://yourname-bibliometric-app.streamlit.app`

### Step 3 — Link dari website kamu

```html
<!-- Tombol di websitemu -->
<a href="https://yourname-bibliometric-app.streamlit.app" target="_blank">
  Buka Analisis Bibliometrik
</a>

<!-- Atau embed sebagai iframe (opsional) -->
<iframe src="https://yourname-bibliometric-app.streamlit.app?embed=true"
        height="800" width="100%" frameborder="0"></iframe>
```

---

## 🔒 Menyembunyikan Kode (Repo Private)

Kalau kamu tidak mau orang lain lihat kode:
1. Set GitHub repo ke **Private**
2. Streamlit Cloud tetap bisa deploy dari private repo
3. User hanya lihat web app — kode tidak bisa diakses

---

## 📁 Struktur File

```
bibliometric-app/
├── app.py              ← main Streamlit app
└── requirements.txt    ← dependencies
```

---

## ✨ Fitur App

| Tab | Fitur |
|-----|-------|
| 📈 Trends | Annual volume, cumulative growth, YoY rate, doc-type mix |
| 📰 Journals | Top journals, Bradford's Law zones |
| ✍️ Authors | Top authors, Lotka's Law, citation leaders |
| 🌍 Country & OA | Geographic output, Open Access breakdown |
| 🔑 Keywords | Author KW vs Index KW, word clouds |
| 📊 Citations | Lorenz curve, distribution, top-cited papers |
| 🔗 Networks | Co-authorship + keyword co-occurrence (static + interactive HTML) |
| 🧠 Topics | LDA topic modelling dari abstrak |
| 🚀 Research Fronts | Emerging vs established themes map |
| 🔬 MCDM Methods | Auto-extract weighting & ranking methods per paper |
| ⬇️ Export | Download semua CSV dalam ZIP |

**Semua chart bisa didownload dengan kualitas 600 DPI** — siap publikasi.

---

## 💡 Tips

- Upload **beberapa file .bib** sekaligus → digabung otomatis (untuk 767 artikel, export per batch lalu upload semua)
- Gunakan sidebar untuk mengatur **Top N items**, **jumlah LDA topics**, dan **kualitas download**
- Format yang didukung: `.bib` (Scopus), `.ris`, `.csv` (Scopus & WoS)
