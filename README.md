# IA Dashboards – *One-File Streamlit Intelligence Suite*
Transforms the raw **`IA_Shaurya_IAPBL.csv`** feed into a **dual-purpose web BI platform**:

| Module (Sidebar ► Choose view) | Target Persona | Strategic Pay-off |
|--------------------------------|----------------|-------------------|
| **Detailed Explorer** | Data analysts, functional leads | Unlimited slice-and-dice across every categorical/ numerical pairing to surface granular revenue drivers. |
| **Executive Overview** | CEO / CXO / Board | KPI cockpit **plus** an ML-driven, renewal-adjusted 12-month city-level revenue forecast. |

Every visual is front-loaded with a **3-line managerial takeaway** to keep insights actionable.

---

## 1 ▪ Repository Layout
IA_dashboards/
│
├─ IA_Shaurya_IAPBL.csv # raw data (leave filename unchanged)
├─ app.py # single-file Streamlit app (both dashboards)
└─ requirements.txt # deterministic package list
> *Why single-file?* Zero onboarding friction for first-time contributors; deploys with one click on Streamlit Cloud.

---

## 2 ▪ Tech Stack & Versions
| Library | Version | Comment |
|---------|---------|---------|
| **Python** | ≥ 3.9 | Tested 3.10, 3.11 |
| **Streamlit** | 1.35.0 | Multipage radio in sidebar |
| **scikit-learn** | 1.5.0 | GridSearchCV, pipelines |
| pandas / numpy | latest stable | ETL, maths |
| **Plotly** | 5.22.0 | Interactive visuals |

Everything pins inside `requirements.txt`.

---

## 3 ▪ Local Quick-Start  ⏱️ < 2 min
```bash
# 1 Clone or download
git clone https://github.com/<your-handle>/IA_dashboards.git
cd IA_dashboards

# 2 (Optional) virtual env
python -m venv venv && source venv/bin/activate   # Win: venv\Scripts\activate

# 3 Install deps
pip install -r requirements.txt

# 4 Launch dashboards
streamlit run app.py
