import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
from groq import Groq
from dotenv import load_dotenv
import traceback

# Load API key securely
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Streamlit App UI
st.set_page_config(page_title="Budget vs. Actuals AI", page_icon="📊", layout="wide")
st.title("📊 Budget vs. Actuals AI – Variance Analysis & Commentary")
st.caption("Format data wajib: Tahun, Detail Akun, Anggaran, Realisasi")
st.write("Upload file Excel dan dapatkan analisis otomatis (AI jika tersedia).")

uploaded_file = st.file_uploader("📂 Upload your dataset (Excel format)", type=["xlsx"])
if not uploaded_file:
    st.info("Silakan upload file Excel terlebih dahulu.")
    st.stop()

# Read file
df = pd.read_excel(uploaded_file)

# Required columns
required_columns = ["Tahun", "Detail Akun", "Anggaran", "Realisasi"]
if not all(col in df.columns for col in required_columns):
    st.error(f"⚠️ File harus mempunyai kolom: {required_columns}")
    st.stop()

# Ensure numeric
df["Anggaran"] = pd.to_numeric(df["Anggaran"], errors="coerce").fillna(0)
df["Realisasi"] = pd.to_numeric(df["Realisasi"], errors="coerce").fillna(0)

# Calculate Variance
df["Variance"] = df["Realisasi"] - df["Anggaran"]
# Avoid division by zero
df["Variance %"] = np.where(df["Anggaran"] == 0, np.nan, (df["Variance"] / df["Anggaran"]) * 100)

# Year filter in sidebar
years = sorted(df["Tahun"].dropna().unique().tolist())
years_display = ["All"] + [str(y) for y in years]
selected_year = st.sidebar.selectbox("Filter Tahun", years_display, index=0)
if selected_year != "All":
    # handle numeric or string tahun
    try:
        sel_y_val = int(selected_year)
        df_filtered = df[df["Tahun"] == sel_y_val]
    except Exception:
        df_filtered = df[df["Tahun"].astype(str) == selected_year]
else:
    df_filtered = df.copy()

st.subheader("📊 Data Preview with Variance Calculation")
st.dataframe(df_filtered)

# Charts
st.subheader("📈 Variance by Detail Akun")
fig_bar = px.bar(
    df_filtered.groupby("Detail Akun", as_index=False)["Variance"].sum().sort_values("Variance", ascending=False),
    x="Detail Akun",
    y="Variance",
    color="Variance",
    title="📊 Variance by Detail Akun",
    text_auto=".2s",
)
st.plotly_chart(fig_bar, use_container_width=True)

st.subheader("📉 Trend Anggaran vs Realisasi per Tahun (per Detail Akun)")
fig_line = px.line(
    df_filtered,
    x="Tahun",
    y=["Anggaran", "Realisasi"],
    color="Detail Akun",
    markers=True,
    title="📉 Budget vs Actual Performance by Tahun & Detail Akun",
)
st.plotly_chart(fig_line, use_container_width=True)

# ---------------------------
# AI Section with error handling and fallback
# ---------------------------
st.subheader("🤖 AI-Powered Variance Analysis (Groq)")

def generate_local_summary(df_in):
    """Simple deterministic summary (fallback when API unavailable)."""
    agg = df_in.groupby("Detail Akun", as_index=False).agg({
        "Anggaran": "sum",
        "Realisasi": "sum",
        "Variance": "sum"
    })
    top_over = agg.sort_values("Variance", ascending=False).head(5)
    top_under = agg.sort_values("Variance", ascending=True).head(5)

    total_anggaran = df_in["Anggaran"].sum()
    total_realisasi = df_in["Realisasi"].sum()
    total_var = total_realisasi - total_anggaran

    lines = []
    lines.append(f"Total Anggaran: {total_anggaran:,.2f}")
    lines.append(f"Total Realisasi: {total_realisasi:,.2f}")
    lines.append(f"Total Variance: {total_var:,.2f}")
    lines.append("")
    lines.append("Top akun dengan realisasi melebihi anggaran (positif):")
    for _, r in top_over.iterrows():
        lines.append(f"- {r['Detail Akun']}: Variance {r['Variance']:,.2f} (A:{r['Anggaran']:,.2f} R:{r['Realisasi']:,.2f})")
    lines.append("")
    lines.append("Top akun dengan realisasi di bawah anggaran (negatif):")
    for _, r in top_under.iterrows():
        lines.append(f"- {r['Detail Akun']}: Variance {r['Variance']:,.2f} (A:{r['Anggaran']:,.2f} R:{r['Realisasi']:,.2f})")
    lines.append("")
    if total_var < 0:
        lines.append("Rekomendasi: Periksa akun-akun bervariance negatif dan pertimbangkan penghematan atau peninjauan ulang anggaran.")
    else:
        lines.append("Rekomendasi: Kelebihan realisasi — verifikasi apakah ini akibat over-delivery atau penjadwalan/transfer biaya.")
    return "\n".join(lines)

# Try calling Groq API, but catch and show full error; fallback to local summary
try:
    client = Groq(api_key=GROQ_API_KEY)
    system_msg = {"role": "system", "content": "You are an AI financial analyst providing variance analysis insights on budget vs. actuals."}
    user_msg_text = f"Here is the budget vs. actual variance summary:\n{df_filtered.to_string()}\nPlease provide key insights and recommendations."
    user_msg = {"role": "user", "content": user_msg_text}

    # call API
    response = client.chat.completions.create(
        messages=[system_msg, user_msg],
        model="llama-3.1-8b-instant",
    )
    ai_text = response.choices[0].message.content
    st.markdown("**AI Summary (Groq):**")
    st.write(ai_text)

except Exception as e:
    st.error("🚨 Permintaan ke API Groq gagal. Menampilkan detail error untuk debugging.")
    # Attempt to extract HTTP response info if present
    try:
        resp = getattr(e, "response", None)
        status = getattr(resp, "status_code", None)
        body_text = getattr(resp, "text", None)
        st.write("**Exception trace:**")
        st.code(traceback.format_exc())
        if status is not None or body_text is not None:
            st.write("**HTTP status / body (jika tersedia):**")
            st.code(f"Status: {status}\nBody: {body_text}")
    except Exception:
        st.code(traceback.format_exc())

    st.info("Menjalankan ringkasan fallback lokal (tanpa AI).")
    local_summary = generate_local_summary(df_filtered)
    st.markdown("**Fallback Summary (lokal):**")
    st.text(local_summary)

# Optional: allow user to ask follow-up question to AI (also protected with try/except)
st.subheader("🗣️ Chat with AI About Variance Analysis (opsional)")
user_query = st.text_input("🔍 Tanyakan sesuatu pada AI (ketik lalu tekan Enter):")
if user_query:
    st.write("Memproses pertanyaan...")
    try:
        client = Groq(api_key=GROQ_API_KEY)
        messages = [
            {"role": "system", "content": "You are an AI financial analyst helping users understand their budget vs. actual variance analysis."},
            {"role": "user", "content": f"Variance Data:\n{df_filtered.to_string()}\nQuestion: {user_query}"}
        ]
        chat_response = client.chat.completions.create(messages=messages, model="llama-3.1-8b-instant")
        st.write(chat_response.choices[0].message.content)
    except Exception:
        st.error("Gagal memproses chat via Groq. Menampilkan fallback jawaban singkat.")
        st.code(traceback.format_exc())
        # Simple fallback answer
        st.write("Fallback: Mohon cek kembali API key / akses model; sementara, ringkasan lokal sudah disediakan di atas.")
