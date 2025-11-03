if st.button("🚀 Fetch & Score", type="primary", use_container_width=True):
    # 1) Load mapping + exemplars from DATASETS
    try:
        mapping = pd.read_csv(MAPPING_PATH)
    except Exception:
        mapping = pd.read_excel(MAPPING_PATH)
    mapping.columns = [c.lower().strip() for c in mapping.columns]
    assert {"column","question_id","attribute"}.issubset(mapping.columns)
    if "prompt_hint" not in mapping.columns: mapping["prompt_hint"] = ""
    mapping = mapping[mapping["attribute"].isin(ORDERED_ATTRS)].copy()

    exemplars = []
    with open(EXEMPLARS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip(): exemplars.append(json.loads(line))

    # 2) Centroids
    with st.spinner("Building centroids..."):
        q_centroids, attr_centroids, global_centroids, by_qkey, question_texts = build_centroids(exemplars)

    # 3) Kobo
    with st.spinner("Fetching Kobo submissions..."):
        df = fetch_kobo_dataframe()
    if df.empty:
        st.error("No Kobo submissions found (or auth/URL wrong).")
        st.stop()

    st.caption("Kobo sample:")
    st.dataframe(df.head(), use_container_width=True)

    # 4) Score
    with st.spinner("Scoring..."):
        scored_df = score_dataframe(df, mapping, q_centroids, attr_centroids, global_centroids, by_qkey, question_texts)

    st.success("✅ Scoring complete.")
    st.dataframe(scored_df.head(50), use_container_width=True)

    # 5) Download
    from io import BytesIO
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as w:
        scored_df.to_excel(w, index=False)
    st.download_button("⬇️ Download Excel",
        data=bio.getvalue(),
        file_name="Individual_Advisory_Scoring_Sheet.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True)
