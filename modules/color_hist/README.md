# use color histogram as feature

quantizer = faiss.IndexFlat(faiss_dim, faiss.METRIC_L1)
index = faiss.IndexIDMap2(quantizer)
