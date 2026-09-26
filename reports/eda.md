# EDA

- Source 1 entities: **8000**, Source 2+3 records: **18562**, true pairs: **13750**
- Singletons (no true match): **11.6%** — each is worth a full 1.0 for an empty prediction
- Matches per Source-1 entity: `{0: 925, 1: 2741, 2: 2634, 3: 1166, 4: 444, 5: 73, 6: 17}`
- Unmatched S2/S3 records (pure distractors): **4812**
- Records claimed by more than one Source-1 entity: **0** → exclusivity **holds**; the one-owner constraint is safe to apply.
- Train country labels (S1): `{'IN': 2022, 'United States': 1315, 'US': 1369, 'India': 1990, 'USA': 1304}`
- Train country labels (S2/S3): `{'United States': 3049, 'IN': 4651, 'USA': 3042, 'US': 3100, 'India': 4720}`
- Test: S1 **3000**, S2+S3 **6895**, labels `{'IN': 1964, 'USA': 1360, 'India': 1958, 'France': 986, 'FR': 978, 'US': 1317, 'United States': 1332}`
