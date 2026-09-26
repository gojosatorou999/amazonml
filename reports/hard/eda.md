# EDA

- Source 1 entities: **8000**, Source 2+3 records: **18595**, true pairs: **13775**
- Singletons (no true match): **11.4%** — each is worth a full 1.0 for an empty prediction
- Matches per Source-1 entity: `{0: 915, 1: 2723, 2: 2677, 3: 1163, 4: 416, 5: 91, 6: 15}`
- Unmatched S2/S3 records (pure distractors): **4820**
- Records claimed by more than one Source-1 entity: **0** → exclusivity **holds**; the one-owner constraint is safe to apply.
- Train country labels (S1): `{'IN': 1980, 'United States': 1384, 'US': 1346, 'India': 1994, 'USA': 1296}`
- Train country labels (S2/S3): `{'United States': 2988, 'IN': 4658, 'India': 4689, 'US': 3074, 'USA': 3186}`
- Test: S1 **3000**, S2+S3 **7058**, labels `{'IN': 2016, 'India': 1996, 'United States': 1396, 'FR': 933, 'France': 958, 'US': 1355, 'USA': 1404}`
