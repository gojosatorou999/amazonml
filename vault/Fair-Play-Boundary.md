# Fair-Play Boundary

The statement bans **external data lookup** on pain of immediate disqualification:
commercial ER APIs, government business registries, **geocoding APIs**, any internet
data augmentation. Code and methodology are reviewed for the top teams.

## Allowed (and we lean on all of it)
- Pretrained open-weight models, MIT/Apache-2.0, ≤8B params — downloaded weights are
  not a "data lookup service".
- Bundled library tables (`unidecode` transliteration, `rapidfuzz`) — static, offline.
- **Transductive use of the provided test set**: IDF statistics, embedding index,
  abbreviation mining, and cross-source S2↔S3 similarity all computed over train+test
  records. This is provided data, not external data. It is a real edge and it is legal.

## Forbidden — do not let these creep in
- Any geocoding, PIN/ZIP→city lookup table pulled from the web.
- Any curated external gazetteer of legal suffixes or street abbreviations.
  → We **mine the abbreviation map from the training pairs themselves** ([[Winning-Features]] #7).
  That is both safe and, we will argue in the writeup, better: it learns the variants
  that actually occur in *this* data, including transliteration pairs no public list has.
- Anything that needs a network call at inference time.

`requirements.txt` is pinned and the pipeline must run fully offline. Make this explicit
in the README so reviewers can verify it in seconds.
