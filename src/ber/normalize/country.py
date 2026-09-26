"""Resolve country *labels* into country *groups* -- without a country list.

The data spells one country several ways ('US', 'USA', 'United States';
'IN', 'India') and the test set adds a country absent from training.  The
statement forbids hard-coding the label set, and a lookup table of country
names would be exactly that.

So we resolve the labels with the same signal the whole pipeline runs on:
records that are near-duplicates of each other.  Take each Source-1 record's
single best cross-source neighbour under a strict TF-IDF similarity; if that
pair is near-identical, its two labels are evidence that they denote the same
country.  Labels whose mutual evidence is a meaningful share of their volume
are merged (union-find).  'US' and 'India' never merge because near-identical
records never straddle them; 'FR' and 'France' merge on the test file alone.

This is entity resolution applied to the country column itself.
"""
from __future__ import annotations

from collections import Counter

import numpy as np


class _DSU:
    def __init__(self, items):
        self.p = {x: x for x in items}

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def resolve_country_groups(
    labels_a: list[str],
    labels_b: list[str],
    best_sim: np.ndarray,
    all_labels: list[str],
    sim_floor: float = 0.75,
    min_share: float = 0.05,
    min_pairs: int = 5,
) -> dict[str, str]:
    """Map every label to a canonical group id.

    labels_a[i], labels_b[i] are the labels of record i and its best neighbour,
    best_sim[i] their similarity.
    """
    vol = Counter(all_labels)
    link: Counter[tuple[str, str]] = Counter()
    for la, lb, s in zip(labels_a, labels_b, best_sim):
        if s >= sim_floor and la != lb:
            link[tuple(sorted((la, lb)))] += 1

    dsu = _DSU(set(all_labels))
    for (la, lb), n in link.items():
        if n >= min_pairs and n / min(vol[la], vol[lb]) >= min_share:
            dsu.union(la, lb)

    groups: dict[str, list[str]] = {}
    for lab in vol:
        groups.setdefault(dsu.find(lab), []).append(lab)
    # canonical name = most frequent label in the group (stable, readable)
    canon = {root: max(members, key=lambda m: (vol[m], m)) for root, members in groups.items()}
    return {lab: canon[dsu.find(lab)] for lab in vol}
