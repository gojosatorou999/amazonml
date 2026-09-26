// Typed client for the Python API in src/ber/service/app.py.

export type Shap = { f: string; v: number; c: number };

export type Candidate = {
  id: string;
  name: string;
  address: string;
  country: string;
  p: number; // probability used by the decision rule
  p1: number; // stage-1 (pruner) probability
  emitted: boolean;
  shap: Shap[];
  base: number;
  also_claimed_by?: { id: string; name: string }[];
  owned_by?: { id: string; name: string; p: number }[];
};

export type Pruned = { id: string; name: string; address: string; p1: number };

export type DecisionRule = { mode: "expected_f" | "threshold"; excl: boolean; cal: boolean; thr?: number };

export type Resolution = {
  candidates: Candidate[];
  curve: number[]; // E[F0.5] for emitting the top-k, k = 0..n
  chosen: string[];
  pruned: Pruned[];
  rule?: DecisionRule;
};

export type EntityRow = {
  id: string;
  name: string;
  address: string;
  country: string;
  n_cand: number;
  n_match: number;
  unseen: boolean;
  contested: boolean;
};

export type Entity = EntityRow & Resolution;

export type QueryResult = Resolution & {
  query: { name: string; address: string; country: string };
  pool_size: number;
  compete: boolean;
  expected_f?: number | null;
};

export type QueueItem = EntityRow & { gain: number; pair: string; pair_name: string; p: number };

export type Sample = { label: string; name: string; address: string; country: string };

export type Score = {
  macro_f: number;
  singleton_f: number;
  matched_f: number;
  precision_micro: number;
  recall_micro: number;
  n: number;
};

export type Overview = {
  test: {
    n_s1: number;
    n_r: number;
    pool_pairs: number;
    cand_pairs: number;
    cand_per_s1: { mean: number; median: number; p95: number; max: number; zero_share: number };
    matched_pairs: number;
    empty_share: number;
    holdout_score?: Score;
    holdout_cand_recall?: number;
    holdout_pool_recall?: number;
    timings: Record<string, number>;
  };
  groups: Record<string, string>;
  train: {
    blocking?: {
      n_s1: number;
      n_r: number;
      pool_recall: number;
      cand_recall: number;
      cand_per_s1: number;
      pool_pairs: number;
      cand_pairs: number;
      prune_tau: number;
    };
    variants?: Record<string, { score: Score; cfg: DecisionRule }>;
    loco?: Record<string, { score: Score }> | null;
    selection?: Record<string, number>;
    chosen?: string;
    unseen_share?: number;
    importance: [string, number][];
    n_features?: number;
  };
  lexicon: { abbrev: Record<string, string>; legal: string[]; cues: string[] };
  decision: DecisionRule;
  tau: number;
};

export class ApiError extends Error {}

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, init);
  } catch {
    throw new ApiError("Can't reach the server. Start it with `make serve` and reload.");
  }
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new ApiError(body.error ?? `The server answered ${res.status}.`);
  return body as T;
}

export const api = {
  health: () => call<{ resolver: "ready" | "warming" | "error"; error?: string }>("/api/health"),
  overview: () => call<Overview>("/api/overview"),
  entities: (q: string, filter: string, offset = 0, limit = 60) =>
    call<{ total: number; items: EntityRow[] }>(
      `/api/entities?q=${encodeURIComponent(q)}&filter=${filter}&offset=${offset}&limit=${limit}`,
    ),
  entity: (id: string) => call<Entity>(`/api/entity/${encodeURIComponent(id)}`),
  review: () => call<QueueItem[]>("/api/review"),
  samples: () => call<Sample[]>("/api/samples"),
  resolve: (q: { name: string; address: string; country: string; compete: boolean }) =>
    call<QueryResult>("/api/resolve", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(q),
    }),
};

/** Human names for model features, written for a reader who never saw the code. */
export const FEATURE_NAMES: Record<string, string> = {
  ctx_excl_j: "Share of this record's claims",
  ctx_p: "First-pass match probability",
  ctx_gap_i: "Distance behind this entity's best candidate",
  ctx_margin_i: "Lead over the runner-up",
  ctx_sum_i: "Other candidates' combined weight",
  ctx_sum_j: "Other entities claiming this record",
  ctx_cnt_j: "Entities competing for this record",
  ctx_rank_i: "Rank among this entity's candidates",
  ctx_rank_j: "Rank among the record's claimants",
  ctx_gap_j: "Distance behind the record's best claimant",
  ctx_support: "Backed by a matching sibling record",
  ctx_nsib: "Confident sibling records",
  ctx_nhi_i: "Confident candidates for this entity",
  id_soft: "Distinctive name words agree",
  id_ratio: "Distinctive name spelling",
  id_tset: "Distinctive name words overlap",
  id_miss_a: "Rare name word missing from record",
  id_miss_b: "Rare record word missing from name",
  c_soft: "Core name agrees",
  c_ratio: "Core name spelling",
  c_tset: "Core name words overlap",
  c_jw: "Core name prefix similarity",
  c_cat_ratio: "Name ignoring spaces",
  c_equal: "Identical core name",
  n_soft: "Full name agrees",
  n_ratio: "Full name spelling",
  n_tsort: "Full name, any word order",
  n_tset: "Full name words overlap",
  n_partial: "Name contained in the other",
  n_miss_a: "Rare word only in entity name",
  n_miss_b: "Rare word only in record name",
  tail_jacc: "Legal form agrees",
  first_eq: "Same first name word",
  acr_hit: "Acronym matches",
  a_soft: "Address words agree",
  a_ratio: "Address spelling",
  a_tset: "Address words overlap",
  a_tsort: "Address, any order",
  a_miss_a: "Rare address word only in entity",
  a_miss_b: "Rare address word only in record",
  st_soft: "Street agrees",
  tl_soft: "Locality and city agree",
  pc_state: "Postcode agreement",
  pc3: "Postcode area agrees",
  house_state: "House number agreement",
  house_part: "House number partly agrees",
  num_jacc: "Numbers in address overlap",
  num_conflict: "Numbers in address conflict",
  country_grp: "Same country",
  country_label: "Same country label",
  name_x_addr: "Name and address both agree",
  ident_x_addr: "Distinctive name and address both agree",
  dens_s1: "How common the name is in Source 1",
  dens_r: "How common the name is in Sources 2–3",
  rrf: "Blocking relevance",
  n_channels: "Blocking channels that found it",
  pool_rank: "Blocking rank",
  lm_a: "Entity address has a landmark",
  lm_b: "Record address has a landmark",
  src: "Source of the record",
};

export const featureName = (f: string) =>
  FEATURE_NAMES[f] ??
  f
    .replace(/_(rank|sim)$/, (_, k) => (k === "rank" ? " rank" : " similarity"))
    .replace(/_rev/, " (reverse)")
    .replace(/_/g, " ")
    .replace(/^name char/, "Name spelling search")
    .replace(/^full char/, "Name and address search")
    .replace(/^word/, "Rare word search")
    .replace(/^keys/, "Exact key search");

export const sourceOf = (id: string): 1 | 2 | 3 => (id.startsWith("S2") ? 2 : id.startsWith("S3") ? 3 : 1);
