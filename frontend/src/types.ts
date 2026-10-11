export type Mutation = {
  chain: string;
  position: number;
  insertion: number;
  from: string;
  to: string;
  region: string;
  label: string;
};
export type Variant = {
  variant_id: string;
  heavy_sequence: string;
  light_sequence: string;
  binding_mean: number;
  binding_std: number;
  stability_mean: number;
  stability_std: number;
  stability_lcb: number;
  selected: boolean;
  selection_rank: number | null;
  well: string | null;
  pareto: boolean;
  measured: boolean;
  eligible: boolean;
  mutations: Mutation[];
  exclusions: string[];
  acquisition: number;
  ehvi: number;
  net_charge: number;
  hydrophobic_patch: number;
  apr: number;
  pseudo_perplexity: number | null;
  developability_penalty: number;
};
export type Config = {
  embedding: "descriptor" | "esm2" | "ablang2";
  exploration: number;
  diversity: number;
  min_stability: number;
  confidence_beta: number;
  max_abs_charge: number;
  max_hydrophobic_patch: number;
  max_apr: number;
  seed: number;
  batch_size: 24;
};
export type Annotation = {
  chain: string;
  method: string;
  identity: number;
  regions: { name: string; start: number; end: number; sequence: string }[];
};
export type RunInfo = {
  id: string;
  revision: number;
  status: string;
  created_at: string;
  error: string | null;
};
export type Dataset = {
  id: string;
  name: string;
  revision: number;
  demo: boolean;
  variant_count: number;
  observation_count: number;
  parent: {
    variant_id: string;
    heavy_sequence: string;
    light_sequence: string;
  };
  annotations: Annotation[];
  runs: RunInfo[];
};
export type Result = {
  variants: Variant[];
  selected_ids: string[];
  annotations: Annotation[];
  config: Config;
  summary: {
    library_size: number;
    observed_variants: number;
    training_points: number;
    total_observations: number;
    eligible: number;
    pareto_count: number;
    selected: number;
    mean_binding: number;
    mean_stability: number;
    min_stability_lcb: number;
    parent_binding: number;
    mean_pairwise_distance: number;
    exploration_weight: number;
    elapsed_seconds: number;
  };
  provenance: {
    algorithm: string;
    surrogate: string;
    embedding: string;
    seed: number;
    score_direction: string;
    training_ids: string[];
  };
  warnings: string[];
};
export type Run = RunInfo & {
  dataset_id: string;
  progress: string;
  stale: boolean;
  result: Result | null;
  config: Config;
};
