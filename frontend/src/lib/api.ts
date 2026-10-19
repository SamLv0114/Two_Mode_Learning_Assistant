/**
 * API client for the ResearchMate backend
 */
import axios, { AxiosError, AxiosInstance, AxiosRequestConfig } from 'axios';

export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const API_URL = API_BASE_URL;

// Create axios instance
const api: AxiosInstance = axios.create({
  baseURL: `${API_URL}/api/v1`,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add auth token to requests
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('access_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

function goToLogin() {
  localStorage.removeItem('access_token');
  localStorage.removeItem('refresh_token');
  window.location.href = '/login';
}

// A 401 means the access token is gone (expired after its fixed lifetime —
// there's no way to renew it in place), not necessarily that the session
// is over: try the refresh token once before giving up. Concurrent 401s
// (several requests in flight when the token expired) must not each fire
// their own refresh call — the refresh token is one-time-use server-side
// (see auth.py's rotation), so a second concurrent refresh would find the
// first one had already redeemed and deleted it, and fail a session that
// was actually still fine. Every caller shares the one in-flight promise.
let refreshInFlight: Promise<string> | null = null;

/** New access token after a 401; sends the user to /login and rejects when the session is over. */
function refreshAccessToken(): Promise<string> {
  if (!refreshInFlight) {
    const refreshToken = localStorage.getItem('refresh_token');
    refreshInFlight = (refreshToken ? authApi.refresh(refreshToken) : Promise.reject(new Error('Session expired')))
      .then((data) => {
        localStorage.setItem('access_token', data.access_token);
        localStorage.setItem('refresh_token', data.refresh_token);
        return data.access_token;
      })
      .catch((error) => {
        goToLogin();
        throw error;
      })
      .finally(() => { refreshInFlight = null; });
  }
  return refreshInFlight;
}

/** Token to retry a 401 with: a login may already have replaced the one the request carried. */
function tokenForRetry(sentToken: string | null): Promise<string> {
  const current = localStorage.getItem('access_token');
  return current && current !== sentToken ? Promise.resolve(current) : refreshAccessToken();
}

// Handle auth errors
api.interceptors.response.use(
  (response) => response,
  async (error: AxiosError) => {
    const originalRequest = error.config as (AxiosRequestConfig & { _retried?: boolean }) | undefined;

    if (error.response?.status !== 401 || !originalRequest || originalRequest._retried) {
      return Promise.reject(error);
    }
    originalRequest._retried = true;
    const sent = String(originalRequest.headers?.Authorization || '').replace(/^Bearer /, '') || null;
    try {
      const token = await tokenForRetry(sent);
      originalRequest.headers = { ...originalRequest.headers, Authorization: `Bearer ${token}` };
      return api(originalRequest);
    } catch {
      return Promise.reject(error);
    }
  }
);

/**
 * User-facing text for a failed call: the API's `detail` when it sent one
 * (a string, or FastAPI's list of validation errors), the message of a
 * non-HTTP error (e.g. a broken stream), otherwise `fallback`.
 */
export function errorMessage(error: unknown, fallback = 'Request failed'): string {
  if (axios.isAxiosError(error)) {
    const detail: unknown = error.response?.data?.detail;
    if (typeof detail === 'string' && detail) return detail;
    if (Array.isArray(detail) && detail[0]?.msg) return String(detail[0].msg);
    return fallback;
  }
  return error instanceof Error && error.message ? error.message : fallback;
}

// Types
export interface User {
  id: number;
  email: string;
  full_name?: string;
  interests: string[];
  focus_areas: string[];
  is_active: boolean;
  created_at: string;
  interaction_count: number;
  model_trained: boolean;
  /** The shared demo account: account actions are hidden and model work is metered. */
  is_demo?: boolean;
}

export interface Token {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
}

export interface Paper {
  id: number;
  rank: number;
  arxiv_id: string;
  title: string;
  authors?: string;
  abstract?: string;
  categories?: string;
  published_date?: string;
  arxiv_url?: string;
  pdf_url?: string;
  citation_count: number;
  relevance_score: number;
  impact_score?: number;
  summary?: string;
  /** Why this paper is in the feed; null for feeds built before the research profile. */
  reasons?: PaperReasons | null;
}

/** The recall channels that found a paper, and the model's judgement of it against the profile. */
export interface PaperReasons {
  reason: string | null;
  goal_fit: 0 | 1 | 2 | 3 | null;
  readiness: 'too_basic' | 'right_level' | 'too_advanced' | null;
  paper_type: string | null;
  channels: ('content' | 'graph' | 'trending' | 'latest')[];
  evidence: string[];
  profile_version?: number;
}

export type PaperType = 'method' | 'survey' | 'benchmark' | 'theory' | 'application' | 'position' | 'other';
export const PAPER_TYPES: PaperType[] = ['method', 'survey', 'benchmark', 'theory', 'application', 'position', 'other'];

export interface SeedPaper { id: string; title: string; year?: number | null }

/** The editable card recommendation is built from (src/agents/research_profile.py). */
export interface ResearchProfile {
  goal: string;
  level: '' | 'beginner' | 'intermediate' | 'advanced';
  known: string[];
  interests: string[];
  preferences: string[];
  avoid_types: PaperType[];
  seed_papers: SeedPaper[];
  /** Papers implemented in the Coding Agent; recommendation treats them as known ground. */
  studied: { arxiv_id: string; title: string; objective: string; at: string }[];
  feedback: { text: string; paper: string; note: string; at: string }[];
  summary: string;
  version: number;
  updated_at: string | null;
  /** False until there is a goal or a topic to recommend from (a row can exist holding only `studied`). */
  set_up: boolean;
}

export type ProfileFields = Pick<ResearchProfile, 'goal' | 'level' | 'known' | 'interests' | 'preferences'
  | 'avoid_types' | 'seed_papers' | 'summary'>;

export interface PaperOption { id: string; title: string; year?: number | null; url: string; source: 'library' | 'semantic_scholar' }

export interface ReadingPathStep {
  stage: 'foundation' | 'core' | 'frontier';
  why: string;
  paper: { title: string; year?: number | null; url: string; arxiv_id?: string | null; citation_count: number;
    db_id?: number | null; source: string };
}

export interface ReadingPath {
  topic: string;
  queries: string[];
  steps: ReadingPathStep[];
  skipped: string;
  ordered_by: 'model' | 'year' | 'none';
  /** True when Semantic Scholar could not be reached and only this library's papers were used. */
  library_only: boolean;
}

export interface FeedResponse {
  papers: Paper[];
  generated_at: string;
  time_window_days: number;
  focus_areas: string[];
  used_ml_ranking: boolean;
  ranking_source?: string;
  ranking_fallback_reason?: string | null;
  selection?: 'llm_judge' | 'prerank' | null;
  /** Known only right after a generation; a feed loaded later does not record it. */
  total_papers_considered: number | null;
}

export interface CurrentFeed {
  papers: Paper[];
  generated_at: string | null;
  feed_mode?: 'recommended' | 'latest' | null;
  time_window_days?: number | null;
  ranking_source?: string;
  ranking_fallback_reason?: string | null;
  /** 'llm_judge' when the model judged the shortlist against the research profile. */
  selection?: 'llm_judge' | 'prerank' | null;
}

export interface ColdStartSeed {
  db_id: number;
  arxiv_id: string;
  title: string;
  abstract: string;
  categories?: string;
  citation_count?: number;
  url: string;
}

export interface ColdStartResponse {
  seeds: ColdStartSeed[];
  count: number;
  interactions_recorded: number;
  onboarding_recommended: boolean;
}

export interface RefineResponse {
  papers: Paper[];
  score: number;
  rounds: number;
  passed: boolean;
  message: string;
}

export interface InteractionStats {
  total: number;
  saved: number;
  viewed: number;
  dismissed: number;
  ready_for_training: boolean;
  interactions_until_training: number;
}

export interface ModelStatus {
  is_trained: boolean;
  interaction_count: number;
  interaction_count_at_training: number;
  training_sample_count: number;
  training_label_counts?: Record<string, number> | null;
  last_trained_at?: string | null;
}

export interface FeedJobStatus {
  status: 'generating' | 'collecting' | 'ranking' | 'done' | 'error' | 'not_found';
  message?: string;
  /** True when the job returned today's existing feed rather than building one. */
  reused?: boolean;
  papers_count?: number;
  used_ml_ranking?: boolean;
  ranking_source?: string;
  ranking_fallback_reason?: string | null;
  selection?: 'llm_judge' | 'prerank' | null;
}

export type AgentEvent =
  | { type: 'session'; session_id: string }
  | { type: 'intent'; value: string; method: string; confidence: number }
  | { type: 'agent'; value: string }
  | { type: 'tool_call'; tool: string }
  | { type: 'tool_result'; tool: string; count: number }
  | { type: 'generating' }
  | { type: 'token'; value: string }
  | { type: 'draft_token'; value: string }
  | { type: 'replace'; value: string }
  | { type: 'done'; tools_called: string[]; citations: { title: string; url: string; type: string }[]; stop_reason?: string; run_id?: string; resumable?: boolean }
  | { type: 'error'; value: string }
  | { type: 'cancelled' }
  | { type: 'run'; run_id: string }
  | { type: 'plan'; tasks: { id: number; title: string; intent: string }[] }
  | { type: 'task_started'; id: number; title: string }
  | { type: 'task_done'; id: number; citations: number }
  | { type: 'gaps'; gaps: { title: string; reason: string }[] }
  | { type: 'critiquing' }
  | {
      type: 'critique_result';
      groundedness: number;
      completeness: number;
      clarity: number;
      aggregate: number;
      critique: string;
      should_retry: boolean;
    }
  | { type: 'refining' };

// Auth API
export const authApi = {
  register: async (data: {
    email: string;
    password: string;
    full_name?: string;
    interests?: string[];
    focus_areas?: string[];
  }) => {
    const response = await api.post<User & { access_token: string; refresh_token: string }>('/auth/register', data);
    return response.data;
  },

  login: async (email: string, password: string) => {
    const formData = new URLSearchParams();
    formData.append('username', email);
    formData.append('password', password);

    const response = await api.post<Token>('/auth/login', formData, {
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    });
    return response.data;
  },

  // Exchanges a refresh token for a new access+refresh pair. Uses a bare
  // axios call, not the shared `api` instance: that instance's response
  // interceptor is what calls this function on a 401, so routing this
  // call through it too would risk a loop if the refresh call itself
  // ever came back 401.
  refresh: async (refreshToken: string) => {
    const response = await axios.post<Token>(`${API_URL}/api/v1/auth/refresh`, {
      refresh_token: refreshToken,
    });
    return response.data;
  },

  demo: async () => (await api.post<Token>('/auth/demo')).data,

  getProfile: async () => {
    const response = await api.get<User>('/auth/me');
    return response.data;
  },

  updateProfile: async (data: {
    full_name?: string;
    interests?: string[];
    focus_areas?: string[];
    password?: string;
  }) => {
    const response = await api.put<User>('/auth/me', data);
    return response.data;
  },
};

// Feed API
export const feedApi = {
  generate: async (params: {
    time_window_days: number;
    focus_areas?: string[];
    custom_interests?: string[];
    use_ml?: boolean;
    mode?: 'recommended' | 'latest';
    /** Draw a new batch instead of reusing today's feed. */
    force_refresh?: boolean;
  }) => {
    const response = await api.post<{ job_id: string; status: string; message: string }>(
      '/feed/generate',
      params,
      { timeout: 20000 }
    );
    return response.data;
  },

  getJobStatus: async (jobId: string) => {
    const response = await api.get<FeedJobStatus>(`/feed/status/${jobId}`, { timeout: 15000 });
    return response.data;
  },

  /** Omit `limit` to let the server return its configured feed size. */
  /** The feed last generated for this user, or generated_at null if there has never been one. */
  getCurrent: async () => (await api.get<CurrentFeed>('/feed/current', { timeout: 15000 })).data,

  getSaved: async () => {
    const response = await api.get<{ papers: Paper[]; total_saved: number }>(
      '/feed/saved'
    );
    return response.data;
  },

  // Evaluator-Optimizer refinement pass over the current feed.
  // Cached feed context is optional; the server recomputes it after expiry.
  refine: async (): Promise<RefineResponse> => {
    const response = await api.post<RefineResponse>('/feed/refine');
    return response.data;
  },

  // Onboarding seeds: diverse high-impact papers for a new user to rate,
  // which bootstraps the profile without waiting for organic interactions.
  getColdStartSeeds: async (count = 12): Promise<ColdStartResponse> => {
    const response = await api.get<ColdStartResponse>('/feed/coldstart', {
      params: { count },
    });
    return response.data;
  },
};

// Research profile API
export const profileApi = {
  get: async () => (await api.get<{ profile: ResearchProfile | null }>('/profile/research')).data.profile,

  /** The user's own description, turned into fields to confirm (one model call). */
  draft: async (description: string) =>
    (await api.post<{ draft: Omit<ProfileFields, 'seed_papers'>; suggested_papers: PaperOption[] }>(
      '/profile/research/draft', { description }, { timeout: 60000 })).data,

  save: async (fields: ProfileFields) =>
    (await api.put<{ profile: ResearchProfile }>('/profile/research', fields)).data.profile,

  /** Plain-language feedback ("too theoretical"), optionally about one feed paper. */
  feedback: async (text: string, paperId?: number) =>
    (await api.post<{ note: string; profile: ResearchProfile }>('/profile/research/feedback',
      { text, paper_id: paperId ?? null }, { timeout: 60000 })).data,

  searchPapers: async (q: string) =>
    (await api.get<{ papers: PaperOption[] }>('/profile/research/papers', { params: { q }, timeout: 30000 })).data.papers,

  readingPath: async (topic: string) =>
    (await api.post<ReadingPath>('/profile/reading-path', { topic }, { timeout: 180000 })).data,
};

// What's Hot API
export interface HFPaper {
  id: string;
  title: string;
  abstract: string;
  authors: string[];
  upvotes: number;
  url: string;
  arxiv_url: string;
  source: string;
  digest: string;
}

export interface GithubRepo {
  name: string;
  full_name: string;
  description: string;
  stars: number;
  /** Stars gained in the window (Trending page); null for the "new repo" fallback. */
  stars_gained?: number | null;
  language: string;
  url: string;
  source: string;
  trend_basis?: 'trending' | 'new';
}

export interface WhatsHotData {
  hf_papers: HFPaper[];
  github_repos: GithubRepo[];
  github_window: string;
  cached_at: string;
}

export const whatsHotApi = {
  get: async (): Promise<WhatsHotData> => {
    const response = await api.get<WhatsHotData>('/whats_hot');
    return response.data;
  },
  refresh: async (): Promise<WhatsHotData> => {
    const response = await api.post<WhatsHotData>('/whats_hot/refresh');
    return response.data;
  },
};

// Interactions API
export const interactionsApi = {
  /** Record that the user viewed, saved or dismissed a paper (item_id is the paper's database id). */
  create: async (data: {
    item_id: number;
    interaction_type: 'viewed' | 'saved' | 'dismissed';
  }) => {
    const response = await api.post('/interactions', data);
    return response.data;
  },

  getStats: async () => {
    const response = await api.get<InteractionStats>('/interactions/stats');
    return response.data;
  },

  getModelStatus: async () => {
    const response = await api.get<ModelStatus>('/interactions/model/status');
    return response.data;
  },

  retrain: async () => {
    const response = await api.post('/interactions/model/retrain');
    return response.data;
  },
};

// Knowledge base documents (questions over them go through Agent Chat)
export const qaApi = {
  uploadDocument: async (file: File, title?: string) => {
    const formData = new FormData();
    formData.append('file', file);
    if (title) formData.append('title', title);

    const response = await api.post('/qa/documents', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
  },

  listDocuments: async () => {
    const response = await api.get('/qa/documents');
    return response.data;
  },

  deleteDocument: async (docId: number) => {
    await api.delete(`/qa/documents/${docId}`);
  },
};

// Chat / Agent API
export const chatApi = {
  send: async (params: { message: string; session_id?: string; enable_eval?: boolean }) => {
    const response = await api.post<{
      reply: string;
      intent: string;
      agent_used: string;
      citations: { title: string; url: string; type: string }[];
      tools_called: string[];
      session_id: string;
      recognition_method: string;
      confidence: number;
      processing_time_ms: number;
      eval_scores?: Record<string, number>;
    }>('/chat/', params);
    return response.data;
  },

  getHistory: async (sessionId: string) => {
    const response = await api.get<{
      session_id: string;
      messages: { role: 'user' | 'assistant'; content: string }[];
      count: number;
    }>(`/chat/history/${sessionId}`);
    return response.data;
  },

  getResearchRuns: async () => {
    const response = await api.get<{ runs: { run_id: string; question: string; status: string; resumable: boolean }[] }>('/chat/research-runs');
    return response.data;
  },
};

export interface RepoToolStep { tool: string; args: Record<string, string>; summary: string }
export interface RepoCitation { text: string; path: string; start: number; end: number; verified: boolean; url: string }
export interface RepoOverview {
  focus: string;
  layout: { path: string; files: number; kinds: string }[];
  entry_points: { path: string; kind: string; detail: string }[];
  central_modules: { path: string; imported_by: number }[];
  import_edges: [string, string][];
  text_files: number; total_files: number; skipped_files: number; archive_truncated: boolean;
  focus_suggestions?: { path: string; source_files: number }[];
}

export interface PracticeEvent { id: number; kind: string; data: Record<string, any>; at?: string | null }
export interface OpenChoice { id: string; detail: string; question: string; options: string[]; recommended: string }
export interface TourStop { path: string; start_line: number; end_line: number; focus: string; question: string }
export interface CodeMap {
  functions: { name: string; start: number; end: number }[];
  links: { evidence_id: string; lines: number[]; function: string }[];
}
export interface ExerciseBlank { id: string; line: number; evidence_id: string; hint: string; prefix: string }
export interface GradeResult { id: string; verdict: 'correct' | 'partially' | 'incorrect' | 'missing'; feedback: string; answer: string }
export interface RepoFileView { path: string; start: number; end: number; total: number; highlight: [number, number]; lines: string[]; url: string }

/** fetch() with the axios client's auth handling: bearer token, and one refresh-and-retry on a 401. */
async function authorizedFetch(path: string, init: RequestInit): Promise<Response> {
  const send = (token: string | null) => fetch(`${API_URL}/api/v1${path}`, {
    ...init,
    headers: { ...init.headers, ...(token ? { Authorization: `Bearer ${token}` } : {}) },
  });
  const token = localStorage.getItem('access_token');
  const response = await send(token);
  return response.status === 401 ? send(await tokenForRetry(token)) : response;
}

/** POST/GET a server-sent event stream with the auth header (EventSource cannot send one). */
export async function readEventStream<E = any>(path: string, init: { method?: string; body?: unknown; signal?: AbortSignal },
  onEvent: (event: E) => void): Promise<void> {
  const response = await authorizedFetch(path, {
    method: init.method || 'GET',
    headers: init.body ? { 'Content-Type': 'application/json' } : {},
    body: init.body ? JSON.stringify(init.body) : undefined,
    signal: init.signal,
  });
  if (!response.ok || !response.body) {
    let detail = `HTTP ${response.status}`;
    try { detail = (await response.json()).detail || detail; } catch { /* not JSON */ }
    throw new Error(detail);
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let index: number;
    while ((index = buffer.indexOf('\n')) !== -1) {
      const line = buffer.slice(0, index);
      buffer = buffer.slice(index + 1);
      if (!line.startsWith('data: ')) continue;
      let event: E;
      try { event = JSON.parse(line.slice(6)); } catch { continue; /* skip a malformed line */ }
      onEvent(event);  // outside the try: a handler bug must surface, not pass as a malformed line
    }
  }
}

export interface PracticeRunSummary {
  run_id: string;
  kind: 'paper' | 'repo';
  source: string;
  status: 'queued' | 'running' | 'waiting' | 'completed' | 'failed' | 'cancelled';
  stage: string;
  title?: string;
  created_at?: string | null;
}

export interface PracticeRun extends PracticeRunSummary {
  error?: string | null;
  updated_at?: string | null;
  input_tokens?: number;
  output_tokens?: number;
  user_executions?: { status: string; code: string; evidence_status: 'agent_version' | 'user_modified_unverified';
    result: { passed?: boolean; stdout?: string; stderr?: string; error?: string } | null }[];
  messages?: { id: number; prompt: string; response: string | null; guided_question?: string | null;
    trace?: RepoToolStep[]; citations?: RepoCitation[] }[];
  comparison_status?: 'queued' | 'running' | 'completed' | 'failed' | null;
  events?: PracticeEvent[];
  comparison_error?: string | null;
  comparison?: {
    repo_url: string; commit_sha: string; module_path: string; module_url: string;
    selection_reason: string; summary: string;
    differences: { teaching: string; repository: string; reason?: string; paper_evidence_ids: string[]; repo_path: string }[];
    limitations: string; relationship: string;
  } | null;
  result: {
    paper_id?: string;
    execution_enabled?: boolean;
    title?: string;
    source_url?: string;
    evidence?: { id: string; text: string; source: string; chunk_index?: string; kind?: string; found_for?: string }[];
    manifest?: { feasible: boolean; method: string; reason: string;
      known_details: { detail: string; evidence_ids: string[]; quote?: string }[]; unknown_details: string[];
      simplifications?: string[]; open_choices?: OpenChoice[] };
    /** Names of the general method-family notes given to generation and review (background, not evidence). */
    method_notes?: string[];
    confirm_plan?: boolean;
    plan?: { choices: { id: string; detail: string; question: string; answer: string; recommended: boolean }[];
      simplifications: string[]; instructions: string } | null;
    gathered?: { round: number; targets: string[]; paper: { query: string; found: string }[]; repo_pieces: number; new_pieces: number }[];
    repo_evidence?: { repo_url: string; commit_sha?: string; label?: string; question?: string; answer?: string;
      citations?: RepoCitation[]; trace?: RepoToolStep[]; used?: number; error?: string } | null;
    personalized?: boolean;
    code_map?: CodeMap | null;
    tour?: TourStop[];
    evidence_source?: 'latex' | 'pdf' | '';
    detail_review?: { round: number; issue_count: number; review_type: string;
      core_mechanism: { implemented: boolean; note: string };
      checks: { detail: string; evidence_ids: string[]; status: 'implemented' | 'contradicted' | 'missing' | 'simplified';
        code_lines: string; note: string }[];
      /** Findings of deterministic code checks (always-true self-checks, random gradients), not of the model. */
      static_issues?: string[] };
    draft_revision?: { issues: string[]; accepted: boolean; diff: string; reason: string };
    timeline?: { attempt: number; diagnosis: string; diff: string;
      change_basis: { change: string; evidence_ids: string[]; kind: string }[];
      execution: { passed?: boolean; stdout?: string; stderr?: string; error?: string; timed_out?: boolean } }[];
    faithfulness?: { status: 'PASS' | 'UNCERTAIN'; reason: string; evidence_ids?: string[];
      unsupported_choices?: string[]; review_type?: string };
    linked_repo?: { url: string; provenance: string; relationship: string; source_section?: string; match_type?: string; label: string };
    linked_paper?: { arxiv_id: string; url: string; provenance: string; relationship: string; source_section?: string; match_type?: string; label: string };
    code?: string;
    explanation?: string;
    attempts?: number;
    execution?: { passed: boolean; stdout: string; stderr?: string; error?: string; timed_out?: boolean };
    questions?: string[];
    repo_url?: string;
    commit_sha?: string;
    file_tree?: string[];
    tree_truncated?: boolean;
    focus?: string;
    overview?: RepoOverview | null;
    inspected_files?: string[];
    architecture?: string;
    entry_points?: string[];
    file_roles?: { path: string; role: string }[];
    limitations?: string;
    stop_reason?: string;
  };
}

export const practiceApi = {
  capabilities: async () => (await api.get<{ enabled: boolean; paper_enabled: boolean; paper_execution_enabled: boolean; repo_enabled: boolean; modes: string[]; reason?: string }>('/practice/capabilities')).data,
  list: async () => (await api.get<{ runs: PracticeRunSummary[] }>('/practice/runs')).data.runs,
  get: async (runId: string) => (await api.get<PracticeRun>(`/practice/runs/${encodeURIComponent(runId)}`)).data,
  createPaper: async (arxivId: string, objective = '', confirmPlan = true) =>
    (await api.post<{ run_id: string }>('/practice/paper', { arxiv_id: arxivId, objective, confirm_plan: confirmPlan })).data,
  confirmPlan: async (runId: string, plan: { choices: Record<string, string>; simplifications: string[]; instructions: string }) =>
    (await api.post(`/practice/runs/${encodeURIComponent(runId)}/plan`, plan)).data,
  streamEvents: (runId: string, after: number, onEvent: (event: any) => void, signal: AbortSignal) =>
    readEventStream(`/practice/runs/${encodeURIComponent(runId)}/events/stream?after=${after}`, { signal }, onEvent),
  streamMessage: (runId: string, message: string, guidedQuestion: string, onEvent: (event: any) => void,
    signal?: AbortSignal, idempotencyKey = crypto.randomUUID()) =>
    readEventStream(`/practice/runs/${encodeURIComponent(runId)}/messages/stream`, {
      method: 'POST', signal, body: { message, guided_question: guidedQuestion, idempotency_key: idempotencyKey } }, onEvent),
  file: async (runId: string, path: string, start = 1, end = 0) =>
    (await api.get<RepoFileView>(`/practice/runs/${encodeURIComponent(runId)}/file`, { params: { path, start, end } })).data,
  exercise: async (runId: string) =>
    (await api.get<{ template: string; blanks: ExerciseBlank[] }>(`/practice/runs/${encodeURIComponent(runId)}/exercise`)).data,
  gradeExercise: async (runId: string, answers: Record<string, string>) =>
    (await api.post<{ results: GradeResult[]; execution: { passed?: boolean; stdout?: string; stderr?: string; error?: string } | null }>(
      `/practice/runs/${encodeURIComponent(runId)}/exercise/grade`, { answers })).data,
  createRepo: async (githubUrl: string) =>
    (await api.post<{ run_id: string }>('/practice/repo', { github_url: githubUrl })).data,
  message: async (runId: string, message: string, guidedQuestion = '', idempotencyKey = crypto.randomUUID()) =>
    (await api.post(`/practice/runs/${encodeURIComponent(runId)}/messages`, { message, guided_question: guidedQuestion, idempotency_key: idempotencyKey })).data,
  cancel: async (runId: string) =>
    (await api.post(`/practice/runs/${encodeURIComponent(runId)}/cancel`)).data,
  execute: async (runId: string, code: string, idempotencyKey: string) =>
    (await api.post(`/practice/runs/${encodeURIComponent(runId)}/executions`, { code, idempotency_key: idempotencyKey })).data,
  compare: async (runId: string) =>
    (await api.post(`/practice/runs/${encodeURIComponent(runId)}/comparison`)).data,
};


export interface PublicStats { papers_indexed: number; demo_available: boolean }

/** Unauthenticated facts for the landing page. */
export const publicApi = {
  stats: async () => (await api.get<PublicStats>('/public/stats', { timeout: 8000 })).data,
};
