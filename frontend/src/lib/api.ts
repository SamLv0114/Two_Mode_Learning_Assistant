/**
 * API client for the Learning Assistant backend
 */
import axios, { AxiosError, AxiosInstance } from 'axios';

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

// Handle auth errors
api.interceptors.response.use(
  (response) => response,
  (error: AxiosError) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('access_token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

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
}

export interface Token {
  access_token: string;
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
}

export interface Article {
  id: number;
  rank: number;
  source: string;
  title: string;
  url: string;
  author?: string;
  published_date?: string;
  upvotes: number;
  relevance_score: number;
  summary?: string;
}

export interface FeedResponse {
  papers: Paper[];
  articles: Article[];
  generated_at: string;
  time_window_days: number;
  focus_areas: string[];
  used_ml_ranking: boolean;
  total_papers_considered: number;
  total_articles_considered: number;
}

export interface InteractionStats {
  total: number;
  saved: number;
  viewed: number;
  dismissed: number;
  ready_for_training: boolean;
  interactions_until_training: number;
}

export interface FeedJobStatus {
  status: 'generating' | 'collecting' | 'ranking' | 'done' | 'error' | 'not_found';
  message?: string;
  papers_count?: number;
  articles_count?: number;
  used_ml_ranking?: boolean;
}

export type AgentEvent =
  | { type: 'session'; session_id: string }
  | { type: 'intent'; value: string; method: string; confidence: number }
  | { type: 'agent'; value: string }
  | { type: 'tool_call'; tool: string }
  | { type: 'tool_result'; tool: string; count: number }
  | { type: 'generating' }
  | { type: 'token'; value: string }
  | { type: 'done'; tools_called: string[]; citations: { title: string; url: string; type: string }[] }
  | { type: 'error'; value: string };

// Auth API
export const authApi = {
  register: async (data: {
    email: string;
    password: string;
    full_name?: string;
    interests?: string[];
    focus_areas?: string[];
  }) => {
    const response = await api.post<User & { access_token: string }>('/auth/register', data);
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
  }) => {
    const response = await api.post<{ job_id: string; status: string; message: string }>(
      '/feed/generate',
      params
    );
    return response.data;
  },

  getJobStatus: async (jobId: string) => {
    const response = await api.get<FeedJobStatus>(`/feed/status/${jobId}`);
    return response.data;
  },

  getPapers: async (limit = 20, offset = 0) => {
    const response = await api.get<Paper[]>('/feed/papers', {
      params: { limit, offset },
    });
    return response.data;
  },

  getArticles: async (limit = 10, offset = 0) => {
    const response = await api.get<Article[]>('/feed/articles', {
      params: { limit, offset },
    });
    return response.data;
  },

  getSaved: async () => {
    const response = await api.get<{ papers: Paper[]; articles: Article[]; total_saved: number }>(
      '/feed/saved'
    );
    return response.data;
  },
};

// Interactions API
export const interactionsApi = {
  create: async (data: {
    item_type: 'paper' | 'article';
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
    const response = await api.get('/interactions/model/status');
    return response.data;
  },

  retrain: async () => {
    const response = await api.post('/interactions/model/retrain');
    return response.data;
  },
};

// Q&A API
export const qaApi = {
  ask: async (params: {
    question: string;
    n_context?: number;
    filter_type?: string;
  }) => {
    const response = await api.post('/qa/ask', params);
    return response.data;
  },

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

  getStreamUrl: () => `${API_BASE_URL}/api/v1/chat/stream`,

  getAuthToken: () =>
    typeof window !== 'undefined' ? localStorage.getItem('access_token') : null,
};

export default api;
