const API_BASE = '/api';

function getToken(): string | null {
  return localStorage.getItem('dappy_token');
}

function authHeaders(): Record<string, string> {
  const token = getToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...authHeaders(),
      ...init?.headers,
    },
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new ApiError(body.detail || 'Request failed', res.status);
  }

  return res.json();
}

export class ApiError extends Error {
  constructor(message: string, public status: number) {
    super(message);
    this.name = 'ApiError';
  }
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
  user_id: string;
  username: string;
  display_name: string | null;
  expires_in: number;
}

export interface User {
  user_id: string;
  username: string;
  email: string;
  display_name: string | null;
  created_at: string;
}

export interface ChatResponse {
  assistant_message: string;
  conversation_history: Record<string, unknown>[];
  metadata: Record<string, unknown>;
  debug?: Record<string, unknown>;
}

export const auth = {
  signup(username: string, email: string, password: string, display_name?: string) {
    return request<TokenResponse>('/auth/signup', {
      method: 'POST',
      body: JSON.stringify({ username, email, password, display_name }),
    });
  },

  login(username: string, password: string) {
    return request<TokenResponse>('/auth/login', {
      method: 'POST',
      body: JSON.stringify({ username, password }),
    });
  },

  me() {
    return request<User>('/auth/me');
  },
};

export interface IngestRequest {
  source_type: string;
  source: string;
  session_id?: string;
}

export interface IngestResponse {
  status: string;
  chunks_parsed: number;
  memories_created: number;
  errors: string[];
  session_id: string;
}

export const user = {
  deleteAllMemories() {
    return request<{ status: string; message: string; deleted: Record<string, unknown> }>('/user/memories', {
      method: 'DELETE',
    });
  },
  
  ingestMemories(data: IngestRequest) {
    return request<IngestResponse>('/user/ingest', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  },
};

export const chat = {
  send(sessionId: string, message: string): Promise<ChatResponse> {
    return request<ChatResponse>('/chat', {
      method: 'POST',
      body: JSON.stringify({ session_id: sessionId, message }),
    });
  },

  async *stream(sessionId: string, message: string): AsyncGenerator<string> {
    const res = await fetch(`${API_BASE}/chat/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...authHeaders(),
      },
      body: JSON.stringify({ session_id: sessionId, message }),
    });

    if (!res.ok) {
      const body = await res.json().catch(() => ({ detail: res.statusText }));
      throw new ApiError(body.detail || 'Stream failed', res.status);
    }

    const reader = res.body?.getReader();
    if (!reader) return;

    const decoder = new TextDecoder();
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      yield decoder.decode(value, { stream: true });
    }
  },
};
