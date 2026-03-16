import { createContext, useContext, useState, useEffect, useCallback, type ReactNode } from 'react';
import { auth as authApi, type User, type TokenResponse, ApiError } from './api';

interface AuthState {
  user: User | null;
  token: string | null;
  loading: boolean;
}

interface AuthContextValue extends AuthState {
  login: (username: string, password: string) => Promise<void>;
  signup: (username: string, email: string, password: string, displayName?: string) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextValue | null>(null);

function saveSession(data: TokenResponse) {
  localStorage.setItem('dappy_token', data.access_token);
  localStorage.setItem('dappy_user', JSON.stringify({
    user_id: data.user_id,
    username: data.username,
    display_name: data.display_name,
  }));
}

function clearSession() {
  localStorage.removeItem('dappy_token');
  localStorage.removeItem('dappy_user');
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<AuthState>({
    user: null,
    token: localStorage.getItem('dappy_token'),
    loading: true,
  });

  useEffect(() => {
    const token = localStorage.getItem('dappy_token');
    if (!token) {
      setState({ user: null, token: null, loading: false });
      return;
    }

    authApi.me()
      .then((user) => setState({ user, token, loading: false }))
      .catch(() => {
        clearSession();
        setState({ user: null, token: null, loading: false });
      });
  }, []);

  const login = useCallback(async (username: string, password: string) => {
    const data = await authApi.login(username, password);
    saveSession(data);
    const user = await authApi.me();
    setState({ user, token: data.access_token, loading: false });
  }, []);

  const signup = useCallback(async (username: string, email: string, password: string, displayName?: string) => {
    const data = await authApi.signup(username, email, password, displayName);
    saveSession(data);
    const user = await authApi.me();
    setState({ user, token: data.access_token, loading: false });
  }, []);

  const logout = useCallback(() => {
    clearSession();
    setState({ user: null, token: null, loading: false });
  }, []);

  return (
    <AuthContext.Provider value={{ ...state, login, signup, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used within AuthProvider');
  return ctx;
}
