/**
 * Authentication state management using Zustand
 */
import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { authApi, User } from './api';

// Ignore profile responses started under a previous login session.
let sessionVersion = 0;
let profileRequestVersion = 0;

interface AuthState {
  user: User | null;
  token: string | null;
  isLoading: boolean;
  isAuthenticated: boolean;

  // Actions
  login: (email: string, password: string) => Promise<void>;
  register: (data: {
    email: string;
    password: string;
    full_name?: string;
    interests?: string[];
    focus_areas?: string[];
  }) => Promise<void>;
  logout: () => void;
  fetchProfile: () => Promise<void>;
  updateProfile: (data: {
    full_name?: string;
    interests?: string[];
    focus_areas?: string[];
    password?: string;
  }) => Promise<void>;
}

export const useAuth = create<AuthState>()(
  persist(
    (set, get) => ({
      user: null,
      token: null,
      isLoading: false,
      isAuthenticated: false,

      login: async (email: string, password: string) => {
        set({ isLoading: true });
        try {
          const response = await authApi.login(email, password);
          sessionVersion += 1;
          localStorage.setItem('access_token', response.access_token);
          localStorage.setItem('refresh_token', response.refresh_token);
          set({
            token: response.access_token,
            isAuthenticated: true,
          });
          // Fetch user profile after login
          await get().fetchProfile();
        } finally {
          set({ isLoading: false });
        }
      },

      register: async (data) => {
        set({ isLoading: true });
        try {
          const response = await authApi.register(data);
          sessionVersion += 1;
          localStorage.setItem('access_token', response.access_token);
          localStorage.setItem('refresh_token', response.refresh_token);
          set({
            user: response,
            token: response.access_token,
            isAuthenticated: true,
          });
        } finally {
          set({ isLoading: false });
        }
      },

      logout: () => {
        sessionVersion += 1;
        localStorage.removeItem('access_token');
        localStorage.removeItem('refresh_token');
        set({
          user: null,
          token: null,
          isAuthenticated: false,
        });
      },

      fetchProfile: async () => {
        const requestVersion = sessionVersion;
        const profileVersion = ++profileRequestVersion;
        const isCurrent = () => requestVersion === sessionVersion && profileVersion === profileRequestVersion;
        const token = localStorage.getItem('access_token');
        if (!token) {
          if (isCurrent()) {
            set({ user: null, token: null, isAuthenticated: false });
          }
          return;
        }

        set({ isLoading: true });
        try {
          const user = await authApi.getProfile();
          if (!isCurrent()) return;
          set({
            user,
            token: localStorage.getItem('access_token'),
            isAuthenticated: true,
          });
        } catch (error) {
          if (!isCurrent()) return;
          // Token invalid, and the response interceptor's own refresh
          // attempt (if any) already failed too by the time this runs.
          localStorage.removeItem('access_token');
          localStorage.removeItem('refresh_token');
          set({
            user: null,
            token: null,
            isAuthenticated: false,
          });
        } finally {
          if (isCurrent()) {
            set({ isLoading: false });
          }
        }
      },

      updateProfile: async (data) => {
        set({ isLoading: true });
        try {
          const user = await authApi.updateProfile(data);
          set({ user });
        } finally {
          set({ isLoading: false });
        }
      },
    }),
    {
      name: 'auth-storage',
      partialize: (state) => ({
        token: state.token,
        isAuthenticated: state.isAuthenticated,
      }),
    }
  )
);
