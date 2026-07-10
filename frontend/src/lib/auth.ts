/**
 * Authentication state management using Zustand
 */
import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { authApi, User } from './api';

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
          localStorage.setItem('access_token', response.access_token);
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
          localStorage.setItem('access_token', response.access_token);
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
        localStorage.removeItem('access_token');
        set({
          user: null,
          token: null,
          isAuthenticated: false,
        });
      },

      fetchProfile: async () => {
        const token = localStorage.getItem('access_token');
        if (!token) {
          set({ isAuthenticated: false });
          return;
        }

        set({ isLoading: true });
        try {
          const user = await authApi.getProfile();
          set({
            user,
            token,
            isAuthenticated: true,
          });
        } catch (error) {
          // Token invalid
          localStorage.removeItem('access_token');
          set({
            user: null,
            token: null,
            isAuthenticated: false,
          });
        } finally {
          set({ isLoading: false });
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
