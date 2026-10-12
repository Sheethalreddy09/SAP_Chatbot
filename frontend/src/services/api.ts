import type { User, Chat, Message, Citation, WebSource } from '../types';

const API_BASE_URL = 'http://localhost:8005/api';

export const getStoredToken = (): string | null => {
  return localStorage.getItem('sap_assistant_token');
};

export const setStoredToken = (token: string): void => {
  localStorage.setItem('sap_assistant_token', token);
};

export const clearStoredToken = (): void => {
  localStorage.removeItem('sap_assistant_token');
};

const authHeaders = (): Record<string, string> => {
  const token = getStoredToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
};

// Auth API
export const apiRegister = async (name: string, email: string, password: string) => {
  const resp = await fetch(`${API_BASE_URL}/auth/register`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, email, password }),
  });
  if (!resp.ok) {
    const errorData = await resp.json().catch(() => ({}));
    throw new Error(errorData.detail || 'Registration failed');
  }
  return resp.json();
};

export const apiLogin = async (email: string, password: string) => {
  const resp = await fetch(`${API_BASE_URL}/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password }),
  });
  if (!resp.ok) {
    const errorData = await resp.json().catch(() => ({}));
    throw new Error(errorData.detail || 'Invalid email or password');
  }
  return resp.json();
};

export const apiGetMe = async (): Promise<User> => {
  const resp = await fetch(`${API_BASE_URL}/auth/me`, {
    headers: { ...authHeaders() },
  });
  if (!resp.ok) {
    throw new Error('Unauthorized');
  }
  return resp.json();
};

// Chats API
export const apiGetChats = async (search?: string): Promise<Chat[]> => {
  const url = new URL(`${API_BASE_URL}/chats`);
  if (search) url.searchParams.append('search', search);

  const resp = await fetch(url.toString(), {
    headers: { ...authHeaders() },
  });
  if (!resp.ok) throw new Error('Failed to load chats');
  return resp.json();
};

export const apiCreateChat = async (title?: string): Promise<Chat> => {
  const resp = await fetch(`${API_BASE_URL}/chats`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify({ title: title || 'New SAP Chat' }),
  });
  if (!resp.ok) throw new Error('Failed to create chat');
  return resp.json();
};

export const apiGetChat = async (chatId: string): Promise<Chat & { messages: Message[] }> => {
  const resp = await fetch(`${API_BASE_URL}/chats/${chatId}`, {
    headers: { ...authHeaders() },
  });
  if (!resp.ok) throw new Error('Chat not found');
  return resp.json();
};

export const apiUpdateChat = async (chatId: string, title: string): Promise<Chat> => {
  const resp = await fetch(`${API_BASE_URL}/chats/${chatId}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify({ title }),
  });
  if (!resp.ok) throw new Error('Failed to update chat');
  return resp.json();
};

export const apiDeleteChat = async (chatId: string): Promise<void> => {
  const resp = await fetch(`${API_BASE_URL}/chats/${chatId}`, {
    method: 'DELETE',
    headers: { ...authHeaders() },
  });
  if (!resp.ok) throw new Error('Failed to delete chat');
};

// Messages API
export const apiGetMessages = async (chatId: string): Promise<Message[]> => {
  const resp = await fetch(`${API_BASE_URL}/chats/${chatId}/messages`, {
    headers: { ...authHeaders() },
  });
  if (!resp.ok) throw new Error('Failed to load messages');
  return resp.json();
};

export const apiSendMessage = async (chatId: string, content: string): Promise<Message> => {
  const resp = await fetch(`${API_BASE_URL}/chats/${chatId}/messages`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify({ content }),
  });
  if (!resp.ok) throw new Error('Failed to send message');
  return resp.json();
};

export const apiSavePartialMessage = async (chatId: string, content: string): Promise<void> => {
  try {
    await fetch(`${API_BASE_URL}/chats/${chatId}/messages/save-partial`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...authHeaders() },
      body: JSON.stringify({ content }),
    });
  } catch (err) {
    console.error('Failed to save partial message:', err);
  }
};

// SSE Streaming
export interface StreamStatus {
  type: 'status';
  stage: 'thinking' | 'retrieval' | 'reranking' | 'grounding' | 'generation' | string;
  message: string;
}

export interface StreamMetadata {
  message_id?: string;
  source_type?: 'knowledge_base' | 'web' | 'refusal' | 'error' | 'interrupted';
  grounding_score?: number;
  citations?: Citation[];
  web_sources?: WebSource[];
  is_in_rag_pipeline?: boolean;
  kb_score?: number;
  web_score?: number;
  winning_score?: number;
  full_answer?: string;
}

export interface StreamCallbacks {
  onStatus?: (status: StreamStatus) => void;
  onMetadata?: (meta: StreamMetadata) => void;
  onToken?: (token: string) => void;
  onDone?: (meta: StreamMetadata, fullAnswer?: string) => void;
  onError?: (err: any) => void;
}

export const apiStreamMessage = async (
  chatId: string,
  content: string,
  callbacks: StreamCallbacks,
  signal?: AbortSignal
): Promise<void> => {
  try {
    const resp = await fetch(`${API_BASE_URL}/chats/${chatId}/messages?stream=true`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...authHeaders(),
      },
      body: JSON.stringify({ content }),
      signal,
    });

    if (!resp.ok) {
      throw new Error(`Failed to stream message: ${resp.statusText}`);
    }

    const reader = resp.body?.getReader();
    if (!reader) throw new Error('ReadableStream not supported');

    const decoder = new TextDecoder();
    let buffer = '';
    let isCompleted = false;
    let finalReceivedAnswer: string | undefined = undefined;
    let lastDoneMeta: StreamMetadata = {};

    const finishStream = (meta?: StreamMetadata, fullAnswer?: string) => {
      if (!isCompleted) {
        isCompleted = true;
        const effectiveMeta = meta || lastDoneMeta;
        callbacks.onDone?.(effectiveMeta, fullAnswer || finalReceivedAnswer);
      }
    };

    const processBlock = (block: string) => {
      const trimmed = block.trim();
      if (!trimmed) return;

      const lines = trimmed.split('\n');
      for (const line of lines) {
        const trimmedLine = line.trim();
        if (trimmedLine.startsWith('data:')) {
          const rawData = trimmedLine.slice(5).trim();
          if (!rawData) continue;

          try {
            const parsed = JSON.parse(rawData);

            if (parsed.type === 'status') {
              callbacks.onStatus?.(parsed as StreamStatus);
            } else if (parsed.type === 'token') {
              const tokenContent = parsed.content ?? parsed.token ?? '';
              if (tokenContent) {
                callbacks.onToken?.(tokenContent);
              }
            } else if (parsed.type === 'done') {
              lastDoneMeta = parsed;
              const ans = parsed.answer || parsed.full_answer;
              if (ans) {
                finalReceivedAnswer = ans;
              }
              finishStream(parsed, ans);
            } else if (parsed.type === 'error') {
              callbacks.onError?.(new Error(parsed.message || 'Generation error occurred'));
            } else if (parsed.citations !== undefined || parsed.message_id !== undefined) {
              lastDoneMeta = { ...lastDoneMeta, ...parsed };
              callbacks.onMetadata?.(parsed);
            }
          } catch (e) {
            console.error('SSE JSON parse error:', e, rawData);
          }
        }
      }
    };

    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        if (buffer.trim()) {
          processBlock(buffer);
          buffer = '';
        }
        break;
      }

      buffer += decoder.decode(value, { stream: true });
      const normalized = buffer.replace(/\r\n/g, '\n');
      const blocks = normalized.split('\n\n');
      buffer = blocks.pop() || '';

      for (const block of blocks) {
        processBlock(block);
      }
    }

    finishStream();
  } catch (err: any) {
    if (signal?.aborted || err?.name === 'AbortError') {
      console.log('Streaming aborted by user.');
      return;
    }
    callbacks.onError?.(err);
  }
};
