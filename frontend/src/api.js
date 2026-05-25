const API_BASE = "http://127.0.0.1:8000";

function authHeaders(token) {
  if (!token) return { "Content-Type": "application/json" };
  return {
    "Content-Type": "application/json",
    Authorization: `Bearer ${token}`,
  };
}

async function parseOrThrow(response, fallbackMessage) {
  if (response.ok) return response.json();

  let detail = fallbackMessage;
  try {
    const data = await response.json();
    if (typeof data?.detail === "string") detail = data.detail;
  } catch {
    // ignore
  }
  throw new Error(detail);
}

export async function register(payload) {
  const response = await fetch(`${API_BASE}/api/auth/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return parseOrThrow(response, "Не удалось зарегистрироваться");
}

export async function login(payload) {
  const response = await fetch(`${API_BASE}/api/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return parseOrThrow(response, "Не удалось выполнить вход");
}

export async function fetchMe(token) {
  const response = await fetch(`${API_BASE}/api/auth/me`, {
    headers: authHeaders(token),
  });
  return parseOrThrow(response, "Сессия недействительна");
}

export async function becomeAdmin(token) {
  const response = await fetch(`${API_BASE}/api/auth/become-admin`, {
    method: "POST",
    headers: authHeaders(token),
  });
  return parseOrThrow(response, "Не удалось выдать права администратора");
}

export async function fetchAdminAnalytics(token) {
  const response = await fetch(`${API_BASE}/api/auth/admin/analytics`, {
    headers: authHeaders(token),
  });
  return parseOrThrow(response, "Не удалось загрузить аналитику");
}

export async function fetchChats(token) {
  const response = await fetch(`${API_BASE}/api/chats`, {
    headers: authHeaders(token),
  });
  return parseOrThrow(response, "Не удалось получить список чатов");
}

export async function createChat(title, token) {
  const response = await fetch(`${API_BASE}/api/chats`, {
    method: "POST",
    headers: authHeaders(token),
    body: JSON.stringify({ title }),
  });
  return parseOrThrow(response, "Не удалось создать чат");
}

export async function deleteChat(chatId, token) {
  const response = await fetch(`${API_BASE}/api/chats/${chatId}`, {
    method: "DELETE",
    headers: authHeaders(token),
  });
  return parseOrThrow(response, "Не удалось удалить чат");
}

export async function fetchMessages(chatId, token) {
  const response = await fetch(`${API_BASE}/api/chats/${chatId}/messages`, {
    headers: authHeaders(token),
  });
  return parseOrThrow(response, "Не удалось загрузить сообщения");
}

export async function sendMessage(chatId, content, token) {
  const response = await fetch(`${API_BASE}/api/chats/${chatId}/messages`, {
    method: "POST",
    headers: authHeaders(token),
    body: JSON.stringify({ content }),
  });
  return parseOrThrow(response, "Не удалось отправить сообщение");
}
