const API_BASE = "http://127.0.0.1:8000";

export async function fetchChats() {
  const response = await fetch(`${API_BASE}/api/chats`);
  if (!response.ok) throw new Error("Не удалось получить список чатов");
  return response.json();
}

export async function createChat(title) {
  const response = await fetch(`${API_BASE}/api/chats`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title }),
  });
  if (!response.ok) throw new Error("Не удалось создать чат");
  return response.json();
}

export async function fetchMessages(chatId) {
  const response = await fetch(`${API_BASE}/api/chats/${chatId}/messages`);
  if (!response.ok) throw new Error("Не удалось загрузить сообщения");
  return response.json();
}

export async function sendMessage(chatId, content) {
  const response = await fetch(`${API_BASE}/api/chats/${chatId}/messages`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ content }),
  });
  if (!response.ok) throw new Error("Не удалось отправить сообщение");
  return response.json();
}
