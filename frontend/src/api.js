const API_BASE = "http://127.0.0.1:8000";

function authHeaders(token) {
  if (!token) {
    return {
      "Content-Type": "application/json",
    };
  }

  return {
    "Content-Type": "application/json",
    Authorization: `Bearer ${token}`,
  };
}

function authOnlyHeaders(token) {
  return {
    Authorization: `Bearer ${token}`,
  };
}

function sleep(ms) {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

async function fetchWithRetry(
  url,
  options = {},
  retries = 10,
  delay = 500
) {
  try {
    return await fetch(url, options);
  } catch (error) {
    if (retries <= 0) {
      throw error;
    }

    await sleep(delay);

    return fetchWithRetry(
      url,
      options,
      retries - 1,
      delay
    );
  }
}

async function parseOrThrow(response, fallbackMessage) {
  if (response.ok) {
    return response.json();
  }

  let detail = fallbackMessage;

  try {
    const data = await response.json();

    if (typeof data?.detail === "string") {
      detail = data.detail;
    }
  } catch {
    // ignore
  }

  throw new Error(detail);
}

export async function register(payload) {
  const response = await fetchWithRetry(
    `${API_BASE}/api/auth/register`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    }
  );

  return parseOrThrow(
    response,
    "Не удалось зарегистрироваться"
  );
}

export async function login(payload) {
  const response = await fetchWithRetry(
    `${API_BASE}/api/auth/login`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    }
  );

  return parseOrThrow(
    response,
    "Не удалось выполнить вход"
  );
}

export async function fetchMe(token) {
  const response = await fetchWithRetry(
    `${API_BASE}/api/auth/me`,
    {
      headers: authHeaders(token),
    }
  );

  return parseOrThrow(
    response,
    "Сессия недействительна"
  );
}

export async function becomeAdmin(token) {
  const response = await fetchWithRetry(
    `${API_BASE}/api/auth/become-admin`,
    {
      method: "POST",
      headers: authHeaders(token),
    }
  );

  return parseOrThrow(
    response,
    "Не удалось выдать права администратора"
  );
}

export async function fetchAdminAnalytics(token) {
  const response = await fetchWithRetry(
    `${API_BASE}/api/subscriptions/admin/analytics`,
    {
      headers: authHeaders(token),
    }
  );

  return parseOrThrow(
    response,
    "Не удалось загрузить аналитику"
  );
}

export async function fetchSubscriptionPlans(token) {
  const response = await fetchWithRetry(
    `${API_BASE}/api/subscriptions/plans`,
    {
      headers: authHeaders(token),
    }
  );

  return parseOrThrow(
    response,
    "Не удалось загрузить тарифы"
  );
}

export async function fetchCurrentSubscription(token) {
  const response = await fetchWithRetry(
    `${API_BASE}/api/subscriptions/me`,
    {
      headers: authHeaders(token),
    }
  );

  return parseOrThrow(
    response,
    "Не удалось загрузить подписку"
  );
}

export async function updateSubscription(plan, token) {
  const response = await fetchWithRetry(
    `${API_BASE}/api/subscriptions/me`,
    {
      method: "PUT",
      headers: authHeaders(token),
      body: JSON.stringify({ plan }),
    }
  );

  return parseOrThrow(
    response,
    "Не удалось изменить подписку"
  );
}

export async function fetchChats(token) {
  const response = await fetchWithRetry(
    `${API_BASE}/api/chats`,
    {
      headers: authHeaders(token),
    }
  );

  return parseOrThrow(
    response,
    "Не удалось получить список чатов"
  );
}

export async function createChat(title, token) {
  const response = await fetchWithRetry(
    `${API_BASE}/api/chats`,
    {
      method: "POST",
      headers: authHeaders(token),
      body: JSON.stringify({ title }),
    }
  );

  return parseOrThrow(
    response,
    "Не удалось создать чат"
  );
}

export async function deleteChat(chatId, token) {
  const response = await fetchWithRetry(
    `${API_BASE}/api/chats/${chatId}`,
    {
      method: "DELETE",
      headers: authHeaders(token),
    }
  );

  return parseOrThrow(
    response,
    "Не удалось удалить чат"
  );
}

export async function updateChat(chatId, title, token) {
  const response = await fetchWithRetry(
    `${API_BASE}/api/chats/${chatId}`,
    {
      method: "PATCH",
      headers: authHeaders(token),
      body: JSON.stringify({ title }),
    }
  );

  return parseOrThrow(
    response,
    "Не удалось изменить название чата"
  );
}

export async function fetchMessages(chatId, token) {
  const response = await fetchWithRetry(
    `${API_BASE}/api/chats/${chatId}/messages`,
    {
      headers: authHeaders(token),
    }
  );

  return parseOrThrow(
    response,
    "Не удалось загрузить сообщения"
  );
}

export async function sendMessage(chatId, content, file, token) {
  const formData = new FormData();

  formData.append("content", content || "");

  if (file) {
    formData.append("file", file);
  }

  const response = await fetchWithRetry(
    `${API_BASE}/api/chats/${chatId}/messages`,
    {
      method: "POST",
      headers: authOnlyHeaders(token),
      body: formData,
    }
  );

  return parseOrThrow(
    response,
    "Не удалось отправить сообщение"
  );
}

export async function downloadAnalysisResult(
  analysisId,
  token
) {
  const response = await fetchWithRetry(
    `${API_BASE}/api/analysis-results/${analysisId}/download`,
    {
      headers: authOnlyHeaders(token),
    }
  );

  if (!response.ok) {
    throw new Error(
      "Не удалось скачать результат анализа"
    );
  }

  const blob = await response.blob();

  const url = window.URL.createObjectURL(blob);

  const link = document.createElement("a");

  link.href = url;
  link.download = "analysis_result.docx";

  document.body.appendChild(link);

  link.click();

  link.remove();

  window.URL.revokeObjectURL(url);
}

