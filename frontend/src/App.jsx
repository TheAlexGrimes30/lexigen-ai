import { useEffect, useMemo, useRef, useState } from "react";
import {
  createChat,
  deleteChat,
  downloadAnalysisResult,
  fetchChats,
  fetchMe,
  fetchMessages,
  login,
  register,
  sendMessage,
} from "./api";

const TOKEN_KEY = "lexigen_token";
const THEME_KEY = "lexigen_theme";

function getInitialTheme() {
  const savedTheme = localStorage.getItem(THEME_KEY);

  return savedTheme === "light" || savedTheme === "dark"
    ? savedTheme
    : "dark";
}

export default function App() {
  const fileInputRef = useRef(null);

  const [activeTab, setActiveTab] = useState("login");
  const [authMode, setAuthMode] = useState("login");
  const [theme, setTheme] = useState(getInitialTheme);
  const [authToken, setAuthToken] = useState(
    localStorage.getItem(TOKEN_KEY) || ""
  );
  const [currentUser, setCurrentUser] = useState(null);

  const [authName, setAuthName] = useState("");
  const [authEmail, setAuthEmail] = useState("");
  const [authPassword, setAuthPassword] = useState("");
  const [authPasswordConfirm, setAuthPasswordConfirm] = useState("");
  const [isPasswordVisible, setIsPasswordVisible] = useState(false);

  const [chats, setChats] = useState([]);
  const [selectedChatId, setSelectedChatId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [newChatTitle, setNewChatTitle] = useState("");
  const [messageText, setMessageText] = useState("");
  const [selectedFile, setSelectedFile] = useState(null);
  const [isMessageSending, setIsMessageSending] = useState(false);
  const [error, setError] = useState("");
  const [isChatSidebarVisible, setIsChatSidebarVisible] = useState(true);

  const isLightTheme = theme === "light";

  const selectedChat = useMemo(
    () => chats.find((chat) => chat.id === selectedChatId) || null,
    [chats, selectedChatId]
  );

  const navItems = useMemo(() => {
    const items = [
      { key: "home", label: "Главная" },
      { key: "profile", label: "Личный кабинет" },
      { key: "chats", label: "Ваши чаты" },
    ];

    if (currentUser?.role === "admin") {
      items.splice(2, 0, { key: "analytics", label: "Аналитика" });
    }

    return items;
  }, [currentUser]);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem(THEME_KEY, theme);
  }, [theme]);

  useEffect(() => {
    if (!authToken) {
      setCurrentUser(null);
      setChats([]);
      setMessages([]);
      setSelectedChatId(null);
      return;
    }

    bootstrapSession();
  }, [authToken]);

  useEffect(() => {
    if (!selectedChatId || !authToken) return;

    loadMessages(selectedChatId, authToken);
  }, [selectedChatId, authToken]);

  async function bootstrapSession() {
    try {
      setError("");

      const user = await fetchMe(authToken);

      setCurrentUser(user);

      await loadChats(authToken);
    } catch (e) {
      clearSession();
      setError(e.message || "Сессия истекла. Войдите заново.");
    }
  }

  function clearSession() {
    localStorage.removeItem(TOKEN_KEY);
    setAuthToken("");
    setCurrentUser(null);
    setChats([]);
    setMessages([]);
    setSelectedChatId(null);
    setActiveTab("home");
  }

  function handleLogout() {
    clearSession();
    setAuthMode("login");
    setAuthPassword("");
    setActiveTab("login");
  }

  function toggleTheme() {
    setTheme((prevTheme) => (
      prevTheme === "dark" ? "light" : "dark"
    ));
  }

  async function handleAuthSubmit(e) {
    e.preventDefault();

    if (authMode === "register" && authPassword !== authPasswordConfirm) {
      setError("Пароли не совпадают");
      return;
    }

    try {
      setError("");

      const payload =
        authMode === "register"
          ? {
              name: authName.trim(),
              email: authEmail.trim(),
              password: authPassword,
              password_confirm: authPasswordConfirm,
            }
          : {
              email: authEmail.trim(),
              password: authPassword,
            };

      const data = authMode === "register"
        ? await register(payload)
        : await login(payload);

      localStorage.setItem(TOKEN_KEY, data.access_token);

      setAuthToken(data.access_token);
      setCurrentUser(data.user);
      setAuthPassword("");
      setAuthPasswordConfirm("");
      setAuthName("");
      setActiveTab("chats");
    } catch (e) {
      setError(e.message || "Ошибка авторизации");
    }
  }

  async function loadChats(token = authToken) {
    if (!token) return;

    try {
      setError("");

      const data = await fetchChats(token);

      setChats(data);

      if (!selectedChatId && data.length > 0) {
        setSelectedChatId(data[0].id);
      }

      if (selectedChatId && !data.find((chat) => chat.id === selectedChatId)) {
        setSelectedChatId(data[0]?.id || null);
      }
    } catch (e) {
      setError(e.message || "Ошибка загрузки чатов");
    }
  }

  async function loadMessages(chatId, token = authToken) {
    if (!token) return;

    try {
      setError("");

      const data = await fetchMessages(chatId, token);

      setMessages(data);
    } catch (e) {
      setError(e.message || "Ошибка загрузки сообщений");
      setMessages([]);
    }
  }

  async function onCreateChat(e) {
    e.preventDefault();

    if (!newChatTitle.trim() || !authToken) return;

    try {
      setError("");

      const chat = await createChat(newChatTitle.trim(), authToken);
      const updated = [chat, ...chats];

      setChats(updated);
      setSelectedChatId(chat.id);
      setNewChatTitle("");
      setActiveTab("chats");
    } catch (e) {
      setError(e.message || "Ошибка создания чата");
    }
  }

  function onFileChange(e) {
    const file = e.target.files?.[0] || null;

    if (!file) {
      setSelectedFile(null);
      return;
    }

    const lowerName = file.name.toLowerCase();

    if (!lowerName.endsWith(".docx") && !lowerName.endsWith(".pdf")) {
      setError("Можно загрузить только DOCX или PDF");
      e.target.value = "";
      setSelectedFile(null);
      return;
    }

    setError("");
    setSelectedFile(file);
  }

  function clearSelectedFile() {
    setSelectedFile(null);

    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  }

  async function onSendMessage(e) {
    e.preventDefault();

    if (
      !selectedChatId ||
      (!messageText.trim() && !selectedFile) ||
      !authToken ||
      isMessageSending
    ) {
      return;
    }

    const textToSend = messageText.trim();
    const fileToSend = selectedFile;

    try {
      setError("");
      setIsMessageSending(true);
      setMessageText("");
      clearSelectedFile();

      const optimisticMessage = {
        id: `local-${Date.now()}`,
        chat_id: selectedChatId,
        user_id: currentUser?.id || "local",
        role: "user",
        content: fileToSend
          ? `${textToSend || "Документ отправлен на анализ"}\nФайл: ${fileToSend.name}`
          : textToSend,
        created_at: new Date().toISOString(),
      };

      setMessages((prev) => [...prev, optimisticMessage]);

      const newMessages = await sendMessage(
        selectedChatId,
        textToSend,
        fileToSend,
        authToken
      );

      setMessages((prev) => [
        ...prev.filter((msg) => msg.id !== optimisticMessage.id),
        ...newMessages,
      ]);
    } catch (e) {
      setError(e.message || "Ошибка отправки сообщения");
    } finally {
      setIsMessageSending(false);
    }
  }

  async function onDownloadAnalysis(analysisId) {
    try {
      setError("");

      await downloadAnalysisResult(
        analysisId,
        authToken
      );
    } catch (e) {
      setError(e.message || "Ошибка скачивания результата анализа");
    }
  }

  async function onDeleteChat(chatId) {
    if (!chatId || !authToken) return;

    try {
      setError("");

      await deleteChat(chatId, authToken);

      const updatedChats = chats.filter((chat) => chat.id !== chatId);

      setChats(updatedChats);

      if (updatedChats.length === 0) {
        setSelectedChatId(null);
        setMessages([]);
        return;
      }

      if (selectedChatId === chatId) {
        setSelectedChatId(updatedChats[0].id);
      }
    } catch (e) {
      setError(e.message || "Ошибка удаления чата");
    }
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="logo">LexigenAI</div>

        <div className="topbar-right">
          <nav className="nav">
            {currentUser
              ? navItems.map((item) => (
                  <button
                    key={item.key}
                    className={`nav-btn ${activeTab === item.key ? "active" : ""}`}
                    onClick={() => setActiveTab(item.key)}
                  >
                    {item.label}
                  </button>
                ))
              : [
                  { key: "login", label: "Вход" },
                  { key: "register", label: "Регистрация" },
                ].map((item) => (
                  <button
                    key={item.key}
                    className={`nav-btn ${authMode === item.key ? "active" : ""}`}
                    onClick={() => {
                      setAuthMode(item.key);
                      setActiveTab(item.key);
                      setAuthPassword("");
                      setAuthPasswordConfirm("");
                      setError("");
                    }}
                  >
                    {item.label}
                  </button>
                ))}
          </nav>

          <button
            type="button"
            className="theme-toggle-btn"
            onClick={toggleTheme}
          >
            {isLightTheme ? "Тёмная тема" : "Светлая тема"}
          </button>
        </div>
      </header>

      <main className={`page ${activeTab === "chats" ? "chats-page" : ""}`}>
        {!currentUser && authMode === "login" && (
          <section className="card auth-card">
            <h2>Страница входа</h2>

            <form onSubmit={handleAuthSubmit} className="auth-form">
              <input
                value={authEmail}
                onChange={(e) => setAuthEmail(e.target.value)}
                placeholder="Email"
                required
              />

              <input
                type={isPasswordVisible ? "text" : "password"}
                value={authPassword}
                onChange={(e) => setAuthPassword(e.target.value)}
                placeholder="Пароль"
                required
              />

              <label className="password-toggle">
                <span>Показать пароль</span>
                <input
                  type="checkbox"
                  checked={isPasswordVisible}
                  onChange={(e) => setIsPasswordVisible(e.target.checked)}
                />
              </label>

              <button type="submit">Войти</button>
            </form>
          </section>
        )}

        {!currentUser && authMode === "register" && (
          <section className="card auth-card">
            <h2>Страница регистрации</h2>

            <form onSubmit={handleAuthSubmit} className="auth-form">
              <input
                value={authName}
                onChange={(e) => setAuthName(e.target.value)}
                placeholder="Имя"
                required
              />

              <input
                value={authEmail}
                onChange={(e) => setAuthEmail(e.target.value)}
                placeholder="Email"
                required
              />

              <input
                type={isPasswordVisible ? "text" : "password"}
                value={authPassword}
                onChange={(e) => setAuthPassword(e.target.value)}
                placeholder="Пароль"
                required
              />

              <input
                type={isPasswordVisible ? "text" : "password"}
                value={authPasswordConfirm}
                onChange={(e) => setAuthPasswordConfirm(e.target.value)}
                placeholder="Подтверждение пароля"
                required
              />

              <label className="password-toggle">
                <span>Показать пароль</span>
                <input
                  type="checkbox"
                  checked={isPasswordVisible}
                  onChange={(e) => setIsPasswordVisible(e.target.checked)}
                />
              </label>

              <button type="submit">Создать аккаунт</button>
            </form>
          </section>
        )}

        {currentUser && activeTab === "home" && (
          <section className="card hero">
            <h1>Система анализа кредитных договоров</h1>
            <p>
              LexigenAI помогает юристам и клиентам анализировать условия кредитных договоров,
              выявлять риски, спорные пункты и формировать рекомендации.
            </p>
            <ul>
              <li>Разбор условий договора и юридических рисков</li>
              <li>Диалоговый помощник по вопросам кредитного права</li>
              <li>История чатов и быстрый доступ к предыдущим обсуждениям</li>
            </ul>
          </section>
        )}

        {currentUser && activeTab === "profile" && (
          <section className="card stub">
            <h2>Личный кабинет</h2>
            <p>Имя: {currentUser.name}</p>
            <p>Email: {currentUser.email}</p>
            <p>Роль: {currentUser.role}</p>

            <button
              type="button"
              className="profile-logout-btn"
              onClick={handleLogout}
            >
              Выйти из аккаунта
            </button>
          </section>
        )}

        {currentUser && currentUser.role === "admin" && activeTab === "analytics" && (
          <section className="card stub">
            <h2>Аналитика</h2>
            <p>Раздел доступен только администратору.</p>
          </section>
        )}

        {currentUser && activeTab === "chats" && (
          <section className={`chat-layout ${isChatSidebarVisible ? "" : "sidebar-hidden"}`}>
            {isChatSidebarVisible && (
              <aside className="chat-sidebar card">
                <h3>Ваши чаты</h3>

                <form onSubmit={onCreateChat} className="new-chat-form">
                  <input
                    value={newChatTitle}
                    onChange={(e) => setNewChatTitle(e.target.value)}
                    placeholder="Название нового чата"
                  />
                  <button type="submit">Создать чат</button>
                </form>

                <div className="chat-list">
                  {chats.map((chat) => (
                    <div
                      key={chat.id}
                      className={`chat-item-row ${chat.id === selectedChatId ? "active" : ""}`}
                    >
                      <button
                        className={`chat-item ${chat.id === selectedChatId ? "active" : ""}`}
                        onClick={() => setSelectedChatId(chat.id)}
                      >
                        {chat.title}
                      </button>

                      <button
                        type="button"
                        className="chat-item-delete"
                        onClick={(e) => {
                          e.stopPropagation();
                          onDeleteChat(chat.id);
                        }}
                        title="Удалить чат"
                        aria-label="Удалить чат"
                      >
                        🗑
                      </button>
                    </div>
                  ))}
                </div>
              </aside>
            )}

            <div className="chat-main card">
              <div className="chat-main-header">
                <h3>{selectedChat ? selectedChat.title : "Выберите чат"}</h3>

                <button
                  type="button"
                  className="toggle-sidebar-btn"
                  onClick={() => setIsChatSidebarVisible((prev) => !prev)}
                >
                  {isChatSidebarVisible ? "Скрыть панель чатов" : "Показать панель чатов"}
                </button>
              </div>

              <div className="messages">
                {messages.length === 0 && (
                  <div className="empty-state">
                    Сообщений пока нет. Начните диалог.
                  </div>
                )}

                {messages.map((msg) => (
                  <div key={msg.id} className={`message ${msg.role}`}>
                    <div className="message-role">
                      {msg.role === "user"
                        ? "Вы"
                        : msg.role === "assistant"
                          ? "Ассистент"
                          : "Система"}
                    </div>

                    <div className="message-content">
                      {msg.content}
                    </div>

                    {msg.analysis_result_id && (
                      <div className="analysis-actions">
                        <button
                          type="button"
                          onClick={() =>
                            onDownloadAnalysis(
                              msg.analysis_result_id
                            )
                          }
                        >
                          Скачать DOCX
                        </button>
                      </div>
                    )}
                  </div>
                ))}

                {isMessageSending && (
                  <div className="message assistant loading-message">
                    <div className="message-role">Ассистент</div>
                    <div className="typing-indicator" aria-label="Ассистент готовит ответ">
                      <span></span>
                      <span></span>
                      <span></span>
                    </div>
                  </div>
                )}
              </div>

              <form onSubmit={onSendMessage} className="message-form">
                <button
                  type="button"
                  className="attach-file-btn"
                  disabled={!selectedChatId || isMessageSending}
                  onClick={() => fileInputRef.current?.click()}
                  title="Добавить DOCX или PDF"
                >
                  +
                </button>

                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".docx,.pdf"
                  className="hidden-file-input"
                  onChange={onFileChange}
                  disabled={!selectedChatId || isMessageSending}
                />

                <input
                  value={messageText}
                  onChange={(e) => setMessageText(e.target.value)}
                  placeholder={
                    selectedFile
                      ? "Можно добавить комментарий к документу..."
                      : "Введите сообщение..."
                  }
                  disabled={!selectedChatId || isMessageSending}
                />

                <button
                  type="submit"
                  disabled={!selectedChatId || isMessageSending}
                >
                  {isMessageSending ? "Обработка..." : "Отправить"}
                </button>
              </form>

              {selectedFile && (
                <div className="selected-file">
                  <span>Файл выбран: {selectedFile.name}</span>

                  <button
                    type="button"
                    onClick={clearSelectedFile}
                  >
                    Убрать
                  </button>
                </div>
              )}
            </div>
          </section>
        )}

        {error && <div className="error-box">{error}</div>}
      </main>
    </div>
  );
}