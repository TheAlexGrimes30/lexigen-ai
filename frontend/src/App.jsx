import { useEffect, useMemo, useState } from "react";
import { createChat, fetchChats, fetchMessages, sendMessage } from "./api";

const NAV_ITEMS = [
  { key: "home", label: "Главная" },
  { key: "profile", label: "Личный кабинет" },
  { key: "analytics", label: "Аналитика" },
  { key: "chats", label: "Ваши чаты" },
];

export default function App() {
  const [activeTab, setActiveTab] = useState("home");
  const [chats, setChats] = useState([]);
  const [selectedChatId, setSelectedChatId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [newChatTitle, setNewChatTitle] = useState("");
  const [messageText, setMessageText] = useState("");
  const [error, setError] = useState("");
  const [isChatSidebarVisible, setIsChatSidebarVisible] = useState(true);

  const selectedChat = useMemo(
    () => chats.find((chat) => chat.id === selectedChatId) || null,
    [chats, selectedChatId]
  );

  useEffect(() => {
    loadChats();
  }, []);

  useEffect(() => {
    if (!selectedChatId) return;
    loadMessages(selectedChatId);
  }, [selectedChatId]);

  async function loadChats() {
    try {
      setError("");
      const data = await fetchChats();
      setChats(data);
      if (!selectedChatId && data.length > 0) {
        setSelectedChatId(data[0].id);
      }
    } catch (e) {
      setError(e.message || "Ошибка загрузки чатов");
    }
  }

  async function loadMessages(chatId) {
    try {
      setError("");
      const data = await fetchMessages(chatId);
      setMessages(data);
    } catch (e) {
      setError(e.message || "Ошибка загрузки сообщений");
      setMessages([]);
    }
  }

  async function onCreateChat(e) {
    e.preventDefault();
    if (!newChatTitle.trim()) return;

    try {
      setError("");
      const chat = await createChat(newChatTitle.trim());
      const updated = [chat, ...chats];
      setChats(updated);
      setSelectedChatId(chat.id);
      setNewChatTitle("");
      setActiveTab("chats");
    } catch (e) {
      setError(e.message || "Ошибка создания чата");
    }
  }

  async function onSendMessage(e) {
    e.preventDefault();
    if (!selectedChatId || !messageText.trim()) return;

    try {
      setError("");
      const newMessages = await sendMessage(selectedChatId, messageText.trim());
      setMessages((prev) => [...prev, ...newMessages]);
      setMessageText("");
    } catch (e) {
      setError(e.message || "Ошибка отправки сообщения");
    }
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="logo">LexigenAI</div>
        <nav className="nav">
          {NAV_ITEMS.map((item) => (
            <button
              key={item.key}
              className={`nav-btn ${activeTab === item.key ? "active" : ""}`}
              onClick={() => setActiveTab(item.key)}
            >
              {item.label}
            </button>
          ))}
        </nav>
      </header>

      <main className={`page ${activeTab === "chats" ? "chats-page" : ""}`}>
        {activeTab === "home" && (
          <section className="card hero">
            <h1>Система анализа кредитных договоров</h1>
            <p>
              LexigenAI помогает юристам и клиентам анализировать условия кредитных
              договоров, выявлять риски, спорные пункты и формировать рекомендации.
            </p>
            <ul>
              <li>Разбор условий договора и юридических рисков</li>
              <li>Диалоговый помощник по вопросам кредитного права</li>
              <li>История чатов и быстрый доступ к предыдущим обсуждениям</li>
            </ul>
          </section>
        )}

        {activeTab === "profile" && (
          <section className="card stub">
            <h2>Личный кабинет</h2>
            <p>Раздел для профиля пользователя и настроек доступа.</p>
          </section>
        )}

        {activeTab === "analytics" && (
          <section className="card stub">
            <h2>Аналитика</h2>
            <p>Здесь будет статистика по чатам, рискам и типам договорных нарушений.</p>
          </section>
        )}

        {activeTab === "chats" && (
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
                    <button
                      key={chat.id}
                      className={`chat-item ${chat.id === selectedChatId ? "active" : ""}`}
                      onClick={() => setSelectedChatId(chat.id)}
                    >
                      {chat.title}
                    </button>
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
                  <div className="empty-state">Сообщений пока нет. Начните диалог.</div>
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
                    <div>{msg.content}</div>
                  </div>
                ))}
              </div>

              <form onSubmit={onSendMessage} className="message-form">
                <input
                  value={messageText}
                  onChange={(e) => setMessageText(e.target.value)}
                  placeholder="Введите сообщение..."
                  disabled={!selectedChatId}
                />
                <button type="submit" disabled={!selectedChatId}>
                  Отправить
                </button>
              </form>

              {error && <div className="error-box">{error}</div>}
            </div>
          </section>
        )}
      </main>
    </div>
  );
}
