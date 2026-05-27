import { useEffect, useMemo, useRef, useState } from "react";
import {
  createChat,
  deleteChat,
  downloadAnalysisResult,
  fetchAdminAnalytics,
  fetchChats,
  fetchCurrentSubscription,
  fetchMe,
  fetchMessages,
  fetchSubscriptionPlans,
  login,
  register,
  sendMessage,
  updateChat,
  updateSubscription,
} from "./api";

import AdminAnalyticsPage from "./components/AdminAnalyticsPage";
import AuthPage from "./components/AuthPage";
import ChatPage from "./components/ChatPage";
import HomePage from "./components/HomePage";
import ProfilePage from "./components/ProfilePage";
import Topbar from "./components/Topbar";

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
  const [currentSubscription, setCurrentSubscription] = useState(null);
  const [subscriptionPlans, setSubscriptionPlans] = useState([]);
  const [isSubscriptionUpdating, setIsSubscriptionUpdating] = useState(false);
  const [adminAnalytics, setAdminAnalytics] = useState(null);
  const [isAdminAnalyticsLoading, setIsAdminAnalyticsLoading] = useState(false);

  const [authName, setAuthName] = useState("");
  const [authEmail, setAuthEmail] = useState("");
  const [authPassword, setAuthPassword] = useState("");
  const [authPasswordConfirm, setAuthPasswordConfirm] = useState("");
  const [isPasswordVisible, setIsPasswordVisible] = useState(false);

  const [chats, setChats] = useState([]);
  const [selectedChatId, setSelectedChatId] = useState(null);
  const [editingChatId, setEditingChatId] = useState(null);
  const [editingChatTitle, setEditingChatTitle] = useState("");

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
      setCurrentSubscription(null);
      setSubscriptionPlans([]);
      setAdminAnalytics(null);
      setChats([]);
      setMessages([]);
      setSelectedChatId(null);
      return;
    }

    bootstrapSession();
  }, [authToken]);

  useEffect(() => {
    if (!selectedChatId || !authToken) {
      return;
    }

    loadMessages(selectedChatId, authToken);
  }, [selectedChatId, authToken]);

  async function bootstrapSession() {
    try {
      setError("");

      const user = await fetchMe(authToken);

      setCurrentUser(user);

      await Promise.all([
        loadChats(authToken),
        loadSubscriptionData(authToken),
        user.role === "admin"
          ? loadAdminAnalytics(authToken)
          : Promise.resolve(),
      ]);
    } catch (e) {
      clearSession();
      setError(e.message || "Сессия истекла. Войдите заново.");
    }
  }

  function clearSession() {
    localStorage.removeItem(TOKEN_KEY);
    setAuthToken("");
    setCurrentUser(null);
    setCurrentSubscription(null);
    setSubscriptionPlans([]);
    setAdminAnalytics(null);
    setChats([]);
    setMessages([]);
    setSelectedChatId(null);
    setActiveTab("home");
  }

  function resetAuthFields() {
    setAuthPassword("");
    setAuthPasswordConfirm("");
    setError("");
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
    if (!token) {
      return;
    }

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

  function startEditChat(chat) {
    setEditingChatId(chat.id);
    setEditingChatTitle(chat.title);
  }

  function cancelEditChat() {
    setEditingChatId(null);
    setEditingChatTitle("");
  }

  async function saveChatTitle(chatId) {
    const title = editingChatTitle.trim();

    if (!title || !authToken) {
      return;
    }

    try {
      setError("");

      const updatedChat = await updateChat(
        chatId,
        title,
        authToken
      );

      setChats((prev) =>
        prev.map((chat) =>
          chat.id === chatId ? updatedChat : chat
        )
      );

      cancelEditChat();
    } catch (e) {
      setError(e.message || "Ошибка изменения названия чата");
    }
  }

  async function loadSubscriptionData(token = authToken) {
    if (!token) {
      return;
    }

    try {
      const [plans, subscription] = await Promise.all([
        fetchSubscriptionPlans(token),
        fetchCurrentSubscription(token),
      ]);

      setSubscriptionPlans(plans);
      setCurrentSubscription(subscription);
    } catch (e) {
      setError(e.message || "Ошибка загрузки подписки");
    }
  }

  async function onSelectSubscription(plan) {
    if (!authToken || isSubscriptionUpdating) {
      return;
    }

    try {
      setError("");
      setIsSubscriptionUpdating(true);

      const subscription = await updateSubscription(
        plan,
        authToken
      );

      setCurrentSubscription(subscription);

      if (currentUser?.role === "admin") {
        await loadAdminAnalytics(authToken);
      }
    } catch (e) {
      setError(e.message || "Ошибка изменения подписки");
    } finally {
      setIsSubscriptionUpdating(false);
    }
  }

  async function loadAdminAnalytics(token = authToken) {
    if (!token) {
      return;
    }

    try {
      setIsAdminAnalyticsLoading(true);

      const analytics = await fetchAdminAnalytics(token);

      setAdminAnalytics(analytics);
    } catch (e) {
      setError(e.message || "Ошибка загрузки аналитики");
    } finally {
      setIsAdminAnalyticsLoading(false);
    }
  }

  async function loadMessages(chatId, token = authToken) {
    if (!token) {
      return;
    }

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

    if (!newChatTitle.trim() || !authToken) {
      return;
    }

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
    if (!chatId || !authToken) {
      return;
    }

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
      <Topbar
        currentUser={currentUser}
        navItems={navItems}
        activeTab={activeTab}
        authMode={authMode}
        isLightTheme={isLightTheme}
        onSetActiveTab={setActiveTab}
        onSetAuthMode={setAuthMode}
        onResetAuthFields={resetAuthFields}
        onToggleTheme={toggleTheme}
      />

      <main className={`page ${activeTab === "chats" ? "chats-page" : ""}`}>
        {!currentUser && (
          <AuthPage
            authMode={authMode}
            authName={authName}
            authEmail={authEmail}
            authPassword={authPassword}
            authPasswordConfirm={authPasswordConfirm}
            isPasswordVisible={isPasswordVisible}
            onSetAuthName={setAuthName}
            onSetAuthEmail={setAuthEmail}
            onSetAuthPassword={setAuthPassword}
            onSetAuthPasswordConfirm={setAuthPasswordConfirm}
            onSetIsPasswordVisible={setIsPasswordVisible}
            onSubmit={handleAuthSubmit}
          />
        )}

        {currentUser && activeTab === "home" && (
          <HomePage
            onOpenChats={() => setActiveTab("chats")}
            onOpenProfile={() => setActiveTab("profile")}
          />
        )}

        {currentUser && activeTab === "profile" && (
          <ProfilePage
            currentUser={currentUser}
            currentSubscription={currentSubscription}
            subscriptionPlans={subscriptionPlans}
            isSubscriptionUpdating={isSubscriptionUpdating}
            onSelectSubscription={onSelectSubscription}
            onLogout={handleLogout}
          />
        )}

        {currentUser && currentUser.role === "admin" && activeTab === "analytics" && (
          <AdminAnalyticsPage
            adminAnalytics={adminAnalytics}
            isAdminAnalyticsLoading={isAdminAnalyticsLoading}
            onRefresh={() => loadAdminAnalytics(authToken)}
          />
        )}

        {currentUser && activeTab === "chats" && (
          <ChatPage
            fileInputRef={fileInputRef}
            chats={chats}
            selectedChat={selectedChat}
            selectedChatId={selectedChatId}
            messages={messages}
            newChatTitle={newChatTitle}
            messageText={messageText}
            selectedFile={selectedFile}
            isMessageSending={isMessageSending}
            isChatSidebarVisible={isChatSidebarVisible}
            editingChatId={editingChatId}
            editingChatTitle={editingChatTitle}
            onSetIsChatSidebarVisible={setIsChatSidebarVisible}
            onSetSelectedChatId={setSelectedChatId}
            onSetNewChatTitle={setNewChatTitle}
            onSetMessageText={setMessageText}
            onSetEditingChatTitle={setEditingChatTitle}
            onCreateChat={onCreateChat}
            onStartEditChat={startEditChat}
            onCancelEditChat={cancelEditChat}
            onSaveChatTitle={saveChatTitle}
            onDeleteChat={onDeleteChat}
            onFileChange={onFileChange}
            onSendMessage={onSendMessage}
            onClearSelectedFile={clearSelectedFile}
            onDownloadAnalysis={onDownloadAnalysis}
          />
        )}

        {error && <div className="error-box">{error}</div>}
      </main>
    </div>
  );
}
