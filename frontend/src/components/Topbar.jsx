export default function Topbar({
  currentUser,
  navItems,
  activeTab,
  authMode,
  isLightTheme,
  onSetActiveTab,
  onSetAuthMode,
  onResetAuthFields,
  onToggleTheme,
}) {
  const guestItems = [
    { key: "login", label: "Вход" },
    { key: "register", label: "Регистрация" },
  ];

  return (
    <header className="topbar">
      <div className="logo">LexigenAI</div>

      <div className="topbar-right">
        <nav className="nav">
          {currentUser
            ? navItems.map((item) => (
                <button
                  key={item.key}
                  className={`nav-btn ${activeTab === item.key ? "active" : ""}`}
                  onClick={() => onSetActiveTab(item.key)}
                >
                  {item.label}
                </button>
              ))
            : guestItems.map((item) => (
                <button
                  key={item.key}
                  className={`nav-btn ${authMode === item.key ? "active" : ""}`}
                  onClick={() => {
                    onSetAuthMode(item.key);
                    onSetActiveTab(item.key);
                    onResetAuthFields();
                  }}
                >
                  {item.label}
                </button>
              ))}
        </nav>

        <button
          type="button"
          className="theme-toggle-btn"
          onClick={onToggleTheme}
        >
          {isLightTheme ? "Тёмная тема" : "Светлая тема"}
        </button>
      </div>
    </header>
  );
}
