export default function AuthPage({
  authMode,
  authName,
  authEmail,
  authPassword,
  authPasswordConfirm,
  isPasswordVisible,
  onSetAuthName,
  onSetAuthEmail,
  onSetAuthPassword,
  onSetAuthPasswordConfirm,
  onSetIsPasswordVisible,
  onSubmit,
}) {
  const isRegister = authMode === "register";

  return (
    <section className="card auth-card">
      <h2>{isRegister ? "Страница регистрации" : "Страница входа"}</h2>

      <form onSubmit={onSubmit} className="auth-form">
        {isRegister && (
          <input
            value={authName}
            onChange={(e) => onSetAuthName(e.target.value)}
            placeholder="Имя"
            required
          />
        )}

        <input
          value={authEmail}
          onChange={(e) => onSetAuthEmail(e.target.value)}
          placeholder="Email"
          required
        />

        <input
          type={isPasswordVisible ? "text" : "password"}
          value={authPassword}
          onChange={(e) => onSetAuthPassword(e.target.value)}
          placeholder="Пароль"
          required
        />

        {isRegister && (
          <input
            type={isPasswordVisible ? "text" : "password"}
            value={authPasswordConfirm}
            onChange={(e) => onSetAuthPasswordConfirm(e.target.value)}
            placeholder="Подтверждение пароля"
            required
          />
        )}

        <label className="password-toggle">
          <span>Показать пароль</span>
          <input
            type="checkbox"
            checked={isPasswordVisible}
            onChange={(e) => onSetIsPasswordVisible(e.target.checked)}
          />
        </label>

        <button type="submit">
          {isRegister ? "Создать аккаунт" : "Войти"}
        </button>
      </form>
    </section>
  );
}
