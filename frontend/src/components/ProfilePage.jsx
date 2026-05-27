export default function ProfilePage({
  currentUser,
  currentSubscription,
  subscriptionPlans,
  isSubscriptionUpdating,
  onSelectSubscription,
  onLogout,
}) {
  return (
    <section className="card stub">
      <h2>Личный кабинет</h2>

      <div className="profile-info">
        <p>Имя: {currentUser.name}</p>
        <p>Email: {currentUser.email}</p>
        <p>Роль: {currentUser.role}</p>
        <p>
          Текущая подписка:{" "}
          <strong>
            {currentSubscription?.title || "Без подписки"}
          </strong>
        </p>
      </div>

      <div className="subscription-section">
        <h3>Подписки</h3>

        <p className="subscription-note">
          Без подписки пользователь может загрузить документ только один раз.
          Пользователи с подпиской могут анализировать документы без лимита.
          Администратор может анализировать документы без подписки.
        </p>

        <div className="subscription-grid">
          {subscriptionPlans.map((plan) => (
            <div
              key={plan.plan}
              className={`subscription-card ${
                currentSubscription?.plan === plan.plan ? "active" : ""
              }`}
            >
              <h4>{plan.title}</h4>

              <div className="subscription-price">
                {plan.price_rub.toLocaleString("ru-RU")} ₽
              </div>

              <p>{plan.description}</p>

              <button
                type="button"
                disabled={
                  isSubscriptionUpdating ||
                  currentSubscription?.plan === plan.plan
                }
                onClick={() => onSelectSubscription(plan.plan)}
              >
                {currentSubscription?.plan === plan.plan ? "Активна" : "Выбрать"}
              </button>
            </div>
          ))}
        </div>
      </div>

      <button
        type="button"
        className="profile-logout-btn"
        onClick={onLogout}
      >
        Выйти из аккаунта
      </button>
    </section>
  );
}
