export default function AdminAnalyticsPage({
  adminAnalytics,
  isAdminAnalyticsLoading,
  onRefresh,
}) {
  return (
    <section className="card stub">
      <div className="admin-analytics-header">
        <h2>Аналитика</h2>

        <button
          type="button"
          onClick={onRefresh}
          disabled={isAdminAnalyticsLoading}
        >
          {isAdminAnalyticsLoading ? "Обновление..." : "Обновить"}
        </button>
      </div>

      <div className="analytics-grid">
        <div className="analytics-card">
          <span>Всего пользователей</span>
          <strong>{adminAnalytics?.total_users ?? 0}</strong>
        </div>

        <div className="analytics-card">
          <span>Без подписки</span>
          <strong>{adminAnalytics?.without_subscription ?? 0}</strong>
        </div>

        <div className="analytics-card">
          <span>Basic</span>
          <strong>{adminAnalytics?.by_plan?.basic ?? 0}</strong>
        </div>

        <div className="analytics-card">
          <span>Pro</span>
          <strong>{adminAnalytics?.by_plan?.pro ?? 0}</strong>
        </div>

        <div className="analytics-card">
          <span>Enterprise</span>
          <strong>{adminAnalytics?.by_plan?.enterprise ?? 0}</strong>
        </div>
      </div>
    </section>
  );
}
