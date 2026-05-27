export default function HomePage({ onOpenChats, onOpenProfile }) {
  return (
    <section className="home-page">
      <div className="card hero hero-modern">
        <div className="hero-content">
          <span className="hero-badge">LexigenAI · Credit Law Assistant</span>

          <h1>Анализ кредитных договоров с RAG по базе знаний</h1>

          <p>
            Загружайте DOCX/PDF договоры, получайте юридический анализ рисков,
            слабых условий и рекомендаций, а затем скачивайте результат в DOCX.
          </p>

          <div className="hero-actions">
            <button
              type="button"
              onClick={onOpenChats}
            >
              Начать анализ
            </button>

            <button
              type="button"
              className="secondary-action"
              onClick={onOpenProfile}
            >
              Посмотреть подписку
            </button>
          </div>
        </div>

        <div className="hero-panel">
          <div className="hero-panel-item">
            <strong>RAG</strong>
            <span>по нормам кредитного права</span>
          </div>

          <div className="hero-panel-item">
            <strong>DOCX/PDF</strong>
            <span>загрузка документов в чат</span>
          </div>

          <div className="hero-panel-item">
            <strong>DOCX</strong>
            <span>скачивание результата анализа</span>
          </div>
        </div>
      </div>

      <div className="home-features">
        <div className="card feature-card">
          <div className="feature-icon">⚖️</div>
          <h3>Юридические риски</h3>
          <p>
            Ассистент выделяет спорные условия, пробелы договора и потенциальные риски.
          </p>
        </div>

        <div className="card feature-card">
          <div className="feature-icon">📄</div>
          <h3>Документы в чате</h3>
          <p>
            Пользователь может прикрепить договор, а система автоматически отправит текст
            в RAG-анализ.
          </p>
        </div>

        <div className="card feature-card">
          <div className="feature-icon">💼</div>
          <h3>Подписки</h3>
          <p>
            Basic, Pro и Enterprise снимают ограничение на количество анализируемых документов.
          </p>
        </div>
      </div>
    </section>
  );
}
