export default function MessageList({
  messages,
  isMessageSending,
  onDownloadAnalysis,
}) {
  return (
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
                onClick={() => onDownloadAnalysis(msg.analysis_result_id)}
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
  );
}
