export default function MessageForm({
  fileInputRef,
  selectedChatId,
  selectedFile,
  messageText,
  isMessageSending,
  onSetMessageText,
  onFileChange,
  onSendMessage,
  onClearSelectedFile,
}) {
  return (
    <>
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
          onChange={(e) => onSetMessageText(e.target.value)}
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
            onClick={onClearSelectedFile}
          >
            Убрать
          </button>
        </div>
      )}
    </>
  );
}
