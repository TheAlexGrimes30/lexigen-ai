export default function ChatSidebar({
  chats,
  selectedChatId,
  newChatTitle,
  editingChatId,
  editingChatTitle,
  onSetSelectedChatId,
  onSetNewChatTitle,
  onCreateChat,
  onStartEditChat,
  onCancelEditChat,
  onSetEditingChatTitle,
  onSaveChatTitle,
  onDeleteChat,
}) {
  return (
    <aside className="chat-sidebar card">
      <h3>Ваши чаты</h3>

      <form onSubmit={onCreateChat} className="new-chat-form">
        <input
          value={newChatTitle}
          onChange={(e) => onSetNewChatTitle(e.target.value)}
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
            {editingChatId === chat.id ? (
              <form
                className="chat-edit-form"
                onSubmit={(e) => {
                  e.preventDefault();
                  onSaveChatTitle(chat.id);
                }}
              >
                <input
                  value={editingChatTitle}
                  onChange={(e) => onSetEditingChatTitle(e.target.value)}
                  autoFocus
                  onKeyDown={(e) => {
                    if (e.key === "Escape") {
                      onCancelEditChat();
                    }
                  }}
                />

                <button
                  type="submit"
                  className="chat-edit-save"
                  title="Сохранить"
                >
                  ✓
                </button>

                <button
                  type="button"
                  className="chat-edit-cancel"
                  onClick={onCancelEditChat}
                  title="Отменить"
                >
                  ×
                </button>
              </form>
            ) : (
              <>
                <button
                  className={`chat-item ${chat.id === selectedChatId ? "active" : ""}`}
                  onClick={() => onSetSelectedChatId(chat.id)}
                >
                  {chat.title}
                </button>

                <button
                  type="button"
                  className="chat-item-edit"
                  onClick={(e) => {
                    e.stopPropagation();
                    onStartEditChat(chat);
                  }}
                  title="Редактировать чат"
                  aria-label="Редактировать чат"
                >
                  ✏️
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
              </>
            )}
          </div>
        ))}
      </div>
    </aside>
  );
}
