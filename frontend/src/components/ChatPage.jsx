import ChatSidebar from "./ChatSidebar";
import MessageForm from "./MessageForm";
import MessageList from "./MessageList";

export default function ChatPage({
  fileInputRef,
  chats,
  selectedChat,
  selectedChatId,
  messages,
  newChatTitle,
  messageText,
  selectedFile,
  isMessageSending,
  isChatSidebarVisible,
  editingChatId,
  editingChatTitle,
  onSetIsChatSidebarVisible,
  onSetSelectedChatId,
  onSetNewChatTitle,
  onSetMessageText,
  onSetEditingChatTitle,
  onCreateChat,
  onStartEditChat,
  onCancelEditChat,
  onSaveChatTitle,
  onDeleteChat,
  onFileChange,
  onSendMessage,
  onClearSelectedFile,
  onDownloadAnalysis,
}) {
  return (
    <section className={`chat-layout ${isChatSidebarVisible ? "" : "sidebar-hidden"}`}>
      {isChatSidebarVisible && (
        <ChatSidebar
          chats={chats}
          selectedChatId={selectedChatId}
          newChatTitle={newChatTitle}
          editingChatId={editingChatId}
          editingChatTitle={editingChatTitle}
          onSetSelectedChatId={onSetSelectedChatId}
          onSetNewChatTitle={onSetNewChatTitle}
          onCreateChat={onCreateChat}
          onStartEditChat={onStartEditChat}
          onCancelEditChat={onCancelEditChat}
          onSetEditingChatTitle={onSetEditingChatTitle}
          onSaveChatTitle={onSaveChatTitle}
          onDeleteChat={onDeleteChat}
        />
      )}

      <div className="chat-main card">
        <div className="chat-main-header">
          <h3>{selectedChat ? selectedChat.title : "Выберите чат"}</h3>

          <button
            type="button"
            className="toggle-sidebar-btn"
            onClick={() => onSetIsChatSidebarVisible((prev) => !prev)}
          >
            {isChatSidebarVisible ? "Скрыть панель чатов" : "Показать панель чатов"}
          </button>
        </div>

        <MessageList
          messages={messages}
          isMessageSending={isMessageSending}
          onDownloadAnalysis={onDownloadAnalysis}
        />

        <MessageForm
          fileInputRef={fileInputRef}
          selectedChatId={selectedChatId}
          selectedFile={selectedFile}
          messageText={messageText}
          isMessageSending={isMessageSending}
          onSetMessageText={onSetMessageText}
          onFileChange={onFileChange}
          onSendMessage={onSendMessage}
          onClearSelectedFile={onClearSelectedFile}
        />
      </div>
    </section>
  );
}
