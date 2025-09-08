from langchain_community.chat_message_histories import ChatMessageHistory

_histories = {}

def get_history(session_id: str) -> ChatMessageHistory:
    if session_id not in _histories:
        _histories[session_id] = ChatMessageHistory()
    return _histories[session_id]