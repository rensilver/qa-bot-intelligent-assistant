from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

from qa_bot_intelligent_assistant.main import respond, to_chat_history

def test_to_chat_history_converts_user_and_assistant_turns():
    history = [
        {"role": "user", "content": "What is X?"},
        {"role": "assistant", "content": "X is ..."},
    ]

    result = to_chat_history(history)

    assert result == [HumanMessage("What is X?"), AIMessage("X is ...")]

def test_to_chat_history_with_no_turns_returns_empty_list():
    assert to_chat_history([]) == []

def test_respond_passes_question_and_converted_history_to_chain():
    history = [
        {"role": "user", "content": "What is X?"},
        {"role": "assistant", "content": "X is ..."},
    ]

    with patch("qa_bot_intelligent_assistant.main.rag_chain") as mock_chain:
        mock_chain.invoke.return_value = "final answer"

        message_box, new_history = respond("what about page 2?", history)

    mock_chain.invoke.assert_called_once_with(
        {
            "question": "what about page 2?",
            "chat_history": [HumanMessage("What is X?"), AIMessage("X is ...")],
        }
    )
    assert message_box == ""
    assert new_history == history + [
        {"role": "user", "content": "what about page 2?"},
        {"role": "assistant", "content": "final answer"},
    ]
