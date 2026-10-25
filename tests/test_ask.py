from types import SimpleNamespace

from app import ask


def test_follow_up_questions_share_the_chat_history(monkeypatch, capsys):
    class FakeChat:
        def __init__(self):
            self.questions = []

        def send_message(self, question):
            self.questions.append(question)
            return type(
                "Response",
                (),
                {
                    "text": f"回答: {question}",
                    "model_version": "gemini-routed-model",
                    "usage_metadata": SimpleNamespace(
                        prompt_token_count=10,
                        candidates_token_count=20,
                    ),
                },
            )()

    class FakeClient:
        def __init__(self, api_key):
            assert api_key == "test-key"

            class FakeChats:
                def create(self, model):
                    assert model == ask.MODEL
                    return chat

            self.chats = FakeChats()

    chat = FakeChat()
    monkeypatch.setattr(ask, "get_api_key", lambda: "test-key")
    monkeypatch.setattr(ask.genai, "Client", FakeClient)
    monkeypatch.setattr("sys.argv", ["app.ask", "最初の質問"])
    answers = iter(["続けて質問", ""])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))

    ask.main()

    assert chat.questions == ["最初の質問", "続けて質問"]
    output = capsys.readouterr().out
    assert "回答: 最初の質問" in output
    assert "回答: 続けて質問" in output
    assert output.count("gemini-routed-model 10/20 ") == 2
    assert "秒" in output
    assert output.index("[質問] 最初の質問") < output.index("[回答] 回答: 最初の質問")
    assert output.index("[回答] 回答: 最初の質問") < output.index("gemini-routed-model")
