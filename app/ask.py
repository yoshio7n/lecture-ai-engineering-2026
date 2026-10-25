"""第1回: 素のLLMに会話形式で質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 既定の質問から開始する

初回の回答後、空行を入力するまで同じ会話の履歴を保って質問できる。
ツールは持たない「素の」呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse
import time

from google import genai
from google.genai.chats import Chat
from google.genai import errors

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def ask(chat: Chat, question: str) -> tuple[str, int | None, int | None, float]:
    started_at = time.perf_counter()
    response = chat.send_message(question)
    elapsed_seconds = time.perf_counter() - started_at
    usage = response.usage_metadata
    if usage is None:
        return response.text, None, None, elapsed_seconds
    return (
        response.text,
        usage.prompt_token_count,
        usage.candidates_token_count,
        elapsed_seconds,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    args = parser.parse_args()

    try:
        client = genai.Client(api_key=get_api_key())
        chat = client.chats.create(model=MODEL)
        question = args.question
        while question:
            print(f"[質問] {question}")
            answer, input_tokens, output_tokens, elapsed_seconds = ask(chat, question)
            print(f"\n[回答] {answer}")
            print(f"[model] {MODEL}")
            input_count = input_tokens if input_tokens is not None else "不明"
            output_count = output_tokens if output_tokens is not None else "不明"
            print(f"[トークン数] 入力: {input_count} / 出力: {output_count}")
            print(f"[応答時間] {elapsed_seconds:.2f} 秒")
            question = input("\n続けて質問してください（終了するには空行）: ").strip()
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)
    except EOFError:
        pass


if __name__ == "__main__":
    main()
