import openai

# OpenAI APIキーを設定
openai.api_key = 'APIkey' #自身のAPIキーを埋め込むこと!(従量課金制)

def main():
    print("生成AIチャットへようこそ。終了するには 'exit' と入力してください。")
    while True:
        user_input = input("あなた: ")
        if user_input.lower() == 'exit':
            print("チャットを終了します。")
            break

        # OpenAI APIへのリクエスト
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "あなたは親切で知識豊富なアシスタントです。"},
                    {"role": "user", "content": user_input}
                ]
            )
            ai_reply = response.choices[0].message.content
            print(f"AI: {ai_reply}\n")
        except Exception as e:
            print(f"エラーが発生しました: {e}")

if __name__ == "__main__":
    main()
