import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    print("生成AIチャットへようこそ。終了するには 'exit' と入力してください。")

    # モデルとトークナイザーの読み込み
    model_name = "gpt2"  # または "EleutherAI/gpt-neo-125M" など
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # モデルをデバイスに移動（GPUがある場合はGPUを使用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # チャット履歴を保持
    chat_history = ""

    while True:
        user_input = input("あなた: ")
        if user_input.lower() == 'exit':
            print("チャットを終了します。")
            break

        chat_history += f"\nユーザー: {user_input}\nAI:"

        # 入力のトークン化
        inputs = tokenizer.encode(chat_history, return_tensors='pt').to(device)

        # モデルによるテキスト生成
        outputs = model.generate(
            inputs,
            max_length=inputs.shape[1] + 50,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95
        )

        # 応答のデコード
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 最新の応答を取得
        ai_reply = output_text[len(chat_history):].strip().split('\n')[0]
        print(f"AI: {ai_reply}")

        chat_history += f" {ai_reply}"

if __name__ == "__main__":
    main()
