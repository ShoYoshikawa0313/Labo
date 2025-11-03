from openai import OpenAI

client = OpenAI(
    base_url = 'http://10.34.35.193:11434/v1',
    api_key='ollama', # required, but unused
)

# --- プロンプトの準備 ---
# 1. instructionとinputを定義
instruction = "Increase the hERG inhibition value of the following molecule while ensuring it remains similar to the original molecule."
input_text = "CC1CN(S(=O)(=O)c2cn(C)nc2C(F)(F)F)C(c2ccccc2)CO1"

# 2. instructionとinputを改行で結合
full_instruction = instruction + '\n' + input_text

# 3. モデル指定のフォーマットで最終的なプロンプトを作成
final_prompt = f"[INST] {full_instruction} [/INST]"


# --- API呼び出し ---
# client.chat.completions.createに渡すパラメータ
params = {
    "model": "drugassist-instruct",
    "messages": [
        {
            "role": "user",
            "content": final_prompt
        }
    ],
    "stream": False,
}

import time

# APIを呼び出して結果を表示
start = time.time()
response = client.chat.completions.create(**params).choices[0].message.content
end = time.time()
print(response)
print(end - start)