import os
import time
import csv
import json
import pandas as pd
from tqdm import tqdm
import openai

# -------------------------------
# CONFIGURATION
NUM_SAMPLES = 300
INPUT_FILES = [
    "pubmed_gpt4_raw_data.csv",
    "writing_gpt4_raw_data.csv",
    "xsum_gpt4_raw_data.csv",
    "pubmed_gemini.raw_data.csv",
    "writing_gemini.raw_data.csv",
    "xsum_gemini.raw_data.csv",
    "pubmed_turbo_3.5_.raw_data.csv",
    "writing_turbo_3.5_.raw_data.csv",
    "xsum_turbo_3.5_.raw_data.csv",
]
TEMPERATURES = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1]
MODEL_NAME = "gpt-4.1-mini"
OUTPUT_DIR = "output"
API_KEY = "OPEN-AI Key"
WAIT_SECONDS = 3
# -------------------------------

COST_PER_INPUT_1K = 0.01
COST_PER_OUTPUT_1K = 0.03

client = openai.OpenAI(api_key=API_KEY)

function_schema = {
    "name": "generate_variants",
    "description": "Return exactly 10 paraphrased versions of the input text.",
    "parameters": {
        "type": "object",
        "properties": {
            "variants": {
                "type": "array",
                "items": {"type": "string"},
                "description": "10 high-quality rewordings"
            }
        },
        "required": ["variants"]
    }
}

class GPT4Paraphraser:
    def __init__(self, temperature, writer, file_handle):
        self.temperature = temperature
        self.writer = writer
        self.file_handle = file_handle
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def build_messages(self, text: str):
        return [
            {
                "role": "user",
                "content": (
                    "You are a JSON generator. Rephrase the input text into exactly 10 high-quality, fluent, "
                    "semantically consistent paraphrases. Return ONLY this format:\n\n"
                    '{"variants": [\n'
                    '{"text": "..."},\n'
                    '{"text": "..."},\n'
                    '... 10 items total\n]}'
                    f'\n\nInput:\n"""{text}"""'
                )
            }
        ]

    def generate_and_stream_variations(self, texts, tags):
        for i, (text, tag) in enumerate(tqdm(zip(texts, tags), total=len(texts), desc=f"Generating (T={self.temperature})", unit="row")):
            if not text.strip():
                self.writer.writerow([text, tag] + [""] * 10)
                continue

            for attempt in range(2):  # 1 retry allowed
                try:
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        temperature=self.temperature,
                        messages=self.build_messages(text),
                        functions=[function_schema],
                        function_call={"name": "generate_variants"}
                    )

                    usage = response.usage
                    self.total_prompt_tokens += usage.prompt_tokens
                    self.total_completion_tokens += usage.completion_tokens

                    args = response.choices[0].message.function_call.arguments
                    data = json.loads(args)
                    variants = [v.strip() for v in data.get("variants", [])]

                    if len(variants) == 10 and all(variants):
                        self.writer.writerow([text, tag] + variants)
                        break
                except Exception as e:
                    print(f"⚠️ Row {i+1} attempt {attempt+1} failed: {e}")
                    time.sleep(1)
            else:
                self.writer.writerow([text, tag] + ["FAILED_VARIANT"] * 10)

            self.file_handle.flush()
            time.sleep(WAIT_SECONDS)

    def report_costs(self):
        input_cost = (self.total_prompt_tokens / 1000) * COST_PER_INPUT_1K
        output_cost = (self.total_completion_tokens / 1000) * COST_PER_OUTPUT_1K
        total_cost = input_cost + output_cost
        print("\n===== Token Usage Report =====")
        print(f"Prompt Tokens    : {self.total_prompt_tokens}")
        print(f"Completion Tokens: {self.total_completion_tokens}")
        print(f"Estimated Cost   : ${total_cost:.4f}")
        print("================================")

def main():
    for file in INPUT_FILES:
        df = pd.read_csv(file)
        df = df.dropna()
        df = df[df["label"].isin(["human_answers", "chatgpt_answers"])]
        df = df.sample(n=min(NUM_SAMPLES, len(df)), random_state=42)

        originals = df["text"].tolist()
        tags = ["human" if lbl == "human_answers" else "ai" for lbl in df["label"].tolist()]

        for temp in TEMPERATURES:
            base = os.path.splitext(file)[0]
            out_path = os.path.join(OUTPUT_DIR, f"{base}_variants_gpt4_temp{temp}.csv")
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

            with open(out_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                header = ["original", "tag"] + [f"variant_{i+1}" for i in range(10)]
                writer.writerow(header)

                gen = GPT4Paraphraser(temperature=temp, writer=writer, file_handle=f)
                gen.generate_and_stream_variations(originals, tags)
                gen.report_costs()

if __name__ == "__main__":
    main()
