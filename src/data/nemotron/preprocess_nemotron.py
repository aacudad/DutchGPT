import json
import re

def preprocess_and_format(jsonl_path, output_path):
    with open(jsonl_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            try:
                data = json.loads(line)
                text = data.get("input", "") + data.get("output", "")

                # Replace <think> with <denken>
                text = text.replace("<think>", "<denken>").replace("</think>", "</denken>")

                # Extract user-assistant turns
                pattern = r"<\|start_header_id\|>user<\|end_header_id\|>\n\n(.*?)<\|eot_id\|><\|start_header_id\|>assistant<\|end_header_id\|>\n\n(.*?)<\|eot_id\|>"
                matches = re.findall(pattern, text, re.DOTALL)

                for user_msg, assistant_msg in matches:
                    formatted = (
                        "<bos><bos><start_of_turn>user\n"
                        f"{user_msg.strip()}\n"
                        "<end_of_turn>\n"
                        "<start_of_turn>model\n"
                        f"{assistant_msg.strip()}\n"
                        "<end_of_turn>\n"
                    )
                    json.dump({"text": formatted}, outfile, ensure_ascii=False)
                    outfile.write('\n')

            except json.JSONDecodeError:
                continue  # Skip malformed lines

# Run the function
preprocess_and_format("chat.jsonl", "formatted_chat.jsonl")
