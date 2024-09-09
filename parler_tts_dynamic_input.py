import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
import os

# Set the device for model inference
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load the pre-trained Parler-TTS model and tokenizer
model = ParlerTTSForConditionalGeneration.from_pretrained("./models/parler-tts-large-v1").to(device)
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-large-v1")

# Read the prompt from input.txt
with open("input.txt", "r") as file:
    prompt = file.read().strip()

# Define the description with a focus on high-quality and accurate audio
description = "Jon's voice is monotone yet slightly fast in delivery, with very clear audio."

# Set a maximum sequence length
max_length = 512

# Tokenize the description and prompt text, applying padding and truncation
input_tokens = tokenizer(description, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length).to(device)
prompt_tokens = tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length).to(device)

# Generate the speech, explicitly setting the attention masks
generation = model.generate(
    input_ids=input_tokens.input_ids,
    attention_mask=input_tokens.attention_mask,
    prompt_input_ids=prompt_tokens.input_ids,
    prompt_attention_mask=prompt_tokens.attention_mask
)

# Convert the generated output to a numpy array and remove unnecessary dimensions
audio_arr = generation.cpu().numpy().squeeze()

# Ensure the output directory exists
output_file = "output/parler_tts_out.wav"
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Save the generated audio to a file
sf.write(output_file, audio_arr, model.config.sampling_rate)
print(f"Speech generated and saved to {output_file}")
