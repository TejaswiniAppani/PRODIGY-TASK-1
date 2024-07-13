import tensorflow as tf
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
SEED = 34
MAX_LEN = 70
input_sequence = "I don't know about you, but there's only one thing I want to do after a long day of work"
tf.random.set_seed(SEED)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
model = TFGPT2LMHeadModel.from_pretrained("gpt2-large", pad_token_id=tokenizer.eos_token_id)
input_ids = tokenizer.encode(input_sequence, return_tensors='tf')
greedy_output = model.generate(input_ids, max_length=MAX_LEN)
print("Greedy Output:\n" + 100 * '-')
print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))
beam_outputs = model.generate(
    input_ids,
    max_length=MAX_LEN,
    num_beams=5,
    no_repeat_ngram_size=2,
    num_return_sequences=5,
    early_stopping=True
)
print("Beam Search Outputs:\n" + 100 * '-')
for i, beam_output in enumerate(beam_outputs):
    print(f"{i}: {tokenizer.decode(beam_output, skip_special_tokens=True)}")
sample_outputs = model.generate(
    input_ids,
    do_sample=True,
    max_length=MAX_LEN,
    top_k=50,
    top_p=0.85
)
print("Sampling Output:\n" + 100 * '-')
print(tokenizer.decode(sample_outputs[0], skip_special_tokens=True))
prompts = [
    'Miley Cyrus was caught shoplifting from Abercrombie and Fitch on Hollywood Boulevard today.',
    'Legolas and Gimli advanced on the orcs, raising their weapons with a harrowing war cry.'
]
for prompt in prompts:
    input_ids = tokenizer.encode(prompt, return_tensors='tf')
    sample_outputs = model.generate(
        input_ids,
        do_sample=True,
        max_length=MAX_LEN,
        top_k=50,
        top_p=0.85
    )

    print(f"\nPrompt: {prompt}\n" + 100 * '-')
    print(tokenizer.decode(sample_outputs[0], skip_special_tokens=True))
