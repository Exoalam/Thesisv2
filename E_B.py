from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from normalizer import normalize 

model = AutoModelForSeq2SeqLM.from_pretrained("csebuetnlp/banglat5_nmt_en_bn")
tokenizer = AutoTokenizer.from_pretrained("csebuetnlp/banglat5_nmt_en_bn", use_fast=False)

input_sentence = "How are you?"
input_ids = tokenizer(normalize(input_sentence), return_tensors="pt").input_ids
generated_tokens = model.generate(input_ids)
decoded_tokens = tokenizer.batch_decode(generated_tokens)[0]

print(decoded_tokens)
