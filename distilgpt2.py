import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = 'distilgpt2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)

def distilgpt2_generate_text(text):
    input = tokenizer.encode(text, return_tensors="pt")
    output = model.generate(input, max_length= 500, do_sample=True, pad_token_id = tokenizer.eos_token_id)
    return tokenizer.decode(output[0], skip_special_tokens=True)


iface = gr.Interface(
    fn=distilgpt2_generate_text,
    inputs='text', 
    outputs='text',
    title="Text generation (KendraGPT)",
    description="Genarate text using my pretrained model. Type a prompt and see how the model generates text"
)
iface.launch()