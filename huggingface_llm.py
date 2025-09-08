from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch


def get_hf_llm():
    model_name = "meta-llama/Llama-2-7b-chat-hf"  

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    device = 0 if torch.cuda.is_available() else -1

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        device=device,
    )

    return HuggingFacePipeline(pipeline=pipe)
