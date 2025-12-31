# List of Hugging Face text generation models with free API support
MODELS = [
    # Meta models
    "meta-llama/Meta-Llama-3-8B",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Meta-Llama-3-70B",
    "meta-llama/Meta-Llama-3-70B-Instruct",
    "meta-llama/Llama-2-7b",
    "meta-llama/Llama-2-7b-chat",
    "meta-llama/Llama-2-13b",
    "meta-llama/Llama-2-13b-chat",
    
    # Mistral models
    "mistralai/Mistral-7B-v0.1",
    "mistralai/Mistral-7B-Instruct-v0.1",
    "mistralai/Mixtral-8x7B-v0.1",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mistralai/Mistral-7B-v0.2",
    "mistralai/Mistral-7B-Instruct-v0.2",
    
    # Google models
    "google/flan-t5-base",
    "google/flan-t5-large",
    "google/flan-t5-xl",
    "google/flan-ul2",
    "google/gemma-2b",
    "google/gemma-2b-it",
    "google/gemma-7b",
    "google/gemma-7b-it",
    
    # Microsoft models
    "microsoft/phi-2",
    "microsoft/phi-1_5",
    
    # BigScience models
    "bigscience/bloom-560m",
    "bigscience/bloom-1b7",
    "bigscience/bloom-3b",
    "bigscience/bloom-7b1",
    
    # EleutherAI models
    "EleutherAI/gpt-neo-125m",
    "EleutherAI/gpt-neo-1.3B",
    "EleutherAI/gpt-neo-2.7B",
    "EleutherAI/gpt-j-6b",
    "EleutherAI/pythia-70m",
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1b",
    "EleutherAI/pythia-1.4b",
    
    # TII models
    "tiiuae/falcon-7b",
    "tiiuae/falcon-7b-instruct",
    "tiiuae/falcon-40b",
    "tiiuae/falcon-40b-instruct",
    
    # Other popular models
    "stabilityai/stablelm-base-alpha-7b",
    "stabilityai/stablelm-tuned-alpha-7b",
    "databricks/dolly-v2-3b",
    "databricks/dolly-v2-7b",
    "databricks/dolly-v2-12b",
    "togethercomputer/RedPajama-INCITE-Base-3B-v1",
    "togethercomputer/RedPajama-INCITE-Instruct-3B-v1",
    "mosaicml/mpt-7b",
    "mosaicml/mpt-7b-instruct",
    "mosaicml/mpt-30b",
    "mosaicml/mpt-30b-instruct",
    
    # Code generation models
    "Salesforce/codegen-350M-mono",
    "Salesforce/codegen-2B-mono",
    "Salesforce/codegen-6B-mono",
    "bigcode/starcoder",
    "bigcode/starcoderbase",
    
    # Smaller/efficient models
    "sshleifer/tiny-gpt2",
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "distilgpt2",
]

# Copy the MODELS list and test the ones you're interested in
print(f"Total models in list: {len(MODELS)}")