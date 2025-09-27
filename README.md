# entity-extraction-finetuning

Code for finetuning Qwen3 models to produce correctly formatted entity extraction responses for RAG systems.

## Setup
1. Clone the repository.
2. Install dependencies with `environment.yml`.
3. Set up environment
Create a `.env` file with the following content:
```
OPENAI_API_KEY="..." # Your OpenAI API key
LLM_BINDING=openai # LLM binding type (e.g., openai)
LLM_MODEL=gpt-4o # LLM model name (e.g., gpt-4o)
LLM_BINDING_HOST=... # LLM API endpoint

PATH_QWEN_4B= # Path to your local Qwen3/4B model
PATH_QWEN_8B= # Path to your local Qwen3/4B model
```

## Pipeline
### 1. Generate (text, extraction output) datasets:
  ```
  python generate_text_entities_dataset.py
  ```
  This step is optional as the datasets are pre-generated and saved in `./text_entities_dataset`.
### 2. Set the expected token length of text in `generate_long_text_dataset.py`.
  Generate (long text, extraction output) datasets:
  ```
  python generate_long_text_dataset.py
  ```
  Generate (extraction output with wrong format, with correct format) datasets (for finetuning on format):
  ```
  python generate_incorrect_correct_dataset.py
  ```
### 3. Finetune the model, either with the whole entity extraction procedure:
  ```
  python extraction_finetune.py
  ```
  or directly on incorrectly formatted responses:
  ```
  python format_finetune.py
  ```
### 4. Evaluate the model:
  ```
  python evaluate.py --model_size 4B --lora_path finetuned-4B-200-lora --use_lora --dataset text_to_correct_200_dev.json
  ```
  See `evaluate.py` for the avaliable arguments.
  Model responses and results will be saved in `./eval_outputs` and `./evaluation_summary`.
