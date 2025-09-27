# Step 1.
# Generate (text, extraction output) datasets.
#
# Call a powerful LLM to generate entity extraction responses
# to text in the CoNLL-2003 dataset,
# filter the responses with extraction format and ground-truth entities from dataset,
# and save the correct answers as dataset for finetuning.
# 
# This script uses the older version of LightRAG delimiters,
# which are replaced in the next step, `generate_long_text_dataset.py`.
#
# To avoid wasting resources, the generated datasets are provided in `text_entities_dataset`.

from dotenv import load_dotenv
import os
import asyncio
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
from datasets import load_dataset, DatasetDict, Dataset
import json
import re

from prompt import PROMPTS

# Configuration
MODEL_NAME = "gpt-4o-mini"
BATCH_SIZE = 20
MAX_TRAIN_SIZE = 14000
MAX_DEV_SIZE = 3200
MAX_TEST_SIZE = 500

# Load environment variables, initialize OpenAI client
load_dotenv(dotenv_path=".env", override=True)

base_url = os.getenv("LLM_BINDING_HOST")
client = AsyncOpenAI(base_url=base_url)

# Load Dataset
print("Loading CoNLL-2003 dataset...")
dataset: DatasetDict = load_dataset("conll2003.py", cache_dir="conll2003", trust_remote_code=True)
train_data: Dataset = dataset["train"]
dev_data: Dataset = dataset["validation"]
test_data: Dataset = dataset["test"]

if MAX_TRAIN_SIZE:
    train_data = train_data.select(range(5000, MAX_TRAIN_SIZE))
if MAX_DEV_SIZE:
    dev_data = dev_data.select(range(1000, MAX_DEV_SIZE))
if MAX_TEST_SIZE:
    test_data = test_data.select(range(MAX_TEST_SIZE)) 
    
print(f"Dataset loaded. Processing {len(train_data)} + {len(dev_data)} + {len(test_data)} samples.")


async def get_llm_response(text: str):
    """
    Calls LLM for entity extraction.
    """
    # The older version of prompts are used here
    prompt_template = PROMPTS["entity_extraction"]
    
    # The instructional part of the prompt goes into the system message
    system_prompt_template, user_prompt_template = prompt_template.split("#############################\n---Real Data---")
    
    system_prompt = system_prompt_template.format(
        language=PROMPTS["DEFAULT_LANGUAGE"],
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types = ["person", "organization", "location", "miscellaneous"],
        examples= "\n".join(PROMPTS["entity_extraction_examples"]).format(
            tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
            record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
            completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        ),
    )

    # The data-specific part of the prompt goes into the user message
    user_prompt = user_prompt_template.format(
        entity_types = ["person", "organization", "location", "miscellaneous"],
        input_text=text
    )

    try:
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            # The API does not support json_object response format with this prompt structure
            # response_format={"type": "json_object"}, 
            temperature=0,
        )
        return {"sentence": text, "response": response.choices[0].message.content}
    except Exception as e:
        return {"sentence": text, "error": str(e)}


def parse_ground_truth_entities(tokens, ner_tags, ner_tag_names):
    """Parses entities from CoNLL-2003 tokens and NER tags."""
    entities = set()
    current_entity_tokens = []
    current_entity_type = None

    for token, tag_idx in zip(tokens, ner_tags):
        tag_name = ner_tag_names[tag_idx]
        if tag_name.startswith("B-"):
            if current_entity_tokens:
                entities.add((" ".join(current_entity_tokens), current_entity_type))
            current_entity_tokens = [token]
            current_entity_type = tag_name.split("-")[1]
        elif tag_name.startswith("I-") and current_entity_type == tag_name.split("-")[1]:
            current_entity_tokens.append(token)
        else:
            if current_entity_tokens:
                entities.add((" ".join(current_entity_tokens), current_entity_type))
            current_entity_tokens = []
            current_entity_type = None
    
    if current_entity_tokens:
        entities.add((" ".join(current_entity_tokens), current_entity_type))
    
    return entities

def parse_llm_response_entities(response_text):
    """
    Parses entities from the LLM extraction output using regex.
    """
    tuple_delimiter = re.escape(PROMPTS["DEFAULT_TUPLE_DELIMITER"])
    pattern = re.compile(
        r'\(\s*"?entity"?\s*' + tuple_delimiter +
        r'\s*"(.*?)"\s*' + tuple_delimiter +
        r'\s*"(.*?)"'
    )
    
    matches = pattern.findall(response_text)
    
    # Normalize types to match CoNLL format (e.g., 'person' -> 'PER')
    entity_type_to_tags = {
        "person": "PER",
        "organization": "ORG",
        "location": "LOC",
        "miscellaneous": "MISC"
    }
    llm_entities = set()
    for name, entity_type in matches:
        if entity_type in entity_type_to_tags:
            llm_entities.add((name, entity_type_to_tags[entity_type]))
            
    return llm_entities

def is_format_correct(response_text):
    """
    Validates if the LLM response strictly adheres to the custom format,
    including the use of all three specified delimiters.
    """
    tuple_delimiter = PROMPTS["DEFAULT_TUPLE_DELIMITER"]
    record_delimiter = PROMPTS["DEFAULT_RECORD_DELIMITER"]
    completion_delimiter = PROMPTS["DEFAULT_COMPLETION_DELIMITER"]

    # Verify the existence of the completion delimiter
    if not response_text.endswith(completion_delimiter):
        return False
    
    # Remove the completion delimiter and split by the record delimiter.
    main_content = response_text.removesuffix(completion_delimiter).strip()
    if not main_content:
        return False # Empty response
    records = main_content.split(record_delimiter)

    if not records:
        return False # Must have at least one record

    # Validate each record.
    for record in records:
        record = record.strip()
        if not record: 
            continue

        # Verify correct tuple structure
        if not (record.startswith('(') and record.endswith(')')):
            return False

        # Verify the number of tuple delimiters in each record
        num_delimiters = record.count(tuple_delimiter)
        if '"entity"' in record:
            if num_delimiters != 3: return False
        elif '"relationship"' in record:
            if num_delimiters != 5: return False
        elif '"content_keywords"' in record:
            if num_delimiters != 1: return False
        else:
            return False
            
    return True

async def generate_and_filter(data, data_type):
    # Main logic
    all_results = []
    
    # Prepare items sentences from the dataset
    tokens = [example["tokens"] for example in data]
    sentences = [" ".join(example["tokens"]) for example in data]
    ner_tags_list = [example["ner_tags"] for example in data]
    ner_tag_names = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']

    print(f"Sending {len(sentences)} sentences to {MODEL_NAME} in batches of {BATCH_SIZE}...")

    # Create batches of tasks
    tasks = [get_llm_response(sentence) for sentence in sentences]
    
    # Progress bar
    for i in range(0, len(tasks), BATCH_SIZE):
        batch = tasks[i:i+BATCH_SIZE]
        results = await tqdm_asyncio.gather(*batch, desc=f"Processing batch {i//BATCH_SIZE + 1}")
        all_results.extend(results)

    # Answer Filtering
    print("\nFiltering results for correctly extracted entities...")
    correctly_extracted_results = []
    for i, result in enumerate(all_results):
        if "error" in result:
            continue

        # Check for perfect extraction output format
        if not is_format_correct(result["response"]):
           continue

        # Parse ground truth entities from the dataset
        ground_truth_entities = parse_ground_truth_entities(tokens[i], ner_tags_list[i], ner_tag_names)
        
        # Parse entities from LLM extraction response
        llm_entities = parse_llm_response_entities(result["response"])

        # The extraction is considered correct if the ground truth entities are all extracted (a subset of the LLM's extracted entities).
        if ground_truth_entities.issubset(llm_entities) or ground_truth_entities == llm_entities:
            correctly_extracted_results.append(result)

    print(f"Found {len(correctly_extracted_results)}/{len(all_results) - sum(1 for r in all_results if 'error' in r)} responses with perfect entity extraction and format.")

    # Save to a json file
    with open(f"./text_to_correct_{data_type}.json", "w") as f:
        json.dump(correctly_extracted_results, f, indent=2)
    print(f"\nSaved {len(correctly_extracted_results)} correctly extracted results to ./text_to_correct_{data_type}.json")


async def main():
    await generate_and_filter(train_data, "train")
    await generate_and_filter(dev_data, "dev")
    #await generate_and_filter(test_data, "test")

if __name__ == "__main__":
    asyncio.run(main())