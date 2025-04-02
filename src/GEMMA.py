from unsloth import FastModel
import torch

fourbit_models = [
    # 4bit dynamic quants for superior accuracy and low memory use
    "unsloth/gemma-3-1b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-27b-it-unsloth-bnb-4bit",

    # Other popular models!
    "unsloth/Llama-3.1-8B",
    "unsloth/Llama-3.2-3B",
    "unsloth/Llama-3.3-70B",
    "unsloth/mistral-7b-instruct-v0.3",
    "unsloth/Phi-4",
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/gemma-3-4b-it",
    max_seq_length = 4096, # Choose any for long context!
    load_in_4bit = True,  # 4 bit quantization to reduce memory
    load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
    full_finetuning = False, # [NEW!] We have full finetuning now!
    # token = "hf_...", # use one if using gated models
)


# We now add LoRA adapters so we only need to update a small amount of parameters!

# In[2]:


model = FastModel.get_peft_model(
    model,
    finetune_vision_layers     = False, # Turn off for just text!
    finetune_language_layers   = True,  # Should leave on!
    finetune_attention_modules = True,  # Attention good for GRPO
    finetune_mlp_modules       = True,  # SHould leave on always!

    r = 8,           # Larger = higher accuracy, but might overfit
    lora_alpha = 8,  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
)


# <a name="Data"></a>
# ### Data Prep
# We now use the `Gemma-3` format for conversation style finetunes. We use [Maxime Labonne's FineTome-100k](https://huggingface.co/datasets/mlabonne/FineTome-100k) dataset in ShareGPT style. Gemma-3 renders multi turn conversations like below:
# 
# ```
# <bos><start_of_turn>user
# Hello!<end_of_turn>
# <start_of_turn>model
# Hey there!<end_of_turn>
# ```
# 
# We use our `get_chat_template` function to get the correct chat template. We support `zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, phi3, llama3, phi4, qwen2.5, gemma3` and more.

# In[3]:


from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma-3",
)


# In[ ]:


# from datasets import load_dataset

# # Replace with your actual path
# dataset = load_dataset("json", data_files="dutch_ir_pairs_huggingface_combined.json")

# # To check the contents
# print(dataset)


# In[33]:


# dataset = dataset['train']


# In[4]:


from datasets import load_dataset
# dataset = load_dataset("mlabonne/FineTome-100k", split = "train")


# In[35]:


# dataset = load_dataset("BramVanroy/alpaca-cleaned-dutch", split="train_sft")


# In[5]:


from datasets import load_dataset, concatenate_datasets, DatasetDict

data = load_dataset("json", data_files="dutch_ir_pairs_huggingface_combined.json")
data_1 = load_dataset("json", data_files="dutch_ir_pairs_huggingface_nemotron_combined.json")
data_2 = load_dataset("json", data_files="huggingface_legal_dataset.json")
data_3 = load_dataset("json", data_files = "dutch_parquet_huggingface_train_combined.json")
combined_dict = {}
for split in data.keys():  # Assuming all datasets have the same splits
    # Concatenate the datasets for this split
    combined_split = concatenate_datasets([data[split], data_1[split], data_2[split], data_3[split]])
    combined_dict[split] = combined_split

# Step 2: Create a new DatasetDict with the combined splits
combined_dataset = DatasetDict(combined_dict)

dataset = combined_dataset['train']


# We now use `standardize_data_formats` to try converting datasets to the correct format for finetuning purposes!

# In[6]:


from unsloth.chat_templates import standardize_data_formats
dataset = standardize_data_formats(dataset)

# data = standardize_data_formats(data)


# Let's see how row 100 looks like!

# In[39]:


# dataset[0]


# We now have to apply the chat template for `Gemma-3` onto the conversations, and save it to `text`

# In[7]:


def apply_chat_template(examples):
    texts = tokenizer.apply_chat_template(examples["conversations"])
    return {"text": texts}

dataset = dataset.map(apply_chat_template, batched=True)

# Process the `data` list using a list comprehension


# In[ ]:


#check the whole dataset for any None values


# Let's see how the chat template did! Notice `Gemma-3` default adds a `<bos>`!

# <a name="Train"></a>
# ### Train the model
# Now let's use Huggingface TRL's `SFTTrainer`! More docs here: [TRL SFT docs](https://huggingface.co/docs/trl/sft_trainer). We do 60 steps to speed things up, but you can set `num_train_epochs=1` for a full run, and turn off `max_steps=None`.

# In[8]:


from trl import SFTTrainer, SFTConfig
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    eval_dataset = None, # Can set up evaluation!
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 8,
        gradient_accumulation_steps = 4, # Use GA to mimic batch size!
        warmup_steps = 5,
        #num_train_epochs = 2, # Set this for 1 full training run.
        max_steps = 1000,
        save_steps = 100,
        learning_rate = 2e-4, # Reduce to 2e-5 for long training runs
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to = "none", # Use this for WandB etc
    ),
)


# We also use Unsloth's `train_on_completions` method to only train on the assistant outputs and ignore the loss on the user's inputs. This helps increase accuracy of finetunes!

# In[9]:


from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<start_of_turn>user\n",
    response_part = "<start_of_turn>model\n",
)


# Let's verify masking the instruction part is done! Let's print the 100th row again:

# In[10]:


tokenizer.decode(trainer.train_dataset[100]["input_ids"])


# Now let's print the masked out example - you should see only the answer is present:

# In[11]:


tokenizer.decode([tokenizer.pad_token_id if x == -100 else x for x in trainer.train_dataset[100]["labels"]]).replace(tokenizer.pad_token, " ")


# In[ ]:





# In[12]:


# @title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)  # Changed from 3 to 1 for the 2nd GPU
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")


# Let's train the model! To resume a training run, set `trainer.train(resume_from_checkpoint = True)`

# In[13]:


#clear the cache

torch.cuda.empty_cache()


# In[14]:


trainer_stats = trainer.train()


# In[17]:


# @title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


# <a name="Inference"></a>
# ### Inference
# Let's run the model via Unsloth native inference! According to the `Gemma-3` team, the recommended settings for inference are `temperature = 1.0, top_p = 0.95, top_k = 64`

# In[18]:


from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma-3",
)
messages = [{
    "role": "user",
    "content": [{
        "type" : "text",
        "text" : "\nHoe kan ik mijn audiodiensten voor podcasts verkopen?\n",
    }]
}]
text = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True, # Must add for generation
)
outputs = model.generate(
    **tokenizer([text], return_tensors = "pt").to("cuda"),
    max_new_tokens = 2048, # Increase for longer outputs!
    # Recommended Gemma-3 settings!
    temperature = 1.0, top_p = 0.95, top_k = 64,
)
tokenizer.batch_decode(outputs)


#  You can also use a `TextStreamer` for continuous inference - so you can see the generation token by token, instead of waiting the whole time!

# In[19]:


messages = [{
    "role": "user",
    "content": [{"type" : "text", "text" : "Leg uit wat relativiteitstheorie doet en quantum mechanica",}]
}]
text = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True, # Must add for generation
)

from transformers import TextStreamer
_ = model.generate(
    **tokenizer([text], return_tensors = "pt").to("cuda"),
    max_new_tokens = 200, # Increase for longer outputs!
    # Recommended Gemma-3 settings!
    temperature = 1.0, top_p = 0.95, top_k = 64,
    streamer = TextStreamer(tokenizer, skip_prompt = True),
)


# <a name="Save"></a>
# ### Saving, loading finetuned models
# To save the final model as LoRA adapters, either use Huggingface's `push_to_hub` for an online save or `save_pretrained` for a local save.
# 
# **[NOTE]** This ONLY saves the LoRA adapters, and not the full model. To save to 16bit or GGUF, scroll down!

# In[86]:


model.save_pretrained("gemma-3-DUTCH")  # Local saving
tokenizer.save_pretrained("gemma-3-DUTCH")
# model.push_to_hub("HF_ACCOUNT/gemma-3", token = "...") # Online saving
# tokenizer.push_to_hub("HF_ACCOUNT/gemma-3", token = "...") # Online saving
