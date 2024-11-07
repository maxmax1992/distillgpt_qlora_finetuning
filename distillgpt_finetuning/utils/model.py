from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType


def get_text_embedding_model():
    # Step 1: Load the tokenizer and model
    model_name = "distilgpt2"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load the model with 4-bit quantization
    distillgpt_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # load_in_4bit=True,       # Enable 4-bit quantization - uncomment this if your setup supports this
        device_map="auto"        # Automatically map modules to devices
    )

    # Uncomment this if your env supports bitsandbytes package (I'm using M3 mac so I can't use this)
    # Step 2: Prepare the model for k-bit (quantized) training
    # model = prepare_model_for_kbit_training(model)

    # Step 3: Set up LoRA configuration
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # Task type for causal language modeling
        inference_mode=False,          # Set to False for training mode
        target_modules=['c_attn'],
        r=8,                           # Rank of the LoRA matrices
        lora_alpha=16,                 # Scaling factor for LoRA updates
        lora_dropout=0.3               # Dropout probability for LoRA layers
    )

    # Step 4: Apply QLoRA to the model
    distillgpt_model = get_peft_model(distillgpt_model, config)

    # Since we're having very small dataset reduce number of trainable parameters
    print("Use reduced set of trainable parameters")
    # Print the trainable parameters
    distillgpt_model.print_trainable_parameters()
    # Now, the model is ready to be trained using PyTorch
    return distillgpt_model, tokenizer


# Define the main classification model with Lora
class DocumentClassifier(nn.Module):
    def __init__(self, distillgpt_model, tokenizer, n_classes):
        super(DocumentClassifier, self).__init__()
        self.tokenizer = tokenizer
        self.text_embedding_model = distillgpt_model
        self.device = distillgpt_model.device
        self.emb_projection_size = 128
        self.text_emb_projection_map = nn.Linear(768, self.emb_projection_size)
        # text features
        self.text_embedding_feature_names = ['submitter', 'title', 'comments', 'journal-ref', 'report-no', 'abstract']
        self.collapse_features = ['categories', 'authors']
        # numerical features
        self.numerical_feature_names = ['days_since_update', 'n_versions', 'n_authors', 'n_categories']
        input_emb_size = (len(self.text_embedding_feature_names) + len(self.collapse_features)) * self.emb_projection_size
        self.prediction_layer = nn.Linear(input_emb_size + len(self.numerical_feature_names), n_classes)

    
    def forward(self, input_batch):
        batch_embs = []
        for item in input_batch:
            # print(item)
            sentences = [item[key] for key in self.text_embedding_feature_names]
            text_embeddings = self.text_emb_projection_map(self.get_text_embeddings(sentences))
            collapsed_embeddings_1 = self.text_emb_projection_map(self.get_text_embeddings(item[self.collapse_features[0]]).sum(axis=0))
            collapsed_embeddings_2 = self.text_emb_projection_map(self.get_text_embeddings(item[self.collapse_features[1]]).sum(axis=0))
            text_embeddings = rearrange(text_embeddings, 'n d -> (n d)')
            numerical_feats_tensor = torch.tensor([item[key] for key in self.numerical_feature_names]).float().to(self.text_embedding_model.device)
            batch_embs.append(torch.concat([text_embeddings, collapsed_embeddings_1, collapsed_embeddings_2, numerical_feats_tensor]))


        x = torch.stack(batch_embs)
        x = self.prediction_layer(F.relu(x))
        x = F.log_softmax(x, dim=1)
        return x

    def get_text_embeddings(self, sequence_of_text):
        # Tokenize the input text
        inputs = self.tokenizer(sequence_of_text, return_tensors="pt", padding=True, truncation=True)

        # Move tensors to the appropriate device
        input_ids = inputs["input_ids"].to(self.text_embedding_model.device)
        attention_mask = inputs["attention_mask"].to(self.text_embedding_model.device)

        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # Forward pass
        outputs = self.text_embedding_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]  # Get last layer hidden states
        embeddings = hidden_states.mean(dim=1)     #
        return embeddings


def get_model(n_classes: int) -> Tuple[torch.nn.Module, torch.nn.Module]:
    distillgpt_model, tokenizer = get_text_embedding_model()
    model = DocumentClassifier(distillgpt_model, tokenizer, n_classes)
    return model.to(distillgpt_model.device)
