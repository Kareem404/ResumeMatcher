import torch
from transformers import BertModel, AutoTokenizer
import torch.nn as nn
import torch.nn.functional as F
from peft import get_peft_model
import os
from dotenv import load_dotenv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

load_dotenv()
token = os.getenv("HF_TOKEN") # gets the hugging face token to be able to use the BERT model

tokenizer = AutoTokenizer.from_pretrained(
    "google-bert/bert-base-uncased", 
     token = token
)

class ATS_Model(nn.Module):
    """
    Model class that implements a dual-encoder architecture using BERT.
    - Uses a BERT transformer for resume and job description
    - Both transformers have their own LoRA adapters to save memory. Helpful if the dataset is not big
    - The CLS token of both transformers is then concatenated together and passed through a classification head of 3 layers 
    """
    def __init__(self, lora_config, seq_len: int):
        super().__init__()
        self.resume_transformer = BertModel.from_pretrained(
                                    "google-bert/bert-base-uncased",
                                    torch_dtype=torch.float16,
                                    attn_implementation="sdpa", 
                                    token = token
                                )
        self.resume_transformer = get_peft_model(self.resume_transformer, lora_config) # apply LoRA to reduce number of trainable params
        
        self.jd_transformer = BertModel.from_pretrained(
                                    "google-bert/bert-base-uncased",
                                    torch_dtype=torch.float16,
                                    attn_implementation="sdpa", 
                                    token = token
                               )
        self.jd_transformer = get_peft_model(self.jd_transformer, lora_config)
        
        self.fc1 = nn.Linear(in_features=768*2, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=2)

    def forward(self, resume_dict, jd_dict):
        res_embd = self.resume_transformer(
            resume_dict['input_ids'].squeeze(1), 
            attention_mask = resume_dict['attention_mask'].squeeze(1)
        ).pooler_output

        jd_embd = self.jd_transformer(
            jd_dict['input_ids'].squeeze(1), 
            attention_mask = jd_dict['attention_mask'].squeeze(1)
        ).pooler_output

        # concatenate the output of the embeddings layers 
        x = torch.cat((res_embd, jd_embd), dim=1)

        # pass the values to the linear layers for classification
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        logits = self.fc3(x) # do not apply softmax as this will automatically be applied using the criterion 

        return logits