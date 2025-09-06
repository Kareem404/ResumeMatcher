import torch
from peft import LoraConfig, get_peft_model
from model import tokenizer, ATS_Model
from utils import train_model, get_dataloaders, SEQ_LEN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# initialize LoRA configuration
lora_config = LoraConfig(
    r = 16, 
    lora_alpha=16, 
    lora_dropout=0.1, 
    target_modules=['query', 'value'] # adapters adapt the Q and V matrecies in the attention heads 
)

model = ATS_Model(lora_config=lora_config, seq_len=SEQ_LEN)

train_dataloader, test_dataloader = get_dataloaders()

trained_model, train_losses, val_losses = train_model(
    model=model, 
    epochs=30, 
    train_dataloader=train_dataloader, 
    test_dataloader=test_dataloader
)