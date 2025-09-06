import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from model import tokenizer, ATS_Model, device

SEQ_LEN = 512

class JD_Resume_Dataset(Dataset):
    def __init__(self, dataset: pd.DataFrame, tokenizer: AutoTokenizer, seq_len: int):
        super().__init__()
        self.dataset = dataset
        self.resume_texts = self.dataset['resume_text']
        self.job_descriptions = self.dataset['job_description_text']
        self.labels = self.dataset['label']
        # self.encoded_labels = encoded_labels
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, index) -> tuple:
        resume_text = self.resume_texts[index]
        job_description = self.job_descriptions[index]
        label = self.labels[index]

        # tokenize both resume text and job description
        resume_text_tokenize = self.tokenizer(resume_text, 
                                              return_tensors="pt", 
                                              padding='max_length',
                                              truncation = True,  
                                              max_length = self.seq_len)
        
        job_description_tokenize = self.tokenizer(job_description, 
                                                  return_tensors="pt", 
                                                  padding='max_length',
                                                  truncation = True, 
                                                  max_length = self.seq_len)

        return resume_text_tokenize, job_description_tokenize, label

def get_dataloaders():
    resumes = torch.load('resumes.pt')
    jds = torch.load('JDs.pt')
    targets = torch.load('targets.pt')
    dataset_x = pd.DataFrame({'resume_text': resumes,
                            'job_description_text': jds, 
                            })
    dataset_y = pd.DataFrame({'label': targets})
    X_train, X_test, y_train, y_test = train_test_split(dataset_x, dataset_y, test_size=0.2, random_state=42)
    train_df = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
    test_df = pd.concat([X_test, y_test], axis=1).reset_index(drop=True)

    training_dataset = JD_Resume_Dataset(train_df, tokenizer, seq_len=SEQ_LEN)
    testing_dataset = JD_Resume_Dataset(test_df, tokenizer, seq_len=SEQ_LEN)

    train_dataloader = DataLoader(training_dataset, 
                                batch_size=8, 
                                shuffle=True)
    test_dataloader = DataLoader(testing_dataset, 
                                batch_size=8, 
                                shuffle=True)
    return train_dataloader, test_dataloader

def train_model(model: ATS_Model, 
                epochs: int, 
                train_dataloader: DataLoader, 
                test_dataloader: DataLoader) -> tuple:
    criterion = nn.CrossEntropyLoss() # applies softmax to logits then applies crossentropy
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4) # use the AdamW optimzier

    train_losses = []
    val_losses = []
    model.to(device)
    scaler = torch.cuda.amp.GradScaler()


    for epoch in range(epochs):
        print(f"Epoch: {epoch+1}")
        
        # apply a forward/backward pass for the whole epoch
        print('Training...')
        batch_train_losses = []
        for resume_text, jd_text, labels in tqdm(train_dataloader):
            resume_text = resume_text.to(device)
            jd_text = jd_text.to(device)
            labels = labels.to(device)

            optim.zero_grad() # reset the gradients in optimizer
            with torch.cuda.amp.autocast():
                preds = model(resume_text, jd_text) # get the predictions from model
                loss = criterion(preds, labels) # calculate the loss
            batch_train_losses.append(loss.item())

            scaler.scale(loss).backward() # apply backprop
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # apply gradient clipping
            scaler.step(optim)
            scaler.update()

            # optim.step() # update weights
        
        train_loss = sum(batch_train_losses)/len(batch_train_losses)
        train_losses.append(train_loss)
        
        # apply a forward pass for validation
        batch_val_losses = []
        print('Validation...')
        with torch.no_grad():
            for resume_text, jd_text, labels in tqdm(test_dataloader):
                resume_text = resume_text.to(device)
                jd_text = jd_text.to(device)
                labels = labels.to(device)
                
                with torch.cuda.amp.autocast():
                    preds = model(resume_text, jd_text) # get the predictions from model
                    loss = criterion(preds, labels) # calculate the loss
                batch_val_losses.append(loss.item())
        
        val_loss = sum(batch_val_losses)/len(batch_val_losses)
        val_losses.append(val_loss)

        print(f'Training loss: {train_loss}')
        print(f'Validation loss: {val_loss}')

        print(50*"-")

    return model, train_losses, val_losses