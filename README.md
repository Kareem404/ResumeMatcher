# ResumeMatcher
a Proof-of-concept (PoC) that checks if a resume is a best fit to a Job description or not. A deep learning model was designed using dual-encoders trained on a generated synthetic dataset.

# 1. Architecture
The approach relies on dual-encoder transformers finetuned with a classification head. Two [bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) transformers were used and they were finetuned using LoRA adapters. After that, the output of the transformers' classification token (CLS) is then sent to a classification head of three layers to classify whether the Resume is a good match for the job description or not.

# 2. Training
To train the model, you need to install a GPU-enabled version PyTorch. Make sure it is compatibale with your CUDA version. Then, install dependencies: <br> 
```
pip install -r requirements.txt
```
<br> After that, simply run the training script:<br>
```
python train.py
```

# 3. Deployment
Coming soon!
