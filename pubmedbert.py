from transformers import AutoTokenizer, AutoModel, AutoModelForTokenClassification, TrainingArguments, Trainer, pipeline
import platform
import torch
import numpy as np
import pandas as pd

model_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"

