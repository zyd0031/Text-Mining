#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import string
import csv
import string
import numpy as np
from itertools import groupby
import emoji
import pandas as pd


# In[2]:


def read_file_as_lists(filename, delimiter='\t'):
    with open(filename) as stream:
        reader = csv.reader(stream, delimiter=delimiter, quoting=csv.QUOTE_NONE)
        labeled_tokens = [zip(*g) for k, g in groupby(reader, lambda x: not [s for s in x if s.strip()]) if not k]
        tokens, labels = zip(*labeled_tokens)
        return [list(t) for t in tokens], [list(l) for l in labels]


# In[3]:


train_tokens, train_labels = read_file_as_lists("/kaggle/input/w-net-data/wnut17train.conll")
dev_tokens, dev_labels = read_file_as_lists("/kaggle/input/w-net-data/emerging.dev.conll")
test_tokens, test_labels = read_file_as_lists("/kaggle/input/w-net-data/emerging.test.annotated")


# In[4]:


def clean_tokens(token_list):
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
#     spell = Speller(lang='en')
    for tokens in token_list:
        for i in range(len(tokens)):
            if re.match(url_pattern, tokens[i]):
                tokens[i] = '<URL>' 
            elif emoji.emoji_count(tokens[i]) > 0:
                tokens[i] = '<emoji>'
#             else:
#                 tokens[i] = spell(tokens[i])
        
    return token_list


# In[5]:


train_tokens = clean_tokens(train_tokens)
dev_tokens = clean_tokens(dev_tokens)
test_tokens = clean_tokens(test_tokens)


# In[6]:


print("Train")
print(len(train_tokens))
print('*'*50)
print("Dev")
print(len(dev_tokens))
print('*'*50)
print("Test")
print(len(test_tokens))


# In[7]:


print("Train")
print(len([token for sublist in train_tokens for token in sublist]))
print('*'*50)
print("Dev")
print(len([token for sublist in dev_tokens for token in sublist]))
print('*'*50)
print("Test")
print(len([token for sublist in test_tokens for token in sublist]))


# In[8]:


df_train_entities = pd.DataFrame({"train_entities": [entity for sublist in train_labels for entity in sublist if entity != 'O']})
df_dev_entities = pd.DataFrame({"dev_entities": [entity for sublist in dev_labels for entity in sublist if entity != 'O']})
df_test_entities = pd.DataFrame({"test_entities": [entity for sublist in test_labels for entity in sublist if entity != 'O']})


# In[9]:


df_train_entities_count = df_train_entities["train_entities"].value_counts().sum()
df_dev_entities_count = df_dev_entities["dev_entities"].value_counts().sum()
df_test_entities_count = df_test_entities["test_entities"].value_counts().sum()
print(f"df_train_entities_count: {df_train_entities_count}")
print(f"df_dev_entities_count: {df_dev_entities_count}")
print(f"df_test_entities_count: {df_test_entities_count}")


# In[10]:


types = ["person", "location", "corporation", "product", "creative-work", "group"]
for data in [df_train_entities, df_dev_entities, df_test_entities]:
    print(data.columns[0])
    for type_ in types:
        print(f"{type_}: {data[data.columns[0]].str.endswith(type_).sum()}")
    print("*"*50)


# In[11]:


def get_label2index(data):
    labels = list(set([label for sublist in data for label in sublist]))
    index2label = dict(enumerate(labels))
    label2index = {value: key for key, value in index2label.items()}
    return label2index

label2index = get_label2index(train_labels)


# In[12]:


index2label = {value: key for key, value in label2index.items()}


# In[13]:


train_labels = [[label2index.get(i) for i in sublist] for sublist in train_labels]
dev_labels = [[label2index.get(i) for i in sublist] for sublist in dev_labels]
test_labels = [[label2index.get(i) for i in sublist] for sublist in test_labels]


# In[14]:


from datasets import Dataset

train_data = Dataset.from_dict({'tokens': train_tokens, 'labels': train_labels})
dev_data = Dataset.from_dict({'tokens': dev_tokens, 'labels': dev_labels})
test_data = Dataset.from_dict({'tokens': test_tokens, 'labels': test_labels})


# In[15]:


from datasets import DatasetDict

dataset = DatasetDict({
    'train': train_data,
    'dev': dev_data,
    'test': test_data
})


# In[16]:


from transformers import AutoTokenizer

model_checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


# In[17]:


B2I = {
    2:10,
    4:0,
    5:8,
    6:1,
    7:12,
    11:9
}


# In[18]:


def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            new_labels.append(-100)
        else:
            label = labels[word_id]
            if label in B2I.keys():
                label = B2I.get(label)
            new_labels.append(label)
    return new_labels


# In[19]:


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    all_labels = examples["labels"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs


# In[20]:


tokenized_datasets = dataset.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=dataset["train"].column_names,
)


# In[21]:


from transformers import DataCollatorForTokenClassification

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)


# In[22]:


train = data_collator(tokenized_datasets["train"])
dev = data_collator(tokenized_datasets["dev"])
test = data_collator(tokenized_datasets["test"])                  


# In[23]:


get_ipython().system('pip install seqeval')


# In[24]:


get_ipython().system('pip install evaluate')


# In[25]:


import evaluate

metric = evaluate.load("seqeval")


# In[26]:


import numpy as np


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[index2label.get(l) for l in label if l != -100] for label in labels]
    true_predictions = [
        [index2label.get(p) for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }


# In[27]:


from transformers import AutoModelForTokenClassification

model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    id2label=index2label,
    label2id=label2index,
)


# In[28]:


from transformers import TrainingArguments

args = TrainingArguments(
    "bert-finetuned-ner",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=False,
)


# In[29]:


from transformers import Trainer

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["dev"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)
trainer.train()


# In[30]:


results = trainer.evaluate(eval_dataset=tokenized_datasets["test"])
results


# In[31]:


def postprocess(predictions, labels):
    predictions = predictions.detach().cpu().clone().numpy()
    labels = labels.detach().cpu().clone().numpy()

    true_labels = [[index2label.get(l) for l in label if l != -100] for label in labels]
    true_predictions = [
        [index2label.get(p) for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    return true_labels, true_predictions


# In[32]:


from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate import Accelerator
from transformers import get_scheduler
from accelerate import Accelerator
from tqdm.auto import tqdm
import torch
from tqdm.auto import tqdm


def Training(lr, batch_size):
    
    train_dataloader = DataLoader(
        tokenized_datasets["train"],
        shuffle=True,
        collate_fn=data_collator,
        batch_size=batch_size,
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["dev"], collate_fn=data_collator, batch_size=batch_size
    )
    model = AutoModelForTokenClassification.from_pretrained(
        model_checkpoint,
        id2label=index2label,
        label2id=label2index,
    )
    optimizer = AdamW(model.parameters(), lr=lr)
    accelerator = Accelerator()
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )
    num_train_epochs = 3
    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    accelerator = Accelerator()
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )
    
    progress_bar = tqdm(range(num_training_steps))
    for epoch in range(num_train_epochs):
        # Training
        model.train()
        for batch in train_dataloader:
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        # Evaluation
        model.eval()
        for batch in eval_dataloader:
            with torch.no_grad():
                outputs = model(**batch)

            predictions = outputs.logits.argmax(dim=-1)
            labels = batch["labels"]

            predictions = accelerator.pad_across_processes(predictions, dim=1, pad_index=-100)
            labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)

            predictions_gathered = accelerator.gather(predictions)
            labels_gathered = accelerator.gather(labels)

            true_predictions, true_labels = postprocess(predictions_gathered, labels_gathered)
            metric.add_batch(predictions=true_predictions, references=true_labels)

        results = metric.compute()
        print(
            f"epoch {epoch}:",
            {
                key: results[f"overall_{key}"]
                for key in ["precision", "recall", "f1", "accuracy"]
            },
        )


# In[33]:


learning_rates = [1e-4, 5e-5, 1e-5]
batch_sizes = [8, 16]

for learning_rate in learning_rates:
    for batch_size in batch_sizes:
        print(f"learning rate:{learning_rate}")
        print(f"batch_size:{batch_size}")
        Training(lr = learning_rate, batch_size = batch_size)
        print()
        print()


# In[34]:


lr = 5e-5
batch_size = 8

train_dataloader = DataLoader(
    tokenized_datasets["train"],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=batch_size,
)
eval_dataloader = DataLoader(
    tokenized_datasets["dev"], collate_fn=data_collator, batch_size=batch_size
)
model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    id2label=index2label,
    label2id=label2index,
)
optimizer = AdamW(model.parameters(), lr=lr)
accelerator = Accelerator()
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)
num_train_epochs = 3
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
accelerator = Accelerator()
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)

progress_bar = tqdm(range(num_training_steps))
for epoch in range(num_train_epochs):
    # Training
    model.train()
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)


# In[46]:


# Evaluation
true_predictions_list = []
true_labels_list = []

model.eval()

test_dataloader = DataLoader(
    tokenized_datasets["test"], collate_fn=data_collator, batch_size=batch_size
)
device = "cuda:0"
for batch in test_dataloader:
    with torch.no_grad():
        model.to(device)  
        batch = {key: value.to(device) for key, value in batch.items()}  

        outputs = model(**batch)
        predictions = torch.argmax(outputs.logits, dim=-1)
        labels = batch["labels"]

        predictions = accelerator.pad_across_processes(predictions, dim=1, pad_index=-100)
        labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)

        predictions_gathered = accelerator.gather(predictions)
        labels_gathered = accelerator.gather(labels)

        true_predictions, true_labels = postprocess(predictions_gathered, labels_gathered)

        true_predictions_list.append(true_predictions)
        true_labels_list.append(true_labels)


# In[51]:


true_labels = [subsublist for sublist in true_labels_list for subsublist in sublist]
true_predictions = [subsublist for sublist in true_predictions for subsublist in sublist]


# In[54]:


from seqeval.metrics import precision_score, recall_score, f1_score

precision = precision_score(true_labels, true_predictions)

recall = recall_score(true_labels, true_predictions)

f1 = f1_score(true_labels, true_predictions)

print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)


# In[36]:


true_predictions_list = [item for sublist in true_predictions_list for subsublist in sublist for item in subsublist]
true_labels_list = [item for sublist in true_labels_list for subsublist in sublist for item in subsublist]


# In[37]:


from sklearn.metrics import classification_report

report = classification_report(true_labels_list, true_predictions_list)

print(report)


# In[38]:


from sklearn.metrics import f1_score

micro_f1 = f1_score(true_labels_list, true_predictions_list, average='micro')
macro_f1 = f1_score(true_labels_list, true_predictions_list, average='macro')
print(f"micro_f1:{micro_f1}")
print(f"macro_f1:{macro_f1}")


# In[39]:


entity_true_labels = [label[2:] if len(label) >1 else label for label in true_labels_list] 
entity_prediction_labels = [label[2:] if len(label) >1 else label for label in true_predictions_list] 

report = classification_report(entity_true_labels, entity_prediction_labels)
print(report)


# In[40]:


micro_f1 = f1_score(entity_true_labels, entity_prediction_labels, average='micro')
macro_f1 = f1_score(entity_true_labels, entity_prediction_labels, average='macro')
print(f"micro_f1:{micro_f1}")
print(f"macro_f1:{macro_f1}")


# In[ ]:




