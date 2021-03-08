from collections import defaultdict
import time
import datetime
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
import seaborn as sns


class SarcasmHeadlinesDataset(Dataset):

    def __init__(self, headlines, targets, tokenizer, max_len):
        self.headlines = headlines
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.headlines)

    def __getitem__(self, item):
        headline = str(self.headlines[item])
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(
            headline,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
            padding='max_length'
        )

        return {
            'headline': headline,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }


def create_data_loader(df, tokenizer, max_len, batch_size):
  ds = SarcasmHeadlinesDataset(
    headlines=df.headline.to_numpy(),
    targets=df.is_sarcastic.to_numpy(),
    tokenizer=tokenizer,
    max_len=max_len
  )

  return DataLoader(
    ds,
    batch_size=batch_size,
    num_workers=4
  )


class SarcasmDetector(torch.nn.Module):

    def __init__(self, n_classes):
        super(SarcasmDetector, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = torch.nn.Dropout(p=0.3)
        self.out = torch.nn.Linear(self.bert.config.hidden_size, n_classes)

    def __init__(self, n_classes, pretrained_bert):
        super(SarcasmDetector, self).__init__()
        self.bert = pretrained_bert
        self.drop = torch.nn.Dropout(p=0.3)
        self.out = torch.nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        bert_out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        output = self.drop(bert_out['pooler_output'])
        return self.out(output)


class SentimentAnalyzer(torch.nn.Module):

    def __init__(self, n_classes):
        super(SentimentAnalyzer, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = torch.nn.Dropout(p=0.3)
        self.out = torch.nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        bert_out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        output = self.drop(bert_out['pooler_output'])
        return self.out(output)


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def train_epoch(
        model,
        data_loader,
        loss_fn,
        optimizer,
        device,
        scheduler,
        n_examples
):
    model = model.train()

    losses = []
    correct_predictions = 0

    batch_num = 0
    dl_len = len(data_loader)
    t0 = time.time()

    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)

        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if batch_num % 40 == 0 and not batch_num == 0:
            elapsed = format_time(time.time() - t0)
            print(f'{batch_num} / {dl_len}, Elapsed - {elapsed}')

        batch_num += 1

    print(f"Training epcoh took: {format_time(time.time() - t0)}")
    return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(model, data_loader, loss_fn, device, n_examples):
  model = model.eval()

  losses = []
  correct_predictions = 0

  with torch.no_grad():
    for d in data_loader:
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      targets = d["targets"].to(device)

      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
      _, preds = torch.max(outputs, dim=1)

      loss = loss_fn(outputs, targets)

      correct_predictions += torch.sum(preds == targets)
      losses.append(loss.item())

  return correct_predictions.double() / n_examples, np.mean(losses)


def get_predictions(model, data_loader):
    model = model.eval()

    headlines = []
    predictions = []
    prediction_probs = []
    real_values = []

    with torch.no_grad():
        for d in data_loader:
            texts = d["headline"]
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)

            probs = F.softmax(outputs, dim=1)

            headlines.extend(texts)
            predictions.extend(preds)
            prediction_probs.extend(probs)
            real_values.extend(targets)

    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()
    return headlines, predictions, prediction_probs, real_values


def show_confusion_matrix(confusion_matrix):
  hmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
  hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
  hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
  plt.ylabel('True sentiment')
  plt.xlabel('Predicted sentiment')
  plt.show()


if __name__ == '__main__':
    # Choose the device to run the model on, cpu/gpu
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")


    # Read the dataset from the json
    df = pd.read_json('sarcasm_datasets/Sarcasm_Headlines_Dataset_v2.json', lines=True)
    print('Number of training sentences: {:,}\n'.format(df.shape[0]))

    sarcasm_class_names = ['non-sarcastic', 'sarcastic']
    sentiment_class_names = ['negative', 'positive']

    from sklearn.model_selection import train_test_split
    RANDOM_SEED = 42
    df_train, df_test = train_test_split(df, test_size=0.1, random_state=RANDOM_SEED)
    df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=RANDOM_SEED)
    print('Done creating dataframes')
    # Loading Tokenizer
    PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
    MAX_LEN = 70
    BATCH_SIZE = 32

    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    print('Loaded tokenizer')

    train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
    val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
    test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)
    print('Created data loaders')

    model = SarcasmDetector(len(sarcasm_class_names))
    # model.load_state_dict(torch.load('best_model_state.bin')) # Uncomment to reload the existing best model save
    model = model.to(device)
    print('Created model')

    EPOCHS = 1

    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    total_steps = len(train_data_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
      optimizer,
      num_warmup_steps=0,
      num_training_steps=total_steps
    )
    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    print('Created optimizer, scheduler and loss_fn')

    history = defaultdict(list)
    best_accuracy = 0

    print('Starting to train...')
    for epoch in range(EPOCHS):

      print(f'Epoch {epoch + 1}/{EPOCHS}')
      print('-' * 10)

      train_acc, train_loss = train_epoch(
        model,
        train_data_loader,
        loss_fn,
        optimizer,
        device,
        scheduler,
        len(df_train)
      )

      print(f'Train loss {train_loss} accuracy {train_acc}')

      val_acc, val_loss = eval_model(
        model,
        val_data_loader,
        loss_fn,
        device,
        len(df_val)
      )

      print(f'Val   loss {val_loss} accuracy {val_acc}')
      print()

      history['train_acc'].append(train_acc)
      history['train_loss'].append(train_loss)
      history['val_acc'].append(val_acc)
      history['val_loss'].append(val_loss)

      if val_acc > best_accuracy:
        print(f'New best accuracy: {str(val_acc)}, saving the model...')
        torch.save(model.state_dict(), 'best_model_state.bin')
        best_accuracy = val_acc

    plt.plot(history['train_acc'], label='train accuracy')
    plt.plot(history['val_acc'], label='validation accuracy')

    plt.title('Training history')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.ylim([0, 1])
    plt.show()

    y_review_texts, y_pred, y_pred_probs, y_test = get_predictions(
        model,
        test_data_loader
    )

    print(classification_report(y_test, y_pred, target_names=sarcasm_class_names))

    cm = confusion_matrix(y_test, y_pred)
    df_cm = pd.DataFrame(cm, index=sarcasm_class_names, columns=sarcasm_class_names)
    show_confusion_matrix(df_cm)

    '''
    # For later
    idx = 2

review_text = y_review_texts[idx]
true_sentiment = y_test[idx]
pred_df = pd.DataFrame({
  'class_names': class_names,
  'values': y_pred_probs[idx]
})

print("\n".join(wrap(review_text)))
print()
print(f'True sentiment: {class_names[true_sentiment]}')

sns.barplot(x='values', y='class_names', data=pred_df, orient='h')
plt.ylabel('sentiment')
plt.xlabel('probability')
plt.xlim([0, 1])
plt.show()

# Raw text
raw_headline = "I love completing my todos! Best app ever!!!"

encoded_headline = tokenizer.encode_plus(
            raw_headline,
            add_special_tokens=True,
            max_length=MAX_LEN,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
            padding='max_length'
        )

input_ids = encoded_headline['input_ids'].to(device)
attention_mask = encoded_headline['attention_mask'].to(device)

output = model(input_ids, attention_mask)
_, prediction = torch.max(output, dim=1)

print(f'Review text  : {review_text}')
print(f'Is sarcastic : {class_names[prediction]}')
    '''