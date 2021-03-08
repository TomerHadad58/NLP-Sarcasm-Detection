from collections import defaultdict
import time
import datetime
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
import seaborn as sns
from datasets import SarcasmHeadlinesDataset, SentimentAnalysisTweetsDataset, SarcasmRedditDataset, \
    SarcasmRedditDataset_V2, SarcasmRedditDataset_V3, SarcasmRedditDuelBertDataset, SarcasmRedditDataset_Exp1, \
    SarcasmTweetsDataset
from models import SarcasmDetector, SentimentAnalyzer
from sklearn.model_selection import train_test_split


def create_headline_sarcasm_data_loader(df, tokenizer, max_len, batch_size):
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


def create_tweets_sarcasm_data_loader(df, tokenizer, max_len, batch_size):
    ds = SarcasmTweetsDataset(
        tweets=df.tweet.to_numpy(),
        targets=df.label.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4
    )


def create_reddit_sarcasm_data_loader(df, tokenizer, max_len, batch_size):
    ds = SarcasmRedditDataset(
        comments=df.comment.to_numpy(),
        parent_comments=df.parent_comment.to_numpy(),
        targets=df.label.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4
    )


def create_reddit_sarcasm_exp1_data_loader(df, tokenizer, max_len, batch_size):
    ds = SarcasmRedditDataset_Exp1(
        comments=df.comment.to_numpy(),
        parent_comments=df.parent_comment.to_numpy(),
        targets=df.label.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4
    )


def create_reddit_sarcasm_data_loader_v2(df, tokenizer, max_len, batch_size):
    ds = SarcasmRedditDataset_V2(
        comments=df.comment.to_numpy(),
        parent_comments=df.parent_comment.to_numpy(),
        targets=df.label.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4
    )


def create_reddit_sarcasm_data_loader_v3(df, tokenizer, max_len, batch_size):
    ds = SarcasmRedditDataset_V3(
        comments=df.comment.to_numpy(),
        targets=df.label.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4
    )


def create_reddit_sarcasm_duel_bert_data_loader(df, tokenizer, max_len, batch_size):
    ds = SarcasmRedditDuelBertDataset(
        comments=df.comment.to_numpy(),
        parent_comments=df.parent_comment.to_numpy(),
        targets=df.label.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4
    )


def create_sentiment_data_loader(df, tokenizer, max_len, batch_size):
    ds = SentimentAnalysisTweetsDataset(
        tweets=df.tweet.to_numpy(),
        targets=df.target.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4
    )


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


def train_epoch_duel_bert(
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
        comment_input_ids = d["comment_input_ids"].to(device)
        comment_attention_mask = d["comment_attention_mask"].to(device)
        parent_input_ids = d["parent_input_ids"].to(device)
        parent_attention_mask = d["parent_attention_mask"].to(device)
        targets = d["targets"].to(device)

        outputs = model(
            comment_input_ids=comment_input_ids,
            comment_attention_mask=comment_attention_mask,
            parent_input_ids=parent_input_ids,
            parent_attention_mask=parent_attention_mask
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


def eval_model_duel_bert(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()

    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in data_loader:
            comment_input_ids = d["comment_input_ids"].to(device)
            comment_attention_mask = d["comment_attention_mask"].to(device)
            parent_input_ids = d["parent_input_ids"].to(device)
            parent_attention_mask = d["parent_attention_mask"].to(device)
            targets = d["targets"].to(device)

            outputs = model(
                comment_input_ids=comment_input_ids,
                comment_attention_mask=comment_attention_mask,
                parent_input_ids=parent_input_ids,
                parent_attention_mask=parent_attention_mask
            )
            _, preds = torch.max(outputs, dim=1)

            loss = loss_fn(outputs, targets)

            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)


def get_predictions(model, data_loader, text_col_name, device):
    model = model.eval()

    texts = []
    predictions = []
    prediction_probs = []
    real_values = []

    with torch.no_grad():
        for d in data_loader:
            text_batch = d[text_col_name]
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)

            probs = F.softmax(outputs, dim=1)

            texts.extend(text_batch)
            predictions.extend(preds)
            prediction_probs.extend(probs)
            real_values.extend(targets)

    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()
    return texts, predictions, prediction_probs, real_values


def get_predictions_duel_bert(model, data_loader, text_col_name, device):
    model = model.eval()

    texts = []
    predictions = []
    prediction_probs = []
    real_values = []

    with torch.no_grad():
        for d in data_loader:
            text_batch = d[text_col_name]
            comment_input_ids = d["comment_input_ids"].to(device)
            comment_attention_mask = d["comment_attention_mask"].to(device)
            parent_input_ids = d["parent_input_ids"].to(device)
            parent_attention_mask = d["parent_attention_mask"].to(device)
            targets = d["targets"].to(device)

            outputs = model(
                comment_input_ids=comment_input_ids,
                comment_attention_mask=comment_attention_mask,
                parent_input_ids=parent_input_ids,
                parent_attention_mask=parent_attention_mask
            )
            _, preds = torch.max(outputs, dim=1)

            probs = F.softmax(outputs, dim=1)

            texts.extend(text_batch)
            predictions.extend(preds)
            prediction_probs.extend(probs)
            real_values.extend(targets)

    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()
    return texts, predictions, prediction_probs, real_values


def show_confusion_matrix(confusion_matrix):
  hmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
  hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
  hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
  plt.ylabel('True sentiment')
  plt.xlabel('Predicted sentiment')
  plt.show()


def run_on_dataset(run_config):
    # Choose the device to run the model on, cpu/gpu
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    RANDOM_SEED = 42
    PRE_TRAINED_MODEL_NAME = run_config.PRE_TRAINED_MODEL_NAME

    train_epoch_func = train_epoch
    eval_model_func = eval_model
    get_predictions_func = get_predictions

    df = run_config.df
    class_names = run_config.class_names
    MAX_LEN = run_config.MAX_LEN
    BATCH_SIZE = run_config.BATCH_SIZE
    EPOCHS = run_config.EPOCHS
    dl_func = run_config.dl_func
    model = run_config.model

    '''
    if dataset_name == 'reddit':
        df = pd.read_csv('sarcasm_datasets/train-balanced-sarcasm.csv', delimiter=',')
        class_names = ['non-sarcastic', 'sarcastic']
        MAX_LEN = 120
        BATCH_SIZE = 16
        dl_func = create_reddit_sarcasm_data_loader
        model = SarcasmDetector(len(class_names))
    if dataset_name == 'reddit_v2':
        df = pd.read_csv('sarcasm_datasets/train-balanced-sarcasm.csv', delimiter=',')
        class_names = ['non-sarcastic', 'sarcastic']
        MAX_LEN = 180
        BATCH_SIZE = 16
        dl_func = create_reddit_sarcasm_data_loader_v2
        model = SarcasmDetector(len(class_names), name="sarcasm_reddit_v2")
    if dataset_name == 'reddit_v3':
        df = pd.read_csv('sarcasm_datasets/train-balanced-sarcasm.csv', delimiter=',')
        class_names = ['non-sarcastic', 'sarcastic']
        MAX_LEN = 180
        BATCH_SIZE = 16
        dl_func = create_reddit_sarcasm_data_loader_v3
        model = SarcasmDetector(len(class_names), name="sarcasm_reddit_v3")
    if dataset_name == 'reddit_duel_bert':
        df = pd.read_csv('sarcasm_datasets/train-balanced-sarcasm.csv', delimiter=',')
        class_names = ['non-sarcastic', 'sarcastic']
        MAX_LEN = 100
        BATCH_SIZE = 16
        dl_func = create_reddit_sarcasm_duel_bert_data_loader
        model = DuelBertSarcasmDetector(len(class_names), name="sarcasm_reddit_duel_bert")
        train_epoch_func = train_epoch_duel_bert
        eval_model_func = eval_model_duel_bert
        get_predictions_func = get_predictions_duel_bert
    if dataset_name == 'twitter':
        df = pd.read_csv('sentiment_datasets/twitter_sentiment_analysis.csv', delimiter=',', names=['target', 'id', 'date', 'query', 'username', 'tweet'])
        class_names = ['negative', 'positive']
        MAX_LEN = 70
        BATCH_SIZE = 32
        dl_func = create_sentiment_data_loader
        model = SentimentAnalyzer(len(class_names))
    '''
    print('Number of training sentences ds: {:,}\n'.format(df.shape[0]))

    df_train, df_test = train_test_split(df, test_size=run_config.test_percent, random_state=RANDOM_SEED)
    df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=RANDOM_SEED)
    print('Done creating dataframes')

    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    print('Loaded tokenizer')

    val_data_loader = dl_func(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
    test_data_loader = dl_func(df_test, tokenizer, MAX_LEN, BATCH_SIZE)
    print('Created data loaders')

    if run_config.model_to_load is not None:
        model.load_state_dict(torch.load(f'saved_models/{run_config.model_to_load}'))
    model.to(device)
    print('Created model')

    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    print('Created optimizer and loss_fn')

    history = defaultdict(list)
    best_accuracy = run_config.init_best_acc

    print('Starting to train...')
    for epoch in range(EPOCHS):

        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)

        df_train_of_epoch, _ = train_test_split(df_train, test_size=(1 - run_config.train_epoch_percent), random_state=int(RANDOM_SEED+epoch+222))
        train_data_loader = dl_func(df_train_of_epoch, tokenizer, MAX_LEN, BATCH_SIZE)
        total_steps = len(train_data_loader) * EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        print(f'Created train data frame, the dataloader and the optimizer for current epoch')

        train_acc, train_loss = train_epoch_func(
            model,
            train_data_loader,
            loss_fn,
            optimizer,
            device,
            scheduler,
            len(df_train_of_epoch)
        )

        print(f'Train loss {train_loss} accuracy {train_acc}')

        val_acc, val_loss = eval_model_func(
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
            str_best_acc = str(val_acc.item()*100)[0:5].replace('.', '_')
            str_train_acc = str(train_acc.item()*100)[0:5].replace('.', '_')
            print(f'New best accuracy: {str_best_acc}, saving the model...')
            torch.save(model.state_dict(), f'saved_models/best_{run_config.save_model_as}_v{str_best_acc}_t{str_train_acc}.bin')
            best_accuracy = val_acc
        if epoch+1 == EPOCHS:
            str_val_acc = str(val_acc.item() * 100)[0:5].replace('.', '_')
            str_train_acc = str(train_acc.item() * 100)[0:5].replace('.', '_')
            print(f'Saving last model, val accuracy: {str_val_acc}, train accuracy: {str_train_acc}')
            torch.save(model.state_dict(), f'saved_models/final_{run_config.save_model_as}_v{str_val_acc}_t{str_train_acc}.bin')

    if EPOCHS > 0:
        plt.plot(history['train_acc'], label='train accuracy')
        plt.plot(history['val_acc'], label='validation accuracy')

        plt.title('Training history')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        plt.ylim([0, 1])
        plt.show()

    y_review_texts, y_pred, y_pred_probs, y_test = get_predictions_func(
        model,
        test_data_loader,
        model.text_col_name,
        device
    )

    print(classification_report(y_test, y_pred, target_names=class_names))

    cm = confusion_matrix(y_test, y_pred)
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    show_confusion_matrix(df_cm)


class RunConfig:
    def __init__(self, df, class_names, MAX_LEN, BATCH_SIZE, EPOCHS, dl_func, model, PRE_TRAINED_MODEL_NAME='bert-base-cased', model_to_load=None,
                 train_func=train_epoch, eval_func=eval_model, predict_func=get_predictions, save_model_as='default',
                 test_percent=0.1,train_epoch_percent=0.5, init_best_acc=0):
        self.df = df
        self.class_names = class_names
        self.MAX_LEN = MAX_LEN
        self.BATCH_SIZE = BATCH_SIZE
        self.EPOCHS = EPOCHS
        self.PRE_TRAINED_MODEL_NAME = PRE_TRAINED_MODEL_NAME
        self.dl_func = dl_func
        self.model = model
        self.model_to_load = model_to_load
        self.train_func = train_func
        self.eval_func = eval_func
        self.predict_func = predict_func
        self.save_model_as = save_model_as
        self.test_percent = test_percent
        self.train_epoch_percent = train_epoch_percent
        self.init_best_acc = init_best_acc


if __name__ == '__main__':
    sentiment_model = SentimentAnalyzer(2)
    sentiment_model.load_state_dict(torch.load('saved_models/best_sentiment_tweets.bin'))
    sentiment_trained_bert = sentiment_model.bert
    '''
    ask_reddit_config = RunConfig(
        df=pd.read_csv('sarcasm_datasets/AskReddit_Sarcasm_Detection.csv', delimiter=','),
        class_names=['non-sarcastic', 'sarcastic'],
        MAX_LEN=120,
        BATCH_SIZE=16,
        dl_func=create_reddit_sarcasm_exp1_data_loader,
        model=SarcasmDetector(2),
        EPOCHS=12,
        train_epoch_percent=0.1,
        save_model_as='sarcasm_ask_reddit_exp1')
    ask_reddit_from_pretrained_sentiment_config = RunConfig(
        df=pd.read_csv('sarcasm_datasets/AskReddit_Sarcasm_Detection.csv', delimiter=','),
        class_names=['non-sarcastic', 'sarcastic'],
        MAX_LEN=120,
        BATCH_SIZE=16,
        dl_func=create_reddit_sarcasm_exp1_data_loader,
        model=SarcasmDetector(2, pretrained_bert=sentiment_trained_bert),
        EPOCHS=12,
        train_epoch_percent=0.1,
        save_model_as='sarcasm_ask_reddit_from_sentiment_exp1')
    sarcasm_headlines_config = RunConfig(
        df=pd.read_json('sarcasm_datasets/Sarcasm_Headlines_Dataset_v2.json', lines=True),
        class_names=['non-sarcastic', 'sarcastic'],
        MAX_LEN=90,
        BATCH_SIZE=8,
        dl_func=create_headline_sarcasm_data_loader,
        model=SarcasmDetector(2, text_col_name='headline'),
        EPOCHS=12,
        train_epoch_percent=0.1,
        save_model_as='sarcasm_headlines_exp2')
    sarcasm_headlines_from_pretrained_sentiment_config = RunConfig(
        df=pd.read_json('sarcasm_datasets/Sarcasm_Headlines_Dataset_v2.json', lines=True),
        class_names=['non-sarcastic', 'sarcastic'],
        MAX_LEN=90,
        BATCH_SIZE=8,
        dl_func=create_headline_sarcasm_data_loader,
        model=SarcasmDetector(2, text_col_name='headline', pretrained_bert=sentiment_trained_bert),
        EPOCHS=12,
        train_epoch_percent=0.1,
        save_model_as='sarcasm_headlines_from_sentiment_exp2')
    twitter_sentiment_config = RunConfig(
        df=pd.read_csv('sentiment_datasets/twitter_sentiment_analysis.csv', delimiter=',', names=['target', 'id', 'date', 'query', 'username', 'tweet']),
        class_names=['negative', 'positive'],
        MAX_LEN=70,
        BATCH_SIZE=64,
        dl_func=create_sentiment_data_loader,
        model=SentimentAnalyzer(2),
        EPOCHS=10,
        train_epoch_percent=0.01,
        test_percent=0.005,
        model_to_load='best_sentiment_tweets.bin',
        save_model_as='sentiment_tweets')
    test_headlines_with_reddit_model_config = RunConfig(
        df=pd.read_json('sarcasm_datasets/Sarcasm_Headlines_Dataset_v2.json', lines=True),
        class_names=['non-sarcastic', 'sarcastic'],
        MAX_LEN=90,
        BATCH_SIZE=8,
        dl_func=create_headline_sarcasm_data_loader,
        model=SarcasmDetector(2, text_col_name='headline'),
        model_to_load='best_sarcasm_ask_reddit_exp1_v75_76_t82_16.bin',
        EPOCHS=0)
    test_headlines_with_reddit_transfer_model_config = RunConfig(
        df=pd.read_json('sarcasm_datasets/Sarcasm_Headlines_Dataset_v2.json', lines=True),
        class_names=['non-sarcastic', 'sarcastic'],
        MAX_LEN=90,
        BATCH_SIZE=8,
        dl_func=create_headline_sarcasm_data_loader,
        model=SarcasmDetector(2, text_col_name='headline'),
        model_to_load='best_sarcasm_ask_reddit_from_sentiment_exp1_v75_73_t82_58.bin',
        EPOCHS=0)
    test_reddit_with_headlines_model_config = RunConfig(
        df=pd.read_csv('sarcasm_datasets/AskReddit_Sarcasm_Detection.csv', delimiter=','),
        class_names=['non-sarcastic', 'sarcastic'],
        MAX_LEN=120,
        BATCH_SIZE=16,
        dl_func=create_reddit_sarcasm_exp1_data_loader,
        model=SarcasmDetector(2),
        model_to_load='best_sarcasm_headlines_exp2_v91_40_t93_63.bin',
        EPOCHS=0)
    test_reddit_with_headlines_transfer_model_config = RunConfig(
        df=pd.read_csv('sarcasm_datasets/AskReddit_Sarcasm_Detection.csv', delimiter=','),
        class_names=['non-sarcastic', 'sarcastic'],
        MAX_LEN=120,
        BATCH_SIZE=16,
        dl_func=create_reddit_sarcasm_exp1_data_loader,
        model=SarcasmDetector(2),
        model_to_load='best_sarcasm_headlines_from_sentiment_exp2_v91_47_t94_83.bin',
        EPOCHS=0)
    
    test_sarcasm_tweets_with_headlines_model_config = RunConfig( #its shit
        df=pd.read_csv('sarcasm_datasets/sarcastic_tweets.csv', delimiter=','),
        class_names=['non-sarcastic', 'sarcastic'],
        MAX_LEN=70,
        BATCH_SIZE=16,
        dl_func=create_tweets_sarcasm_data_loader,
        model=SarcasmDetector(2, text_col_name='tweet'),
        model_to_load='best_sarcasm_headlines_exp2_v91_40_t93_63.bin',
        EPOCHS=0,
        test_percent=0.99)
    sarcasm_tweets_config = RunConfig(
        df=pd.read_csv('sarcasm_datasets/sarcastic_tweets.csv', delimiter=','),
        class_names=['non-sarcastic', 'sarcastic'],
        MAX_LEN=70,
        BATCH_SIZE=16,
        dl_func=create_tweets_sarcasm_data_loader,
        model=SarcasmDetector(2, text_col_name='tweet'),
        EPOCHS=12,
        train_epoch_percent=0.1,
        save_model_as='sarcasm_tweets')
    sarcasm_tweets_from_pretrained_sentiment_config = RunConfig(
        df=pd.read_csv('sarcasm_datasets/sarcastic_tweets.csv', delimiter=','),
        class_names=['non-sarcastic', 'sarcastic'],
        MAX_LEN=70,
        BATCH_SIZE=16,
        dl_func=create_tweets_sarcasm_data_loader,
        model=SarcasmDetector(2, text_col_name='tweet', pretrained_bert=sentiment_trained_bert),
        EPOCHS=12,
        train_epoch_percent=0.1,
        save_model_as='sarcasm_tweets_from_sentiment')
    '''

    ask_reddit_config = RunConfig(
        df=pd.read_csv('sarcasm_datasets/AskReddit_Sarcasm_Detection.csv', delimiter=','),
        class_names=['non-sarcastic', 'sarcastic'],
        MAX_LEN=70,
        BATCH_SIZE=32,
        dl_func=create_reddit_sarcasm_exp1_data_loader,
        model=SarcasmDetector(2),
        EPOCHS=12,
        train_epoch_percent=0.1,
        save_model_as='sarcasm_ask_reddit_exp2')
    ask_reddit_from_pretrained_sentiment_config = RunConfig(
        df=pd.read_csv('sarcasm_datasets/AskReddit_Sarcasm_Detection.csv', delimiter=','),
        class_names=['non-sarcastic', 'sarcastic'],
        MAX_LEN=70,
        BATCH_SIZE=32,
        dl_func=create_reddit_sarcasm_exp1_data_loader,
        model=SarcasmDetector(2, pretrained_bert=sentiment_trained_bert),
        EPOCHS=12,
        train_epoch_percent=0.1,
        save_model_as='sarcasm_ask_reddit_from_sentiment_exp2')

    run_on_dataset(ask_reddit_config)
    run_on_dataset(ask_reddit_from_pretrained_sentiment_config)
