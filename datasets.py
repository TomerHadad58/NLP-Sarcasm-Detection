import torch
from torch.utils.data import Dataset


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


class SarcasmTweetsDataset(Dataset):

    def __init__(self, tweets, targets, tokenizer, max_len):
        self.tweets = tweets
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, item):
        tweet = str(self.tweets[item])
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(
            tweet,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
            padding='max_length'
        )

        return {
            'tweet': tweet,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }


class SentimentAnalysisTweetsDataset(Dataset):

    def __init__(self, tweets, targets, tokenizer, max_len):
        self.tweets = tweets
        for i in range(len(targets)):
            targets[i] = int(targets[i]/4)
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, item):
        tweet = str(self.tweets[item])
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(
            tweet,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
            padding='max_length'
        )

        return {
            'tweet': tweet,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }


class SarcasmRedditDataset(Dataset): #concat comments after tokenize

    def __init__(self, comments, parent_comments, targets, tokenizer, max_len):
        self.comments = comments
        self.parent_comments = parent_comments
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, item):
        comment = str(self.comments[item])
        parent_comment = str(self.parent_comments[item])
        target = self.targets[item]

        encoding_parent_comment = self.tokenizer.encode_plus(
            parent_comment,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
            padding='max_length'
        )

        encoding_comment = self.tokenizer.encode_plus(
            comment,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
            padding='max_length'
        )

        #, torch.tensor([102]) , torch.tensor([1])
        combined_input_ids = torch.cat([encoding_parent_comment['input_ids'].flatten(), encoding_comment['input_ids'].flatten()])
        combined_attention_mask = torch.cat([encoding_parent_comment['attention_mask'].flatten(), encoding_comment['attention_mask'].flatten()])

        return {
            'comment': comment,
            'parent_comment': parent_comment,
            'input_ids': combined_input_ids,
            'attention_mask': combined_attention_mask,
            'targets': torch.tensor(target, dtype=torch.long)
        }


class SarcasmRedditDataset_Exp1(Dataset): #concat comment to parent, leave parent with smaller max_len and no padding

    def __init__(self, comments, parent_comments, targets, tokenizer, max_len):
        self.comments = comments
        self.parent_comments = parent_comments
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, item):
        comment = str(self.comments[item])
        parent_comment = str(self.parent_comments[item])
        target = self.targets[item]

        encoding_parent_comment = self.tokenizer.encode_plus(
            parent_comment,
            add_special_tokens=True,
            max_length=(int(self.max_len/2)),
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        parent_input_ids = encoding_parent_comment['input_ids'].flatten()
        parent_attention_mask = encoding_parent_comment['attention_mask'].flatten()

        encoding_comment = self.tokenizer.encode_plus(
            comment,
            add_special_tokens=True,
            max_length=(int(self.max_len*1.5) - len(parent_input_ids)),
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
            padding='max_length'
        )
        comment_input_ids = encoding_comment['input_ids'].flatten()
        comment_attention_mask = encoding_comment['attention_mask'].flatten()

        combined_input_ids = torch.cat([parent_input_ids, torch.tensor([102]), comment_input_ids])
        combined_attention_mask = torch.cat([parent_attention_mask, torch.tensor([1]), comment_attention_mask])

        return {
            'comment': comment,
            'parent_comment': parent_comment,
            'input_ids': combined_input_ids,
            'attention_mask': combined_attention_mask,
            'targets': torch.tensor(target, dtype=torch.long)
        }


class SarcasmRedditDataset_V2(Dataset): #concat comments before tokenize

    def __init__(self, comments, parent_comments, targets, tokenizer, max_len):
        self.comments = comments
        self.parent_comments = parent_comments
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, item):
        comment = str(self.comments[item])
        parent_comment = str(self.parent_comments[item])
        target = self.targets[item]
        combined_comment = parent_comment + ' ' + comment
        encoding = self.tokenizer.encode_plus(
            combined_comment,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
            padding='max_length'
        )

        return {
            'comment': comment,
            'parent_comment': parent_comment,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }


class SarcasmRedditDataset_V3(Dataset): #use only main commnet, no parent

    def __init__(self, comments, targets, tokenizer, max_len):
        self.comments = comments
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, item):
        comment = str(self.comments[item])
        target = self.targets[item]
        encoding = self.tokenizer.encode_plus(
            comment,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
            padding='max_length'
        )

        return {
            'comment': comment,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }


class SarcasmRedditDuelBertDataset(Dataset):

    def __init__(self, comments, parent_comments, targets, tokenizer, max_len):
        self.comments = comments
        self.parent_comments = parent_comments
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, item):
        comment = str(self.comments[item])
        parent_comment = str(self.parent_comments[item])
        target = self.targets[item]

        encoding_comment = self.tokenizer.encode_plus(
            comment,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
            padding='max_length'
        )

        encoding_parent_comment = self.tokenizer.encode_plus(
            parent_comment,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
            padding='max_length'
        )

        return {
            'comment': comment,
            'parent_comment': parent_comment,
            'comment_input_ids': encoding_comment['input_ids'].flatten(),
            'comment_attention_mask': encoding_comment['attention_mask'].flatten(),
            'parent_input_ids': encoding_parent_comment['input_ids'].flatten(),
            'parent_attention_mask': encoding_parent_comment['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }