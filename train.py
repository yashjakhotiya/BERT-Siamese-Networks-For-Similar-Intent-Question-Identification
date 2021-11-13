# from preprocessing import preprocess, get_train_dataset, get_test_dataset
from preprocessing import preprocess
from transformers import BertTokenizer, BertModel
import torch
from torch import nn, optim
import copy
import random

text = ["Georgia Tech is among the best schools for computer science", "India performed really well in this world cup", "India played very good in this world cup", "Georgia Tech is a top 10 school for computer science"]
train_dataset = [[preprocess(text[0]), preprocess(text[1]), 0], [preprocess(text[0]), preprocess(text[2]), 0],
[preprocess(text[0]), preprocess(text[3]), 1], [preprocess(text[1]), preprocess(text[2]), 1],
[preprocess(text[1]), preprocess(text[3]), 0], [preprocess(text[2]), preprocess(text[3]), 0]]
test_dataset = copy.deepcopy(train_dataset)

tokenized_input_1 = [i[0] for i in train_dataset]
tokenized_input_2 = [i[1] for i in train_dataset]
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
encoded_input_1 = tokenizer(tokenized_input_1, return_tensors='pt', is_split_into_words=True, padding=True)
encoded_input_2 = tokenizer(tokenized_input_2, return_tensors='pt', is_split_into_words=True, padding=True)
train_Y = [i[2] for i in train_dataset]
num_classes = 2

class SimilarityModel(nn.Module):
    def __init__(self):
        super(SimilarityModel, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.feedforward_1 = nn.Linear(768*2, 256)
        self.relu = nn.ReLU()
        self.feedforward_2 = nn.Linear(256, 2)
        self.log_softmax = nn.LogSoftmax(dim=0)

    def forward(self, encoded_input_1, encoded_input_2, train=False):
        if train:
          self.train()
        else:
          self.eval()
        pooler_output_1 = self.bert(**encoded_input_1).pooler_output
        pooler_output_2 = self.bert(**encoded_input_2).pooler_output
        # print(pooler_output_1)
        concatenated_output = torch.cat([pooler_output_1, pooler_output_2])
        return self.log_softmax(self.feedforward_2(self.relu(self.feedforward_1(concatenated_output))))

def shuffle_sentences(input_1, input_2, labels):
    shuffled_input_1 = []
    shuffled_input_2 = []
    shuffled_labels      = []
    indices = list(range(len(input_1)))
    random.shuffle(indices)
    for i in indices:
        shuffled_input_1.append(input_1[i])
        shuffled_input_2.append(input_2[i])
        shuffled_labels.append(labels[i])
    return (shuffled_input_1, shuffled_input_2, shuffled_labels)

model = SimilarityModel()
optimizer = optim.Adam(model.parameters(), lr=0.1)
batch_size = len(tokenized_input_1)
for epoch in range(10):
    total_loss = 0.0
    (shuffled_input_1, shuffled_input_2, shuffled_labels) = shuffle_sentences(encoded_input_1, encoded_input_2, train_Y)
    for batch in range(0, len(tokenized_input_1), batch_size):
    #Randomly shuffle examples in each epoch
        input_1 = shuffled_input_1[batch:(batch + batch_size)]
        input_2 = shuffled_input_2[batch:(batch + batch_size)]
        labels = shuffled_labels[batch:(batch + batch_size)]
        labels_onehot = torch.zeros(len(labels), num_classes).cuda()
        for i in range(len(labels)):
            labels_onehot[i][labels[i]] = 1.0
        model.zero_grad()
        log_probs = model.forward(input_1.cuda(), input_2.cuda(), train=True)
        loss = 0
        for batch_idx in range(labels_onehot.shape[0]):
            loss_batch = torch.neg(log_probs[batch_idx]).dot(labels_onehot[batch_idx])
            loss += loss_batch
        loss /= labels_onehot.shape[0]
        loss.backward()
        optimizer.step()
        total_loss += loss

    print(f"loss on epoch {epoch} = {total_loss}")
