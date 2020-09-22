import re
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

def accuracy(predictions, labels):
    classes = torch.argmax(predictions, dim=2)
    return torch.mean((classes == labels).float())

def main():
    """ The data """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #import torchtext
    #from torchtext.data.utils import get_tokenizer
    #TEXT = torchtext.data.Field(tokenize=get_tokenizer("basic_english"),
    #                            init_token='<sos>',
    #                            eos_token='<eos>',
    #                            lower=True)
    #train = torchtext.data.TabularDataset(
    #        path='../data/training.datahead', format='TSV', skip_header=False,
    #        fields=[
    #            ('source', TEXT),
    #            ('target', TEXT),
    #            ])
    #TEXT.build_vocab(train)
    #train_set, val_set = torch.utils.data.random_split(train, [5, 5])
    #
    #def batchify(data, bsz):
    #    data = TEXT.numericalize([data.source.text, data.target.text])
    #    # Divide the dataset into bsz parts.
    #    nbatch = data.size(0) // bsz
    #    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    #    data = data.narrow(0, 0, nbatch * bsz)
    #    # Evenly divide the data across the bsz batches.
    #    data = data.view(bsz, -1).t().contiguous()
    #    return data.to(device)
    #
    #batch_size = 20
    #eval_batch_size = 10
    #train_data = batchify(train_set, batch_size)
    #val_data = batchify(val_set, batch_size)
    from tqdm import tqdm
    from torch.utils.data.sampler import SubsetRandomSampler
    with open("../../data/training.data") as f:
        sentences = f.readlines()
    print(len(sentences))

    NUM_INSTANCES = 100000
    MAX_SENT_LEN = 10
    eng_sentences, deu_sentences = [], []
    eng_words, deu_words = set(), set()
    for i in tqdm(range(NUM_INSTANCES)):
        rand_idx = np.random.randint(len(sentences))
        # find only letters in sentences
        eng_sent, deu_sent = ["<sos>"], ["<sos>"]
        eng_sent += re.findall(r"\w+", sentences[rand_idx].split("\t")[0])
        deu_sent += re.findall(r"\w+", sentences[rand_idx].split("\t")[1])

        # change to lowercase
        eng_sent = [x.lower() for x in eng_sent]
        deu_sent = [x.lower() for x in deu_sent]
        eng_sent.append("<eos>")
        deu_sent.append("<eos>")

        if len(eng_sent) >= MAX_SENT_LEN:
            eng_sent = eng_sent[:MAX_SENT_LEN]
        else:
            for _ in range(MAX_SENT_LEN - len(eng_sent)):
                eng_sent.append("<pad>")

        if len(deu_sent) >= MAX_SENT_LEN:
            deu_sent = deu_sent[:MAX_SENT_LEN]
        else:
            for _ in range(MAX_SENT_LEN - len(deu_sent)):
                deu_sent.append("<pad>")

        # add parsed sentences
        eng_sentences.append(eng_sent)
        deu_sentences.append(deu_sent)

        # update unique words
        eng_words.update(eng_sent)
        deu_words.update(deu_sent)

    eng_words, deu_words = list(eng_words), list(deu_words)

    # encode each token into index
    for i in tqdm(range(len(eng_sentences))):
        eng_sentences[i] = [eng_words.index(x) for x in eng_sentences[i]]
        deu_sentences[i] = [deu_words.index(x) for x in deu_sentences[i]]

    ENG_VOCAB_SIZE = len(eng_words)
    DEU_VOCAB_SIZE = len(deu_words)
    NUM_EPOCHS = 100
    HIDDEN_SIZE = 128
    EMBEDDING_DIM = 30
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-2
    DEVICE = device

    np.random.seed(777)   # for reproducibility
    dataset = MTDataset(eng_sentences, deu_sentences)
    NUM_INSTANCES = len(dataset)
    TEST_RATIO = 0.3
    TEST_SIZE = int(NUM_INSTANCES * 0.3)
    batch_size = 32

    indices = list(range(NUM_INSTANCES))

    test_idx = np.random.choice(indices, size = TEST_SIZE, replace = False)
    train_idx = list(set(indices) - set(test_idx))
    train_sampler, test_sampler = SubsetRandomSampler(train_idx), SubsetRandomSampler(test_idx)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)


    encoder = Encoder(ENG_VOCAB_SIZE, HIDDEN_SIZE, EMBEDDING_DIM, DEVICE).to(DEVICE)
    decoder = Decoder(DEU_VOCAB_SIZE, HIDDEN_SIZE, EMBEDDING_DIM).to(DEVICE)
    seq2seq = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(seq2seq.parameters(), lr = LEARNING_RATE)

    loss_trace = []
    for epoch in tqdm(range(NUM_EPOCHS)):
        print('\n\n\nEpoch:', epoch)
        current_loss = 0
        running_accuracy = 0
        for i, (x, y) in enumerate(train_loader):
            x, y  = x.to(DEVICE), y.to(DEVICE)
            outputs = seq2seq(x, y)
            loss = criterion(outputs.resize(outputs.size(0) * outputs.size(1), outputs.size(-1)), y.resize(y.size(0) * y.size(1)))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            current_loss += loss.item()
            running_accuracy += accuracy(outputs, y)
        loss_trace.append(current_loss)
        print("Loss:", current_loss)
        print("Accuracy:", float(running_accuracy / len(train_loader)))

        with torch.no_grad():
            current_loss = 0
            running_accuracy = 0
            for i, (x, y) in enumerate(test_loader):
                x, y  = x.to(DEVICE), y.to(DEVICE)
                outputs = seq2seq(x, y)
                loss = criterion(outputs.resize(outputs.size(0) * outputs.size(1), outputs.size(-1)), y.resize(y.size(0) * y.size(1)))
                current_loss += loss.item()
                running_accuracy += accuracy(outputs, y)
            loss_trace.append(current_loss)
            print("Validation loss:", current_loss)
            print("Validation accuracy:", float(running_accuracy / len(train_loader)))

    torch.save(seq2seq.state_dict(), 'seq2seq.pt')
    torch.save(encoder.state_dict(), 'encoder.pt')
    torch.save(decoder.state_dict(), 'decoder.pt')
    torch.save(seq2seq, 'seq2seq_all.pt')
    torch.save(encoder, 'encoder_all.pt')
    torch.save(decoder, 'decoder_all.pt')

class MTDataset(torch.utils.data.Dataset):
    def __init__(self, eng_sentences, deu_sentences):
        # import and initialize dataset    
        self.source = np.array(eng_sentences, dtype = int)
        self.target = np.array(deu_sentences, dtype = int)

    def __getitem__(self, idx):
        # get item by index
        return self.source[idx], self.target[idx]

    def __len__(self):
        # returns length of data
        return len(self.source)

""" The model """
class Encoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, embedding_dim, device):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.device = device
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_size)

    def forward(self, x, h0):
        # x = (BATCH_SIZE, MAX_SENT_LEN) = (128, 10)
        x = self.embedding(x)
        x = x.permute(1, 0, 2)
        # x = (MAX_SENT_LEN, BATCH_SIZE, EMBEDDING_DIM) = (10, 128, 30)
        out, h0 = self.gru(x, h0)
        # out = (MAX_SENT_LEN, BATCH_SIZE, HIDDEN_SIZE) = (128, 10, 16)
        # h0 = (1, BATCH_SIZE, HIDDEN_SIZE) = (1, 128, 16)
        return out, h0


class Decoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, embedding_dim):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_size)
        self.dense = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim = 1)

    def forward(self, x, h0):
        # x = (BATCH_SIZE) = (128)
        x = self.embedding(x).unsqueeze(0)
        # x = (1, BATCH_SIZE, EMBEDDING_DIM) = (1, 128, 30)
        x, h0 = self.gru(x, h0)
        x = self.dense(x.squeeze(0))
        x = self.softmax(x)
        return x, h0


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, source, target, tf_ratio = .5):
        # target = (BATCH_SIZE, MAX_SENT_LEN) = (128, 10)
        # source = (BATCH_SIZE, MAX_SENT_LEN) = (128, 10)
        dec_outputs = torch.zeros(target.size(0), target.size(1), self.decoder.vocab_size).to(self.device)
        h0 = torch.zeros(1, source.size(0), self.encoder.hidden_size).to(self.device)

        _, h0 = self.encoder(source, h0)
        # dec_input = (BATCH_SIZE) = (128)
        dec_input = target[:, 0]

        for k in range(target.size(1)):
            # out = (BATCH_SIZE, VOCAB_SIZE) = (128, XXX)
            # h0 = (1, BATCH_SIZE, HIDDEN_SIZE) = (1, 128, 16)
            out, h0 = self.decoder(dec_input, h0)
            dec_outputs[:, k, :] = out
            dec_input = target[:, k]
            if np.random.choice([True, False], p = [tf_ratio, 1-tf_ratio]):
                dec_input = target[:, k]
            else:
                dec_input = out.argmax(1).detach()

        return dec_outputs

if __name__ == "__main__":
    main()
