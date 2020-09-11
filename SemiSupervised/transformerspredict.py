import re
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformerstransfer import Seq2Seq, Encoder, Decoder

def get_sentence_tensor(sent):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MAX_SENT_LEN = 10
    eng_sent= ["<sos>"]
    eng_sent = re.findall(r"\w+", sent)
    eng_words = set()
    eng_sent = [x.lower() for x in eng_sent]
    eng_sent.append("<eos>")
    if len(eng_sent) >= MAX_SENT_LEN:
        eng_sent = eng_sent[:MAX_SENT_LEN]
    else:
        for _ in range(MAX_SENT_LEN - len(eng_sent)):
            eng_sent.append("<pad>")

    eng_words.update(eng_sent)
    eng_words = list(eng_words)
    eng_sent = [eng_words.index(x) for x in eng_sent]
    x,y = torch.from_numpy(np.array([eng_sent], dtype=int)).to(device), torch.from_numpy(np.array([eng_sent], dtype=int)).to(device)
    return x, y


def get_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    HIDDEN_SIZE = 128

    encoder = torch.load('encoder_all.pt')
    decoder = torch.load('decoder_all.pt')
    seq_state_dict = torch.load('seq2seq.pt')
    seq2seq = Seq2Seq(encoder, decoder, device)
    seq2seq.load_state_dict(seq_state_dict)
    seq2seq.eval()
    return seq2seq

def predict(sent):
    model = get_model()
    x,y = get_sentence_tensor(sent)
    pred = model(x,y)
    return pred

if __name__ == "__main__":
    print(predict("This is a test sentence."))