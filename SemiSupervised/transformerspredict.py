import re
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformerstransfer import Seq2Seq, Encoder, Decoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_SENT_LEN = 10
HIDDEN_SIZE = 128

sent = "This is a test sent."
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
#model = Net()
#model.load_state_dict(torch.load(PATH))
#model.eval()
encoder = torch.load('encoder_all.pt')
decoder = torch.load('decoder_all.pt')
seq_state_dict = torch.load('seq2seq.pt')
seq2seq = Seq2Seq(encoder, decoder, device)
seq2seq.load_state_dict(seq_state_dict)
seq2seq.eval()

x,y = torch.from_numpy(np.array([eng_sent], dtype=int)).to(device), torch.from_numpy(np.array([eng_sent], dtype=int)).to(device)
print(x.size(), y.size())
print(x)
pred = seq2seq(x,y)
print(pred)