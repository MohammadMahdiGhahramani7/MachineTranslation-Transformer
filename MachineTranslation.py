import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from my_transformer import Transformer
from my_train import train
import warnings
warnings.filterwarnings("ignore")

English_sens = ['i am fine', 'you are fine', 'he is fine', 'they are fine']
Spanish_sens = ['<st> yo estoy bien <end>', '<st> tu eres bien <end>', '<st> el es bien <end>', '<st> ellos son bien <end>']
                
def create_tokens(tokenizer, dataset):
  for sample in dataset:
    yield tokenizer(sample)

tokenizer_src = get_tokenizer('basic_english', language='en')
tokenizer_trg = get_tokenizer('toktok', language='es')

vocab_src = build_vocab_from_iterator(create_tokens(tokenizer_src, English_sens), specials=["<oov>", "<sos>"])
vocab_src.set_default_index(vocab_src["<oov>"])

vocab_trg = build_vocab_from_iterator(create_tokens(tokenizer_trg, Spanish_sens), specials=["<oov>", "<sos>"])
vocab_trg.set_default_index(vocab_trg["<oov>"])

idx_to_word_src = {vocab_src[w]:w for w in vocab_src.get_itos()}
idx_to_word_trg = {vocab_trg[w]:w for w in vocab_trg.get_itos()}

text_pipeline_src = lambda x: vocab_src(tokenizer_src(x))
text_pipeline_trg = lambda x: vocab_trg(tokenizer_trg(x))

def sent_padding(sent_vec, maxlen):
  sent_vec = torch.tensor(sent_vec)
  maxlen -= len(sent_vec)
  return F.pad(sent_vec, (0, maxlen))
  
class MyDataset(Dataset):

  def __init__(self, SRC, TRG, seq_len_src, seq_len_trg, device):
    self.SRC = SRC
    self.TRG = TRG
    self.seq_len_src = seq_len_src
    self.seq_len_trg = seq_len_trg
    self.device = device

  def __len__(self):
    return len(self.SRC)
  
  def __getitem__(self, idx):
    src, trg = self.SRC[idx], self.TRG[idx]

    src = sent_padding(text_pipeline_src(src), maxlen=self.seq_len_src)
    trg = sent_padding(text_pipeline_trg(trg), maxlen=self.seq_len_trg)

    return src.to(self.device), trg.to(self.device)
    
seq_len_src = 8
seq_len_trg_PRIME = 6
seq_len_trg = seq_len_trg_PRIME - 1

batch_size = 2
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataloader = DataLoader(MyDataset(English_sens, Spanish_sens, seq_len_src, seq_len_trg_PRIME, device), batch_size=batch_size)

src_vocab = len(vocab_src)
trg_vocab = len(vocab_trg)
d_model = 32
N = 1
heads = 2
max_seq_len = max(seq_len_src, seq_len_trg)


model = Transformer(src_vocab, trg_vocab, d_model, N, heads, max_seq_len).to(device)

epochs = 1000
print_step = 100
lr = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

loss = train(model, optimizer, dataloader, epochs, print_step)

@torch.no_grad()
def translate(sentence, device):

  sen_SRC = sent_padding(text_pipeline_src(sentence), maxlen=seq_len_src).unsqueeze(0).to(device)

  sen_TRG = '<st>'

  while '<end>' not in sen_TRG:

    length = len(sen_TRG.split())

    trg_input = sent_padding(text_pipeline_trg(sen_TRG), maxlen=seq_len_trg).unsqueeze(0)[:, :-1].to(device)

    preds = model(sen_SRC, trg_input, src_mask=None, trg_mask=None).squeeze()

    next_word_idx = torch.argmax(preds, dim=-1)[length-1] #IMPORTANT
    
    sen_TRG += (' ' + idx_to_word_trg[next_word_idx.item()])

  return sen_TRG


sentence = 'they are fine'
print(sentence)
print(translate(sentence, device))
