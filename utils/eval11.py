import argparse
import logging
import math
import os
import time

import editdistance
import torch
from torch.utils import data
import torchaudio
from torch.autograd import Variable
import torch.nn.functional as F
import torchaudio.transforms as T
import numpy as np
import fastwer
from model import Transducer, RNNModel
from DataLoader2 import  TokenAcc, index_map

parser = argparse.ArgumentParser(description='MXNet Autograd RNN/LSTM Acoustic Model on TIMIT.')
parser.add_argument('model', help='trained model filename')
parser.add_argument('--beam', type=int, default=0, help='apply beam search, beam width')
parser.add_argument('--ctc', default=False, action='store_true', help='decode CTC acoustic model')
parser.add_argument('--bi', default=False, action='store_true', help='bidirectional LSTM')
parser.add_argument('--dataset', default='test', help='decoding data set')
parser.add_argument('--out', type=str, default='', help='decoded result output dir')
args = parser.parse_args()

logdir = args.out if args.out else os.path.dirname(args.model) + '/decode.log'
# if args.out: os.makedirs(args.out, exist_ok=True)
logging.basicConfig(format='%(asctime)s: %(message)s', datefmt="%H:%M:%S", filename=logdir, level=logging.INFO)

# Parameters for the mfcc transformer
sample_rate = 16000
n_fft = 1024
win_length = 320 #20ms
hop_length = 160 #10ms 
n_mels = 80
n_mfcc = 80  #23
mfcc_transform = T.MFCC(sample_rate=sample_rate, n_mfcc=n_mfcc,
 melkwargs={'n_fft': n_fft, 'n_mels': n_mels, 'win_length': win_length, 'hop_length': hop_length})

# Load model
Model = RNNModel if args.ctc else Transducer
model = Model(80, 29, 250, 3, bidirectional=args.bi)
state = torch.load(args.model, map_location='cpu')
# model.load_state_dict(torch.load(args.model, map_location='cpu'))
model.load_state_dict(state['state_dict'])

#use_gpu = torch.cuda.is_available()
use_gpu = True
if use_gpu:
    model.cuda()
def data_processing(data, data_type="test"):
    mfccs = []
    labels = []
    input_lengths = []
    label_lengths = []
    fileid_audio = " "
    for (waveform, _, utterance, speaker_id, chapter_id, utterance_id ) in data:
        if data_type == 'test':
            mfcc = mfcc_transform(waveform).squeeze(0).transpose(0, 1)
            label = utterance.lower().split()
            fileid_audio = str(speaker_id) + "-" + str(chapter_id) + "-" + str(utterance_id) 

    return mfcc, label, fileid_audio
# data set
# feat = 'ark:copy-feats scp:mydata/data/{}/feats.scp ark:- | apply-cmvn --utt2spk=ark:mydata/data/{}/utt2spk scp:mydata/data/{}/cmvn.scp ark:- ark:- |\
#  add-deltas --delta-order=2 ark:- ark:- | nnet-forward mydata/data/final.feature_transform ark:- ark:- |'.format(args.dataset, args.dataset, args.dataset)
# with open('mydata/data/'+args.dataset+'/sample_text', 'r') as f:
    # label = {}
    # for line in f:
    #     line = line.split()
    #     label[line[0]] = line[1:]
test_url = "dev-clean"
if not os.path.isdir("./data"):
    os.makedirs("./data")

test_dataset = torchaudio.datasets.LIBRISPEECH("./data", url=test_url, download=False)

test_loader = data.DataLoader(dataset=test_dataset,
                                batch_size=1,
                                shuffle=False,
                                collate_fn=lambda x: data_processing(x, 'test'))
# Phone map
with open('conf/word.200000.map', 'r') as f:
    pmap = {}
    for line in f:
        line = line.lower().split()
        # line = line.split()
        if len(line) < 3: pmap[line[0]] = line[0]
        else: pmap[line[0]] = line[2]
# print(pmap)

def distance(y, t, blank='<eps>'):
    def remap(y, blank):
        prev = blank
        seq = []
        for i in y:
            if i != blank and i != prev: seq.append(i)
            prev = i
        return seq
    y = remap(y, blank)
    t = remap(t, blank)
    # print(y)
    return y, t, editdistance.eval(y, t)
# calculate sentence level character error rate(CER)
def calculate_cer(y,t):
    y = list(' '.join(y))
    t = list(' '.join(t))
    char_t_len = len(t)
    # print(f" {t} len: {char_t_len}")
    return char_t_len, editdistance.eval(y, t)



model.eval()
def decode():
    logging.info('Decoding transduction model:')
    total_word, total_char, total_cer, total_wer, fast_cer, fast_wer = 0,0,0,0,0,0
    for i, (xs, label,k) in enumerate(test_loader):
       
        xs = Variable(torch.FloatTensor(xs[None, ...]), volatile=True)
        if use_gpu:
            xs = xs.cuda()
        if args.beam > 0:
            y,nll = model.beam_search(xs, args.beam)
            # print("beam {}".format(y))
        else:
            y, nll = model.greedy_decode(xs)
        # y = [pmap.get(i) for i in y if pmap.get(i)]
        # t = [pmap.get(i) for i in label if pmap.get(i)]
        # y, t, wer = distance(y, t)
        # total_wer += wer; total_word += len(t)
        #Compute CER
        # sen_len, cer= calculate_cer(y,t)
        # total_cer += cer; total_char += sen_len;
        y = ' '.join(y)
        with open("raw_output.log", "a") as fw:
            fw.write(f'[{k}]:  {y}\n')
    with open("raw_output.log","a") as f:
        fw.write("Done!!")
    #     logging.info('[{}]: {}'.format(k, ' '.join(t)))
    #     logging.info('[{}]: {}\nlog-likelihood: {:.2f}\n'.format(k, ' '.join(y),nll))
    # logging.info('{} set {} CER {:.2f}% and WER {:.2f}%\n'.format(
    #     args.dataset.capitalize(), 'CTC' if args.ctc else 'Transducer', 100*total_cer/total_char,100*total_wer/total_word))

decode()
