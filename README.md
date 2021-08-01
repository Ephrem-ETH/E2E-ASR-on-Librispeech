# E2E ASR experiments on Librispeech (1000 hours) 
This experiment is conducted on 1000 hours of librispeech data for the task of E2E ASR using an intermediate character level representation and Connectionist Temporal Classifications(CTC) loss function.

We used a prefix beam search decoding strategy to decode the model output and return word transcription. 




## File description
* model.py: rnnt joint model
* train_ctc.py: ctc acoustic model training script
* eval.py: rnnt & ctc decode
* DataLoader.py: Feature extraction (MFCC) 





 



### Train CTC acoustic model
```
python train_ctc.py --lr 1e-3 --bi --dropout 0.5 --out exp/ctc_bi_lr1e-3 --schedule
```
##### Results
###### Loss curve
<img src="img/loss.png"/>


### Decode 
```
python eval.py <path to best model> [--ctc] --bi
```







## Requirements
* Python 3.6
* PyTorch >= 0.4
* numpy 1.14



