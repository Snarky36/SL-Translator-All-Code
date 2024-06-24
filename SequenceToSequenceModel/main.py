import torch
import OneFileSeq2Seq
from OneFileSeq2Seq import blueScore
from torchtext.data.metrics import bleu_score

if __name__ == '__main__':
    # print(torch.cuda.current_device())
    # print(torch.cuda.is_available())
    #OneFileSeq2Seq.trainModel(10)
    english = OneFileSeq2Seq.transalate("Ein Mann mit einem orangefarbenen Hut, der etwas anstarrt.")
    print(english)
    print(blueScore())
