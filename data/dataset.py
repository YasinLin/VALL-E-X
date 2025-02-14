# Copyright      2023                           (authors: Feiteng Li)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
modified from lhoste.dataset.speech_synthesis.py
"""

import torch
import math
import h5py
from tokenizers import Tokenizer
import numpy as np
from typing import Callable, Dict, List, Sequence, Union
import torch
from lhotse import validate
from lhotse.cut import CutSet
from lhotse.dataset.collation import collate_audio
from lhotse.dataset.input_strategies import BatchIO, PrecomputedFeatures
from lhotse.utils import ifnone

from vallex.data.collation import TextTokenCollater,get_text_token_collater
from vallex.utils.g2p import PhonemeBpeTokenizer
from tqdm import tqdm

_pad        = '_'
_punctuation = ',.!?-~…'
_letters = 'NQabdefghijklmnopstuvwxyzɑæʃʑçɯɪɔɛɹðəɫɥɸʊɾʒθβŋɦ⁼ʰ`^#*=ˈˌ→↓↑ '
symbols = [_pad] + list(_punctuation) + list(_letters)

language_name_dict = {
    'Englise': 'en',
    'Chinese': 'zh',
    'ja': 'ja',
}
language_dict = {
    'en': 0,
    'zh': 1,
    'ja': 2,
}
lang2token = {
    'zh': "[ZH]",
    'ja': "[JA]",
    "en": "[EN]",
    'mix': "",
}


text_tokenizer = PhonemeBpeTokenizer(tokenizer_path="./vallex/utils/g2p/bpe_69.json")
def seq2phone(tokens: Union[List, np.ndarray]):
    """
    Convert tokenized phoneme ID sequence back to phoneme string
    :param tokens: phoneme tokens
    :return: recovered phoneme sequence
    """
    phones = "".join([symbols[i] for i in tokens])
    return phones



class SpeechSynthesisDataset(torch.utils.data.Dataset):
    """
    The PyTorch Dataset for the speech synthesis(e.g. TTS) task.
    Each item in this dataset is a dict of:

    .. code-block::

        {
            'audio': (B x NumSamples) float tensor
            'audio_lens': (B, ) int tensor
            'text': str
            'audio_features': (B x NumFrames x NumFeatures) float tensor
            'audio_features_lens': (B, ) int tensor
            'text_tokens': (B x NumTextTokens) long tensor
            'text_tokens_lens': (B, ) int tensor
        }
    """

    def __init__(
        self,
        text_token_collater: TextTokenCollater,
        cut_transforms: List[Callable[[CutSet], CutSet]] = None,
        feature_input_strategy: BatchIO = PrecomputedFeatures(),
        feature_transforms: Union[Sequence[Callable], Callable] = None,
    ) -> None:
        super().__init__()

        self.text_token_collater = text_token_collater
        self.cut_transforms = ifnone(cut_transforms, [])
        self.feature_input_strategy = feature_input_strategy

        if feature_transforms is None:
            feature_transforms = []
        elif not isinstance(feature_transforms, Sequence):
            feature_transforms = [feature_transforms]

        assert all(
            isinstance(transform, Callable) for transform in feature_transforms
        ), "Feature transforms must be Callable"
        self.feature_transforms = feature_transforms

    def __getitem__(self, cuts: CutSet) -> Dict[str, torch.Tensor]:
        global text_tokenizer
        validate_for_tts(cuts)

        for transform in self.cut_transforms:
            cuts = transform(cuts)

        if False:  # not used
            audio, audio_lens = collate_audio(cuts)
        else:  # for sharing tokenized features in different machines
            audio, audio_lens = None, None

        audio_features, audio_features_lens = self.feature_input_strategy(cuts)

        for transform in self.feature_transforms:
            audio_features = transform(audio_features)

        datas = []

        # language_id = []
        token_infos = []
        phone_tokens_all = []
        for cut in cuts:
            phone_tokens = cut.supervisions[0].custom["tokens"]['text']
            phone_tokens_all.append(phone_tokens)
            
            # text = cut.supervisions[0].text
            # language = language_name_dict[cut.supervisions[0].language]
            # lang_token = lang2token[language]
            # text = lang_token + text + lang_token
            # phone_tokens, langs = text_tokenizer.tokenize(text=f"_{text}".strip())
            # phone_tokens_all.append(phone_tokens)
            
            
            # t_tokens, t_tokens_lens = self.text_token_collater(
            #     [phone_tokens]
            # )
            # language_id += [language_dict[lang] for lang in langs]
            # token_infos = (t_tokens, t_tokens_lens)

        text_tokens, text_tokens_lens = self.text_token_collater(
            phone_tokens_all
        )
        # for (t_tokens, t_tokens_lens) in token_infos:
        #     if text_tokens != None:
        #         text_tokens += t_tokens
        #     else:
        #         text_tokens = t_tokens
        #     if text_tokens_lens != None:
        #         text_tokens_lens += t_tokens_lens
        #     else:
        #         text_tokens_lens = t_tokens_lens

        language_id = np.array([language_dict[language_name_dict[cut.supervisions[0].language]] for cut in cuts])
        return {
            "utt_id": [cut.id for cut in cuts],
            "text": [cut.supervisions[0].text for cut in cuts],
            "audio": audio,
            "audio_lens": audio_lens,
            "audio_features": audio_features,
            "audio_features_lens": audio_features_lens,
            "text_tokens": text_tokens,
            "text_tokens_lens": text_tokens_lens,
            "language": torch.LongTensor(language_id),
        }


def validate_for_tts(cuts: CutSet) -> None:
    validate(cuts)
    for cut in cuts:
        assert (
            len(cut.supervisions) == 1
        ), "Only the Cuts with single supervision are supported."



class DynamicBatchSampler(torch.utils.data.Sampler):
    def __init__(self, sampler, num_tokens_fn, num_buckets=100, min_size=0, max_size=1000,
                 max_tokens=None, max_sentences=None, drop_last=False):
        """

        :param sampler:
        :param num_tokens_fn: 根据idx返回样本的长度的函数
        :param num_buckets: 利用桶原理将相似长度的样本放在一个batchsize中，桶的数量
        :param min_size: 最小长度的样本， 小于这个值的样本会被过滤掉。 依据这个值来创建样桶
        :param max_size: 最大长度的样本
        :param max_sentences: batch_size, 但是这里可以通过max_sentences 和 max_tokens 共同控制最终的大小
        """
        super(DynamicBatchSampler, self).__init__(sampler)
        self.sampler = sampler
        self.num_tokens_fn = num_tokens_fn
        self.num_buckets = num_buckets

        self.min_size = min_size
        self.max_size = max_size

        assert max_size <= max_tokens, "max_size should be smaller than max tokens"
        assert max_tokens is not None or max_sentences is not None, \
            "max_tokens and max_sentences should not be null at the same time, please specify one parameter at least"
        self.max_tokens = max_tokens if max_tokens is not None else float('Inf')
        self.max_sentences = max_sentences if max_sentences is not None else float('Inf')
        self.drop_last = drop_last

    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)
    def is_batch_full(self, num_tokens, batch):
        if len(batch) == 0:
            return False
        if len(batch) == self.max_sentences:
            return True
        if num_tokens > self.max_tokens:
            return True
        return False

    def __iter__(self):
        buckets = [[] for _ in range(self.num_buckets)]
        sample_len = [0] * self.num_buckets

        for idx in self.sampler:
            idx_length = self.num_tokens_fn(idx)
            if not (self.min_size <= idx_length <= self.max_size):
                print("sentence at index {} of size {} exceeds max_tokens, the sentence is ignored".format(idx, idx_length))
                continue

            index_buckets = math.floor((idx_length - self.min_size) / (self.max_size - self.min_size + 1)
                                       * self.num_buckets)
            sample_len[index_buckets] = max(sample_len[index_buckets], idx_length)

            num_tokens = (len(buckets[index_buckets]) + 1) * sample_len[index_buckets]
            if self.is_batch_full(num_tokens, buckets[index_buckets]):
                # yield this batch
                yield buckets[index_buckets]
                buckets[index_buckets] = []
                sample_len[index_buckets] = 0

            buckets[index_buckets].append(idx)

        # process left-over
        leftover_batch = []
        leftover_sample_len = 0
        leftover = [idx for bucket in buckets for idx in bucket]
        for idx in leftover:
            idx_length = self.num_tokens_fn(idx)
            leftover_sample_len = max(leftover_sample_len, idx_length)
            num_tokens = (len(leftover_batch) + 1) * leftover_sample_len
            if self.is_batch_full(num_tokens, leftover_batch):
                yield leftover_batch
                leftover_batch = []
                leftover_sample_len = 0
            leftover_batch.append(idx)

        if len(leftover_batch) > 0 and not self.drop_last:
            yield leftover_batch

    def __len__(self):
        # we do not know the exactly batch size, so do not call len(dataloader)
        pass


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, h5_path, ann_path, tokenizer_path):
        self.h5_path = h5_path
        with open(ann_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        ls = [l.split("|") for l in lines]
        ls_T = list(zip(*ls))
        del ls_T[-1]
        self.h5_paths, self.durations, self.langs, self.texts = \
            list(ls_T[0]), list(ls_T[1]), list(ls_T[2]), list(ls_T[3])
        self.durations = [float(dur) for dur in self.durations]
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

        self._archive = None

    def __len__(self):
        return len(self.h5_paths)

    def get_dur(self, idx):
        return self.durations[idx]

    @property
    def archive(self):
        if self._archive is None:  # lazy loading here!
            self._archive = h5py.File(self.h5_path, "r")
        return self._archive
    def __getitem__(self, idx):
        archive = self.archive
        h5_path = self.h5_paths[idx]
        sub = archive[h5_path]
        audio_tokens = sub['audio'][()]
        phone_tokens = sub['text'][()]
        dur = self.durations[idx]
        lang = self.langs[idx]
        text = self.texts[idx]
        # tokenization should be done within dataloader
        phones = seq2phone(phone_tokens)
        phones = phones.replace(" ", "_")
        if not len(phones):
            cptpho_tokens = self.tokenizer.encode(text).ids
        else:
            cptpho_tokens = self.tokenizer.encode(phones).ids
        assert len(cptpho_tokens)
        return {
            'utt_id': h5_path,
            'text': text,
            'audio': None,
            'audio_lens': None,
            'audio_features': audio_tokens,
            'audio_features_lens': len(audio_tokens.T),
            'text_tokens': np.array(cptpho_tokens),
            'text_tokens_lens': len(cptpho_tokens),
            'language': language_dict[lang],
        }

def collate(batch):
    utt_id_s = [b['utt_id'] for b in batch]
    text_s = [b['text'] for b in batch]

    audio_s = [b['audio'] for b in batch]
    audio_lens_s = [b['audio_lens'] for b in batch]

    audio_features_lens_s = [b['audio_features_lens'] for b in batch]
    # create an empty tensor with maximum audio feature length
    audio_features_s = torch.zeros([len(batch), max(audio_features_lens_s), 8], dtype=torch.int64) - 1 # audio pad with -1

    text_tokens_lens_s = [b['text_tokens_lens'] for b in batch]
    # create an empty tensor with maximum text tokens length
    text_tokens_s = torch.zeros([len(batch), max(text_tokens_lens_s)], dtype=torch.int64) + 3 # [PAD] token id 3

    language_s = [b['language'] for b in batch]

    for i, b in enumerate(batch):
        audio_features = b['audio_features']
        audio_features_lens = b['audio_features_lens']
        audio_features_s[i, :audio_features_lens, :] = torch.LongTensor(audio_features.T)

        text_tokens = b['text_tokens']
        text_tokens_lens = b['text_tokens_lens']
        text_tokens_s[i, :text_tokens_lens] = torch.LongTensor(text_tokens)

    batch = {
        'utt_id': utt_id_s,
        'text': text_s,
        'audio': audio_s,
        'audio_lens': audio_lens_s,
        'audio_features': audio_features_s,
        'audio_features_lens': torch.LongTensor(np.array(audio_features_lens_s)),
        'text_tokens': text_tokens_s,
        'text_tokens_lens': torch.LongTensor(np.array(text_tokens_lens_s)),
        'languages': torch.LongTensor(np.array(language_s)),
    }
    return batch

def create_dataloader(data_dir="/root/valle/egs/mix", n_gpus=1, rank=0, num_workers=0, num_buckets=10, max_duration=120):
    train_dataset = AudioDataset(h5_path=f"{data_dir}/audio_sum.hdf5",
                                 ann_path=f"{data_dir}/audio_ann_sum.txt",
                                 tokenizer_path=f"{data_dir}/bpe_69.json")
    ran_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=n_gpus,
            rank=rank,
            shuffle=True,
        )
    dynamic_sampler = DynamicBatchSampler(ran_sampler, train_dataset.get_dur, num_buckets=num_buckets, max_size=20,
                                          max_tokens=max_duration)


    train_loader = torch.utils.data.DataLoader(train_dataset, num_workers=num_workers, collate_fn=collate,
                                               batch_sampler=dynamic_sampler)

    return train_loader
