"""
Description taken from the official website of wenetspeech
(https://wenet-e2e.github.io/WenetSpeech/)

We release a 10000+ hours multi-domain transcribed Mandarin Speech Corpus
collected from YouTube and Podcast. Optical character recognition (OCR) and
automatic speech recognition (ASR) techniques are adopted to label each YouTube
and Podcast recording, respectively. To improve the quality of the corpus,
we use a novel end-to-end label error detection method to further validate and
filter the data.

See https://github.com/wenet-e2e/WenetSpeech for more details about WenetSpeech
"""

import json
import logging
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from tqdm.auto import tqdm

from lhotse import (
    compute_num_samples,
    fix_manifests,
    validate_recordings_and_supervisions,
)
from lhotse.audio import AudioSource, Recording, RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, add_durations



from denoiser import pretrained
from denoiser.dsp import convert_audio
import torchaudio
import torch
import soundfile
import os
import librosa
import random
# from tqdm import tqdm
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from datetime import datetime, timedelta

WETNET_SPEECH_PARTS = ("L", "M", "S", "DEV", "TEST_NET", "TEST_MEETING")


def prepare_wenet_speech(
    corpus_dir: Pathlike,
    resample_corpus_dir: Pathlike,
    dataset_parts: Union[str, Sequence[str]] = "all",
    output_dir: Optional[Pathlike] = None,
    num_jobs: int = 1,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions
    :param corpus_dir: Pathlike, the path of the data dir.
    :param dataset_parts: Which parts of dataset to prepare, all for all the
                          parts.
    :param output_dir: Pathlike, the path where to write the manifests.
    :num_jobs Number of workers to extract manifests.
    :return: a Dict whose key is the dataset part, and the value is Dicts with
             the keys 'recordings' and 'supervisions'.
    """
    corpus_dir = Path(corpus_dir)
    resample_corpus_dir = Path(resample_corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    subsets = WETNET_SPEECH_PARTS if "all" in dataset_parts else dataset_parts

    manifests = defaultdict(dict)
    for sub in subsets:
        if sub not in WETNET_SPEECH_PARTS:
            raise ValueError(f"No such part of dataset in WenetSpeech : {sub}")
        manifests[sub] = {"recordings": [], "supervisions": []}

    raw_manifests_path = corpus_dir / "WenetSpeech.json"
    assert raw_manifests_path.is_file(), f"No such file : {raw_manifests_path}"
    logging.info(f"Loading raw manifests from : {raw_manifests_path}")
    print("loadding json")
    raw_manifests = json.load(open(raw_manifests_path, "r", encoding="utf8"))

    print("loadding data")
    with ProcessPoolExecutor(num_jobs) as ex:
        for recording, segments in tqdm(
            ex.map(
                parse_utterance,
                raw_manifests["audios"],
                repeat(corpus_dir),
                repeat(resample_corpus_dir),
                repeat(subsets),
            ),
            desc="Processing WenetSpeech JSON entries",
        ):
            for part in segments:
                manifests[part]["recordings"].append(recording)
                manifests[part]["supervisions"].extend(segments[part])

    print("sss")
    for sub in subsets:
        recordings, supervisions = fix_manifests(
            recordings=RecordingSet.from_recordings(manifests[sub]["recordings"]),
            supervisions=SupervisionSet.from_segments(manifests[sub]["supervisions"]),
        )
        validate_recordings_and_supervisions(
            recordings=recordings, supervisions=supervisions
        )

        if output_dir is not None:
            supervisions.to_file(
                output_dir / f"wenetspeech_supervisions_{sub}.jsonl.gz"
            )
            recordings.to_file(output_dir / f"wenetspeech_recordings_{sub}.jsonl.gz")

        manifests[sub] = {
            "recordings": recordings,
            "supervisions": supervisions,
        }

    return manifests


denoise_model = None

def get_denoise_model(cpu=True):
    global denoise_model
    if not denoise_model:
        if cpu:
            denoise_model = pretrained.dns64().cpu()
        else:
            denoise_model = pretrained.dns64().cuda()
    return denoise_model

def denoisy(input, output, tsr, cpu=True):
    wav, sr = torchaudio.load(input)
    try:
        wav, sr = denoisy_np(wav, sr, tsr, cpu)
        soundfile.write(output, wav, sr)
    except Exception as es:
        print(es)
        return False

    return True
    
    

def denoisy_np(wav, sr, tsr, cpu=True):
    model = get_denoise_model(cpu)
    if cpu:
        wav=wav.cpu()
    else:
        wav=wav.cuda()
    wav = convert_audio(wav, sr, model.sample_rate, model.chin)
    with torch.no_grad():
        denoised = model(wav[None])[0]
        np_wav = denoised.data.cpu().numpy()
        wav = librosa.resample(np_wav.reshape(-1), model.sample_rate, tsr)
        return wav, tsr



def resample(f, output_file):
    denoisy(f, output_file, 24000, False)
        

def parse_utterance(
    audio: Any, root_path: Path, resample_corpus_dir:Path, subsets: Sequence
) -> Tuple[Recording, Dict[str, List[SupervisionSegment]]]:
    sampling_rate = 24000
    # 转成wav并去燥
    f = root_path / audio["path"]
    output_file = resample_corpus_dir / audio["path"]
    
    if not output_file.exists():
        # 获取上级目录路径
        parent_directory = output_file.parent

        # 判断上级目录是否存在，如果不存在则创建它
        if not parent_directory.exists():
            parent_directory.mkdir(parents=True) 

        denoisy(f, output_file, sampling_rate, False)
    
    recording = Recording(
        id=audio["aid"],
        sources=[
            AudioSource(
                type="file",
                channels=[0],
                source=str(root_path / audio["path"]),
            )
        ],
        num_samples=compute_num_samples(
            duration=audio["duration"], sampling_rate=sampling_rate
        ),
        sampling_rate=sampling_rate,
        duration=audio["duration"],
    )
    segments = defaultdict(dict)
    for sub in subsets:
        segments[sub] = []
    for seg in audio["segments"]:
        segment = SupervisionSegment(
            id=seg["sid"],
            recording_id=audio["aid"],
            start=seg["begin_time"],
            duration=add_durations(
                seg["end_time"], -seg["begin_time"], sampling_rate=sampling_rate
            ),
            language="Chinese",
            text=seg["text"].strip(),
        )
        for sub in seg["subsets"]:
            if sub in subsets:
                segments[sub].append(segment)
    return recording, segments


if __name__ == "__main__":
    corpus_dir = Path("egs/wenet_speech/download/wenet_speech")
    output_dir = Path("egs/wenet_speech/data/manifests")
    resample_corpus_dir = Path("egs/wenet_speech/data/wav")
    prepare_wenet_speech(corpus_dir, resample_corpus_dir, "all", output_dir, num_jobs=1)
