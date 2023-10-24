"""
About the Aishell corpus
Aishell is an open-source Chinese Mandarin speech corpus published by Beijing Shell Shell Technology Co.,Ltd.
publicly available on https://www.openslr.org/33
"""

import logging
import os
import shutil
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Union

from tqdm.auto import tqdm as tqdmauto

from lhotse import validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.qa import fix_manifests
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, resumable_download, safe_extract

import random
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import torchaudio
import torch
import librosa
import whisper
import soundfile
from denoiser import pretrained
from denoiser.dsp import convert_audio
denoise_model = None

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda", 0)


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
        wav = librosa.resample(np_wav.reshape(-1), orig_sr = model.sample_rate, target_sr = tsr)
        return wav, tsr

def resample(f, output_file):
    denoisy(f, output_file, 24000, False)

def text_normalize(line: str):
    """
    Modified from https://github.com/wenet-e2e/wenet/blob/main/examples/multi_cn/s0/local/aishell_data_prep.sh#L54
    sed 's/ａ/a/g' | sed 's/ｂ/b/g' |\
    sed 's/ｃ/c/g' | sed 's/ｋ/k/g' |\
    sed 's/ｔ/t/g' > $dir/transcripts.t

    """
    line = line.replace("ａ", "a")
    line = line.replace("ｂ", "b")
    line = line.replace("ｃ", "c")
    line = line.replace("ｋ", "k")
    line = line.replace("ｔ", "t")
    line = line.upper()
    return line

def get_sub_dir_names(directory):
    subdirnames = []
    for name in os.listdir(directory):
        path = os.path.join(directory, name)
        if os.path.isdir(path):
            subdirnames.append(name)
    return subdirnames




def process_with_progress(func, data_list, *args, **kwargs):
    pregress_total = len(data_list)
    progress_bar = tqdm(total=pregress_total)  # 初始化进度条
    results = []
    max_workers = 5
    if "max_workers" in kwargs:
        max_workers = kwargs.pop("max_workers")

    listen_progress = None
    if "listen_progress" in kwargs:
        listen_progress = kwargs.pop("listen_progress")

    future_time = None  # 加上 10 秒

    def time_condition_with_process(pregress, total, td):
        nonlocal future_time
        if not future_time:
            future_time = datetime.now() + td
        if datetime.now() > future_time or pregress >= total:
            future_time = datetime.now() + td
            return True
        return False

    condition_info_with_process = None
    if "time_condition_kwargs" in kwargs:
        condition_info_with_process = {
            "fun": time_condition_with_process,
            "kwargs": kwargs.pop("time_condition_kwargs")
        }

    def wrapped_condition_with_process(pregress, total, index, data):
        return pregress >= total or not condition_info_with_process \
            or condition_info_with_process['fun'](
                pregress, total,
                **(condition_info_with_process['kwargs'] if 'kwargs' in condition_info_with_process else {}))

    def wrapped_listen_progress(pregress, total, index, data):
        nonlocal listen_progress
        if listen_progress and wrapped_condition_with_process(pregress, total, index, data):
            listen_progress(pregress, total, index, data)

    def indexed(iterable):
        # 将元素与其索引打包作为元组 (index, element)
        return zip(range(len(iterable)), iterable)

    # 包装处理函数，添加额外的参数
    def wrapped_func(data_index_tuple):
        index, data = data_index_tuple
        kwdict = kwargs
        kwdict["index"] = index
        return index, func(data, *args, **kwdict)

    completed_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:  # 创建线程池
        futures = [executor.submit(wrapped_func, data) for data in indexed(data_list)]
        for future in as_completed(futures):
            index, result = future.result()
            results.append({"result": result, "index": index})
            progress_bar.update(1)  # 每完成一个任务，更新进度条
            completed_count += 1
            wrapped_listen_progress(completed_count, pregress_total, index, result)

    progress_bar.close()  # 关闭进度条
    return results


def gen_customer(data, corpus_dir: Pathlike, index=None, output_dir: Optional[Pathlike] = None, rs=False, spk=None):
    filepath = corpus_dir  / data['filepath'][data['filepath'].index("audio/formated"):]
    data['filepath'] = filepath
    if rs:
        filename = data['filename']
        output_file = output_dir / f'wav/{spk}_{filename}'
        if not os.path.isfile(output_file):
            resample(filepath, output_file)
        data['filepath'] = output_file
    return data

if not os.path.exists("./whisper/"): os.mkdir("./whisper/")
try:
    whisper_model = whisper.load_model("large",download_root=os.path.join(os.getcwd(), "whisper")).cpu()
except Exception as e:
    logging.info(e)
    raise Exception(
        "\n Whisper download failed or damaged, please go to "
        "'https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/large-v2.pt'"
        "\n manually download model and put it to {} .".format(os.getcwd() + "\whisper"))
    
def transcribe_one(audio_path):
    text_path = str(audio_path)[:len(str(audio_path))-4]+".txt"
    if not os.path.isfile(text_path):
        global whisper_model
        whisper_model.to(device)
        # load audio and pad/trim it to fit 30 seconds
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)


        # make log-Mel spectrogram and move to the same device as the model
        mel = whisper.log_mel_spectrogram(audio).to(whisper_model.device)

        options = whisper.DecodingOptions(temperature=1.0, best_of=5, fp16=False if device == torch.device("cpu") else True, sample_len=150, language="Chinese")
        result = whisper.decode(whisper_model, mel, options)
        text = result.text
        print(text)
        with open(text_path, 'w') as file:
            file.write(result.text);
    else:
        with open(text_path, 'r') as file:
            text = file.readline();
    return text

def prepare_customize(
    corpus_dir: Pathlike, output_dir: Optional[Pathlike] = None
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions
    :param corpus_dir: Pathlike, the path of the data dir.
    :param output_dir: Pathlike, the path where to write the manifests.
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'recordings' and 'supervisions'.
    """
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    
    label_dir_path = corpus_dir / "audio/label"
    
    sub_dir_names = get_sub_dir_names(label_dir_path)
    
    rs = True
    index = 0
    manifests = defaultdict(dict)
    recordings = []
    supervisions = []
    for sub_dir_name in sub_dir_names:
        with open(f'{label_dir_path}/{sub_dir_name}/labels_data.json', 'r') as rfile:
            lines = json.load(rfile)
            results = process_with_progress(gen_customer, lines, max_workers=1, corpus_dir=corpus_dir,output_dir=output_dir,rs=rs, spk=sub_dir_name)
            for result in results:
                line = result['result']
                audio_path = line['filepath']
                text = transcribe_one(audio_path)
                speaker = sub_dir_name
                idx = index
                if not audio_path.is_file():
                    logging.warning(f"No such file: {audio_path}")
                    continue
                recording = Recording.from_file(audio_path)
                recording.id = idx
                recordings.append(recording)
                segment = SupervisionSegment(
                    id=idx,
                    recording_id=idx,
                    start=0.0,
                    duration=recording.duration,
                    channel=0,
                    language="Chinese",
                    speaker=speaker,
                    text=text.strip(),
                )
                supervisions.append(segment)
                index = index + 1
    recording_set = RecordingSet.from_recordings(recordings)
    supervision_set = SupervisionSet.from_segments(supervisions)
    recording_set, supervision_set = fix_manifests(recording_set, supervision_set)
    validate_recordings_and_supervisions(recording_set, supervision_set)

    if output_dir is not None:
        supervision_set.to_file(
            output_dir / f"aishell_supervisions_train.jsonl.gz"
        )
        recording_set.to_file(output_dir / f"aishell_recordings_train.jsonl.gz")

    manifests["train"] = {"recordings": recording_set, "supervisions": supervision_set}

    return manifests


if __name__ == "__main__":
    corpus_dir = Path("egs/customize/download")
    output_dir = Path("egs/customize/data")
    prepare_customize(corpus_dir, output_dir)
    # transcribe_one("egs/customize/download/audio/formated/xiongchao/1_vocals_mono_48k_1.wav")