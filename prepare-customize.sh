#!/usr/bin/env bash

set -eou pipefail

# fix segmentation fault reported in https://github.com/k2-fsa/icefall/issues/674
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

nj=16
stage=-1
stop_stage=2

# We assume dl_dir (download dir) contains the following
# directories and files. If not, they will be downloaded
# by this script automatically.
#
#  - $dl_dir/aishell
#      You can download aishell from https://www.openslr.org/33/
#

dl_dir=$PWD/egs/customize/download

dataset_parts="-p train"  # debug
# dataset_parts="-p test" 

text_extractor=""
audio_extractor="Encodec"  # or Fbank
audio_feats_dir=egs/customize/data/tokenized

. egs/customize/shared/parse_options.sh || exit 1


# All files generated by this script are saved in "data".
# You can safely remove "data" and rerun this script to regenerate it.
mkdir -p egs/customize/data

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  log "Stage 2: Tokenize/Fbank customize"
  mkdir -p ${audio_feats_dir}
  if [ ! -e ${audio_feats_dir}/.customize.tokenize.done ]; then
    python3 bin/tokenizer.py --dataset-parts "${dataset_parts}" \
        --audio-extractor ${audio_extractor} \
        --prefix "customize"\
        --batch-duration 400 \
        --src-dir "egs/customize/data/manifests" \
        --output-dir "${audio_feats_dir}"
  fi
  touch ${audio_feats_dir}/.customize.tokenize.done
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 3: Prepare customize train/dev/test"
  if [ ! -e ${audio_feats_dir}/.customize.train.done ]; then
    # dev 14326
    lhotse subset --first 400 \
        ${audio_feats_dir}/customize_cuts_train.jsonl.gz \
        ${audio_feats_dir}/cuts_dev.jsonl.gz

    lhotse subset --last 8979 \
        ${audio_feats_dir}/customize_cuts_train.jsonl.gz \
        ${audio_feats_dir}/cuts_dev_others.jsonl.gz
        
    #test
    lhotse subset --first 400 \
        ${audio_feats_dir}/cuts_dev_others.jsonl.gz \
        ${audio_feats_dir}/cuts_test.jsonl.gz

    #train
    lhotse subset --last 8579 \
        ${audio_feats_dir}/cuts_dev_others.jsonl.gz \
        ${audio_feats_dir}/cuts_train.jsonl.gz

    touch ${audio_feats_dir}/.customize.train.done
  fi
fi

python3 ./bin/display_manifest_statistics.py --manifest-dir ${audio_feats_dir}