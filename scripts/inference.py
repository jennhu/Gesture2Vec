import argparse
import math
import pickle
import pprint
from pathlib import Path

import librosa
import numpy as np
import time

import torch
from scipy.signal import savgol_filter
import joblib as jl

import utils
from pymo.preprocessing import *
from pymo.viz_tools import *
from pymo.writers import *
from utils.data_utils import SubtitleWrapper, normalize_string
from utils.train_utils import set_logger

from data_loader.data_preprocessor import DataPreprocessor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def generate_gestures(args, pose_decoder, lang_model, words, seed_seq=None):
    out_list = []
    clip_length = words[-1][2]

    # pre seq
    pre_seq = torch.zeros((1, args.n_pre_poses, pose_decoder.pose_dim))
    if seed_seq is not None:
        pre_seq[0, :, :] = torch.Tensor(seed_seq[0:args.n_pre_poses])
    else:
        mean_pose = args.data_mean
        mean_pose = torch.squeeze(torch.Tensor(mean_pose))
        pre_seq[0, :, :] = mean_pose.repeat(args.n_pre_poses, 1)

    # divide into inference units and do inferences
    unit_time = args.n_poses / args.motion_resampling_framerate
    stride_time = (args.n_poses - args.n_pre_poses) / args.motion_resampling_framerate
    if clip_length < unit_time:
        num_subdivision = 1
    else:
        num_subdivision = math.ceil((clip_length - unit_time) / stride_time) + 1

    print('{}, {}, {}, {}'.format(num_subdivision, unit_time, clip_length, stride_time))
    num_subdivision = min(num_subdivision, 500)  # DEBUG: generate only for the first N divisions

    out_poses = None
    start = time.time()
    for i in range(0, num_subdivision):
        start_time = i * stride_time
        end_time = start_time + unit_time

        # prepare text input
        word_seq = DataPreprocessor.get_words_in_time_range(word_list=words, start_time=start_time, end_time=end_time)
        word_indices = np.zeros(len(word_seq) + 2)
        word_indices[0] = lang_model.SOS_token
        word_indices[-1] = lang_model.EOS_token
        for w_i, word in enumerate(word_seq):
            print(word[0], end=', ')
            word_indices[w_i + 1] = lang_model.get_word_index(word[0])
        print(' ({}, {})'.format(start_time, end_time))
        in_text = torch.LongTensor(word_indices).unsqueeze(0).to(device)

        # prepare pre seq
        if i > 0:
            pre_seq[0, :, :] = out_poses.squeeze(0)[-args.n_pre_poses:]
        pre_seq = pre_seq.float().to(device)

        # inference
        words_lengths = torch.LongTensor([in_text.shape[1]])
        out_poses = pose_decoder(in_text, words_lengths, pre_seq, None)
        out_seq = out_poses[0, :, :].data.cpu().numpy()

        # smoothing motion transition
        if len(out_list) > 0:
            last_poses = out_list[-1][-args.n_pre_poses:]
            out_list[-1] = out_list[-1][:-args.n_pre_poses]  # delete the last part

            for j in range(len(last_poses)):
                n = len(last_poses)
                prev = last_poses[j]
                next = out_seq[j]
                out_seq[j] = prev * (n - j) / (n + 1) + next * (j + 1) / (n + 1)

        out_list.append(out_seq)

    print('Avg. inference time: {:.2} s'.format((time.time() - start) / num_subdivision))

    # aggregate results
    out_poses = np.vstack(out_list)

    return out_poses


def main(checkpoint_path, transcript_path):
    args, generator, loss_fn, lang_model, out_dim = utils.train_utils.load_checkpoint_and_model(
        checkpoint_path, device, what='baseline')
    pprint.pprint(vars(args))
    save_path = '../output/infer_sample'
    os.makedirs(save_path, exist_ok=True)

    # load lang_model
    vocab_cache_path = os.path.join(os.path.split(args.train_data_path[0])[0], 'vocab_cache.pkl')
    with open(vocab_cache_path, 'rb') as f:
        lang_model = pickle.load(f)

    # prepare input
    # Todo: Why? Whas it for the hitmap check?
    big_infer = False
    if not big_infer:
        transcript = SubtitleWrapper(transcript_path).get()
    else:
        adress = "/local-scratch/pjomeyaz/GENEA_DATASET/trinityspeechgesture.scss.tcd.ie/data/GENEA_Challenge_2020_data_release/Test_data/Transcripts/TestSeq00"
        offset = 0
        transcript_list = []
        for i in range(1, 6):
            current_transcript = SubtitleWrapper(adress + str(i) + ".json").get()
            for word in current_transcript:
                word['start_time'] = "{:.2f}".format(offset + float(word['start_time'][0:-1])) + 's'
                word['end_time'] = "{:.2f}".format(offset + float(word['end_time'][0:-1])) + 's'
            offset = float(current_transcript[-1]['end_time'][0:-1])
            transcript_list.append(current_transcript)
        transcript = [item for subist in transcript_list for item in subist]

    word_list = []
    for wi in range(len(transcript)):
        word_s = float(transcript[wi]['start_time'][:-1])
        word_e = float(transcript[wi]['end_time'][:-1])
        word = transcript[wi]['word']

        word = normalize_string(word)
        if len(word) > 0:
            word_list.append([word, word_s, word_e])

    # inference
    out_poses = generate_gestures(args, generator, lang_model, word_list)

    # unnormalize
    mean = np.array(args.data_mean).squeeze()
    std = np.array(args.data_std).squeeze()
    std = np.clip(std, a_min=0.01, a_max=None)
    out_poses = np.multiply(out_poses, std) + mean

    # make a BVH
    filename_prefix = '{}'.format(transcript_path.stem)
    make_bvh(save_path, filename_prefix, out_poses)


def make_bvh(save_path, filename_prefix, poses):
    writer = BVHWriter()
    pipeline = jl.load('../resource/data_pipe.sav')

    # smoothing
    n_poses = poses.shape[0]
    out_poses = np.zeros((n_poses, poses.shape[1]))

    for i in range(poses.shape[1]):
        out_poses[:, i] = savgol_filter(poses[:, i], 15, 2)  # NOTE: smoothing on rotation matrices is not optimal

    # rotation matrix to euler angles
    out_poses = out_poses.reshape((out_poses.shape[0], -1, 9))
    out_poses = out_poses.reshape((out_poses.shape[0], out_poses.shape[1], 3, 3))
    out_euler = np.zeros((out_poses.shape[0], out_poses.shape[1] * 3))
    for i in range(out_poses.shape[0]):  # frames
        r = R.from_matrix(out_poses[i])
        out_euler[i] = r.as_euler('ZXY', degrees=True).flatten()

    bvh_data = pipeline.inverse_transform([out_euler])

    out_bvh_path = os.path.join(save_path, filename_prefix + '_generated.bvh')
    with open(out_bvh_path, 'w') as f:
        writer.write(bvh_data[0], f)


if __name__ == '__main__':
    '''
    ../output/baseline/baseline_model_checkpoint_100.bin
     /local-scratch/pjomeyaz/GENEA_DATASET/trinityspeechgesture.scss.tcd.ie/data/GENEA_Challenge_2020_data_release/Test_data/Transcripts/TestSeq001.json
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_path", type=Path)
    parser.add_argument("transcript_path", type=Path)
    args = parser.parse_args()

    main(args.ckpt_path, args.transcript_path)
