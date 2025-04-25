#!/usr/bin/env python -u
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch 
import torch.nn as nn
import numpy as np
import os 
import sys
import librosa
import mimetypes

def get_file_extension(file_path):
    """
    返回音频文件的扩展名
    """

    _, ext = os.path.splitext(file_path)
    return ext

def is_audio_file(file_path):
    """
    检查给定文件路径是否为音频文件
    如果是音频文件则返回 True，否则返回 False
    """
    file_ext = ["wav", "aac", "ac3", "aiff", "flac", "m4a", "mp3", "ogg", "opus", "wma", "webm"]

    ext = get_file_extension(file_path)
    if ext.replace('.','') in file_ext:
        return True

    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type and mime_type.startswith('audio'):
        return True
    return False
    
def read_and_config_file(args, input_path, decode=0):
    """
    读取并处理输入的文件或目录，以提取音频文件路径或配置信息。
    
    参数：
    args：参数对象
    input_path (str)：包含音频数据或文件路径的文件或目录路径
    decode (bool)：是否为解码模式，
                   如果为 True（decode=1），则直接处理音频文件（查找 .wav 或 .flac）或 .scp 文件；
                   如果为 False（decode=0），则将输入文件视为配置文件，每行包含音频文件路径。
    
    返回：
    processed_list (list)：处理后的文件路径列表，或包含 'inputs' 和可选 'condition_audio' 的字典列表。
    """
    processed_list = []  # 初始化列表以保存处理后的文件路径或配置信息
    
    # 支持的音频类型如下（已测试），但不限于此。
    file_ext = ["wav", "aac", "ac3", "aiff", "flac", "m4a", "mp3", "ogg", "opus", "wma", "webm"]
    
    if decode:
        if args.task == 'target_speaker_extraction':
            if args.network_reference.cue== 'lip':
                # 如果 decode 为 True，则在目录或单个文件中查找视频文件
                if os.path.isdir(input_path):
                    # 在输入目录中查找所有 .mp4、.avi、.mov、.MOV、.webm 文件
                    processed_list = librosa.util.find_files(input_path, ext="mp4")
                    processed_list += librosa.util.find_files(input_path, ext="avi")
                    processed_list += librosa.util.find_files(input_path, ext="mov")
                    processed_list += librosa.util.find_files(input_path, ext="MOV")
                    processed_list += librosa.util.find_files(input_path, ext="webm")
                else:
                    # 如果是单个文件且为 .mp4/.avi/.mov/.webm，则添加到列表
                    if input_path.lower().endswith(".mp4") or input_path.lower().endswith(".avi") or input_path.lower().endswith(".mov") or input_path.lower().endswith(".webm"):
                        processed_list.append(input_path)
                    else:
                        # 从输入的文本文件读取文件路径（每行一个路径）
                        with open(input_path) as fid:
                            for line in fid:
                                path_s = line.strip().split()  # 拆分路径（以空格分隔）
                                processed_list.append(path_s[0])  # 添加第一个路径（输入音频路径）
                return processed_list

        # 如果 decode 为 True，则在目录或单个文件中查找音频文件
        if os.path.isdir(input_path):
            # 在输入目录中查找所有 .wav 文件
            processed_list = librosa.util.find_files(input_path, ext=file_ext)
        else:
            # 如果是单个音频文件，添加到列表
            #if input_path.lower().endswith(".wav") or input_path.lower().endswith(".flac"):
            if is_audio_file(input_path):
                processed_list.append(input_path)
            else:
                # 从输入的文本文件读取文件路径（每行一个路径）
                with open(input_path) as fid:
                    for line in fid:
                        path_s = line.strip().split()  # 拆分路径（以空格分隔）
                        processed_list.append(path_s[0])  # 添加第一个路径（输入音频路径）
        return processed_list

    # 如果 decode 为 False，将输入文件视为配置文件
    with open(input_path) as fid:
        for line in fid:
            tmp_paths = line.strip().split()  # 拆分路径（以空格分隔）
            if len(tmp_paths) == 2:
                # 如果每行有两个路径，则将第二个视为 'condition_audio'
                sample = {'inputs': tmp_paths[0], 'condition_audio': tmp_paths[1]}
            elif len(tmp_paths) == 1:
                # 如果每行只有一个路径，则视为 'inputs'
                sample = {'inputs': tmp_paths[0]}
            processed_list.append(sample)  # 将处理后的样本添加到列表
    return processed_list

