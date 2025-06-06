import os
import pickle
import shutil
from io import BufferedReader

import numpy as np
import torch
import torch.nn as nn
import yaml
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from yeaudio.audio import AudioSegment

from mvector.data_utils.featurizer import AudioFeaturizer
from mvector.infer_utils.speaker_diarization import SpeakerDiarization
from mvector.models import build_model
from mvector.utils.checkpoint import load_pretrained
from mvector.utils.utils import dict_to_object, print_arguments

class MVectorPredictor:
    def __init__(self,
                 configs,
                 threshold=0.6,
                 audio_db_path=None,
                 model_path='models/CAMPPlus_Fbank/best_model/',
                 use_gpu=True):
        """
        声纹识别预测工具
        :param configs: 配置参数
        :param threshold: 判断是否为同一个人的阈值
        :param audio_db_path: 声纹库路径
        :param model_path: 导出的预测模型文件夹路径
        :param use_gpu: 是否使用GPU预测
        """
        if use_gpu:
            assert (torch.cuda.is_available()), 'GPU不可用'
            self.device = torch.device("cuda")
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            self.device = torch.device("cpu")
        self.threshold = threshold
        # 读取配置文件
        if isinstance(configs, str):
            with open(configs, 'r', encoding='utf-8') as f:
                configs = yaml.load(f.read(), Loader=yaml.FullLoader)
            print_arguments(configs=configs)
        self.configs = dict_to_object(configs)
        self._audio_featurizer = AudioFeaturizer(feature_method=self.configs.preprocess_conf.feature_method,
                                                 use_hf_model=self.configs.preprocess_conf.get('use_hf_model', False),
                                                 method_args=self.configs.preprocess_conf.get('method_args', {}))
        # 获取模型
        backbone = build_model(input_size=self._audio_featurizer.feature_dim, configs=self.configs)
        self.predictor = nn.Sequential(backbone)
        self.predictor.to(self.device)
        # 加载模型
        if os.path.isdir(model_path):
            model_path = os.path.join(model_path, 'model.pth')
        assert os.path.exists(model_path), f"{model_path} 模型不存在！"
        self.predictor = load_pretrained(self.predictor, model_path, use_gpu=use_gpu)
        logger.info(f"成功加载模型参数：{model_path}")
        self.predictor.eval()

        # 声纹库的声纹特征
        self.audio_feature = None
        # 每个用户的平均声纹特征
        self.audio_feature_mean = None
        # 声纹特征对应的用户名
        self.users_name = []
        # 声纹特征对应的声纹文件路径
        self.users_audio_path = []
        # 每个用户的平均声纹特征对应的用户名称
        self.users_name_mean = []
        # 目标特征维度
        self.target_feature_dim = None
        # 加载声纹库
        self.audio_db_path = audio_db_path
        if self.audio_db_path is not None:
            self.audio_indexes_path = os.path.join(audio_db_path, "audio_indexes.bin")
            # 加载声纹库中的声纹
            self.__load_audio_db(self.audio_db_path)
        # 说话人日志
        self.speaker_diarize = SpeakerDiarization()

    # 加载声纹特征索引
    def __load_audio_indexes(self):
        # 如果存在声纹特征索引文件就加载
        if not os.path.exists(self.audio_indexes_path): return
        with open(self.audio_indexes_path, "rb") as f:
            indexes = pickle.load(f)

        loaded_features_2d = indexes.get("audio_feature")
        loaded_users_name = indexes.get("users_name", [])
        loaded_users_audio_path = indexes.get("users_audio_path", [])

        if loaded_features_2d is None or not loaded_users_name or not loaded_users_audio_path:
            self.audio_feature = None
            self.users_name = []
            self.users_audio_path = []
            logger.info("Loaded audio_indexes.bin is empty or incomplete.")
            return

        if not isinstance(loaded_features_2d, np.ndarray):
            logger.error(f"Loaded audio_feature is not a numpy array. Type: {type(loaded_features_2d)}. Skipping.")
            self.audio_feature = None; self.users_name = []; self.users_audio_path = []
            return
            
        if loaded_features_2d.ndim == 1: 
            logger.warning(f"Loaded audio_feature is 1D, attempting to reshape to (1, D). Shape: {loaded_features_2d.shape}")
            loaded_features_2d = loaded_features_2d[np.newaxis, :]

        if loaded_features_2d.ndim != 2:
            logger.error(f"Loaded audio_feature from pickle is not 2D after initial reshape: {loaded_features_2d.shape}. Cannot process.")
            self.audio_feature = None; self.users_name = []; self.users_audio_path = []
            return
        
        if loaded_features_2d.shape[0] == 0:
            logger.info("Loaded audio_features_2d is empty. Initializing empty audio_feature.")
            self.audio_feature = None; self.users_name = []; self.users_audio_path = []
            return

        valid_audio_features_list = []
        valid_users_name_list = []
        valid_users_audio_path_list = []

        min_len = min(len(loaded_users_name), len(loaded_users_audio_path), loaded_features_2d.shape[0])
        if min_len != loaded_features_2d.shape[0] or min_len != len(loaded_users_name) or min_len != len(loaded_users_audio_path):
            logger.warning(f"Mismatch in lengths of loaded arrays: features({loaded_features_2d.shape[0]}), names({len(loaded_users_name)}), paths({len(loaded_users_audio_path)}). Using min_len: {min_len}")

        for i in range(min_len):
            current_path = loaded_users_audio_path[i]
            if not os.path.exists(current_path):
                logger.warning(f"Audio path '{current_path}' from index for user '{loaded_users_name[i]}' does not exist. Skipping feature.")
                continue

            feature_1d = loaded_features_2d[i, :].copy()
            adjusted_feature_1d = self._adjust_feature_dim(feature_1d) 

            valid_audio_features_list.append(adjusted_feature_1d)
            valid_users_name_list.append(loaded_users_name[i])
            valid_users_audio_path_list.append(current_path)

        if valid_audio_features_list:
            self.audio_feature = np.array(valid_audio_features_list)
            self.users_name = valid_users_name_list
            self.users_audio_path = valid_users_audio_path_list
            logger.info(f"Loaded and adjusted {self.audio_feature.shape[0]} features from audio_indexes.bin. Target dim: {self.target_feature_dim}")
            if self.audio_feature.ndim != 2 or (self.target_feature_dim is not None and self.audio_feature.shape[1] != self.target_feature_dim):
                 logger.error(f"CRITICAL: self.audio_feature shape {self.audio_feature.shape} inconsistent with target_dim {self.target_feature_dim} after loading indexes.")
        else:
            self.audio_feature = None
            self.users_name = []
            self.users_audio_path = []
            logger.info("No valid features loaded from audio_indexes.bin after filtering.")

    # 保存声纹特征索引
    def __write_index(self):
        with open(self.audio_indexes_path, "wb") as f:
            pickle.dump({"users_name": self.users_name,
                         "audio_feature": self.audio_feature,
                         "users_audio_path": self.users_audio_path}, f)

    # 加载声纹库中的声纹
    def __load_audio_db(self, audio_db_path):
        # 先加载声纹特征索引
        self.__load_audio_indexes()
        os.makedirs(audio_db_path, exist_ok=True)
        audios_path = []
        for name in os.listdir(audio_db_path):
            audio_dir = os.path.join(audio_db_path, name)
            if not os.path.isdir(audio_dir): continue
            for file in os.listdir(audio_dir):
                audios_path.append(os.path.join(audio_dir, file).replace('\\', '/'))
        # 声纹库没数据就跳过
        if len(audios_path) == 0: return
        logger.info('正在加载声纹库数据...')
        input_audios = []
        for audio_path in tqdm(audios_path, desc='加载声纹库数据'):
            # 如果声纹特征已经在索引就跳过
            if audio_path in self.users_audio_path: continue
            # 读取声纹库音频
            audio_segment = self._load_audio(audio_path)
            # 获取用户名
            user_name = os.path.basename(os.path.dirname(audio_path))
            self.users_name.append(user_name)
            self.users_audio_path.append(audio_path)
            input_audios.append(audio_segment.samples)
            # 处理一批数据
            if len(input_audios) == self.configs.dataset_conf.eval_conf.batch_size:
                features = self.predict_batch(input_audios)
                if self.audio_feature is None:
                    self.audio_feature = features
                else:
                    self.audio_feature = np.vstack((self.audio_feature, features))
                input_audios = []
        # 处理不满一批的数据
        if len(input_audios) != 0:
            features = self.predict_batch(input_audios)
            if self.audio_feature is None:
                self.audio_feature = features
            else:
                self.audio_feature = np.vstack((self.audio_feature, features))
        assert len(self.audio_feature) == len(self.users_name) == len(self.users_audio_path), '加载的数量对不上！'
        # 将声纹特征保存到索引文件中
        self.__write_index()
        # 计算平均特征，用于检索
        for name in set(self.users_name):
            indexes = [idx for idx, val in enumerate(self.users_name) if val == name]
            feature = self.audio_feature[indexes].mean(axis=0)
            if self.audio_feature_mean is None:
                self.audio_feature_mean = feature
            else:
                self.audio_feature_mean = np.vstack((self.audio_feature_mean, feature))
            self.users_name_mean.append(name)
        if len(self.audio_feature_mean.shape) == 1:
            self.audio_feature_mean = self.audio_feature_mean[np.newaxis, :]
        logger.info(f'声纹库数据加载完成，一共有{len(self.audio_feature_mean)}个用户，分别是：{self.users_name_mean}')

    # 特征进行归一化
    @staticmethod
    def normalize_features(features):
        return features / np.linalg.norm(features, axis=1, keepdims=True)

    def _adjust_feature_dim(self, feature: np.ndarray) -> np.ndarray:
        """Adjusts the feature to the target dimension by padding or truncating."""
        if not isinstance(feature, np.ndarray) or feature.ndim != 1:
            logger.error(f"Feature to adjust must be a 1D numpy array, got {type(feature)} with ndim {feature.ndim if isinstance(feature, np.ndarray) else 'N/A'}.")
            if isinstance(feature, np.ndarray) and feature.ndim == 2 and feature.shape[0] == 1:
                logger.warning(f"Adjusting feature with shape {feature.shape} by flattening.")
                feature = feature.flatten()
            elif isinstance(feature, np.ndarray) and feature.ndim == 2 and feature.shape[1] == 1:
                logger.warning(f"Adjusting feature with shape {feature.shape} by flattening (column vector).")
                feature = feature.flatten()
            else:
                logger.error("Cannot adjust feature as it's not 1D or convertible to 1D.")
                return feature

        if self.target_feature_dim is None:
            self.target_feature_dim = feature.shape[0]
            logger.info(f"Target feature dimension initialized to: {self.target_feature_dim} from a feature of shape {feature.shape}")
            return feature

        current_dim = feature.shape[0]
        if current_dim == self.target_feature_dim:
            return feature
        elif current_dim < self.target_feature_dim:
            padding = np.zeros(self.target_feature_dim - current_dim, dtype=feature.dtype)
            adjusted_feature = np.concatenate((feature, padding))
            logger.warning(f"Feature padded from {current_dim} to {self.target_feature_dim}.")
            return adjusted_feature
        else:
            adjusted_feature = feature[:self.target_feature_dim]
            logger.warning(f"Feature truncated from {current_dim} to {self.target_feature_dim}.")
            return adjusted_feature

    def __retrieval(self, np_feature):
        if isinstance(np_feature, list):
            np_feature = np.array(np_feature)
        labels = []
        np_feature = self.normalize_features(np_feature.astype(np.float32))
        similarities = cosine_similarity(np_feature, self.audio_feature_mean)
        for sim in similarities:
            idx = np.argmax(sim)
            sim = sim[idx]
            if sim >= self.threshold:
                sim = round(float(sim), 5)
                labels.append([self.users_name_mean[idx], sim])
            else:
                labels.append([None, None])
        return labels

    def _load_audio(self, audio_data, sample_rate=16000):
        """加载音频
        :param audio_data: 需要识别的数据，支持文件路径，文件对象，字节，numpy，AudioSegment对象。如果是字节的话，必须是完整的字节文件
        :param sample_rate: 如果传入的事numpy数据，需要指定采样率
        :return: 识别的文本结果和解码的得分数
        """
        # 加载音频文件，并进行预处理
        if isinstance(audio_data, str):
            audio_segment = AudioSegment.from_file(audio_data)
        elif isinstance(audio_data, BufferedReader):
            audio_segment = AudioSegment.from_file(audio_data)
        elif isinstance(audio_data, np.ndarray):
            audio_segment = AudioSegment.from_ndarray(audio_data, sample_rate)
        elif isinstance(audio_data, bytes):
            audio_segment = AudioSegment.from_bytes(audio_data)
        elif isinstance(audio_data, AudioSegment):
            audio_segment = audio_data
        else:
            raise Exception(f'不支持该数据类型，当前数据类型为：{type(audio_data)}')
        assert audio_segment.duration >= self.configs.dataset_conf.dataset.min_duration, \
            f'音频太短，最小应该为{self.configs.dataset_conf.dataset.min_duration}s，当前音频为{audio_segment.duration}s'
        # 重采样
        if audio_segment.sample_rate != self.configs.dataset_conf.dataset.sample_rate:
            audio_segment.resample(self.configs.dataset_conf.dataset.sample_rate)
        # decibel normalization
        if self.configs.dataset_conf.dataset.use_dB_normalization:
            audio_segment.normalize(target_db=self.configs.dataset_conf.dataset.target_dB)
        return audio_segment

    def predict(self,
                audio_data,
                sample_rate=16000):
        """预测一个音频的特征

        :param audio_data: 需要识别的数据，支持文件路径，文件对象，字节，numpy，AudioSegment对象。如果是字节的话，必须是完整并带格式的字节文件
        :param sample_rate: 如果传入的事numpy数据，需要指定采样率
        :return: 声纹特征向量
        """
        # 加载音频文件，并进行预处理
        input_data = self._load_audio(audio_data=audio_data, sample_rate=sample_rate)
        input_data = torch.tensor(input_data.samples, dtype=torch.float32).unsqueeze(0)
        audio_feature = self._audio_featurizer(input_data).to(self.device)
        # 执行预测
        feature = self.predictor(audio_feature).data.cpu().numpy()[0]
        return feature

    def predict_batch(self, audios_data, sample_rate=16000, batch_size=32):
        """预测一批音频的特征

        :param audios_data: 需要识别的数据，支持文件路径，文件对象，字节，numpy，AudioSegment对象。如果是字节的话，必须是完整并带格式的字节文件
        :param sample_rate: 如果传入的事numpy数据，需要指定采样率
        :return: 声纹特征向量
        """
        audios_data1 = []
        for audio_data in audios_data:
            # 加载音频文件，并进行预处理
            input_data = self._load_audio(audio_data=audio_data, sample_rate=sample_rate)
            audios_data1.append(input_data.samples)
        # 找出音频长度最长的
        batch = sorted(audios_data1, key=lambda a: a.shape[0], reverse=True)
        max_audio_length = batch[0].shape[0]
        input_size = len(batch)
        # 以最大的长度创建0张量
        inputs = np.zeros((input_size, max_audio_length), dtype=np.float32)
        input_lens_ratio = []
        for x in range(input_size):
            tensor = audios_data1[x]
            seq_length = tensor.shape[0]
            # 将数据插入都0张量中，实现了padding
            inputs[x, :seq_length] = tensor[:]
            input_lens_ratio.append(seq_length / max_audio_length)
        inputs = torch.tensor(inputs, dtype=torch.float32)
        input_lens_ratio = torch.tensor(input_lens_ratio, dtype=torch.float32)
        audio_feature = self._audio_featurizer(inputs, input_lens_ratio).to(self.device)
        # 执行预测
        features_list = []
        for i in range(0, input_size, batch_size):
            batch_audio_feature = audio_feature[i:i + batch_size]
            feature_output = self.predictor(batch_audio_feature).data.cpu().numpy()
            features_list.extend(feature_output)
        
        raw_features_2d = np.array(features_list)

        if raw_features_2d.ndim == 2 and raw_features_2d.shape[0] > 0:
            adjusted_batch_features_list = []
            for i in range(raw_features_2d.shape[0]):
                feature_row_1d = raw_features_2d[i, :].copy()
                if feature_row_1d.ndim != 1:
                    logger.error(f"Unexpected feature row dimension in predict_batch: {feature_row_1d.shape}")
                    adjusted_batch_features_list.append(feature_row_1d)
                    continue
                
                adj_feat_row = self._adjust_feature_dim(feature_row_1d)
                adjusted_batch_features_list.append(adj_feat_row)
            
            if adjusted_batch_features_list:
                if self.target_feature_dim is not None:
                    final_features_np = np.empty((len(adjusted_batch_features_list), self.target_feature_dim), dtype=raw_features_2d.dtype)
                    for idx, feat in enumerate(adjusted_batch_features_list):
                        if feat.shape[0] == self.target_feature_dim:
                            final_features_np[idx, :] = feat
                        else:
                            logger.error(f"Post-adjustment feature dim mismatch in predict_batch. Expected {self.target_feature_dim}, got {feat.shape[0]}. Using original for this item.")
                            final_features_np = np.array(adjusted_batch_features_list)
                else:
                    logger.warning("target_feature_dim is None after batch adjustment in predict_batch. Returning raw features.")
                    final_features_np = raw_features_2d
                return final_features_np
            else:
                 return raw_features_2d
        else:
            return raw_features_2d

    def contrast(self, audio_data1, audio_data2):
        """声纹对比

        param audio_data1: 需要对比的音频1，支持文件路径，文件对象，字节，numpy，AudioSegment对象。如果是字节的话，必须是完整的字节文件
        param audio_data2: 需要对比的音频2，支持文件路径，文件对象，字节，numpy，AudioSegment对象。如果是字节的话，必须是完整的字节文件

        return: 两个音频的相似度
        """
        feature1 = self.predict(audio_data1)
        feature2 = self.predict(audio_data2)
        # 对角余弦值
        dist = np.dot(feature1, feature2) / (np.linalg.norm(feature1) * np.linalg.norm(feature2))
        return dist

    def register(self,
                 audio_data,
                 user_name: str,
                 sample_rate=16000):
        """声纹注册
        :param audio_data: 需要识别的数据，支持文件路径，文件对象，字节，numpy。如果是字节的话，必须是完整的字节文件
        :param user_name: 注册用户的名字
        :param sample_rate: 如果传入的事numpy数据，需要指定采样率
        :return: 识别的文本结果和解码的得分数
        """
        # 加载音频文件
        audio_segment = self._load_audio(audio_data=audio_data, sample_rate=sample_rate)
        raw_feature = self.predict(audio_data=audio_segment)

        if not isinstance(raw_feature, np.ndarray) or raw_feature.ndim != 1:
             logger.error(f"Raw feature from self.predict() is not 1D numpy array, shape: {raw_feature.shape if isinstance(raw_feature, np.ndarray) else type(raw_feature)}. Cannot register.")
             return False, "特征提取失败"

        feature = self._adjust_feature_dim(raw_feature.copy())

        feature_to_stack = feature[np.newaxis, :]

        if self.audio_feature is None:
            self.audio_feature = feature_to_stack
        else:
            if self.audio_feature.shape[1] != self.target_feature_dim:
                logger.error(f"CRITICAL: Dimension mismatch for existing self.audio_feature. "
                             f"Shape: {self.audio_feature.shape}, Target dim: {self.target_feature_dim}. "
                             "This indicates a persistent inconsistency not resolved by loading. "
                             "Attempting to register new feature might fail or corrupt further.")
                if self.audio_feature.shape[1] != feature_to_stack.shape[1]:
                    logger.critical(f"Cannot vstack. self.audio_feature.shape[1] ({self.audio_feature.shape[1]}) != feature_to_stack.shape[1] ({feature_to_stack.shape[1]})")
                    return False, "声纹库维度不一致"

            self.audio_feature = np.vstack((self.audio_feature, feature_to_stack))
        
        # 保存音频文件
        user_audio_dir = os.path.join(self.audio_db_path, user_name)
        os.makedirs(user_audio_dir, exist_ok=True)
        if not os.listdir(user_audio_dir):
            audio_path = os.path.join(user_audio_dir, '0.wav')
        else:
            existing_files = [f for f in os.listdir(user_audio_dir) if f.endswith('.wav') and f[:-4].isdigit()]
            next_num = 0
            if existing_files:
                next_num = max([int(f[:-4]) for f in existing_files]) + 1
            audio_path = os.path.join(user_audio_dir, f'{next_num}.wav')
            
        audio_segment.to_wav_file(audio_path)
        self.users_audio_path.append(audio_path.replace('\\\\', '/'))
        self.users_name.append(user_name)
        
        try:
            self.__write_index()
        except Exception as e:
            logger.error(f"Error writing index file: {e}")
            return False, f"写入索引失败: {e}"

        if self.audio_feature_mean is not None and self.audio_feature_mean.shape[1] != self.target_feature_dim:
            logger.warning(f"self.audio_feature_mean dimension ({self.audio_feature_mean.shape[1]}) "
                           f"differs from target dimension ({self.target_feature_dim}). Rebuilding mean features.")
            self.audio_feature_mean = None 
            self.users_name_mean = []
            unique_user_names = sorted(list(set(self.users_name)))
            if unique_user_names:
                temp_mean_features_list = []
                for u_name in unique_user_names:
                    u_indexes = [idx for idx, val in enumerate(self.users_name) if val == u_name]
                    if u_indexes:
                        mean_feat = self.audio_feature[u_indexes].mean(axis=0)
                        temp_mean_features_list.append(mean_feat)
                if temp_mean_features_list:
                    self.audio_feature_mean = np.array(temp_mean_features_list)
                    self.users_name_mean = unique_user_names

        if user_name in self.users_name_mean:
            idx_mean = self.users_name_mean.index(user_name)
            current_user_indexes = [i for i, u in enumerate(self.users_name) if u == user_name]
            if current_user_indexes:
                mean_feature_for_user = self.audio_feature[current_user_indexes].mean(axis=0)
                self.audio_feature_mean[idx_mean] = mean_feature_for_user
        else:
            adjusted_feature_for_mean = feature[np.newaxis, :]

            if self.audio_feature_mean is None:
                if not self.users_name_mean:
                     self.audio_feature_mean = adjusted_feature_for_mean
                     self.users_name_mean.append(user_name)
                else:
                     self.audio_feature_mean = np.vstack((self.audio_feature_mean, adjusted_feature_for_mean))
                     self.users_name_mean.append(user_name)

            else:
                 current_user_indexes = [i for i, u in enumerate(self.users_name) if u == user_name]
                 mean_feature_for_user = self.audio_feature[current_user_indexes].mean(axis=0)

                 self.audio_feature_mean = np.vstack((self.audio_feature_mean, mean_feature_for_user[np.newaxis, :]))
                 self.users_name_mean.append(user_name)
                 
        return True, "注册成功"

    def recognition(self, audio_data, threshold=None, sample_rate=16000):
        """声纹识别

        Args:
            audio_data (str, file-like object, bytes, numpy.ndarray, AudioSegment): 需要识别的数据
            threshold (float): 判断的阈值，如果为None则用创建对象时使用的阈值。默认为None。
            sample_rate (int): 如果传入的是numpy数组，需要指定采样率。默认为16000。
        Returns:
            str: 识别的用户名称，如果为None，即没有识别到用户。
        """
        if threshold:
            self.threshold = threshold
        feature = self.predict(audio_data, sample_rate=sample_rate)
        result = self.__retrieval(np_feature=np.array([feature]))[0]
        return result

    def get_users(self):
        """获取所有用户

        return: 所有用户
        """
        return self.users_name

    def remove_user(self, user_name):
        """删除用户

        :param user_name: 用户名
        :return:
        """
        if user_name in self.users_name and user_name in self.users_name_mean:
            indexes = [i for i in range(len(self.users_name)) if self.users_name[i] == user_name]
            for index in sorted(indexes, reverse=True):
                del self.users_name[index]
                del self.users_audio_path[index]
                self.audio_feature = np.delete(self.audio_feature, index, axis=0)
            self.__write_index()
            shutil.rmtree(os.path.join(self.audio_db_path, user_name))
            # 删除检索内的特征
            index = self.users_name_mean.index(user_name)
            del self.users_name_mean[index]
            self.audio_feature_mean = np.delete(self.audio_feature_mean, index, axis=0)
            return True
        else:
            return False

    def speaker_diarization(self, audio_data, sample_rate=16000, speaker_num=None, search_audio_db=False):
        """说话人日志识别

        Args:
            audio_data: 需要识别的数据，支持文件路径，文件对象，字节，numpy。如果是字节的话，必须是完整并带格式的字节文件
            sample_rate (int): 如果传入的是numpy数据，需要指定采样率
            speaker_num (int): 预期的说话人数量，提供说话人数量可以提高准确率
            search_audio_db (bool): 是否在数据库中搜索与输入音频最匹配的音频进行识别
        Returns:
            list: 说话人日志识别结果
        """
        input_data = self._load_audio(audio_data=audio_data, sample_rate=sample_rate)
        segments = self.speaker_diarize.segments_audio(input_data)
        segments_data = [segment[2] for segment in segments]
        features = self.predict_batch(segments_data, sample_rate=sample_rate)
        labels, spk_center_embeddings = self.speaker_diarize.clustering(features, speaker_num=speaker_num)
        outputs = self.speaker_diarize.postprocess(segments, labels)
        if search_audio_db:
            assert self.audio_feature is not None, "数据库中没有音频数据，请先指定说话人特征数据库或者注册说话人"
            names = self.__retrieval(np_feature=spk_center_embeddings)
            results = []
            for output in outputs:
                name = names[output['speaker']][0]
                result = {
                    'speaker': name if name else f"陌生人{output['speaker']}",
                    'start': output['start'],
                    'end': output['end']
                }
                results.append(result)
            outputs = results
        return outputs
