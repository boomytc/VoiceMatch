import argparse
import json
import yamlargparse
import torch.nn as nn

class network_wrapper(nn.Module):
    """
    一个包装类，用于加载不同的神经网络模型，适用于语音增强（SE）、语音分离（SS）和目标说话人提取（TSE）等任务。
    它管理参数解析、模型配置加载，并根据任务和模型名称实例化相应模型。
    """
    
    def __init__(self):
        """
        初始化网络包装器，不预定义任何模型或参数。
        """
        super(network_wrapper, self).__init__()
        self.args = None  # 命令行参数的占位符
        self.config_path = None  # YAML配置文件的路径
        self.model_name = None  # 基于任务要加载的模型名称

    def load_args_se(self):
        """
        使用YAML配置文件加载语音增强任务的参数。
        设置配置路径并解析所有必需的参数，如输入/输出路径、模型设置和FFT参数。
        """
        self.config_path = 'config/inference/' + self.model_name + '.yaml'
        parser = yamlargparse.ArgumentParser("Settings")

        # 通用模型和推理设置
        parser.add_argument('--config', help='配置文件路径', action=yamlargparse.ActionConfigFile)
        parser.add_argument('--mode', type=str, default='inference', help='模式：训练或推理')
        parser.add_argument('--checkpoint-dir', dest='checkpoint_dir', type=str, default='checkpoints/FRCRN_SE_16K', help='检查点目录')
        parser.add_argument('--input-path', dest='input_path', type=str, help='噪声音频输入的路径')
        parser.add_argument('--output-dir', dest='output_dir', type=str, help='增强音频输出的目录')
        parser.add_argument('--use-cuda', dest='use_cuda', default=1, type=int, help='启用CUDA（1=是，0=否）')
        parser.add_argument('--num-gpu', dest='num_gpu', type=int, default=1, help='使用的GPU数量')

        # 模型特定设置
        parser.add_argument('--network', type=str, help='选择SE模型：FRCRN_SE_16K, MossFormer2_SE_48K')
        parser.add_argument('--sampling-rate', dest='sampling_rate', type=int, default=16000, help='采样率')
        parser.add_argument('--one-time-decode-length', dest='one_time_decode_length', type=float, default=60.0, help='一次性解码的最大段长度')
        parser.add_argument('--decode-window', dest='decode_window', type=float, default=1.0, help='解码块大小')

        # 用于特征提取的FFT参数
        parser.add_argument('--window-len', dest='win_len', type=int, default=400, help='帧的窗口长度')
        parser.add_argument('--window-inc', dest='win_inc', type=int, default=100, help='帧的窗口移动步长')
        parser.add_argument('--fft-len', dest='fft_len', type=int, default=512, help='特征提取的FFT长度')
        parser.add_argument('--num-mels', dest='num_mels', type=int, default=60, help='梅尔频谱图bins数量')
        parser.add_argument('--window-type', dest='win_type', type=str, default='hamming', help='窗口类型：hamming或hanning')

        # 从配置文件解析参数
        self.args = parser.parse_args(['--config', self.config_path])

    def load_args_ss(self):
        """
        使用YAML配置文件加载语音分离任务的参数。
        该方法设置输入/输出路径、模型配置和基于MossFormer2的语音分离模型的编码器/解码器设置等参数。
        """
        self.config_path = 'config/inference/' + self.model_name + '.yaml'
        parser = yamlargparse.ArgumentParser("Settings")

        # 通用模型和推理设置
        parser.add_argument('--config', default=self.config_path, help='配置文件路径', action=yamlargparse.ActionConfigFile)
        parser.add_argument('--mode', type=str, default='inference', help='模式：训练或推理')
        parser.add_argument('--checkpoint-dir', dest='checkpoint_dir', type=str, default='checkpoints/FRCRN_SE_16K', help='检查点目录')
        parser.add_argument('--input-path', dest='input_path', type=str, help='混合音频输入的路径')
        parser.add_argument('--output-dir', dest='output_dir', type=str, help='分离音频输出的目录')
        parser.add_argument('--use-cuda', dest='use_cuda', default=1, type=int, help='启用CUDA（1=是，0=否）')
        parser.add_argument('--num-gpu', dest='num_gpu', type=int, default=1, help='使用的GPU数量')

        # 语音分离的模型特定设置
        parser.add_argument('--network', type=str, help='选择SS模型：MossFormer2_SS_16K')
        parser.add_argument('--sampling-rate', dest='sampling_rate', type=int, default=16000, help='采样率')
        parser.add_argument('--num-spks', dest='num_spks', type=int, default=2, help='要分离的说话人数量')
        parser.add_argument('--one-time-decode-length', dest='one_time_decode_length', type=float, default=60.0, help='一次性解码的最大段长度')
        parser.add_argument('--decode-window', dest='decode_window', type=float, default=1.0, help='解码块大小')

        # 编码器设置
        parser.add_argument('--encoder_kernel-size', dest='encoder_kernel_size', type=int, default=16, help='Conv1D编码器的卷积核大小')
        parser.add_argument('--encoder-embedding-dim', dest='encoder_embedding_dim', type=int, default=512, help='编码器的嵌入维度')

        # MossFormer模型参数
        parser.add_argument('--mossformer-squence-dim', dest='mossformer_sequence_dim', type=int, default=512, help='MossFormer的序列维度')
        parser.add_argument('--num-mossformer_layer', dest='num_mossformer_layer', type=int, default=24, help='MossFormer层数')
        
        # 从配置文件解析参数
        self.args = parser.parse_args(['--config', self.config_path])
    
    def load_config_json(self, config_json_path):
        with open(config_json_path, 'r') as file:
            return json.load(file)
      
    def combine_config_and_args(self, json_config, args):
        # 将argparse.Namespace转换为字典
        args_dict = vars(args)
        
        # 从args_dict中移除`config`键（它是JSON文件的路径）
        args_dict.pop("config", None)
        
        # 合并JSON配置和args_dict，优先使用args_dict
        combined_config = {**json_config, **{k: v for k, v in args_dict.items() if v is not None}}
        return combined_config
    
    def load_args_sr(self):
        """
        使用YAML配置文件加载语音超分辨率任务的参数。
        设置配置路径并解析所有必需的参数，如输入/输出路径、模型设置和FFT参数。
        """
        self.config_path = 'config/inference/' + self.model_name + '.yaml'
        parser = yamlargparse.ArgumentParser("Settings")

        # 通用模型和推理设置
        parser.add_argument('--config', help='配置文件路径', action=yamlargparse.ActionConfigFile)
        parser.add_argument('--config_json', type=str, help='config.json文件的路径')
        parser.add_argument('--mode', type=str, default='inference', help='模式：训练或推理')
        parser.add_argument('--checkpoint-dir', dest='checkpoint_dir', type=str, default='checkpoints/FRCRN_SE_16K', help='检查点目录')
        parser.add_argument('--input-path', dest='input_path', type=str, help='噪声音频输入的路径')
        parser.add_argument('--output-dir', dest='output_dir', type=str, help='增强音频输出的目录')
        parser.add_argument('--use-cuda', dest='use_cuda', default=1, type=int, help='启用CUDA（1=是，0=否）')
        parser.add_argument('--num-gpu', dest='num_gpu', type=int, default=1, help='使用的GPU数量')

        # 模型特定设置
        parser.add_argument('--network', type=str, help='选择SE模型：FRCRN_SE_16K, MossFormer2_SE_48K')
        parser.add_argument('--sampling-rate', dest='sampling_rate', type=int, default=16000, help='采样率')
        parser.add_argument('--one-time-decode-length', dest='one_time_decode_length', type=float, default=60.0, help='一次性解码的最大段长度')
        parser.add_argument('--decode-window', dest='decode_window', type=float, default=1.0, help='解码块大小')

        # 从配置文件解析参数
        self.args = parser.parse_args(['--config', self.config_path])
        json_config = self.load_config_json(self.args.config_json)
        self.args = self.combine_config_and_args(json_config, self.args)
        self.args = argparse.Namespace(**self.args)

    def load_args_tse(self):
        """
        使用YAML配置文件加载目标说话人提取（TSE）任务的参数。
        参数包括输入/输出路径、CUDA配置和解码参数。
        """
        self.config_path = 'config/inference/' + self.model_name + '.yaml'
        parser = yamlargparse.ArgumentParser("Settings")

        # 通用模型和推理设置
        parser.add_argument('--config', default=self.config_path, help='配置文件路径', action=yamlargparse.ActionConfigFile)
        parser.add_argument('--mode', type=str, default='inference', help='模式：训练或推理')
        parser.add_argument('--checkpoint-dir', dest='checkpoint_dir', type=str, default='checkpoint_dir/AV_MossFormer2_TSE_16K', help='检查点目录')
        parser.add_argument('--input-path', dest='input_path', type=str, help='混合音频输入的路径')
        parser.add_argument('--output-dir', dest='output_dir', type=str, help='分离音频输出的目录')
        parser.add_argument('--use-cuda', dest='use_cuda', default=1, type=int, help='启用CUDA（1=是，0=否）')
        parser.add_argument('--num-gpu', dest='num_gpu', type=int, default=1, help='使用的GPU数量')

        # 目标说话人提取的模型特定设置
        parser.add_argument('--network', type=str, help='选择TSE模型（当前支持AV_MossFormer2_TSE_16K）')
        parser.add_argument('--sampling-rate', dest='sampling_rate', type=int, default=16000, help='采样率（当前支持16 kHz）')
        parser.add_argument('--network_reference', type=dict, help='包含辅助参考信号参数的字典')
        parser.add_argument('--network_audio', type=dict, help='包含网络参数的字典')

        # 用于流式或基于块的解码的参数
        parser.add_argument('--one-time-decode-length', dest='one_time_decode_length', type=int, default=60, help='一次性解码的最大段长度')
        parser.add_argument('--decode-window', dest='decode_window', type=int, default=1, help='流式处理的块长度')

        # 从配置文件解析参数
        self.args = parser.parse_args(['--config', self.config_path])

    def __call__(self, task, model_name):
        """
        根据任务类型（例如'speech_enhancement'、'speech_separation'或'target_speaker_extraction'）
        调用相应的参数加载函数。然后根据所选任务和模型名称加载相应的模型。
        
        参数：
        - task (str)：任务类型（'speech_enhancement'、'speech_separation'、'target_speaker_extraction'）。
        - model_name (str)：要加载的模型名称（例如'FRCRN_SE_16K'）。
        
        返回：
        - self.network：实例化的神经网络模型。
        """
        
        self.model_name = model_name  # 根据用户输入设置模型名称
        
        # 加载特定于任务的参数
        if task == 'speech_enhancement':
            self.load_args_se()  # 加载语音增强的参数
        elif task == 'speech_separation':
            self.load_args_ss()  # 加载语音分离的参数
        elif task == 'speech_super_resolution':
            self.load_args_sr()  # 加载语音超分辨率的参数
        elif task == 'target_speaker_extraction':
            self.load_args_tse()  # 加载目标说话人提取的参数
        else:
            # 如果任务不受支持，则打印错误消息
            print(f'{task}不受支持，请从以下选项中选择：'
                  'speech_enhancement, speech_separation, speech_super_resolution或target_speaker_extraction')
            return

        #print(self.args)  # 显示解析的参数
        self.args.task = task 
        self.args.network = self.model_name  # 将网络名称设置为模型名称

        # 根据所选模型初始化相应的网络
        if self.args.network == 'FRCRN_SE_16K':
            from networks import CLS_FRCRN_SE_16K
            self.network = CLS_FRCRN_SE_16K(self.args)  # 加载FRCRN模型
        elif self.args.network == 'MossFormer2_SE_48K':
            from networks import CLS_MossFormer2_SE_48K
            self.network = CLS_MossFormer2_SE_48K(self.args)  # 加载MossFormer2_SE模型
        elif self.args.network == 'MossFormer2_SR_48K':
            from networks import CLS_MossFormer2_SR_48K
            self.network = CLS_MossFormer2_SR_48K(self.args)  # 加载MossFormer2_SR模型
        elif self.args.network == 'MossFormerGAN_SE_16K':
            from networks import CLS_MossFormerGAN_SE_16K
            self.network = CLS_MossFormerGAN_SE_16K(self.args)  # 加载MossFormerGAN模型
        elif self.args.network == 'MossFormer2_SS_16K':
            from networks import CLS_MossFormer2_SS_16K
            self.network = CLS_MossFormer2_SS_16K(self.args)  # 加载用于分离的MossFormer2
        elif self.args.network == 'AV_MossFormer2_TSE_16K':
            from networks import CLS_AV_MossFormer2_TSE_16K
            self.network = CLS_AV_MossFormer2_TSE_16K(self.args)  # 加载用于目标说话人提取的AV MossFormer2模型
        else:
            # 如果找不到匹配的网络，则打印错误消息
            print("未找到网络！")
            return
        
        return self.network  # 返回实例化的网络模型
