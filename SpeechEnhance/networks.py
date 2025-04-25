import torch
import torch.nn as nn
import soundfile as sf
import os
import subprocess
import librosa
from tqdm import tqdm
import numpy as np
from pydub import AudioSegment
from utils.decode import decode_one_audio
from dataloader.dataloader import DataReader

MAX_WAV_VALUE = 32768.0

class SpeechModel:
    """
    SpeechModel类是一个基类，设计用于处理语音处理任务，
    如加载、处理和解码音频数据。它初始化计算设备
    （CPU或GPU）并保存模型相关的属性。该类具有灵活性，旨在
    被特定的语音模型扩展，用于语音增强、语音分离、
    目标说话人提取等任务。

    属性:
    - args: 包含配置设置的参数解析器对象。
    - device: 模型运行的设备（CPU或GPU）。
    - model: 用于语音处理任务的实际模型（由子类加载）。
    - name: 模型名称的占位符。
    - data: 存储与模型相关的任何附加数据的字典，如音频输入。
    """

    def __init__(self, args):
        """
        通过根据系统可用性确定计算设备
        （GPU或CPU）来初始化SpeechModel类，用于运行模型。

        参数:
        - args: 包含设置的参数解析器对象，如是否使用CUDA（GPU）。
        """
        # 检查是否有GPU可用
        if torch.cuda.is_available():
            # 使用自定义方法查找具有最多空闲内存的GPU
            free_gpu_id = self.get_free_gpu()
            if free_gpu_id is not None:
                args.use_cuda = 1
                torch.cuda.set_device(free_gpu_id)
                self.device = torch.device('cuda')
            else:
                # 如果未检测到GPU，使用CPU
                #print("未找到GPU。使用CPU。")
                args.use_cuda = 0
                self.device = torch.device('cpu')
        else:
            # 如果未检测到GPU，使用CPU
            args.use_cuda = 0
            self.device = torch.device('cpu')

        self.args = args
        self.model = None
        self.name = None
        self.data = {}
        self.print = False

    def get_free_gpu(self):
        """
        使用'nvidia-smi'识别具有最多空闲内存的GPU并返回其索引。

        此函数查询系统上可用的GPU并确定哪个GPU具有
        最多的空闲内存。它使用`nvidia-smi`命令行工具收集
        GPU内存使用数据。如果成功，它返回具有最多空闲内存的GPU的索引。
        如果查询失败或发生错误，它返回None。

        返回:
        int: 具有最多空闲内存的GPU的索引，如果未找到GPU或发生错误则返回None。
        """
        try:
            # 运行nvidia-smi查询GPU内存使用情况和空闲内存
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.free', '--format=csv,nounits,noheader'], stdout=subprocess.PIPE)
            gpu_info = result.stdout.decode('utf-8').strip().split('\n')

            free_gpu = None
            max_free_memory = 0
            for i, info in enumerate(gpu_info):
                used, free = map(int, info.split(','))
                if free > max_free_memory:
                    max_free_memory = free
                    free_gpu = i
            return free_gpu
        except Exception as e:
            print(f"查找空闲GPU时出错: {e}")
            return None

    def download_model(self, model_name):
        checkpoint_dir = self.args.checkpoint_dir
        from huggingface_hub import snapshot_download
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        print(f'正在下载{model_name}的检查点')
        try:
            snapshot_download(repo_id=f'alibabasglab/{model_name}', local_dir=checkpoint_dir)
            return True
        except:
            return False
            
    def load_model(self):
        """
        从指定目录加载预训练模型检查点。它检查
        检查点目录中的最佳模型('last_best_checkpoint')。如果找到模型，
        它将模型状态加载到当前模型实例中。

        如果找不到检查点，它会尝试从huggingface下载模型。
        如果下载失败，它会打印警告消息。

        步骤:
        - 搜索最佳模型检查点或最近的检查点。
        - 从检查点文件加载模型的状态字典。

        引发:
        - FileNotFoundError: 如果既没有找到'last_best_checkpoint'也没有找到'last_checkpoint'文件。
        """
        # 定义最佳模型和最后检查点的路径
        best_name = os.path.join(self.args.checkpoint_dir, 'last_best_checkpoint')
        # 检查最后的最佳检查点是否存在
        if not os.path.isfile(best_name):
            if not self.download_model(self.name):
                # 如果下载不成功
                print(f'警告: 下载模型{self.name}不成功。请重试或手动从https://huggingface.co/alibabasglab/{self.name}/tree/main下载!')
                return

        if isinstance(self.model, nn.ModuleList):
            with open(best_name, 'r') as f:
                model_name = f.readline().strip()
                checkpoint_path = os.path.join(self.args.checkpoint_dir, model_name)
                self._load_model(self.model[0], checkpoint_path, model_key='mossformer')
                model_name = f.readline().strip()
                checkpoint_path = os.path.join(self.args.checkpoint_dir, model_name)
                self._load_model(self.model[1], checkpoint_path, model_key='generator')
        else:
            # 从文件中读取模型的检查点名称
            with open(best_name, 'r') as f:
                model_name = f.readline().strip()
            # 形成模型检查点的完整路径
            checkpoint_path = os.path.join(self.args.checkpoint_dir, model_name)
            self._load_model(self.model, checkpoint_path, model_key='model')

    def _load_model(self, model, checkpoint_path, model_key=None):
        # 将检查点文件加载到内存中（map_location确保与不同设备的兼容性）
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        # 将模型的状态字典（权重和偏差）加载到当前模型中
        if model_key in checkpoint:
            pretrained_model = checkpoint[model_key]
        else:
            pretrained_model = checkpoint
        state = model.state_dict()
        for key in state.keys():
            if key in pretrained_model and state[key].shape == pretrained_model[key].shape:
                state[key] = pretrained_model[key]
            elif key.replace('module.', '') in pretrained_model and state[key].shape == pretrained_model[key.replace('module.', '')].shape:
                 state[key] = pretrained_model[key.replace('module.', '')]
            elif 'module.'+key in pretrained_model and state[key].shape == pretrained_model['module.'+key].shape:
                 state[key] = pretrained_model['module.'+key]
            elif self.print: print(f'{key}未加载')
        model.load_state_dict(state)

    def decode(self):
        """
        使用加载的模型解码输入音频数据并确保输出与原始音频长度匹配。

        此方法通过语音模型处理音频（例如，用于增强、分离等），
        并将结果音频截断以匹配原始输入的长度。该方法支持多位说话人，
        如果模型处理多说话人音频。

        返回:
        output_audio: 处理后的音频，截断为输入音频长度。 
                  如果处理多说话人音频，则返回每位说话人的截断音频输出列表。
        """
        # 使用加载的模型在给定设备上解码音频（例如CPU或GPU）
        output_audios = []
        for i in range(len(self.data['audio'])):
            output_audio = decode_one_audio(self.model, self.device, self.data['audio'][i], self.args)
            # 确保解码输出与输入音频的长度匹配
            if isinstance(output_audio, list):
                # 如果是多说话人音频（输出列表），将每位说话人的音频截断到输入长度
                for spk in range(self.args.num_spks):
                    output_audio[spk] = output_audio[spk][:self.data['audio_len']]
            else:
                # 单个输出，截断到输入音频长度
                output_audio = output_audio[:self.data['audio_len']]
            output_audios.append(output_audio)
            
        if isinstance(output_audios[0], list):
            output_audios_np = []
            for spk in range(self.args.num_spks):
                output_audio_buf = []
                for i in range(len(output_audios)):
                    output_audio_buf.append(output_audios[i][spk])
                    #output_audio_buf = np.vstack((output_audio_buf, output_audios[i][spk])).T
                output_audios_np.append(np.array(output_audio_buf))
        else:
            output_audios_np = np.array(output_audios)
        return output_audios_np

    def process(self, input_path, online_write=False, output_path=None):
        """
        从指定的输入路径加载并处理音频文件。可选地，
        将输出音频文件写入指定的输出目录。
        
        参数:
            input_path (str): 输入音频文件或文件夹的路径。
            online_write (bool): 是否实时将处理后的音频写入磁盘。
            output_path (str): 写入输出文件的可选路径。如果为None，输出
                               将存储在self.result中。
        
        返回:
            dict或ndarray: 处理后的音频结果，根据处理的音频文件数量，
                            可能是字典或单个数组。
                            如果启用了online_write，则返回None。
        """
        
        self.result = {}
        self.args.input_path = input_path
        data_reader = DataReader(self.args)  # 初始化数据读取器以加载音频文件


        # 检查是否启用了在线写入
        if online_write:
            output_wave_dir = self.args.output_dir  # 设置默认输出目录
            if isinstance(output_path, str):  # 如果提供了特定的输出路径，则使用它
                output_wave_dir = os.path.join(output_path, self.name)
            # 如果输出目录不存在，则创建
            if not os.path.isdir(output_wave_dir):
                os.makedirs(output_wave_dir)
        
        num_samples = len(data_reader)  # 获取要处理的样本总数
        print(f'运行 {self.name} ...')  # 显示正在使用的模型

        if self.args.task == 'target_speaker_extraction':
            from utils.video_process import process_tse
            assert online_write == True
            process_tse(self.args, self.model, self.device, data_reader, output_wave_dir)
        else:
            # 在推断过程中禁用梯度计算以提高效率
            with torch.no_grad():
                for idx in tqdm(range(num_samples)):  # 遍历所有音频样本
                    self.data = {}
                    # 从数据读取器读取音频、波形ID和音频长度
                    input_audio, wav_id, input_len, scalars, audio_info = data_reader[idx]
                    # 将输入音频和元数据存储在self.data中
                    self.data['audio'] = input_audio
                    self.data['id'] = wav_id
                    self.data['audio_len'] = input_len
                    self.data.update(audio_info)
                    
                    # 执行音频解码/处理
                    output_audios = self.decode()

                    # 执行音频重新归一化
                    if not isinstance(output_audios, list):
                        if len(scalars) > 1:
                            for i in range(len(scalars)):
                                output_audios[:,i] = output_audios[:,i] * scalars[i]
                        else:
                                output_audios = output_audios * scalars[0]
                        
                    if online_write:
                        # 如果启用了在线写入，将输出音频保存到文件中
                        if isinstance(output_audios, list):
                            # 对于多说话人输出，分别保存每个说话人的输出
                            for spk in range(self.args.num_spks):
                                output_file = os.path.join(output_wave_dir, wav_id.replace('.'+self.data['ext'], f'_s{spk+1}.'+self.data['ext']))
                                self.write_audio(output_file, key=None, spk=spk, audio=output_audios)
                        else:
                            # 单说话人或标准输出
                            output_file = os.path.join(output_wave_dir, wav_id)
                            self.write_audio(output_file, key=None, spk=None, audio=output_audios)
                    else:
                        # 如果不写入磁盘，则将输出存储在结果字典中
                        self.result[wav_id] = output_audios
            
            # 如果不写入磁盘，则返回处理结果
            if not online_write:
                if len(self.result) == 1:
                    # 如果只有一个结果，直接返回它
                    return next(iter(self.result.values()))
                else:
                    # 否则，返回整个结果字典
                    return self.result

    def write_audio(self, output_path, key=None, spk=None, audio=None):
        """
        此函数将音频信号写入输出文件，根据提供的参数和实例的内部设置
        应用必要的转换，如重采样、通道处理和格式转换。
        
        参数:
            output_path (str): 音频将保存的文件路径。
            key (str, optional): 如果未提供音频，则用于从内部结果字典中检索音频的键。
            spk (str, optional): 特定说话人标识符，用于从多说话人数据集或结果中
                                提取特定说话人的音频。
            audio (numpy.ndarray, optional): 包含要写入的音频数据的numpy数组。
                                如果提供，则忽略key和spk。
        """
        
        if audio is not None:
            if spk is not None:
                result_ = audio[spk]
            else:
                result_ = audio
        else:
            if spk is not None:
                result_ = self.result[key][spk]
            else:
                result_ = self.result[key]
                
        if self.data['sample_rate'] != self.args.sampling_rate:
            if self.data['channels'] == 2:
                left_channel = librosa.resample(result_[0,:], orig_sr=self.args.sampling_rate, target_sr=self.data['sample_rate'])
                right_channel = librosa.resample(result_[1,:], orig_sr=self.args.sampling_rate, target_sr=self.data['sample_rate'])
                result = np.vstack((left_channel, right_channel)).T
            else:
                result = librosa.resample(result_[0,:], orig_sr=self.args.sampling_rate, target_sr=self.data['sample_rate'])
        else:
            if self.data['channels'] == 2:
                left_channel = result_[0,:]
                right_channel = result_[1,:]
                result = np.vstack((left_channel, right_channel)).T
            else:
                result = result_[0,:]
                
        if self.data['sample_width'] == 4: ##32位浮点
            MAX_WAV_VALUE = 2147483648.0
            np_type = np.int32
        elif self.data['sample_width'] == 2: ##16位整数
            MAX_WAV_VALUE = 32768.0
            np_type = np.int16
        else:
            self.data['sample_width'] = 2 ##16位整数
            MAX_WAV_VALUE = 32768.0
            np_type = np.int16
                        
        result = result * MAX_WAV_VALUE
        result = result.astype(np_type)
        audio_segment = AudioSegment(
            result.tobytes(),  # 原始音频数据（字节）
            frame_rate=self.data['sample_rate'],  # 采样率
            sample_width=self.data['sample_width'],          # 每个样本的字节数
            channels=self.data['channels']               # 通道数
        )
        audio_format = 'ipod' if self.data['ext'] in ['m4a', 'aac'] else self.data['ext']
        audio_segment.export(output_path, format=audio_format)
                    
    def write(self, output_path, add_subdir=False, use_key=False):
        """
        将处理后的音频结果写入指定的输出路径。

        参数:
            output_path (str): 处理后的音频将保存的目录或文件路径。如果未
                               提供，默认为self.args.output_dir。
            add_subdir (bool): 如果为True，将模型名称作为子目录附加到输出路径。
            use_key (bool): 如果为True，使用结果字典的键（音频文件ID）作为文件名。

        返回:
            None: 输出写入磁盘，不返回数据。
        """

        # 确保输出路径是字符串。如果未提供，使用默认输出目录
        if not isinstance(output_path, str):
            output_path = self.args.output_dir

        # 如果启用了add_subdir，为模型名称创建子目录
        if add_subdir:
            if os.path.isfile(output_path):
                print(f'文件存在: {output_path}, 移除它并重试!')
                return
            output_path = os.path.join(output_path, self.name)
            if not os.path.isdir(output_path):
                os.makedirs(output_path)

        # 使用键作为文件名时确保正确设置目录
        if use_key and not os.path.isdir(output_path):
            if os.path.exists(output_path):
                print(f'文件存在: {output_path}, 移除它并重试!')
                return
            os.makedirs(output_path)
        # 如果不使用键且输出路径是目录，则检查冲突
        if not use_key and os.path.isdir(output_path):
            print(f'目录存在: {output_path}, 移除它并重试!')
            return

        # 遍历结果字典以将处理后的音频写入磁盘
        for key in self.result:
            if use_key:
                # 如果使用键，根据结果字典的键（音频ID）格式化文件名
                if isinstance(self.result[key], list):  # 对于多说话人输出
                    for spk in range(self.args.num_spks):
                        output_file = os.path.join(output_path, key.replace('.'+self.data['ext'], f'_s{spk+1}.'+self.data['ext']))
                        self.write_audio(output_file, key, spk)
                else:
                    output_file = os.path.join(output_path, key)
                    self.write_audio(output_path, key)
            else:
                # 如果不使用键，直接将音频写入指定的输出路径
                if isinstance(self.result[key], list):  # 对于多说话人输出
                    for spk in range(self.args.num_spks):
                        output_file = output_path.replace('.'+self.data['ext'], f'_s{spk+1}.'+self.data['ext'])
                        self.write_audio(output_file, key, spk)
                else:
                    self.write_audio(output_path, key)
                    
# 特定子任务的模型类

class CLS_FRCRN_SE_16K(SpeechModel):
    """
    SpeechModel的子类，使用
    FRCRN架构实现16 kHz语音增强的语音增强模型。
    
    参数:
        args (Namespace): 包含模型配置和路径的参数解析器。
    """

    def __init__(self, args):
        # 初始化父SpeechModel类
        super(CLS_FRCRN_SE_16K, self).__init__(args)
        
        # 导入16 kHz的FRCRN语音增强模型
        from models.frcrn_se.frcrn import FRCRN_SE_16K
        
        # 初始化模型
        self.model = FRCRN_SE_16K(args).model
        self.name = 'FRCRN_SE_16K'
        
        # 加载预训练模型检查点
        self.load_model()
        
        # 将模型移动到适当的设备（GPU/CPU）
        if args.use_cuda == 1:
            self.model.to(self.device)
        
        # 将模型设置为评估模式（无梯度计算）
        self.model.eval()

class CLS_MossFormer2_SE_48K(SpeechModel):
    """
    SpeechModel的子类，实现MossFormer2架构用于
    48 kHz语音增强。
    
    参数:
        args (Namespace): 包含模型配置和路径的参数解析器。
    """

    def __init__(self, args):
        # 初始化父SpeechModel类
        super(CLS_MossFormer2_SE_48K, self).__init__(args)
        
        # 导入48 kHz的MossFormer2语音增强模型
        from models.mossformer2_se.mossformer2_se_wrapper import MossFormer2_SE_48K
        
        # 初始化模型
        self.model = MossFormer2_SE_48K(args).model
        self.name = 'MossFormer2_SE_48K'
        
        # 加载预训练模型检查点
        self.load_model()
        
        # 将模型移动到适当的设备（GPU/CPU）
        if args.use_cuda == 1:
            self.model.to(self.device)
        
        # 将模型设置为评估模式（无梯度计算）
        self.model.eval()

class CLS_MossFormer2_SR_48K(SpeechModel):
    """
    SpeechModel的子类，实现MossFormer2架构用于
    48 kHz语音超分辨率。
    
    参数:
        args (Namespace): 包含模型配置和路径的参数解析器。
    """

    def __init__(self, args):
        # 初始化父SpeechModel类
        super(CLS_MossFormer2_SR_48K, self).__init__(args)
        
        # 导入48 kHz的MossFormer2语音增强模型
        from models.mossformer2_sr.mossformer2_sr_wrapper import MossFormer2_SR_48K
        
        # 初始化模型
        self.model = nn.ModuleList()
        self.model.append(MossFormer2_SR_48K(args).model_m)
        self.model.append(MossFormer2_SR_48K(args).model_g)
        self.name = 'MossFormer2_SR_48K'
        
        # 加载预训练模型检查点
        self.load_model()
        
        # 将模型移动到适当的设备（GPU/CPU）
        if args.use_cuda == 1:
            for model in self.model:
                model.to(self.device)
        
        # 将模型设置为评估模式（无梯度计算）
        for model in self.model:
            model.eval()
        self.model[1].remove_weight_norm()

class CLS_MossFormerGAN_SE_16K(SpeechModel):
    """
    SpeechModel的子类，实现MossFormerGAN架构用于
    16 kHz语音增强，利用基于GAN的语音处理。
    
    参数:
        args (Namespace): 包含模型配置和路径的参数解析器。
    """

    def __init__(self, args):
        # 初始化父SpeechModel类
        super(CLS_MossFormerGAN_SE_16K, self).__init__(args)
        
        # 导入16 kHz的MossFormerGAN语音增强模型
        from models.mossformer_gan_se.generator import MossFormerGAN_SE_16K
        
        # 初始化模型
        self.model = MossFormerGAN_SE_16K(args).model
        self.name = 'MossFormerGAN_SE_16K'
        
        # 加载预训练模型检查点
        self.load_model()
        
        # 将模型移动到适当的设备（GPU/CPU）
        if args.use_cuda == 1:
            self.model.to(self.device)
        
        # 将模型设置为评估模式（无梯度计算）
        self.model.eval()

class CLS_MossFormer2_SS_16K(SpeechModel):
    """
    SpeechModel的子类，实现MossFormer2架构用于
    16 kHz语音分离。
    
    参数:
        args (Namespace): 包含模型配置和路径的参数解析器。
    """

    def __init__(self, args):
        # 初始化父SpeechModel类
        super(CLS_MossFormer2_SS_16K, self).__init__(args)
        
        # 导入16 kHz的MossFormer2语音分离模型
        from models.mossformer2_ss.mossformer2 import MossFormer2_SS_16K
        
        # 初始化模型
        self.model = MossFormer2_SS_16K(args).model
        self.name = 'MossFormer2_SS_16K'
        
        # 加载预训练模型检查点
        self.load_model()
        
        # 将模型移动到适当的设备（GPU/CPU）
        if args.use_cuda == 1:
            self.model.to(self.device)
        
        # 将模型设置为评估模式（无梯度计算）
        self.model.eval()


class CLS_AV_MossFormer2_TSE_16K(SpeechModel):
    """
    SpeechModel的子类，实现使用
    AV-MossFormer2架构的音频-视觉（AV）模型，用于16 kHz的目标说话人提取（TSE）。
    该模型利用音频和视觉线索执行说话人提取。
    
    参数:
        args (Namespace): 包含模型配置和路径的参数解析器。
    """

    def __init__(self, args):
        # 初始化父SpeechModel类
        super(CLS_AV_MossFormer2_TSE_16K, self).__init__(args)
        
        # 导入16 kHz的AV-MossFormer2目标语音增强模型
        from models.av_mossformer2_tse.av_mossformer2 import AV_MossFormer2_TSE_16K
        
        # 初始化模型
        self.model = AV_MossFormer2_TSE_16K(args).model
        self.name = 'AV_MossFormer2_TSE_16K'
        
        # 加载预训练模型检查点
        self.load_model()
        
        # 将模型移动到适当的设备（GPU/CPU）
        if args.use_cuda == 1:
            self.model.to(self.device)
        
        # 将模型设置为评估模式（无梯度计算）
        self.model.eval()


