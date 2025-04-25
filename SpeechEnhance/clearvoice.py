from network_wrapper import network_wrapper
import os
import warnings
warnings.filterwarnings("ignore")

class ClearVoice:
    """ 提供给最终用户用于执行语音处理的主要类接口
        此类提供所需的模型以执行指定任务
    """
    def __init__(self, task, model_names):
        """ 加载指定任务所需的模型。执行所有给定模型并返回所有结果。
   
        参数:
        ----------
        task: str
            匹配以下任务之一的任务: 
            'speech_enhancement' (语音增强)
            'speech_separation' (语音分离)
            'target_speaker_extraction' (目标说话人提取)
        model_names: str 或 list of str
            匹配以下模型之一的模型名称: 
            'FRCRN_SE_16K'
            'MossFormer2_SE_48K'
            'MossFormerGAN_SE_16K'
            'MossFormer2_SS_16K'
            'AV_MossFormer2_TSE_16K'

        返回值:
        --------
        一个 ModelsList 对象，可以运行以获取所需结果
        """        
        self.network_wrapper = network_wrapper()
        self.models = []
        for model_name in model_names:
            model = self.network_wrapper(task, model_name)
            self.models += [model]  
            
    def __call__(self, input_path, online_write=False, output_path=None):
        results = {}
        for model in self.models:
            result = model.process(input_path, online_write, output_path)
            if not online_write:
                results[model.name] = result

        if not online_write:
            if len(results) == 1:
                return results[model.name]
            else:
                return results

    def write(self, results, output_path):
        add_subdir = False
        use_key = False
        if len(self.models) > 1: add_subdir = True #multi_model is True        
        for model in self.models:
            if isinstance(results, dict):
                if model.name in results: 
                   if len(results[model.name]) > 1: use_key = True
                       
                else:
                   if len(results) > 1: use_key = True #multi_input is True
            break

        for model in self.models:
            model.write(output_path, add_subdir, use_key)
