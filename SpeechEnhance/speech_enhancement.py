from clearvoice import ClearVoice
import os
import sys
import tqdm
import torch
from multiprocessing import Pool, cpu_count, set_start_method
import argparse
import time

def process_file(task_args):
    """单个文件处理函数"""
    input_path, gpu_id, do_se, do_sr, output_path = task_args
    
    # 设置当前进程使用的GPU (如果可用)
    if torch.cuda.is_available() and torch.cuda.device_count() > gpu_id:
        try:
            torch.cuda.set_device(gpu_id)
        except Exception as e:
            print(f"警告: 无法在进程中设置 GPU {gpu_id}: {e}")

    # 初始化模型 (根据需要)
    cv_se = None
    if do_se:
        cv_se = ClearVoice(
            task='speech_enhancement',
            model_names=['MossFormer2_SE_48K']
        )

    cv_sr = None
    if do_sr:
        cv_sr = ClearVoice(
            task='speech_super_resolution',
            model_names=['MossFormer2_SR_48K']
        )

    # 文件路径处理
    input_dir = os.path.dirname(input_path)
    input_filename = os.path.basename(input_path)
    input_name_no_ext = os.path.splitext(input_filename)[0]
    input_ext = os.path.splitext(input_filename)[1]

    # 任务执行
    current_input = input_path
    final_output_path = None
    temp_files = []
    tasks_done = []
    se_path = None

    try:
        # 执行语音增强 (如果需要)
        if do_se:
            se_start_time = time.time()
            output_wav_se = cv_se(
                input_path=current_input,
                online_write=False
            )
            se_end_time = time.time()
            print(f"\n\033[33m【模型耗时】语音增强模型: {se_end_time - se_start_time:.2f} 秒\033[0m")

            # 确定 SE 输出路径
            if do_sr: 
                se_path = os.path.join(input_dir, f"{input_name_no_ext}_se_temp{input_ext}")
                temp_files.append(se_path)
            elif not do_sr:
                se_path = os.path.join(input_dir, f"{input_name_no_ext}_se{input_ext}")
                # 使用用户指定输出路径（仅 SE 单任务时生效）
                if output_path and not do_sr:
                    se_path = output_path
            cv_se.write(output_wav_se, output_path=se_path)
            current_input = se_path
            tasks_done.append("增强")
            final_output_path = se_path

        # 执行语音超分辨率 (如果需要)
        if do_sr:
            sr_start_time = time.time()
            output_wav_sr = cv_sr(
                input_path=current_input,
                online_write=False
            )
            sr_end_time = time.time()
            print(f"\n\033[33m【模型耗时】超分辨率模型: {sr_end_time - sr_start_time:.2f} 秒\033[0m")

            # 确定 SR 输出路径
            suffix = "_sr" if not do_se else "_se_sr"
            sr_path = os.path.join(input_dir, f"{input_name_no_ext}{suffix}{input_ext}")
            # 使用用户指定输出路径
            if output_path:
                sr_path = output_path
            cv_sr.write(output_wav_sr, output_path=sr_path)
            tasks_done.append("超分辨率")
            final_output_path = sr_path

        return (True, final_output_path)

    except Exception as e:
        print(f"\n处理文件 {input_filename} 时出错: {str(e)}")
        if se_path and os.path.exists(se_path) and se_path in temp_files:
             try:
                 os.remove(se_path)
             except OSError:
                 pass
        return (False, None)

    finally:
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except OSError as err:
                    print(f"删除临时文件 {temp_file} 时出错: {err}")


def process_directory(input_dir, do_se=True, do_sr=True):
    start_time_batch = time.time()
    # 获取可用的GPU数量
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("警告：未检测到GPU，将使用CPU处理。多进程可能不会带来显著加速。")
        num_gpus = 1
        num_processes = max(1, cpu_count() - 2)
    else:
        num_processes = num_gpus

    # 获取所有音频文件
    audio_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.wav', '.mp3', '.flac', '.m4a', '.ogg'))]

    if not audio_files:
        print(f"在目录 {input_dir} 中未找到支持的音频文件（.wav, .mp3, .flac, .m4a, .ogg）")
        return

    # 准备任务列表，为每个文件分配GPU ID 和任务标志
    tasks = []
    for i, audio_file in enumerate(audio_files):
        input_path = os.path.join(input_dir, audio_file)
        gpu_id = i % num_gpus
        tasks.append((input_path, gpu_id, do_se, do_sr, None))

    # 创建进程池
    start_method = 'spawn' if sys.platform == 'win32' else 'fork'
    try:
        set_start_method(start_method, force=True)
    except RuntimeError:
         print(f"注意：无法强制设置多处理启动方法为 '{start_method}'。可能已设置。")

    with Pool(processes=num_processes) as pool:
        results = list(tqdm.tqdm(
            pool.imap_unordered(process_file, tasks),
            total=len(tasks),
            desc="处理文件进度"
        ))

    # 统计处理结果
    success_count = sum(1 for success, _ in results if success)
    failed_files = len(tasks) - success_count

    end_time_batch = time.time()
    duration_batch = end_time_batch - start_time_batch

    tasks_performed_str = []
    if do_se: tasks_performed_str.append("语音增强")
    if do_sr: tasks_performed_str.append("超分辨率")
    tasks_str = " 和 ".join(tasks_performed_str) if tasks_performed_str else "未执行任何任务"

    print(f"\n\033[32m--- 批处理完成 ({tasks_str}) ---\033[0m")
    print(f"\033[36m总文件数: {len(tasks)}\033[0m")
    print(f"\033[32m成功处理: {success_count}\033[0m")
    if failed_files > 0:
        print(f"\033[31m处理失败: {failed_files}\033[0m")
    print(f"\033[36m总耗时: {duration_batch:.2f} 秒\033[0m")
    avg_time = duration_batch / success_count if success_count > 0 else 0
    print(f"\033[33m平均每个音频处理时间: {avg_time:.2f} 秒\033[0m")
    print(f"\033[35m输出文件保存在目录: {input_dir}\033[0m")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用 ClearVoice 进行语音增强和/或超分辨率处理（单文件或批量）。")
    parser.add_argument("input_path", help="输入音频文件或文件夹路径。")
    parser.add_argument("-se", "--speech_enhancement", action="store_true", help="执行语音增强任务。")
    parser.add_argument("-sr", "--speech_super_resolution", action="store_true", help="执行语音超分辨率任务。")
    parser.add_argument("-o", "--output_path", help="指定单文件模式下的输出路径。")

    args = parser.parse_args()
    input_path = args.input_path
    do_se = args.speech_enhancement
    do_sr = args.speech_super_resolution
    if not do_se and not do_sr:
        print("未指定任务 (-se 或 -sr)，默认执行语音增强和超分辨率。")
        do_se = True
        do_sr = True

    if os.path.isdir(input_path):
        process_directory(input_path, do_se, do_sr)
    elif os.path.isfile(input_path):
        start_time = time.time()
        success, final_path = process_file((input_path, 0, do_se, do_sr, args.output_path))
        duration = time.time() - start_time
        if success:
            tasks_done = []
            if do_se: tasks_done.append("增强")
            if do_sr: tasks_done.append("超分辨率")
            tasks_str = " 和 ".join(tasks_done)
            print(f"\n\033[32m处理完成！{tasks_str}处理后的音频已保存到: {final_path}\033[0m")
            print(f"\033[36m总运行时间: {duration:.2f} 秒\033[0m")
            sys.exit(0)
        else:
            sys.exit(1)
    else:
        print(f"错误：提供的路径 '{input_path}' 既不是文件也不是目录")
        sys.exit(1)
