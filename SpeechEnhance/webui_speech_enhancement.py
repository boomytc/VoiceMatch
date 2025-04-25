# -*- coding: utf-8 -*-
import gradio as gr
import os
import time
import torch
import tempfile
import shutil
import sys
from clearvoice import ClearVoice

# --- 全局模型变量 ---
cv_se = None
cv_sr = None
model_load_lock = False # 简单的锁，防止并发加载

# --- 模型加载函数 ---
def load_models(do_se, do_sr, progress=gr.Progress(track_tqdm=True)):
    """按需加载模型"""
    global cv_se, cv_sr, model_load_lock
    if model_load_lock:
        print("等待其他模型加载完成...")
        while model_load_lock:
            time.sleep(0.1)

    model_load_lock = True
    load_se_needed = do_se and cv_se is None
    load_sr_needed = do_sr and cv_sr is None

    try:
        if load_se_needed:
            print("正在加载语音增强模型 (MossFormer2_SE_48K)...")
            progress(0.1, desc="正在加载语音增强模型...")
            cv_se = ClearVoice(
                task='speech_enhancement',
                model_names=['MossFormer2_SE_48K']
            )
            print("语音增强模型加载完成。")
            progress(0.5 if load_sr_needed else 1.0, desc="语音增强模型加载完成。")

        if load_sr_needed:
            print("正在加载语音超分辨率模型 (MossFormer2_SR_48K)...")
            progress(0.6, desc="正在加载语音超分辨率模型...")
            cv_sr = ClearVoice(
                task='speech_super_resolution',
                model_names=['MossFormer2_SR_48K']
            )
            print("语音超分辨率模型加载完成。")
            progress(1.0, desc="语音超分辨率模型加载完成。")

    except Exception as e:
        print(f"加载模型时出错: {e}")
        # 重置模型变量，以便下次尝试重新加载
        if load_se_needed: cv_se = None
        if load_sr_needed: cv_sr = None
        raise gr.Error(f"加载模型失败: {e}")
    finally:
        model_load_lock = False

# --- 音频处理核心函数 ---
def enhance_speech(input_audio_path, do_se, do_sr, progress=gr.Progress(track_tqdm=True)):
    """处理单个音频文件"""
    if not input_audio_path:
        raise gr.Error("错误：未提供输入音频文件。") # 使用 gr.Error

    if not do_se and not do_sr:
        raise gr.Error("错误：请至少选择一个处理任务（语音增强 或 语音超分辨率）。") # 使用 gr.Error

    start_time_total = time.time()
    status_messages = []
    output_dir = tempfile.mkdtemp() # 创建临时目录存放输出
    temp_files = [] # 存储中间临时文件路径

    try:
        # 1. 加载模型 (如果需要)
        progress(0, desc="检查并加载模型...")
        load_models(do_se, do_sr, progress) # 加载函数内部处理进度条

        # 2. 文件路径处理
        input_filename = os.path.basename(input_audio_path)
        input_name_no_ext, input_ext = os.path.splitext(input_filename)
        if not input_ext:
             input_ext = ".wav" # Gradio可能不提供扩展名，默认wav

        # 3. 任务执行
        current_input = input_audio_path
        final_output_path = None
        tasks_done = []

        # -- 执行语音增强 (如果需要) --
        if do_se:
            progress(0.1, desc="开始语音增强...")
            status_messages.append("开始语音增强...")
            se_start_time = time.time()
            try:
                output_wav_se = cv_se(
                    input_path=current_input,
                    online_write=False
                )
            except Exception as e:
                raise gr.Error(f"语音增强处理失败: {e}")
            se_end_time = time.time()
            se_duration = se_end_time - se_start_time
            status_messages.append(f"语音增强完成，耗时: {se_duration:.2f} 秒。")
            print(f"语音增强耗时: {se_duration:.2f} 秒")
            progress(0.5 if do_sr else 0.9, desc="语音增强完成 ({se_duration:.2f}s)")

            # 确定 SE 输出路径
            if do_sr: # 如果 SR 也要执行，SE 输出是临时的
                se_path = os.path.join(output_dir, f"{input_name_no_ext}_se_temp{input_ext}")
                temp_files.append(se_path)
            else: # 如果只执行 SE
                se_path = os.path.join(output_dir, f"{input_name_no_ext}_se{input_ext}")

            cv_se.write(output_wav_se, output_path=se_path)
            current_input = se_path # 更新下个任务的输入
            tasks_done.append("增强")
            final_output_path = se_path # 如果这是最后一个任务，这就是最终输出

        # -- 执行语音超分辨率 (如果需要) --
        if do_sr:
            progress(0.6, desc="开始语音超分辨率...")
            status_messages.append("开始语音超分辨率...")
            sr_start_time = time.time()
            try:
                output_wav_sr = cv_sr(
                    input_path=current_input, # 输入可能是原始文件或 SE 输出
                    online_write=False
                )
            except Exception as e:
                 raise gr.Error(f"语音超分辨率处理失败: {e}")
            sr_end_time = time.time()
            sr_duration = sr_end_time - sr_start_time
            status_messages.append(f"语音超分辨率完成，耗时: {sr_duration:.2f} 秒。")
            print(f"语音超分辨率耗时: {sr_duration:.2f} 秒")
            progress(0.9, desc="语音超分辨率完成 ({sr_duration:.2f}s)")

            # 确定 SR 输出路径
            suffix = "_sr" if not do_se else "_se_sr" # 根据是否执行了 SE 调整后缀
            sr_path = os.path.join(output_dir, f"{input_name_no_ext}{suffix}{input_ext}")

            cv_sr.write(output_wav_sr, output_path=sr_path)
            tasks_done.append("超分辨率")
            final_output_path = sr_path # SR 任务总是最后执行（如果执行的话）

        # 4. 计算总时间并返回结果
        end_time_total = time.time()
        total_duration = end_time_total - start_time_total
        tasks_str = " 和 ".join(tasks_done)
        status_messages.append(f"处理完成！任务: {tasks_str}。")
        status_messages.append(f"总耗时: {total_duration:.2f} 秒。")
        status_messages.append(f"输出文件: {os.path.basename(final_output_path)}")
        print(f"总处理耗时: {total_duration:.2f} 秒")
        progress(1.0, desc="处理完成！")

        # 将最终文件移动到 Gradio 可以访问的位置（如果需要）
        # Gradio 通常能处理临时目录中的文件路径
        target_temp_path = os.path.join(tempfile.gettempdir(), os.path.basename(final_output_path))
        if os.path.exists(target_temp_path):
            print(f"警告：目标临时文件已存在，将覆盖: {target_temp_path}")
            try:
                os.remove(target_temp_path)
            except OSError as e:
                print(f"错误：无法删除已存在的目标文件: {e}")
                raise  # 重新引发错误，因为无法继续

        final_gradio_path = shutil.move(final_output_path, target_temp_path)
        final_output_path = final_gradio_path # 更新路径为Gradio可访问的路径
        # 清理包含临时文件的目录，因为最终文件已被移走
        if os.path.exists(output_dir):
             try:
                 shutil.rmtree(output_dir)
             except Exception as cleanup_error:
                 print(f"清理源临时目录 {output_dir} 时出错: {cleanup_error}")

        return final_output_path, "\n".join(status_messages)

    except Exception as e:
        # 不再需要手动处理错误返回，gr.Error 会处理
        # 清理仍然需要执行
        if os.path.exists(output_dir):
            try:
                shutil.rmtree(output_dir)
            except Exception as cleanup_error:
                print(f"错误处理中清理临时目录 {output_dir} 时出错: {cleanup_error}")
        # 重新抛出异常，让 Gradio 的错误处理机制捕获
        raise e

    # finally 块不再需要，因为成功和失败路径都已处理清理和返回

# --- Gradio 界面定义 ---
with gr.Blocks(title="语音增强/超分") as demo:
    gr.Markdown(
        """
        # 语音增强 & 超分辨率 🚀
        上传你的音频文件，选择需要的处理任务，然后点击“开始处理”。
        """
    )
    with gr.Row():
        with gr.Column(scale=1):
            input_audio = gr.Audio(type="filepath", label="上传音频文件 (Upload Audio)")
            with gr.Row():
                se_checkbox = gr.Checkbox(label="语音增强 (Speech Enhancement)", value=True)
                sr_checkbox = gr.Checkbox(label="语音超分辨率 (Speech Super-Resolution)", value=False)
            process_button = gr.Button("开始处理 (Start Processing)", variant="primary")
        with gr.Column(scale=1):
            output_audio = gr.Audio(label="处理结果 (Processed Audio)", type="filepath")
            status_textbox = gr.Textbox(label="处理状态 (Processing Status)", lines=5, interactive=False)

    process_button.click(
        fn=enhance_speech,
        inputs=[input_audio, se_checkbox, sr_checkbox],
        outputs=[output_audio, status_textbox]
    )

# --- 启动界面 ---
if __name__ == "__main__":
    # 检查是否有可用的GPU
    gpu_available = torch.cuda.is_available()
    print(f"GPU 可用: {'是' if gpu_available else '否'}")
    if not gpu_available:
        print("警告: 未检测到 CUDA GPU。模型将在 CPU 上运行，速度可能较慢。")

    # 设置多进程启动方法
    torch.multiprocessing.set_start_method('spawn', force=True)
    print("设置 multiprocessing 启动方法为 'spawn'")

    demo.queue().launch(inbrowser=True) # 使用 queue() 支持更长的处理时间