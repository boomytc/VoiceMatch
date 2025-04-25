import gradio as gr
import os
import shutil
import time
import glob
from pathlib import Path

# 假设 mvector 包在项目路径下或者已安装
from mvector.predict import MVectorPredictor

# --- 全局配置 ---
CONFIG_PATH = 'configs/cam++.yml'
MODEL_PATH = 'models/CAMPPlus_Fbank/best_model/'
AUDIO_DB_PATH = 'audio_db/'
BACKUP_ROOT_PATH = 'backups/' # 备份文件存放根目录
THRESHOLD = 0.6
USE_GPU = True

# --- 初始化预测器 ---
try:
    predictor = MVectorPredictor(
        configs=CONFIG_PATH,
        model_path=MODEL_PATH,
        threshold=THRESHOLD,
        audio_db_path=AUDIO_DB_PATH,
        use_gpu=USE_GPU
    )
    print("MVectorPredictor 初始化成功.")
except Exception as e:
    print(f"错误：无法初始化 MVectorPredictor: {e}")
    # 在 Gradio 界面中显示错误信息可能更好
    predictor = None # 设置为 None 以便后续检查

# --- 确保目录存在 ---
os.makedirs(AUDIO_DB_PATH, exist_ok=True)
os.makedirs(BACKUP_ROOT_PATH, exist_ok=True)

# --- Gradio UI 函数 ---

def get_user_list():
    """获取当前用户列表"""
    if predictor:
        try:
            users = predictor.get_users()
            # 去除重复用户
            unique_users = list(set(users)) if users else []
            return unique_users
        except Exception as e:
            print(f"获取用户列表时出错: {e}")
            return []
    return []

# --- 注册功能逻辑 ---
def handle_register(username, mode, single_file_path, batch_dir_path):
    if not predictor:
        return "错误：预测器未初始化", gr.update(choices=get_user_list())
    if not username:
        return "错误：请输入用户名", gr.update(choices=get_user_list())

    status_message = ""
    try:
        if mode == "单个文件":
            if not single_file_path:
                return "错误：请选择单个音频文件", gr.update(choices=get_user_list())
            # predictor.register 接受音频数据（numpy 数组或路径）
            # Gradio 的 gr.Audio(type="filepath") 返回文件路径
            result = predictor.register(user_name=username, audio_data=single_file_path)
            if result:
                status_message = f"用户 '{username}' 使用文件 '{os.path.basename(single_file_path)}' 注册成功。"
            else:
                 # register 可能返回 False 或 None 表示失败，或引发异常
                status_message = f"用户 '{username}' 使用文件 '{os.path.basename(single_file_path)}' 注册失败。可能已存在或音频无效。"

        elif mode == "文件夹批量":
            if not batch_dir_path:
                return "错误：请选择包含音频文件的文件夹", gr.update(choices=get_user_list())

            # batch_dir_path 是 gr.File(file_count="directory") 返回的临时目录路径
            audio_files = glob.glob(os.path.join(batch_dir_path, '*.wav')) + \
                          glob.glob(os.path.join(batch_dir_path, '*.mp3')) # 可根据需要添加更多格式

            if not audio_files:
                return f"错误：在文件夹 '{batch_dir_path}' 中未找到支持的音频文件 (.wav, .mp3)", gr.update(choices=get_user_list())

            success_count = 0
            fail_count = 0
            registered_files = []
            for audio_file in audio_files:
                try:
                    # 注意：批量注册时，每个文件都会尝试注册到同一个用户名下
                    # MVectorPredictor 的 register 会自动处理，如果特征已存在则可能跳过或更新
                    result = predictor.register(user_name=username, audio_data=audio_file)
                    if result:
                        success_count += 1
                        registered_files.append(os.path.basename(audio_file))
                    else:
                        fail_count += 1
                except Exception as e:
                    print(f"注册文件 '{audio_file}' 时出错: {e}")
                    fail_count += 1
            status_message = f"文件夹批量注册完成: 用户 '{username}'，成功 {success_count} 个文件，失败 {fail_count} 个。"
            if registered_files:
                status_message += f"\n注册的文件: {', '.join(registered_files)}"
        else:
            return "错误：无效的注册模式", gr.update(choices=get_user_list())

    except Exception as e:
        status_message = f"注册时发生错误: {e}"
        print(f"注册错误详情: {e}")

    # 更新用户列表下拉菜单
    updated_users = get_user_list()
    return status_message, gr.update(choices=updated_users, value=updated_users[0] if updated_users else None)

# --- 切换注册模式UI ---
def toggle_register_ui(mode):
    if mode == "单个文件":
        return gr.update(visible=True), gr.update(visible=False)
    elif mode == "文件夹批量":
        return gr.update(visible=False), gr.update(visible=True)
    else:
        return gr.update(visible=True), gr.update(visible=False) # 默认为单个文件

# --- 对比功能逻辑 ---
def handle_contrast(audio_path1, audio_path2):
    if not predictor:
        return "错误：预测器未初始化", None
    if not audio_path1 or not audio_path2:
        return "错误：请同时提供两个音频文件进行对比", None

    status_message = ""
    score = None
    try:
        # predictor.contrast 需要两个音频路径
        score = predictor.contrast(audio_data1=audio_path1, audio_data2=audio_path2)
        if score is not None:
             # 保留几位小数，例如4位
            score = round(score, 4)
            status_message = f"文件 '{os.path.basename(audio_path1)}' 和 '{os.path.basename(audio_path2)}' 对比完成。"
        else:
             # contrast 可能返回 None 表示失败
            status_message = f"无法计算相似度。请检查音频文件是否有效。"

    except Exception as e:
        status_message = f"对比时发生错误: {e}"
        print(f"对比错误详情: {e}")
        score = None # 确保出错时返回 None

    return status_message, score

# --- 识别功能逻辑 ---
def handle_recognize(audio_path):
    if not predictor:
        return "错误：预测器未初始化", ""
    if not audio_path:
        return "错误：请选择需要识别的音频文件", ""

    status_message = ""
    result_message = ""
    try:
        # predictor.recognition 需要音频路径
        recognized_user, score = predictor.recognition(audio_data=audio_path)

        if recognized_user:
             # 保留几位小数，例如4位
            score = round(score, 4)
            status_message = f"文件 '{os.path.basename(audio_path)}' 识别完成。"
            result_message = f"识别结果: {recognized_user} (置信度: {score})"
        else:
            status_message = f"文件 '{os.path.basename(audio_path)}' 处理完成。"
            result_message = "未识别到匹配的用户 (可能低于阈值或库中无此人)"

    except Exception as e:
        status_message = f"识别时发生错误: {e}"
        print(f"识别错误详情: {e}")
        result_message = "识别出错"

    return status_message, result_message

# --- 管理声纹库功能逻辑 ---

def handle_refresh_users():
    """刷新用户列表下拉菜单"""
    users = get_user_list()
    # 同时清空详情和播放器
    return gr.update(choices=users, value=users[0] if users else None), "用户列表已刷新", "", gr.update(choices=[], value=None), None

def handle_view_details(username):
    """查看选定用户的详情（音频文件列表和路径）"""
    if not predictor:
        return "错误：预测器未初始化", "", gr.update(choices=[], value=None), None
    if not username:
        return "请先在下拉菜单中选择一个用户", "", gr.update(choices=[], value=None), None

    details = ""
    audio_file_paths = []
    status_message = f"正在获取用户 '{username}' 的详情..."
    audio_player_update = None

    try:
        user_audio_dir = os.path.join(predictor.audio_db_path, username)
        if not os.path.isdir(user_audio_dir):
            status_message = f"状态：用户 '{username}' 的音频目录不存在。"
            return status_message, details, gr.update(choices=[], value=None), audio_player_update

        # 列出目录下的 .wav 文件（或 predictor 支持的其他格式）
        # 使用 glob 获取完整路径
        found_files = glob.glob(os.path.join(user_audio_dir, '*.wav')) # 假设主要是 wav

        if not found_files:
            details = f"用户 '{username}' 当前没有注册的音频文件。"
            status_message = f"状态：用户 '{username}' 的详情已获取。"
            return status_message, details, gr.update(choices=[], value=None), audio_player_update

        # 提取文件名用于显示，保留完整路径用于下拉菜单和播放器
        file_names = [os.path.basename(f) for f in found_files]
        audio_file_paths = found_files # 完整路径列表

        details = f"用户 '{username}' 已注册的音频文件 ({len(file_names)}个):\n" + "\n".join(file_names)
        status_message = f"状态：已获取用户 '{username}' 的详情。"

        # 更新播放器为第一个文件 (如果存在)
        if audio_file_paths:
            audio_player_update = gr.update(value=audio_file_paths[0])

        # 更新下拉菜单选项和值
        return status_message, details, gr.update(choices=audio_file_paths, value=audio_file_paths[0]), audio_player_update

    except Exception as e:
        print(f"查看用户详情时出错: {e}")
        status_message = f"错误：查看用户 '{username}' 详情时出错: {e}"
        return status_message, details, gr.update(choices=[], value=None), audio_player_update

def handle_delete_user(username):
    """删除选定的用户"""
    if not predictor:
        return "错误：预测器未初始化", gr.update(choices=get_user_list())
    if not username:
        return "请先在下拉菜单中选择一个用户", gr.update(choices=get_user_list())

    try:
        # MVectorPredictor 的 remove_user 会处理特征索引和可选的音频文件
        # 确认 predictor.remove_user 是否删除文件，如果否则需手动删
        # 假设 predictor.remove_user 会删除关联的文件目录
        success = predictor.remove_user(user_name=username)

        # 手动删除文件夹以确保干净 (如果 predictor 不删除)
        user_audio_dir = os.path.join(predictor.audio_db_path, username)
        if os.path.exists(user_audio_dir):
             try:
                 shutil.rmtree(user_audio_dir)
                 print(f"手动删除了用户目录: {user_audio_dir}")
                 success = True # 即使 predictor.remove_user 返回 False，只要目录删了就算成功
             except Exception as e:
                 print(f"手动删除用户目录 '{user_audio_dir}' 失败: {e}")
                 # 根据需要决定是否覆盖 success 状态

        if success:
            status_message = f"用户 '{username}' 已成功删除。"
        else:
             # 检查用户是否真的不存在了
            if username not in predictor.get_users() and not os.path.exists(user_audio_dir):
                 status_message = f"用户 '{username}' 已成功删除 (remove_user可能返回False但实际已清理)。"
            else:
                 status_message = f"删除用户 '{username}' 失败。可能用户不存在或发生内部错误。"

    except Exception as e:
        status_message = f"删除用户时发生错误: {e}"
        print(f"删除用户错误详情: {e}")

    # 更新用户列表
    users = get_user_list()
    return status_message, gr.update(choices=users, value=users[0] if users else None)

def handle_rename_user(old_name, new_name):
    """重命名用户 (复杂操作)"""
    if not predictor:
        return "错误：预测器未初始化", gr.update(choices=get_user_list()), ""
    if not old_name:
        return "请先在下拉菜单中选择要重命名的用户", gr.update(choices=get_user_list()), ""
    if not new_name or old_name == new_name:
        return "请输入有效的新用户名，且不能与旧用户名相同", gr.update(choices=get_user_list()), ""
    if new_name in predictor.get_users():
        return f"错误：新用户名 '{new_name}' 已存在，请选择其他名称", gr.update(choices=get_user_list()), ""

    status_message = ""
    old_dir = os.path.join(predictor.audio_db_path, old_name)
    new_dir = os.path.join(predictor.audio_db_path, new_name)

    if not os.path.isdir(old_dir):
        return f"错误：找不到用户 '{old_name}' 的音频目录", gr.update(choices=get_user_list()), ""

    try:
        # 1. 备份旧用户音频文件路径
        audio_files_to_re_register = glob.glob(os.path.join(old_dir, '*.wav')) # + 其他格式

        # 2. 删除旧用户特征 (从索引中移除)
        predictor.remove_user(user_name=old_name)

        # 3. 重命名文件夹
        shutil.move(old_dir, new_dir)
        print(f"重命名目录: '{old_dir}' -> '{new_dir}'")

        # 4. 使用新用户名重新注册所有音频文件
        success_count = 0
        fail_count = 0
        registered_files = []
        for audio_file_path_in_new_dir in glob.glob(os.path.join(new_dir, '*.wav')):
            try:
                # 注意：批量注册时，每个文件都会尝试注册到同一个用户名下
                # MVectorPredictor 的 register 会自动处理，如果特征已存在则可能跳过或更新
                result = predictor.register(user_name=new_name, audio_data=audio_file_path_in_new_dir)
                if result:
                    success_count += 1
                    registered_files.append(os.path.basename(audio_file_path_in_new_dir))
                else:
                    fail_count += 1
            except Exception as reg_e:
                print(f"重注册文件 '{audio_file_path_in_new_dir}' 时出错: {reg_e}")
                fail_count += 1

        status_message = f"用户 '{old_name}' 已重命名为 '{new_name}'。重新注册: 成功 {success_count}, 失败 {fail_count}."
        if registered_files:
            status_message += f"\n注册的文件: {', '.join(registered_files)}"

    except Exception as e:
        status_message = f"重命名用户时发生错误: {e}"
        print(f"重命名用户错误详情: {e}")
        # 尝试回滚？（可能很困难，至少要恢复文件夹名称）
        if os.path.exists(new_dir) and not os.path.exists(old_dir):
            try:
                shutil.move(new_dir, old_dir)
                print("错误发生后尝试回滚目录重命名")
            except Exception as roll_e:
                print(f"回滚目录重命名失败: {roll_e}")

    # 更新用户列表并清空新名称输入框
    users = get_user_list()
    return status_message, gr.update(choices=users, value=new_name if new_name in users else (users[0] if users else None)), ""

def handle_clear_db(confirm_text):
    """清空整个声纹库"""
    if not predictor:
        return "错误：预测器未初始化", gr.update(choices=get_user_list())
    if confirm_text != "确认清空":
        return "请输入 '确认清空' 以确认操作", gr.update(choices=get_user_list())

    status_message = ""
    try:
        users_to_remove = list(predictor.get_users()) # 获取当前用户列表副本
        print(f"准备清空声纹库，将移除用户: {users_to_remove}")

        for user in users_to_remove:
            try:
                predictor.remove_user(user_name=user)
                user_dir = os.path.join(predictor.audio_db_path, user)
                if os.path.exists(user_dir):
                    shutil.rmtree(user_dir)
                    print(f"已删除用户目录: {user_dir}")
            except Exception as e_remove:
                print(f"移除用户 '{user}' 或其目录时出错: {e_remove}")
                # 继续尝试删除其他用户

        # 检查并删除可能的索引文件 (如果 predictor.remove_user 未完全清理)
        # 注意：需要知道 predictor 内部索引文件的确切名称和位置
        # predictor.audio_indexes_path 在 reference_gui.py 中有提到，但不确定是否在 predictor 对象上可访问
        # 假设存在一个索引文件需要删除
        # index_file = os.path.join(predictor.audio_db_path, "audio_indexes.json") # 假设的文件名
        # if os.path.exists(index_file):
        #     os.remove(index_file)
        #     print(f"已删除索引文件: {index_file}")

        # 确认 audio_db 目录是否为空（或只剩必要文件）
        remaining_items = os.listdir(predictor.audio_db_path)
        if not remaining_items: # 如果目录为空
             status_message = "声纹库已成功清空。"
        else:
             # 可能还有 predictor 内部的其他文件，只要用户目录和索引没了就算成功
             user_dirs_remain = any(os.path.isdir(os.path.join(predictor.audio_db_path, item)) for item in remaining_items)
             if not user_dirs_remain: # 假设没有用户目录残留
                status_message = "声纹库已清空 (可能保留了非用户数据文件)。"
             else:
                status_message = "清空声纹库完成，但似乎有用户目录残留。请手动检查。"

    except Exception as e:
        status_message = f"清空声纹库时发生错误: {e}"
        print(f"清空声纹库错误详情: {e}")

    # 更新用户列表 (应该为空了)
    users = get_user_list()
    return status_message, gr.update(choices=users, value=None)

def handle_backup_db():
    """备份声纹库"""
    if not predictor:
        return "错误：预测器未初始化"

    try:
        backup_name = f"audio_db_backup_{time.strftime('%Y%m%d_%H%M%S')}"
        backup_path = os.path.join(BACKUP_ROOT_PATH, backup_name)
        os.makedirs(backup_path, exist_ok=True) # 确保父目录存在

        # 使用 shutil.copytree 备份整个 audio_db 目录
        shutil.copytree(predictor.audio_db_path, backup_path, dirs_exist_ok=True)

        status_message = f"声纹库已成功备份到: {backup_path}"
        print(status_message)

    except Exception as e:
        status_message = f"备份声纹库时发生错误: {e}"
        print(f"备份错误详情: {e}")

    return status_message

# --- 加载音频到播放器 --- (新函数)
def load_audio_to_player(audio_file_path):
    if audio_file_path and os.path.exists(audio_file_path):
        return gr.update(value=audio_file_path)
    return gr.update(value=None) # 如果路径无效或为空，则清空播放器

# --- 动态启用清空按钮 ---
def enable_clear_button(confirm_text):
    if confirm_text == "确认清空":
        return gr.update(interactive=True)
    else:
        return gr.update(interactive=False)

# --- 构建 Gradio 界面 ---
with gr.Blocks(title="声纹识别系统 (Gradio)") as demo:
    gr.Markdown("# 声纹识别系统 (Gradio)")

    if not predictor:
        gr.Markdown("## 警告：声纹识别引擎初始化失败！请检查配置和模型路径。")
    else:
        with gr.Tabs():
            # == 1. 注册用户 Tab ==
            with gr.TabItem("注册用户"):
                gr.Markdown("## 注册声纹")
                status_register = gr.Textbox(label="状态", interactive=False)
                with gr.Row():
                    register_username = gr.Textbox(label="用户名")
                    register_mode = gr.Radio(["单个文件", "文件夹批量"], label="注册模式", value="单个文件")
                with gr.Column(visible=True) as single_file_ui: # 默认显示
                    register_audio_single = gr.Audio(label="选择单个音频文件", type="filepath")
                with gr.Column(visible=False) as batch_folder_ui:
                    register_audio_files_batch = gr.File(label="选择包含音频文件的文件夹 (上传文件夹中的所有音频)", file_count="directory")

                register_btn = gr.Button("开始注册")

            # == 2. 对比音频 Tab ==
            with gr.TabItem("对比音频"):
                gr.Markdown("## 计算两个音频的相似度")
                status_contrast = gr.Textbox(label="状态", interactive=False)
                with gr.Row():
                    contrast_audio1 = gr.Audio(label="音频文件 1", type="filepath")
                    contrast_audio2 = gr.Audio(label="音频文件 2", type="filepath")
                contrast_btn = gr.Button("计算相似度")
                contrast_score = gr.Number(label="相似度得分")

            # == 3. 识别用户 Tab ==
            with gr.TabItem("识别用户"):
                gr.Markdown("## 识别说话人")
                status_recognize = gr.Textbox(label="状态", interactive=False)
                recognize_audio = gr.Audio(label="选择需要识别的音频文件", type="filepath")
                recognize_btn = gr.Button("开始识别")
                recognize_result = gr.Textbox(label="识别结果")

            # == 4. 管理声纹库 Tab ==
            with gr.TabItem("管理声纹库"):
                gr.Markdown("## 声纹库管理")
                status_manage = gr.Textbox(label="操作状态", interactive=False, lines=3)
                with gr.Row():
                    user_list_dropdown = gr.Dropdown(label="选择用户", choices=get_user_list(), interactive=True)
                    refresh_users_btn = gr.Button("刷新列表")
                with gr.Tabs():
                    with gr.TabItem("用户操作"):
                         with gr.Row():
                            view_details_btn = gr.Button("查看详情")
                            delete_user_btn = gr.Button("删除用户")
                         user_details_output = gr.Textbox(label="用户详情 (音频列表)", interactive=False, lines=5)
                         user_audio_files_dropdown = gr.Dropdown(label="选择音频文件", choices=[], interactive=True)
                         audio_player = gr.Audio(label="播放音频", type="filepath")
                    with gr.TabItem("重命名"):
                        rename_new_name = gr.Textbox(label="新用户名")
                        rename_user_btn = gr.Button("确认重命名")
                    with gr.TabItem("库管理"):
                        gr.Markdown("### 危险操作")
                        clear_db_confirm = gr.Textbox(label="输入 '确认清空' 以启用按钮")
                        clear_db_btn = gr.Button("清空整个声纹库", interactive=False)
                        backup_db_btn = gr.Button("备份声纹库")

        # --- 事件处理逻辑 --- 

        # 1. 注册用户 Tab 事件
        register_mode.change(toggle_register_ui, 
                             inputs=[register_mode], 
                             outputs=[single_file_ui, batch_folder_ui])
        
        register_btn.click(handle_register, 
                           inputs=[register_username, register_mode, register_audio_single, register_audio_files_batch],
                           outputs=[status_register, user_list_dropdown])

        # 2. 对比音频 Tab 事件
        contrast_btn.click(handle_contrast,
                           inputs=[contrast_audio1, contrast_audio2],
                           outputs=[status_contrast, contrast_score])

        # 3. 识别用户 Tab 事件
        recognize_btn.click(handle_recognize,
                            inputs=[recognize_audio],
                            outputs=[status_recognize, recognize_result])

        # 4. 管理声纹库 Tab 事件
        refresh_users_btn.click(handle_refresh_users,
                                inputs=None,
                                # 更新下拉菜单、状态、清空详情、清空文件下拉、清空播放器
                                outputs=[user_list_dropdown, status_manage, user_details_output, user_audio_files_dropdown, audio_player])

        view_details_btn.click(handle_view_details,
                               inputs=[user_list_dropdown],
                               # 更新状态、详情文本、文件下拉菜单、播放器
                               outputs=[status_manage, user_details_output, user_audio_files_dropdown, audio_player])

        delete_user_btn.click(handle_delete_user,
                              inputs=[user_list_dropdown],
                              # 更新状态、用户列表
                              outputs=[status_manage, user_list_dropdown])

        rename_user_btn.click(handle_rename_user,
                              inputs=[user_list_dropdown, rename_new_name],
                              # 更新状态、用户列表、清空新名称输入框
                              outputs=[status_manage, user_list_dropdown, rename_new_name]) # 重置输入框

        # 动态启用清空按钮
        clear_db_confirm.input(enable_clear_button,
                               inputs=[clear_db_confirm],
                               outputs=[clear_db_btn])

        clear_db_btn.click(handle_clear_db,
                           inputs=[clear_db_confirm],
                           # 更新状态、用户列表
                           outputs=[status_manage, user_list_dropdown])

        backup_db_btn.click(handle_backup_db,
                            inputs=None,
                            # 更新状态
                            outputs=[status_manage])

        # 当用户音频文件下拉菜单变化时，更新播放器
        user_audio_files_dropdown.change(load_audio_to_player,
                                         inputs=[user_audio_files_dropdown],
                                         outputs=[audio_player])

# --- 启动 Gradio 应用 ---
if __name__ == "__main__":
    if predictor is None:
        print("无法启动 Gradio 应用，因为 MVectorPredictor 初始化失败。")
    else:
        demo.launch(inbrowser=True) # share=True 可以创建公共链接
        print(f"Gradio 应用已启动，请在浏览器中打开链接。")
