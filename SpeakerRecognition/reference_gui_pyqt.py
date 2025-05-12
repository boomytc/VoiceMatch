import sys
import os
import glob
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QGridLayout, QLabel, QLineEdit, 
                            QPushButton, QFileDialog, QTextEdit, QTabWidget,
                            QGroupBox, QMessageBox, QListWidget, QInputDialog, QRadioButton,
                            QDialog, QScrollArea)
from PySide6.QtCore import QUrl
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
import shutil

from mvector.predict import MVectorPredictor


class ReferenceGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("声纹识别系统")
        self.setGeometry(100, 100, 800, 600)
        
        # 创建预测器
        self.predictor = MVectorPredictor(
            configs='configs/cam++.yml',
            model_path='models/CAMPPlus_Fbank/best_model/',
            threshold=0.6,
            audio_db_path='audio_db/',
            use_gpu=True
        )
        
        # 创建媒体播放器
        self.media_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.media_player.setAudioOutput(self.audio_output)
        
        # 创建中心部件
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # 创建主布局
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # 创建标签页
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)
        
        # 创建各个功能标签页
        self.create_register_tab()
        self.create_contrast_tab()
        self.create_recognition_tab()
        self.create_manage_db_tab()  # 新增声纹库管理标签页
        
        # 添加状态栏
        self.statusBar().showMessage("就绪")

    def create_register_tab(self):
        """创建注册用户音频标签页（整合单个注册和批量注册）"""
        register_tab = QWidget()
        layout = QGridLayout(register_tab)
        
        # 用户名
        layout.addWidget(QLabel("用户名:"), 0, 0)
        self.register_username = QLineEdit()
        layout.addWidget(self.register_username, 0, 1, 1, 2)
        
        # 注册模式选择
        self.register_mode_group = QGroupBox("注册模式")
        register_mode_layout = QVBoxLayout()
        
        # 单个文件模式
        self.single_file_radio = QRadioButton("单个文件注册")
        self.single_file_radio.setChecked(True)
        self.single_file_radio.toggled.connect(self.toggle_register_mode)
        register_mode_layout.addWidget(self.single_file_radio)
        
        # 文件夹批量模式
        self.folder_radio = QRadioButton("文件夹批量注册")
        self.folder_radio.toggled.connect(self.toggle_register_mode)
        register_mode_layout.addWidget(self.folder_radio)
        
        self.register_mode_group.setLayout(register_mode_layout)
        layout.addWidget(self.register_mode_group, 1, 0, 1, 3)
        
        # 单个文件注册控件组
        self.single_file_group = QGroupBox("单个文件")
        single_file_layout = QGridLayout()
        
        single_file_layout.addWidget(QLabel("音频路径:"), 0, 0)
        self.register_audio_path = QLineEdit()
        single_file_layout.addWidget(self.register_audio_path, 0, 1)
        
        self.register_select_btn = QPushButton("选择文件")
        self.register_select_btn.clicked.connect(self.select_register_audio)
        single_file_layout.addWidget(self.register_select_btn, 0, 2)
        
        self.single_file_group.setLayout(single_file_layout)
        layout.addWidget(self.single_file_group, 2, 0, 1, 3)
        
        # 文件夹批量注册控件组
        self.folder_group = QGroupBox("文件夹批量")
        folder_layout = QGridLayout()
        
        folder_layout.addWidget(QLabel("文件夹路径:"), 0, 0)
        self.register_folder_path = QLineEdit()
        folder_layout.addWidget(self.register_folder_path, 0, 1)
        
        self.folder_select_btn = QPushButton("选择文件夹")
        self.folder_select_btn.clicked.connect(self.select_register_folder)
        folder_layout.addWidget(self.folder_select_btn, 0, 2)
        
        self.folder_group.setLayout(folder_layout)
        layout.addWidget(self.folder_group, 3, 0, 1, 3)
        self.folder_group.setVisible(False)  # 默认隐藏文件夹选择组
        
        # 注册按钮
        self.register_btn = QPushButton("注册")
        self.register_btn.clicked.connect(self.register_user)
        layout.addWidget(self.register_btn, 4, 0, 1, 3)
        
        # 结果显示
        layout.addWidget(QLabel("结果:"), 5, 0)
        self.register_result = QTextEdit()
        self.register_result.setReadOnly(True)
        layout.addWidget(self.register_result, 6, 0, 1, 3)
        
        self.tabs.addTab(register_tab, "注册用户音频")
    
    def toggle_register_mode(self):
        """切换注册模式（单个文件/文件夹批量）"""
        if self.single_file_radio.isChecked():
            self.single_file_group.setVisible(True)
            self.folder_group.setVisible(False)
        else:
            self.single_file_group.setVisible(False)
            self.folder_group.setVisible(True)
    
    # ===== 注册用户音频 =====
    def select_register_audio(self):
        """选择注册用户的音频文件"""
        filename, _ = QFileDialog.getOpenFileName(self, "选择音频文件", "./dataset", "音频文件 (*.wav *.mp3)")
        if filename:
            self.register_audio_path.setText(filename)
    
    def select_register_folder(self):
        """选择注册用户的音频文件夹"""
        folder = QFileDialog.getExistingDirectory(self, "选择音频文件夹", "./dataset")
        if folder:
            self.register_folder_path.setText(folder)
    
    def register_user(self):
        """注册用户（整合单个注册和批量注册）"""
        username = self.register_username.text()
        
        if not username:
            QMessageBox.critical(self, "错误", "请输入用户名")
            return
        
        # 根据选择的模式执行不同的注册逻辑
        if self.single_file_radio.isChecked():
            # 单个文件注册
            audio_path = self.register_audio_path.text()
            
            if not audio_path:
                QMessageBox.critical(self, "错误", "请选择音频文件")
                return
                
            try:
                self.statusBar().showMessage("正在注册...")
                self.register_result.clear()
                
                # 注册用户
                self.predictor.register(user_name=username, audio_data=audio_path)
                
                self.register_result.append(f"用户 '{username}' 注册成功！")
                self.register_result.append(f"使用的音频: {os.path.basename(audio_path)}")
                self.statusBar().showMessage("注册成功")
                
            except Exception as e:
                self.register_result.append(f"注册失败: {str(e)}")
                self.statusBar().showMessage(f"注册失败: {str(e)}")
        else:
            # 文件夹批量注册
            folder_path = self.register_folder_path.text()
            
            if not folder_path or not os.path.isdir(folder_path):
                QMessageBox.critical(self, "错误", "请选择有效的音频文件夹")
                return
                
            try:
                self.statusBar().showMessage("正在批量注册...")
                self.register_result.clear()
                
                # 获取文件夹下所有.wav文件的路径
                audio_files = glob.glob(os.path.join(folder_path, "*.wav"))
                
                if not audio_files:
                    self.register_result.append(f"文件夹 '{folder_path}' 中没有找到.wav文件")
                    self.statusBar().showMessage("批量注册失败: 没有找到.wav文件")
                    return
                    
                # 批量注册
                for i, audio_file in enumerate(audio_files):
                    self.predictor.register(user_name=username, audio_data=audio_file)
                    self.register_result.append(f"已注册 ({i+1}/{len(audio_files)}): {os.path.basename(audio_file)}")
                    QApplication.processEvents()  # 确保UI更新
                
                self.register_result.append(f"\n用户 '{username}' 批量注册成功！")
                self.register_result.append(f"共注册了 {len(audio_files)} 个音频文件")
                self.statusBar().showMessage("批量注册成功")
                
            except Exception as e:
                self.register_result.append(f"批量注册失败: {str(e)}")
                self.statusBar().showMessage(f"批量注册失败: {str(e)}")
    
    # ===== 获取两个音频的相似度 =====
    def create_contrast_tab(self):
        """创建获取两个音频的相似度标签页"""
        contrast_tab = QWidget()
        layout = QGridLayout(contrast_tab)
        
        # 音频1路径
        layout.addWidget(QLabel("音频1路径:"), 0, 0)
        self.contrast_audio_path1 = QLineEdit()
        layout.addWidget(self.contrast_audio_path1, 0, 1)
        
        # 选择文件按钮1
        self.contrast_select_btn1 = QPushButton("选择文件")
        self.contrast_select_btn1.clicked.connect(lambda: self.select_contrast_audio(1))
        layout.addWidget(self.contrast_select_btn1, 0, 2)
        
        # 音频2路径
        layout.addWidget(QLabel("音频2路径:"), 1, 0)
        self.contrast_audio_path2 = QLineEdit()
        layout.addWidget(self.contrast_audio_path2, 1, 1)
        
        # 选择文件按钮2
        self.contrast_select_btn2 = QPushButton("选择文件")
        self.contrast_select_btn2.clicked.connect(lambda: self.select_contrast_audio(2))
        layout.addWidget(self.contrast_select_btn2, 1, 2)
        
        # 阈值
        layout.addWidget(QLabel("判断阈值:"), 2, 0)
        self.contrast_threshold = QLineEdit("0.6")
        layout.addWidget(self.contrast_threshold, 2, 1, 1, 2)
        
        # 获取相似度按钮
        self.get_contrast_btn = QPushButton("获取相似度")
        self.get_contrast_btn.clicked.connect(self.get_contrast)
        layout.addWidget(self.get_contrast_btn, 3, 0, 1, 3)
        
        # 结果显示
        layout.addWidget(QLabel("结果:"), 4, 0)
        self.contrast_result = QTextEdit()
        self.contrast_result.setReadOnly(True)
        layout.addWidget(self.contrast_result, 5, 0, 1, 3)
        
        self.tabs.addTab(contrast_tab, "获取音频相似度")
    
    def select_contrast_audio(self, audio_num):
        """选择对比的音频文件"""
        filename, _ = QFileDialog.getOpenFileName(self, "选择音频文件", "./dataset", "音频文件 (*.wav *.mp3)")
        if filename:
            if audio_num == 1:
                self.contrast_audio_path1.setText(filename)
            else:
                self.contrast_audio_path2.setText(filename)
    
    def get_contrast(self):
        """获取两个音频的相似度"""
        audio_path1 = self.contrast_audio_path1.text()
        audio_path2 = self.contrast_audio_path2.text()
        
        if not audio_path1 or not audio_path2:
            QMessageBox.critical(self, "错误", "请选择两个音频文件")
            return
            
        try:
            threshold = float(self.contrast_threshold.text())
        except ValueError:
            QMessageBox.critical(self, "错误", "阈值必须是一个有效的数字")
            return
            
        try:
            self.statusBar().showMessage("正在计算相似度...")
            self.contrast_result.clear()
            
            # 获取相似度
            similarity = self.predictor.contrast(audio_data1=audio_path1, audio_data2=audio_path2)
            
            self.contrast_result.append(f"音频1: {os.path.basename(audio_path1)}")
            self.contrast_result.append(f"音频2: {os.path.basename(audio_path2)}")
            self.contrast_result.append(f"相似度: {similarity:.5f}")
            
            if similarity > threshold:
                self.contrast_result.append(f"\n结论: 这两个音频很可能是同一个人的声音 (相似度 > {threshold})")
            else:
                self.contrast_result.append(f"\n结论: 这两个音频可能不是同一个人的声音 (相似度 <= {threshold})")
                
            self.statusBar().showMessage("计算相似度成功")
            
        except Exception as e:
            self.contrast_result.append(f"计算相似度失败: {str(e)}")
            self.statusBar().showMessage(f"计算相似度失败: {str(e)}")
    
    # ===== 识别用户音频 =====
    def create_recognition_tab(self):
        """创建识别用户音频标签页"""
        recognition_tab = QWidget()
        layout = QGridLayout(recognition_tab)
        
        # 音频路径
        layout.addWidget(QLabel("音频路径:"), 0, 0)
        self.recognition_audio_path = QLineEdit()
        layout.addWidget(self.recognition_audio_path, 0, 1)
        
        # 选择文件按钮
        self.recognition_select_btn = QPushButton("选择文件")
        self.recognition_select_btn.clicked.connect(self.select_recognition_audio)
        layout.addWidget(self.recognition_select_btn, 0, 2)
        
        # 阈值
        layout.addWidget(QLabel("判断阈值:"), 1, 0)
        self.recognition_threshold = QLineEdit("0.6")
        layout.addWidget(self.recognition_threshold, 1, 1, 1, 2)
        
        # 识别按钮
        self.recognition_btn = QPushButton("识别")
        self.recognition_btn.clicked.connect(self.recognize_user)
        layout.addWidget(self.recognition_btn, 2, 0, 1, 3)
        
        # 结果显示
        layout.addWidget(QLabel("结果:"), 3, 0)
        self.recognition_result = QTextEdit()
        self.recognition_result.setReadOnly(True)
        layout.addWidget(self.recognition_result, 4, 0, 1, 3)
        
        self.tabs.addTab(recognition_tab, "识别用户音频")
    
    def select_recognition_audio(self):
        """选择识别的音频文件"""
        filename, _ = QFileDialog.getOpenFileName(self, "选择音频文件", "./dataset", "音频文件 (*.wav *.mp3)")
        if filename:
            self.recognition_audio_path.setText(filename)
    
    def recognize_user(self):
        """识别用户"""
        audio_path = self.recognition_audio_path.text()
        
        if not audio_path:
            QMessageBox.critical(self, "错误", "请选择音频文件")
            return
            
        try:
            threshold = float(self.recognition_threshold.text())
        except ValueError:
            QMessageBox.critical(self, "错误", "阈值必须是一个有效的数字")
            return
            
        try:
            self.statusBar().showMessage("正在识别...")
            self.recognition_result.clear()
            
            # 识别用户
            name, score = self.predictor.recognition(audio_data=audio_path, threshold=threshold)
            
            self.recognition_result.append(f"音频: {os.path.basename(audio_path)}")
            
            if name:
                self.recognition_result.append(f"识别结果: {name}")
                self.recognition_result.append(f"得分: {score:.5f}")
                self.recognition_result.append("\n结论: 成功识别为注册用户")
            else:
                self.recognition_result.append("识别结果: 未识别到匹配的用户")
                self.recognition_result.append("\n结论: 可能是未注册用户或识别分数低于阈值")
                
            self.statusBar().showMessage("识别完成")
            
        except Exception as e:
            self.recognition_result.append(f"识别失败: {str(e)}")
            self.statusBar().showMessage(f"识别失败: {str(e)}")

    # ===== 声纹库管理 =====
    def create_manage_db_tab(self):
        """创建声纹库管理标签页"""
        manage_db_tab = QWidget()
        layout = QGridLayout(manage_db_tab)
        
        # 用户列表
        layout.addWidget(QLabel("声纹库用户列表:"), 0, 0, 1, 3)
        self.user_list = QListWidget()
        layout.addWidget(self.user_list, 1, 0, 1, 3)
        
        # 刷新按钮
        self.refresh_btn = QPushButton("刷新用户列表")
        self.refresh_btn.clicked.connect(self.refresh_user_list)
        layout.addWidget(self.refresh_btn, 2, 0)
        
        # 删除用户按钮
        self.delete_user_btn = QPushButton("删除选中用户")
        self.delete_user_btn.clicked.connect(self.delete_selected_user)
        layout.addWidget(self.delete_user_btn, 2, 1)
        
        # 查看用户详情按钮
        self.view_user_btn = QPushButton("查看用户详情")
        self.view_user_btn.clicked.connect(self.view_user_details)
        layout.addWidget(self.view_user_btn, 2, 2)
        
        # 重命名用户按钮
        self.rename_user_btn = QPushButton("重命名用户")
        self.rename_user_btn.clicked.connect(self.rename_user)
        layout.addWidget(self.rename_user_btn, 3, 0)
        
        # 清空声纹库按钮
        self.clear_db_btn = QPushButton("清空声纹库")
        self.clear_db_btn.clicked.connect(self.clear_audio_db)
        layout.addWidget(self.clear_db_btn, 3, 1)
        
        # 备份声纹库按钮
        self.backup_db_btn = QPushButton("备份声纹库")
        self.backup_db_btn.clicked.connect(self.backup_audio_db)
        layout.addWidget(self.backup_db_btn, 3, 2)
        
        # 结果显示
        layout.addWidget(QLabel("操作结果:"), 4, 0)
        self.manage_db_result = QTextEdit()
        self.manage_db_result.setReadOnly(True)
        layout.addWidget(self.manage_db_result, 5, 0, 1, 3)
        
        self.tabs.addTab(manage_db_tab, "声纹库管理")
        
        # 初始加载用户列表
        self.refresh_user_list()

    def refresh_user_list(self):
        """刷新用户列表"""
        try:
            self.statusBar().showMessage("正在刷新用户列表...")
            self.user_list.clear()
            
            # 获取所有用户
            users = self.predictor.get_users()
            
            # 统计每个用户的音频数量
            user_counts = {}
            for user in users:
                if user not in user_counts:
                    user_counts[user] = 1
                else:
                    user_counts[user] += 1
            
            # 去重并排序
            unique_users = sorted(set(users))
            
            # 添加到列表中
            for user in unique_users:
                count = user_counts.get(user, 0)
                self.user_list.addItem(f"{user} ({count}条音频)")
            
            self.statusBar().showMessage(f"用户列表刷新完成，共{len(unique_users)}个用户")
            self.manage_db_result.clear()
            self.manage_db_result.append(f"用户列表刷新完成，共{len(unique_users)}个用户")
            
        except Exception as e:
            self.statusBar().showMessage(f"刷新用户列表失败: {str(e)}")
            self.manage_db_result.append(f"刷新用户列表失败: {str(e)}")
    
    def delete_selected_user(self):
        """删除选中的用户"""
        selected_items = self.user_list.selectedItems()
        if not selected_items:
            QMessageBox.critical(self, "错误", "请先选择要删除的用户")
            return
            
        selected_item = selected_items[0]
        user_name = selected_item.text().split(" (")[0]  # 提取用户名
        
        reply = QMessageBox.question(self, "确认删除", 
                                    f"确定要删除用户 '{user_name}' 及其所有声纹数据吗？",
                                    QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            try:
                self.statusBar().showMessage(f"正在删除用户 '{user_name}'...")
                self.manage_db_result.clear()
                
                # 删除用户
                result = self.predictor.remove_user(user_name=user_name)
                
                if result:
                    self.manage_db_result.append(f"用户 '{user_name}' 删除成功！")
                    self.statusBar().showMessage(f"用户 '{user_name}' 删除成功")
                    # 刷新用户列表
                    self.refresh_user_list()
                else:
                    self.manage_db_result.append(f"用户 '{user_name}' 删除失败，可能用户不存在")
                    self.statusBar().showMessage(f"用户 '{user_name}' 删除失败")
                
            except Exception as e:
                self.manage_db_result.append(f"删除用户失败: {str(e)}")
                self.statusBar().showMessage(f"删除用户失败: {str(e)}")
    
    def view_user_details(self):
        """查看用户详情"""
        selected_items = self.user_list.selectedItems()
        if not selected_items:
            QMessageBox.critical(self, "错误", "请先选择要查看的用户")
            return
            
        selected_item = selected_items[0]
        user_name = selected_item.text().split(" (")[0]  # 提取用户名
        
        try:
            self.statusBar().showMessage(f"正在查看用户 '{user_name}' 的详情...")
            
            # 获取用户音频文件
            user_dir = os.path.join(self.predictor.audio_db_path, user_name)
            if not os.path.exists(user_dir):
                self.manage_db_result.clear()
                self.manage_db_result.append(f"用户 '{user_name}' 的音频文件夹不存在")
                self.statusBar().showMessage("查看用户详情失败")
                return
                
            audio_files = [f for f in os.listdir(user_dir) if f.endswith('.wav')]
            
            # 创建用户详情对话框
            details_dialog = QDialog(self)
            details_dialog.setWindowTitle(f"用户 '{user_name}' 的详情")
            details_dialog.setMinimumSize(600, 400)
            
            dialog_layout = QVBoxLayout(details_dialog)
            
            # 用户基本信息
            info_group = QGroupBox("基本信息")
            info_layout = QVBoxLayout()
            info_layout.addWidget(QLabel(f"用户名: {user_name}"))
            info_layout.addWidget(QLabel(f"音频文件数量: {len(audio_files)}"))
            info_layout.addWidget(QLabel(f"音频文件夹路径: {user_dir}"))
            info_group.setLayout(info_layout)
            dialog_layout.addWidget(info_group)
            
            # 音频文件列表
            files_group = QGroupBox("音频文件列表")
            files_layout = QVBoxLayout()
            
            # 创建滚动区域
            scroll_area = QScrollArea()
            scroll_area.setWidgetResizable(True)
            scroll_content = QWidget()
            scroll_layout = QVBoxLayout(scroll_content)
            
            for audio_file in audio_files:
                file_path = os.path.join(user_dir, audio_file)
                file_size = os.path.getsize(file_path) / 1024  # KB
                
                file_widget = QWidget()
                file_layout = QHBoxLayout(file_widget)
                file_layout.setContentsMargins(0, 0, 0, 0)
                
                # 文件信息
                file_label = QLabel(f"{audio_file} ({file_size:.2f} KB)")
                file_layout.addWidget(file_label, 1)
                
                # 播放按钮
                play_btn = QPushButton("播放")
                play_btn.setProperty("file_path", file_path)
                play_btn.clicked.connect(self.play_audio)
                file_layout.addWidget(play_btn)
                
                scroll_layout.addWidget(file_widget)
            
            scroll_content.setLayout(scroll_layout)
            scroll_area.setWidget(scroll_content)
            files_layout.addWidget(scroll_area)
            
            files_group.setLayout(files_layout)
            dialog_layout.addWidget(files_group)
            
            # 关闭按钮
            close_btn = QPushButton("关闭")
            close_btn.clicked.connect(details_dialog.accept)
            dialog_layout.addWidget(close_btn)
            
            # 在主窗口中也显示一些基本信息
            self.manage_db_result.clear()
            self.manage_db_result.append(f"用户名: {user_name}")
            self.manage_db_result.append(f"音频文件数量: {len(audio_files)}")
            self.manage_db_result.append(f"已打开详情对话框，可查看和播放音频文件")
            
            self.statusBar().showMessage("查看用户详情完成")
            
            # 显示对话框
            details_dialog.exec()
            
        except Exception as e:
            self.manage_db_result.clear()
            self.manage_db_result.append(f"查看用户详情失败: {str(e)}")
            self.statusBar().showMessage(f"查看用户详情失败: {str(e)}")
    
    def play_audio(self):
        """播放音频文件"""
        sender = self.sender()
        if not sender:
            return
            
        file_path = sender.property("file_path")
        if not file_path or not os.path.exists(file_path):
            QMessageBox.critical(self, "错误", f"音频文件不存在: {file_path}")
            return
            
        try:
            # 设置音量
            self.audio_output.setVolume(1.0)
            
            # 设置媒体源并播放
            self.media_player.setSource(QUrl.fromLocalFile(file_path))
            self.media_player.play()
            
            self.statusBar().showMessage(f"正在播放: {os.path.basename(file_path)}")
            
        except Exception as e:
            QMessageBox.critical(self, "播放失败", f"无法播放音频文件: {str(e)}")
            self.statusBar().showMessage(f"播放失败: {str(e)}")
    
    def rename_user(self):
        """重命名用户"""
        selected_items = self.user_list.selectedItems()
        if not selected_items:
            QMessageBox.critical(self, "错误", "请先选择要重命名的用户")
            return
            
        selected_item = selected_items[0]
        old_name = selected_item.text().split(" (")[0]  # 提取用户名
        
        new_name, ok = QInputDialog.getText(self, "重命名用户", 
                                           f"请输入用户 '{old_name}' 的新名称:")
        
        if ok and new_name:
            if new_name == old_name:
                QMessageBox.information(self, "提示", "新名称与原名称相同，无需修改")
                return
                
            try:
                self.statusBar().showMessage(f"正在重命名用户 '{old_name}' 为 '{new_name}'...")
                self.manage_db_result.clear()
                
                # 检查新名称是否已存在
                users = self.predictor.get_users()
                if new_name in set(users):
                    QMessageBox.critical(self, "错误", f"用户名 '{new_name}' 已存在")
                    return
                
                # 重命名用户文件夹
                old_dir = os.path.join(self.predictor.audio_db_path, old_name)
                new_dir = os.path.join(self.predictor.audio_db_path, new_name)
                
                if not os.path.exists(old_dir):
                    self.manage_db_result.append(f"用户 '{old_name}' 的音频文件夹不存在")
                    self.statusBar().showMessage("重命名用户失败")
                    return
                    
                # 先删除用户
                self.predictor.remove_user(user_name=old_name)
                
                # 创建新文件夹
                os.makedirs(new_dir, exist_ok=True)
                
                # 复制音频文件
                audio_files = [f for f in os.listdir(old_dir) if f.endswith('.wav')]
                for audio_file in audio_files:
                    old_file = os.path.join(old_dir, audio_file)
                    new_file = os.path.join(new_dir, audio_file)
                    shutil.copy2(old_file, new_file)
                    
                    # 重新注册
                    self.predictor.register(user_name=new_name, audio_data=new_file)
                
                # 删除旧文件夹
                shutil.rmtree(old_dir)
                
                self.manage_db_result.append(f"用户 '{old_name}' 已成功重命名为 '{new_name}'")
                self.statusBar().showMessage(f"用户重命名成功")
                
                # 刷新用户列表
                self.refresh_user_list()
                
            except Exception as e:
                self.manage_db_result.append(f"重命名用户失败: {str(e)}")
                self.statusBar().showMessage(f"重命名用户失败: {str(e)}")
    
    def clear_audio_db(self):
        """清空声纹库"""
        reply = QMessageBox.question(self, "确认清空", 
                                    "确定要清空整个声纹库吗？此操作不可恢复！",
                                    QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            confirm_text, ok = QInputDialog.getText(self, "二次确认", 
                                                 "请输入'确认清空'以继续操作:")
            
            if ok and confirm_text == "确认清空":
                try:
                    self.statusBar().showMessage("正在清空声纹库...")
                    self.manage_db_result.clear()
                    
                    # 获取所有用户
                    users = set(self.predictor.get_users())
                    
                    # 逐个删除用户
                    for user in users:
                        self.predictor.remove_user(user_name=user)
                    
                    # 删除索引文件
                    if os.path.exists(self.predictor.audio_indexes_path):
                        os.remove(self.predictor.audio_indexes_path)
                    
                    self.manage_db_result.append("声纹库已成功清空")
                    self.statusBar().showMessage("声纹库已清空")
                    
                    # 刷新用户列表
                    self.refresh_user_list()
                    
                except Exception as e:
                    self.manage_db_result.append(f"清空声纹库失败: {str(e)}")
                    self.statusBar().showMessage(f"清空声纹库失败: {str(e)}")
    
    def backup_audio_db(self):
        """备份声纹库"""
        backup_dir = QFileDialog.getExistingDirectory(self, "选择备份目录")
        if not backup_dir:
            return
            
        try:
            self.statusBar().showMessage("正在备份声纹库...")
            self.manage_db_result.clear()
            
            # 创建备份文件夹
            import time
            backup_name = f"audio_db_backup_{time.strftime('%Y%m%d_%H%M%S')}"
            backup_path = os.path.join(backup_dir, backup_name)
            os.makedirs(backup_path, exist_ok=True)
            
            # 复制声纹库文件
            shutil.copytree(self.predictor.audio_db_path, backup_path, dirs_exist_ok=True)
            
            self.manage_db_result.append(f"声纹库已成功备份到: {backup_path}")
            self.statusBar().showMessage("声纹库备份完成")
            
        except Exception as e:
            self.manage_db_result.append(f"备份声纹库失败: {str(e)}")
            self.statusBar().showMessage(f"备份声纹库失败: {str(e)}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ReferenceGUI()
    window.show()
    sys.exit(app.exec())
