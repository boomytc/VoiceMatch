# VoiceMatch

VoiceMatch是一个综合性语音处理工具包，集成了声纹识别和语音增强两大核心功能模块，旨在提供高质量的语音处理解决方案。

## 📋 目录

- [功能特点](#功能特点)
- [项目结构](#项目结构)
- [安装指南](#安装指南)
- [使用方法](#使用方法)
  - [声纹识别](#声纹识别)
  - [语音增强](#语音增强)
- [模型说明](#模型说明)
- [贡献指南](#贡献指南)
- [许可证](#许可证)

## 🌟 功能特点

### 声纹识别模块 (SpeakerRecognition)
- **声纹注册**：支持用户声纹特征提取与注册
- **声纹识别**：能够识别已注册用户的声音
- **说话人区分**：区分不同说话人的声音
- **多种模型支持**：包括CAMPPlus、ECAPA-TDNN、ERes2Net等先进声纹识别模型
- **图形界面**：提供基于PySide6的用户友好界面
- **Web界面**：基于Gradio的Web交互界面

### 语音增强模块 (SpeechEnhance)
- **语音增强**：去除背景噪音，提高语音清晰度
- **语音超分辨率**：提升语音质量和采样率
- **批量处理**：支持单文件和批量文件处理
- **多模型支持**：包括MossFormer2系列高性能语音处理模型
- **图形界面**：提供直观的用户界面
- **多平台支持**：支持Windows、Linux和macOS

## 🗂️ 项目结构

```
VoiceMatch/
├── SpeakerRecognition/           # 声纹识别模块
│   ├── configs/                  # 模型配置文件
│   ├── mvector/                  # 声纹识别核心库
│   │   ├── data_utils/           # 数据处理工具
│   │   ├── infer_utils/          # 推理工具
│   │   ├── loss/                 # 损失函数
│   │   ├── metric/               # 评估指标
│   │   ├── models/               # 模型定义
│   │   ├── optimizer/            # 优化器
│   │   └── utils/                # 工具函数
│   ├── reference_gui_pyqt.py     # PySide6图形界面
│   └── webui_reference.py        # Gradio Web界面
│
├── SpeechEnhance/                # 语音增强模块
│   ├── checkpoints/              # 模型检查点
│   ├── dataloader/               # 数据加载器
│   ├── models/                   # 模型定义
│   │   ├── mossformer2_se/       # 语音增强模型
│   │   └── mossformer2_sr/       # 语音超分辨率模型
│   ├── utils/                    # 工具函数
│   ├── clearvoice.py             # 主接口类
│   ├── gui_speech_enhancement.py # 图形界面
│   ├── speech_enhancement.py     # 命令行工具
│   └── webui_speech_enhancement.py # Web界面
│
├── gui_speaker_recognize.sh      # 声纹识别GUI启动脚本
├── gui_speech_enhance.sh         # 语音增强GUI启动脚本
├── speech_enhance.sh             # 语音增强命令行启动脚本
└── requirements.txt              # 项目依赖
```

## 🔧 安装指南

### 环境要求
- Python 3.8+
- CUDA支持（推荐用于GPU加速，但非必须）
- 操作系统：Windows、Linux或macOS

### 安装步骤

1. 克隆仓库
```bash
git clone https://github.com/boomytc/VoiceMatch.git
cd VoiceMatch
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 下载预训练模型（首次运行时会自动下载）
   - 声纹识别模型将保存在 `SpeakerRecognition/models/` 目录
   - 语音增强模型将保存在 `SpeechEnhance/checkpoints/` 目录
     - [MossFormer2_SE_48K](https://huggingface.co/alibabasglab/MossFormer2_SE_48K)（语音增强模型）
     - [MossFormer2_SR_48K](https://huggingface.co/alibabasglab/MossFormer2_SR_48K)（语音超分辨率模型）

## 📚 使用方法

### 声纹识别

#### 图形界面（推荐）
```bash
# Linux/macOS
./gui_speaker_recognize.sh

# Windows
bash gui_speaker_recognize.sh
```

#### Web界面
```bash
cd SpeakerRecognition
python webui_reference.py
```

#### 主要功能
1. **声纹注册**：录制或上传音频，为用户注册声纹
2. **声纹识别**：上传音频，识别说话人身份
3. **声纹库管理**：查看、删除已注册的声纹

### 语音增强

#### 图形界面（推荐）
```bash
# Linux/macOS
./gui_speech_enhance.sh

# Windows
bash gui_speech_enhance.sh
```

#### 命令行工具
```bash
# 单文件处理
./speech_enhance.sh 输入文件路径 -se -sr

# 批量处理
./speech_enhance.sh 输入目录路径 -se -sr
```
参数说明：
- `-se`：执行语音增强
- `-sr`：执行语音超分辨率
- `-o 输出路径`：指定输出文件路径（仅单文件模式）

#### Web界面
```bash
cd SpeechEnhance
python webui_speech_enhancement.py
```

## 🧠 模型说明

### 声纹识别模型

VoiceMatch支持多种先进的声纹识别模型：

1. **CAMPPlus**：默认模型，提供出色的声纹识别性能和较低的计算需求
2. **ECAPA-TDNN**：适用于复杂环境下的声纹识别
3. **ERes2Net**：提供最高精度的声纹识别，但计算需求较高

模型配置文件位于 `SpeakerRecognition/configs/` 目录。

### 语音增强模型

VoiceMatch集成了多种语音处理模型：

1. **[MossFormer2_SE_48K](https://huggingface.co/alibabasglab/MossFormer2_SE_48K)**：48kHz语音增强模型，用于去除背景噪音
2. **[MossFormer2_SR_48K](https://huggingface.co/alibabasglab/MossFormer2_SR_48K)**：48kHz语音超分辨率模型，用于提升音频质量

