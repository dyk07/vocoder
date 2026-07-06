# to-do list

## 20260630
封装成一个独立的python文件：输入索引文件夹（存储GT的若干音频）和一个模型生成的文件夹（存储和GT同名的生成音频），打印出如下指标：M-STFT, PESQ, Periodicity, V/UV F1, UTMOS, VISQOL

数据集用LibriTTS，Out of distribution用MUSDB18数据集；
MUSDB18 一般是随便选10s的clip；
libriTTS 应该是全长；
按行业规范随机选150-250条即可，测试时可选150条


bigvgan的目录为dev-other.txt, dev-clean.txt.

用libri tts dev set里边的这些音频作为测试样本就可以了
你先写六个指标的集成环境，然后直接在这上边测，到时候把结果告诉我 我看看能不能对上就行了


这只有208个音频 你应该非常好做，就是你笔记本拿bigvgan的checkpoint，先用它仓库提取一下音频的mel图，然后就可以运行推理合成音频，之后打分测试了
https://www.openslr.org/60/
这个月搞定 客观分打分工具和主观分打分平台，确保完全正确，客观分弄完了之后我告诉你主观测评平台要怎么写是一个python GUI程序
争取一个月搞定
下个月你就要去运行各个基线模型测出分数然后制表了