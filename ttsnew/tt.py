import os

# 환경 변수 설정
os.environ['TORCH_HOME'] = 'C:\\torch_cache'

import torch
import numpy as np
from IPython.display import Audio, display

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

Tacotron2 = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_tacotron2', trust_repo=True, map_location=device)
Tacotron2.to(device)

# Tacotron2와 WaveGlow 사전 학습 모델 다운로드
Tacotron2 = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_tacotron2', trust_repo=True, map_location=torch.device('cpu'))
WaveGlow = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_waveglow', trust_repo=True, map_location=torch.device('cpu'))

Tacotron2.to(torch.device('cpu'))
WaveGlow.to(torch.device('cpu'))
Tacotron2.eval()
WaveGlow.eval()

# 텍스트를 음성으로 변환하는 함수
def text_to_speech(text):
    # 텍스트를 시퀀스로 변환
    sequence = np.array(Tacotron2.text_to_sequence(text, ['english_cleaners']))[None, :]
    sequence = torch.from_numpy(sequence).to(torch.int64).to(torch.device('cpu'))
    
    # 멜-스펙트로그램 생성
    with torch.no_grad():
        mel_outputs, mel_outputs_postnet, _, alignments = Tacotron2.infer(sequence)
    
    # 음성 신호 생성
    with torch.no_grad():
        audio = WaveGlow.infer(mel_outputs_postnet, sigma=0.666)
    
    # 음성을 numpy 배열로 변환
    audio_numpy = audio[0].data.cpu().numpy()
    return audio_numpy

# 텍스트 입력
text = "Hello, this is a text to speech conversion example using Tacotron2 and WaveGlow."

# 음성 생성
audio = text_to_speech(text)

# 음성 재생
display(Audio(audio, rate=22050))
