from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from transformers import pipeline
import torch
import time

# Dependencies:
# python3 -m venv python-env && source python-env/bin/activate
# pip install --upgrade pip
# pip install "transformers>=4.35" "torch>=2.1,<2.4.0" "torchvision<0.19.0" "onnx<1.16.2" "peft==0.6.2" --extra-index-url https://download.pytorch.org/whl/cpu
# pip install moviepy

from moviepy.editor import VideoFileClip
from transformers.pipelines.audio_utils import ffmpeg_read
def get_audio(video_file):
    """
    Extract audio signal from a given video file, then convert it to float,
    then mono-channel format and resample it to the expected sample rate

    Parameters:
        video_file: path to input video file
    Returns:
      resampled_audio: mono-channel float audio signal with 16000 Hz sample rate
                       extracted from video
      duration: duration of video fragment in seconds
    """
    input_video = VideoFileClip(str(video_file))
    duration = input_video.duration
    audio_file = video_file + ".wav"
    input_video.audio.write_audiofile(audio_file, verbose=False, logger=None)
    with open(audio_file, "rb") as f:
        inputs = f.read()
    audio = ffmpeg_read(inputs, pipe.feature_extractor.sampling_rate)
    print("pipe.feature_extractor.sampling_rate=", pipe.feature_extractor.sampling_rate)
    return {
        "raw": audio,
        "sampling_rate": pipe.feature_extractor.sampling_rate,
    }, duration


PT_MODEL_ID='./whisper_pymodel/'
PT_MODEL_ID="./openai_whisper-base/"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

processor = AutoProcessor.from_pretrained(PT_MODEL_ID)

#  attn_implementation='flash_attention_2',
pt_model = AutoModelForSpeechSeq2Seq.from_pretrained(PT_MODEL_ID, torch_dtype=torch_dtype, low_cpu_mem_usage=True,
                                                     use_safetensors=True, cache_dir='data',).to(device)
pt_model.eval()
print(f"device={device}")

pipe = pipeline(
    "automatic-speech-recognition",
    model=pt_model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
)

output_file='../../downloaded_video.mp4'
inputs, duration = get_audio(output_file)

def get_audio2(audio_file, sampling_rate):
    with open(audio_file, "rb") as f:
        inputs = f.read()
    audio = ffmpeg_read(inputs, sampling_rate)
    return {
        "raw": audio,
        "sampling_rate": sampling_rate,
    }
# mp3
inputs = get_audio2('../../output.mp3', pipe.feature_extractor.sampling_rate)
task='translate'
language='en'

# my test audio
task='transcribe'
language='zh'
inputs = get_audio2('../../test_audio.ogx', pipe.feature_extractor.sampling_rate)

t1 = time.time()
transcription = pipe(inputs, generate_kwargs={"task": task, "language":language}, return_timestamps=True)["chunks"]
t2 = time.time()
print(f"time= {(t2-t1)*1000} ms")
print("transcription=", transcription)