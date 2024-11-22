# Refer: https://github.com/ggerganov/whisper.cpp?tab=readme-ov-file#quick-start

# Download model
# sh ./models/download-ggml-model.sh base

# build command
# make -j

# Convert audo to wav
# ffmpeg -i ./samples/test_audio.ogx -acodec pcm_s16le -ar 16000 ./samples/test_audio.wav
# ffmpeg -i ../../../../test_audio.ogx -acodec pcm_s16le -ar 16000 ./test_audio.wav

model_my_cvt=./whisper.cpp/models/ggml-base.bin

inp_audio=./whisper.cpp/samples/jfk.wav
inp_audio=./whisper.cpp/samples/test_audio.wav

./whisper.cpp/main -f $inp_audio -m $model_my_cvt --language zh --threads 20 --prompt "请用简体中文输出" -bs 5