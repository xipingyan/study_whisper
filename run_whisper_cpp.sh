# Refer: https://github.com/ggerganov/whisper.cpp?tab=readme-ov-file#quick-start

# Download model
# sh ./models/download-ggml-model.sh base

# build command
# make -j

# Convert audo to wav
# ffmpeg -i ./samples/test_audio.ogx -acodec pcm_s16le -ar 16000 ./samples/test_audio.wav
# ffmpeg -i ../../../../test_audio.ogx -acodec pcm_s16le -ar 16000 ./test_audio.wav

inp_audio=./whisper.cpp/samples/jfk.wav
inp_audio=./whisper.cpp/samples/test_audio.wav

./whisper.cpp/main -f $inp_audio -m ./whisper.cpp/models/for-tests-ggml-base.bin --language en  --output-txt true --translate true