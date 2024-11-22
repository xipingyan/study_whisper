
cd whisper.cpp

# Whisper repos:
#  git clone https://github.com/openai/whisper

mkdir models/whisper-base
python3 ./models/convert-h5-to-ggml.py ../openai_whisper-base/ ../whisper/ ./models/whisper-base
mv ./models/whisper-base/ggml-model.bin models/ggml-base.bin
rmdir models/whisper-base
cd ..