# Original pytorch model

Refer: https://huggingface.co/openai/whisper-base

#### Model

    https://huggingface.co/openai/whisper-base/tree/main

#### Virtual Python ENV.

    python3 -m venv python-env && source python-env/bin/activate
    pip install --upgrade pip
    pip install "transformers>=4.35" "torch>=2.1,<2.4.0" "torchvision<0.19.0" "onnx<1.16.2" "peft==0.6.2" --extra-index-url https://download.pytorch.org/whl/cpu
    pip install moviepy

#### Run audio to text

    audio sampling rate: 16000 Hz

    You can change your input audio in the run_whisper.py.

    $ ./run.sh 

    device=cpu
    You have passed task=transcribe, but also have set `forced_decoder_ids` to [[1, None], [2, 50359]] which creates a conflict. `forced_decoder_ids` will be ignored in favor of task=transcribe.
    The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
    time= 217.51093864440918 ms
    transcription= [{'timestamp': (0.0, 2.0), 'text': '再来一次'}]

# whisper.cpp

Refer: [whisper.cpp](https://github.com/ggerganov/whisper.cpp.git) <br>
Convert original pytorch model to whisper.cpp format. [Refer](https://github.com/ggerganov/whisper.cpp/tree/master/models#3-convert-with-convert-pt-to-ggmlpy) <br>

    $ ./cvt_pytorch_model_to_whispercpp.sh

    $ git submodule update --init
    $ cd whisper.cpp
    $ make -j20

    $ cd ..
    $ run_whisper_cpp.sh

    [00:00:00.000 --> 00:00:02.820]  再来一次再来一次