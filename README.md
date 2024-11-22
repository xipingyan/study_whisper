# study_whisper

Refer: https://huggingface.co/openai/whisper-base

#### Model

    https://huggingface.co/openai/whisper-base/tree/main

#### Run audio to text

    audio sampling rate: 16000 Hz

    $ ./run.sh 

    device=cpu
    You have passed task=transcribe, but also have set `forced_decoder_ids` to [[1, None], [2, 50359]] which creates a conflict. `forced_decoder_ids` will be ignored in favor of task=transcribe.
    The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
    time= 217.51093864440918 ms
    transcription= [{'timestamp': (0.0, 2.0), 'text': '再来一次'}]

