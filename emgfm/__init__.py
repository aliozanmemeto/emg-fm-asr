"""emgfm -- silent-speech recognition from surface EMG through a pretrained speech foundation model.

Pipeline (one utterance):
    8-channel EMG at 1000 Hz
      -> EMGAdapter (conv stack, total stride 20, + Transformer)      [trained]
      -> predicted Granite input features: 160-d = 2 stacked 80-bin log-mel frames per 20 ms
      -> Granite Speech 4.0-1B conformer encoder, adapted with LoRA   [LoRA + input projection trained]
      -> Granite's Q-Former projector and 1B LLM                       [frozen]
      -> transcript (beam search)

Training has two stages ("align, then let go"):
    align   the adapter alone learns to reproduce the real spectrogram of the spoken utterance
            (a silent utterance is paired to its vocalised twin by Gaddy's monotonic DTW);
            Granite is frozen and no transcript is used.
    refine  adapter + encoder LoRA are trained on Granite's own next-token loss, with NO spectrogram
            term -- any spectrogram anchor in this stage holds silent decoding back (see README).
"""

MODEL = "ibm-granite/granite-4.0-1b-speech"
PROMPT_TEXT = "<|audio|>can you transcribe the speech into a written format?"

EMG_HZ = 1000            # cache sample rate of the filtered EMG (Gaddy's front-end minus his subsampling)
FRAME_MS = 20            # one Granite encoder frame; 1000 Hz / 20 ms = 20 EMG samples per frame
MEL_DIM = 160            # Granite encoder INPUT: 2 stacked 80-bin log-mel frames
ENC_DIM = 1024           # Granite encoder OUTPUT (= projector input)
WINDOW, QUERIES = 15, 3  # projector: 15 encoder frames -> 3 LLM tokens

ENCODER_LORA_TARGETS = "to_q,to_kv,to_out,up_proj,down_proj"
BN_BUFFERS = ("running_mean", "running_var", "num_batches_tracked")

SILENT, VOCAL, NONPARALLEL = "silent_parallel", "voiced_parallel", "nonparallel"
CONDITIONS = [("silent", SILENT), ("vocal", VOCAL)]   # the two benchmark conditions (dev/test)
