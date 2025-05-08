import os
import sys

from mmgp import offload

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'xcodec_mini_infer'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'xcodec_mini_infer', 'descriptaudiocodec'))
import re
import copy
from tqdm import tqdm
from collections import Counter
import argparse
import numpy as np
import torch
import torchaudio
import time
from datetime import datetime
from torchaudio.transforms import Resample
import soundfile as sf
from einops import rearrange
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList
from omegaconf import OmegaConf
from codecmanipulator import CodecManipulator
from mmtokenizer import _MMSentencePieceTokenizer
from xcodec_mini_infer.models.soundstream_hubert_new import SoundStream
from xcodec_mini_infer.vocoder import build_codec_model, process_audio
from xcodec_mini_infer.post_process_audio import replace_low_freq_with_energy_matched
import gradio as gr

parser = argparse.ArgumentParser()
# Model Configuration:
parser.add_argument("--max_new_tokens", type=int, default=3000,
                    help="The maximum number of new tokens to generate in one pass during text generation.")
parser.add_argument("--run_n_segments", type=int, default=2,
                    help="The number of segments to process during the generation.")
# Prompt
parser.add_argument("--genre_txt", type=str, default="prompt_examples/genrerock.txt",
                    help="The file path to a text file containing genre tags that describe the musical style or characteristics (e.g., instrumental, genre, mood, vocal timbre, vocal gender). This is used as part of the generation prompt.")
parser.add_argument("--lyrics_txt", type=str, default="prompt_examples/lastxmas.txt",
                    help="The file path to a text file containing the lyrics for the music generation. These lyrics will be processed and split into structured segments to guide the generation process.")
parser.add_argument("--use_audio_prompt", action="store_true",
                    help="If set, the model will use an audio file as a prompt during generation. The audio file should be specified using --audio_prompt_path.")
parser.add_argument("--audio_prompt_path", type=str, default="",
                    help="The file path to an audio file to use as a reference prompt when --use_audio_prompt is enabled.")
parser.add_argument("--prompt_start_time", type=float, default=0.0,
                    help="The start time in seconds to extract the audio prompt from the given audio file.")
parser.add_argument("--prompt_end_time", type=float, default=30.0,
                    help="The end time in seconds to extract the audio prompt from the given audio file.")
parser.add_argument("--use_dual_tracks_prompt", action="store_true",
                    help="If set, the model will use dual tracks as a prompt during generation. The vocal and instrumental files should be specified using --vocal_track_prompt_path and --instrumental_track_prompt_path.")
parser.add_argument("--vocal_track_prompt_path", type=str, default="",
                    help="The file path to a vocal track file to use as a reference prompt when --use_dual_tracks_prompt is enabled.")
parser.add_argument("--instrumental_track_prompt_path", type=str, default="",
                    help="The file path to an instrumental track file to use as a reference prompt when --use_dual_tracks_prompt is enabled.")
# Output 
parser.add_argument("--output_dir", type=str, default="./output",
                    help="The directory where generated outputs will be saved.")
parser.add_argument("--keep_intermediate", action="store_true",
                    help="If set, intermediate outputs will be saved during processing.")
parser.add_argument("--disable_offload_model", action="store_true",
                    help="If set, the model will not be offloaded from the GPU to CPU after Stage 1 inference.")
parser.add_argument("--cuda_idx", type=int, default=0)
# Config for xcodec and upsampler
parser.add_argument('--basic_model_config', default='./xcodec_mini_infer/final_ckpt/config.yaml',
                    help='YAML files for xcodec configurations.')
parser.add_argument('--resume_path', default='./xcodec_mini_infer/final_ckpt/ckpt_00360000.pth',
                    help='Path to the xcodec checkpoint.')
parser.add_argument('--config_path', type=str, default='./xcodec_mini_infer/decoders/config.yaml',
                    help='Path to Vocos config file.')
parser.add_argument('--vocal_decoder_path', type=str, default='./xcodec_mini_infer/decoders/decoder_131000.pth',
                    help='Path to Vocos decoder weights.')
parser.add_argument('--inst_decoder_path', type=str, default='./xcodec_mini_infer/decoders/decoder_151000.pth',
                    help='Path to Vocos decoder weights.')
parser.add_argument('-r', '--rescale', action='store_true', help='Rescale output to avoid clipping.')
parser.add_argument("--profile", type=int, default=3)
parser.add_argument("--verbose", type=int, default=1)
parser.add_argument("--compile", action="store_true")
parser.add_argument("--sdpa", action="store_true")
parser.add_argument("--icl", action="store_true")
parser.add_argument("--turbo-stage2", action="store_true")
# Gradio server
parser.add_argument("--server_name", type=str, default="localhost",
                    help="The server name for the wWbUI. By default it exposes the service to all network interfaces. Set to localhost, if you want to restrict access to the local machine.")
parser.add_argument("--server_port", type=int, default=7860, help="The port number for the WebUI.")

args = parser.parse_args()

# set up arguments
profile = args.profile
compile = args.compile
sdpa = args.sdpa
use_icl = args.icl

if use_icl:
    args.stage1_model = "m-a-p/YuE-s1-7B-anneal-en-icl"
else:
    args.stage1_model = "m-a-p/YuE-s1-7B-anneal-en-cot"

args.stage2_model = "m-a-p/YuE-s2-1B-general"

args.stage2_batch_size = [20, 20, 20, 4, 3, 2][profile]

if sdpa:
    attn_implementation = "sdpa"
else:
    attn_implementation = "flash_attention_2"

if args.use_audio_prompt and not args.audio_prompt_path:
    raise FileNotFoundError(
        "Please offer audio prompt filepath using '--audio_prompt_path', when you enable 'use_audio_prompt'!")
if args.use_dual_tracks_prompt and not args.vocal_track_prompt_path and not args.instrumental_track_prompt_path:
    raise FileNotFoundError(
        "Please offer dual tracks prompt filepath using '--vocal_track_prompt_path' and '--inst_decoder_path', when you enable '--use_dual_tracks_prompt'!")
stage1_model = args.stage1_model
stage2_model = args.stage2_model
cuda_idx = args.cuda_idx
max_new_tokens = args.max_new_tokens
stage1_output_dir = os.path.join(args.output_dir, f"stage1")
stage2_output_dir = stage1_output_dir.replace('stage1', 'stage2')
os.makedirs(stage1_output_dir, exist_ok=True)
os.makedirs(stage2_output_dir, exist_ok=True)

# load tokenizer and model
device = torch.device(f"cuda:{cuda_idx}" if torch.cuda.is_available() else "cpu")
mmtokenizer = _MMSentencePieceTokenizer("./mm_tokenizer_v0.2_hf/tokenizer.model")
model = AutoModelForCausalLM.from_pretrained(
    stage1_model,
    torch_dtype=torch.bfloat16,
    attn_implementation=attn_implementation,  # To enable flashattn, you have to install flash-attn
)
# to device, if gpu is available
model.to(device)
model.eval()

model_stage2 = AutoModelForCausalLM.from_pretrained(
    stage2_model,
    torch_dtype=torch.float16,
    attn_implementation=attn_implementation,
)
model_stage2.to(device)
model_stage2.eval()


# remove test on arguments for method 'model.generate' in case transformers patch not applied
def nop(nada):
    pass


model._validate_model_kwargs = nop
model_stage2._validate_model_kwargs = nop

pipe = {"transformer": model, "stage2": model_stage2}

quantizeTransformer = profile == 3 or profile == 4 or profile == 5

codectool = CodecManipulator("xcodec", 0, 1)
codectool_stage2 = CodecManipulator("xcodec", 0, 8)
model_config = OmegaConf.load(args.basic_model_config)
codec_model = eval(model_config.generator.name)(**model_config.generator.config).to(device)
parameter_dict = torch.load(args.resume_path, map_location="cpu", weights_only=False)
codec_model.load_state_dict(parameter_dict['codec_model'])
codec_model.to('cpu')
codec_model.eval()
kwargs = {}
if profile == 5:
    kwargs["budgets"] = {"transformer": 500, "*": 3000}
    kwargs["pinnedMemory"] = True
elif profile == 4:
    kwargs["budgets"] = {"transformer": 3000, "*": 5000}
elif profile == 2:
    kwargs["budgets"] = 5000

offload.profile(pipe, profile_no=profile, compile=compile, quantizeTransformer=quantizeTransformer,
                verboseLevel=args.verbose if args.verbose is not None else 1, **kwargs)  # pinnedMemory=False,


class BlockTokenRangeProcessor(LogitsProcessor):
    def __init__(self, start_id, end_id):
        self.blocked_token_ids = list(range(start_id, end_id))
        self.start_id = start_id
        self.end_id = end_id

    def __call__(self, input_ids, scores):
        # scores[:, self.blocked_token_ids] = -float("inf")
        scores[:, self.start_id: self.end_id] = -float("inf")

        return scores


def load_audio_mono(filepath, sampling_rate=16000):
    audio, sr = torchaudio.load(filepath)
    # Convert to mono
    audio = torch.mean(audio, dim=0, keepdim=True)
    # Resample if needed
    if sr != sampling_rate:
        resampler = Resample(orig_freq=sr, new_freq=sampling_rate)
        audio = resampler(audio)
    return audio


def encode_audio(codec_model, audio_prompt, device, target_bw=0.5):
    if len(audio_prompt.shape) < 3:
        audio_prompt.unsqueeze_(0)
    with torch.no_grad():
        raw_codes = codec_model.encode(audio_prompt.to(device), target_bw=target_bw)
    raw_codes = raw_codes.transpose(0, 1)
    raw_codes = raw_codes.cpu().numpy().astype(np.int16)
    return raw_codes


def split_lyrics(lyrics):
    pattern = r"\[(\w+)\](.*?)\n(?=\[|\Z)"
    segments = re.findall(pattern, lyrics, re.DOTALL)
    structured_lyrics = [f"[{seg[0]}]\n{seg[1].strip()}\n\n" for seg in segments]
    return structured_lyrics


def get_song_id(seed, genres, top_p, temperature, repetition_penalty, max_new_tokens):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M-%S.%f")[:-3]

    genres = re.sub(r'[^a-zA-Z0-9_-]', '_', genres.replace(' ', '-'))
    genres = re.sub(r'_+', '_', genres).strip('_')
    genres = genres[:180]

    song_id = f"{timestamp}_{genres}_seed{seed}_tp{top_p}_T{temperature}_rp{repetition_penalty}_maxtk{max_new_tokens}"

    return song_id[:240]


def stage1_inference(genres, lyrics_input, run_n_segments, max_new_tokens, seed, state=None, callback=None):
    # Tips:
    # genre tags support instrumental，genre，mood，vocal timbr and vocal gender
    # all kinds of tags are needed
    genres = genres.strip()

    lyrics = split_lyrics(lyrics_input)
    # instruction
    full_lyrics = "\n".join(lyrics)
    prompt_texts = [f"Generate music from the given lyrics segment by segment.\n[Genre] {genres}\n{full_lyrics}"]
    prompt_texts += lyrics

    # Here is suggested decoding config
    top_p = 0.93
    temperature = 1.0
    repetition_penalty = 1.2
    # special tokens
    start_of_segment = mmtokenizer.tokenize('[start_of_segment]')
    end_of_segment = mmtokenizer.tokenize('[end_of_segment]')
    # Format text prompt
    run_n_segments = min(run_n_segments, len(lyrics))
    for i, p in enumerate(tqdm(prompt_texts[1:run_n_segments + 1]), 1):
        # print(f"---Stage 1: Generating Sequence {i} out of {run_n_segments}")
        state["stage"] = f"Stage 1: Generating Sequence {i} out of {run_n_segments}"
        section_text = p.replace('[start_of_segment]', '').replace('[end_of_segment]', '')
        guidance_scale = 1.5 if i <= 1 else 1.2
        if i == 1:
            if args.use_dual_tracks_prompt or args.use_audio_prompt:
                if args.use_dual_tracks_prompt:
                    vocals_ids = load_audio_mono(args.vocal_track_prompt_path)
                    instrumental_ids = load_audio_mono(args.instrumental_track_prompt_path)
                    vocals_ids = encode_audio(codec_model, vocals_ids, device, target_bw=0.5)
                    instrumental_ids = encode_audio(codec_model, instrumental_ids, device, target_bw=0.5)
                    vocals_ids = codectool.npy2ids(vocals_ids[0])
                    instrumental_ids = codectool.npy2ids(instrumental_ids[0])
                    min_size = min(len(vocals_ids), len(instrumental_ids))
                    vocals_ids = vocals_ids[0: min_size]
                    instrumental_ids = instrumental_ids[0: min_size]
                    ids_segment_interleaved = rearrange([np.array(vocals_ids), np.array(instrumental_ids)],
                                                        'b n -> (n b)')
                    audio_prompt_codec = ids_segment_interleaved[
                                         int(args.prompt_start_time * 50 * 2): int(args.prompt_end_time * 50 * 2)]
                    audio_prompt_codec = audio_prompt_codec.tolist()
                elif args.use_audio_prompt:
                    audio_prompt = load_audio_mono(args.audio_prompt_path)
                    raw_codes = encode_audio(codec_model, audio_prompt, device, target_bw=0.5)
                    # Format audio prompt
                    code_ids = codectool.npy2ids(raw_codes[0])
                    audio_prompt_codec = code_ids[int(args.prompt_start_time * 50): int(
                        args.prompt_end_time * 50)]  # 50 is tps of xcodec
                audio_prompt_codec_ids = [mmtokenizer.soa] + codectool.sep_ids + audio_prompt_codec + [mmtokenizer.eoa]
                sentence_ids = mmtokenizer.tokenize(
                    "[start_of_reference]") + audio_prompt_codec_ids + mmtokenizer.tokenize("[end_of_reference]")
                head_id = mmtokenizer.tokenize(prompt_texts[0]) + sentence_ids
            else:
                head_id = mmtokenizer.tokenize(prompt_texts[0])
            prompt_ids = head_id + start_of_segment + mmtokenizer.tokenize(section_text) + [
                mmtokenizer.soa] + codectool.sep_ids
        else:
            prompt_ids = end_of_segment + start_of_segment + mmtokenizer.tokenize(section_text) + [
                mmtokenizer.soa] + codectool.sep_ids

        prompt_ids = torch.as_tensor(prompt_ids).unsqueeze(0).to(device)
        input_ids = torch.cat([raw_output, prompt_ids], dim=1) if i > 1 else prompt_ids
        # Use window slicing in case output sequence exceeds the context of model
        max_context = 16384 - max_new_tokens - 1
        if input_ids.shape[-1] > max_context:
            print(
                f'Section {i}: output length {input_ids.shape[-1]} exceeding context length {max_context}, now using the last {max_context} tokens.')
            input_ids = input_ids[:, -(max_context):]
        with torch.no_grad():
            output_seq = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                min_new_tokens=100,
                do_sample=True,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                eos_token_id=mmtokenizer.eoa,
                pad_token_id=mmtokenizer.eoa,
                logits_processor=LogitsProcessorList(
                    [BlockTokenRangeProcessor(0, 32002), BlockTokenRangeProcessor(32016, 32017)]),
                guidance_scale=guidance_scale,
                callback=callback,
            )
            torch.cuda.empty_cache()
            if output_seq[0][-1].item() != mmtokenizer.eoa:
                tensor_eoa = torch.as_tensor([[mmtokenizer.eoa]]).to(model.device)
                output_seq = torch.cat((output_seq, tensor_eoa), dim=1)
        if i > 1:
            raw_output = torch.cat([raw_output, prompt_ids, output_seq[:, input_ids.shape[-1]:]], dim=1)
        else:
            raw_output = output_seq

    # save raw output and check sanity
    ids = raw_output[0].cpu().numpy()
    soa_idx = np.where(ids == mmtokenizer.soa)[0].tolist()
    eoa_idx = np.where(ids == mmtokenizer.eoa)[0].tolist()
    if len(soa_idx) != len(eoa_idx):
        raise ValueError(f'invalid pairs of soa and eoa, Num of soa: {len(soa_idx)}, Num of eoa: {len(eoa_idx)}')

    vocals = []
    instrumentals = []
    range_begin = 1 if args.use_audio_prompt or args.use_dual_tracks_prompt else 0
    for i in range(range_begin, len(soa_idx)):
        codec_ids = ids[soa_idx[i] + 1:eoa_idx[i]]
        if codec_ids[0] == 32016:
            codec_ids = codec_ids[1:]
        codec_ids = codec_ids[:2 * (codec_ids.shape[0] // 2)]
        vocals_ids = codectool.ids2npy(rearrange(codec_ids, "(n b) -> b n", b=2)[0])
        vocals.append(vocals_ids)
        instrumentals_ids = codectool.ids2npy(rearrange(codec_ids, "(n b) -> b n", b=2)[1])
        instrumentals.append(instrumentals_ids)
    vocals = np.concatenate(vocals, axis=1)
    instrumentals = np.concatenate(instrumentals, axis=1)
    song_id = get_song_id(seed, genres, top_p, temperature, repetition_penalty, max_new_tokens)
    vocal_save_path = os.path.join(stage1_output_dir, f"{song_id}_vtrack.npy")
    inst_save_path = os.path.join(stage1_output_dir, f"{song_id}_itrack.npy")
    np.save(vocal_save_path, vocals)
    np.save(inst_save_path, instrumentals)
    stage1_output_set = []
    stage1_output_set.append(vocal_save_path)
    stage1_output_set.append(inst_save_path)
    return stage1_output_set


def stage2_generate(model, prompt, batch_size=16, segment_duration=6, state=None, callback=None):
    codec_ids = codectool.unflatten(prompt, n_quantizer=1)
    codec_ids = codectool.offset_tok_ids(
        codec_ids,
        global_offset=codectool.global_offset,
        codebook_size=codectool.codebook_size,
        num_codebooks=codectool.num_codebooks,
    ).astype(np.int32)

    # Prepare prompt_ids based on batch size or single input
    if batch_size > 1:
        codec_list = []
        for i in range(batch_size):
            idx_begin = i * segment_duration * 50
            idx_end = (i + 1) * segment_duration * 50
            codec_list.append(codec_ids[:, idx_begin:idx_end])

        codec_ids = np.concatenate(codec_list, axis=0)
        prompt_ids = np.concatenate(
            [
                np.tile([mmtokenizer.soa, mmtokenizer.stage_1], (batch_size, 1)),
                codec_ids,
                np.tile([mmtokenizer.stage_2], (batch_size, 1)),
            ],
            axis=1
        )
    else:
        prompt_ids = np.concatenate([
            np.array([mmtokenizer.soa, mmtokenizer.stage_1]),
            codec_ids.flatten(),  # Flatten the 2D array to 1D
            np.array([mmtokenizer.stage_2])
        ]).astype(np.int32)
        prompt_ids = prompt_ids[np.newaxis, ...]

    codec_ids = torch.as_tensor(codec_ids).to(device)
    prompt_ids = torch.as_tensor(prompt_ids).to(device)
    len_prompt = prompt_ids.shape[-1]

    block_list = LogitsProcessorList(
        [BlockTokenRangeProcessor(0, 46358), BlockTokenRangeProcessor(53526, mmtokenizer.vocab_size)])

    # Teacher forcing generate loop

    max_tokens = codec_ids.shape[1] * 8
    i = 0
    real_max_length = codec_ids.shape[1] * 8 + prompt_ids.shape[1]
    session_cache = {"real_max_length": real_max_length}
    codec_ids.shape[1]
    for frames_idx in range(codec_ids.shape[1]):
        if i % 96 == 0:
            # print(f"Tokens: {i} out of {max_tokens}")
            callback(i, real_max_length)

        cb0 = codec_ids[:, frames_idx:frames_idx + 1]
        # print(f"insert cb0: {cb0}")
        prompt_ids = torch.cat([prompt_ids, cb0], dim=1)
        input_ids = prompt_ids

        with torch.no_grad():
            stage2_output = model.generate(input_ids=input_ids,
                                           min_new_tokens=7,
                                           max_new_tokens=7,
                                           eos_token_id=mmtokenizer.eoa,
                                           pad_token_id=mmtokenizer.eoa,
                                           logits_processor=block_list,
                                           session_cache=session_cache,
                                           )

        assert stage2_output.shape[1] - prompt_ids.shape[
            1] == 7, f"output new tokens={stage2_output.shape[1] - prompt_ids.shape[1]}"
        prompt_ids = stage2_output
        i += 8

    del session_cache
    torch.cuda.empty_cache()

    # Return output based on batch size
    if batch_size > 1:
        output = prompt_ids.cpu().numpy()[:, len_prompt:]
        output_list = [output[i] for i in range(batch_size)]
        output = np.concatenate(output_list, axis=0)
    else:
        output = prompt_ids[0].cpu().numpy()[len_prompt:]

    return output


def stage2_inference(model, stage1_output_set, stage2_output_dir, batch_size=4, segment_duration=6, state=None,
                     callback=None):
    stage2_result = []
    for i in tqdm(range(len(stage1_output_set))):
        if i == 0:
            # print("---Stage 2.1: Sampling Vocal track")
            prefix = "Stage 2.1: Sampling Vocal track"
        else:
            # print("---Stage 2.2: Sampling Instrumental track")
            prefix = "Stage 2.2: Sampling Instrumental track"

        output_filename = os.path.join(stage2_output_dir, os.path.basename(stage1_output_set[i]))

        if os.path.exists(output_filename) and False:
            print(f'{output_filename} stage2 has done.')
            stage2_result.append(output_filename)
            continue

        # Load the prompt
        prompt = np.load(stage1_output_set[i]).astype(np.int32)
        segment_length = 3
        # Only accept 6s segments ( = segment_duration )
        output_duration = prompt.shape[-1] // 50 // segment_duration * segment_duration
        num_batch = output_duration // segment_duration

        any_trail = output_duration * 50 != prompt.shape[-1]

        if num_batch <= batch_size:
            # If num_batch is less than or equal to batch_size, we can infer the entire prompt at once
            # print("Only one segment to process for this track")               
            max_segments = 2 if any_trail else 1
            if max_segments == 1:
                state["stage"] = prefix
            else:
                state["stage"] = prefix + f", segment 1 out of {max_segments}"
            output = stage2_generate(model, prompt[:, :output_duration * 50], batch_size=num_batch,
                                     segment_duration=segment_duration, state=state, callback=callback)
        else:
            # If num_batch is greater than batch_size, process in chunks of batch_size
            segments = []
            num_segments = (num_batch // batch_size) + (1 if num_batch % batch_size != 0 else 0)

            max_segments = num_segments + 1 if any_trail else num_segments
            for seg in range(num_segments):
                # print(f"Segment {seg+1} out of {max_segments}")
                state["stage"] = prefix + f", segment {seg + 1} out of {max_segments}"
                start_idx = seg * batch_size * 300
                # Ensure the end_idx does not exceed the available length
                end_idx = min((seg + 1) * batch_size * 300, output_duration * 50)  # Adjust the last segment
                current_batch_size = batch_size if seg != num_segments - 1 or num_batch % batch_size == 0 else num_batch % batch_size
                segment = stage2_generate(
                    model,
                    prompt[:, start_idx:end_idx],
                    batch_size=current_batch_size,
                    segment_duration=segment_duration,
                    state=state,
                    callback=callback
                )
                segments.append(segment)

            # Concatenate all the segments
            output = np.concatenate(segments, axis=0)

        # Process the ending part of the prompt
        if any_trail:
            # print(f"Segment {max_segments} / {max_segments}")
            state["stage"] = prefix + f", segment {max_segments} out of {max_segments}"
            ending = stage2_generate(model, prompt[:, output_duration * 50:], batch_size=1,
                                     segment_duration=segment_duration, state=state, callback=callback)
            output = np.concatenate([output, ending], axis=0)
        output = codectool_stage2.ids2npy(output)

        # Fix invalid codes (a dirty solution, which may harm the quality of audio)
        # We are trying to find better one
        fixed_output = copy.deepcopy(output)
        for i, line in enumerate(output):
            for j, element in enumerate(line):
                if element < 0 or element > 1023:
                    counter = Counter(line)
                    most_frequant = sorted(counter.items(), key=lambda x: x[1], reverse=True)[0][0]
                    fixed_output[i, j] = most_frequant
        # save output
        np.save(output_filename, fixed_output)
        stage2_result.append(output_filename)
    return stage2_result


def build_callback(state, progress, status):
    def callback(tokens_processed, max_tokens):
        prefix = state["prefix"]
        status = prefix + state["stage"]
        tokens_processed += 1
        if state.get("abort", False):
            status_msg = status + " - Aborting"
            raise Exception("abort")
            # pipe._interrupt = True
        # elif step_idx  == num_inference_steps:
        #     status_msg = status + " - VAE Decoding"    
        else:
            status_msg = status  # + " - Denoising"

        progress(tokens_processed / max_tokens, desc=status_msg, unit=" %")

    return callback


def abort_generation(state):
    if "in_progress" in state:
        state["abort"] = True
        return gr.Button(interactive=False)
    else:
        return gr.Button(interactive=True)


def refresh_gallery(state):
    file_list = state.get("file_list", None)
    if len(file_list) > 0:
        return file_list[0], file_list
    else:
        return None, file_list


def finalize_gallery(state):
    if "in_progress" in state:
        del state["in_progress"]
    time.sleep(0.2)
    return gr.Button(interactive=True)


def generate_song(genres_input, lyrics_input, run_n_segments, seed, max_new_tokens, vocal_track_prompt,
                  instrumental_track_prompt, prompt_start_time, prompt_end_time, repeat_generation, state,
                  progress=gr.Progress()):
    args.use_audio_prompt = False
    args.use_dual_tracks_prompt = False
    # Call the function and print the result

    if "abort" in state:
        del state["abort"]
    state["in_progress"] = True
    state["selected"] = 0
    file_list = state.get("file_list", [])
    if len(file_list) == 0:
        state["file_list"] = file_list

    if use_icl:
        if prompt_start_time > prompt_end_time:
            raise gr.Error(f"'Start time' should be less than 'End Time'")
        if (prompt_end_time - prompt_start_time) > 30:
            raise gr.Error(f"The duration for the audio prompt should not exceed 30s")
        if vocal_track_prompt == None:
            raise gr.Error(f"You must provide at least a Vocal audio prompt")
        args.prompt_start_time = prompt_start_time
        args.prompt_end_time = prompt_end_time

        if instrumental_track_prompt == None:
            args.use_audio_prompt = True
            args.audio_prompt_path = vocal_track_prompt
        else:
            args.use_dual_tracks_prompt = True
            args.vocal_track_prompt_path = vocal_track_prompt
            args.instrumental_track_prompt_path = instrumental_track_prompt

    segment_duration = 3 if args.turbo_stage2 else 6

    import random

    if seed <= 0:
        seed = random.randint(0, 999999999)

    genres_input = genres_input.replace("\r", "").split("\n")
    song_no = 0
    total_songs = repeat_generation * len(genres_input)

    start_time = time.time()
    for genres_no, genres in enumerate(genres_input):
        for gen_no in range(repeat_generation):
            song_no += 1
            prefix = ""
            status = f"Song {song_no}/{total_songs}"
            if len(genres_input) > 1:
                prefix += f"Genres {genres_no + 1}/{len(genres_input)} > "
            if repeat_generation > 1:
                prefix += f"Generation {gen_no + 1}/{repeat_generation} > "
            state["prefix"] = prefix

            # return "output/cot_inspiring-female-uplifting-pop-airy-vocal-electronic-bright-vocal-vocal_tp0@93_T1@0_rp1@2_maxtk3000_mixed_e0a99c45-7f63-41c9-826f-9bde7417db4c.mp3"

            torch.cuda.manual_seed(seed)
            random.seed(seed)

            callback = build_callback(state, progress, status)

            # if True:
            try:
                stage1_output_set = stage1_inference(genres, lyrics_input, run_n_segments, max_new_tokens, seed, state,
                                                     callback)

                # random_id ="5b4b4613-1cc2-4d84-af7a-243f853f168b"
                # stage1_output_set = [ "output/stage1/inspiring-female-uplifting-pop-airy-vocal-electronic-bright-vocal_tp0@93_T1@0_rp1@2_maxtk3000_5b4b4613-1cc2-4d84-af7a-243f853f168b_vtrack.npy", 
                #                       "output/stage1/inspiring-female-uplifting-pop-airy-vocal-electronic-bright-vocal_tp0@93_T1@0_rp1@2_maxtk3000_5b4b4613-1cc2-4d84-af7a-243f853f168b_itrack.npy"]

                stage2_result = stage2_inference(model_stage2, stage1_output_set, stage2_output_dir,
                                                 batch_size=args.stage2_batch_size, segment_duration=segment_duration,
                                                 state=state, callback=callback)
            except Exception as e:
                s = str(e)
                if "abort" in s:
                    stage2_result = None
                else:
                    raise

            if stage2_result == None:
                end_time = time.time()
                yield f"Song Generation Aborted. Total Generation Time: {end_time - start_time:.1f}s"
                return

            print(stage2_result)
            print('Stage 2 DONE.\n')

            # convert audio tokens to audio
            def save_audio(wav: torch.Tensor, path, sample_rate: int, rescale: bool = False):
                folder_path = os.path.dirname(path)
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                limit = 0.99
                max_val = wav.abs().max()
                wav = wav * min(limit / max_val, 1) if rescale else wav.clamp(-limit, limit)
                torchaudio.save(str(path), wav, sample_rate=sample_rate, encoding='PCM_S', bits_per_sample=16)

            # reconstruct tracks
            recons_output_dir = os.path.join(args.output_dir, "recons")
            recons_mix_dir = os.path.join(recons_output_dir, 'mix')
            os.makedirs(recons_mix_dir, exist_ok=True)
            tracks = []
            for npy in stage2_result:
                codec_result = np.load(npy)
                decodec_rlt = []
                with torch.no_grad():
                    decoded_waveform = codec_model.decode(
                        torch.as_tensor(codec_result.astype(np.int16), dtype=torch.long).unsqueeze(0).permute(1, 0,
                                                                                                              2).to(
                            device))
                decoded_waveform = decoded_waveform.cpu().squeeze(0)
                decodec_rlt.append(torch.as_tensor(decoded_waveform, device="cpu"))
                decodec_rlt = torch.cat(decodec_rlt, dim=-1)
                save_path = os.path.join(recons_output_dir, os.path.splitext(os.path.basename(npy))[0] + ".mp3")
                tracks.append(save_path)
                save_audio(decodec_rlt, save_path, 16000)
            # mix tracks
            for inst_path in tracks:
                try:
                    if (inst_path.endswith('.wav') or inst_path.endswith('.mp3')) \
                            and '_itrack' in inst_path:
                        # find pair
                        vocal_path = inst_path.replace('_itrack', '_vtrack')
                        if not os.path.exists(vocal_path):
                            continue
                        # mix
                        recons_mix = os.path.join(recons_mix_dir,
                                                  os.path.basename(inst_path).replace('_itrack', '_mixed'))
                        vocal_stem, sr = sf.read(inst_path)
                        instrumental_stem, _ = sf.read(vocal_path)
                        mix_stem = (vocal_stem + instrumental_stem) / 1
                        sf.write(recons_mix, mix_stem, sr)
                except Exception as e:
                    print(e)

            # vocoder to upsample audios
            vocal_decoder, inst_decoder = build_codec_model(args.config_path, args.vocal_decoder_path,
                                                            args.inst_decoder_path)
            vocoder_output_dir = os.path.join(args.output_dir, 'vocoder')
            vocoder_stems_dir = os.path.join(vocoder_output_dir, 'stems')
            vocoder_mix_dir = os.path.join(vocoder_output_dir, 'mix')
            os.makedirs(vocoder_mix_dir, exist_ok=True)
            os.makedirs(vocoder_stems_dir, exist_ok=True)
            for npy in stage2_result:
                if '_itrack' in npy:
                    # Process instrumental
                    instrumental_output = process_audio(
                        npy,
                        os.path.join(vocoder_stems_dir, 'itrack.mp3'),
                        args.rescale,
                        args,
                        inst_decoder,
                        codec_model
                    )
                else:
                    # Process vocal
                    vocal_output = process_audio(
                        npy,
                        os.path.join(vocoder_stems_dir, 'vtrack.mp3'),
                        args.rescale,
                        args,
                        vocal_decoder,
                        codec_model
                    )
            # mix tracks
            try:
                mix_output = instrumental_output + vocal_output
                vocoder_mix = os.path.join(vocoder_mix_dir, os.path.basename(recons_mix))
                save_audio(mix_output, vocoder_mix, 44100, args.rescale)
                print(f"Created mix: {vocoder_mix}")
            except RuntimeError as e:
                print(e)
                print(f"mix {vocoder_mix} failed! inst: {instrumental_output.shape}, vocal: {vocal_output.shape}")

            # Post process
            output_file = os.path.join(args.output_dir, os.path.basename(recons_mix))
            replace_low_freq_with_energy_matched(
                a_file=recons_mix,  # 16kHz
                b_file=vocoder_mix,  # 48kHz
                c_file=output_file,
                cutoff_freq=5500.0
            )
            file_list.insert(0, output_file)
            if song_no < total_songs:
                yield status
            else:
                end_time = time.time()
                yield f"Total Generation Time: {end_time - start_time:.1f}s"
            seed += 1

            # return output_file


def create_demo():
    with gr.Blocks() as demo:
        gr.Markdown("<div align=center><H1>YuE<SUP>GP</SUP> v3</div>")

        gr.Markdown(
            "<H1><DIV ALIGN=CENTER>YuE is a groundbreaking series of open-source foundation models designed for music generation, specifically for transforming lyrics into full songs (lyrics2song).</DIV></H1>")
        gr.Markdown(
            "<H2><B>GPU Poor version by DeepBeepMeep</B> (<A HREF='https://github.com/deepbeepmeep/YuEGP'>Updates</A> / <A HREF='https://github.com/multimodal-art-projection/YuE'>Original</A>). Switch to profile 1 for fast generation (requires a 16 GB VRAM GPU), 1 min of song will take only 4 minutes</H2>")
        if use_icl:
            gr.Markdown(
                "<H3>With In Context Learning Mode in addition to the lyrics and genres info, you can provide audio prompts to describe your expectations. You can generate a song with either: </H3>")
            gr.Markdown("<H3>- a single mixed (song/instruments) Audio track prompt</H3>")
            gr.Markdown("<H3>- a Vocal track and an Instrumental track prompt</H3>")
            gr.Markdown(
                "Given some Lyrics and sample audio songs, you can try different Genres Prompt by separating each prompt by a carriage return.")
        else:
            gr.Markdown(
                "Given some Lyrics, you can try different Genres Prompt by separating each prompt by a carriage return.")

        with gr.Row():
            with gr.Column():
                with open(os.path.join("prompt_examples", "lyrics.txt")) as f:
                    lyrics_file = f.read()
                # lyrics_file.replace("\n", "\n\r")

                genres_input = gr.Text(label="Genres Prompt (one Genres Prompt per line for multiple generations)",
                                       value="inspiring female uplifting pop airy vocal electronic bright vocal",
                                       lines=3)
                lyrics_input = gr.Text(label="Lyrics", lines=20, value=lyrics_file)
                repeat_generation = gr.Slider(1, 25.0, value=1.0, step=1,
                                              label="Number of Generated Songs per Genres Prompt")

            with gr.Column():
                state = gr.State({})
                number_sequences = gr.Slider(1, 10, value=2, step=1,
                                             label="Number of Sequences (paragraphs in Lyrics, the higher this number, the higher the VRAM consumption)")
                max_new_tokens = gr.Slider(300, 6000, value=3000, step=300,
                                           label="Number of tokens per sequence (1000 tokens = 10s, the higher this number, the higher the VRAM consumption) ")

                seed = gr.Slider(0, 999999999, value=123, step=1, label="Seed (0 for random)")
                with gr.Row():
                    with gr.Column():
                        gen_status = gr.Text(label="Status", interactive=False)
                        generate_btn = gr.Button("Generate")
                        abort_btn = gr.Button("Abort")
                        output = gr.Audio(label="Last Generated Song")
                        files_history = gr.Files(label="History of Generated Songs (From most Recent to Oldest)",
                                                 type='filepath', height=150)
                        abort_btn.click(abort_generation, state, abort_btn)
                        gen_status.change(refresh_gallery, inputs=[state], outputs=[output, files_history])

        with gr.Row(visible=use_icl):  # use_icl
            with gr.Column():
                vocal_track_prompt = gr.Audio(label="Audio track prompt / Vocal track prompt", type='filepath')
            with gr.Column():
                instrumental_track_prompt = gr.Audio(
                    label="Intrumental track prompt (optional if Vocal track prompt set)", type='filepath')
        with gr.Row(visible=use_icl):
            with gr.Column():
                prompt_start_time = gr.Slider(0.0, 300.0, value=0.0, step=0.5, label="Audio Prompt Start time")
                prompt_end_time = gr.Slider(0.0, 300.0, value=30.0, step=0.5, label="Audio Prompt End time")

        abort_btn.click(abort_generation, state, abort_btn)

        generate_btn.click(
            fn=generate_song,
            inputs=[
                genres_input,
                lyrics_input,
                number_sequences,
                seed,
                max_new_tokens,
                vocal_track_prompt,
                instrumental_track_prompt,
                prompt_start_time,
                prompt_end_time,
                repeat_generation,
                state
            ],
            outputs=[gen_status]  # ,state

        ).then(
            finalize_gallery,
            [state],
            [abort_btn]
        )

    return demo


if __name__ == "__main__":
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    demo = create_demo()
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        allowed_paths=[args.output_dir])
