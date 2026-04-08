import os, sys, json, gc
import glob
import typing as tp
import io
import numpy as np

from einops import rearrange
from safetensors.torch import load_file
 

from .util_config import get_model_config


def _get_sao():
    """Lazily import stable_audio_tools when needed.
    Prefers local custom-extensions clone to avoid PyPI pins. Returns
    (get_pretrained_model, create_model_from_config, generate_diffusion_cond,
     generate_diffusion_uncond, prepare_audio, load_ckpt_state_dict)
    """
    current_path = os.path.dirname(os.path.abspath(__file__))
    stable_audio_path = os.path.abspath(
        os.path.join(current_path, '../../custom-extensions/stable-audio-tools')
    )

    # Prefer local clone first
    if os.path.isdir(stable_audio_path) and stable_audio_path not in sys.path:
        sys.path.insert(0, stable_audio_path)
        print(
            f"[comfyui-stable-audio-sampler] Using local stable-audio-tools: {stable_audio_path}"
        )
        try:
            from stable_audio_tools import get_pretrained_model, create_model_from_config  # type: ignore
            from stable_audio_tools.inference.generation import (  # type: ignore
                generate_diffusion_cond, generate_diffusion_uncond,
            )
            from stable_audio_tools.inference.utils import prepare_audio  # type: ignore
            from stable_audio_tools.models.utils import load_ckpt_state_dict  # type: ignore
            return (
                get_pretrained_model,
                create_model_from_config,
                generate_diffusion_cond,
                generate_diffusion_uncond,
                prepare_audio,
                load_ckpt_state_dict,
            )
        except ImportError:
            # Fall back to installed package
            pass

    # Try installed package
    try:
        from stable_audio_tools import get_pretrained_model, create_model_from_config  # type: ignore
        from stable_audio_tools.inference.generation import (
            generate_diffusion_cond, generate_diffusion_uncond,  # type: ignore
        )
        from stable_audio_tools.inference.utils import prepare_audio  # type: ignore
        from stable_audio_tools.models.utils import load_ckpt_state_dict  # type: ignore
        return (
            get_pretrained_model,
            create_model_from_config,
            generate_diffusion_cond,
            generate_diffusion_uncond,
            prepare_audio,
            load_ckpt_state_dict,
        )
    except ImportError as e:
        raise ImportError(
            "stable_audio_tools not found. Clone it to custom-extensions/stable-audio-tools "
            "or install via `pip install --no-deps stable-audio-tools` and add runtime deps manually."
        ) from e

# except ImportError as e:
#     checker = PackageDependencyChecker()
#     discrepancies = checker.check_version_discrepancies('requirements.txt')
#     #instructions = checker.generate_user_instructions(discrepancies)

#     # Find dependent discrepancies for all packages with issues
#     dependent_discrepancies = []
#     for discrepancy in discrepancies:
#         package_name = discrepancy['package_name']
#         dependent_discrepancies.extend(checker.check_dependents_discrepancies(package_name))
    
#     # Analyze discrepancies and suggest solutions
#     solutions = checker.analyze_discrepancies(discrepancies + dependent_discrepancies)
#     solution_suggestions = checker.suggest_solutions(solutions)
    
#     out = "\nSuggested solutions:\n"
#     for suggestion in solution_suggestions:
#         out += f"{suggestion}\n"
    
#     raise ValueError(f"<<StableAudioSampler>>: You Have some Environment Problems...\n\n{out}")

# Test current setup
# Add in Audio2Audio

# Comfy libs
def add_comfy_path():
    current_path = os.path.dirname(os.path.abspath(__file__))
    comfy_path = os.path.abspath(os.path.join(current_path, '../../../comfy'))
    if comfy_path not in sys.path:
        sys.path.insert(0, comfy_path)


add_comfy_path()

from comfy.utils import ProgressBar # type: ignore
import folder_paths # type: ignore

MAX_FP32 = np.iinfo(np.int32).max
SCHEDULERS = ["dpmpp-3m-sde", "dpmpp-2m-sde", "k-heun", "k-lms", "k-dpmpp-2s-ancestral", "k-dpm-2", "k-dpm-fast"]
ACKPT_FOLDER = "models/audio_checkpoints/"
TEMP_FOLDER = "temp/"

def get_device():
    try:
        import torch  # type: ignore
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

base_path = os.path.dirname(os.path.realpath(__file__))
os.makedirs(ACKPT_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)

# Our any instance wants to be a wildcard string
any = AnyType("*")
def get_models_path(ckpt_name):
    if not ckpt_name:
        return None
    return f"{ACKPT_FOLDER}{ckpt_name}"

model_files = [os.path.basename(file) for file in glob.glob(f"{ACKPT_FOLDER}*.safetensors")] + [os.path.basename(file) for file in glob.glob("models/audio_checkpoints/*.ckpt")]
config_files = [os.path.basename(file) for file in glob.glob(f"{ACKPT_FOLDER}*.json")]
if len(model_files) == 0:
    model_files.append(f"Put models in {ACKPT_FOLDER}")

def repo_path(repo, filename):
    path = os.path.join(repo, filename)
    instance_path = os.path.normpath(path)
    if sys.platform == 'win32':
        instance_path = instance_path.replace('\\', "/")
    return instance_path

import re
def replace_variables(template, values_dict):
    """Replace variables from a template where {} encloses a variable key from values_dict."""
    pattern = r'\{(\w+)\}'

    def replacer(match):
        variable_name = match.group(1)
        value = values_dict.get(variable_name, match.group(0))
        if isinstance(value, (str, int, float, bool)):
            return str(value)
        return match.group(0)

    result = re.sub(pattern, replacer, template)
    return result

# def wav_bytes_to_tensor(wav_bytes: bytes, model, sample_rate) -> tp.Tuple[int, torch.Tensor]:
#     # Load the audio data and sample rate using torchaudio
#     audio_tensor, in_sr = torchaudio.load(io.BytesIO(wav_bytes))
#     #audio_tensor = torch.from_numpy(audio_tensor).float().div(32767)
#     print("Before Transform", audio_tensor)
#     audio_tensor.float().div(32767)
#     print("Converted", audio_tensor)
#     if audio_tensor.dim() == 1:
#         audio_tensor = audio_tensor.unsqueeze(0) # [1, n]
#     elif audio_tensor.dim() == 2:
#         audio_tensor = audio_tensor.transpose(0, 1) # [n, 2] -> [2, n]    print(sample_rate)
#     print("Unsquoze", audio_tensor)
#     if in_sr != sample_rate:
#         resample_tf = T.Resample(in_sr, sample_rate).to(audio_tensor.device)
#         audio_tensor = resample_tf(audio_tensor)
#     print("Resampled", audio_tensor)
#     dtype = next(model.parameters()).dtype
#     audio_tensor = audio_tensor.to(dtype)
#     print("Retyped", audio_tensor)
#     return sample_rate, audio_tensor

def wav_bytes_to_tensor(wav_bytes: tp.Union[bytes, dict], model, sample_rate, sample_size: int):
    import torch  # local import
    import torchaudio  # local import
    from torchaudio import transforms as T
    # Handle dictionary input case
    if isinstance(wav_bytes, dict):
        if 'waveform' in wav_bytes:
            # Handle ComfyUI LoadAudio format
            audio_tensor = wav_bytes['waveform']
            in_sr = wav_bytes.get('sample_rate', sample_rate)
        elif 'path' in wav_bytes:
            # Load from file path
            audio_tensor, in_sr = torchaudio.load(wav_bytes['path'])
        elif 'tensor' in wav_bytes:
            # Direct tensor data
            audio_tensor = wav_bytes['tensor']
            in_sr = wav_bytes.get('sample_rate', sample_rate)
        elif 'filename' in wav_bytes:
            # Load from filename
            audio_tensor, in_sr = torchaudio.load(wav_bytes['filename'])
        else:
            print("Audio dictionary contents:", wav_bytes.keys())
            raise ValueError("Invalid audio dictionary format - must contain 'waveform', 'path', 'tensor', or 'filename' key")
    else:
        # Original bytes handling
        audio_tensor, in_sr = torchaudio.load(io.BytesIO(wav_bytes))

    # Handle different tensor shapes
    if audio_tensor.dim() == 3:
        # If we have a 3D tensor [batch, channels, samples], squeeze out the batch dimension
        audio_tensor = audio_tensor.squeeze(0)
    elif audio_tensor.dim() == 1:
        # If we have a 1D tensor [samples], add channel dimension
        audio_tensor = audio_tensor.unsqueeze(0)
    
    # Ensure we have [channels, samples] format
    if audio_tensor.shape[0] > audio_tensor.shape[-1]:
        audio_tensor = audio_tensor.transpose(0, -1)
    
    if in_sr != sample_rate:
        resample_tf = T.Resample(in_sr, sample_rate).to(audio_tensor.device)
        audio_tensor = resample_tf(audio_tensor)

    num_channels, num_samples = audio_tensor.shape
    if num_samples > sample_size:
        audio_tensor = audio_tensor[:, :sample_size]
    elif num_samples < sample_size:
        padding = torch.zeros((num_channels, sample_size - num_samples), dtype=audio_tensor.dtype, device=audio_tensor.device)
        audio_tensor = torch.cat((audio_tensor, padding), dim=1)

    dtype = next(model.parameters()).dtype
    audio_tensor = audio_tensor.to(dtype)
    
    return sample_rate, audio_tensor


def generate_audio(cond_batch, steps, cfg_scale, sigma_min, sigma_max, sampler_type, device, save, save_prefix, modelinfo, batch_size=1, seed=-1, after_generate="randomize", counter=0, init_noise_level=1.0, init_audio=None, quantum=True):
    import torch  # local import
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    model, sample_rate, sample_size, _device = modelinfo
    b_pos, b_neg = cond_batch
    p_conditioning, p_batch_size = b_pos
    n_conditioning, n_batch_size = b_neg
    sample_size = p_conditioning[0]['seconds_total'] * sample_rate

    #dprint("Model Loaded:", model)
    print("[comfyui-stable-audio-sampler, nodes.py, generate_audio] Positive Conditioning:", p_conditioning)
    print("[comfyui-stable-audio-sampler, nodes.py, generate_audio] Negative Conditioning:", n_conditioning)
    print("[comfyui-stable-audio-sampler, nodes.py, generate_audio] Sample Size:", sample_size)
    print("[comfyui-stable-audio-sampler, nodes.py, generate_audio] Sample Rate:", sample_rate)
    print("[comfyui-stable-audio-sampler, nodes.py, generate_audio] Seconds:", sample_size / sample_rate)

    # if init_audio is not None:
    #     print(len(init_audio))
    #     in_sr, init_audio = init_audio
    #     # Turn into torch tensor, converting from int16 to float32
    #     init_audio = torch.from_numpy(init_audio).float().div(32767)
        
    #     if init_audio.dim() == 1:
    #         init_audio = init_audio.unsqueeze(0) # [1, n]
    #     elif init_audio.dim() == 2:
    #         init_audio = init_audio.transpose(0, 1) # [n, 2] -> [2, n]

    #     if in_sr != sample_rate:
    #         resample_tf = T.Resample(in_sr, sample_rate).to(init_audio.device)
    #         init_audio = resample_tf(init_audio)

    #     audio_length = init_audio.shape[-1]

    #     if audio_length > sample_size:

    #         input_sample_size = audio_length + (model.min_input_length - (audio_length % model.min_input_length)) % model.min_input_length

    #     init_audio = (sample_rate, init_audio)

    wt = None if init_audio is None else wav_bytes_to_tensor(init_audio, model, sample_rate, sample_size)
    (
        get_pretrained_model,
        create_model_from_config,
        generate_diffusion_cond,
        generate_diffusion_uncond,
        prepare_audio,
        load_ckpt_state_dict,
    ) = _get_sao()

    output = generate_diffusion_cond(
        model,
        steps=steps,
        cfg_scale=cfg_scale,
        conditioning=p_conditioning,
        negative_conditioning=n_conditioning,
        sample_size=sample_size,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        sampler_type=sampler_type,
        device=_device,
        seed=seed,
        batch_size=p_batch_size,
        init_noise_level=init_noise_level, 
        init_audio=wt
    )
    
    gendata = locals()
    gendata['prompt'] = p_conditioning[0]['prompt']
    gendata['negative_prompt'] = n_conditioning[0]['prompt']

    print("[comfyui-stable-audio-sampler, nodes.py, generate_audio] Raw Output:", output)

    output = rearrange(output, "b d n -> d (b n)")
    print("[comfyui-stable-audio-sampler, nodes.py, generate_audio] Rearranged Output:", output)

    output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
    print("[comfyui-stable-audio-sampler, nodes.py, generate_audio] Transformed Output:", output)
    
    filepaths = None
    if save:
        filepaths = save_audio_files(output, sample_rate, save_prefix, counter, data=gendata)

    spectrogram = make_spectrogram_image(output, sample_rate)
    
    audio_bytes = output.numpy().tobytes()

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    return audio_bytes, sample_rate, spectrogram, filepaths


def get_model(model_filename=None, config=None, repo=None, half_precision=False, device_override=None):
    #print(model_filename, config, repo, half_precision)
    if model_filename:
        model_path = get_models_path(model_filename) #f"models/audio_checkpoints/{model_filename}"
        if model_filename.endswith(".safetensors") or model_filename.endswith(".ckpt"):
            if not config:
                model_config = get_model_config()
            else:
                with open(config, 'r') as f:
                    model_config = json.load(f)
            (_, create_model_from_config, _, _, _, load_ckpt_state_dict) = _get_sao()
            model = create_model_from_config(model_config)
            print(f"[comfyui-stable-audio-sampler, nodes.py, get_model] Model path: {model_path}")
            model.load_state_dict(load_ckpt_state_dict(model_path))
        else:
            repo_id = "stabilityai/stable-audio-open-1.0" if not repo else repo
            print(f"[comfyui-stable-audio-sampler, nodes.py, get_model] Loading pretrained model {repo_id}")
            (get_pretrained_model, _, _, _, _, _) = _get_sao()
            model, model_config = get_pretrained_model(repo_id)
        sample_rate = model_config["sample_rate"]
        sample_size = model_config["sample_size"]
    elif repo:
        if repo == "stabilityai/stable-audio-open-1.0":
            print(f"[comfyui-stable-audio-sampler, nodes.py, get_model] Loading pretrained model {repo}")
            (get_pretrained_model, _, _, _, _, _) = _get_sao()
            model, model_config = get_pretrained_model(repo)
        else:
            json_path = config or repo_path(repo, "model_config.json")
            model_path = repo_path(repo, "model.safetensors")
            with open(json_path) as f:
                model_config = json.load(f)
            (_, create_model_from_config, _, _, _, load_ckpt_state_dict) = _get_sao()
            model = create_model_from_config(model_config)
            model.load_state_dict(load_ckpt_state_dict(model_path), strict=False)
        sample_rate = model_config["sample_rate"]
        sample_size = model_config["sample_size"]
    else:
        raise ValueError("You must specify an Audio Checkpoint or a Repo to load from.")
    
    _device = (get_device() if not device_override else device_override)
    model = model.to(_device).requires_grad_(False) #.eval().requires_grad_(False)
    
    if half_precision and _device != "cpu":
        import torch  # local import
        model.to(torch.float16)
    
    return (model, sample_rate, sample_size, _device)

def load_model(model_config=None, model_ckpt_path=None, pretrained_name=None, pretransform_ckpt_path=None, device="cuda", model_half=False):
    global model, sample_rate, sample_size
    
    if pretrained_name is not None:
        print(f"[comfyui-stable-audio-sampler, nodes.py, load_model] Loading pretrained model {pretrained_name}")
        (get_pretrained_model, _, _, _, _, _) = _get_sao()
        model, model_config = get_pretrained_model(pretrained_name)

    elif model_config is not None and model_ckpt_path is not None:
        print(f"[comfyui-stable-audio-sampler, nodes.py, load_model] Creating model from config")
        (_, create_model_from_config, _, _, _, load_ckpt_state_dict) = _get_sao()
        model = create_model_from_config(model_config)

        print(f"[comfyui-stable-audio-sampler, nodes.py, load_model] Loading model checkpoint from {model_ckpt_path}")
        # Load checkpoint (allow missing/unexpected keys for broader compatibility)
        model.load_state_dict(load_ckpt_state_dict(model_ckpt_path), strict=False)

    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]

    if pretransform_ckpt_path is not None:
        print(f"[comfyui-stable-audio-sampler, nodes.py, load_model] Loading pretransform checkpoint from {pretransform_ckpt_path}")
        (_, _, _, _, _, load_ckpt_state_dict) = _get_sao()
        model.pretransform.load_state_dict(load_ckpt_state_dict(pretransform_ckpt_path), strict=False)
        print(f"[comfyui-stable-audio-sampler, nodes.py, load_model] Done loading pretransform")

    model.to(device).eval().requires_grad_(False)

    if model_half:
        import torch  # local import
        model.to(torch.float16)
        
    print(f"[comfyui-stable-audio-sampler, nodes.py, load_model] Done loading model")

    return model, model_config

import shutil
from urllib.parse import quote
def save_audio_files(output, sample_rate, filename_prefix, counter, data=None, save_temp=True):
    from datetime import datetime
    import torchaudio  # local import
    
    filename_prefix += ""
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get current datetime and format it
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create filename with datetime
    wavname = f"{current_time}_{filename_prefix}" if not data else f"{current_time}_{replace_variables(filename_prefix, data)}"
    
    filepaths = []
    for i, audio in enumerate(output):
        if i > 0: # TODO fix batches
            break
        fpath = f"{quote(wavname)}_{counter:04}.wav"
        file_path = os.path.join(output_dir, fpath)
        print(f"[comfyui-stable-audio-sampler, nodes.py, save_audio_files] Saving audio to {file_path}")
        torchaudio.save(file_path, audio.unsqueeze(0), sample_rate)
        filepaths.append(fpath)
        # Saves to temporary path so it can be used for streaming loops
        if save_temp:
            tpath = os.path.join(TEMP_FOLDER, "stableaudiosampler.wav")
            print(f"[comfyui-stable-audio-sampler, nodes.py, save_audio_files] Saving temp audio to: {tpath}")
            shutil.copyfile(file_path, tpath)
        counter += 1
    return filepaths

def make_spectrogram_image(output_tensor, sample_rate):
    try:
        from aeiou.viz import audio_spectrogram_image  # type: ignore
        return audio_spectrogram_image(output_tensor, sample_rate=sample_rate)
    except Exception as e:
        print("[comfyui-stable-audio-sampler] aeiou.viz unavailable; using basic waveform preview:", e)
        try:
            import numpy as np
            from PIL import Image  # type: ignore
            x = output_tensor.numpy()
            if x.ndim == 2:
                mono = np.mean(np.abs(x), axis=0)
            else:
                mono = np.abs(x).squeeze()
            mono = mono.astype(np.float32)
            maxv = np.max(mono) if mono.size else 1.0
            mono = mono / (maxv + 1e-6)
            width = min(1024, mono.shape[-1] if mono.size else 128)
            idx = np.linspace(0, mono.shape[-1]-1, width).astype(int) if mono.size else np.arange(128)
            vals = (mono[idx] * 127).astype(np.uint8) if mono.size else np.zeros(128, dtype=np.uint8)
            h = 128
            img = np.zeros((h, width), dtype=np.uint8)
            for i, v in enumerate(vals):
                img[h-1-v:, i] = 255
            rgb = np.stack([img]*3, axis=-1)
            return Image.fromarray(rgb)
        except Exception as e2:
            print("[comfyui-stable-audio-sampler] PIL fallback failed:", e2)
            return None

def create_image_batch(spectrograms, batch_size):
    import torch  # local import
    import numpy as np
    images = []
    for spec in spectrograms:
        arr = None
        try:
            from PIL import Image  # type: ignore
            if spec is not None and isinstance(spec, Image.Image):
                arr = np.array(spec.convert("RGB"))
        except Exception:
            pass
        if arr is None:
            try:
                arr = np.array(spec)
            except Exception:
                arr = np.zeros((128, 128, 3), dtype=np.uint8)
        images.append(torch.tensor(arr))
    return torch.stack(images)

class StableAudioSampler:
    def __init__(self):
        self.counter = 0
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio_model": ("SAOMODEL", {"forceInput": True}),
                "positive": ("CONDITIONING", {"forceInput": True}),
                "negative": ("CONDITIONING", {"forceInput": True}),
                "seed": ("INT", {"default": -1, "min": -1, "max": MAX_FP32}),
                "steps": ("INT", {"default": 100, "min": 1, "max": 10000}),
                "cfg_scale": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                # "sample_size": ("INT", {"default": 65536, "min": 1, "max": 1000000}),
                "sigma_min": ("FLOAT", {"default": 0.3, "min": 0.01, "max": 1000.0, "step": 0.01}),
                "sigma_max": ("FLOAT", {"default": 500.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                "sampler_type": (SCHEDULERS, {"default": "dpmpp-3m-sde"}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.01}),
                "save": ("BOOLEAN", {"default": True}),
                "save_prefix": ("STRING", {"default": "{prompt}-{seed}-{cfg_scale}-{steps}-{sigma_min}"}),
                "quantum": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "audio": (any, )
            }
        }

    RETURN_TYPES = ("AUDIO", "INT", "IMAGE")
    RETURN_NAMES = ("audio", "sample_rate", "image")
    FUNCTION = "sample"
    OUTPUT_NODE = True

    CATEGORY = "audio/samplers"

    def sample(self, audio_model, positive, negative, seed, steps, cfg_scale, sigma_min, sigma_max, sampler_type, denoise, save, save_prefix, quantum=True, audio=None):
        audio_bytes, sample_rate, spectrogram, filepaths = generate_audio(
            (positive, negative), 
            steps, 
            cfg_scale, 
            sigma_min, 
            sigma_max, 
            sampler_type, 
            get_device(), 
            save, 
            save_prefix, 
            audio_model, 
            seed=seed, 
            counter=self.counter, 
            init_noise_level=denoise, 
            init_audio=audio,
            quantum=quantum
        )
        # Convert int16 bytes to Comfy AUDIO payload {waveform [B,C,S] float32, sample_rate}
        import numpy as _np
        import torch as _torch
        int16_arr = _np.frombuffer(audio_bytes, dtype=_np.int16)
        if int16_arr.size == 0:
            float_arr = _np.zeros((2, 0), dtype=_np.float32)
        else:
            chans = 2  # model outputs stereo per config
            samples = int(int16_arr.size // chans)
            float_arr = (int16_arr[:samples*chans].reshape(chans, samples).astype(_np.float32) / 32767.0)
        waveform = _torch.from_numpy(float_arr).unsqueeze(0).contiguous()  # [1,C,S]
        audio_payload = {"waveform": waveform, "sample_rate": int(sample_rate)}

        spectrograms = create_image_batch([spectrogram], 1)
        return {"ui": {"paths": filepaths}, "result": (audio_payload, sample_rate, spectrograms)}
        #return (audio_bytes, sample_rate, spectrograms)

class StableLoadAudioModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_filename": (model_files, ),
            },
            "optional": {
                "model_config": (config_files, ),
                "repo": ("STRING", {"default": "stabilityai/stable-audio-open-1.0"}),
                "half_precision": ("BOOLEAN", {"default": False}),
                "force_cpu": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("SAOMODEL", )
    RETURN_NAMES = ("audio_model", )
    FUNCTION = "load"

    CATEGORY = "audio/loaders"

    def load(self, model_filename, model_config=None, repo=None, half_precision=None, force_cpu=None):
        mpath = get_models_path(model_config)
        modelinfo = get_model(model_filename=model_filename, config=mpath, repo=repo, half_precision=half_precision, device_override=None if not force_cpu else "cpu")
        return (modelinfo,)
    
class StableAudioPrompt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conditioning": ("CONDITIONING", {"forceInput": True}),
                "prompt": ("STRING", {"multiline": True}),
            }
        }
 
    RETURN_TYPES = ("CONDITIONING", )
    RETURN_NAMES = ("conditioning", )
    FUNCTION = "go"

    CATEGORY = "audio/conditioning"

    def go(self, conditioning, prompt):
        print("[comfyui-stable-audio-sampler, nodes.py, go] PROMPT", prompt)
        cond, batch_size = conditioning
        print("[comfyui-stable-audio-sampler, nodes.py, go]  cond, batch_size", cond, batch_size)
        o = []
        #cond[0]['prompt'] = prompt
        for v in cond:
            v['prompt'] = prompt
            o.append(v.copy())
        #c = conditioning[0]
        # conditioning = [{
        #     "prompt": prompt,
        #     "seconds_start": seconds_start,
        #     "seconds_total": seconds_total
        # }]
        print("[comfyui-stable-audio-sampler, nodes.py, go] o, batch_size", o, batch_size)
        return ((o, batch_size), )

import time

class StableAudioConditioning:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "seconds_start": ("INT", {"default": 0, "min": 0, "max": 60, "step": 1, "display": "number"}),
                "seconds_total": ("INT", {"default": 30, "min": 0, "max": 60, "step": 1, "display": "number"}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 50, "step": 1, "display": "number"}),
            }
        }
 
    RETURN_TYPES = ("CONDITIONING", )
    RETURN_NAMES = ("conditioning", )
    FUNCTION = "go"

    CATEGORY = "audio/conditioning"

    def go(self, seconds_start, seconds_total, batch_size):
        conditioning = [{
            "prompt": None,
            "seconds_start": seconds_start,
            "seconds_total": seconds_total
        }]
        return ((conditioning, batch_size), )

    @classmethod
    def IS_CHANGED(s, **kwargs):
        return time.time()

NODE_CLASS_MAPPINGS = {
    "StableAudioSampler": StableAudioSampler,
    "StableAudioLoadModel": StableLoadAudioModel,
    "StableAudioPrompt": StableAudioPrompt,
    "StableAudioConditioning": StableAudioConditioning,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StableAudioSampler": "Stable Audio Sampler",
    "StableAudioLoadModel": "Load Stable Audio Model",
    "StableAudioPrompt": "Stable Audio Prompt",
    "StableAudioConditioning": "Stable Audio Pre-Conditioning"
}
