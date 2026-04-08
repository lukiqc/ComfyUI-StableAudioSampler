"""
@author: lks-ai
@title: StableAudioSampler
@nickname: stableaudio
@description: A Simple integration of Stable Audio Diffusion with knobs and stuff!
"""

from .nodes import StableAudioSampler, NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
print("[comfyui-stable-audio-sampler] StableAudioSampler loaded successfully.")

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
WEB_DIRECTORY = "./web"
