# Model and license manifest

Reviewed 2026-08-18. This is an engineering inventory, not legal advice. Model
weights are downloaded into user-controlled caches and are not redistributed
with this repository.

| Component | Identifier/version | Upstream terms | Operational note |
|---|---|---|---|
| Whisper ASR weights | `openai/whisper-large-v3[-turbo]` | MIT upstream; Hugging Face metadata says Apache-2.0 | Preserve the stricter provenance record until the metadata conflict is resolved |
| Faster Whisper | `faster-whisper==1.2.1` | MIT | CTranslate2 adapter |
| CTranslate2 | `ctranslate2==4.7.2` | MIT | Core faster-whisper runtime |
| MLX Whisper | `mlx-whisper==0.4.3` | MIT | Apple Silicon adapter |
| WhisperX | `whisperx==3.8.6` | BSD-2-Clause | Alignment model terms can differ by automatically selected language model |
| Parakeet TDT | `nvidia/parakeet-tdt-0.6b-v3` | CC-BY-4.0 | Attribution is required |
| NVIDIA NeMo | `nemo_toolkit[asr]==3.0.0` | Apache-2.0 | Runtime for Parakeet |
| Manga OCR weights/code | `kha-white/manga-ocr-base`, `manga-ocr==0.1.16` | Apache-2.0 | Japanese recognition |
| PaddleOCR detector | `PP-OCRv5_mobile_det`, `paddleocr==3.7.0` | Apache-2.0 | Lightweight fallback detector |
| OpenCV | `opencv-contrib-python==4.10.0.84` | Apache-2.0 | DBNet and temporal connected components |
| NLLB translation | `facebook/nllb-200-distilled-600M` | CC-BY-NC-4.0 | Non-commercial restriction; replace for commercial deployment |
| User DBNet detector | path supplied with `--detector-model` | User supplied | The operator must verify the weight file's terms |
| PyTorch CUDA image | `pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime` | BSD-3-Clause plus bundled NVIDIA component terms | Review image notices before redistribution |

Primary upstream references:

- [OpenAI Whisper](https://github.com/openai/whisper)
- [WhisperX](https://github.com/m-bain/whisperX)
- [NVIDIA Parakeet model card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
- [Manga OCR model card](https://huggingface.co/kha-white/manga-ocr-base)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [NLLB model card](https://huggingface.co/facebook/nllb-200-distilled-600M)

Before release, archive the exact model revisions used by the acceptance run,
review dynamically selected WhisperX alignment-model licenses, preserve required
attribution, and confirm that the intended NLLB use is non-commercial. The
profile translation dictionary can replace NLLB for known cards; a commercial
translation backend must implement the existing local `Translator` protocol.
