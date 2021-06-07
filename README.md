# FFT3DFilter
Denoising filter for Avisynth 2.6.

FFT3DFilter is 3D Frequency Domain filter - strong denoiser and moderate sharpener.

Technical info:
FFT3DFilter uses Fast Fourier Transform method for image processing in frequency domain.
It is based on some advanced mathematical algorithmes of optimal filtration.
It works not locally, but makes some delocalized (block) processing.
In 3D mode, it results in effect similar to partial motion compensation.
This filter can reduce noise without visible quality loss and artefactes,
even with quite strong settings.
It can greatly improve compression and reduce encoded file size.
