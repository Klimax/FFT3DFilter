//
//	FFT3DFilter plugin for Avisynth 2.6 - 3D Frequency Domain filter
//  Functions for CreateThread and class for implementing Sharpen and Dehalo filter
//
//  Copyright(C) 2018 Daniel Klíma aka Klimax
//
//	This program is free software; you can redistribute it and/or modify
//	it under the terms of the GNU General Public License version 2 as published by
//	the Free Software Foundation.
//
//	This program is distributed in the hope that it will be useful,
//	but WITHOUT ANY WARRANTY; without even the implied warranty of
//	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//	GNU General Public License for more details.
//
//	You should have received a copy of the GNU General Public License
//	along with this program; if not, write to the Free Software
//	Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
//
//-----------------------------------------------------------------------------------------
//

#pragma once
#include "windows.h"
#include <functional>

typedef float fftwf_complex[2];

class SharpenFilter {
private:
	int start_block, blocks, CPUFlags, outwidth, outpitch, bh;
	fftwf_complex *gridsample;
	float sharpen, sigmaSquaredSharpenMin, sigmaSquaredSharpenMax, *wsharpen, dehalo, *wdehalo, ht2n, degrid;

	void Sharpen_AVX512() noexcept;
	void Sharpen_AVX2() noexcept;
	void Sharpen_AVX() noexcept;
	void Sharpen_SSE3() noexcept;
	void Sharpen_SSE2() noexcept;
	void Sharpen_SSE() noexcept;
	void Sharpen_C() noexcept;

	void Sharpen_degrid_AVX512() noexcept;
	void Sharpen_degrid_AVX2() noexcept;
	void Sharpen_degrid_AVX() noexcept;
	void Sharpen_degrid_SSE3() noexcept;
	void Sharpen_degrid_SSE2() noexcept;
	void Sharpen_degrid_SSE() noexcept;
	void Sharpen_degrid_C() noexcept;

public:
	SharpenFilter::SharpenFilter(int start_block, int blocks, int CPUFlags, int outwidth, int outpitch, int bh, fftwf_complex *gridsample,
		float sharpen, float sigmaSquaredSharpenMin, float sigmaSquaredSharpenMax, float *wsharpen, float dehalo, float *wdehalo, float ht2n, float degrid);

	std::function<void(class SharpenFilter&)> Sharpen;

	fftwf_complex *outcur;
};
