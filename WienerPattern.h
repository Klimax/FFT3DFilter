//
//	FFT3DFilter plugin for Avisynth 2.6 - 3D Frequency Domain filter
//  Functions for CreateThread and class for implementing Wiener pattern filter
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

class PatternFilter {
private:
	int start_block, blocks;
	fftwf_complex *gridsample;
	int outwidth, outpitch, bh;
	float lowlimit, degrid; //lowlimit = (beta - 1) / beta >= 0
	int CPUFlags;
	float sharpen, sigmaSquaredSharpenMin, sigmaSquaredSharpenMax, *wsharpen, dehalo, *wdehalo, ht2n;

	void ApplyPattern2D_AVX512() noexcept;
	void ApplyPattern2D_AVX2() noexcept;
	void ApplyPattern2D_AVX() noexcept;
	void ApplyPattern2D_SSE3() noexcept;
	void ApplyPattern2D_SSE2() noexcept;
#ifndef SSE2BUILD
	void ApplyPattern2D_SSE() noexcept;
	void ApplyPattern2D_C() noexcept;
#endif

	void ApplyPattern3D2_AVX512() noexcept;
	void ApplyPattern3D2_AVX2() noexcept;
	void ApplyPattern3D2_AVX() noexcept;
	void ApplyPattern3D2_SSE3() noexcept;
	void ApplyPattern3D2_SSE2() noexcept;
#ifndef SSE2BUILD
	void ApplyPattern3D2_SSE() noexcept;
	void ApplyPattern3D2_C() noexcept;
#endif

	void ApplyPattern3D3_AVX512() noexcept;
	void ApplyPattern3D3_AVX2() noexcept;
	void ApplyPattern3D3_AVX() noexcept;
	void ApplyPattern3D3_SSE3() noexcept;
	void ApplyPattern3D3_SSE2() noexcept;
#ifndef SSE2BUILD
	void ApplyPattern3D3_SSE() noexcept;
	void ApplyPattern3D3_C() noexcept;
#endif

	void ApplyPattern3D4_AVX512() noexcept;
	void ApplyPattern3D4_AVX2() noexcept;
	void ApplyPattern3D4_AVX() noexcept;
	void ApplyPattern3D4_SSE3() noexcept;
	void ApplyPattern3D4_SSE2() noexcept;
#ifndef SSE2BUILD
	void ApplyPattern3D4_SSE() noexcept;
	void ApplyPattern3D4_C() noexcept;
#endif

	void ApplyPattern3D5_AVX512() noexcept;
	void ApplyPattern3D5_AVX2() noexcept;
	void ApplyPattern3D5_AVX() noexcept;
	void ApplyPattern3D5_SSE3() noexcept;
	void ApplyPattern3D5_SSE2() noexcept;
#ifndef SSE2BUILD
	void ApplyPattern3D5_SSE() noexcept;
	void ApplyPattern3D5_C() noexcept;
#endif

	void ApplyPattern2D_degrid_AVX512() noexcept;
	void ApplyPattern2D_degrid_AVX2() noexcept;
	void ApplyPattern2D_degrid_AVX() noexcept;
	void ApplyPattern2D_degrid_SSE3() noexcept;
	void ApplyPattern2D_degrid_SSE2() noexcept;
#ifndef SSE2BUILD
	void ApplyPattern2D_degrid_SSE() noexcept;
	void ApplyPattern2D_degrid_C() noexcept;
#endif

	void ApplyPattern3D2_degrid_AVX512() noexcept;
	void ApplyPattern3D2_degrid_AVX2() noexcept;
	void ApplyPattern3D2_degrid_AVX() noexcept;
	void ApplyPattern3D2_degrid_SSE3() noexcept;
	void ApplyPattern3D2_degrid_SSE2() noexcept;
#ifndef SSE2BUILD
	void ApplyPattern3D2_degrid_SSE() noexcept;
	void ApplyPattern3D2_degrid_C() noexcept;
#endif

	void ApplyPattern3D3_degrid_AVX512() noexcept;
	void ApplyPattern3D3_degrid_AVX2() noexcept;
	void ApplyPattern3D3_degrid_AVX() noexcept;
	void ApplyPattern3D3_degrid_SSE3() noexcept;
	void ApplyPattern3D3_degrid_SSE2() noexcept;
#ifndef SSE2BUILD
	void ApplyPattern3D3_degrid_SSE() noexcept;
	void ApplyPattern3D3_degrid_C() noexcept;
#endif

	void ApplyPattern3D4_degrid_AVX512() noexcept;
	void ApplyPattern3D4_degrid_AVX2() noexcept;
	void ApplyPattern3D4_degrid_AVX() noexcept;
	void ApplyPattern3D4_degrid_SSE3() noexcept;
	void ApplyPattern3D4_degrid_SSE2() noexcept;
#ifndef SSE2BUILD
	void ApplyPattern3D4_degrid_SSE() noexcept;
	void ApplyPattern3D4_degrid_C() noexcept;
#endif

	void ApplyPattern3D5_degrid_AVX512() noexcept;
	void ApplyPattern3D5_degrid_AVX2() noexcept;
	void ApplyPattern3D5_degrid_AVX() noexcept;
	void ApplyPattern3D5_degrid_SSE3() noexcept;
	void ApplyPattern3D5_degrid_SSE2() noexcept;
#ifndef SSE2BUILD
	void ApplyPattern3D5_degrid_SSE() noexcept;
	void ApplyPattern3D5_degrid_C() noexcept;
#endif

public:
	PatternFilter(int start_block, int blocks, int outwidth, int outpitch, int bh, int CPUFlags, fftwf_complex *gridsample,
		float lowlimit, float degrid, float sharpen, float sigmaSquaredSharpenMin,
		float sigmaSquaredSharpenMax, float *wsharpen, float dehalo, float *wdehalo, float ht2n);

	std::function<void(PatternFilter&)> ApplyPattern3D5;
	std::function<void(PatternFilter&)> ApplyPattern3D4;
	std::function<void(PatternFilter&)> ApplyPattern3D3;
	std::function<void(PatternFilter&)> ApplyPattern3D2;
	std::function<void(PatternFilter&)> ApplyPattern2D;

	fftwf_complex *outcur, *outprev2, *outprev, *outnext, *outnext2;
	float *pattern3d, pfactor;
};
