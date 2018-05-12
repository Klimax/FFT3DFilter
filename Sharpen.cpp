//
//	FFT3DFilter plugin for Avisynth 2.6 - 3D Frequency Domain filter
//  Functions for CreateThread and initialization of SharpenFilter
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

#include "Sharpen.h"
#include "enums.h"

DWORD WINAPI Sharpen_MT(LPVOID lpParam)
{
	SharpenFilter* in = (SharpenFilter*)lpParam;

	in->Sharpen(*in);
	return 0;
}

SharpenFilter::SharpenFilter(int start_block, int blocks, int CPUFlags, int outwidth, int outpitch, int bh, fftwf_complex *gridsample,
	float sharpen, float sigmaSquaredSharpenMin, float sigmaSquaredSharpenMax, float *wsharpen, float dehalo, float *wdehalo, float ht2n, float degrid) :
	start_block(start_block), blocks(blocks), CPUFlags(CPUFlags), outwidth(outwidth), outpitch(outpitch), bh(bh), gridsample(gridsample),
	sharpen(sharpen), sigmaSquaredSharpenMin(sigmaSquaredSharpenMin), sigmaSquaredSharpenMax(sigmaSquaredSharpenMax),
	wsharpen(wsharpen), dehalo(dehalo), wdehalo(wdehalo), ht2n(ht2n), degrid(degrid), outcur(nullptr),
	Sharpen(&SharpenFilter::Sharpen_SSE2)
{
	if (degrid == 0.0f)
	{
		if (CPUFlags & CPUK_AVX512)
			Sharpen = &SharpenFilter::Sharpen_AVX512;
		else if (CPUFlags & CPUK_AVX2)
			Sharpen = &SharpenFilter::Sharpen_AVX2;
		else if (CPUFlags & CPUK_AVX)
			Sharpen = &SharpenFilter::Sharpen_AVX;
		else if (CPUFlags & CPUK_SSE3)
			Sharpen = &SharpenFilter::Sharpen_SSE3;
		else if (CPUFlags & CPUK_SSE2)
			Sharpen = &SharpenFilter::Sharpen_SSE2;
#ifndef SSE2BUILD
		else if (CPUFlags & CPUK_SSE)
			Sharpen = &SharpenFilter::Sharpen_SSE;
		else
			Sharpen = &SharpenFilter::Sharpen_C;
#endif
	}
	else
	{
		if (CPUFlags & CPUK_AVX512)
			Sharpen = &SharpenFilter::Sharpen_degrid_AVX512;
		else if (CPUFlags & CPUK_AVX2)
			Sharpen = &SharpenFilter::Sharpen_degrid_AVX2;
		else if (CPUFlags & CPUK_AVX)
			Sharpen = &SharpenFilter::Sharpen_degrid_AVX;
		else if (CPUFlags & CPUK_SSE3)
			Sharpen = &SharpenFilter::Sharpen_degrid_SSE3;
		else if (CPUFlags & CPUK_SSE2)
			Sharpen = &SharpenFilter::Sharpen_degrid_SSE2;
#ifndef SSE2BUILD
		else if (CPUFlags & CPUK_SSE)
			Sharpen = &SharpenFilter::Sharpen_degrid_SSE;
		else
			Sharpen = &SharpenFilter::Sharpen_degrid_C;
#endif
	}
}