//
//	FFT3DFilter plugin for Avisynth 2.6 - 3D Frequency Domain filter
//  Functions for CreateThread and initialization of PatternFilter
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

#include "WienerPattern.h"
#include "enums.h"

PatternFilter::PatternFilter(int start_block, int blocks, int outwidth, int outpitch, int bh, int CPUFlags, fftwf_complex *gridsample,
	float lowlimit, float degrid, float sharpen, float sigmaSquaredSharpenMin,
	float sigmaSquaredSharpenMax, float *wsharpen, float dehalo, float *wdehalo, float ht2n) :
	start_block(start_block), blocks(blocks), outwidth(outwidth), outpitch(outpitch), bh(bh), CPUFlags(CPUFlags), gridsample(gridsample),
	lowlimit(lowlimit), degrid(degrid), sharpen(sharpen), sigmaSquaredSharpenMin(sigmaSquaredSharpenMin),
	sigmaSquaredSharpenMax(sigmaSquaredSharpenMax), wsharpen(wsharpen), dehalo(dehalo), wdehalo(wdehalo), ht2n(ht2n),
	outcur(nullptr), outprev2(nullptr), outprev(nullptr), outnext(nullptr), outnext2(nullptr), pattern3d(nullptr),
	ApplyPattern3D5(&PatternFilter::ApplyPattern3D5_SSE2), ApplyPattern3D4(&PatternFilter::ApplyPattern3D4_SSE2),
	ApplyPattern3D3(&PatternFilter::ApplyPattern3D3_SSE2), ApplyPattern3D2(&PatternFilter::ApplyPattern3D2_SSE2),
	ApplyPattern2D(&PatternFilter::ApplyPattern2D_SSE2)
{
	if (degrid == 0.0f)
	{
		if (CPUFlags & CPUK_AVX512)
		{
			ApplyPattern3D5 = &PatternFilter::ApplyPattern3D5_AVX512;
			ApplyPattern3D4 = &PatternFilter::ApplyPattern3D4_AVX512;
			ApplyPattern3D3 = &PatternFilter::ApplyPattern3D3_AVX512;
			ApplyPattern3D2 = &PatternFilter::ApplyPattern3D2_AVX512;
			ApplyPattern2D = &PatternFilter::ApplyPattern2D_AVX512;
		}
		else if (CPUFlags & CPUK_AVX2)
		{
			ApplyPattern3D5 = &PatternFilter::ApplyPattern3D5_AVX2;
			ApplyPattern3D4 = &PatternFilter::ApplyPattern3D4_AVX2;
			ApplyPattern3D3 = &PatternFilter::ApplyPattern3D3_AVX2;
			ApplyPattern3D2 = &PatternFilter::ApplyPattern3D2_AVX2;
			ApplyPattern2D = &PatternFilter::ApplyPattern2D_AVX2;
		}
		else if (CPUFlags & CPUK_AVX)
		{
			ApplyPattern3D5 = &PatternFilter::ApplyPattern3D5_AVX;
			ApplyPattern3D4 = &PatternFilter::ApplyPattern3D4_AVX;
			ApplyPattern3D3 = &PatternFilter::ApplyPattern3D3_AVX;
			ApplyPattern3D2 = &PatternFilter::ApplyPattern3D2_AVX;
			ApplyPattern2D = &PatternFilter::ApplyPattern2D_AVX;
		}
		else if (CPUFlags & CPUK_SSE3)
		{
			ApplyPattern3D5 = &PatternFilter::ApplyPattern3D5_SSE3;
			ApplyPattern3D4 = &PatternFilter::ApplyPattern3D4_SSE3;
			ApplyPattern3D3 = &PatternFilter::ApplyPattern3D3_SSE3;
			ApplyPattern3D2 = &PatternFilter::ApplyPattern3D2_SSE3;
			ApplyPattern2D = &PatternFilter::ApplyPattern2D_SSE3;
		}
		else if (CPUFlags & CPUK_SSE2)
		{
			ApplyPattern3D5 = &PatternFilter::ApplyPattern3D5_SSE2;
			ApplyPattern3D4 = &PatternFilter::ApplyPattern3D4_SSE2;
			ApplyPattern3D3 = &PatternFilter::ApplyPattern3D3_SSE2;
			ApplyPattern3D2 = &PatternFilter::ApplyPattern3D2_SSE2;
			ApplyPattern2D = &PatternFilter::ApplyPattern2D_SSE2;
		}
#ifndef SSE2BUILD
		else if (CPUFlags & CPUK_SSE)
		{
			ApplyPattern3D5 = &PatternFilter::ApplyPattern3D5_SSE;
			ApplyPattern3D4 = &PatternFilter::ApplyPattern3D4_SSE;
			ApplyPattern3D3 = &PatternFilter::ApplyPattern3D3_SSE;
			ApplyPattern3D2 = &PatternFilter::ApplyPattern3D2_SSE;
			ApplyPattern2D = &PatternFilter::ApplyPattern2D_SSE;
		}
		else
		{
			ApplyPattern3D5 = &PatternFilter::ApplyPattern3D5_C;
			ApplyPattern3D4 = &PatternFilter::ApplyPattern3D4_C;
			ApplyPattern3D3 = &PatternFilter::ApplyPattern3D3_C;
			ApplyPattern3D2 = &PatternFilter::ApplyPattern3D2_C;
			ApplyPattern2D = &PatternFilter::ApplyPattern2D_C;
		}
#endif
	}
	else
	{
		if (CPUFlags & CPUK_AVX512)
		{
			ApplyPattern3D5 = &PatternFilter::ApplyPattern3D5_degrid_AVX512;
			ApplyPattern3D4 = &PatternFilter::ApplyPattern3D4_degrid_AVX512;
			ApplyPattern3D3 = &PatternFilter::ApplyPattern3D3_degrid_AVX512;
			ApplyPattern3D2 = &PatternFilter::ApplyPattern3D2_degrid_AVX512;
			ApplyPattern2D = &PatternFilter::ApplyPattern2D_degrid_AVX512;
		}
		else if (CPUFlags & CPUK_AVX2)
		{
			ApplyPattern3D5 = &PatternFilter::ApplyPattern3D5_degrid_AVX2;
			ApplyPattern3D4 = &PatternFilter::ApplyPattern3D4_degrid_AVX2;
			ApplyPattern3D3 = &PatternFilter::ApplyPattern3D3_degrid_AVX2;
			ApplyPattern3D2 = &PatternFilter::ApplyPattern3D2_degrid_AVX2;
			ApplyPattern2D = &PatternFilter::ApplyPattern2D_degrid_AVX2;
		}
		else if (CPUFlags & CPUK_AVX)
		{
			ApplyPattern3D5 = &PatternFilter::ApplyPattern3D5_degrid_AVX;
			ApplyPattern3D4 = &PatternFilter::ApplyPattern3D4_degrid_AVX;
			ApplyPattern3D3 = &PatternFilter::ApplyPattern3D3_degrid_AVX;
			ApplyPattern3D2 = &PatternFilter::ApplyPattern3D2_degrid_AVX;
			ApplyPattern2D = &PatternFilter::ApplyPattern2D_degrid_AVX;
		}
		else if (CPUFlags & CPUK_SSE3)
		{
			ApplyPattern3D5 = &PatternFilter::ApplyPattern3D5_degrid_SSE3;
			ApplyPattern3D4 = &PatternFilter::ApplyPattern3D4_degrid_SSE3;
			ApplyPattern3D3 = &PatternFilter::ApplyPattern3D3_degrid_SSE3;
			ApplyPattern3D2 = &PatternFilter::ApplyPattern3D2_degrid_SSE3;
			ApplyPattern2D = &PatternFilter::ApplyPattern2D_degrid_SSE3;
		}
		else if (CPUFlags & CPUK_SSE2)
		{
			ApplyPattern3D5 = &PatternFilter::ApplyPattern3D5_degrid_SSE2;
			ApplyPattern3D4 = &PatternFilter::ApplyPattern3D4_degrid_SSE2;
			ApplyPattern3D3 = &PatternFilter::ApplyPattern3D3_degrid_SSE2;
			ApplyPattern3D2 = &PatternFilter::ApplyPattern3D2_degrid_SSE2;
			ApplyPattern2D = &PatternFilter::ApplyPattern2D_degrid_SSE2;
		}
#ifndef SSE2BUILD
		else if (CPUFlags & CPUK_SSE)
		{
			ApplyPattern3D5 = &PatternFilter::ApplyPattern3D5_degrid_SSE;
			ApplyPattern3D4 = &PatternFilter::ApplyPattern3D4_degrid_SSE;
			ApplyPattern3D3 = &PatternFilter::ApplyPattern3D3_degrid_SSE;
			ApplyPattern3D2 = &PatternFilter::ApplyPattern3D2_degrid_SSE;
			ApplyPattern2D = &PatternFilter::ApplyPattern2D_degrid_SSE;
		}
		else
		{
			ApplyPattern3D5 = &PatternFilter::ApplyPattern3D5_degrid_C;
			ApplyPattern3D4 = &PatternFilter::ApplyPattern3D4_degrid_C;
			ApplyPattern3D3 = &PatternFilter::ApplyPattern3D3_degrid_C;
			ApplyPattern3D2 = &PatternFilter::ApplyPattern3D2_degrid_C;
			ApplyPattern2D = &PatternFilter::ApplyPattern2D_degrid_C;
		}
#endif
	}
}