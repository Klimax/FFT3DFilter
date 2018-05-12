//
//	FFT3DFilter plugin for Avisynth 2.6 - 3D Frequency Domain filter
//  Functions for CreateThread and initialization of WienerFilter
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

#include "Wiener.h"
#include "enums.h"

DWORD WINAPI ApplyWiener3D5_MT(LPVOID lpParam)
{
	WienerFilter* in = (WienerFilter*)lpParam;

	in->ApplyWiener3D5(*in);
	return 0;
}

DWORD WINAPI ApplyWiener3D4_MT(LPVOID lpParam)
{
	WienerFilter* in = (WienerFilter*)lpParam;

	in->ApplyWiener3D4(*in);
	return 0;
}

DWORD WINAPI ApplyWiener3D3_MT(LPVOID lpParam)
{
	WienerFilter* in = (WienerFilter*)lpParam;

	in->ApplyWiener3D3(*in);
	return 0;
}

DWORD WINAPI ApplyWiener3D2_MT(LPVOID lpParam)
{
	WienerFilter* in = (WienerFilter*)lpParam;

	in->ApplyWiener3D2(*in);
	return 0;
}

DWORD WINAPI ApplyWiener2D_MT(LPVOID lpParam)
{
	WienerFilter* in = (WienerFilter*)lpParam;

	in->ApplyWiener2D(*in);
	return 0;
}

WienerFilter::WienerFilter(int start_block, int blocks, int outwidth, int outpitch, int bh, int CPUFlags, fftwf_complex *gridsample,
	float lowlimit, float degrid, float sharpen, float sigmaSquaredSharpenMin,
	float sigmaSquaredSharpenMax, float *wsharpen, float dehalo, float *wdehalo, float ht2n) :
	start_block(start_block), blocks(blocks), outwidth(outwidth), outpitch(outpitch), bh(bh), CPUFlags(CPUFlags), gridsample(gridsample),
	lowlimit(lowlimit), degrid(degrid), sharpen(sharpen), sigmaSquaredSharpenMin(sigmaSquaredSharpenMin),
	sigmaSquaredSharpenMax(sigmaSquaredSharpenMax), wsharpen(wsharpen), dehalo(dehalo), wdehalo(wdehalo), ht2n(ht2n),
	outcur(nullptr), outprev2(nullptr), outprev(nullptr), outnext(nullptr), outnext2(nullptr),
	ApplyWiener3D5(&WienerFilter::ApplyWiener3D5_SSE2), ApplyWiener3D4(&WienerFilter::ApplyWiener3D4_SSE2),
	ApplyWiener3D3(&WienerFilter::ApplyWiener3D3_SSE2), ApplyWiener3D2(&WienerFilter::ApplyWiener3D2_SSE2),
	ApplyWiener2D(&WienerFilter::ApplyWiener2D_SSE2)
{
	if (degrid == 0.0f)
	{
		if (CPUFlags & CPUK_AVX512)
		{
			ApplyWiener3D5 = &WienerFilter::ApplyWiener3D5_AVX512;
			ApplyWiener3D4 = &WienerFilter::ApplyWiener3D4_AVX512;
			ApplyWiener3D3 = &WienerFilter::ApplyWiener3D3_AVX512;
			ApplyWiener3D2 = &WienerFilter::ApplyWiener3D2_AVX512;
			ApplyWiener2D = &WienerFilter::ApplyWiener2D_AVX512;
		}
		else if (CPUFlags & CPUK_AVX2)
		{
			ApplyWiener3D5 = &WienerFilter::ApplyWiener3D5_AVX2;
			ApplyWiener3D4 = &WienerFilter::ApplyWiener3D4_AVX2;
			ApplyWiener3D3 = &WienerFilter::ApplyWiener3D3_AVX2;
			ApplyWiener3D2 = &WienerFilter::ApplyWiener3D2_AVX2;
			ApplyWiener2D = &WienerFilter::ApplyWiener2D_AVX2;
		}
		else if (CPUFlags & CPUK_AVX)
		{
			ApplyWiener3D5 = &WienerFilter::ApplyWiener3D5_AVX;
			ApplyWiener3D4 = &WienerFilter::ApplyWiener3D4_AVX;
			ApplyWiener3D3 = &WienerFilter::ApplyWiener3D3_AVX;
			ApplyWiener3D2 = &WienerFilter::ApplyWiener3D2_AVX;
			ApplyWiener2D = &WienerFilter::ApplyWiener2D_AVX;
		}
		else if (CPUFlags & CPUK_SSE3)
		{
			ApplyWiener3D5 = &WienerFilter::ApplyWiener3D5_SSE3;
			ApplyWiener3D4 = &WienerFilter::ApplyWiener3D4_SSE3;
			ApplyWiener3D3 = &WienerFilter::ApplyWiener3D3_SSE3;
			ApplyWiener3D2 = &WienerFilter::ApplyWiener3D2_SSE3;
			ApplyWiener2D = &WienerFilter::ApplyWiener2D_SSE3;
		}
		else if (CPUFlags & CPUK_SSE2)
		{
			ApplyWiener3D5 = &WienerFilter::ApplyWiener3D5_SSE2;
			ApplyWiener3D4 = &WienerFilter::ApplyWiener3D4_SSE2;
			ApplyWiener3D3 = &WienerFilter::ApplyWiener3D3_SSE2;
			ApplyWiener3D2 = &WienerFilter::ApplyWiener3D2_SSE2;
			ApplyWiener2D = &WienerFilter::ApplyWiener2D_SSE2;
		}
#ifndef SSE2BUILD
		else if (CPUFlags & CPUK_SSE)
		{
			ApplyWiener3D5 = &WienerFilter::ApplyWiener3D5_SSE;
			ApplyWiener3D4 = &WienerFilter::ApplyWiener3D4_SSE;
			ApplyWiener3D3 = &WienerFilter::ApplyWiener3D3_SSE;
			ApplyWiener3D2 = &WienerFilter::ApplyWiener3D2_SSE;
			ApplyWiener2D = &WienerFilter::ApplyWiener2D_SSE;
		}
		else
		{
			ApplyWiener3D5 = &WienerFilter::ApplyWiener3D5_C;
			ApplyWiener3D4 = &WienerFilter::ApplyWiener3D4_C;
			ApplyWiener3D3 = &WienerFilter::ApplyWiener3D3_C;
			ApplyWiener3D2 = &WienerFilter::ApplyWiener3D2_C;
			ApplyWiener2D = &WienerFilter::ApplyWiener2D_C;
		}
#endif
	}
	else
	{
		if (CPUFlags & CPUK_AVX512)
		{
			ApplyWiener3D5 = &WienerFilter::ApplyWiener3D5_degrid_AVX512;
			ApplyWiener3D4 = &WienerFilter::ApplyWiener3D4_degrid_AVX512;
			ApplyWiener3D3 = &WienerFilter::ApplyWiener3D3_degrid_AVX512;
			ApplyWiener3D2 = &WienerFilter::ApplyWiener3D2_degrid_AVX512;
			ApplyWiener2D = &WienerFilter::ApplyWiener2D_degrid_AVX512;
		}
		else if (CPUFlags & CPUK_AVX2)
		{
			ApplyWiener3D5 = &WienerFilter::ApplyWiener3D5_degrid_AVX2;
			ApplyWiener3D4 = &WienerFilter::ApplyWiener3D4_degrid_AVX2;
			ApplyWiener3D3 = &WienerFilter::ApplyWiener3D3_degrid_AVX2;
			ApplyWiener3D2 = &WienerFilter::ApplyWiener3D2_degrid_AVX2;
			ApplyWiener2D = &WienerFilter::ApplyWiener2D_degrid_AVX2;
		}
		else if (CPUFlags & CPUK_AVX)
		{
			ApplyWiener3D5 = &WienerFilter::ApplyWiener3D5_degrid_AVX;
			ApplyWiener3D4 = &WienerFilter::ApplyWiener3D4_degrid_AVX;
			ApplyWiener3D3 = &WienerFilter::ApplyWiener3D3_degrid_AVX;
			ApplyWiener3D2 = &WienerFilter::ApplyWiener3D2_degrid_AVX;
			ApplyWiener2D = &WienerFilter::ApplyWiener2D_degrid_AVX;
		}
		else if (CPUFlags & CPUK_SSE3)
		{
			ApplyWiener3D5 = &WienerFilter::ApplyWiener3D5_degrid_SSE3;
			ApplyWiener3D4 = &WienerFilter::ApplyWiener3D4_degrid_SSE3;
			ApplyWiener3D3 = &WienerFilter::ApplyWiener3D3_degrid_SSE3;
			ApplyWiener3D2 = &WienerFilter::ApplyWiener3D2_degrid_SSE3;
			ApplyWiener2D = &WienerFilter::ApplyWiener2D_degrid_SSE3;
		}
		else if (CPUFlags & CPUK_SSE2)
		{
			ApplyWiener3D5 = &WienerFilter::ApplyWiener3D5_degrid_SSE2;
			ApplyWiener3D4 = &WienerFilter::ApplyWiener3D4_degrid_SSE2;
			ApplyWiener3D3 = &WienerFilter::ApplyWiener3D3_degrid_SSE2;
			ApplyWiener3D2 = &WienerFilter::ApplyWiener3D2_degrid_SSE2;
			ApplyWiener2D = &WienerFilter::ApplyWiener2D_degrid_SSE2;
		}
#ifndef SSE2BUILD
		else if (CPUFlags & CPUK_SSE)
		{
			ApplyWiener3D5 = &WienerFilter::ApplyWiener3D5_degrid_SSE;
			ApplyWiener3D4 = &WienerFilter::ApplyWiener3D4_degrid_SSE;
			ApplyWiener3D3 = &WienerFilter::ApplyWiener3D3_degrid_SSE;
			ApplyWiener3D2 = &WienerFilter::ApplyWiener3D2_degrid_SSE;
			ApplyWiener2D = &WienerFilter::ApplyWiener2D_degrid_SSE;
		}
		else
		{
			ApplyWiener3D5 = &WienerFilter::ApplyWiener3D5_degrid_C;
			ApplyWiener3D4 = &WienerFilter::ApplyWiener3D4_degrid_C;
			ApplyWiener3D3 = &WienerFilter::ApplyWiener3D3_degrid_C;
			ApplyWiener3D2 = &WienerFilter::ApplyWiener3D2_degrid_C;
			ApplyWiener2D = &WienerFilter::ApplyWiener2D_degrid_C;
		}
#endif
	}
}