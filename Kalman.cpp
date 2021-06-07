//
//	FFT3DFilter plugin for Avisynth 2.6 - 3D Frequency Domain filter
//  Functions for CreateThread and initialization of KalmanFilter
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

#include "Kalman.h"
#include "enums.h"

KalmanFilter::KalmanFilter(fftwf_complex *covar_in, fftwf_complex *covarProcess_in, int start_block,
	int blocks, int outwidth, int outpitch, int bh, int CPUFlags, float kratio2) :
	covar_in(covar_in), covarProcess_in(covarProcess_in), start_block(start_block),
	blocks(blocks), outwidth(outwidth), outpitch(outpitch), bh(bh), CPUFlags(CPUFlags), kratio2(kratio2),
	outcur(nullptr), outLast(nullptr), covarNoiseNormed2(nullptr), covarNoiseNormed(0),
	ApplyKalman(&KalmanFilter::ApplyKalman_SSE2), ApplyKalmanPattern(&KalmanFilter::ApplyKalmanPattern_SSE2)
{
	if (CPUFlags & CPUK_AVX512)
	{
		ApplyKalman = &KalmanFilter::ApplyKalman_AVX512;
		ApplyKalmanPattern = &KalmanFilter::ApplyKalmanPattern_AVX512;
	}
	else if (CPUFlags & CPUK_AVX2)
	{
		ApplyKalman = &KalmanFilter::ApplyKalman_AVX2;
		ApplyKalmanPattern = &KalmanFilter::ApplyKalmanPattern_AVX2;
	}
	else if (CPUFlags & CPUK_AVX)
	{
		ApplyKalman = &KalmanFilter::ApplyKalman_AVX;
		ApplyKalmanPattern = &KalmanFilter::ApplyKalmanPattern_AVX;
	}
	else if (CPUFlags & CPUK_SSE4_1)
	{
		ApplyKalman = &KalmanFilter::ApplyKalman_SSE4;
		ApplyKalmanPattern = &KalmanFilter::ApplyKalmanPattern_SSE4;
	}
	else if (CPUFlags & CPUK_SSE2)
	{
		ApplyKalman = &KalmanFilter::ApplyKalman_SSE2;
		ApplyKalmanPattern = &KalmanFilter::ApplyKalmanPattern_SSE2;
	}
#ifndef SSE2BUILD
	else if (CPUFlags & CPUK_SSE)
	{
		ApplyKalman = &KalmanFilter::ApplyKalman_SSE;
		ApplyKalmanPattern = &KalmanFilter::ApplyKalmanPattern_SSE;
	}
	else
	{
		ApplyKalman = &KalmanFilter::ApplyKalman_C;
		ApplyKalmanPattern = &KalmanFilter::ApplyKalmanPattern_C;
	}
#endif
}