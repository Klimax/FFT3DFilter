//
//	FFT3DFilter plugin for Avisynth 2.6 - 3D Frequency Domain filter
//  Functions for CreateThread and class for implementing Kalman filter
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

class KalmanFilter {
private:
	int start_block, blocks, outwidth, outpitch, bh, CPUFlags;
	fftwf_complex *covar_in, *covarProcess_in;
	float kratio2;

	void ApplyKalman_AVX512() noexcept;
	void ApplyKalman_AVX2() noexcept;
	void ApplyKalman_AVX() noexcept;
	void ApplyKalman_SSE4() noexcept;
	void ApplyKalman_SSE2() noexcept;
	void ApplyKalman_SSE() noexcept;
	void ApplyKalman_C() noexcept;

	void ApplyKalmanPattern_AVX512() noexcept;
	void ApplyKalmanPattern_AVX2() noexcept;
	void ApplyKalmanPattern_AVX() noexcept;
	void ApplyKalmanPattern_SSE4() noexcept;
	void ApplyKalmanPattern_SSE2() noexcept;
	void ApplyKalmanPattern_SSE() noexcept;
	void ApplyKalmanPattern_C() noexcept;

public:
	KalmanFilter(fftwf_complex *covar_in, fftwf_complex *covarProcess_in, int start_block,
		int blocks, int outwidth, int outpitch, int bh, int CPUFlags, float kratio2);

	std::function<void(KalmanFilter&)> ApplyKalmanPattern;
	std::function<void(KalmanFilter&)> ApplyKalman;

	fftwf_complex * outcur, *outLast;
	float covarNoiseNormed, *covarNoiseNormed2;
};
