//
//	FFT3DFilter plugin for Avisynth 2.6 - 3D Frequency Domain filter
//  Class for managing filters
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
#include <vector>
#include <thread>
#include "Wiener.h"
#include "WienerPattern.h"
#include "Sharpen.h"
#include "Kalman.h"

typedef float fftwf_complex[2];

class FilterContainer
{
private:
	std::vector<WienerFilter> WienerFilters;
	std::vector<PatternFilter> PatternFilters;
	std::vector<SharpenFilter> SharpenFilters;
	std::vector<KalmanFilter> KalmanFilters;
	std::vector<std::thread> threads;
	std::vector<HANDLE> handles;
	int thread_offset = 0;

public:
	FilterContainer() = default;
	FilterContainer(int howmanyblocks, int ncpu, int CPUFlags, int outwidth, int outpitch, int bh, float degrid, float beta, fftwf_complex* gridsample, float sharpen, float sigmaSquaredSharpenMin, float sigmaSquaredSharpenMax, float *wsharpen, float dehalo, float *wdehalo, float ht2n, fftwf_complex *covar, fftwf_complex *covarProcess, float kratio2);
	~FilterContainer() = default;

	void Init(int howmanyblocks, int ncpu, int CPUFlags, int outwidth, int outpitch, int bh, float degrid, float beta, fftwf_complex* gridsample, float sharpen, float sigmaSquaredSharpenMin, float sigmaSquaredSharpenMax, float *wsharpen, float dehalo, float *wdehalo, float ht2n, fftwf_complex *covar, fftwf_complex *covarProcess, float kratio2);

	void ApplyWiener2D(fftwf_complex *outcur, float sigmaSquaredNoiseNormed);
	void ApplyWiener3D2(fftwf_complex *outcur, fftwf_complex *outprev, float sigmaSquaredNoiseNormed);
	void ApplyWiener3D3(fftwf_complex *out, fftwf_complex *outprev, fftwf_complex *outnext, float sigmaSquaredNoiseNormed);
	void ApplyWiener3D4(fftwf_complex *out, fftwf_complex *outprev2, fftwf_complex *outprev, fftwf_complex *outnext, float sigmaSquaredNoiseNormed);
	void ApplyWiener3D5(fftwf_complex *out, fftwf_complex *outprev2, fftwf_complex *outprev, fftwf_complex *outnext, fftwf_complex *outnext2, float sigmaSquaredNoiseNormed);

	void ApplyPattern2D(fftwf_complex *outcur, float pfactor, float* pattern3d);
	void ApplyPattern3D2(fftwf_complex *outcur, fftwf_complex *outprev, float* pattern3d);
	void ApplyPattern3D3(fftwf_complex *out, fftwf_complex *outprev, fftwf_complex *outnext, float* pattern3d);
	void ApplyPattern3D4(fftwf_complex *out, fftwf_complex *outprev2, fftwf_complex *outprev, fftwf_complex *outnext, float* pattern3d);
	void ApplyPattern3D5(fftwf_complex *out, fftwf_complex *outprev2, fftwf_complex *outprev, fftwf_complex *outnext, fftwf_complex *outnext2, float* pattern3d);

	void Sharpen(fftwf_complex *out);

	void ApplyKalman(fftwf_complex *outcur, fftwf_complex *outLast, float covarNoiseNormed);
	void ApplyKalmanPattern(fftwf_complex *outcur, fftwf_complex *outLast, float *covarNoiseNormed);
};
