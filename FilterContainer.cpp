//
//	FFT3DFilter plugin for Avisynth 2.6 - 3D Frequency Domain filter
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

#include "FilterContainer.h"
#include "windows.h"

FilterContainer::FilterContainer(int howmanyblocks, int ncpu, int CPUFlags, int outwidth, int outpitch, int bh, float degrid, float beta, fftwf_complex* gridsample, float sharpen, float sigmaSquaredSharpenMin, float sigmaSquaredSharpenMax, float *wsharpen, float dehalo, float *wdehalo, float ht2n, fftwf_complex *covar, fftwf_complex *covarProcess, float kratio2) :
	thread_offset(0)
{
	Init(howmanyblocks, ncpu, CPUFlags, outwidth, outpitch, bh, degrid, beta, gridsample, sharpen, sigmaSquaredSharpenMin, sigmaSquaredSharpenMax, wsharpen, dehalo, wdehalo, ht2n, covar, covarProcess, kratio2);
}

void FilterContainer::Init(int howmanyblocks, int ncpu, int CPUFlags, int outwidth, int outpitch, int bh, float degrid, float beta,
	fftwf_complex* gridsample, float sharpen, float sigmaSquaredSharpenMin, float sigmaSquaredSharpenMax, float *wsharpen, float dehalo,
	float *wdehalo, float ht2n, fftwf_complex *covar, fftwf_complex *covarProcess, float kratio2)
{
	const int blocks_per_thread = howmanyblocks / ncpu;
	thread_offset = outpitch * bh*blocks_per_thread;

	int blocks(0);

	for (int i2 = 0; i2 < ncpu; i2++) {
		if (i2 == ncpu - 1) { blocks = howmanyblocks; }
		else { blocks = (i2 + 1) * blocks_per_thread; }

		WienerFilters.emplace_back(i2 * blocks_per_thread, blocks, outwidth, outpitch, bh, CPUFlags, gridsample,
			(beta - 1) / beta, degrid, sharpen, sigmaSquaredSharpenMin, sigmaSquaredSharpenMax, wsharpen,
			dehalo, wdehalo, ht2n);
	}

	for (int i2 = 0; i2 < ncpu; i2++) {
		if (i2 == ncpu - 1) { blocks = howmanyblocks; }
		else { blocks = (i2 + 1) * blocks_per_thread; }

		PatternFilters.emplace_back(i2 * blocks_per_thread, blocks, outwidth, outpitch, bh, CPUFlags, gridsample,
			(beta - 1) / beta, degrid, sharpen, sigmaSquaredSharpenMin, sigmaSquaredSharpenMax, wsharpen,
			dehalo, wdehalo, ht2n);
	}

	for (int i2 = 0; i2 < ncpu; i2++) {
		if (i2 == ncpu - 1) { blocks = howmanyblocks; }
		else { blocks = (i2 + 1) * blocks_per_thread; }

		SharpenFilters.emplace_back(i2 * blocks_per_thread, blocks, CPUFlags, outwidth, outpitch, bh, gridsample,
			sharpen, sigmaSquaredSharpenMin, sigmaSquaredSharpenMax, wsharpen, dehalo, wdehalo, ht2n, degrid);
	}

	for (int i2 = 0; i2 < ncpu; i2++) {
		if (i2 == ncpu - 1) { blocks = howmanyblocks; }
		else { blocks = (i2 + 1) * blocks_per_thread; }

		KalmanFilters.emplace_back(i2 * thread_offset + covar, i2 * thread_offset + covarProcess, i2 * blocks_per_thread,
			blocks, outwidth, outpitch, bh, CPUFlags, kratio2);
	}
}

//Wiener
void FilterContainer::ApplyWiener3D4(fftwf_complex *out, fftwf_complex *outprev2, fftwf_complex *outprev, fftwf_complex *outnext, float sigmaSquaredNoiseNormed)
{
	for (unsigned int i = 0; i < WienerFilters.size(); i++) {
		WienerFilters[i].outcur = i * thread_offset + out;
		WienerFilters[i].outprev2 = i * thread_offset + outprev2;
		WienerFilters[i].outprev = i * thread_offset + outprev;
		WienerFilters[i].outnext = i * thread_offset + outnext;
		WienerFilters[i].sigmaSquaredNoiseNormed = sigmaSquaredNoiseNormed;

		auto thread = std::thread(WienerFilters[i].ApplyWiener3D4, WienerFilters[i]);
		handles.emplace_back(thread.native_handle());
		thread.detach();
	}

	WaitForMultipleObjects(handles.size(), handles.data(), true, INFINITE);

	handles.clear();
}

void FilterContainer::ApplyWiener3D3(fftwf_complex *out, fftwf_complex *outprev, fftwf_complex *outnext, float sigmaSquaredNoiseNormed)
{
	for (unsigned int i = 0; i < WienerFilters.size(); i++) {
		WienerFilters[i].outcur = i * thread_offset + out;
		WienerFilters[i].outprev = i * thread_offset + outprev;
		WienerFilters[i].outnext = i * thread_offset + outnext;
		WienerFilters[i].sigmaSquaredNoiseNormed = sigmaSquaredNoiseNormed;

		auto thread = std::thread(WienerFilters[i].ApplyWiener3D3, WienerFilters[i]);
		handles.emplace_back(thread.native_handle());
		thread.detach();
	}

	WaitForMultipleObjects(handles.size(), handles.data(), true, INFINITE);

	handles.clear();
}

void FilterContainer::ApplyWiener3D2(fftwf_complex *out, fftwf_complex *outprev, float sigmaSquaredNoiseNormed)
{
	for (unsigned int i = 0; i < WienerFilters.size(); i++) {
		WienerFilters[i].outcur = i * thread_offset + out;
		WienerFilters[i].outprev = i * thread_offset + outprev;
		WienerFilters[i].sigmaSquaredNoiseNormed = sigmaSquaredNoiseNormed;

		auto thread = std::thread(WienerFilters[i].ApplyWiener3D2, WienerFilters[i]);
		handles.emplace_back(thread.native_handle());
		thread.detach();
	}

	WaitForMultipleObjects(handles.size(), handles.data(), true, INFINITE);

	handles.clear();
}

void FilterContainer::ApplyWiener2D(fftwf_complex *out, float sigmaSquaredNoiseNormed)
{
	for (unsigned int i = 0; i < WienerFilters.size(); i++) {
		WienerFilters[i].outcur = i * thread_offset + out;
		WienerFilters[i].sigmaSquaredNoiseNormed = sigmaSquaredNoiseNormed;

		auto thread = std::thread(WienerFilters[i].ApplyWiener2D, WienerFilters[i]);
		handles.emplace_back(thread.native_handle());
		thread.detach();
	}

	WaitForMultipleObjects(handles.size(), handles.data(), true, INFINITE);

	handles.clear();
}

void FilterContainer::ApplyWiener3D5(fftwf_complex *out, fftwf_complex *outprev2, fftwf_complex *outprev, fftwf_complex *outnext, fftwf_complex *outnext2, float sigmaSquaredNoiseNormed)
{
	for (unsigned int i = 0; i < WienerFilters.size(); i++) {
		WienerFilters[i].outcur = i * thread_offset + out;
		WienerFilters[i].outprev2 = i * thread_offset + outprev2;
		WienerFilters[i].outprev = i * thread_offset + outprev;
		WienerFilters[i].outnext = i * thread_offset + outnext;
		WienerFilters[i].outnext2 = i * thread_offset + outnext2;
		WienerFilters[i].sigmaSquaredNoiseNormed = sigmaSquaredNoiseNormed;

		auto thread = std::thread(WienerFilters[i].ApplyWiener3D5, WienerFilters[i]);
		handles.emplace_back(thread.native_handle());
		thread.detach();
	}

	WaitForMultipleObjects(handles.size(), handles.data(), true, INFINITE);

	handles.clear();
}

//Patterns
void FilterContainer::ApplyPattern3D4(fftwf_complex *out, fftwf_complex *outprev2, fftwf_complex *outprev, fftwf_complex *outnext, float* pattern3d)
{
	for (unsigned int i = 0; i < PatternFilters.size(); i++) {
		PatternFilters[i].outcur = i * thread_offset + out;
		PatternFilters[i].outprev2 = i * thread_offset + outprev2;
		PatternFilters[i].outprev = i * thread_offset + outprev;
		PatternFilters[i].outnext = i * thread_offset + outnext;
		PatternFilters[i].pattern3d = pattern3d;

		auto thread = std::thread(PatternFilters[i].ApplyPattern3D4, PatternFilters[i]);
		handles.emplace_back(thread.native_handle());
		thread.detach();
	}

	WaitForMultipleObjects(handles.size(), handles.data(), true, INFINITE);

	handles.clear();
}

void FilterContainer::ApplyPattern3D3(fftwf_complex *out, fftwf_complex *outprev, fftwf_complex *outnext, float* pattern3d)
{
	for (unsigned int i = 0; i < PatternFilters.size(); i++) {
		PatternFilters[i].outcur = i * thread_offset + out;
		PatternFilters[i].outprev = i * thread_offset + outprev;
		PatternFilters[i].outnext = i * thread_offset + outnext;
		PatternFilters[i].pattern3d = pattern3d;

		auto thread = std::thread(PatternFilters[i].ApplyPattern3D3, PatternFilters[i]);
		handles.emplace_back(thread.native_handle());
		thread.detach();
	}

	WaitForMultipleObjects(handles.size(), handles.data(), true, INFINITE);

	handles.clear();
}

void FilterContainer::ApplyPattern3D2(fftwf_complex *out, fftwf_complex *outprev, float* pattern3d)
{
	for (unsigned int i = 0; i < PatternFilters.size(); i++) {
		PatternFilters[i].outcur = i * thread_offset + out;
		PatternFilters[i].outprev = i * thread_offset + outprev;
		PatternFilters[i].pattern3d = pattern3d;

		auto thread = std::thread(PatternFilters[i].ApplyPattern3D2, PatternFilters[i]);
		handles.emplace_back(thread.native_handle());
		thread.detach();
	}

	WaitForMultipleObjects(handles.size(), handles.data(), true, INFINITE);

	handles.clear();
}

void FilterContainer::ApplyPattern2D(fftwf_complex *outcur, float pfactor, float* pattern3d)
{
	for (unsigned int i = 0; i < PatternFilters.size(); i++) {
		PatternFilters[i].outcur = i * thread_offset + outcur;
		PatternFilters[i].pfactor = pfactor;
		PatternFilters[i].pattern3d = pattern3d;

		auto thread = std::thread(PatternFilters[i].ApplyPattern2D, PatternFilters[i]);
		handles.emplace_back(thread.native_handle());
		thread.detach();
	}

	WaitForMultipleObjects(handles.size(), handles.data(), true, INFINITE);

	handles.clear();
}

void FilterContainer::ApplyPattern3D5(fftwf_complex *out, fftwf_complex *outprev2, fftwf_complex *outprev, fftwf_complex *outnext, fftwf_complex *outnext2, float* pattern3d)
{
	for (unsigned int i = 0; i < PatternFilters.size(); i++) {
		PatternFilters[i].outcur = i * thread_offset + out;
		PatternFilters[i].outprev2 = i * thread_offset + outprev2;
		PatternFilters[i].outprev = i * thread_offset + outprev;
		PatternFilters[i].outnext = i * thread_offset + outnext;
		PatternFilters[i].outnext2 = i * thread_offset + outnext2;
		PatternFilters[i].pattern3d = pattern3d;

		auto thread = std::thread(PatternFilters[i].ApplyPattern3D5, PatternFilters[i]);
		handles.emplace_back(thread.native_handle());
		thread.detach();
	}

	WaitForMultipleObjects(handles.size(), handles.data(), true, INFINITE);

	handles.clear();
}

void FilterContainer::Sharpen(fftwf_complex *out)
{
	for (unsigned int i = 0; i < SharpenFilters.size(); i++) {
		SharpenFilters[i].outcur = i * thread_offset + out;

		auto thread = std::thread(SharpenFilters[i].Sharpen, SharpenFilters[i]);
		handles.emplace_back(thread.native_handle());
		thread.detach();
	}

	WaitForMultipleObjects(handles.size(), handles.data(), true, INFINITE);

	handles.clear();
}

void FilterContainer::ApplyKalman(fftwf_complex *outcur, fftwf_complex *outLast, float covarNoiseNormed)
{
	for (unsigned int i = 0; i < KalmanFilters.size(); i++) {
		KalmanFilters[i].outcur = i * thread_offset + outcur;
		KalmanFilters[i].outLast = i * thread_offset + outLast;
		KalmanFilters[i].covarNoiseNormed = covarNoiseNormed;

		auto thread = std::thread(KalmanFilters[i].ApplyKalman, KalmanFilters[i]);
		handles.emplace_back(thread.native_handle());
		thread.detach();
	}

	WaitForMultipleObjects(handles.size(), handles.data(), true, INFINITE);

	handles.clear();
}

void FilterContainer::ApplyKalmanPattern(fftwf_complex *outcur, fftwf_complex *outLast, float *covarNoiseNormed)
{
	for (unsigned int i = 0; i < KalmanFilters.size(); i++) {
		KalmanFilters[i].outcur = i * thread_offset + outcur;
		KalmanFilters[i].outLast = i * thread_offset + outLast;
		KalmanFilters[i].covarNoiseNormed2 = covarNoiseNormed;

		auto thread = std::thread(KalmanFilters[i].ApplyKalmanPattern, KalmanFilters[i]);
		handles.emplace_back(thread.native_handle());
		thread.detach();
	}

	WaitForMultipleObjects(handles.size(), handles.data(), true, INFINITE);

	handles.clear();
}