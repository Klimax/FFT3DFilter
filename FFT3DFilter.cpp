/*
	FFT3DFilter plugin for Avisynth 2.6 - 3D Frequency Domain filter

	Copyright(C)2004-2006 A.G.Balakhnin aka Fizick, bag@hotmail.ru, http://avisynth.org.ru
	Copyright(C) 2018 Daniel Klíma aka Klimax

	This program is free software; you can redistribute it and/or modify
	it under the terms of the GNU General Public License version 2 as published by
	the Free Software Foundation.

	This program is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with this program; if not, write to the Free Software
	Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.

	Plugin uses external FFTW library version 3 (http://www.fftw.org)
	as Windows binary DLL (compiled with gcc under MinGW by Alessio Massaro),
	which support for threads and have AMD K7 (3dNow!) support in addition to SSE/SSE2.
	It may be downloaded from ftp://ftp.fftw.org/pub/fftw/fftw3win32mingw.zip
	You must put FFTW3.DLL file from this package to some directory in path
	(for example, C:\WINNT\System32).

	The algorithm is based on the 3D IIR/3D Frequency Domain Filter from:
	MOTION PICTURE RESTORATION. by Anil Christopher Kokaram. Ph.D. Thesis. May 1993.
	http://www.mee.tcd.ie/~ack/papers/a4ackphd.ps.gz

	Version 0.1, 23 November 2004 - initial
	Version 0.2, 3 December 2004 - add beta parameter of noise margin
	Version 0.3, 21 December 2004 - add bt parameter of temporal size
	Version 0.4, 16 January 2005 - algorithm optimized for speed for bt=2 (now default),
		mode bt=3 is temporary disabled, changed default bw=bh=32, filtered region now centered.
	Version 0.5, 28 January 2005 - added YUY2 support
	Version 0.6, 29 January 2005 - added Kalman filter mode for bt=0, ratio parameter
	Version 0.7, 30 January 2005 - re-enabled Wiener filter mode with 3 frames (bt=3)
	Version 0.8, 05 February2005 - added option to sharpen, and bt=-1
	Version 0.8.1, 6 February2005 - skip sharpening of the lowest frequencies to prevent parasitic lines near border
	Version 0.8.2,  February 15, 2005 - added internal buffer to process whole frame (borders included) for any bw, bh (a little slower)
	Version 0.8.3, March 16, 2005 - fixed sharpen mode (bt=-1) for YUY2
	Version 0.8.4, April 3, 2005 - delayed FFTW3.DLL loading
	Version 0.9 - April 3,2005 - variable overlapping size
	Version 0.9.1 - April 7,2005 - some assembler 3DNow! optimization for mode bt=3
	Version 0.9.2 - April 10,2005 - some assembler 3DNow! optimization for mode bt=2,
		option measure=true is now default as more fast
	Version 0.9.3 - April 24,2005 - bug fixed for bt=2 with 3DNow;	bt=3 now default;
	modifyed sharpen to horizontal only (still experimental)
	Version 1.0 - June 22, 2005 - improved edges processing (by padding);
		added svr parameter to control vertical sharpening
	Version 1.0.1 - July 05, 2005 - fixed bug for YUY2 chroma planes
	Version 1.1 - July 8,2005 - improved sharpen mode to prevent grid artifactes and to limit sharpening,
		added parameters smin, smax; renamed parameter ratio to kratio.
	Version 1.2 - July 12, 2005 - changed parameters defaults (bw=bh=48, ow=bw/3, oh=bh/3) to prevent grid artifactes
	Version 1.3 - July 20, 2005 - added interlaced mode
	Version 1.3.1 - July 21, 2005 - fixed bug for YUY2 interlaced
	Version 1.4 - July 23, 2005 - corrected neutral level for chroma processing, added wintype to decrease grid artefactes
	Version 1.5 - July 26, 2005 - added noise pattern method and its parameters pframe, px, py, pshow, pcutoff, pfactor
	Version 1.5.1 - July 29, 2005 - fixed bug with pshow
	Version 1.5.2 - July 31, 2005 - fixed bug with Kalman mode (bt=0) for Athlon (introduced in v1.5)
	Version 1.6 - August 01, 2005 - added mode bt=4; optimized SSE version for bt=2,3
	Version 1.7 - August 29, 2005 - added SSE version for for sharpen and pattern modes bt=2,3 ; restuctured code, GPL v2
	Version 1.8 - September 6, 2005 - improved internal fft cache; added degrid=0; changed wintype=0
	Version 1.8.1 - October 26, 2005 - fixed bug with sharpen>0 AND degrid>0 for bt not equal 1.
	Version 1.8.2 - November 04, 2005 - really set default degrid=1.0 (was = 0)
	Version 1.8.3 - November 28, 2005 - fixed bug with first frame for Kalman YV12 (thanks to Tsp)
	Version 1.8.4 - November 29, 2005 - added multiplane modes plane=3,4
	Version 1.8.5 - 4 December 2005 - fixed bug with memory leakage (thanks to tsp).
	Version 1.9 - April 25, 2006 - added dehalo options; corrected sharpen mode a little;
		re-enabled 3DNow and SSE optimization for degrid=0;  added SSE optimization for bt=3,-1 with degrid>0 (faster by 15%)
	Version 1.9.1 - May 10, 2006 - added SSE optimization for bt=4 with degrid>0 (faster by 30%)
	Version 1.9.2 - September 6, 2006 - added new mode bt=5
	Version 2.0.0 - november 6, 2006 - added motion compensation mc parameter, window reorganized, multi-cpu
	Version 2.1.0 - January 17, 2007 - removed motion compensation mc parameter
	Version 2.1.1 - February 19, 2007 - fixed bug with bw not mod 4 (restored v1.9.2 window method)
	Klimax edition:
	Version 3.0.0 - 2017 - MultiThreading support, revectorised, Avisynth 2.6 header and support, new vectorisations,
					upgraded to MSVC 2017, corrected error message, added support for env variable, measurment code for performance testing, AVX 512 support
					note: ApplyKalmanPattern is broken - unknown casue
					Massive refactoring
	Version 3.0.1 - Limited to C++17, because avisynth.h has some problems under c++20 compilation
        Version 3.1 - 2021 - Changed to VS 2019, threads are created using std::thread and thus trampolines are removed,
                           -sanity checking on multithreading and ncpu parameter versus number of threads of current CPU
*/

#include "fft3dfilter.h"
#include "info.h"
#include <intrin.h>
#include <utility>
#include <vector>
#include <string>

#ifdef MEASURING
#define MEASURMENT(function,...) \
QueryPerformanceCounter(&PerformanceCount);\
function(__VA_ARGS__);\
QueryPerformanceCounter(&PerformanceCount2);\
PerformanceCount.QuadPart *= 1000000;\
PerformanceCount2.QuadPart *= 1000000;\
instrumentation.AddInstance(std::string(#function),PerformanceCount.QuadPart,PerformanceCount2.QuadPart);
#else
#define MEASURMENT(function,...) \
function(__VA_ARGS__);
#endif // MEASURING


#ifdef DEBUGDUMP
#define DUMP()
#else
#define DUMP()

#endif // MEASURING

// The following is the implementation
// of the defined functions.

AVS_Linkage* AVS_linkage = nullptr;

//Here is the acutal constructor code used
FFT3DFilter::FFT3DFilter(PClip _child, float _sigma, float _beta, int _plane, int _bw, int _bh, int _bt, int _ow, int _oh,
	float _kratio, float _sharpen, float _scutoff, float _svr, float _smin, float _smax,
	bool _measure, bool _interlaced, int _wintype,
	int _pframe, int _px, int _py, bool _pshow, float _pcutoff, float _pfactor,
	float _sigma2, float _sigma3, float _sigma4, float _degrid,
	float _dehalo, float _hr, float _ht, int _ncpu, int _multiplane, IScriptEnvironment* env) :

	GenericVideoFilter(_child), sigma(_sigma), beta(_beta), plane(_plane), bw(_bw), bh(_bh), bt(_bt), ow(_ow), oh(_oh),
	kratio(_kratio), sharpen(_sharpen), scutoff(_scutoff), svr(_svr), smin(_smin), smax(_smax),
	measure(_measure), interlaced(_interlaced), wintype(_wintype),
	pframe(_pframe), px(_px), py(_py), pshow(_pshow), pcutoff(_pcutoff), pfactor(_pfactor),
	sigma2(_sigma2), sigma3(_sigma3), sigma4(_sigma4), degrid(_degrid),
	dehalo(_dehalo), hr(_hr), ht(_ht), ncpu(_ncpu), multiplane(_multiplane),
	pattern2d(nullptr), pattern3d(nullptr), wsharpen(nullptr), wdehalo(nullptr), gridsample(nullptr),
	in(nullptr), out(nullptr), outprev(nullptr), outnext(nullptr), outtemp(nullptr), outprev2(nullptr), outnext2(nullptr), outrez(nullptr),
	mean(nullptr), pwin(nullptr), planeBase(0), isPatternSet(false)
{
	// This is the implementation of the constructor.
	// The child clip (source clip) is inherited by the GenericVideoFilter,
	//  where the following variables gets defined:
	//   PClip child;   // Contains the source clip.
	//   VideoInfo vi;  // Contains videoinfo on the source clip.

#ifdef MEASURING
	QueryPerformanceFrequency(&Frequency);
	instrumentation.SetFrequency(Frequency.QuadPart);
#endif

	if (ow * 2 > bw) env->ThrowError("FFT3DFilter: Must not be 2*ow > bw");
	if (oh * 2 > bh) env->ThrowError("FFT3DFilter: Must not be 2*oh > bh");
	if (ow < 0) ow = bw / 3; // changed from bw/4 to bw/3 in v.1.2
	if (oh < 0) oh = bh / 3; // changed from bh/4 to bh/3 in v.1.2

	if (bt < -1 || bt >5) env->ThrowError("FFT3DFilter: bt must be -1(Sharpen), 0(Kalman), 1,2,3,4,5(Wiener)");

	if (vi.IsYV12())
	{
		if (plane == 0)
		{ // Y
			nox = (vi.width - ow + (bw - ow - 1)) / (bw - ow); //removed mirrors (added below) in v.1.2
			noy = (vi.height - oh + (bh - oh - 1)) / (bh - oh);
		}
		else if (plane == 1 || plane == 2) // U,V
		{
			nox = (vi.width / 2 - ow + (bw - ow - 1)) / (bw - ow);
			noy = (vi.height / 2 - oh + (bh - oh - 1)) / (bh - oh);
		}
	}
	else if (vi.IsYUY2())
	{
		if (plane == 0)
		{ // Y
			nox = (vi.width - ow + (bw - ow - 1)) / (bw - ow);
			noy = (vi.height - oh + (bh - oh - 1)) / (bh - oh);
		}
		else if (plane == 1 || plane == 2) // U,V
		{
			nox = (vi.width / 2 - ow + (bw - ow - 1)) / (bw - ow);
			noy = (vi.height - oh + (bh - oh - 1)) / (bh - oh);
		}
		else
			env->ThrowError("FFT3DFilter: internal plane must be 0,1,2");
	}
	else
		env->ThrowError("FFT3DFilter: video must be YV12 or YUY2");


	// padding by 1 block per side
	nox += 2;
	noy += 2;
	mirw = bw - ow; // set mirror size as block interval
	mirh = bh - oh;

	if (beta < 1)
		env->ThrowError("FFT3DFilter: beta must not be less than 1.0");

	int istat(0);

	DetectFeatures(env);

	int malign = 32;
	if (CPUFlags == CPUK_AVX512)
	{
		malign = 64;
	}

	InitFunctors();

#ifdef DEBUGDUMP
	debugdump.SetMaxinstruction(MaxFeatures);
#endif

	hinstLib = LoadLibrary(L"fftw3.dll"); // added in v 0.8.4 for delayed loading
	if (hinstLib != nullptr)
	{
		fftwf_free = (fftwf_free_proc)GetProcAddress(hinstLib, "fftwf_free");
		fftwf_malloc = (fftwf_malloc_proc)GetProcAddress(hinstLib, "fftwf_malloc");
		fftwf_plan_many_dft_r2c = (fftwf_plan_many_dft_r2c_proc)GetProcAddress(hinstLib, "fftwf_plan_many_dft_r2c");
		fftwf_plan_many_dft_c2r = (fftwf_plan_many_dft_c2r_proc)GetProcAddress(hinstLib, "fftwf_plan_many_dft_c2r");
		fftwf_destroy_plan = (fftwf_destroy_plan_proc)GetProcAddress(hinstLib, "fftwf_destroy_plan");
		fftwf_execute_dft_r2c = (fftwf_execute_dft_r2c_proc)GetProcAddress(hinstLib, "fftwf_execute_dft_r2c");
		fftwf_execute_dft_c2r = (fftwf_execute_dft_c2r_proc)GetProcAddress(hinstLib, "fftwf_execute_dft_c2r");
		fftwf_init_threads = (fftwf_init_threads_proc)GetProcAddress(hinstLib, "fftwf_init_threads");
		fftwf_plan_with_nthreads = (fftwf_plan_with_nthreads_proc)GetProcAddress(hinstLib, "fftwf_plan_with_nthreads");
		istat = fftwf_init_threads();
	}
	if (istat == 0 || hinstLib == nullptr || fftwf_free == nullptr || fftwf_malloc == nullptr || fftwf_plan_many_dft_r2c == nullptr ||
		fftwf_plan_many_dft_c2r == nullptr || fftwf_destroy_plan == nullptr || fftwf_execute_dft_r2c == nullptr || fftwf_execute_dft_c2r == nullptr)
		env->ThrowError("FFT3DFilter: Can not load FFTW3.DLL !");


	coverwidth = nox * (bw - ow) + ow;
	coverheight = noy * (bh - oh) + oh;
	coverpitch = ((coverwidth + 7) / 8) * 8;
	coverbuf = (BYTE*)_aligned_malloc(coverheight * coverpitch, malign);

	const int insize = bw * bh * nox * noy;
	in = (float*)_aligned_malloc(sizeof(float) * insize, malign);
	outwidth = bw / 2 + 1; // width (pitch) of complex fft block

	if ((CPUFlags & CPUK_AVX2) || (CPUFlags & CPUK_AVX)) { outpitch = ((outwidth + 1) / 2) * 4; } // must be divisible by 4 (full 256b operations) for AVX // somehow breaks Wintype 0 and 1
	else { outpitch = ((outwidth + 1) / 2) * 2; } // must be even for SSE - v1.7 //Also it is demanded by FFTW to fit full array

	outsize = outpitch * bh * nox * noy; // replace outwidth to outpitch here and below in v1.7

	const auto CurCPU = GetCurrentProcessorNumber();
	SetThreadIdealProcessor(GetCurrentThread(), CurCPU);
	unsigned char CurNode;
	GetNumaProcessorNode(CurCPU, &CurNode);
	//MemoryPages = (unsigned char*)VirtualAllocExNuma(GetCurrentProcess(),nullptr,sizeof(fftwf_complex) * outsize * 14,MEM_COMMIT | MEM_RESERVE,PAGE_EXECUTE_READWRITE,CurNode);

	if (howmanyblocks / ncpu < 2) { ncpu = nox * noy; }
	const int NumCPUs = GetMaximumProcessorCount(ALL_PROCESSOR_GROUPS);
	if (NumCPUs < ncpu) { ncpu = NumCPUs - 1; }

	if (bt == 0) // Kalman
	{
		outLast = (fftwf_complex*)_aligned_malloc(sizeof(fftwf_complex) * outsize, malign);
		covar = (fftwf_complex*)_aligned_malloc(sizeof(fftwf_complex) * outsize, malign);
		covarProcess = (fftwf_complex*)_aligned_malloc(sizeof(fftwf_complex) * outsize, malign);
	}

	outrez = (fftwf_complex*)_aligned_malloc(sizeof(fftwf_complex) * outsize, malign); //v1.8
	gridsample = (fftwf_complex*)_aligned_malloc(sizeof(fftwf_complex) * outsize, malign); //v1.8

	// fft cache - added in v1.8
	cachesize = bt + 2;
	cachewhat = (int*)_aligned_malloc(sizeof(int) * cachesize, 32);
	cachefft = (fftwf_complex * *)_aligned_malloc(sizeof(fftwf_complex*) * cachesize, malign);

	for (int i = 0; i < cachesize; i++)
	{
		cachefft[i] = nullptr;
		cachefft[i] = (fftwf_complex*)_aligned_malloc(sizeof(fftwf_complex) * outsize, malign);
		cachewhat[i] = -1; // init as nonexistant
	}


	int planFlags(FFTW_ESTIMATE);
	// use FFTW_ESTIMATE or FFTW_MEASURE (more optimal plan, but with time calculation at load stage)
#ifdef FFTW_ATOM
	if (measure)
		planFlags = FFTW_MEASURE;
#else
	if (measure)
		planFlags = FFTW_PATIENT | FFTW_DESTROY_INPUT;
#endif
	const int rank = 2; // 2d
	ndim[0] = bh; // size of block along height
	ndim[1] = bw; // size of block along width
	const int istride = 1;
	const int ostride = 1;
	const int idist = bw * bh;
	const int odist = outpitch * bh;//  v1.7 (was outwidth)
	inembed[0] = bh;
	inembed[1] = bw;
	onembed[0] = bh;
	onembed[1] = outpitch;//  v1.7 (was outwidth)
	howmanyblocks = nox * noy;

	fftwf_plan_with_nthreads(ncpu);

	plan = fftwf_plan_many_dft_r2c(rank, ndim, howmanyblocks,
		in, inembed, istride, idist, outrez, onembed, ostride, odist, planFlags);
	if (plan == nullptr)
		env->ThrowError("FFT3DFilter: FFTW plan error");

	planinv = fftwf_plan_many_dft_c2r(rank, ndim, howmanyblocks,
		outrez, onembed, ostride, odist, in, inembed, istride, idist, planFlags);
	if (planinv == nullptr)
		env->ThrowError("FFT3DFilter: FFTW plan error");

	fftwf_plan_with_nthreads(1);

	wanxl = (float*)_aligned_malloc(ow * sizeof(float), malign);
	wanxr = (float*)_aligned_malloc(ow * sizeof(float), malign);
	wanyl = (float*)_aligned_malloc(oh * sizeof(float), malign);
	wanyr = (float*)_aligned_malloc(oh * sizeof(float), malign);

	wsynxl = (float*)_aligned_malloc(ow * sizeof(float), malign);
	wsynxr = (float*)_aligned_malloc(ow * sizeof(float), malign);
	wsynyl = (float*)_aligned_malloc(oh * sizeof(float), malign);
	wsynyr = (float*)_aligned_malloc(oh * sizeof(float), malign);

	wsharpen = (float*)fftwf_malloc(bh * outpitch * sizeof(float));
	wdehalo = (float*)fftwf_malloc(bh * outpitch * sizeof(float));

	GenWindows();

	// init nlast
	nlast = -999; // init as nonexistant
	btcurlast = -999; // init as nonexistant

	norm = 1.0f / (bw * bh); // do not forget set FFT normalization factor

	sigmaSquaredNoiseNormed2D = sigma * sigma / norm;
	sigmaNoiseNormed2D = sigma / sqrtf(norm);
	sigmaMotionNormed = sigma * kratio / sqrtf(norm);
	sigmaSquaredSharpenMinNormed = smin * smin / norm;
	sigmaSquaredSharpenMaxNormed = smax * smax / norm;
	ht2n = ht * ht / norm; // halo threshold squared and normed - v1.9

	// init Kalman
	if (bt == 0) // Kalman
	{
		fill_complex(outLast, outsize, 0, 0);
		fill_complex(covar, outsize, sigmaSquaredNoiseNormed2D, sigmaSquaredNoiseNormed2D); // fixed bug in v.1.1
		fill_complex(covarProcess, outsize, sigmaSquaredNoiseNormed2D, sigmaSquaredNoiseNormed2D);// fixed bug in v.1.1
	}

	mean = (float*)_aligned_malloc(nox * noy * sizeof(float), malign);
	for (int i = 0; i < nox * noy; i++) { mean[i] = 0.0f; }

	pwin = (float*)_aligned_malloc(bh * outpitch * sizeof(float), malign); // pattern window array
	for (int i = 0; i < bh * outpitch; i++) { pwin[i] = 0.0f; }

	float fw2(0.0f), fh2(0.0f);
	for (int j = 0; j < bh; j++)
	{
		if (j < bh / 2)
			fh2 = (j * 2.0f / bh) * (j * 2.0f / bh);
		else
			fh2 = ((bh - 1 - j) * 2.0f / bh) * ((bh - 1 - j) * 2.0f / bh);
		for (int i = 0; i < outwidth; i++)
		{
			fw2 = (i * 2.0f / bw) * (j * 2.0f / bw);
			pwin[i] = (fh2 + fw2) / (fh2 + fw2 + pcutoff * pcutoff);
		}
		pwin += outpitch;
	}
	pwin -= outpitch * bh; // restore pointer

	pattern2d = (float*)_aligned_malloc(bh * outpitch * sizeof(float), malign); // noise pattern window array
	for (int i = 0; i < bh * outpitch; i++) { pattern2d[i] = 0.0f; }

	pattern3d = (float*)_aligned_malloc(bh * outpitch * sizeof(float), malign); // noise pattern window array
	for (int i = 0; i < bh * outpitch; i++) { pattern3d[i] = 0.0f; }

	if ((sigma2 != sigma || sigma3 != sigma || sigma4 != sigma) && pfactor == 0)
	{// we have different sigmas, so create pattern from sigmas
		SigmasToPattern(sigma, sigma2, sigma3, sigma4, bh, outwidth, outpitch, norm, pattern2d);
		isPatternSet = true;
		pfactor = 1;
	} // pattern must be estimated in all other cases

	// prepare  window compensation array gridsample
	// allocate large array for simplicity :)
	// but use one block only for speed
	// Attention: other block could be the same, but we do not calculate them!
	plan1 = fftwf_plan_many_dft_r2c(rank, ndim, 1,
		in, inembed, istride, idist, outrez, onembed, ostride, odist, planFlags); // 1 block

	memset(coverbuf, 255, coverheight * coverpitch);
	InitOverlapPlane(*this, in, coverbuf);
	// make FFT 2D
	fftwf_execute_dft_r2c(plan1, in, gridsample);

	filters.Init(howmanyblocks, ncpu, CPUFlags, outwidth, outpitch, bh, degrid, beta, gridsample,
		sharpen, sigmaSquaredSharpenMinNormed, sigmaSquaredSharpenMaxNormed, wsharpen, dehalo,
		wdehalo, ht2n, covar, covarProcess, kratio * kratio);
}
//-------------------------------------------------------------------------------------------

void FFT3DFilter::InitFunctors()
{
	if (wintype != 2)
	{
		if (CPUFlags & CPUK_AVX512)
			InitOverlapPlane = &FFT3DFilter::InitOverlapPlane_AVX512;
		else if (CPUFlags & CPUK_AVX2)
			InitOverlapPlane = &FFT3DFilter::InitOverlapPlane_AVX2;
		else if (CPUFlags & CPUK_AVX)
			InitOverlapPlane = &FFT3DFilter::InitOverlapPlane_AVX;
		else if (CPUFlags & CPUK_SSE4_1)
			InitOverlapPlane = &FFT3DFilter::InitOverlapPlane_SSE4;
		else if (CPUFlags & CPUK_SSSE3)
			InitOverlapPlane = &FFT3DFilter::InitOverlapPlane_SSSE3;
		else if (CPUFlags & CPUK_SSE2)
			InitOverlapPlane = &FFT3DFilter::InitOverlapPlane_SSE2;
#ifndef SSE2BUILD
		else if (CPUFlags & CPUK_SSE)
			InitOverlapPlane = &FFT3DFilter::InitOverlapPlane_SSE;
		else
			InitOverlapPlane = &FFT3DFilter::InitOverlapPlane_C;
#endif
	}
	else
	{
		if (CPUFlags & CPUK_AVX512)
			InitOverlapPlane = &FFT3DFilter::InitOverlapPlane_wt2_AVX512;
		else if (CPUFlags & CPUK_AVX2)
			InitOverlapPlane = &FFT3DFilter::InitOverlapPlane_wt2_AVX2;
		else if (CPUFlags & CPUK_AVX)
			InitOverlapPlane = &FFT3DFilter::InitOverlapPlane_wt2_AVX;
		else if (CPUFlags & CPUK_SSE4_1)
			InitOverlapPlane = &FFT3DFilter::InitOverlapPlane_wt2_SSE4;
		else if (CPUFlags & CPUK_SSSE3)
			InitOverlapPlane = &FFT3DFilter::InitOverlapPlane_wt2_SSSE3;
		else if (CPUFlags & CPUK_SSE2)
			InitOverlapPlane = &FFT3DFilter::InitOverlapPlane_wt2_SSE2;
#ifndef SSE2BUILD
		else if (CPUFlags & CPUK_SSE)
			InitOverlapPlane = &FFT3DFilter::InitOverlapPlane_wt2_SSE;
		else
			InitOverlapPlane = &FFT3DFilter::InitOverlapPlane_wt2_C;
#endif
	}

	if (CPUFlags & CPUK_AVX512)
		DecodeOverlapPlane = &FFT3DFilter::DecodeOverlapPlane_AVX512;
	else if (CPUFlags & CPUK_AVX2)
		DecodeOverlapPlane = &FFT3DFilter::DecodeOverlapPlane_AVX2;
	else if (CPUFlags & CPUK_AVX)
		DecodeOverlapPlane = &FFT3DFilter::DecodeOverlapPlane_AVX;
	else if (CPUFlags & CPUK_SSE4_1)
		DecodeOverlapPlane = &FFT3DFilter::DecodeOverlapPlane_SSE4;
	else if (CPUFlags & CPUK_SSE2)
		DecodeOverlapPlane = &FFT3DFilter::DecodeOverlapPlane_SSE2;
#ifndef SSE2BUILD
	else if (CPUFlags & CPUK_SSE)
		DecodeOverlapPlane = &FFT3DFilter::DecodeOverlapPlane_SSE;
	else
		DecodeOverlapPlane = &FFT3DFilter::DecodeOverlapPlane_C;
#endif
}

void FFT3DFilter::DetectFeatures(IScriptEnvironment* env)
{
	std::vector<wchar_t> TempBuffer;
	std::wstring TempString;

	CPUFlags = env->GetCPUFlags(); //re-enabled in v.1.9
	int cpuInfo[4];

	__cpuidex(cpuInfo, 7, 0);
	if (cpuInfo[1] & 32768)
	{
		CPUFlags = CPUFlags | CPUF_AVX512;
		CPUFlags = CPUFlags | CPUF_AVX2;
		CPUFlags = CPUFlags | CPUF_AVX;
	}
	else if (cpuInfo[1] & 32)
	{
		CPUFlags = CPUFlags | CPUF_AVX2;
		CPUFlags = CPUFlags | CPUF_AVX;
	}
	else
	{
		__cpuid(cpuInfo, 1);
		if (cpuInfo[2] & 268435456) { CPUFlags = CPUFlags | CPUF_AVX; }
	}

	TempBuffer.resize(32747, 0);
	TempString.clear();
	int NewCPUFlags(0);
	int MaxFeatures(CPUK_AVX512);
	const auto retsize = GetEnvironmentVariable(L"FFT3DFilter", TempBuffer.data(), TempBuffer.size());
	if (retsize > 0 && retsize < 11) {

		TempString.assign(TempBuffer.data(), retsize);
		MaxFeatures = std::stoul(TempString);
	}

	switch (MaxFeatures) 
	{
	case CPUK_AVX512:
		if (CPUFlags & CPUF_AVX512) { NewCPUFlags = NewCPUFlags | CPUK_AVX512; }
	case CPUK_AVX2:
		if (CPUFlags & CPUF_AVX2) { NewCPUFlags = NewCPUFlags | CPUK_AVX2; }
	case CPUK_AVX:
		if (CPUFlags & CPUF_AVX) { NewCPUFlags = NewCPUFlags | CPUK_AVX; }
	case CPUK_SSE4_2:
		if (CPUFlags & CPUF_SSE4) { NewCPUFlags = NewCPUFlags | CPUK_SSE4_2; }
	case CPUK_SSE4_1:
		if (CPUFlags & CPUF_SSE4_1) { NewCPUFlags = NewCPUFlags | CPUK_SSE4_1; }
	case CPUK_SSSE3:
		if (CPUFlags & CPUF_SSSE3) { NewCPUFlags = NewCPUFlags | CPUK_SSSE3; }
	case CPUK_SSE3:
		if (CPUFlags & CPUF_SSE3) { NewCPUFlags = NewCPUFlags | CPUK_SSE3; }
	case CPUK_SSE2:
		if (CPUFlags & CPUF_SSE2) { NewCPUFlags = NewCPUFlags | CPUK_SSE2; }
	case CPUK_3DNOW_EXT:
		if (CPUFlags & CPUF_3DNOW_EXT) { NewCPUFlags = NewCPUFlags | CPUK_3DNOW_EXT; }
	case CPUK_3DNOW:
		if (CPUFlags & CPUF_3DNOW) { NewCPUFlags = NewCPUFlags | CPUK_3DNOW; }
	case CPUK_SSE:
		if (CPUFlags & CPUF_SSE) { NewCPUFlags = NewCPUFlags | CPUK_SSE; }
	case CPUK_MMX:
		if (CPUFlags & CPUF_MMX) { NewCPUFlags = NewCPUFlags | CPUK_MMX; }
		break;
	}
	CPUFlags = NewCPUFlags;

#ifdef MEASURING
	instrumentation.SetMaxinstruction(MaxFeatures);
#endif
}

void FFT3DFilter::GenWindows() noexcept
{
	// define analysis and synthesis windows
	// combining window (analize mult by synthesis) is raised cosine (Hanning)
	constexpr float pi = 3.1415926535897932384626433832795f;
	if (wintype == 0) // window type
	{ // , used in all version up to 1.3
	  // half-cosine, the same for analysis and synthesis
	  // define analysis windows
		for (int i = 0; i < ow; i++)
		{
			wanxl[i] = cosf(pi * (i - ow + 0.5f) / (ow * 2)); // left analize window (half-cosine)
			wanxr[i] = cosf(pi * (i + 0.5f) / (ow * 2)); // right analize window (half-cosine)
		}
		for (int i = 0; i < oh; i++)
		{
			wanyl[i] = cosf(pi * (i - oh + 0.5f) / (oh * 2));
			wanyr[i] = cosf(pi * (i + 0.5f) / (oh * 2));
		}
		// use the same windows for synthesis too.
		for (int i = 0; i < ow; i++)
		{
			wsynxl[i] = wanxl[i]; // left  window (half-cosine)

			wsynxr[i] = wanxr[i]; // right  window (half-cosine)
		}
		for (int i = 0; i < oh; i++)
		{
			wsynyl[i] = wanyl[i];
			wsynyr[i] = wanyr[i];
		}
	}
	else if (wintype == 1) // added in v.1.4
	{
		// define analysis windows as more flat (to decrease grid)
		for (int i = 0; i < ow; i++)
		{
			wanxl[i] = sqrt(cosf(pi * (i - ow + 0.5f) / (ow * 2)));
			wanxr[i] = sqrt(cosf(pi * (i + 0.5f) / (oh * 2)));
		}
		for (int i = 0; i < oh; i++)
		{
			wanyl[i] = sqrt(cosf(pi * (i - oh + 0.5f) / (oh * 2)));
			wanyr[i] = sqrt(cosf(pi * (i + 0.5f) / (oh * 2)));
		}
		// define synthesis as supplenent to rised cosine (Hanning)
		for (int i = 0; i < ow; i++)
		{
			wsynxl[i] = wanxl[i] * wanxl[i] * wanxl[i]; // left window
			wsynxr[i] = wanxr[i] * wanxr[i] * wanxr[i]; // right window
		}
		for (int i = 0; i < oh; i++)
		{
			wsynyl[i] = wanyl[i] * wanyl[i] * wanyl[i];
			wsynyr[i] = wanyr[i] * wanyr[i] * wanyr[i];
		}
	}
	else //  (wintype==2) - added in v.1.4
	{
		// define synthesis as rised cosine (Hanning)
		for (int i = 0; i < ow; i++)
		{
			const float temp = cosf(pi * (i - ow + 0.5f) / (ow * 2));
			wsynxl[i] = temp * temp;// left window (rised cosine)
			const float temp2 = cosf(pi * (i + 0.5f) / (ow * 2));
			wsynxr[i] = temp2 * temp2; // right window (falled cosine)
		}
		for (int i = 0; i < oh; i++)
		{
			const float temp = cosf(pi * (i - oh + 0.5f) / (oh * 2));
			wsynyl[i] = temp * temp;
			const float temp2 = cosf(pi * (i + 0.5f) / (oh * 2));
			wsynyr[i] = temp2 * temp2;
		}
	}

	// window for sharpen
	for (int j = 0; j < bh; j++)
	{
		int dj = j;
		if (j >= bh / 2)
			dj = bh - j;
		const float d2v = float(dj * dj) * (svr * svr) / ((bh / 2) * (bh / 2)); // v1.7
		for (int i = 0; i < outwidth; i++)
		{
			const float d2 = d2v + float(i * i) / ((bw / 2) * (bw / 2)); // distance_2 - v1.7
			wsharpen[i] = 1 - exp(-d2 / (2 * scutoff * scutoff));
		}
		wsharpen += outpitch;
	}
	wsharpen -= outpitch * bh; // restore pointer

							   // window for dehalo - added in v1.9
	float wmax = 0;
	for (int j = 0; j < bh; j++)
	{
		int dj = j;
		if (j >= bh / 2)
			dj = bh - j;
		const float d2v = float(dj * dj) * (svr * svr) / ((bh / 2) * (bh / 2));
		for (int i = 0; i < outwidth; i++)
		{
			const float d2 = d2v + float(i * i) / ((bw / 2) * (bw / 2)); // squared distance in frequency domain
			wdehalo[i] = exp(-0.7f * d2 * hr * hr) - exp(-d2 * hr * hr); // some window with max around 1/hr, small at low and high frequencies
			if (wdehalo[i] > wmax) { wmax = wdehalo[i]; } // for normalization
		}
		wdehalo += outpitch;
	}
	wdehalo -= outpitch * bh; // restore pointer

	for (int j = 0; j < bh; j++)
	{
		for (int i = 0; i < outwidth; i++)
		{
			wdehalo[i] /= wmax; // normalize
		}
		wdehalo += outpitch;
	}
	wdehalo -= outpitch * bh; // restore pointer
}

// This is where any actual destructor code used goes
FFT3DFilter::~FFT3DFilter() {
	// This is where you can deallocate any memory you might have used.
	//VirtualFreeEx(GetCurrentProcess(), MemoryPages, 0, MEM_RELEASE);

	fftwf_destroy_plan(plan);
	fftwf_destroy_plan(plan1);
	fftwf_destroy_plan(planinv);
	_aligned_free(in);
	_aligned_free(wanxl);
	_aligned_free(wanxr);
	_aligned_free(wanyl);
	_aligned_free(wanyr);
	_aligned_free(wsynxl);
	_aligned_free(wsynxr);
	_aligned_free(wsynyl);
	_aligned_free(wsynyr);
	fftwf_free(wsharpen);
	fftwf_free(wdehalo);
	_aligned_free(mean);
	_aligned_free(pwin);
	_aligned_free(pattern2d);
	_aligned_free(pattern3d);
	_aligned_free(outrez);
	if (bt == 0) // Kalman
	{
		_aligned_free(outLast);
		_aligned_free(covar);
		_aligned_free(covarProcess);
	}
	_aligned_free(coverbuf);
	_aligned_free(cachewhat);
	for (int i = 0; i < cachesize; i++)
	{
		_aligned_free(cachefft[i]);
	}
	_aligned_free(cachefft);
	_aligned_free(gridsample); //fixed memory leakage in v1.8.5

	if (hinstLib != nullptr)
		FreeLibrary(hinstLib);

#ifdef MEASURING
	instrumentation.SaveData();
#endif
}

//-------------------------------------------------------------------------------------------
void Copyfft(fftwf_complex* outrez, const fftwf_complex* outprev, int outsize, IScriptEnvironment* env)
{ // save outprev to outrez to prevent cache change (inverse fft2d will destroy the array)
	env->BitBlt((BYTE*)& outrez[0][0], outsize * 8, (BYTE*)& outprev[0][0], outsize * 8, outsize * 8, 1); // faster
}

//-------------------------------------------------------------------------------------------
void SortCache(int* cachewhat, fftwf_complex** cachefft, int cachesize, int cachestart, int cachestartold) noexcept
{
	// sort ordered series, put existant ffts to proper places
	int i(0), ctemp(0);
	fftwf_complex* ffttemp(nullptr);

	int offset = cachestart - cachestartold;
	if (offset > 0) // right
	{
		for (i = 0; i < cachesize; i++)
		{
			if ((i + offset) < cachesize)
			{
				//swap
				ctemp = cachewhat[i + offset];
				cachewhat[i + offset] = cachewhat[i];
				cachewhat[i] = ctemp;
				ffttemp = cachefft[i + offset];
				cachefft[i + offset] = cachefft[i];
				cachefft[i] = ffttemp;
			}
		}
	}
	else if (offset < 0)
	{
		for (i = cachesize - 1; i >= 0; i--)
		{
			if ((i + offset) >= 0)
			{
				ctemp = cachewhat[i + offset];
				cachewhat[i + offset] = cachewhat[i];
				cachewhat[i] = ctemp;
				ffttemp = cachefft[i + offset];
				cachefft[i + offset] = cachefft[i];
				cachefft[i] = ffttemp;
			}
		}
	}
}
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
void CopyFrame(const PVideoFrame& src, PVideoFrame& dst, VideoInfo vi, int planeskip, IScriptEnvironment* env)
{
	const BYTE* srcp(nullptr);
	BYTE* dstp(nullptr);
	int src_height(0), src_width(0), src_pitch(0);
	int dst_height(0), dst_width(0), dst_pitch(0);
	int planeNum(0), plane(0);

	if (vi.IsPlanar()) // copy all planes besides given
	{
		for (plane = 0; plane < 3; plane++)
		{
			if (plane != planeskip)
			{
				planeNum = 1 << plane;

				srcp = src->GetReadPtr(planeNum);
				src_height = src->GetHeight(planeNum);
				src_width = src->GetRowSize(planeNum);
				src_pitch = src->GetPitch(planeNum);
				dstp = dst->GetWritePtr(planeNum);
				dst_height = dst->GetHeight(planeNum);
				dst_width = dst->GetRowSize(planeNum);
				dst_pitch = dst->GetPitch(planeNum);
				env->BitBlt(dstp, dst_pitch, srcp, src_pitch, dst_width, dst_height); // copy one plane
			}
		}
	}
	else if (vi.IsYUY2()) // copy all
	{
		srcp = src->GetReadPtr();
		src_height = src->GetHeight();
		src_width = src->GetRowSize();
		src_pitch = src->GetPitch();
		dstp = dst->GetWritePtr();
		dst_height = dst->GetHeight();
		dst_width = dst->GetRowSize();
		dst_pitch = dst->GetPitch();
		env->BitBlt(dstp, dst_pitch, srcp, src_pitch, dst_width, dst_height); // copy full frame
	}
}
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------

PVideoFrame __stdcall FFT3DFilter::GetFrame(int n, IScriptEnvironment* env) {
	// This is the implementation of the GetFrame function.
	// See the header definition for further info.
#ifdef MEASURING
	instrumentation.FrameInstrumentation(n);
#endif
	PVideoFrame prev2, prev, src, next, psrc, dst, next2;
	int pxf(0), pyf(0);
	int cachecur(0), cachestart(0), cachestartold(0);

	if (plane == 0)
		planeBase = 0;
	else
		planeBase = 128; // neutral chroma value

	if (pfactor != 0 && isPatternSet == false && pshow == false) // get noise pattern
	{
		psrc = child->GetFrame(pframe, env); // get noise pattern frame

		// put source bytes to float array of overlapped blocks
		MEASURMENT(FramePlaneToCoverbuf, plane, psrc, vi, coverbuf, coverwidth, coverheight, coverpitch, mirw, mirh, interlaced, env);
		MEASURMENT(InitOverlapPlane, *this, in, coverbuf);
		// make FFT 2D
		MEASURMENT(fftwf_execute_dft_r2c, plan, in, outrez);
		if (px == 0 && py == 0) // try find pattern block with minimal noise sigma
			MEASURMENT(FindPatternBlock, outrez, outwidth, outpitch, bh, nox, noy, px, py, pwin, degrid, gridsample, CPUFlags);
		MEASURMENT(SetPattern, outrez, outwidth, outpitch, bh, nox, px, py, pwin, pattern2d, psigma, degrid, gridsample);
		isPatternSet = true;
	}
	else if (pfactor != 0 && pshow == true)
	{
		// show noise pattern window
		src = child->GetFrame(n, env); // get noise pattern frame
		dst = env->NewVideoFrame(vi);
		CopyFrame(src, dst, vi, plane, env);

		// put source bytes to float array of overlapped blocks
		FramePlaneToCoverbuf(plane, src, vi, coverbuf, coverwidth, coverheight, coverpitch, mirw, mirh, interlaced, env);
		InitOverlapPlane(*this, in, coverbuf);
		// make FFT 2D
		fftwf_execute_dft_r2c(plan, in, outrez);
		if (px == 0 && py == 0) // try find pattern block with minimal noise sigma
			FindPatternBlock(outrez, outwidth, outpitch, bh, nox, noy, pxf, pyf, pwin, degrid, gridsample, CPUFlags);
		else
		{
			pxf = px; // fixed bug in v1.6
			pyf = py;
		}
		SetPattern(outrez, outwidth, outpitch, bh, nox, pxf, pyf, pwin, pattern2d, psigma, degrid, gridsample);

		// change analysis and synthesis window to constant to show
		for (int i = 0; i < ow; i++)
		{
			wanxl[i] = 1;	wanxr[i] = 1;	wsynxl[i] = 1;	wsynxr[i] = 1;
		}
		for (int i = 0; i < oh; i++)
		{
			wanyl[i] = 1;	wanyr[i] = 1;	wsynyl[i] = 1;	wsynyr[i] = 1;
		}

		planeBase = 128;

		// put source bytes to float array of overlapped blocks
		// cur frame
		FramePlaneToCoverbuf(plane, src, vi, coverbuf, coverwidth, coverheight, coverpitch, mirw, mirh, interlaced, env);
		InitOverlapPlane(*this, in, coverbuf);
		// make FFT 2D
		fftwf_execute_dft_r2c(plan, in, outrez);

		PutPatternOnly(outrez, outwidth, outpitch, bh, nox, noy, pxf, pyf);
		// do inverse 2D FFT, get filtered 'in' array
		fftwf_execute_dft_c2r(planinv, outrez, in);

		// make destination frame plane from current overlaped blocks
		DecodeOverlapPlane(*this, in, coverbuf);
		CoverbufToFramePlane(plane, coverbuf, coverwidth, coverheight, coverpitch, dst, vi, mirw, mirh, interlaced, env);
		const int psigmaint = ((int)(10 * psigma)) / 10;
		const int psigmadec = (int)((psigma - psigmaint) * 10);

		TCHAR* messagebuf = (TCHAR*)malloc(80); //1.8.5;
		wsprintf(messagebuf, L" frame=%d, px=%d, py=%d, sigma=%d.%d", n, pxf, pyf, psigmaint, psigmadec);

		DrawString(dst, 0, 0, messagebuf, vi.IsYUY2());

		free(messagebuf); //v1.8.5

		return dst; // return pattern frame to show
	}

	// Request frame 'n' from the child (source) clip.
	src = child->GetFrame(n, env);
	dst = env->NewVideoFrame(vi);

	if (multiplane < 3 || (multiplane == 3 && plane == 1)) // v1.8.4
	{
		MEASURMENT(CopyFrame, src, dst, vi, plane, env);
	}

	int btcur = bt; // bt used for current frame
	if ((bt / 2 > n) || (bt - 1) / 2 > (vi.num_frames - 1 - n))
	{
		btcur = 1; //	do 2D filter for first and last frames
	}
	// return src //first  frame was not processed prior v.0.7

	if (btcur > 0) // Wiener
	{
		sigmaSquaredNoiseNormed = btcur * sigma * sigma / norm; // normalized variation=sigma^2

		if (btcur != btcurlast)
		{
			MEASURMENT(Pattern2Dto3D_C, pattern2d, bh, outpitch, (float)btcur, pattern3d);
		}

		if (btcur == 1) // 2D
		{
			// cur frame
			MEASURMENT(FramePlaneToCoverbuf, plane, src, vi, coverbuf, coverwidth, coverheight, coverpitch, mirw, mirh, interlaced, env);
			MEASURMENT(InitOverlapPlane, *this, in, coverbuf);
			// make FFT 2D
			MEASURMENT(fftwf_execute_dft_r2c, plan, in, outrez);
			if (pfactor != 0)
			{
				MEASURMENT(filters.ApplyPattern2D, outrez, pfactor, pattern2d);
				MEASURMENT(filters.Sharpen, outrez);
			}
			else
			{
				MEASURMENT(filters.ApplyWiener2D, outrez, sigmaSquaredNoiseNormed);
			}

			// do inverse FFT 2D, get filtered 'in' array
			MEASURMENT(fftwf_execute_dft_c2r, planinv, outrez, in);
		}
		else if (btcur == 2)  // 3D2
		{
			cachecur = 2;
			cachestart = n - cachecur;
			cachestartold = nlast - cachecur;
			MEASURMENT(SortCache, cachewhat, cachefft, cachesize, cachestart, cachestartold);
			// cur frame
			out = cachefft[cachecur];
			if (cachewhat[cachecur] != n)
			{
				MEASURMENT(FramePlaneToCoverbuf, plane, src, vi, coverbuf, coverwidth, coverheight, coverpitch, mirw, mirh, interlaced, env);
				MEASURMENT(InitOverlapPlane, *this, in, coverbuf);
				// make FFT 2D
				MEASURMENT(fftwf_execute_dft_r2c, plan, in, out);
				cachewhat[cachecur] = n;
			}
			// prev frame
			outprev = cachefft[cachecur - 1];
			if (cachewhat[cachecur - 1] != n - 1)
			{
				prev = child->GetFrame(n - 1, env);
				MEASURMENT(FramePlaneToCoverbuf, plane, prev, vi, coverbuf, coverwidth, coverheight, coverpitch, mirw, mirh, interlaced, env);
				// calculate prev
				MEASURMENT(InitOverlapPlane, *this, in, coverbuf);
				// make FFT 2D
				MEASURMENT(fftwf_execute_dft_r2c, plan, in, outprev);
				cachewhat[cachecur - 1] = n - 1;
			}
			if (n != nlast + 1)//(not direct sequential access)
			{
				MEASURMENT(Copyfft, outrez, outprev, outsize, env); // save outprev to outrez to prevent its change in cache
			}
			else
			{
				// swap
				outtemp = outrez;
				outrez = outprev;
				outprev = outtemp;
				cachefft[cachecur - 1] = outtemp;
				cachewhat[cachecur - 1] = -1; // will be destroyed
			}
			if (pfactor != 0)
			{
				MEASURMENT(filters.ApplyPattern3D2, out, outrez, pattern3d);
			}
			else
			{
				MEASURMENT(filters.ApplyWiener3D2, out, outrez, sigmaSquaredNoiseNormed);
			} // get result in outpret
			MEASURMENT(filters.Sharpen, outrez);

			// do inverse FFT 3D, get filtered 'in' array
			// note: input "outrez" array is destroyed by execute algo.
			MEASURMENT(fftwf_execute_dft_c2r, planinv, outrez, in);
		}
		else if (btcur == 3) // 3D3
		{
			cachecur = 2;
			cachestart = n - cachecur;
			cachestartold = nlast - cachecur;
			MEASURMENT(SortCache, cachewhat, cachefft, cachesize, cachestart, cachestartold);
			// cur frame
			out = cachefft[cachecur];
			if (cachewhat[cachecur] != n)
			{
				MEASURMENT(FramePlaneToCoverbuf, plane, src, vi, coverbuf, coverwidth, coverheight, coverpitch, mirw, mirh, interlaced, env);
				MEASURMENT(InitOverlapPlane, *this, in, coverbuf);
				// make FFT 2D
				MEASURMENT(fftwf_execute_dft_r2c, plan, in, out);
				cachewhat[cachecur] = n;
			}
			// prev frame
			outprev = cachefft[cachecur - 1];
			if (cachewhat[cachecur - 1] != n - 1)
			{
				// calculate prev
				prev = child->GetFrame(n - 1, env);
				MEASURMENT(FramePlaneToCoverbuf, plane, prev, vi, coverbuf, coverwidth, coverheight, coverpitch, mirw, mirh, interlaced, env);
				MEASURMENT(InitOverlapPlane, *this, in, coverbuf);
				// make FFT 2D
				MEASURMENT(fftwf_execute_dft_r2c, plan, in, outprev);
				cachewhat[cachecur - 1] = n - 1;
			}
			if (n != nlast + 1)
			{
				MEASURMENT(Copyfft, outrez, outprev, outsize, env); // save outprev to outrez to preventits change in cache
			}
			else
			{
				// swap
				outtemp = outrez;
				outrez = outprev;
				outprev = outtemp;
				cachefft[cachecur - 1] = outtemp;
				cachewhat[cachecur - 1] = -1; // will be destroyed
			}
			// calculate next
			outnext = cachefft[cachecur + 1];
			if (cachewhat[cachecur + 1] != n + 1)
			{
				next = child->GetFrame(n + 1, env);
				MEASURMENT(FramePlaneToCoverbuf, plane, next, vi, coverbuf, coverwidth, coverheight, coverpitch, mirw, mirh, interlaced, env);
				MEASURMENT(InitOverlapPlane, *this, in, coverbuf);
				// make FFT 2D
				MEASURMENT(fftwf_execute_dft_r2c, plan, in, outnext);
				cachewhat[cachecur + 1] = n + 1;
			}
			if (pfactor != 0)
			{
				MEASURMENT(filters.ApplyPattern3D3, out, outrez, outnext, pattern3d);
			}
			else
			{
				MEASURMENT(filters.ApplyWiener3D3, out, outrez, outnext, sigmaSquaredNoiseNormed);
			}
			MEASURMENT(filters.Sharpen, outrez);

			// do inverse FFT 2D, get filtered 'in' array
		// note: input "outrez" array is destroyed by execute algo.
			MEASURMENT(fftwf_execute_dft_c2r, planinv, outrez, in);
		}
		else if (btcur == 4) // 3D4
		{
			// cycle prev2, prev, cur and next
			cachecur = 3;
			cachestart = n - cachecur;
			cachestartold = nlast - cachecur;
			MEASURMENT(SortCache, cachewhat, cachefft, cachesize, cachestart, cachestartold);

			// cur frame
			out = cachefft[cachecur];
			if (cachewhat[cachecur] != n)
			{
				MEASURMENT(FramePlaneToCoverbuf, plane, src, vi, coverbuf, coverwidth, coverheight, coverpitch, mirw, mirh, interlaced, env);

				//QueryPerformanceCounter(&PerformanceCount);
				MEASURMENT(InitOverlapPlane, *this, in, coverbuf);
				// make FFT 2D
				MEASURMENT(fftwf_execute_dft_r2c, plan, in, out);
				cachewhat[cachecur] = n;
			}
			// prev2 frame
			outprev2 = cachefft[cachecur - 2];
			if (cachewhat[cachecur - 2] != n - 2)
			{
				// calculate prev2
				prev2 = child->GetFrame(n - 2, env);
				MEASURMENT(FramePlaneToCoverbuf, plane, prev2, vi, coverbuf, coverwidth, coverheight, coverpitch, mirw, mirh, interlaced, env);
				MEASURMENT(InitOverlapPlane, *this, in, coverbuf);
				// make FFT 2D
				MEASURMENT(fftwf_execute_dft_r2c, plan, in, outprev2);
				cachewhat[cachecur - 2] = n - 2;
			}
			if (n != nlast + 1)
			{
				MEASURMENT(Copyfft, outrez, outprev2, outsize, env); // save outprev2 to outrez to prevent its change in cache
			}
			else
			{
				// swap
				outtemp = outrez;
				outrez = outprev2;
				outprev2 = outtemp;
				cachefft[cachecur - 2] = outtemp;
				cachewhat[cachecur - 2] = -1; // will be destroyed
			}
			// prev frame
			outprev = cachefft[cachecur - 1];
			if (cachewhat[cachecur - 1] != n - 1)
			{
				prev = child->GetFrame(n - 1, env);
				MEASURMENT(FramePlaneToCoverbuf, plane, prev, vi, coverbuf, coverwidth, coverheight, coverpitch, mirw, mirh, interlaced, env);
				MEASURMENT(InitOverlapPlane, *this, in, coverbuf);
				// make FFT 2D
				MEASURMENT(fftwf_execute_dft_r2c, plan, in, outprev);
				cachewhat[cachecur - 1] = n - 1;
			}
			// next frame
			outnext = cachefft[cachecur + 1];
			if (cachewhat[cachecur + 1] != n + 1)
			{
				next = child->GetFrame(n + 1, env);

				MEASURMENT(FramePlaneToCoverbuf, plane, next, vi, coverbuf, coverwidth, coverheight, coverpitch, mirw, mirh, interlaced, env);
				MEASURMENT(InitOverlapPlane, *this, in, coverbuf);

				// make FFT 2D
				MEASURMENT(fftwf_execute_dft_r2c, plan, in, outnext);
				cachewhat[cachecur + 1] = n + 1;
			}
			if (pfactor != 0)
			{
				MEASURMENT(filters.ApplyPattern3D4, out, outrez, outprev, outnext, pattern3d);
			}
			else
			{
				MEASURMENT(filters.ApplyWiener3D4, out, outrez, outprev, outnext, sigmaSquaredNoiseNormed);
			}
			MEASURMENT(filters.Sharpen, outrez);

			// do inverse FFT 2D, get filtered 'in' array
		// note: input "outrez" array is destroyed by execute algo.

			MEASURMENT(fftwf_execute_dft_c2r, planinv, outrez, in);
		}
		else if (btcur == 5) // 3D5
		{
			// cycle prev2, prev, cur, next and next2
			cachecur = 3;
			cachestart = n - cachecur;
			cachestartold = nlast - cachecur;
			MEASURMENT(SortCache, cachewhat, cachefft, cachesize, cachestart, cachestartold);
			// cur frame
			out = cachefft[cachecur];
			if (cachewhat[cachecur] != n)
			{
				MEASURMENT(FramePlaneToCoverbuf, plane, src, vi, coverbuf, coverwidth, coverheight, coverpitch, mirw, mirh, interlaced, env);
				MEASURMENT(InitOverlapPlane, *this, in, coverbuf);
				// make FFT 2D
				MEASURMENT(fftwf_execute_dft_r2c, plan, in, out);
				cachewhat[cachecur] = n;
			}
			// prev2 frame
			outprev2 = cachefft[cachecur - 2];
			if (cachewhat[cachecur - 2] != n - 2)
			{
				// calculate prev2
				prev2 = child->GetFrame(n - 2, env);
				MEASURMENT(FramePlaneToCoverbuf, plane, prev2, vi, coverbuf, coverwidth, coverheight, coverpitch, mirw, mirh, interlaced, env);
				MEASURMENT(InitOverlapPlane, *this, in, coverbuf);
				// make FFT 2D
				MEASURMENT(fftwf_execute_dft_r2c, plan, in, outprev2);
				cachewhat[cachecur - 2] = n - 2;
			}
			if (n != nlast + 1)
			{
				MEASURMENT(Copyfft, outrez, outprev2, outsize, env); // save outprev2 to outrez to prevent its change in cache
			}
			else
			{
				// swap
				outtemp = outrez;
				outrez = outprev2;
				outprev2 = outtemp;
				cachefft[cachecur - 2] = outtemp;
				cachewhat[cachecur - 2] = -1; // will be destroyed
			}
			// prev frame
			outprev = cachefft[cachecur - 1];
			if (cachewhat[cachecur - 1] != n - 1)
			{
				prev = child->GetFrame(n - 1, env);
				MEASURMENT(FramePlaneToCoverbuf, plane, prev, vi, coverbuf, coverwidth, coverheight, coverpitch, mirw, mirh, interlaced, env);
				MEASURMENT(InitOverlapPlane, *this, in, coverbuf);
				// make FFT 2D
				MEASURMENT(fftwf_execute_dft_r2c, plan, in, outprev);
				cachewhat[cachecur - 1] = n - 1;
			}
			// next frame
			outnext = cachefft[cachecur + 1];
			if (cachewhat[cachecur + 1] != n + 1)
			{
				next = child->GetFrame(n + 1, env);
				MEASURMENT(FramePlaneToCoverbuf, plane, next, vi, coverbuf, coverwidth, coverheight, coverpitch, mirw, mirh, interlaced, env);
				MEASURMENT(InitOverlapPlane, *this, in, coverbuf);
				// make FFT 2D
				MEASURMENT(fftwf_execute_dft_r2c, plan, in, outnext);
				cachewhat[cachecur + 1] = n + 1;
			}
			// next2 frame
			outnext2 = cachefft[cachecur + 2];
			if (cachewhat[cachecur + 2] != n + 2)
			{
				next2 = child->GetFrame(n + 2, env);
				MEASURMENT(FramePlaneToCoverbuf, plane, next2, vi, coverbuf, coverwidth, coverheight, coverpitch, mirw, mirh, interlaced, env);
				MEASURMENT(InitOverlapPlane, *this, in, coverbuf);
				// make FFT 2D
				MEASURMENT(fftwf_execute_dft_r2c, plan, in, outnext2);
				cachewhat[cachecur + 2] = n + 2;
			}
			if (pfactor != 0)
			{
				MEASURMENT(filters.ApplyPattern3D5, out, outrez, outprev, outnext, outnext2, pattern3d);
			}
			else
			{
				MEASURMENT(filters.ApplyWiener3D5, out, outrez, outprev, outnext, outnext2, sigmaSquaredNoiseNormed);
			}
			MEASURMENT(filters.Sharpen, outrez);

			// do inverse FFT 2D, get filtered 'in' array
		// note: input "outrez" array is destroyed by execute algo.
			MEASURMENT(fftwf_execute_dft_c2r, planinv, outrez, in);
		}
		// make destination frame plane from current overlaped blocks
		MEASURMENT(DecodeOverlapPlane, *this, in, coverbuf);

		MEASURMENT(CoverbufToFramePlane, plane, coverbuf, coverwidth, coverheight, coverpitch, dst, vi, mirw, mirh, interlaced, env);
	}
	else if (bt == 0) //Kalman filter
	{
		// get power spectral density (abs quadrat) for every block and apply filter

		if (n == 0)
		{
			return src; // first frame  not processed
		}

		// put source bytes to float array of overlapped blocks
		// cur frame
		MEASURMENT(FramePlaneToCoverbuf, plane, src, vi, coverbuf, coverwidth, coverheight, coverpitch, mirw, mirh, interlaced, env);
		MEASURMENT(InitOverlapPlane, *this, in, coverbuf);
		// make FFT 2D
		MEASURMENT(fftwf_execute_dft_r2c, plan, in, outrez);
		if (pfactor != 0)
		{
			MEASURMENT(filters.ApplyKalmanPattern, outrez, outLast, pattern2d);
		}
		else
		{
			MEASURMENT(filters.ApplyKalman, outrez, outLast, sigmaSquaredNoiseNormed2D);
		}

		// copy outLast to outrez
		env->BitBlt((BYTE*)& outrez[0][0], outsize * sizeof(fftwf_complex), (BYTE*)& outLast[0][0], outsize * sizeof(fftwf_complex), outsize * sizeof(fftwf_complex), 1);  //v.0.9.2
		MEASURMENT(filters.Sharpen, outrez);

		// do inverse FFT 2D, get filtered 'in' array
	// note: input "out" array is destroyed by execute algo.
	// that is why we must have its copy in "outLast" array
		MEASURMENT(fftwf_execute_dft_c2r, planinv, outrez, in);
		// make destination frame plane from current overlaped blocks
		MEASURMENT(DecodeOverlapPlane, *this, in, coverbuf);
		MEASURMENT(CoverbufToFramePlane, plane, coverbuf, coverwidth, coverheight, coverpitch, dst, vi, mirw, mirh, interlaced, env);

	}
	else if (bt == -1) /// sharpen only
	{
		//		env->MakeWritable(&src);
				// put source bytes to float array of overlapped blocks
		MEASURMENT(FramePlaneToCoverbuf, plane, src, vi, coverbuf, coverwidth, coverheight, coverpitch, mirw, mirh, interlaced, env);
		MEASURMENT(InitOverlapPlane, *this, in, coverbuf);
		// make FFT 2D
		MEASURMENT(fftwf_execute_dft_r2c, plan, in, outrez);
		MEASURMENT(filters.Sharpen, outrez);

		// do inverse FFT 2D, get filtered 'in' array
		MEASURMENT(fftwf_execute_dft_c2r, planinv, outrez, in);
		// make destination frame plane from current overlaped blocks
		MEASURMENT(DecodeOverlapPlane, *this, in, coverbuf);
		MEASURMENT(CoverbufToFramePlane, plane, coverbuf, coverwidth, coverheight, coverpitch, dst, vi, mirw, mirh, interlaced, env);

	}

	if (btcur == bt)
	{// for normal step
		nlast = n; // set last frame to current
	}
	btcurlast = btcur;

	// As we now are finished processing the image, we return the destination image.
	return dst;
}

//-------------------------------------------------------------------------------------------

// This is the function that created the filter, when the filter has been called.
// This can be used for simple parameter checking, so it is possible to create different filters,
// based on the arguments recieved.
AVSValue __cdecl Create_FFT3DFilter(AVSValue args, void*, IScriptEnvironment* env)
{
	// Calls the constructor with the arguments provided.
	const float sigma1 = (float)args[1].AsFloat(2.0f);

	return new FFT3DFilter(args[0].AsClip(), // the 0th parameter is the source clip
		sigma1, // sigma
		(float)args[2].AsFloat(1.0f), // beta
		args[3].AsInt(0), // plane
		args[4].AsInt(48), // bw -new default in v.1.2
		args[5].AsInt(48), // bh -new default in v.1.2
		args[6].AsInt(3), //  bt (=0 for Kalman mode) // new default=3 in v.0.9.3
		args[7].AsInt(-1), //  ow
		args[8].AsInt(-1), //  oh
		(float)args[9].AsFloat(2.0f), // kratio for Kalman mode
		(float)args[10].AsFloat(0.0f), // sharpen strength
		(float)args[11].AsFloat(0.3f), // sharpen cufoff frequency (relative to max) - v1.7
		(float)args[12].AsFloat(1.0f), // svr - sharpen vertical ratio
		(float)args[13].AsFloat(4.0f), // smin -  minimum limit for sharpen (prevent noise amplifying)
		(float)args[14].AsFloat(20.0f), // smax - maximum limit for sharpen (prevent oversharping)
		args[15].AsBool(true), // measure - switched to true in v.0.9.2
		args[16].AsBool(false), // interlaced - v.1.3
		args[17].AsInt(2), // wintype - v1.4, v1.8
		args[18].AsInt(0), //  pframe
		args[19].AsInt(0), //  px
		args[20].AsInt(0), //  py
		args[21].AsBool(false), //  pshow
		(float)args[22].AsFloat(0.1f), //  pcutoff
		(float)args[23].AsFloat(0.0f), //  pfactor
		(float)args[24].AsFloat(sigma1), // sigma2
		(float)args[25].AsFloat(sigma1), // sigma3
		(float)args[26].AsFloat(sigma1), // sigma4
		(float)args[27].AsFloat(1.0f), // degrid
		(float)args[28].AsFloat(0.0f), // dehalo
		(float)args[29].AsFloat(2.0f), // halo radius
		(float)args[30].AsFloat(50.0), // halo threshold - v 1.9
		args[31].AsInt(1), //  ncpu
		args[32].AsInt(0), //  multiplane
		env);
}
//-------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------
class FFT3DFilterMulti : public GenericVideoFilter {
	// FFT3DFilter defines the name of your filter class.
	// This name is only used internally, and does not affect the name of your filter or similar.
	// This filter extends GenericVideoFilter, which incorporates basic functionality.
	// All functions present in the filter must also be present here.

	PClip filtered;
	PClip YClip, UClip, VClip;
	int multiplane;

public:
	// This defines that these functions are present in your class.
	// These functions must be that same as those actually implemented.
	// Since the functions are "public" they are accessible to other classes.
	// Otherwise they can only be called from functions within the class itself.

	FFT3DFilterMulti(PClip _child, float _sigma, float _beta, int _plane, int _bw, int _bh, int _bt, int _ow, int _oh,
		float _kratio, float _sharpen, float _scutoff, float _svr, float _smin, float _smax,
		bool _measure, bool _interlaced, int _wintype,
		int _pframe, int _px, int _py, bool _pshow, float _pcutoff, float _pfactor,
		float _sigma2, float _sigma3, float _sigma4, float _degrid,
		float _dehalo, float _hr, float _ht, int _ncpu, IScriptEnvironment* env);
	// This is the constructor. It does not return any value, and is always used,
	//  when an instance of the class is created.
	// Since there is no code in this, this is the definition.

	~FFT3DFilterMulti();
	// The is the destructor definition. This is called when the filter is destroyed.


	PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env);
	// This is the function that AviSynth calls to get a given frame.
	// So when this functions gets called, the filter is supposed to return frame n.
};


//-------------------------------------------------------------------

// The following is the implementation
// of the defined functions.

//Here is the acutal constructor code used
FFT3DFilterMulti::FFT3DFilterMulti(PClip _child, float _sigma, float _beta, int _multiplane, int _bw, int _bh, int _bt, int _ow, int _oh,
	float _kratio, float _sharpen, float _scutoff, float _svr, float _smin, float _smax,
	bool _measure, bool _interlaced, int _wintype,
	int _pframe, int _px, int _py, bool _pshow, float _pcutoff, float _pfactor,
	float _sigma2, float _sigma3, float _sigma4, float _degrid,
	float _dehalo, float _hr, float _ht, int _ncpu, IScriptEnvironment* env) :

	GenericVideoFilter(_child) {
	// This is the implementation of the constructor.
	// The child clip (source clip) is inherited by the GenericVideoFilter,
	//  where the following variables gets defined:
	//   PClip child;   // Contains the source clip.
	//   VideoInfo vi;  // Contains videoinfo on the source clip.
	multiplane = _multiplane;

	if (_multiplane == 0 || _multiplane == 1 || _multiplane == 2)
	{
		filtered = new FFT3DFilter(_child, _sigma, _beta, _multiplane, _bw, _bh, _bt, _ow, _oh,
			_kratio, _sharpen, _scutoff, _svr, _smin, _smax,
			_measure, _interlaced, _wintype,
			_pframe, _px, _py, _pshow, _pcutoff, _pfactor,
			_sigma2, _sigma3, _sigma4, _degrid, _dehalo, _hr, _ht, _ncpu, _multiplane, env);
	}
	else if (_multiplane == 3 || _multiplane == 4)
	{

		UClip = new FFT3DFilter(_child, _sigma, _beta, 1, _bw, _bh, _bt, _ow, _oh,
			_kratio, _sharpen, _scutoff, _svr, _smin, _smax,
			_measure, _interlaced, _wintype,
			_pframe, _px, _py, _pshow, _pcutoff, _pfactor,
			_sigma2, _sigma3, _sigma4, _degrid, _dehalo, _hr, _ht, _ncpu, _multiplane, env);

		VClip = new FFT3DFilter(_child, _sigma, _beta, 2, _bw, _bh, _bt, _ow, _oh,
			_kratio, _sharpen, _scutoff, _svr, _smin, _smax,
			_measure, _interlaced, _wintype,
			_pframe, _px, _py, _pshow, _pcutoff, _pfactor,
			_sigma2, _sigma3, _sigma4, _degrid, _dehalo, _hr, _ht, _ncpu, _multiplane, env);

		if (_multiplane == 3)
		{
			YClip = _child;
		}
		else
		{
			YClip = new FFT3DFilter(_child, _sigma, _beta, 0, _bw, _bh, _bt, _ow, _oh,
				_kratio, _sharpen, _scutoff, _svr, _smin, _smax,
				_measure, _interlaced, _wintype,
				_pframe, _px, _py, _pshow, _pcutoff, _pfactor,
				_sigma2, _sigma3, _sigma4, _degrid, _dehalo, _hr, _ht, _ncpu, _multiplane, env);
		}

		// replaced by internal processing in v1.9.2
		//			AVSValue argsUToY[1] = { UClip };
		//			UClip = env->Invoke("UToY", AVSValue(argsUToY,1)).AsClip();
		//			AVSValue argsVToY[1] = { VClip };
		//			VClip = env->Invoke("VToY", AVSValue(argsVToY,1)).AsClip();
		//			AVSValue argsYToUV[3] = { UClip, VClip, YClip };
		//			filtered = env->Invoke("YToUV", AVSValue(argsYToUV,3)).AsClip();
	}
	else
		env->ThrowError("FFT3DFilter: plane must be from 0 to 4!");

}

// This is where any actual destructor code used goes
FFT3DFilterMulti::~FFT3DFilterMulti() {
	// This is where you can deallocate any memory you might have used.
}

PVideoFrame __stdcall FFT3DFilterMulti::GetFrame(int n, IScriptEnvironment* env) {
	// This is the implementation of the GetFrame function.
	// See the header definition for further info.
	PVideoFrame dst;
	if (multiplane < 3)
		dst = filtered->GetFrame(n, env);
	else
	{
		PVideoFrame fY = YClip->GetFrame(n, env);
		PVideoFrame fU = UClip->GetFrame(n, env);
		PVideoFrame fV = VClip->GetFrame(n, env);
		dst = env->NewVideoFrame(vi);
		if (vi.IsPlanar())
		{
			env->BitBlt(dst->GetWritePtr(PLANAR_Y), dst->GetPitch(PLANAR_Y), fY->GetReadPtr(PLANAR_Y),
				fY->GetPitch(PLANAR_Y), fY->GetRowSize(PLANAR_Y), fY->GetHeight(PLANAR_Y));
			env->BitBlt(dst->GetWritePtr(PLANAR_U), dst->GetPitch(PLANAR_U), fU->GetReadPtr(PLANAR_U),
				fU->GetPitch(PLANAR_U), fU->GetRowSize(PLANAR_U), fU->GetHeight(PLANAR_U));
			env->BitBlt(dst->GetWritePtr(PLANAR_V), dst->GetPitch(PLANAR_V), fV->GetReadPtr(PLANAR_V),
				fV->GetPitch(PLANAR_V), fV->GetRowSize(PLANAR_V), fV->GetHeight(PLANAR_V));
		}
		else // YUY2 - not optimal
		{
			const int height = dst->GetHeight();
			const int width = dst->GetRowSize();
			BYTE* pdst = dst->GetWritePtr();
			const BYTE* pY = fY->GetReadPtr();
			const BYTE* pU = fU->GetReadPtr();
			const BYTE* pV = fV->GetReadPtr();
			for (int h = 0; h < height; h++)
			{
				for (int w = 0; w < width; w += 4)
				{
					pdst[w] = pY[w];
					pdst[w + 1] = pU[w + 1];
					pdst[w + 2] = pY[w + 2];
					pdst[w + 3] = pV[w + 3];
				}
				pdst += dst->GetPitch();
				pY += fY->GetPitch();
				pU += fU->GetPitch();
				pV += fV->GetPitch();
			}
		}

	}
	return dst;
}

AVSValue __cdecl Create_FFT3DFilterMulti(AVSValue args, void*, IScriptEnvironment* env)
{
	// Calls the constructor with the arguments provided.
	const float sigma1 = (float)args[1].AsFloat(2.0f);
	return new FFT3DFilterMulti(args[0].AsClip(), // the 0th parameter is the source clip
		sigma1, // sigma
		(float)args[2].AsFloat(1.0f), // beta
		args[3].AsInt(0), // plane
		args[4].AsInt(32), // bw - changed default from 48 to 32 in v.1.9.2
		args[5].AsInt(32), // bh - changed default from 48 to 32 in v.1.9.2
		args[6].AsInt(3), //  bt (=0 for Kalman mode) // new default=3 in v.0.9.3
		args[7].AsInt(-1), //  ow
		args[8].AsInt(-1), //  oh
		(float)args[9].AsFloat(2.0f), // kratio for Kalman mode
		(float)args[10].AsFloat(0.0f), // sharpen strength
		(float)args[11].AsFloat(0.3f), // sharpen cutoff frequency (relative to max) - v1.7
		(float)args[12].AsFloat(1.0f), // svr - sharpen vertical ratio
		(float)args[13].AsFloat(4.0f), // smin -  minimum limit for sharpen (prevent noise amplifying)
		(float)args[14].AsFloat(20.0f), // smax - maximum limit for sharpen (prevent oversharping)
		args[15].AsBool(true), // measure - switched to true in v.0.9.2
		args[16].AsBool(false), // interlaced - v.1.3
		args[17].AsInt(2), // wintype - v1.4, v1.8
		args[18].AsInt(0), //  pframe
		args[19].AsInt(0), //  px
		args[20].AsInt(0), //  py
		args[21].AsBool(false), //  pshow
		(float)args[22].AsFloat(0.1), //  pcutoff
		(float)args[23].AsFloat(0), //  pfactor
		(float)args[24].AsFloat(sigma1), // sigma2
		(float)args[25].AsFloat(sigma1), // sigma3
		(float)args[26].AsFloat(sigma1), // sigma4
		(float)args[27].AsFloat(1.0f), // degrid
		(float)args[28].AsFloat(0.0f), // dehalo - v 1.9
		(float)args[29].AsFloat(2.0f), // halo radius - v 1.9
		(float)args[30].AsFloat(50.0f), // halo threshold - v 1.9
		args[31].AsInt(1), //  ncpu
		env);
}

//-------------------------------------------------------------------------------------------

// The following function is the function that actually registers the filter in AviSynth
// It is called automatically, when the plugin is loaded to see which functions this filter contains.

extern "C" __declspec(dllexport) const char* __stdcall AvisynthPluginInit3(IScriptEnvironment* env, AVS_Linkage* vectors) {
	env->AddFunction("FFT3DFilter", "c[sigma]f[beta]f[plane]i[bw]i[bh]i[bt]i[ow]i[oh]i[kratio]f[sharpen]f[scutoff]f[svr]f[smin]f[smax]f[measure]b[interlaced]b[wintype]i[pframe]i[px]i[py]i[pshow]b[pcutoff]f[pfactor]f[sigma2]f[sigma3]f[sigma4]f[degrid]f[dehalo]f[hr]f[ht]f[ncpu]i", Create_FFT3DFilterMulti, 0);
	// The AddFunction has the following parameters:
	// AddFunction(Filtername , Arguments, Function to call,0);

	// Arguments is a string that defines the types and optional names of the arguments for you filter.
	// c - Video Clip
	// i - Integer number
	// f - Float number
	// s - String
	// b - boolean

	AVS_linkage = vectors;

	return "`FFT3DFilter' FFT3DFilter plugin";

}

