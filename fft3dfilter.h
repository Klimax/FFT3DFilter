#include "windows.h"
#include "avisynth.h"
//#include "fftw3.h" // replaced by fftwlite.h for dynamic load
#include "fftwlite.h" // added in v.0.8.4
#ifdef MEASURING
#include "Instrumentation.h"
#endif
#include "enums.h"
#include "FilterContainer.h"
#ifdef DEBUGDUMP
#include "DebugDump.h"
#endif

#include <unordered_set>

//Conversion functions
void PlanarPlaneToCovebuf(const BYTE *srcp, int src_width, int src_height, int src_pitch, BYTE *coverbuf, int coverwidth, int coverheight, int coverpitch, int mirw, int mirh, bool interlaced, IScriptEnvironment* env) noexcept;
void CoverbufToPlanarPlane(const BYTE *coverbuf, int coverpitch, BYTE *dstp, int dst_width, int dst_height, int dst_pitch, int mirw, int mirh, bool interlaced, IScriptEnvironment* env) noexcept;
void YUY2PlaneToCoverbuf(int plane, const BYTE *srcp, int src_width, int src_height, int src_pitch, BYTE *coverbuf, int coverwidth, int coverheight, int coverpitch, int mirw, int mirh, bool interlaced) noexcept;
void CoverbufToYUY2Plane(int plane, const BYTE *coverbuf, int coverwidth, BYTE *dstp, int dst_width, int dst_height, int dst_pitch, int mirw, int mirh, bool interlaced) noexcept;
void FramePlaneToCoverbuf(int plane, const PVideoFrame &src, VideoInfo vi1, BYTE *coverbuf, int coverwidth, int coverheight, int coverpitch, int mirw, int mirh, bool interlaced, IScriptEnvironment* env) noexcept;
void CoverbufToFramePlane(int plane, const BYTE *coverbuf, int coverwidth, int coverheight, int coverpitch, const PVideoFrame &dst, VideoInfo vi1, int mirw, int mirh, bool interlaced, IScriptEnvironment* env) noexcept;

//Paterns
void fill_complex(fftwf_complex *plane, int outsize, float realvalue, float imgvalue) noexcept;
void SigmasToPattern(float sigma, float sigma2, float sigma3, float sigma4, int bh, int outwidth, int outpitch, float norm, float *pattern2d) noexcept;

void FindPatternBlock(fftwf_complex *outcur0, int outwidth, int outpitch, int bh, int nox, int noy, int &px, int &py, float *pwin, float degrid, fftwf_complex *gridsample, int CPUFlags) noexcept;
void SetPattern(fftwf_complex *outcur, int outwidth, int outpitch, int bh, int nox, int px, int py, float *pwin, float *pattern2d, float &psigma, float degrid, fftwf_complex *gridsample) noexcept;

void PutPatternOnly(fftwf_complex *outcur, int outwidth, int outpitch, int bh, int nox, int noy, int px, int py) noexcept;

void Pattern2Dto3D_C(const float *pattern2d, int bh, int outpitch, float mult, float *pattern3d) noexcept;

//-------------------------------------------------------------------------------------------
class FFT3DFilter : public GenericVideoFilter {
	// FFT3DFilter defines the name of your filter class.
	// This name is only used internally, and does not affect the name of your filter or similar.
	// This filter extends GenericVideoFilter, which incorporates basic functionality.
	// All functions present in the filter must also be present here.

#ifdef MEASURING
	LARGE_INTEGER Frequency;
	Instrumentation instrumentation;
	LARGE_INTEGER PerformanceCount, PerformanceCount2;
#endif
#ifdef DEBUGDUMP
	DebugDump debugdump;
#endif
	FilterContainer filters;

	//  parameters
	float sigma; // noise level (std deviation) for high frequncies
	float beta; // relative noise margin for Wiener filter
	int plane; // color plane
	int bw;// block width
	int bh;// block height
	int bt;// block size  along time (mumber of frames), =0 for Kalman, >0 for Wiener
	int ow; // overlap width - v.0.9
	int oh; // overlap height - v.0.9
	float kratio; // threshold to sigma ratio for Kalman filter
	float sharpen; // sharpen factor (0 to 1 and above)
	float scutoff; // sharpen cufoff frequency (relative to max) - v1.7
	float svr; // sharpen vertical ratio (0 to 1 and above) - v.1.0
	float smin; // minimum limit for sharpen (prevent noise amplifying) - v.1.1
	float smax; // maximum limit for sharpen (prevent oversharping) - v.1.1
	bool measure; // fft optimal method
	bool interlaced;
	int wintype; // window type
	int pframe; // noise pattern frame number
	int px; // noise pattern window x-position
	int py; // noise pattern window y-position
	bool pshow; // show noise pattern
	float pcutoff; // pattern cutoff frequency (relative to max)
	float pfactor; // noise pattern denoise strength
	float sigma2; // noise level for middle frequencies
	float sigma3; // noise level for low frequencies
	float sigma4; // noise level for lowest (zero) frequencies
	float degrid; // decrease grid
	float dehalo; // remove halo strength - v.1.9
	float hr; // halo radius - v1.9
	float ht; // halo threshold - v1.9
	int ncpu; // number of threads - v2.0

	int multiplane; // multiplane value

	// additional parameterss
	float *in;
	unsigned char* MemoryPages;
	fftwf_complex *out, *outprev, *outnext, *outtemp, *outprev2, *outnext2;
	fftwf_complex *outrez, *gridsample; //v1.8
	fftwf_plan plan, planinv, plan1;
	int nox, noy;
	int outwidth;
	int outpitch; //v.1.7

	int outsize;
	int howmanyblocks;

	int ndim[2];
	int inembed[2];
	int onembed[2];

	float *wanxl; // analysis
	float *wanxr;
	float *wanyl;
	float *wanyr;

	float *wsynxl; // synthesis
	float *wsynxr;
	float *wsynyl;
	float *wsynyr;

	float *wsharpen;
	float *wdehalo;

	int nlast;// frame number at last step
	int btcurlast;  //v1.7

	fftwf_complex *outLast, *covar, *covarProcess;
	float sigmaSquaredNoiseNormed;
	float sigmaSquaredNoiseNormed2D;
	float sigmaNoiseNormed2D;
	float sigmaMotionNormed;
	float sigmaSquaredSharpenMinNormed;
	float sigmaSquaredSharpenMaxNormed;
	float ht2n; // halo threshold squared normed
	float norm; // normalization factor

	BYTE *coverbuf; //  block buffer covering the frame without remainders (with sufficient width and heigth)
	int coverwidth;
	int coverheight;
	int coverpitch;

	int mirw; // mirror width for padding
	int mirh; // mirror height for padding

	int planeBase; // color base value (0 for luma, 128 for chroma)

	float *mean;

	float *pwin;
	float *pattern2d;
	float *pattern3d;
	bool isPatternSet;
	float psigma;

	// added in v.0.9 for delayed FFTW3.DLL loading
	HINSTANCE hinstLib;
	fftwf_malloc_proc fftwf_malloc;
	fftwf_free_proc fftwf_free;
	fftwf_plan_many_dft_r2c_proc fftwf_plan_many_dft_r2c;
	fftwf_plan_many_dft_c2r_proc fftwf_plan_many_dft_c2r;
	fftwf_destroy_plan_proc fftwf_destroy_plan;
	fftwf_execute_dft_r2c_proc fftwf_execute_dft_r2c;
	fftwf_execute_dft_c2r_proc fftwf_execute_dft_c2r;
	fftwf_init_threads_proc fftwf_init_threads;
	fftwf_plan_with_nthreads_proc fftwf_plan_with_nthreads;

	int CPUFlags;
	std::unordered_set<std::string> CPUFeatures;

	fftwf_complex ** cachefft; //v1.8
	int * cachewhat;//v1.8
	int cachesize;//v1.8

#ifndef SSE2BUILD
	void InitOverlapPlane_C(float *__restrict inp, const BYTE *__restrict srcp) noexcept;
	void InitOverlapPlane_SSE(float *__restrict inp, const BYTE *__restrict srcp) noexcept;
	void InitOverlapPlane_wt2_C(float *__restrict inp, const BYTE *__restrict srcp) noexcept;
	void InitOverlapPlane_wt2_SSE(float *__restrict inp, const BYTE *__restrict srcp) noexcept;
#endif
	void InitOverlapPlane_SSE2(float *__restrict inp, const BYTE *__restrict srcp) noexcept;
	void InitOverlapPlane_SSSE3(float *__restrict inp, const BYTE *__restrict srcp) noexcept;
	void InitOverlapPlane_SSE4(float *__restrict inp, const BYTE *__restrict srcp) noexcept;
	void InitOverlapPlane_AVX(float *__restrict inp, const BYTE *__restrict srcp) noexcept;
	void InitOverlapPlane_AVX2(float *__restrict inp, const BYTE *__restrict srcp) noexcept;
	void InitOverlapPlane_AVX512(float *__restrict inp, const BYTE *__restrict srcp) noexcept;

	void InitOverlapPlane_wt2_SSE2(float *__restrict inp, const BYTE *__restrict srcp) noexcept;
	void InitOverlapPlane_wt2_SSSE3(float *__restrict inp, const BYTE *__restrict srcp) noexcept;
	void InitOverlapPlane_wt2_SSE4(float *__restrict inp, const BYTE *__restrict srcp) noexcept;
	void InitOverlapPlane_wt2_AVX(float *__restrict inp, const BYTE *__restrict srcp) noexcept;
	void InitOverlapPlane_wt2_AVX2(float *__restrict inp, const BYTE *__restrict srcp) noexcept;
	void InitOverlapPlane_wt2_AVX512(float *__restrict inp, const BYTE *__restrict srcp) noexcept;

#ifndef SSE2BUILD
	void DecodeOverlapPlane_C(const float *__restrict in, BYTE *__restrict dstp) noexcept;
	void DecodeOverlapPlane_SSE(const float *__restrict in, BYTE *__restrict dstp) noexcept;
#endif
	void DecodeOverlapPlane_SSE2(const float *__restrict in, BYTE *__restrict dstp) noexcept;
	void DecodeOverlapPlane_SSE4(const float *__restrict in, BYTE *__restrict dstp) noexcept;
	void DecodeOverlapPlane_AVX(const float *__restrict in, BYTE *__restrict dstp) noexcept;
	void DecodeOverlapPlane_AVX2(const float *__restrict in, BYTE *__restrict dstp) noexcept;
	void DecodeOverlapPlane_AVX512(const float *__restrict in, BYTE *__restrict dstp) noexcept;

	std::function<void(FFT3DFilter&, const float *, BYTE *)> DecodeOverlapPlane;
	std::function<void(FFT3DFilter&, float *, const BYTE *)> InitOverlapPlane;

	void InitFunctors();
	void DetectFeatures(IScriptEnvironment* env);
	void DetectFeatures();
	void GenWindows() noexcept;

public:
	// This defines that these functions are present in your class.
	// These functions must be that same as those actually implemented.
	// Since the functions are "public" they are accessible to other classes.
	// Otherwise they can only be called from functions within the class itself.

	FFT3DFilter(PClip _child, float _sigma, float _beta, int _plane, int _bw, int _bh, int _bt, int _ow, int _oh,
		float _kratio, float _sharpen, float _scutoff, float _svr, float _smin, float _smax,
		bool _measure, bool _interlaced, int _wintype,
		int _pframe, int _px, int _py, bool _pshow, float _pcutoff, float _pfactor,
		float _sigma2, float _sigma3, float _sigma4, float _degrid,
		float _dehalo, float _hr, float _ht, int _ncpu, int _multiplane, IScriptEnvironment* env);
	// This is the constructor. It does not return any value, and is always used,
	//  when an instance of the class is created.
	// Since there is no code in this, this is the definition.

	~FFT3DFilter();
	// The is the destructor definition. This is called when the filter is destroyed.


	PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env);
	// This is the function that AviSynth calls to get a given frame.
	// So when this functions gets called, the filter is supposed to return frame n.
};