/*
	FFT3DFilter plugin for Avisynth 2.6 - 3D Frequency Domain filter
	Functions for working with patterns

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
*/

#include "fft3dfilter.h"

#ifndef SSE2BUILD
void FindPatternBlock_C(fftwf_complex *outcur0, int outwidth, int outpitch, int bh, int nox, int noy, int &px, int &py, float *pwin, float degrid, fftwf_complex *gridsample) noexcept;
void FindPatternBlock_SSE(fftwf_complex *outcur0, int outwidth, int outpitch, int bh, int nox, int noy, int &px, int &py, float *pwin, float degrid, fftwf_complex *gridsample) noexcept;
#endif
void FindPatternBlock_SSE2(fftwf_complex *outcur0, int outwidth, int outpitch, int bh, int nox, int noy, int &px, int &py, float *pwin, float degrid, fftwf_complex *gridsample) noexcept;
void FindPatternBlock_SSE3(fftwf_complex *outcur0, int outwidth, int outpitch, int bh, int nox, int noy, int &px, int &py, float *pwin, float degrid, fftwf_complex *gridsample) noexcept;
void FindPatternBlock_AVX(fftwf_complex *outcur0, int outwidth, int outpitch, int bh, int nox, int noy, int &px, int &py, float *pwin, float degrid, fftwf_complex *gridsample) noexcept;
void FindPatternBlock_AVX2(fftwf_complex *outcur0, int outwidth, int outpitch, int bh, int nox, int noy, int &px, int &py, float *pwin, float degrid, fftwf_complex *gridsample) noexcept;

void SetPattern_C(fftwf_complex *outcur, int outwidth, int outpitch, int bh, int nox, int px, int py, float *pwin, float *pattern2d, float &psigma, float degrid, fftwf_complex *gridsample);

void fill_complex(fftwf_complex *plane, int outsize, float realvalue, float imgvalue) noexcept
{
	// it is not fast, but called only in constructor
	for (int w = 0; w < outsize; w++) {
		plane[w][0] = realvalue;
		plane[w][1] = imgvalue;
	}
}
//-------------------------------------------------------------------
void SigmasToPattern(float sigma, float sigma2, float sigma3, float sigma4, int bh, int outwidth, int outpitch, float norm, float *pattern2d) noexcept
{
	// it is not fast, but called only in constructor
	float sigmacur(0.0f);
	const float ft2 = sqrt(0.5f) / 2; // frequency for sigma2
	const float ft3 = sqrt(0.5f) / 4; // frequency for sigma3
	for (int h = 0; h < bh; h++)
	{
		for (int w = 0; w < outwidth; w++)
		{
			const float fy = (bh - 2.0f*abs(h - bh / 2)) / bh; // normalized to 1
			const float fx = (w*1.0f) / outwidth;  // normalized to 1
			const float f = sqrt((fx*fx + fy * fy)*0.5f); // normalized to 1
			if (f < ft3)
			{ // low frequencies
				sigmacur = sigma4 + (sigma3 - sigma4)*f / ft3;
			}
			else if (f < ft2)
			{ // middle frequencies
				sigmacur = sigma3 + (sigma2 - sigma3)*(f - ft3) / (ft2 - ft3);
			}
			else
			{// high frequencies
				sigmacur = sigma + (sigma2 - sigma)*(1 - f) / (1 - ft2);
			}
			pattern2d[w] = sigmacur * sigmacur / norm;
		}
		pattern2d += outpitch;
	}
}
//Klimax: Bugged?
//-------------------------------------------------------------------------------------------
void FindPatternBlock(fftwf_complex *outcur, int outwidth, int outpitch, int bh, int nox, int noy, int &px, int &py, float *pwin, float degrid, fftwf_complex *gridsample, int CPUFlags) noexcept
{
	if (CPUFlags & CPUK_AVX2)
		FindPatternBlock_AVX2(outcur, outwidth, outpitch, bh, nox, noy, px, py, pwin, degrid, gridsample);
	else if (CPUFlags & CPUK_AVX)
		FindPatternBlock_AVX(outcur, outwidth, outpitch, bh, nox, noy, px, py, pwin, degrid, gridsample);
	else if (CPUFlags & CPUK_SSE3)
		FindPatternBlock_SSE3(outcur, outwidth, outpitch, bh, nox, noy, px, py, pwin, degrid, gridsample);
	else if (CPUFlags & CPUK_SSE2)
		FindPatternBlock_SSE2(outcur, outwidth, outpitch, bh, nox, noy, px, py, pwin, degrid, gridsample);
#ifndef SSE2BUILD
	else if (CPUFlags & CPUK_SSE)
		FindPatternBlock_SSE(outcur, outwidth, outpitch, bh, nox, noy, px, py, pwin, degrid, gridsample);
	else
		FindPatternBlock_C(outcur, outwidth, outpitch, bh, nox, noy, px, py, pwin, degrid, gridsample);
#endif
}
//-------------------------------------------------------------------------------------------
void SetPattern(fftwf_complex *outcur, int outwidth, int outpitch, int bh, int nox, int px, int py, float *pwin, float *pattern2d, float &psigma, float degrid, fftwf_complex *gridsample) noexcept
{
	SetPattern_C(outcur, outwidth, outpitch, bh, nox, px, py, pwin, pattern2d, psigma, degrid, gridsample);
}
//-------------------------------------------------------------------------------------------
void PutPatternOnly(fftwf_complex *outcur, int outwidth, int outpitch, int bh, int nox, int noy, int px, int py) noexcept
{
	int w(0), block(0);
	const int pblock = py * nox + px;
	const int blocks = nox * noy;

	for (block = 0; block < pblock; block++)
	{
		for (int h = 0; h < bh; h++)
		{
			for (w = 0; w < outwidth; w++)
			{
				outcur[w][0] = 0;
				outcur[w][1] = 0;
			}
			outcur += outpitch;
		}
	}

	outcur += bh * outpitch;

	for (block = pblock + 1; block < blocks; block++)
	{
		for (int h = 0; h < bh; h++)
		{
			for (w = 0; w < outwidth; w++)
			{
				outcur[w][0] = 0;
				outcur[w][1] = 0;
			}
			outcur += outpitch;
		}
	}

}