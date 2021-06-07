//
//	FFT3DFilter plugin for Avisynth 2.6 - 3D Frequency Domain filter
//  AVX2 version of filtering functions
//
//	Derived from C version of function. (Copyright(C)2004-2006 A.G.Balakhnin aka Fizick, bag@hotmail.ru, http://avisynth.org.ru)
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

#include "windows.h"
#include "fftwlite.h"
#include <intrin.h>
#include "Kalman.h"

void KalmanFilter::ApplyKalman_AVX2() noexcept
{
	// return result in outLast
	int w(0);
	fftwf_complex *__restrict covar = covar_in, *__restrict covarProcess = covarProcess_in;
	const float sigmaSquaredMotionNormed = covarNoiseNormed * kratio2;
	const int outwidth4 = outwidth - outwidth % 4;
	const __m256 covarNoiseNormed8 = _mm256_set1_ps(covarNoiseNormed);

	for (int block = start_block; block < blocks; block++)
	{
		for (int h = 0; h < bh; h++) // 
		{
			__m256 cur = _mm256_load_ps(outcur[0]);
			__m256 last = _mm256_load_ps(outLast[0]);
			for (w = 0; w < outwidth4; w = w + 4)
			{
				// use one of possible method for motion detection:
				__m256 r3 = _mm256_sub_ps(cur, last);
				r3 = _mm256_mul_ps(r3, r3);
				r3 = _mm256_cmp_ps(r3, _mm256_set1_ps(sigmaSquaredMotionNormed), 0x0e);
				const __m256 mask = _mm256_or_ps(r3, _mm256_shuffle_ps(r3, r3, _MM_SHUFFLE(2, 3, 0, 1))); //positive mask - greater then

				const __m256 covar4 = _mm256_load_ps(covar[w]);
				const __m256 covarProcess4 = _mm256_load_ps(covarProcess[w]);

				const __m256 sum = _mm256_add_ps(covar4, covarProcess4);
				const __m256 gain = _mm256_div_ps(sum, _mm256_add_ps(sum, covarNoiseNormed8));

				r3 = _mm256_mul_ps(gain, gain);
				r3 = _mm256_mul_ps(r3, covarNoiseNormed8);

				_mm256_store_ps(covarProcess[w], _mm256_blendv_ps(r3, covarNoiseNormed8, mask));

				r3 = _mm256_mul_ps(gain, sum);
				r3 = _mm256_sub_ps(sum, r3);
				_mm256_store_ps(covar[w], _mm256_blendv_ps(r3, covarNoiseNormed8, mask));

				r3 = _mm256_mul_ps(gain, last);
				const __m256 r2 = _mm256_sub_ps(last, r3);
				r3 = _mm256_fmadd_ps(gain, cur, r2);
				r3 = _mm256_blendv_ps(r3, cur, mask);
				cur = _mm256_load_ps(outcur[w + 4]);
				last = _mm256_load_ps(outLast[w + 4]);
				_mm256_store_ps(outLast[w], r3);
			}
			_mm_prefetch((const char*)(outcur + outpitch), _MM_HINT_T0);
			_mm_prefetch((const char*)(outLast + outpitch), _MM_HINT_T0);
			_mm_prefetch((const char*)(covar + outpitch), _MM_HINT_T0);
			_mm_prefetch((const char*)(covarProcess + outpitch), _MM_HINT_T0);
			for (; w < outwidth; w++)
			{
				// use one of possible method for motion detection:
				if ((outcur[w][0] - outLast[w][0])*(outcur[w][0] - outLast[w][0]) > sigmaSquaredMotionNormed ||
					(outcur[w][1] - outLast[w][1])*(outcur[w][1] - outLast[w][1]) > sigmaSquaredMotionNormed)
				{
					// big pixel variation due to motion etc
					// reset filter
					covar[w][0] = covarNoiseNormed;
					covar[w][1] = covarNoiseNormed;
					covarProcess[w][0] = covarNoiseNormed;
					covarProcess[w][1] = covarNoiseNormed;
					outLast[w][0] = outcur[w][0];
					outLast[w][1] = outcur[w][1];
					//return result in outLast
				}
				else
				{ // small variation
				  // useful sum
					float sumre = (covar[w][0] + covarProcess[w][0]);
					float sumim = (covar[w][1] + covarProcess[w][1]);
					// real gain, imagine gain
					float GainRe = sumre / (sumre + covarNoiseNormed);
					float GainIm = sumim / (sumim + covarNoiseNormed);
					// update process
					covarProcess[w][0] = (GainRe*GainRe*covarNoiseNormed);
					covarProcess[w][1] = (GainIm*GainIm*covarNoiseNormed);
					// update variation
					covar[w][0] = (1 - GainRe)*sumre;
					covar[w][1] = (1 - GainIm)*sumim;
					outLast[w][0] = (GainRe*outcur[w][0] + (1 - GainRe)*outLast[w][0]);
					outLast[w][1] = (GainIm*outcur[w][1] + (1 - GainIm)*outLast[w][1]);
					//return filtered result in outLast
				}
			}
			outcur += outpitch;
			outLast += outpitch;
			covar += outpitch;
			covarProcess += outpitch;
		}
	}
}

void KalmanFilter::ApplyKalmanPattern_AVX2() noexcept
{
	// return result in outLast
	int w(0);
	fftwf_complex *covar = covar_in, *covarProcess = covarProcess_in;
	const __m256 kratio8 = _mm256_set1_ps(kratio2);

	const int outwidth4 = outwidth - outwidth % 4;

	for (int block = start_block; block < blocks; block++)
	{
		for (int h = 0; h < bh; h++) // 
		{
			for (w = 0; w < outwidth4; w = w + 4)
			{
				// use one of possible method for motion detection:
				const __m256 cur = _mm256_load_ps(outcur[w]);
				const __m256 last = _mm256_load_ps(outLast[w]);
				__m256 r3 = _mm256_sub_ps(cur, last);
				__m256 cnn4 = _mm256_broadcast_ps((__m128*) &covarNoiseNormed2[w]);
				cnn4 = _mm256_unpacklo_ps(cnn4, cnn4);
				r3 = _mm256_mul_ps(r3, r3);
				r3 = _mm256_cmp_ps(r3, _mm256_mul_ps(cnn4, kratio8), 0x0e);
				const __m256 mask = _mm256_or_ps(r3, _mm256_shuffle_ps(r3, r3, _MM_SHUFFLE(2, 3, 0, 1))); //positive mask - greater then

				const __m256 covar4 = _mm256_load_ps(covar[w]);
				const __m256 covarProcess4 = _mm256_load_ps(covarProcess[w]);

				const __m256 sum = _mm256_add_ps(covar4, covarProcess4);
				const __m256 gain = _mm256_div_ps(sum, _mm256_add_ps(sum, cnn4));

				r3 = _mm256_mul_ps(gain, gain);
				r3 = _mm256_mul_ps(r3, cnn4);

				_mm256_store_ps(covarProcess[w], _mm256_blendv_ps(r3, cnn4, mask));

				r3 = _mm256_mul_ps(gain, sum);
				r3 = _mm256_sub_ps(sum, r3);
				_mm256_store_ps(covar[w], _mm256_blendv_ps(r3, cnn4, mask));

				r3 = _mm256_mul_ps(gain, last);
				const __m256 r2 = _mm256_sub_ps(last, r3);
				r3 = _mm256_fmadd_ps(gain, cur, r2);

				_mm256_store_ps(outLast[w], _mm256_blendv_ps(r3, cur, mask));
			}
			_mm_prefetch((const char*)(outcur + outpitch), _MM_HINT_T0);
			_mm_prefetch((const char*)(outLast + outpitch), _MM_HINT_T0);
			_mm_prefetch((const char*)(covar + outpitch), _MM_HINT_T0);
			_mm_prefetch((const char*)(covarProcess + outpitch), _MM_HINT_T0);
			_mm_prefetch((const char*)(covarNoiseNormed2 + outpitch), _MM_HINT_T0);
			for (; w < outwidth; w++)
			{
				// use one of possible method for motion detection:
				if ((outcur[w][0] - outLast[w][0])*(outcur[w][0] - outLast[w][0]) > covarNoiseNormed2[w] * kratio2 ||
					(outcur[w][1] - outLast[w][1])*(outcur[w][1] - outLast[w][1]) > covarNoiseNormed2[w] * kratio2)
				{
					// big pixel variation due to motion etc
					// reset filter
					covar[w][0] = covarNoiseNormed2[w];
					covar[w][1] = covarNoiseNormed2[w];
					covarProcess[w][0] = covarNoiseNormed2[w];
					covarProcess[w][1] = covarNoiseNormed2[w];
					outLast[w][0] = outcur[w][0];
					outLast[w][1] = outcur[w][1];
					//return result in outLast
				}
				else
				{ // small variation
				  // useful sum
					float sumre = (covar[w][0] + covarProcess[w][0]);
					float sumim = (covar[w][1] + covarProcess[w][1]);
					// real gain, imagine gain
					float GainRe = sumre / (sumre + covarNoiseNormed2[w]);
					float GainIm = sumim / (sumim + covarNoiseNormed2[w]);
					// update process
					covarProcess[w][0] = (GainRe*GainRe*covarNoiseNormed2[w]);
					covarProcess[w][1] = (GainIm*GainIm*covarNoiseNormed2[w]);
					// update variation
					covar[w][0] = (1 - GainRe)*sumre;
					covar[w][1] = (1 - GainIm)*sumim;
					outLast[w][0] = (GainRe*outcur[w][0] + (1 - GainRe)*outLast[w][0]);
					outLast[w][1] = (GainIm*outcur[w][1] + (1 - GainIm)*outLast[w][1]);
					//return filtered result in outLast
				}
			}
			outcur += outpitch;
			outLast += outpitch;
			covar += outpitch;
			covarProcess += outpitch;
			covarNoiseNormed2 += outpitch;
		}
		covarNoiseNormed2 -= outpitch * bh;
	}
}