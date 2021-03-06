//
//	FFT3DFilter plugin for Avisynth 2.5 - 3D Frequency Domain filter
//  AVX512 version of filtering functions
//
//	Derived from C version of function. (Copyright(C)2004-2006 A.G.Balakhnin aka Fizick, bag@hotmail.ru, http://avisynth.org.ru)
//  Copyright(C) 2018 Daniel Kl�ma aka Klimax
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

void KalmanFilter::ApplyKalman_AVX512() noexcept
{
	// return result in outLast
	int w(0);
	fftwf_complex *__restrict covar = covar_in, *__restrict covarProcess = covarProcess_in;
	const float sigmaSquaredMotionNormed = covarNoiseNormed * kratio2;
	const int outwidth8 = outwidth - outwidth % 8;
	const __m512 covarNoiseNormed8 = _mm512_maskz_broadcastss_ps(0xFFFF, _mm_broadcast_ss(&covarNoiseNormed));

	for (int block = start_block; block < blocks; block++)
	{
		for (int h = 0; h < bh; h++) // 
		{
			__m512 cur = _mm512_load_ps(outcur[0]);
			__m512 last = _mm512_load_ps(outLast[0]);
			for (w = 0; w < outwidth8; w = w + 8)
			{
				// use one of possible method for motion detection:
				__m512 r3 = _mm512_sub_ps(cur, last);
				r3 = _mm512_mul_ps(r3, r3);
				const __mmask16 k1 = _mm512_cmp_ps_mask(r3, _mm512_set1_ps(sigmaSquaredMotionNormed), 0x0e);
				__mmask16 k2 = k1 << 1;
				const __mmask16 k3 = k2 & 0x5555;
				k2 = k2 >> 1;
				k2 = k2 && 0xAAAA;
				k2 = k1 | k2;
				const __mmask16 mask = _mm512_kor(k3, k2); //positive mask - greater then

				const __m512 covarProcess4 = _mm512_load_ps(covarProcess[w]);

				const __m512 sum = _mm512_add_ps(_mm512_load_ps(covar[w]), covarProcess4);
				const __m512 gain = _mm512_div_ps(sum, _mm512_add_ps(sum, covarNoiseNormed8));

				r3 = _mm512_mul_ps(gain, gain);
				r3 = _mm512_mul_ps(r3, covarNoiseNormed8);

				r3 = _mm512_mask_blend_ps(mask, r3, covarNoiseNormed8);
				_mm512_store_ps(covarProcess[w], r3);

				r3 = _mm512_mul_ps(gain, sum);
				r3 = _mm512_sub_ps(sum, r3);

				r3 = _mm512_mask_blend_ps(mask, r3, covarNoiseNormed8);
				_mm512_store_ps(covar[w], r3);

				r3 = _mm512_mul_ps(gain, last);
				const __m512 r2 = _mm512_sub_ps(last, r3);
				r3 = _mm512_fmadd_ps(gain, cur, r2);

				r3 = _mm512_mask_blend_ps(mask, r3, cur);

				cur = _mm512_load_ps(outcur[w + 8]);
				last = _mm512_load_ps(outLast[w + 8]);

				_mm512_store_ps(outLast[w], r3);
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

void KalmanFilter::ApplyKalmanPattern_AVX512() noexcept
{
	// return result in outLast
	int w(0);
	fftwf_complex *__restrict covar = covar_in, *__restrict covarProcess = covarProcess_in;
	const __m512 kratio8 = _mm512_set1_ps(kratio2);

	const int outwidth8 = outwidth - outwidth % 8;

	for (int block = start_block; block < blocks; block++)
	{
		for (int h = 0; h < bh; h++) // 
		{
			for (w = 0; w < outwidth8; w = w + 8)
			{
				// use one of possible method for motion detection:
				const __m512 cur = _mm512_load_ps(outcur[w]);
				const __m512 last = _mm512_load_ps(outLast[w]);
				__m512 r3 = _mm512_sub_ps(cur, last);

				const __m512 cnn4 = _mm512_permutexvar_ps(_mm512_set_epi32(7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0), _mm512_castps256_ps512(_mm256_load_ps(&covarNoiseNormed2[w])));
				r3 = _mm512_mul_ps(r3, r3);
				const __mmask16 k1 = _mm512_cmp_ps_mask(r3, _mm512_mul_ps(cnn4, kratio8), 0x0e);
				__mmask16 k2 = k1 << 1;
				const __mmask16 k3 = k2 & 0x5555;
				k2 = k2 >> 1;
				k2 = k2 && 0xAAAA;
				k2 = k1 | k2;

				const __mmask16 mask = _mm512_kor(k3, k2); //positive mask - greater then

				const __m512 covar4 = _mm512_load_ps(covar[w]);
				const __m512 covarProcess4 = _mm512_load_ps(covarProcess[w]);

				const __m512 sum = _mm512_add_ps(covar4, covarProcess4);
				const __m512 gain = _mm512_div_ps(sum, _mm512_add_ps(sum, cnn4));

				r3 = _mm512_mul_ps(gain, gain);
				r3 = _mm512_mul_ps(r3, cnn4);

				r3 = _mm512_mask_blend_ps(mask, r3, cnn4);
				_mm512_store_ps(covarProcess[w], r3);

				r3 = _mm512_mul_ps(gain, sum);
				r3 = _mm512_sub_ps(sum, r3);
				r3 = _mm512_mask_blend_ps(mask, r3, cnn4);
				_mm512_store_ps(covar[w], r3);

				r3 = _mm512_mul_ps(gain, last);
				const __m512 r2 = _mm512_sub_ps(last, r3);
				r3 = _mm512_fmadd_ps(gain, cur, r2);

				r3 = _mm512_mask_blend_ps(mask, r3, cur);
				_mm512_store_ps(outLast[w], r3);
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