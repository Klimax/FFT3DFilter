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
//
#include "windows.h"
#include "fftwlite.h"
#include <intrin.h>
#include "Wiener.h"

//-----------------------------------------------------------------------------------------
//
void WienerFilter::ApplyWiener2D_AVX2() noexcept
{
	if (sharpen == 0 && dehalo == 0)// no sharpen, no dehalo
	{
		for (int block = start_block; block < blocks; block++)
		{
			for (int h = 0; h < bh; h++) // middle
			{
				for (int w = 0; w < outwidth; w = w + 4) // not skip first v.1.2
				{
					__m256 r1 = _mm256_load_ps(outcur[w]);
					__m256 r3 = _mm256_mul_ps(r1, r1);
					r3 = _mm256_hadd_ps(r3, r3);
					r3 = _mm256_add_ps(r3, _mm256_set1_ps(1e-15f));
					const __m256 r2 = _mm256_sub_ps(r3, _mm256_set1_ps(sigmaSquaredNoiseNormed));
					r3 = _mm256_div_ps(r2, r3);
					r3 = _mm256_max_ps(r3, _mm256_set1_ps(lowlimit));
					r3 = _mm256_unpacklo_ps(r3, r3);
					r1 = _mm256_mul_ps(r1, r3);
					_mm256_store_ps(outcur[w], r1);
				}
				_mm_prefetch((const char*)(outcur + outpitch), _MM_HINT_T0);

				outcur += outpitch;
			}
		}
	}
	else if (sharpen != 0 && dehalo == 0) // sharpen
	{
		for (int block = start_block; block < blocks; block++)
		{
			for (int h = 0; h < bh; h++) // middle
			{
				for (int w = 0; w < outwidth; w = w + 4) // not skip first
				{
					__m256 r1 = _mm256_load_ps(outcur[w]);
					__m256 r3 = _mm256_mul_ps(r1, r1);
					r3 = _mm256_hadd_ps(r3, r3);
					__m256 psd8 = _mm256_add_ps(r3, _mm256_set1_ps(1e-15f));
					psd8 = _mm256_permutevar_ps(psd8, _mm256_set_epi32(3, 3, 2, 2, 1, 1, 0, 0));

					__m256 r7 = _mm256_mul_ps(_mm256_set1_ps(sharpen), _mm256_broadcast_ps((__m128*)&wsharpen[w]));
					__m256 r4 = _mm256_mul_ps(psd8, _mm256_set1_ps(sigmaSquaredSharpenMax));
					__m256 r5 = _mm256_add_ps(psd8, _mm256_set1_ps(sigmaSquaredSharpenMin));
					const __m256 r6 = _mm256_add_ps(psd8, _mm256_set1_ps(sigmaSquaredSharpenMax));
					r5 = _mm256_mul_ps(r5, r6);
					r4 = _mm256_div_ps(r4, r5);
					r4 = _mm256_sqrt_ps(r4);
					r7 = _mm256_permutevar_ps(r7, _mm256_set_epi32(3, 3, 2, 2, 1, 1, 0, 0));
					r4 = _mm256_mul_ps(r4, r7);

					const __m256 r2 = _mm256_sub_ps(psd8, _mm256_set1_ps(sigmaSquaredNoiseNormed));
					r3 = _mm256_div_ps(r2, psd8);
					r3 = _mm256_max_ps(r3, _mm256_set1_ps(lowlimit));

					r4 = _mm256_add_ps(r4, _mm256_set1_ps(1));
					r3 = _mm256_mul_ps(r3, r4);
					r1 = _mm256_mul_ps(r1, r3);
					_mm256_store_ps(outcur[w], r1);
				}
				_mm_prefetch((const char*)(outcur + outpitch), _MM_HINT_T0);
				_mm_prefetch((const char*)(wsharpen + outpitch), _MM_HINT_T0);

				outcur += outpitch;
				wsharpen += outpitch;
			}
			wsharpen -= outpitch * bh;
		}
	}
	else if (sharpen == 0 && dehalo != 0)
	{
		for (int block = start_block; block < blocks; block++)
		{
			for (int h = 0; h < bh; h++) // middle
			{
				for (int w = 0; w < outwidth; w = w + 4) // not skip first4
				{
					__m256 r1 = _mm256_load_ps(outcur[w]);
					__m256 r3 = _mm256_mul_ps(r1, r1);
					r3 = _mm256_hadd_ps(r3, r3);
					__m256 psd8 = _mm256_add_ps(r3, _mm256_set1_ps(1e-15f));
					psd8 = _mm256_permutevar_ps(psd8, _mm256_set_epi32(3, 3, 2, 2, 1, 1, 0, 0));

					__m256 r5 = _mm256_add_ps(psd8, _mm256_set1_ps(ht2n));
					__m256 r6 = _mm256_mul_ps(psd8, _mm256_set1_ps(dehalo));
					const __m256 r7 = _mm256_broadcast_ps((__m128*)&wdehalo[w]);
					r6 = _mm256_mul_ps(r6, _mm256_permutevar_ps(r7, _mm256_set_epi32(3, 3, 2, 2, 1, 1, 0, 0)));
					r6 = _mm256_add_ps(r5, r6);
					r5 = _mm256_div_ps(r5, r6);

					const __m256 r2 = _mm256_sub_ps(psd8, _mm256_set1_ps(sigmaSquaredNoiseNormed));
					r3 = _mm256_div_ps(r2, psd8);
					r3 = _mm256_max_ps(r3, _mm256_set1_ps(lowlimit));

					r3 = _mm256_mul_ps(r3, r5);
					r1 = _mm256_mul_ps(r1, r3);
					_mm256_store_ps(outcur[w], r1);
				}
				_mm_prefetch((const char*)(outcur + outpitch), _MM_HINT_T0);
				_mm_prefetch((const char*)(wdehalo + outpitch), _MM_HINT_T0);

				outcur += outpitch;
				wdehalo += outpitch;
			}
			wdehalo -= outpitch * bh;
		}
	}
	else if (sharpen != 0 && dehalo != 0)
	{
		for (int block = start_block; block < blocks; block++)
		{
			for (int h = 0; h < bh; h++) // middle
			{
				for (int w = 0; w < outwidth; w = w + 4) // not skip first4
				{
					__m256 r1 = _mm256_load_ps(outcur[w]);
					__m256 r3 = _mm256_mul_ps(r1, r1);
					r3 = _mm256_hadd_ps(r3, r3);
					__m256 psd8 = _mm256_add_ps(r3, _mm256_set1_ps(1e-15f));
					psd8 = _mm256_permutevar_ps(psd8, _mm256_set_epi32(3, 3, 2, 2, 1, 1, 0, 0));

					__m256 r5 = _mm256_mul_ps(_mm256_set1_ps(sharpen), _mm256_broadcast_ps((__m128*)&wsharpen[w]));
					__m256 r6 = _mm256_add_ps(psd8, _mm256_set1_ps(sigmaSquaredSharpenMax));
					__m256 r7 = _mm256_add_ps(psd8, _mm256_set1_ps(sigmaSquaredSharpenMin));
					__m256 r8 = _mm256_mul_ps(psd8, _mm256_set1_ps(sigmaSquaredSharpenMax));

					r7 = _mm256_mul_ps(r6, r7);
					r8 = _mm256_div_ps(r8, r7);
					r6 = _mm256_sqrt_ps(r8);

					r5 = _mm256_permutevar_ps(r5, _mm256_set_epi32(3, 3, 2, 2, 1, 1, 0, 0));
					const __m256 sharp8 = _mm256_mul_ps(r5, r6);

					r5 = _mm256_add_ps(psd8, _mm256_set1_ps(ht2n));
					r6 = _mm256_mul_ps(psd8, _mm256_set1_ps(dehalo));
					r7 = _mm256_broadcast_ps((__m128*)&wdehalo[w]);
					r6 = _mm256_mul_ps(r6, _mm256_permutevar_ps(r7, _mm256_set_epi32(3, 3, 2, 2, 1, 1, 0, 0)));
					r6 = _mm256_add_ps(r5, r6);
					const __m256 dehalo8 = _mm256_div_ps(r5, r6);

					r5 = _mm256_mul_ps(sharp8, dehalo8);

					r5 = _mm256_add_ps(_mm256_set1_ps(1.0f), r5);

					const __m256 r2 = _mm256_sub_ps(psd8, _mm256_set1_ps(sigmaSquaredNoiseNormed));
					r3 = _mm256_div_ps(r2, psd8);
					r3 = _mm256_max_ps(r3, _mm256_set1_ps(lowlimit));

					r3 = _mm256_mul_ps(r3, r5);
					r1 = _mm256_mul_ps(r1, r3);
					_mm256_store_ps(outcur[w], r1);
				}
				_mm_prefetch((const char*)(outcur + outpitch), _MM_HINT_T0);
				_mm_prefetch((const char*)(wsharpen + outpitch), _MM_HINT_T0);
				_mm_prefetch((const char*)(wdehalo + outpitch), _MM_HINT_T0);

				outcur += outpitch;
				wsharpen += outpitch;
				wdehalo += outpitch;
			}
			wsharpen -= outpitch * bh;
			wdehalo -= outpitch * bh;
		}
	}

}

void WienerFilter::ApplyWiener2D_degrid_AVX2() noexcept
{
	if (sharpen == 0 && dehalo == 0)// no sharpen, no dehalo
	{
		for (int block = start_block; block < blocks; block++)
		{
			const __m256 gridfraction8 = _mm256_set1_ps(degrid * outcur[0][0] / gridsample[0][0]);
			for (int h = 0; h < bh; h++) // middle
			{
				for (int w = 0; w < outwidth; w = w + 4) // not skip first v.1.2
				{
					const __m256 gridcorrection = _mm256_mul_ps(gridfraction8, _mm256_load_ps(gridsample[w])); //gridcorrection
					__m256 r1 = _mm256_load_ps(outcur[w]);
					const __m256 r4 = _mm256_sub_ps(r1, gridcorrection); //corrected
					__m256 r3 = _mm256_mul_ps(r4, r4); //corrected^2
					r3 = _mm256_hadd_ps(r3, r3); //psd
					r3 = _mm256_add_ps(r3, _mm256_set1_ps(1e-15f)); //psd
					const __m256 r2 = _mm256_sub_ps(r3, _mm256_set1_ps(sigmaSquaredNoiseNormed));
					r3 = _mm256_div_ps(r2, r3);
					r3 = _mm256_max_ps(r3, _mm256_set1_ps(lowlimit)); //wienerfactor
					r3 = _mm256_unpacklo_ps(r3, r3);
					r1 = _mm256_fmadd_ps(r4, r3, gridcorrection);
					_mm256_store_ps(outcur[w], r1);
				}
				_mm_prefetch((const char*)(outcur + outpitch), _MM_HINT_T0);
				_mm_prefetch((const char*)(gridsample + outpitch), _MM_HINT_T0);

				outcur += outpitch;
				gridsample += outpitch;
			}
			gridsample -= outpitch * bh; // restore pointer to only valid first block
		}
	}
	else if (sharpen != 0 && dehalo == 0) // sharpen
	{
		for (int block = start_block; block < blocks; block++)
		{
			const __m256 gridfraction8 = _mm256_set1_ps(degrid * outcur[0][0] / gridsample[0][0]);
			for (int h = 0; h < bh; h++) // middle
			{
				__m256 r1 = _mm256_load_ps(outcur[0]);
				for (int w = 0; w < outwidth; w = w + 4) // not skip first
				{
					const __m256 gridcorrection = _mm256_mul_ps(gridfraction8, _mm256_load_ps(gridsample[w])); //gridcorrection
					const __m256 r4 = _mm256_sub_ps(r1, gridcorrection); //corrected
					__m256 r3 = _mm256_mul_ps(r4, r4); //corrected^2
					r3 = _mm256_hadd_ps(r3, r3); //psd
					__m256 psd8 = _mm256_add_ps(r3, _mm256_set1_ps(1e-15f)); //psd [0,2]
					psd8 = _mm256_permutevar_ps(psd8, _mm256_set_epi32(3, 3, 2, 2, 1, 1, 0, 0));
					__m256 r2 = _mm256_sub_ps(psd8, _mm256_set1_ps(sigmaSquaredNoiseNormed));
					r3 = _mm256_div_ps(r2, psd8);
					__m256 wienerfactor4 = _mm256_max_ps(r3, _mm256_set1_ps(lowlimit)); //wienerfactor

					__m256 r5 = _mm256_mul_ps(_mm256_set1_ps(sharpen), _mm256_broadcast_ps((__m128*)&wsharpen[w]));
					__m256 r6 = _mm256_add_ps(psd8, _mm256_set1_ps(sigmaSquaredSharpenMax));
					__m256 r7 = _mm256_add_ps(psd8, _mm256_set1_ps(sigmaSquaredSharpenMin));
					__m256 r8 = _mm256_mul_ps(psd8, _mm256_set1_ps(sigmaSquaredSharpenMax));

					r7 = _mm256_mul_ps(r6, r7);
					r8 = _mm256_div_ps(r8, r7);
					r6 = _mm256_sqrt_ps(r8);

					r5 = _mm256_permutevar_ps(r5, _mm256_set_epi32(3, 3, 2, 2, 1, 1, 0, 0));
					r5 = _mm256_fmadd_ps(r5, r6, _mm256_set1_ps(1.0f));
					wienerfactor4 = _mm256_mul_ps(r5, wienerfactor4);

					r2 = _mm256_fmadd_ps(r4, wienerfactor4, gridcorrection);
					r1 = _mm256_load_ps(outcur[w + 4]);
					_mm256_store_ps(outcur[w], r2);
				}
				_mm_prefetch((const char*)(outcur + outpitch), _MM_HINT_T0);
				_mm_prefetch((const char*)(gridsample + outpitch), _MM_HINT_T0);
				_mm_prefetch((const char*)(wsharpen + outpitch), _MM_HINT_T0);

				outcur += outpitch;
				wsharpen += outpitch;
				gridsample += outpitch;
			}
			wsharpen -= outpitch * bh;
			gridsample -= outpitch * bh; // restore pointer to only valid first block
		}
	}
	else if (sharpen == 0 && dehalo != 0)
	{
		for (int block = start_block; block < blocks; block++)
		{
			const __m256 gridfraction8 = _mm256_set1_ps(degrid * outcur[0][0] / gridsample[0][0]);
			for (int h = 0; h < bh; h++) // middle
			{
				for (int w = 0; w < outwidth; w = w + 4) // not skip first
				{
					const __m256 gridcorrection = _mm256_mul_ps(gridfraction8, _mm256_load_ps(gridsample[w])); //gridcorrection
					__m256 r1 = _mm256_load_ps(outcur[w]);
					const __m256 r4 = _mm256_sub_ps(r1, gridcorrection); //corrected
					__m256 r3 = _mm256_mul_ps(r4, r4); //corrected^2
					r3 = _mm256_hadd_ps(r3, r3); //psd
					__m256 psd8 = _mm256_add_ps(r3, _mm256_set1_ps(1e-15f)); //psd [0,1]
					psd8 = _mm256_permutevar_ps(psd8, _mm256_set_epi32(3, 3, 2, 2, 1, 1, 0, 0));
					const __m256 r2 = _mm256_sub_ps(psd8, _mm256_set1_ps(sigmaSquaredNoiseNormed));
					r3 = _mm256_div_ps(r2, psd8);
					__m256 wienerfactor8 = _mm256_max_ps(r3, _mm256_set1_ps(lowlimit)); //wienerfactor

					__m256 r5 = _mm256_add_ps(psd8, _mm256_set1_ps(ht2n));
					__m256 r6 = _mm256_mul_ps(psd8, _mm256_set1_ps(dehalo));
					const __m256 r7 = _mm256_broadcast_ps((__m128*)&wdehalo[w]);
					r6 = _mm256_fmadd_ps(r6, _mm256_permutevar_ps(r7, _mm256_set_epi32(3, 3, 2, 2, 1, 1, 0, 0)), r5);

					r5 = _mm256_div_ps(r5, r6);
					wienerfactor8 = _mm256_mul_ps(r5, wienerfactor8);

					r1 = _mm256_fmadd_ps(r4, wienerfactor8, gridcorrection);
					_mm256_store_ps(outcur[w], r1);
				}
				_mm_prefetch((const char*)(outcur + outpitch), _MM_HINT_T0);
				_mm_prefetch((const char*)(gridsample + outpitch), _MM_HINT_T0);
				_mm_prefetch((const char*)(wdehalo + outpitch), _MM_HINT_T0);

				outcur += outpitch;
				wdehalo += outpitch;
				gridsample += outpitch;
			}
			wdehalo -= outpitch * bh;
			gridsample -= outpitch * bh; // restore pointer to only valid first block
		}
	}
	else if (sharpen != 0 && dehalo != 0)
	{
		for (int block = start_block; block < blocks; block++)
		{
			const __m256 gridfraction8 = _mm256_set1_ps(degrid * outcur[0][0] / gridsample[0][0]);
			for (int h = 0; h < bh; h++) // middle
			{
				for (int w = 0; w < outwidth; w = w + 4) // not skip first
				{
					const __m256 gridcorrection = _mm256_mul_ps(gridfraction8, _mm256_load_ps(gridsample[w])); //gridcorrection
					__m256 r1 = _mm256_load_ps(outcur[w]);
					const __m256 r4 = _mm256_sub_ps(r1, gridcorrection); //corrected
					__m256 r3 = _mm256_mul_ps(r4, r4); //corrected^2
					r3 = _mm256_hadd_ps(r3, r3); //psd
					__m256 psd8 = _mm256_add_ps(r3, _mm256_set1_ps(1e-15f)); //psd [0,2]
					psd8 = _mm256_permutevar_ps(psd8, _mm256_set_epi32(3, 3, 2, 2, 1, 1, 0, 0));
					const __m256 r2 = _mm256_sub_ps(psd8, _mm256_set1_ps(sigmaSquaredNoiseNormed));
					r3 = _mm256_div_ps(r2, psd8);
					__m256 wienerfactor8 = _mm256_max_ps(r3, _mm256_set1_ps(lowlimit)); //wienerfactor

					__m256 r5 = _mm256_mul_ps(_mm256_set1_ps(sharpen), _mm256_broadcast_ps((__m128*)&wsharpen[w]));
					__m256 r6 = _mm256_add_ps(psd8, _mm256_set1_ps(sigmaSquaredSharpenMax));
					__m256 r7 = _mm256_add_ps(psd8, _mm256_set1_ps(sigmaSquaredSharpenMin));
					__m256 r8 = _mm256_mul_ps(psd8, _mm256_set1_ps(sigmaSquaredSharpenMax));

					r7 = _mm256_mul_ps(r6, r7);
					r8 = _mm256_div_ps(r8, r7);
					r6 = _mm256_sqrt_ps(r8);

					r5 = _mm256_permutevar_ps(r5, _mm256_set_epi32(3, 3, 2, 2, 1, 1, 0, 0));
					const __m256 sharp8 = _mm256_mul_ps(r5, r6);

					r5 = _mm256_add_ps(psd8, _mm256_set1_ps(ht2n));
					r6 = _mm256_mul_ps(psd8, _mm256_set1_ps(dehalo));
					r7 = _mm256_broadcast_ps((__m128*)&wdehalo[w]);
					r6 = _mm256_fmadd_ps(r6, _mm256_permutevar_ps(r7, _mm256_set_epi32(3, 3, 2, 2, 1, 1, 0, 0)), r5);
					const __m256 dehalo8 = _mm256_div_ps(r5, r6);

					r5 = _mm256_mul_ps(sharp8, dehalo8);
					r5 = _mm256_add_ps(_mm256_set1_ps(1.0f), r5);
					wienerfactor8 = _mm256_mul_ps(r5, wienerfactor8);

					r1 = _mm256_fmadd_ps(r4, wienerfactor8, gridcorrection);
					_mm256_store_ps(outcur[w], r1);
				}
				_mm_prefetch((const char*)(outcur + outpitch), _MM_HINT_T0);
				_mm_prefetch((const char*)(gridsample + outpitch), _MM_HINT_T0);
				_mm_prefetch((const char*)(wsharpen + outpitch), _MM_HINT_T0);
				_mm_prefetch((const char*)(wdehalo + outpitch), _MM_HINT_T0);

				outcur += outpitch;
				wsharpen += outpitch;
				gridsample += outpitch;
				wdehalo += outpitch;
			}
			wsharpen -= outpitch * bh;
			wdehalo -= outpitch * bh;
			gridsample -= outpitch * bh; // restore pointer to only valid first block
		}
	}

}

void WienerFilter::ApplyWiener3D2_AVX2() noexcept
{
	// dft 3d (very short - 2 points)
	for (int block = start_block; block < blocks; block++)
	{
		for (int h = 0; h < bh; h++)
		{
			__m256 r3 = _mm256_load_ps(outcur[0]);
			__m256 r2 = _mm256_load_ps(outprev[0]);
			for (int w = 0; w < outwidth; w = w + 4)
			{
				__m256 sum4 = _mm256_add_ps(r3, r2);
				__m256 dif4 = _mm256_sub_ps(r3, r2);
				__m256 r1 = _mm256_mul_ps(sum4, sum4);
				r2 = _mm256_mul_ps(dif4, dif4);

				__m256 psd8 = _mm256_hadd_ps(r1, r2);

				psd8 = _mm256_add_ps(psd8, _mm256_set1_ps(1e-15f));

				r1 = _mm256_sub_ps(psd8, _mm256_set1_ps(sigmaSquaredNoiseNormed));
				r1 = _mm256_div_ps(r1, psd8);

				const __m256 WienerFactor8 = _mm256_max_ps(r1, _mm256_set1_ps(lowlimit));

				r1 = _mm256_permute_ps(WienerFactor8, _MM_SHUFFLE(1, 1, 0, 0));
				r2 = _mm256_permute_ps(WienerFactor8, _MM_SHUFFLE(3, 3, 2, 2));

				sum4 = _mm256_mul_ps(sum4, r1);
				dif4 = _mm256_mul_ps(dif4, r2);

				r1 = _mm256_add_ps(sum4, dif4);
				r3 = _mm256_load_ps(outcur[w + 4]);
				r2 = _mm256_load_ps(outprev[w + 4]);
				r1 = _mm256_mul_ps(r1, _mm256_set1_ps(0.5f));
				_mm256_store_ps(outprev[w], r1);
			}
			_mm_prefetch((const char*)(outprev + outpitch), _MM_HINT_T0);
			_mm_prefetch((const char*)(outcur + outpitch), _MM_HINT_T0);

			outcur += outpitch;
			outprev += outpitch;
		}
	}
}

void WienerFilter::ApplyWiener3D2_degrid_AVX2() noexcept
{
	// dft 3d (very short - 2 points)
	for (int block = start_block; block < blocks; block++)
	{
		const __m256 gridfraction8 = _mm256_set1_ps(degrid * outcur[0][0] / gridsample[0][0]);
		for (int h = 0; h < bh; h++)
		{
			__m256 r3 = _mm256_load_ps(outcur[0]);
			__m256 r2 = _mm256_load_ps(outprev[0]);
			for (int w = 0; w < outwidth; w = w + 4)
			{
				__m256 gridcorrection = _mm256_mul_ps(gridfraction8, _mm256_load_ps(gridsample[w])); //gridcorrection
				gridcorrection = _mm256_mul_ps(gridcorrection, _mm256_set1_ps(2.0));
				__m256 sum8 = _mm256_add_ps(r3, r2);
				sum8 = _mm256_sub_ps(sum8, gridcorrection);
				__m256 dif8 = _mm256_sub_ps(r3, r2);
				__m256 r1 = _mm256_mul_ps(sum8, sum8);
				r2 = _mm256_mul_ps(dif8, dif8);

				__m256 psd8 = _mm256_hadd_ps(r1, r2);

				psd8 = _mm256_add_ps(psd8, _mm256_set1_ps(1e-15f));

				r1 = _mm256_sub_ps(psd8, _mm256_set1_ps(sigmaSquaredNoiseNormed));
				r1 = _mm256_div_ps(r1, psd8);

				const __m256 WienerFactor8 = _mm256_max_ps(r1, _mm256_set1_ps(lowlimit));

				r1 = _mm256_permute_ps(WienerFactor8, _MM_SHUFFLE(1, 1, 0, 0));
				r2 = _mm256_permute_ps(WienerFactor8, _MM_SHUFFLE(3, 3, 2, 2));

				sum8 = _mm256_mul_ps(sum8, r1);
				dif8 = _mm256_mul_ps(dif8, r2);

				r1 = _mm256_add_ps(sum8, dif8);
				r3 = _mm256_load_ps(outcur[w + 4]);
				r2 = _mm256_load_ps(outprev[w + 4]);
				r1 = _mm256_add_ps(gridcorrection, r1);
				r1 = _mm256_mul_ps(r1, _mm256_set1_ps(0.5f));
				_mm256_store_ps(outprev[w], r1);
			}
			_mm_prefetch((const char*)(outprev + outpitch), _MM_HINT_T0);
			_mm_prefetch((const char*)(outcur + outpitch), _MM_HINT_T0);
			_mm_prefetch((const char*)(gridsample + outpitch), _MM_HINT_T0);

			outcur += outpitch;
			outprev += outpitch;
			gridsample += outpitch;
		}
		gridsample -= outpitch * bh; // restore pointer to only valid first block
	}
}

void WienerFilter::ApplyWiener3D3_AVX2() noexcept
{
	// dft 3d (very short - 3 points)
	constexpr float sin120 = 0.86602540378443864676372317075294f;//sqrtf(3.0f)*0.5f;

	for (int block = start_block; block < blocks; block++)
	{
		for (int h = 0; h < bh; h++) // first half
		{
			__m256 r2 = _mm256_load_ps(outnext[0]);
			__m256 r3 = _mm256_load_ps(outcur[0]);
			for (int w = 0; w < outwidth; w = w + 4) // 
			{
				__m256 r1 = _mm256_load_ps(outprev[w]);
				const __m256 pn4 = _mm256_add_ps(r1, r2); //r,i,r,i
				__m256 fc4 = _mm256_add_ps(pn4, r3); //r,i,r,i

				__m256 d4 = _mm256_sub_ps(r1, r2); //r,i,r,i!
				d4 = _mm256_mul_ps(d4, _mm256_set_ps(1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f));
				d4 = _mm256_mul_ps(d4, _mm256_set1_ps(sin120));
				d4 = _mm256_permute_ps(d4, _MM_SHUFFLE(2, 3, 0, 1));
				r1 = _mm256_mul_ps(pn4, _mm256_set1_ps(0.5f));
				r1 = _mm256_sub_ps(r3, r1);
				__m256 fp4 = _mm256_add_ps(r1, d4);
				__m256 fn4 = _mm256_sub_ps(r1, d4);

				__m256 psdc4 = _mm256_mul_ps(fc4, fc4);
				const __m256 psdp4 = _mm256_mul_ps(fp4, fp4);
				__m256 psdn4 = _mm256_mul_ps(fn4, fn4);

				psdc4 = _mm256_hadd_ps(psdc4, psdp4);
				psdn4 = _mm256_hadd_ps(psdn4, psdn4);

				psdc4 = _mm256_add_ps(psdc4, _mm256_set1_ps(1e-15f));
				psdn4 = _mm256_add_ps(psdn4, _mm256_set1_ps(1e-15f));

				r1 = _mm256_sub_ps(psdc4, _mm256_set1_ps(sigmaSquaredNoiseNormed));
				r3 = _mm256_sub_ps(psdn4, _mm256_set1_ps(sigmaSquaredNoiseNormed));

				r1 = _mm256_div_ps(r1, psdc4);
				r3 = _mm256_div_ps(r3, psdn4);

				r1 = _mm256_max_ps(r1, _mm256_set1_ps(lowlimit));
				r3 = _mm256_max_ps(r3, _mm256_set1_ps(lowlimit));

				fc4 = _mm256_mul_ps(_mm256_permute_ps(r1, _MM_SHUFFLE(1, 1, 0, 0)), fc4);
				fp4 = _mm256_mul_ps(_mm256_permute_ps(r1, _MM_SHUFFLE(3, 3, 2, 2)), fp4);
				fn4 = _mm256_mul_ps(_mm256_permute_ps(r3, _MM_SHUFFLE(1, 1, 0, 0)), fn4);

				r1 = _mm256_add_ps(fc4, fp4);
				r1 = _mm256_add_ps(r1, fn4);
				r2 = _mm256_load_ps(outnext[w + 4]);
				r3 = _mm256_load_ps(outcur[w + 4]);
				r1 = _mm256_mul_ps(r1, _mm256_set1_ps(0.33333333333f));

				_mm256_store_ps(outprev[w], r1);
			}
			_mm_prefetch((const char*)(outprev + outpitch), _MM_HINT_T0);
			_mm_prefetch((const char*)(outcur + outpitch), _MM_HINT_T0);
			_mm_prefetch((const char*)(outnext + outpitch), _MM_HINT_T0);

			outcur += outpitch;
			outprev += outpitch;
			outnext += outpitch;
		}
	}
}

void WienerFilter::ApplyWiener3D3_degrid_AVX2() noexcept
{
	// dft 3d (very short - 3 points)
	constexpr float sin120 = 0.86602540378443864676372317075294f;//sqrtf(3.0f)*0.5f;

	for (int block = start_block; block < blocks; block++)
	{
		const __m256 gridfraction8 = _mm256_set1_ps(degrid * outcur[0][0] / gridsample[0][0]);
		for (int h = 0; h < bh; h++) // first half
		{
			__m256 r2 = _mm256_load_ps(outnext[0]);
			__m256 r3 = _mm256_load_ps(outcur[0]);
			for (int w = 0; w < outwidth; w = w + 4) // 
			{
				__m256 gridcorrection = _mm256_mul_ps(gridfraction8, _mm256_load_ps(gridsample[w])); //gridcorrection
				gridcorrection = _mm256_mul_ps(gridcorrection, _mm256_set1_ps(3.0f));
				__m256 r1 = _mm256_load_ps(outprev[w]);
				const __m256 pn4 = _mm256_add_ps(r1, r2); //r,i,r,i
				__m256 fc4 = _mm256_add_ps(pn4, r3); //r,i,r,i
				fc4 = _mm256_sub_ps(fc4, gridcorrection);

				__m256 d4 = _mm256_sub_ps(r1, r2); //r,i,r,i!
				d4 = _mm256_mul_ps(d4, _mm256_set_ps(1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f));
				d4 = _mm256_mul_ps(d4, _mm256_set1_ps(sin120));
				d4 = _mm256_permute_ps(d4, _MM_SHUFFLE(2, 3, 0, 1));
				r1 = _mm256_fmadd_ps(pn4, _mm256_set1_ps(-0.5f), r3);
				__m256 fp4 = _mm256_add_ps(r1, d4);
				__m256 fn4 = _mm256_sub_ps(r1, d4);

				__m256 psdc4 = _mm256_mul_ps(fc4, fc4);
				const __m256 psdp4 = _mm256_mul_ps(fp4, fp4);
				__m256 psdn4 = _mm256_mul_ps(fn4, fn4);

				psdc4 = _mm256_hadd_ps(psdc4, psdp4);
				psdn4 = _mm256_hadd_ps(psdn4, psdn4);

				psdc4 = _mm256_add_ps(psdc4, _mm256_set1_ps(1e-15f));
				psdn4 = _mm256_add_ps(psdn4, _mm256_set1_ps(1e-15f));

				r1 = _mm256_sub_ps(psdc4, _mm256_set1_ps(sigmaSquaredNoiseNormed));
				r3 = _mm256_sub_ps(psdn4, _mm256_set1_ps(sigmaSquaredNoiseNormed));

				r1 = _mm256_div_ps(r1, psdc4);
				r3 = _mm256_div_ps(r3, psdn4);

				r1 = _mm256_max_ps(r1, _mm256_set1_ps(lowlimit));
				r3 = _mm256_max_ps(r3, _mm256_set1_ps(lowlimit));

				fc4 = _mm256_mul_ps(_mm256_permute_ps(r1, _MM_SHUFFLE(1, 1, 0, 0)), fc4);
				fp4 = _mm256_mul_ps(_mm256_permute_ps(r1, _MM_SHUFFLE(3, 3, 2, 2)), fp4);
				fn4 = _mm256_mul_ps(_mm256_permute_ps(r3, _MM_SHUFFLE(1, 1, 0, 0)), fn4);

				r1 = _mm256_add_ps(fc4, fp4);
				r1 = _mm256_add_ps(r1, fn4);
				r2 = _mm256_load_ps(outnext[w + 4]);
				r3 = _mm256_load_ps(outcur[w + 4]);
				r1 = _mm256_add_ps(r1, gridcorrection);
				r1 = _mm256_mul_ps(r1, _mm256_set1_ps(0.33333333333f));

				_mm256_store_ps(outprev[w], r1);
			}
			_mm_prefetch((const char*)(outprev + outpitch), _MM_HINT_T0);
			_mm_prefetch((const char*)(outcur + outpitch), _MM_HINT_T0);
			_mm_prefetch((const char*)(outnext + outpitch), _MM_HINT_T0);
			_mm_prefetch((const char*)(gridsample + outpitch), _MM_HINT_T0);

			outcur += outpitch;
			outprev += outpitch;
			outnext += outpitch;
			gridsample += outpitch;
		}
		gridsample -= outpitch * bh; // restore pointer to only valid first block
	}
}

void WienerFilter::ApplyWiener3D4_AVX2() noexcept
{
	// dft with 4 points
	const __m256 sigmaSquaredNoiseNormed8 = _mm256_set1_ps(sigmaSquaredNoiseNormed);
	const __m256 lowlimit8 = _mm256_set1_ps(lowlimit);

	for (int block = start_block; block < blocks; block++)
	{
		for (int h = 0; h < bh; h++) // first half
		{
			__m256 r3 = _mm256_load_ps(&outcur[0][0]);
			__m256 r4 = _mm256_load_ps(&outnext[0][0]);
			for (int w = 0; w < outwidth; w = w + 4)
			{
				__m256 r1 = _mm256_load_ps(&outprev2[w][0]);
				__m256 r2 = _mm256_load_ps(&outprev[w][0]);
				//outcur[w][0] - outnext[w][1] - outprev2[w][0] + outprev[w][1]
				//outcur[w][1] + outnext[w][0] - outprev2[w][1] - outprev[w][0]

				__m256 r5 = _mm256_permute_ps(r4, _MM_SHUFFLE(1, 0, 3, 2)); //r5: swapped outnext
				__m256 r6 = _mm256_permute_ps(r2, _MM_SHUFFLE(1, 0, 3, 2)); //r6: swapped outprev

				__m256 r7 = _mm256_addsub_ps(r3, r5); //outcur + outnext
				r7 = _mm256_sub_ps(r7, r1); //-outprev2
				r7 = _mm256_fmadd_ps(r6, _mm256_set_ps(1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f), r7);

				//outcur[w][0] + outnext[w][1] - outprev2[w][0] - outprev[w][1]
				//outcur[w][1] - outnext[w][0] - outprev2[w][1] + outprev[w][0]
				__m256 r8 = _mm256_fmadd_ps(r5, _mm256_set_ps(-1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f), r3);
				r8 = _mm256_sub_ps(r8, r1); //-outprev2
				r8 = _mm256_fmadd_ps(r6, _mm256_set_ps(-1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f), r8);

				r5 = _mm256_add_ps(r1, r2);
				r6 = _mm256_add_ps(r3, r4);
				r5 = _mm256_add_ps(r5, r6); // fcr1, fci1, fcr2, fci2

				r1 = _mm256_sub_ps(r1, r2);
				r2 = _mm256_sub_ps(r3, r4);
				r6 = _mm256_add_ps(r1, r2); // fp2r1, fp2i1, fp2r2, fp2i2
											//r7: fpr1, fpi1, fpr2, fpi2
											//r8: fnr1, fni1, fnr2, fni2
											//r5: fcr1, fci1, fcr2, fci2
											//r6: fp2r1, fp2i1, fp2r2, fp2i2

				r1 = _mm256_mul_ps(r7, r7);
				r2 = _mm256_mul_ps(r8, r8);
				r3 = _mm256_mul_ps(r5, r5);
				r4 = _mm256_mul_ps(r6, r6);

				r1 = _mm256_hadd_ps(r1, r2);
				r2 = _mm256_hadd_ps(r3, r4);

				r1 = _mm256_add_ps(r1, _mm256_set1_ps(1e-15f));
				r2 = _mm256_add_ps(r2, _mm256_set1_ps(1e-15f));
				//r1: psd_fp1, psd_fp2, psd_fn1, psd_fn2
				//r2: psd_fc1, psd_fc2, psd_fp2_1, psd_fp2_2
				r3 = _mm256_sub_ps(r1, sigmaSquaredNoiseNormed8);
				r4 = _mm256_sub_ps(r2, sigmaSquaredNoiseNormed8);
				r1 = _mm256_div_ps(r3, r1);
				r2 = _mm256_div_ps(r4, r2);
				r1 = _mm256_max_ps(r1, lowlimit8);
				r2 = _mm256_max_ps(r2, lowlimit8);
				//r1: wf_fp1, wf_fp2, wf_fn1, wf_fn2
				//r2: wf_fc1, wf_fc2, wf_fp2_1, wf_fp2_2
				//r5: fcr1, fci1, fcr2, fci2 ; fc1, fc2
				//r6: fp2r1, fp2i1, fp2r2, fp2i2 ; fp2_1, fp2_2
				//r7: fpr1, fpi1, fpr2, fpi2 ; fp1, fp2
				//r8: fnr1, fni1, fnr2, fni2 ; fn1, fn2

				r8 = _mm256_mul_ps(r8, _mm256_unpackhi_ps(r1, r1));
				r6 = _mm256_mul_ps(r6, _mm256_unpackhi_ps(r2, r2));

				r6 = _mm256_fmadd_ps(r5, _mm256_unpacklo_ps(r2, r2), r6);
				r8 = _mm256_fmadd_ps(r7, _mm256_unpacklo_ps(r1, r1), r8);
				r3 = _mm256_load_ps(&outcur[w + 4][0]);
				r4 = _mm256_load_ps(&outnext[w + 4][0]);

				r6 = _mm256_add_ps(r6, r8);
				r6 = _mm256_mul_ps(r6, _mm256_set1_ps(0.25f));

				_mm256_store_ps(&outprev2[w][0], r6);
				// Attention! return filtered "out" in "outprev2" to preserve "out" for next step
			}
			_mm_prefetch((const char*)(outprev2 + outpitch), _MM_HINT_T0);
			_mm_prefetch((const char*)(outprev + outpitch), _MM_HINT_T0);
			_mm_prefetch((const char*)(outcur + outpitch), _MM_HINT_T0);
			_mm_prefetch((const char*)(outnext + outpitch), _MM_HINT_T0);

			outcur += outpitch;
			outprev2 += outpitch;
			outprev += outpitch;
			outnext += outpitch;
		}
	}
}

void WienerFilter::ApplyWiener3D4_degrid_AVX2() noexcept
{
	// dft with 4 points
	_mm_prefetch((const char*)outprev2, _MM_HINT_T0);
	_mm_prefetch((const char*)outprev, _MM_HINT_T0);
	_mm_prefetch((const char*)outcur, _MM_HINT_T0);
	_mm_prefetch((const char*)outnext, _MM_HINT_T0);


	const __m256 sigmaSquaredNoiseNormed8 = _mm256_set1_ps(sigmaSquaredNoiseNormed);
	const __m256 lowlimit8 = _mm256_set1_ps(lowlimit);

	for (int block = start_block; block < blocks; block++)
	{
		const __m256 gridfraction8 = _mm256_set1_ps(degrid * outcur[0][0] / gridsample[0][0]);

		for (int h = 0; h < bh; h++) // first half
		{
			__m256 r3 = _mm256_load_ps(&outcur[0][0]);
			__m256 r4 = _mm256_load_ps(&outnext[0][0]);
			for (int w = 0; w < outwidth; w = w + 4)
			{
				const __m256 gridcorrection = _mm256_mul_ps(_mm256_mul_ps(_mm256_load_ps(&gridsample[w][0]), gridfraction8), _mm256_set1_ps(4));
				__m256 r1 = _mm256_load_ps(&outprev2[w][0]);
				__m256 r2 = _mm256_load_ps(&outprev[w][0]);
				//outcur[w][0] - outnext[w][1] - outprev2[w][0] + outprev[w][1]
				//outcur[w][1] + outnext[w][0] - outprev2[w][1] - outprev[w][0]

				__m256 r5 = _mm256_permute_ps(r4, _MM_SHUFFLE(1, 0, 3, 2)); //r5: swapped outnext
				__m256 r6 = _mm256_permute_ps(r2, _MM_SHUFFLE(1, 0, 3, 2)); //r6: swapped outprev

				__m256 r7 = _mm256_addsub_ps(r3, r5); //outcur + outnext
				r7 = _mm256_sub_ps(r7, r1); //-outprev2
				r7 = _mm256_fmadd_ps(r6, _mm256_set_ps(1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f), r7);

				//outcur[w][0] + outnext[w][1] - outprev2[w][0] - outprev[w][1]
				//outcur[w][1] - outnext[w][0] - outprev2[w][1] + outprev[w][0]
				__m256 r8 = _mm256_fmadd_ps(r5, _mm256_set_ps(-1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f), r3);
				r8 = _mm256_sub_ps(r8, r1); //-outprev2
				r8 = _mm256_fmadd_ps(r6, _mm256_set_ps(-1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f), r8);

				r5 = _mm256_add_ps(r1, r2);
				r6 = _mm256_add_ps(r3, r4);
				r5 = _mm256_add_ps(r5, r6); // fcr1, fci1, fcr2, fci2
				r5 = _mm256_sub_ps(r5, gridcorrection);

				r1 = _mm256_sub_ps(r1, r2);
				r2 = _mm256_sub_ps(r3, r4);
				r6 = _mm256_add_ps(r1, r2); // fp2r1, fp2i1, fp2r2, fp2i2
				//r7: fpr1, fpi1, fpr2, fpi2
				//r8: fnr1, fni1, fnr2, fni2
				//r5: fcr1, fci1, fcr2, fci2
				//r6: fp2r1, fp2i1, fp2r2, fp2i2

				r1 = _mm256_mul_ps(r7, r7);
				r2 = _mm256_mul_ps(r8, r8);
				r3 = _mm256_mul_ps(r5, r5);
				r4 = _mm256_mul_ps(r6, r6);

				r1 = _mm256_hadd_ps(r1, r2);
				r2 = _mm256_hadd_ps(r3, r4);

				r1 = _mm256_add_ps(r1, _mm256_set1_ps(1e-15f));
				r2 = _mm256_add_ps(r2, _mm256_set1_ps(1e-15f));
				//r1: psd_fp1, psd_fp2, psd_fn1, psd_fn2
				//r2: psd_fc1, psd_fc2, psd_fp2_1, psd_fp2_2
				r3 = _mm256_sub_ps(r1, sigmaSquaredNoiseNormed8);
				r4 = _mm256_sub_ps(r2, sigmaSquaredNoiseNormed8);
				r1 = _mm256_div_ps(r3, r1);
				r2 = _mm256_div_ps(r4, r2);
				r1 = _mm256_max_ps(r1, lowlimit8);
				r2 = _mm256_max_ps(r2, lowlimit8);
				//r1: wf_fp1, wf_fp2, wf_fn1, wf_fn2
				//r2: wf_fc1, wf_fc2, wf_fp2_1, wf_fp2_2
				//r5: fcr1, fci1, fcr2, fci2 ; fc1, fc2
				//r6: fp2r1, fp2i1, fp2r2, fp2i2 ; fp2_1, fp2_2
				//r7: fpr1, fpi1, fpr2, fpi2 ; fp1, fp2
				//r8: fnr1, fni1, fnr2, fni2 ; fn1, fn2

				r8 = _mm256_mul_ps(r8, _mm256_unpackhi_ps(r1, r1));
				r6 = _mm256_mul_ps(r6, _mm256_unpackhi_ps(r2, r2));

				r6 = _mm256_fmadd_ps(r5, _mm256_unpacklo_ps(r2, r2), r6);
				r8 = _mm256_fmadd_ps(r7, _mm256_unpacklo_ps(r1, r1), r8);
				r3 = _mm256_load_ps(&outcur[w + 4][0]);
				r4 = _mm256_load_ps(&outnext[w + 4][0]);

				r6 = _mm256_add_ps(r6, r8);
				r6 = _mm256_mul_ps(_mm256_add_ps(r6, gridcorrection), _mm256_set1_ps(0.25f));

				_mm256_store_ps(&outprev2[w][0], r6);
				// Attention! return filtered "out" in "outprev2" to preserve "out" for next step
			}
			_mm_prefetch((const char*)(outprev2 + outpitch), _MM_HINT_T0);
			_mm_prefetch((const char*)(outprev + outpitch), _MM_HINT_T0);
			_mm_prefetch((const char*)(outcur + outpitch), _MM_HINT_T0);
			_mm_prefetch((const char*)(outnext + outpitch), _MM_HINT_T0);
			_mm_prefetch((const char*)(gridsample + outpitch), _MM_HINT_T0);

			outcur += outpitch;
			outprev2 += outpitch;
			outprev += outpitch;
			outnext += outpitch;
			gridsample += outpitch;
		}
		gridsample -= outpitch * bh; // restore pointer to only valid first block
	}
}

void WienerFilter::ApplyWiener3D5_AVX2() noexcept
{
	// dft with 5 points
	const __m256 sincos72 = _mm256_set_ps(0.30901699437494742410229341718282f, 0.95105651629515357211643933337938, 0.30901699437494742410229341718282f, 0.95105651629515357211643933337938,
		0.30901699437494742410229341718282f, 0.95105651629515357211643933337938, 0.30901699437494742410229341718282f, 0.95105651629515357211643933337938);
	const __m256 cossin72 = _mm256_set_ps(0.95105651629515357211643933337938, 0.30901699437494742410229341718282f, 0.95105651629515357211643933337938, 0.30901699437494742410229341718282f,
		0.95105651629515357211643933337938, 0.30901699437494742410229341718282f, 0.95105651629515357211643933337938, 0.30901699437494742410229341718282f);
	const __m256 sincos144 = _mm256_set_ps(-0.80901699437494742410229341718282f, 0.58778525229247312916870595463907f, -0.80901699437494742410229341718282f, 0.58778525229247312916870595463907f,
		-0.80901699437494742410229341718282f, 0.58778525229247312916870595463907f, -0.80901699437494742410229341718282f, 0.58778525229247312916870595463907f);
	const __m256 cossin144 = _mm256_set_ps(0.58778525229247312916870595463907f, -0.80901699437494742410229341718282f, 0.58778525229247312916870595463907f, -0.80901699437494742410229341718282f,
		0.58778525229247312916870595463907f, -0.80901699437494742410229341718282f, 0.58778525229247312916870595463907f, -0.80901699437494742410229341718282f);

	for (int block = start_block; block < blocks; block++)
	{
		for (int h = 0; h < bh; h++) // first half
		{
			__m256 r3 = _mm256_load_ps(outprev[0]);
			__m256 r4 = _mm256_load_ps(outnext[0]);
			for (int w = 0; w < outwidth; w = w + 4) // 
			{
				__m256 r1 = _mm256_load_ps(outprev2[w]);
				__m256 r2 = _mm256_load_ps(outnext2[w]);
				const __m256 r5 = _mm256_load_ps(outcur[w]);

				__m256 r6 = _mm256_fmadd_ps(r1, _mm256_set_ps(-1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f), r2); //sum, dif, sum, dif
				__m256 r7 = _mm256_fmadd_ps(r4, _mm256_set_ps(-1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f), r3); //sum, dif, sum, dif
				r6 = _mm256_mul_ps(r6, cossin72);
				r7 = _mm256_mul_ps(r7, cossin144);
				r7 = _mm256_add_ps(r6, r7);
				r7 = _mm256_add_ps(r7, _mm256_andnot_ps(_mm256_set_ps(~0, 0.0f, ~0, 0.0f, ~0, 0.0f, ~0, 0.0f), r5));
				r6 = _mm256_permute_ps(r7, _MM_SHUFFLE(0, 3, 0, 1));
				__m256 fp2r = _mm256_add_ps(r7, r6);
				__m256 fn2r = _mm256_sub_ps(r7, r6);

				r6 = _mm256_fmadd_ps(r2, _mm256_set_ps(1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f), r1); //dif, sum, dif, sum
				r7 = _mm256_fmadd_ps(r3, _mm256_set_ps(1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f), r4); //dif, sum, dif, sum
				r6 = _mm256_mul_ps(r6, sincos72);
				r7 = _mm256_mul_ps(r7, sincos144);
				r7 = _mm256_add_ps(r6, r7);
				r7 = _mm256_add_ps(r7, _mm256_andnot_ps(_mm256_set_ps(0.0f, ~0, 0.0f, ~0, 0.0f, ~0, 0.0f, ~0), r5));
				r6 = _mm256_permute_ps(r7, _MM_SHUFFLE(0, 3, 0, 1)); //dif, sum -> r6 == sum!
				const __m256 fp2i = _mm256_add_ps(r7, r6);
				const __m256 fn2i = _mm256_sub_ps(r6, r7);

				r6 = _mm256_fmadd_ps(r2, _mm256_set_ps(-1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f), r1); //sum, dif, sum, dif
				r7 = _mm256_fmadd_ps(r4, _mm256_set_ps(-1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f), r3); //sum, dif, sum, dif
				r6 = _mm256_mul_ps(r6, cossin144);
				r7 = _mm256_mul_ps(r7, cossin72);
				r7 = _mm256_add_ps(r6, r7);
				r7 = _mm256_add_ps(r7, _mm256_andnot_ps(_mm256_set_ps(~0, 0.0f, ~0, 0.0f, ~0, 0.0f, ~0, 0.0f), r5));
				r6 = _mm256_permute_ps(r7, _MM_SHUFFLE(0, 3, 0, 1));
				__m256 fpr = _mm256_add_ps(r7, r6);
				__m256 fnr = _mm256_sub_ps(r7, r6);

				r6 = _mm256_fmadd_ps(r1, _mm256_set_ps(1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f), r2); //dif, sum, dif, sum
				r7 = _mm256_fmadd_ps(r3, _mm256_set_ps(1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f), r4); //dif, sum, dif, sum
				r6 = _mm256_mul_ps(r6, sincos144);
				r7 = _mm256_mul_ps(r7, sincos72);
				r7 = _mm256_add_ps(r6, r7);
				r7 = _mm256_add_ps(r7, _mm256_andnot_ps(_mm256_set_ps(0.0f, ~0, 0.0f, ~0, 0.0f, ~0, 0.0f, ~0), r5));
				r6 = _mm256_permute_ps(r7, _MM_SHUFFLE(0, 3, 0, 1)); //dif, sum -> r6 == sum!
				const __m256 fpi = _mm256_add_ps(r7, r6);
				const __m256 fni = _mm256_sub_ps(r6, r7);

				r6 = _mm256_add_ps(r1, r2);
				r7 = _mm256_add_ps(r3, r4);
				r6 = _mm256_add_ps(r5, r6);
				__m256 fc = _mm256_add_ps(r6, r7);

				__m256 psd = _mm256_mul_ps(fc, fc);
				r1 = _mm256_permute_ps(psd, _MM_SHUFFLE(0, 3, 0, 1)); //psd,-,psd,-
				psd = _mm256_add_ps(psd, r1);
				psd = _mm256_add_ps(psd, _mm256_set1_ps(1e-15f));
				r1 = _mm256_sub_ps(psd, _mm256_set1_ps(sigmaSquaredNoiseNormed));
				r1 = _mm256_div_ps(r1, psd);
				r1 = _mm256_max_ps(r1, _mm256_set1_ps(lowlimit));
				r1 = _mm256_moveldup_ps(r1);
				fc = _mm256_mul_ps(r1, fc);

				r1 = _mm256_mul_ps(fp2r, fp2r);
				r2 = _mm256_mul_ps(fp2i, fp2i);
				fp2r = _mm256_shuffle_ps(fp2r, fp2i, _MM_SHUFFLE(2, 0, 2, 0));
				fp2r = _mm256_permute_ps(fp2r, _MM_SHUFFLE(3, 1, 2, 0));
				psd = _mm256_add_ps(r1, r2);
				psd = _mm256_add_ps(psd, _mm256_set1_ps(1e-15f));
				r1 = _mm256_sub_ps(psd, _mm256_set1_ps(sigmaSquaredNoiseNormed));
				r1 = _mm256_div_ps(r1, psd);
				r1 = _mm256_max_ps(r1, _mm256_set1_ps(lowlimit));
				r1 = _mm256_moveldup_ps(r1);
				fp2r = _mm256_mul_ps(r1, fp2r);

				r1 = _mm256_mul_ps(fpr, fpr);
				r2 = _mm256_mul_ps(fpi, fpi);
				fpr = _mm256_shuffle_ps(fpr, fpi, _MM_SHUFFLE(2, 0, 2, 0));
				fpr = _mm256_permute_ps(fpr, _MM_SHUFFLE(3, 1, 2, 0));
				psd = _mm256_add_ps(r1, r2);
				psd = _mm256_add_ps(psd, _mm256_set1_ps(1e-15f));
				r1 = _mm256_sub_ps(psd, _mm256_set1_ps(sigmaSquaredNoiseNormed));
				r1 = _mm256_div_ps(r1, psd);
				r1 = _mm256_max_ps(r1, _mm256_set1_ps(lowlimit));
				r1 = _mm256_moveldup_ps(r1);
				fpr = _mm256_mul_ps(r1, fpr);

				r1 = _mm256_mul_ps(fnr, fnr);
				r2 = _mm256_mul_ps(fni, fni);
				fnr = _mm256_shuffle_ps(fnr, fni, _MM_SHUFFLE(2, 0, 2, 0));
				fnr = _mm256_permute_ps(fnr, _MM_SHUFFLE(3, 1, 2, 0));
				psd = _mm256_add_ps(r1, r2);
				psd = _mm256_add_ps(psd, _mm256_set1_ps(1e-15f));
				r1 = _mm256_sub_ps(psd, _mm256_set1_ps(sigmaSquaredNoiseNormed));
				r1 = _mm256_div_ps(r1, psd);
				r1 = _mm256_max_ps(r1, _mm256_set1_ps(lowlimit));
				r1 = _mm256_moveldup_ps(r1);
				fnr = _mm256_mul_ps(r1, fnr);

				r1 = _mm256_mul_ps(fn2r, fn2r);
				r2 = _mm256_mul_ps(fn2i, fn2i);
				fn2r = _mm256_shuffle_ps(fn2r, fn2i, _MM_SHUFFLE(2, 0, 2, 0));
				fn2r = _mm256_permute_ps(fn2r, _MM_SHUFFLE(3, 1, 2, 0));
				psd = _mm256_add_ps(r1, r2);
				psd = _mm256_add_ps(psd, _mm256_set1_ps(1e-15f));
				r1 = _mm256_sub_ps(psd, _mm256_set1_ps(sigmaSquaredNoiseNormed));
				r1 = _mm256_div_ps(r1, psd);
				r1 = _mm256_max_ps(r1, _mm256_set1_ps(lowlimit));
				r1 = _mm256_moveldup_ps(r1);
				fn2r = _mm256_mul_ps(r1, fn2r);

				r1 = _mm256_add_ps(fp2r, fpr);
				r2 = _mm256_add_ps(fc, fnr);
				r1 = _mm256_add_ps(r1, fn2r);
				r1 = _mm256_add_ps(r1, r2);
				r3 = _mm256_load_ps(outprev[w + 4]);
				r4 = _mm256_load_ps(outnext[w + 4]);
				r1 = _mm256_mul_ps(r1, _mm256_set1_ps(0.2f));
				_mm256_store_ps(outprev2[w], r1);
			}
			_mm_prefetch((const char*)(outprev2 + outpitch), _MM_HINT_T0);
			_mm_prefetch((const char*)(outprev + outpitch), _MM_HINT_T0);
			_mm_prefetch((const char*)(outcur + outpitch), _MM_HINT_T0);
			_mm_prefetch((const char*)(outnext + outpitch), _MM_HINT_T0);
			_mm_prefetch((const char*)(outnext2 + outpitch), _MM_HINT_T0);

			outcur += outpitch;
			outprev2 += outpitch;
			outprev += outpitch;
			outnext += outpitch;
			outnext2 += outpitch;
		}
	}
}

void WienerFilter::ApplyWiener3D5_degrid_AVX2() noexcept
{
	// dft with 5 points
	const __m256 sincos72 = _mm256_set_ps(0.30901699437494742410229341718282f, 0.95105651629515357211643933337938, 0.30901699437494742410229341718282f, 0.95105651629515357211643933337938,
		0.30901699437494742410229341718282f, 0.95105651629515357211643933337938, 0.30901699437494742410229341718282f, 0.95105651629515357211643933337938);
	const __m256 cossin72 = _mm256_set_ps(0.95105651629515357211643933337938, 0.30901699437494742410229341718282f, 0.95105651629515357211643933337938, 0.30901699437494742410229341718282f,
		0.95105651629515357211643933337938, 0.30901699437494742410229341718282f, 0.95105651629515357211643933337938, 0.30901699437494742410229341718282f);
	const __m256 sincos144 = _mm256_set_ps(-0.80901699437494742410229341718282f, 0.58778525229247312916870595463907f, -0.80901699437494742410229341718282f, 0.58778525229247312916870595463907f,
		-0.80901699437494742410229341718282f, 0.58778525229247312916870595463907f, -0.80901699437494742410229341718282f, 0.58778525229247312916870595463907f);
	const __m256 cossin144 = _mm256_set_ps(0.58778525229247312916870595463907f, -0.80901699437494742410229341718282f, 0.58778525229247312916870595463907f, -0.80901699437494742410229341718282f,
		0.58778525229247312916870595463907f, -0.80901699437494742410229341718282f, 0.58778525229247312916870595463907f, -0.80901699437494742410229341718282f);

	for (int block = start_block; block < blocks; block++)
	{
		const __m256 gridfraction8 = _mm256_set1_ps(degrid * outcur[0][0] / gridsample[0][0]);
		for (int h = 0; h < bh; h++) // first half
		{
			__m256 r3 = _mm256_load_ps(outprev[0]);
			__m256 r4 = _mm256_load_ps(outnext[0]);
			for (int w = 0; w < outwidth; w = w + 4) // 
			{
				__m256 gridcorrection = _mm256_mul_ps(_mm256_load_ps(&gridsample[w][0]), _mm256_set1_ps(degrid * outcur[0][0] / gridsample[0][0]));
				gridcorrection = _mm256_mul_ps(gridcorrection, _mm256_set1_ps(5));
				__m256 r1 = _mm256_load_ps(outprev2[w]);
				__m256 r2 = _mm256_load_ps(outnext2[w]);
				const __m256 r5 = _mm256_load_ps(outcur[w]);

				__m256 r6 = _mm256_fmadd_ps(r1, _mm256_set_ps(-1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f), r2); //sum, dif, sum, dif
				__m256 r7 = _mm256_fmadd_ps(r4, _mm256_set_ps(-1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f), r3); //sum, dif, sum, dif
				r6 = _mm256_mul_ps(r6, cossin72);
				r7 = _mm256_mul_ps(r7, cossin144);
				r7 = _mm256_add_ps(r6, r7);
				r7 = _mm256_add_ps(r7, _mm256_andnot_ps(_mm256_set_ps(~0, 0.0f, ~0, 0.0f, ~0, 0.0f, ~0, 0.0f), r5)); //bitwise AND would be faster, but how to use it?
				r6 = _mm256_permute_ps(r7, _MM_SHUFFLE(0, 3, 0, 1));
				__m256 fp2r = _mm256_add_ps(r7, r6);
				__m256 fn2r = _mm256_sub_ps(r7, r6);

				r6 = _mm256_fmadd_ps(r2, _mm256_set_ps(1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f), r1); //dif, sum, dif, sum
				r7 = _mm256_fmadd_ps(r3, _mm256_set_ps(1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f), r4); //dif, sum, dif, sum
				r6 = _mm256_mul_ps(r6, sincos72);
				r7 = _mm256_mul_ps(r7, sincos144);
				r7 = _mm256_add_ps(r6, r7);
				r7 = _mm256_add_ps(r7, _mm256_andnot_ps(_mm256_set_ps(0.0f, ~0, 0.0f, ~0, 0.0f, ~0, 0.0f, ~0), r5)); //bitwise AND would be faster, but how to use it?
				r6 = _mm256_permute_ps(r7, _MM_SHUFFLE(0, 3, 0, 1)); //dif, sum -> r6 == sum!
				const __m256 fp2i = _mm256_add_ps(r7, r6);
				const __m256 fn2i = _mm256_sub_ps(r6, r7);

				r6 = _mm256_fmadd_ps(r2, _mm256_set_ps(-1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f), r1); //sum, dif, sum, dif
				r7 = _mm256_fmadd_ps(r4, _mm256_set_ps(-1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f), r3); //sum, dif, sum, dif
				r6 = _mm256_mul_ps(r6, cossin144);
				r7 = _mm256_mul_ps(r7, cossin72);
				r7 = _mm256_add_ps(r6, r7);
				r7 = _mm256_add_ps(r7, _mm256_andnot_ps(_mm256_set_ps(~0, 0.0f, ~0, 0.0f, ~0, 0.0f, ~0, 0.0f), r5)); //bitwise AND would be faster, but how to use it?
				r6 = _mm256_permute_ps(r7, _MM_SHUFFLE(0, 3, 0, 1));
				__m256 fpr = _mm256_add_ps(r7, r6);
				__m256 fnr = _mm256_sub_ps(r7, r6);

				r6 = _mm256_fmadd_ps(r1, _mm256_set_ps(1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f), r2); //dif, sum, dif, sum
				r7 = _mm256_fmadd_ps(r3, _mm256_set_ps(1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f), r4); //dif, sum, dif, sum
				r6 = _mm256_mul_ps(r6, sincos144);
				r7 = _mm256_mul_ps(r7, sincos72);
				r7 = _mm256_add_ps(r6, r7);
				r7 = _mm256_add_ps(r7, _mm256_andnot_ps(_mm256_set_ps(0.0f, ~0, 0.0f, ~0, 0.0f, ~0, 0.0f, ~0), r5)); //bitwise AND would be faster, but how to use it?
				r6 = _mm256_permute_ps(r7, _MM_SHUFFLE(0, 3, 0, 1)); //dif, sum -> r6 == sum!
				const __m256 fpi = _mm256_add_ps(r7, r6);
				const __m256 fni = _mm256_sub_ps(r6, r7);

				r6 = _mm256_add_ps(r1, r2);
				r7 = _mm256_add_ps(r3, r4);
				r6 = _mm256_add_ps(r5, r6);
				__m256 fc = _mm256_add_ps(r6, r7);
				fc = _mm256_sub_ps(fc, gridcorrection);

				__m256 psd = _mm256_mul_ps(fc, fc);
				r1 = _mm256_permute_ps(psd, _MM_SHUFFLE(0, 3, 0, 1)); //psd,-,psd,-
				psd = _mm256_add_ps(psd, r1);
				psd = _mm256_add_ps(psd, _mm256_set1_ps(1e-15f));
				r1 = _mm256_sub_ps(psd, _mm256_set1_ps(sigmaSquaredNoiseNormed));
				r1 = _mm256_div_ps(r1, psd);
				r1 = _mm256_max_ps(r1, _mm256_set1_ps(lowlimit));
				r1 = _mm256_moveldup_ps(r1);
				fc = _mm256_mul_ps(r1, fc);

				r1 = _mm256_mul_ps(fp2r, fp2r);
				r2 = _mm256_mul_ps(fp2i, fp2i);
				fp2r = _mm256_shuffle_ps(fp2r, fp2i, _MM_SHUFFLE(2, 0, 2, 0));
				fp2r = _mm256_permute_ps(fp2r, _MM_SHUFFLE(3, 1, 2, 0));
				psd = _mm256_add_ps(r1, r2);
				psd = _mm256_add_ps(psd, _mm256_set1_ps(1e-15f));
				r1 = _mm256_sub_ps(psd, _mm256_set1_ps(sigmaSquaredNoiseNormed));
				r1 = _mm256_div_ps(r1, psd);
				r1 = _mm256_max_ps(r1, _mm256_set1_ps(lowlimit));
				r1 = _mm256_moveldup_ps(r1);
				fp2r = _mm256_mul_ps(r1, fp2r);

				r1 = _mm256_mul_ps(fpr, fpr);
				r2 = _mm256_mul_ps(fpi, fpi);
				fpr = _mm256_shuffle_ps(fpr, fpi, _MM_SHUFFLE(2, 0, 2, 0));
				fpr = _mm256_permute_ps(fpr, _MM_SHUFFLE(3, 1, 2, 0));
				psd = _mm256_add_ps(r1, r2);
				psd = _mm256_add_ps(psd, _mm256_set1_ps(1e-15f));
				r1 = _mm256_sub_ps(psd, _mm256_set1_ps(sigmaSquaredNoiseNormed));
				r1 = _mm256_div_ps(r1, psd);
				r1 = _mm256_max_ps(r1, _mm256_set1_ps(lowlimit));
				r1 = _mm256_moveldup_ps(r1);
				fpr = _mm256_mul_ps(r1, fpr);

				r1 = _mm256_mul_ps(fnr, fnr);
				r2 = _mm256_mul_ps(fni, fni);
				fnr = _mm256_shuffle_ps(fnr, fni, _MM_SHUFFLE(2, 0, 2, 0));
				fnr = _mm256_permute_ps(fnr, _MM_SHUFFLE(3, 1, 2, 0));
				psd = _mm256_add_ps(r1, r2);
				psd = _mm256_add_ps(psd, _mm256_set1_ps(1e-15f));
				r1 = _mm256_sub_ps(psd, _mm256_set1_ps(sigmaSquaredNoiseNormed));
				r1 = _mm256_div_ps(r1, psd);
				r1 = _mm256_max_ps(r1, _mm256_set1_ps(lowlimit));
				r1 = _mm256_moveldup_ps(r1);
				fnr = _mm256_mul_ps(r1, fnr);

				r1 = _mm256_mul_ps(fn2r, fn2r);
				r2 = _mm256_mul_ps(fn2i, fn2i);
				fn2r = _mm256_shuffle_ps(fn2r, fn2i, _MM_SHUFFLE(2, 0, 2, 0));
				fn2r = _mm256_permute_ps(fn2r, _MM_SHUFFLE(3, 1, 2, 0));
				psd = _mm256_add_ps(r1, r2);
				psd = _mm256_add_ps(psd, _mm256_set1_ps(1e-15f));
				r1 = _mm256_sub_ps(psd, _mm256_set1_ps(sigmaSquaredNoiseNormed));
				r1 = _mm256_div_ps(r1, psd);
				r1 = _mm256_max_ps(r1, _mm256_set1_ps(lowlimit));
				r1 = _mm256_moveldup_ps(r1);
				fn2r = _mm256_mul_ps(r1, fn2r);

				r1 = _mm256_add_ps(fp2r, fpr);
				r2 = _mm256_add_ps(fc, fnr);
				r1 = _mm256_add_ps(r1, fn2r);
				r1 = _mm256_add_ps(r1, r2);
				r3 = _mm256_load_ps(outprev[w + 4]);
				r4 = _mm256_load_ps(outnext[w + 4]);
				r1 = _mm256_add_ps(r1, gridcorrection);
				r1 = _mm256_mul_ps(r1, _mm256_set1_ps(0.2f));
				_mm256_store_ps(outprev2[w], r1);
			}
			_mm_prefetch((const char*)(outprev2 + outpitch), _MM_HINT_T0);
			_mm_prefetch((const char*)(outprev + outpitch), _MM_HINT_T0);
			_mm_prefetch((const char*)(outcur + outpitch), _MM_HINT_T0);
			_mm_prefetch((const char*)(outnext + outpitch), _MM_HINT_T0);
			_mm_prefetch((const char*)(outnext2 + outpitch), _MM_HINT_T0);
			_mm_prefetch((const char*)(gridsample + outpitch), _MM_HINT_T0);

			outcur += outpitch;
			outprev2 += outpitch;
			outprev += outpitch;
			outnext += outpitch;
			outnext2 += outpitch;
			gridsample += outpitch;
		}
		gridsample -= outpitch * bh; // restore pointer to only valid first block
	}
}