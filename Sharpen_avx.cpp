//
//	FFT3DFilter plugin for Avisynth 2.6 - 3D Frequency Domain filter
//  AVX version of filtering functions
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
#include "Sharpen.h"

// since v1.7 we use outpitch instead of outwidth

//-------------------------------------------------------------------------------------------
//
void SharpenFilter::Sharpen_AVX() noexcept
{
	if (sharpen != 0 && dehalo == 0)
	{
		const __m256 sharpen8 = _mm256_set1_ps(sharpen);
		const __m256 sigmaSquaredSharpenMax8 = _mm256_set1_ps(sigmaSquaredSharpenMax);
		const __m256 sigmaSquaredSharpenMin8 = _mm256_set1_ps(sigmaSquaredSharpenMin);

		for (int block = start_block; block < blocks; block++)
		{
			for (int h = 0; h < bh; h++) // middle
			{
				for (int w = 0; w < outwidth; w = w + 4)
				{
					__m256 cur = _mm256_load_ps(outcur[w]);
					__m256 r1 = _mm256_mul_ps(cur, cur);
					__m256 psd4 = _mm256_hadd_ps(r1, r1); //psd1, psd2, psd1, psd2

					__m256 r2 = _mm256_mul_ps(sharpen8, _mm256_broadcast_ps((__m128*)&wsharpen[w]));
					__m256 r3 = _mm256_mul_ps(psd4, sigmaSquaredSharpenMax8);
					__m256 r4 = _mm256_add_ps(psd4, sigmaSquaredSharpenMin8);
					__m256 r5 = _mm256_add_ps(psd4, sigmaSquaredSharpenMax8);
					r4 = _mm256_mul_ps(r4, r5);
					r3 = _mm256_div_ps(r3, r4);
					r3 = _mm256_sqrt_ps(r3);
					r2 = _mm256_mul_ps(r2, r3);
					r2 = _mm256_add_ps(r2, _mm256_set1_ps(1.0f));
					r2 = _mm256_permutevar_ps(r2, _mm256_set_epi32(3, 3, 2, 2, 1, 1, 0, 0));
					_mm256_store_ps(outcur[w], _mm256_mul_ps(r2, cur));
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
		const __m256 dehalo8 = _mm256_set1_ps(dehalo);
		const __m256 ht2n8 = _mm256_set1_ps(ht2n);

		for (int block = start_block; block < blocks; block++)
		{
			for (int h = 0; h < bh; h++) // middle
			{
				for (int w = 0; w < outwidth; w = w + 4)
				{
					__m256 cur = _mm256_load_ps(outcur[w]);
					__m256 r1 = _mm256_mul_ps(cur, cur);
					__m256 psd4 = _mm256_hadd_ps(r1, r1); //psd1, -, psd2, -

					__m256 r3 = _mm256_add_ps(psd4, ht2n8);
					__m256 r4 = _mm256_mul_ps(psd4, dehalo8);
					r4 = _mm256_mul_ps(r4, _mm256_broadcast_ps((__m128*)&wdehalo[w]));
					r4 = _mm256_add_ps(r3, r4);
					r3 = _mm256_div_ps(r3, r4);

					__m256 r2 = _mm256_permutevar_ps(r3, _mm256_set_epi32(3, 3, 2, 2, 1, 1, 0, 0));
					_mm256_store_ps(outcur[w], _mm256_mul_ps(r2, cur));
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
		const __m256 sharpen8 = _mm256_set1_ps(sharpen);
		const __m256 sigmaSquaredSharpenMax8 = _mm256_set1_ps(sigmaSquaredSharpenMax);
		const __m256 sigmaSquaredSharpenMin8 = _mm256_set1_ps(sigmaSquaredSharpenMin);

		const __m256 dehalo8 = _mm256_set1_ps(dehalo);
		const __m256 ht2n8 = _mm256_set1_ps(ht2n);

		for (int block = start_block; block < blocks; block++)
		{
			for (int h = 0; h < bh; h++) // middle
			{
				for (int w = 0; w < outwidth; w = w + 4)
				{
					__m256 cur = _mm256_load_ps(outcur[w]);
					__m256 r1 = _mm256_mul_ps(cur, cur);
					__m256 psd4 = _mm256_hadd_ps(r1, r1); //psd1, -, psd2, -, psd3, -, psd4, -

					__m256 r2 = _mm256_mul_ps(sharpen8, _mm256_broadcast_ps((__m128*) &wsharpen[w]));
					__m256 r3 = _mm256_mul_ps(psd4, sigmaSquaredSharpenMax8);
					__m256 r4 = _mm256_add_ps(psd4, sigmaSquaredSharpenMin8);
					__m256 r5 = _mm256_add_ps(psd4, sigmaSquaredSharpenMax8);
					r4 = _mm256_mul_ps(r4, r5);
					r3 = _mm256_div_ps(r3, r4);
					r3 = _mm256_sqrt_ps(r3);
					r2 = _mm256_mul_ps(r2, r3);
					r2 = _mm256_add_ps(r2, _mm256_set1_ps(1.0f));

					r3 = _mm256_add_ps(psd4, ht2n8);
					r4 = _mm256_mul_ps(psd4, dehalo8);
					r4 = _mm256_mul_ps(r4, _mm256_broadcast_ps((__m128*)&wdehalo[w]));
					r4 = _mm256_add_ps(r3, r4);
					r3 = _mm256_div_ps(r3, r4);

					r2 = _mm256_mul_ps(r2, r3);

					r2 = _mm256_permutevar_ps(r2, _mm256_set_epi32(3, 3, 2, 2, 1, 1, 0, 0));
					_mm256_store_ps(outcur[w], _mm256_mul_ps(r2, cur));
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
//-------------------------------------------------------------------------------------------
//
void SharpenFilter::Sharpen_degrid_AVX() noexcept
{
	if (sharpen != 0 && dehalo == 0)
	{
		const __m256 sharpen8 = _mm256_set1_ps(sharpen);
		const __m256 sigmaSquaredSharpenMax8 = _mm256_set1_ps(sigmaSquaredSharpenMax);
		const __m256 sigmaSquaredSharpenMin8 = _mm256_set1_ps(sigmaSquaredSharpenMin);

		for (int block = start_block; block < blocks; block++)
		{
			const __m256 gridfraction8 = _mm256_set1_ps(degrid * outcur[0][0] / gridsample[0][0]);
			for (int h = 0; h < bh; h++) // middle
			{
				for (int w = 0; w < outwidth; w = w + 4)
				{
					const __m256 gridcorrection = _mm256_mul_ps(gridfraction8, _mm256_load_ps(gridsample[w])); //gridcorrection
					__m256 cur = _mm256_load_ps(outcur[w]);
					cur = _mm256_sub_ps(cur, gridcorrection);
					__m256 r1 = _mm256_mul_ps(cur, cur);
					__m256 psd4 = _mm256_hadd_ps(r1, r1); //psd1, -, psd2, -

					__m256 sharp4 = _mm256_broadcast_ps((__m128*) &wsharpen[w]);
					__m256 r2 = _mm256_mul_ps(sharpen8, sharp4);
					__m256 r3 = _mm256_mul_ps(psd4, sigmaSquaredSharpenMax8);
					__m256 r4 = _mm256_add_ps(psd4, sigmaSquaredSharpenMin8);
					__m256 r5 = _mm256_add_ps(psd4, sigmaSquaredSharpenMax8);
					r4 = _mm256_mul_ps(r4, r5);
					r3 = _mm256_div_ps(r3, r4);
					r3 = _mm256_sqrt_ps(r3);
					r2 = _mm256_mul_ps(r2, r3);
					r2 = _mm256_add_ps(r2, _mm256_set1_ps(1.0f));

					r2 = _mm256_permutevar_ps(r2, _mm256_set_epi32(3, 3, 2, 2, 1, 1, 0, 0));
					r2 = _mm256_mul_ps(r2, cur);
					_mm256_store_ps(outcur[w], _mm256_add_ps(r2, gridcorrection));
				}
				_mm_prefetch((const char*)(outcur + outpitch), _MM_HINT_T0);
				_mm_prefetch((const char*)(gridsample + outpitch), _MM_HINT_T0);
				_mm_prefetch((const char*)(wsharpen + outpitch), _MM_HINT_T0);

				outcur += outpitch;
				wsharpen += outpitch;
				gridsample += outpitch;
			}
			wsharpen -= outpitch * bh;
			gridsample -= outpitch * bh; // restore pointer to only valid first block - bug fixed in v1.8.1
		}
	}
	if (sharpen == 0 && dehalo != 0)
	{
		const __m256 dehalo8 = _mm256_set1_ps(dehalo);
		const __m256 ht2n8 = _mm256_set1_ps(ht2n);

		for (int block = start_block; block < blocks; block++)
		{
			const __m256 gridfraction8 = _mm256_set1_ps(degrid * outcur[0][0] / gridsample[0][0]);
			for (int h = 0; h < bh; h++) // middle
			{
				for (int w = 0; w < outwidth; w = w + 4)
				{
					const __m256 gridcorrection = _mm256_mul_ps(gridfraction8, _mm256_load_ps(gridsample[w])); //gridcorrection
					__m256 cur = _mm256_load_ps(outcur[w]);
					cur = _mm256_sub_ps(cur, gridcorrection);
					__m256 r1 = _mm256_mul_ps(cur, cur);
					__m256 psd4 = _mm256_hadd_ps(r1, r1); //psd1, -, psd2, -

					__m256 r3 = _mm256_add_ps(psd4, ht2n8);
					__m256 r4 = _mm256_mul_ps(psd4, dehalo8);
					r4 = _mm256_mul_ps(r4, _mm256_broadcast_ps((__m128*)&wdehalo[w]));
					r4 = _mm256_add_ps(r3, r4);
					__m256 r2 = _mm256_div_ps(r3, r4);

					r2 = _mm256_permutevar_ps(r2, _mm256_set_epi32(3, 3, 2, 2, 1, 1, 0, 0));
					r2 = _mm256_mul_ps(r2, cur);
					_mm256_store_ps(outcur[w], _mm256_add_ps(r2, gridcorrection));
				}
				_mm_prefetch((const char*)(outcur + outpitch), _MM_HINT_T0);
				_mm_prefetch((const char*)(gridsample + outpitch), _MM_HINT_T0);
				_mm_prefetch((const char*)(wdehalo + outpitch), _MM_HINT_T0);

				outcur += outpitch;
				wdehalo += outpitch;
				gridsample += outpitch;
			}
			wdehalo -= outpitch * bh;
			gridsample -= outpitch * bh; // restore pointer to only valid first block - bug fixed in v1.8.1
		}
	}
	if (sharpen != 0 && dehalo != 0)
	{
		const __m256 sharpen8 = _mm256_set1_ps(sharpen);
		const __m256 sigmaSquaredSharpenMax8 = _mm256_set1_ps(sigmaSquaredSharpenMax);
		const __m256 sigmaSquaredSharpenMin8 = _mm256_set1_ps(sigmaSquaredSharpenMin);

		const __m256 dehalo8 = _mm256_set1_ps(dehalo);
		const __m256 ht2n8 = _mm256_set1_ps(ht2n);

		for (int block = start_block; block < blocks; block++)
		{
			const __m256 gridfraction8 = _mm256_set1_ps(degrid * outcur[0][0] / gridsample[0][0]);
			for (int h = 0; h < bh; h++) // middle
			{
				for (int w = 0; w < outwidth; w = w + 4)
				{
					const __m256 gridcorrection = _mm256_mul_ps(gridfraction8, _mm256_load_ps(gridsample[w])); //gridcorrection
					__m256 cur = _mm256_load_ps(outcur[w]);
					cur = _mm256_sub_ps(cur, gridcorrection);
					__m256 r1 = _mm256_mul_ps(cur, cur);
					__m256 psd4 = _mm256_hadd_ps(r1, r1); //psd1, -, psd2, -

					__m256 sharp4 = _mm256_broadcast_ps((__m128*) &wsharpen[w]);
					__m256 r2 = _mm256_mul_ps(sharpen8, sharp4);
					__m256 r3 = _mm256_mul_ps(psd4, sigmaSquaredSharpenMax8);
					__m256 r4 = _mm256_add_ps(psd4, sigmaSquaredSharpenMin8);
					__m256 r5 = _mm256_add_ps(psd4, sigmaSquaredSharpenMax8);
					r4 = _mm256_mul_ps(r4, r5);
					r3 = _mm256_div_ps(r3, r4);
					r3 = _mm256_sqrt_ps(r3);
					r2 = _mm256_mul_ps(r2, r3);
					r2 = _mm256_add_ps(r2, _mm256_set1_ps(1.0f));

					__m256 wdehalo4 = _mm256_broadcast_ps((__m128*)&wdehalo[w]);
					r3 = _mm256_add_ps(psd4, ht2n8);
					r4 = _mm256_mul_ps(psd4, dehalo8);
					r4 = _mm256_mul_ps(r4, wdehalo4);
					r4 = _mm256_add_ps(r3, r4);
					r3 = _mm256_div_ps(r3, r4);

					r2 = _mm256_mul_ps(r2, r3);

					r2 = _mm256_permutevar_ps(r2, _mm256_set_epi32(3, 3, 2, 2, 1, 1, 0, 0));
					r2 = _mm256_mul_ps(r2, cur);
					_mm256_store_ps(outcur[w], _mm256_add_ps(r2, gridcorrection));
				}
				_mm_prefetch((const char*)(outcur + outpitch), _MM_HINT_T0);
				_mm_prefetch((const char*)(gridsample + outpitch), _MM_HINT_T0);
				_mm_prefetch((const char*)(wsharpen + outpitch), _MM_HINT_T0);
				_mm_prefetch((const char*)(wdehalo + outpitch), _MM_HINT_T0);

				outcur += outpitch;
				wsharpen += outpitch;
				wdehalo += outpitch;
				gridsample += outpitch;
			}
			wsharpen -= outpitch * bh;
			wdehalo -= outpitch * bh;
			gridsample -= outpitch * bh; // restore pointer to only valid first block - bug fixed in v1.8.1
		}
	}
}