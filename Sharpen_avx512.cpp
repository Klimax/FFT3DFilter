//
//	FFT3DFilter plugin for Avisynth 2.6 - 3D Frequency Domain filter
//  AVX512 version of SSE filtering functions
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
void SharpenFilter::Sharpen_AVX512() noexcept
{
	const int outwidth8 = outwidth - outwidth % 8;

	if (sharpen != 0 && dehalo == 0)
	{
		const __m512 sharpen8 = _mm512_set1_ps(sharpen);
		const __m512 sigmaSquaredSharpenMax8 = _mm512_set1_ps(sigmaSquaredSharpenMax);
		const __m512 sigmaSquaredSharpenMin8 = _mm512_set1_ps(sigmaSquaredSharpenMin);
		for (int block = start_block; block < blocks; block++)
		{
			for (int h = 0; h < bh; h++) // middle
			{
				for (int w = 0; w < outwidth8; w = w + 8)
				{
					__m512 cur = _mm512_load_ps(outcur[w]);
					__m512 r1 = _mm512_mul_ps(cur, cur);
					__m512 r2 = _mm512_shuffle_ps(r1, r1, _MM_SHUFFLE(0, 3, 0, 1));
					__m512 psd8 = _mm512_add_ps(r1, r2); //psd1, -, psd2, -

					__m512 sharp8 = _mm512_castps256_ps512(_mm256_load_ps(&wsharpen[w]));
					sharp8 = _mm512_permutexvar_ps(_mm512_set_epi32(7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0), sharp8);
					r2 = _mm512_mul_ps(sharpen8, sharp8);
					__m512 r3 = _mm512_mul_ps(psd8, sigmaSquaredSharpenMax8);
					__m512 r4 = _mm512_add_ps(psd8, sigmaSquaredSharpenMin8);
					__m512 r5 = _mm512_add_ps(psd8, sigmaSquaredSharpenMax8);
					r4 = _mm512_mul_ps(r4, r5);
					r3 = _mm512_div_ps(r3, r4);
					r3 = _mm512_sqrt_ps(r3);
					r2 = _mm512_mul_ps(r2, r3);
					r2 = _mm512_add_ps(r2, _mm512_set1_ps(1.0f));
					r2 = _mm512_moveldup_ps(r2);
					_mm512_store_ps(outcur[w], _mm512_mul_ps(r2, cur));
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
		const __m512 dehalo8 = _mm512_set1_ps(dehalo);
		const __m512 ht2n8 = _mm512_set1_ps(ht2n);

		for (int block = start_block; block < blocks; block++)
		{
			for (int h = 0; h < bh; h++) // middle
			{
				for (int w = 0; w < outwidth8; w = w + 8)
				{
					__m512 cur = _mm512_load_ps(outcur[w]);
					__m512 r1 = _mm512_mul_ps(cur, cur);
					__m512 r2 = _mm512_shuffle_ps(r1, r1, _MM_SHUFFLE(0, 3, 0, 1));
					__m512 psd8 = _mm512_add_ps(r1, r2); //psd1, -, psd2, -

					__m512 wdehalo8 = _mm512_castps256_ps512(_mm256_load_ps(&wdehalo[w]));
					wdehalo8 = _mm512_permutexvar_ps(_mm512_set_epi32(7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0), wdehalo8);
					__m512 r3 = _mm512_add_ps(psd8, ht2n8);
					__m512 r4 = _mm512_mul_ps(psd8, dehalo8);
					r4 = _mm512_mul_ps(r4, wdehalo8);
					r4 = _mm512_add_ps(r3, r4);
					r3 = _mm512_div_ps(r3, r4);

					r2 = _mm512_moveldup_ps(r3);
					_mm512_store_ps(outcur[w], _mm512_mul_ps(r2, cur));
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
		const __m512 sharpen8 = _mm512_set1_ps(sharpen);
		const __m512 sigmaSquaredSharpenMax8 = _mm512_set1_ps(sigmaSquaredSharpenMax);
		const __m512 sigmaSquaredSharpenMin8 = _mm512_set1_ps(sigmaSquaredSharpenMin);

		const __m512 dehalo8 = _mm512_set1_ps(dehalo);
		const __m512 ht2n8 = _mm512_set1_ps(ht2n);

		for (int block = start_block; block < blocks; block++)
		{
			for (int h = 0; h < bh; h++) // middle
			{
				for (int w = 0; w < outwidth8; w = w + 8)
				{
					__m512 cur = _mm512_load_ps(outcur[w]);
					__m512 r1 = _mm512_mul_ps(cur, cur);
					__m512 r2 = _mm512_shuffle_ps(r1, r1, _MM_SHUFFLE(0, 3, 0, 1));
					__m512 psd8 = _mm512_add_ps(r1, r2); //psd1, -, psd2, -

					__m512 sharp8 = _mm512_castps256_ps512(_mm256_load_ps(&wsharpen[w]));
					sharp8 = _mm512_permutexvar_ps(_mm512_set_epi32(7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0), sharp8);
					r2 = _mm512_mul_ps(sharpen8, sharp8);
					__m512 r3 = _mm512_mul_ps(psd8, sigmaSquaredSharpenMax8);
					__m512 r4 = _mm512_add_ps(psd8, sigmaSquaredSharpenMin8);
					__m512 r5 = _mm512_add_ps(psd8, sigmaSquaredSharpenMax8);
					r4 = _mm512_mul_ps(r4, r5);
					r3 = _mm512_div_ps(r3, r4);
					r3 = _mm512_sqrt_ps(r3);
					r2 = _mm512_mul_ps(r2, r3);
					r2 = _mm512_add_ps(r2, _mm512_set1_ps(1.0f));

					__m512 wdehalo8 = _mm512_castps256_ps512(_mm256_load_ps(&wdehalo[w]));
					wdehalo8 = _mm512_permutexvar_ps(_mm512_set_epi32(7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0), wdehalo8);
					r3 = _mm512_add_ps(psd8, ht2n8);
					r4 = _mm512_mul_ps(psd8, dehalo8);
					r4 = _mm512_mul_ps(r4, wdehalo8);
					r4 = _mm512_add_ps(r3, r4);
					r3 = _mm512_div_ps(r3, r4);

					r2 = _mm512_mul_ps(r2, r3);

					r2 = _mm512_moveldup_ps(r2);
					_mm512_store_ps(outcur[w], _mm512_mul_ps(r2, cur));
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
void SharpenFilter::Sharpen_degrid_AVX512() noexcept
{
	const int outwidth8 = outwidth - outwidth % 8;

	if (sharpen != 0 && dehalo == 0)
	{
		const __m512 sharpen8 = _mm512_set1_ps(sharpen);
		const __m512 sigmaSquaredSharpenMax8 = _mm512_set1_ps(sigmaSquaredSharpenMax);
		const __m512 sigmaSquaredSharpenMin8 = _mm512_set1_ps(sigmaSquaredSharpenMin);

		for (int block = start_block; block < blocks; block++)
		{
			const __m512 gridfraction8 = _mm512_set1_ps(degrid * outcur[0][0] / gridsample[0][0]);
			for (int h = 0; h < bh; h++) // middle
			{
				for (int w = 0; w < outwidth8; w = w + 8)
				{
					const __m512 gridcorrection = _mm512_mul_ps(gridfraction8, _mm512_load_ps(gridsample[w])); //gridcorrection
					__m512 cur = _mm512_load_ps(outcur[w]);
					cur = _mm512_sub_ps(cur, gridcorrection);
					__m512 r1 = _mm512_mul_ps(cur, cur);
					__m512 r2 = _mm512_shuffle_ps(r1, r1, _MM_SHUFFLE(0, 3, 0, 1));
					__m512 psd8 = _mm512_add_ps(r1, r2); //psd1, -, psd2, -

					__m512 sharp8 = _mm512_castps256_ps512(_mm256_load_ps(&wsharpen[w]));
					sharp8 = _mm512_permutexvar_ps(_mm512_set_epi32(7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0), sharp8);
					r2 = _mm512_mul_ps(sharpen8, sharp8);
					__m512 r3 = _mm512_mul_ps(psd8, sigmaSquaredSharpenMax8);
					__m512 r4 = _mm512_add_ps(psd8, sigmaSquaredSharpenMin8);
					__m512 r5 = _mm512_add_ps(psd8, sigmaSquaredSharpenMax8);
					r4 = _mm512_mul_ps(r4, r5);
					r3 = _mm512_div_ps(r3, r4);
					r3 = _mm512_sqrt_ps(r3);
					r2 = _mm512_mul_ps(r2, r3);
					r2 = _mm512_add_ps(r2, _mm512_set1_ps(1.0f));
					r2 = _mm512_moveldup_ps(r2);
					r2 = _mm512_mul_ps(r2, cur);
					_mm512_store_ps(outcur[w], _mm512_add_ps(r2, gridcorrection));
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
		const __m512 dehalo8 = _mm512_set1_ps(dehalo);
		const __m512 ht2n8 = _mm512_set1_ps(ht2n);

		for (int block = start_block; block < blocks; block++)
		{
			const __m512 gridfraction8 = _mm512_set1_ps(degrid * outcur[0][0] / gridsample[0][0]);
			for (int h = 0; h < bh; h++) // middle
			{
				for (int w = 0; w < outwidth8; w = w + 8)
				{
					const __m512 gridcorrection = _mm512_mul_ps(gridfraction8, _mm512_load_ps(gridsample[w])); //gridcorrection
					__m512 cur = _mm512_load_ps(outcur[w]);
					cur = _mm512_sub_ps(cur, gridcorrection);
					__m512 r1 = _mm512_mul_ps(cur, cur);
					__m512 r2 = _mm512_shuffle_ps(r1, r1, _MM_SHUFFLE(0, 3, 0, 1));
					__m512 psd8 = _mm512_add_ps(r1, r2); //psd1, -, psd2, -

					__m512 wdehalo8 = _mm512_castps256_ps512(_mm256_load_ps(&wdehalo[w]));
					wdehalo8 = _mm512_permutexvar_ps(_mm512_set_epi32(7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0), wdehalo8);
					__m512 r3 = _mm512_add_ps(psd8, ht2n8);
					__m512 r4 = _mm512_mul_ps(psd8, dehalo8);
					r4 = _mm512_mul_ps(r4, wdehalo8);
					r4 = _mm512_add_ps(r3, r4);
					r3 = _mm512_div_ps(r3, r4);

					r2 = _mm512_moveldup_ps(r3);
					r2 = _mm512_mul_ps(r2, cur);
					_mm512_store_ps(outcur[w], _mm512_add_ps(r2, gridcorrection));
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
		const __m512 sharpen8 = _mm512_set1_ps(sharpen);
		const __m512 sigmaSquaredSharpenMax8 = _mm512_set1_ps(sigmaSquaredSharpenMax);
		const __m512 sigmaSquaredSharpenMin8 = _mm512_set1_ps(sigmaSquaredSharpenMin);

		const __m512 dehalo8 = _mm512_set1_ps(dehalo);
		const __m512 ht2n8 = _mm512_set1_ps(ht2n);

		for (int block = start_block; block < blocks; block++)
		{
			const __m512 gridfraction8 = _mm512_set1_ps(degrid * outcur[0][0] / gridsample[0][0]);
			for (int h = 0; h < bh; h++) // middle
			{
				for (int w = 0; w < outwidth8; w = w + 8)
				{
					const __m512 gridcorrection = _mm512_mul_ps(gridfraction8, _mm512_load_ps(gridsample[w])); //gridcorrection
					__m512 cur = _mm512_load_ps(outcur[w]);
					cur = _mm512_sub_ps(cur, gridcorrection);
					__m512 r1 = _mm512_mul_ps(cur, cur);
					__m512 r2 = _mm512_shuffle_ps(r1, r1, _MM_SHUFFLE(0, 3, 0, 1));
					__m512 psd8 = _mm512_add_ps(r1, r2); //psd1, -, psd2, -

					__m512 sharp8 = _mm512_castps256_ps512(_mm256_load_ps(&wsharpen[w]));
					sharp8 = _mm512_permutexvar_ps(_mm512_set_epi32(7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0), sharp8);
					r2 = _mm512_mul_ps(sharpen8, sharp8);
					__m512 r3 = _mm512_mul_ps(psd8, sigmaSquaredSharpenMax8);
					__m512 r4 = _mm512_add_ps(psd8, sigmaSquaredSharpenMin8);
					__m512 r5 = _mm512_add_ps(psd8, sigmaSquaredSharpenMax8);
					r4 = _mm512_mul_ps(r4, r5);
					r3 = _mm512_div_ps(r3, r4);
					r3 = _mm512_sqrt_ps(r3);
					r2 = _mm512_mul_ps(r2, r3);
					r2 = _mm512_add_ps(r2, _mm512_set1_ps(1.0f));

					__m512 wdehalo8 = _mm512_castps256_ps512(_mm256_load_ps(&wdehalo[w]));
					wdehalo8 = _mm512_permutexvar_ps(_mm512_set_epi32(7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0), wdehalo8);
					r3 = _mm512_add_ps(psd8, ht2n8);
					r4 = _mm512_mul_ps(psd8, dehalo8);
					r4 = _mm512_mul_ps(r4, wdehalo8);
					r4 = _mm512_add_ps(r3, r4);
					r3 = _mm512_div_ps(r3, r4);

					r2 = _mm512_mul_ps(r2, r3);

					r2 = _mm512_moveldup_ps(r2);

					r2 = _mm512_mul_ps(r2, cur);
					_mm512_store_ps(outcur[w], _mm512_add_ps(r2, gridcorrection));
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