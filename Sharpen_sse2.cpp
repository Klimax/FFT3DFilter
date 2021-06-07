//
//	FFT3DFilter plugin for Avisynth 2.6 - 3D Frequency Domain filter
//  SSE2 version of filtering functions
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
void SharpenFilter::Sharpen_SSE2() noexcept
{
	if (sharpen != 0 && dehalo == 0)
	{
		const __m128 sharpen4 = _mm_load1_ps(&sharpen);
		const __m128 sigmaSquaredSharpenMax4 = _mm_load1_ps(&sigmaSquaredSharpenMax);
		const __m128 sigmaSquaredSharpenMin4 = _mm_load1_ps(&sigmaSquaredSharpenMin);

		for (int block = start_block; block < blocks; block++)
		{
			for (int h = 0; h < bh; h++) // middle
			{
				for (int w = 0; w < outwidth; w = w + 2)
				{
					const __m128 cur = _mm_load_ps(outcur[w]);
					const __m128 r1 = _mm_mul_ps(cur, cur);
					__m128 r2 = _mm_shuffle_ps(r1, r1, _MM_SHUFFLE(0, 3, 0, 1));
					const __m128 psd4 = _mm_add_ps(r1, r2); //psd1, -, psd2, -

					__m128 sharp4 = _mm_loadl_pi(_mm_setzero_ps(), (__m64*) &wsharpen[w]);
					sharp4 = _mm_unpacklo_ps(sharp4, sharp4);
					r2 = _mm_mul_ps(sharpen4, sharp4);
					__m128 r3 = _mm_mul_ps(psd4, sigmaSquaredSharpenMax4);
					__m128 r4 = _mm_add_ps(psd4, sigmaSquaredSharpenMin4);
					const __m128 r5 = _mm_add_ps(psd4, sigmaSquaredSharpenMax4);
					r4 = _mm_mul_ps(r4, r5);
					r3 = _mm_div_ps(r3, r4);
					r3 = _mm_sqrt_ps(r3);
					r2 = _mm_mul_ps(r2, r3);
					r2 = _mm_add_ps(r2, _mm_set1_ps(1.0f));
					r2 = _mm_shuffle_ps(r2, r2, _MM_SHUFFLE(2, 2, 0, 0));
					_mm_store_ps(outcur[w], _mm_mul_ps(r2, cur));
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
		const __m128 dehalo4 = _mm_load1_ps(&dehalo);
		const __m128 ht2n4 = _mm_load1_ps(&ht2n);

		for (int block = start_block; block < blocks; block++)
		{
			for (int h = 0; h < bh; h++) // middle
			{
				for (int w = 0; w < outwidth; w = w + 2)
				{
					const __m128 cur = _mm_load_ps(outcur[w]);
					const __m128 r1 = _mm_mul_ps(cur, cur);
					__m128 r2 = _mm_shuffle_ps(r1, r1, _MM_SHUFFLE(0, 3, 0, 1));
					const __m128 psd4 = _mm_add_ps(r1, r2); //psd1, -, psd2, -

					const __m128 wdehalo4 = _mm_loadl_pi(_mm_setzero_ps(), (__m64*)&wdehalo[w]);
					__m128 r3 = _mm_add_ps(psd4, ht2n4);
					__m128 r4 = _mm_mul_ps(psd4, dehalo4);
					r4 = _mm_mul_ps(r4, _mm_unpacklo_ps(wdehalo4, wdehalo4));
					r4 = _mm_add_ps(r3, r4);
					r3 = _mm_div_ps(r3, r4);

					r2 = _mm_shuffle_ps(r3, r3, _MM_SHUFFLE(2, 2, 0, 0));
					_mm_store_ps(outcur[w], _mm_mul_ps(r2, cur));
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
		const __m128 sharpen4 = _mm_load1_ps(&sharpen);
		const __m128 sigmaSquaredSharpenMax4 = _mm_load1_ps(&sigmaSquaredSharpenMax);
		const __m128 sigmaSquaredSharpenMin4 = _mm_load1_ps(&sigmaSquaredSharpenMin);

		const __m128 dehalo4 = _mm_load1_ps(&dehalo);
		const __m128 ht2n4 = _mm_load1_ps(&ht2n);

		for (int block = start_block; block < blocks; block++)
		{
			for (int h = 0; h < bh; h++) // middle
			{
				for (int w = 0; w < outwidth; w = w + 2)
				{
					const __m128 cur = _mm_load_ps(outcur[w]);
					const __m128 r1 = _mm_mul_ps(cur, cur);
					__m128 r2 = _mm_shuffle_ps(r1, r1, _MM_SHUFFLE(0, 3, 0, 1));
					const __m128 psd4 = _mm_add_ps(r1, r2); //psd1, -, psd2, -

					__m128 sharp4 = _mm_loadl_pi(_mm_setzero_ps(), (__m64*) &wsharpen[w]);
					sharp4 = _mm_unpacklo_ps(sharp4, sharp4);
					r2 = _mm_mul_ps(sharpen4, sharp4);
					__m128 r3 = _mm_mul_ps(psd4, sigmaSquaredSharpenMax4);
					__m128 r4 = _mm_add_ps(psd4, sigmaSquaredSharpenMin4);
					const __m128 r5 = _mm_add_ps(psd4, sigmaSquaredSharpenMax4);
					r4 = _mm_mul_ps(r4, r5);
					r3 = _mm_div_ps(r3, r4);
					r3 = _mm_sqrt_ps(r3);
					r2 = _mm_mul_ps(r2, r3);
					r2 = _mm_add_ps(r2, _mm_set1_ps(1.0f));

					const __m128 wdehalo4 = _mm_loadl_pi(_mm_setzero_ps(), (__m64*)&wdehalo[w]);
					r3 = _mm_add_ps(psd4, ht2n4);
					r4 = _mm_mul_ps(psd4, dehalo4);
					r4 = _mm_mul_ps(r4, _mm_unpacklo_ps(wdehalo4, wdehalo4));
					r4 = _mm_add_ps(r3, r4);
					r3 = _mm_div_ps(r3, r4);

					r2 = _mm_mul_ps(r2, r3);

					r2 = _mm_shuffle_ps(r2, r2, _MM_SHUFFLE(2, 2, 0, 0));
					_mm_store_ps(outcur[w], _mm_mul_ps(r2, cur));
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

void SharpenFilter::Sharpen_degrid_SSE2() noexcept
{
	if (sharpen != 0 && dehalo == 0)
	{
		const __m128 sharpen4 = _mm_load1_ps(&sharpen);
		const __m128 sigmaSquaredSharpenMax4 = _mm_load1_ps(&sigmaSquaredSharpenMax);
		const __m128 sigmaSquaredSharpenMin4 = _mm_load1_ps(&sigmaSquaredSharpenMin);

		for (int block = start_block; block < blocks; block++)
		{
			const __m128 gridfraction4 = _mm_set1_ps(degrid * outcur[0][0] / gridsample[0][0]);
			for (int h = 0; h < bh; h++) // middle
			{
				for (int w = 0; w < outwidth; w = w + 2)
				{
					const __m128 gridcorrection = _mm_mul_ps(gridfraction4, _mm_load_ps(gridsample[w])); //gridcorrection
					__m128 cur = _mm_load_ps(outcur[w]);
					cur = _mm_sub_ps(cur, gridcorrection);
					const __m128 r1 = _mm_mul_ps(cur, cur);
					__m128 r2 = _mm_shuffle_ps(r1, r1, _MM_SHUFFLE(0, 3, 0, 1));
					const __m128 psd4 = _mm_add_ps(r1, r2); //psd1, -, psd2, -

					__m128 sharp4 = _mm_loadl_pi(_mm_setzero_ps(), (__m64*) &wsharpen[w]);
					sharp4 = _mm_unpacklo_ps(sharp4, sharp4);
					r2 = _mm_mul_ps(sharpen4, sharp4);
					__m128 r3 = _mm_mul_ps(psd4, sigmaSquaredSharpenMax4);
					__m128 r4 = _mm_add_ps(psd4, sigmaSquaredSharpenMin4);
					const __m128 r5 = _mm_add_ps(psd4, sigmaSquaredSharpenMax4);
					r4 = _mm_mul_ps(r4, r5);
					r3 = _mm_div_ps(r3, r4);
					r3 = _mm_sqrt_ps(r3);
					r2 = _mm_mul_ps(r2, r3);
					r2 = _mm_add_ps(r2, _mm_set1_ps(1.0f));

					r2 = _mm_shuffle_ps(r2, r2, _MM_SHUFFLE(2, 2, 0, 0));
					r2 = _mm_mul_ps(r2, cur);
					_mm_store_ps(outcur[w], _mm_add_ps(r2, gridcorrection));
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
		const __m128 dehalo4 = _mm_load1_ps(&dehalo);
		const __m128 ht2n4 = _mm_load1_ps(&ht2n);

		for (int block = start_block; block < blocks; block++)
		{
			const __m128 gridfraction4 = _mm_set1_ps(degrid * outcur[0][0] / gridsample[0][0]);
			for (int h = 0; h < bh; h++) // middle
			{
				for (int w = 0; w < outwidth; w = w + 2)
				{
					const __m128 gridcorrection = _mm_mul_ps(gridfraction4, _mm_load_ps(gridsample[w])); //gridcorrection
					__m128 cur = _mm_load_ps(outcur[w]);
					cur = _mm_sub_ps(cur, gridcorrection);
					const __m128 r1 = _mm_mul_ps(cur, cur);
					__m128 r2 = _mm_shuffle_ps(r1, r1, _MM_SHUFFLE(0, 3, 0, 1));
					const __m128 psd4 = _mm_add_ps(r1, r2); //psd1, -, psd2, -

					const __m128 wdehalo4 = _mm_loadl_pi(_mm_setzero_ps(), (__m64*)&wdehalo[w]);
					const __m128 r3 = _mm_add_ps(psd4, ht2n4);
					__m128 r4 = _mm_mul_ps(psd4, dehalo4);
					r4 = _mm_mul_ps(r4, _mm_unpacklo_ps(wdehalo4, wdehalo4));
					r4 = _mm_add_ps(r3, r4);
					r2 = _mm_div_ps(r3, r4);

					r2 = _mm_shuffle_ps(r2, r2, _MM_SHUFFLE(2, 2, 0, 0));
					r2 = _mm_mul_ps(r2, cur);
					_mm_store_ps(outcur[w], _mm_add_ps(r2, gridcorrection));
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
		const __m128 sharpen4 = _mm_load1_ps(&sharpen);
		const __m128 sigmaSquaredSharpenMax4 = _mm_load1_ps(&sigmaSquaredSharpenMax);
		const __m128 sigmaSquaredSharpenMin4 = _mm_load1_ps(&sigmaSquaredSharpenMin);

		const __m128 dehalo4 = _mm_load1_ps(&dehalo);
		const __m128 ht2n4 = _mm_load1_ps(&ht2n);

		for (int block = start_block; block < blocks; block++)
		{
			const __m128 gridfraction4 = _mm_set1_ps(degrid * outcur[0][0] / gridsample[0][0]);
			for (int h = 0; h < bh; h++) // middle
			{
				for (int w = 0; w < outwidth; w = w + 2)
				{
					const __m128 gridcorrection = _mm_mul_ps(gridfraction4, _mm_load_ps(gridsample[w])); //gridcorrection
					__m128 cur = _mm_load_ps(outcur[w]);
					cur = _mm_sub_ps(cur, gridcorrection);
					const __m128 r1 = _mm_mul_ps(cur, cur);
					__m128 r2 = _mm_shuffle_ps(r1, r1, _MM_SHUFFLE(0, 3, 0, 1));
					const __m128 psd4 = _mm_add_ps(r1, r2); //psd1, -, psd2, -

					__m128 sharp4 = _mm_loadl_pi(_mm_setzero_ps(), (__m64*) &wsharpen[w]);
					sharp4 = _mm_unpacklo_ps(sharp4, sharp4);
					r2 = _mm_mul_ps(sharpen4, sharp4);
					__m128 r3 = _mm_mul_ps(psd4, sigmaSquaredSharpenMax4);
					__m128 r4 = _mm_add_ps(psd4, sigmaSquaredSharpenMin4);
					const __m128 r5 = _mm_add_ps(psd4, sigmaSquaredSharpenMax4);
					r4 = _mm_mul_ps(r4, r5);
					r3 = _mm_div_ps(r3, r4);
					r3 = _mm_sqrt_ps(r3);
					r2 = _mm_mul_ps(r2, r3);
					r2 = _mm_add_ps(r2, _mm_set1_ps(1.0f));

					const __m128 wdehalo4 = _mm_loadl_pi(_mm_setzero_ps(), (__m64*)&wdehalo[w]);
					r3 = _mm_add_ps(psd4, ht2n4);
					r4 = _mm_mul_ps(psd4, dehalo4);
					r4 = _mm_mul_ps(r4, _mm_unpacklo_ps(wdehalo4, wdehalo4));
					r4 = _mm_add_ps(r3, r4);
					r3 = _mm_div_ps(r3, r4);

					r2 = _mm_mul_ps(r2, r3);

					r2 = _mm_shuffle_ps(r2, r2, _MM_SHUFFLE(2, 2, 0, 0));
					r2 = _mm_mul_ps(r2, cur);
					_mm_store_ps(outcur[w], _mm_add_ps(r2, gridcorrection));
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