//
//	FFT3DFilter plugin for Avisynth 2.6 - 3D Frequency Domain filter
//  SSE version of filtering functions
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

// since v1.7 we use outpitch instead of outwidth

void WienerFilter::ApplyWiener2D_SSE() noexcept
{
	if (sharpen == 0 && dehalo == 0)// no sharpen, no dehalo
	{
		for (int block = start_block; block < blocks; block++)
		{
			for (int h = 0; h < bh; h++) // middle
			{
				for (int w = 0; w < outwidth; w = w + 2) // not skip first v.1.2
				{
					__m128 r1 = _mm_load_ps(outcur[w]);
					__m128 r3 = _mm_mul_ps(r1, r1);
					__m128 r2 = _mm_shuffle_ps(r3, r3, _MM_SHUFFLE(0, 3, 0, 1));
					r3 = _mm_add_ps(r3, r2);
					r3 = _mm_add_ps(r3, _mm_set1_ps(1e-15f));
					r2 = _mm_sub_ps(r3, _mm_load1_ps(&sigmaSquaredNoiseNormed));
					r3 = _mm_div_ps(r2, r3);
					r3 = _mm_max_ps(r3, _mm_load1_ps(&lowlimit));
					r3 = _mm_shuffle_ps(r3, r3, _MM_SHUFFLE(2, 2, 0, 0));
					r1 = _mm_mul_ps(r1, r3);
					_mm_store_ps(outcur[w], r1);
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
				for (int w = 0; w < outwidth; w = w + 2) // not skip first
				{
					__m128 r1 = _mm_load_ps(outcur[w]);
					__m128 r3 = _mm_mul_ps(r1, r1);
					__m128 r2 = _mm_shuffle_ps(r3, r3, _MM_SHUFFLE(0, 3, 0, 1));
					r3 = _mm_add_ps(r3, r2);
					r3 = _mm_add_ps(r3, _mm_set1_ps(1e-15f));

					__m128 sharp = _mm_loadl_pi(_mm_setzero_ps(), (__m64*) &wsharpen[w]);
					__m128 r7 = _mm_mul_ps(_mm_load1_ps(&sharpen), _mm_unpacklo_ps(sharp, sharp));
					__m128 r4 = _mm_mul_ps(r3, _mm_load1_ps(&sigmaSquaredSharpenMax));
					__m128 r5 = _mm_add_ps(r3, _mm_load1_ps(&sigmaSquaredSharpenMin));
					__m128 r6 = _mm_add_ps(r3, _mm_load1_ps(&sigmaSquaredSharpenMax));
					r5 = _mm_mul_ps(r5, r6);
					r4 = _mm_div_ps(r4, r5);
					r4 = _mm_sqrt_ps(r4);
					r4 = _mm_mul_ps(r4, r7);

					r2 = _mm_sub_ps(r3, _mm_load1_ps(&sigmaSquaredNoiseNormed));
					r3 = _mm_div_ps(r2, r3);
					r3 = _mm_max_ps(r3, _mm_load1_ps(&lowlimit));
					r3 = _mm_shuffle_ps(r3, r3, _MM_SHUFFLE(2, 2, 0, 0));

					r4 = _mm_add_ps(r4, _mm_set1_ps(1));
					r4 = _mm_shuffle_ps(r4, r4, _MM_SHUFFLE(2, 2, 0, 0));
					r3 = _mm_mul_ps(r3, r4);
					r1 = _mm_mul_ps(r1, r3);
					_mm_store_ps(outcur[w], r1);
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
				for (int w = 0; w < outwidth; w = w + 2) // not skip first
				{
					__m128 r1 = _mm_load_ps(outcur[w]);
					__m128 r3 = _mm_mul_ps(r1, r1);
					__m128 r2 = _mm_shuffle_ps(r3, r3, _MM_SHUFFLE(0, 3, 0, 1));
					r3 = _mm_add_ps(r3, r2);
					r3 = _mm_add_ps(r3, _mm_set1_ps(1e-15f));

					__m128 halo = _mm_loadl_pi(_mm_setzero_ps(), (__m64*) &wdehalo[w]);
					__m128 r7 = _mm_mul_ps(r3, _mm_unpacklo_ps(halo, halo));
					__m128 r5 = _mm_add_ps(r3, _mm_load1_ps(&ht2n));
					r7 = _mm_mul_ps(r7, _mm_load1_ps(&dehalo));
					r7 = _mm_add_ps(r5, r7);
					r5 = _mm_div_ps(r5, r7);

					r2 = _mm_sub_ps(r3, _mm_load1_ps(&sigmaSquaredNoiseNormed));
					r3 = _mm_div_ps(r2, r3);
					r3 = _mm_max_ps(r3, _mm_load1_ps(&lowlimit));
					r3 = _mm_shuffle_ps(r3, r3, _MM_SHUFFLE(2, 2, 0, 0));

					r5 = _mm_shuffle_ps(r5, r5, _MM_SHUFFLE(2, 2, 0, 0));
					r3 = _mm_mul_ps(r3, r5);
					r1 = _mm_mul_ps(r1, r3);
					_mm_store_ps(outcur[w], r1);
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
				for (int w = 0; w < outwidth; w = w + 2) // not skip first
				{
					__m128 r1 = _mm_load_ps(outcur[w]);
					__m128 r3 = _mm_mul_ps(r1, r1);
					__m128 r2 = _mm_shuffle_ps(r3, r3, _MM_SHUFFLE(0, 3, 0, 1));
					r3 = _mm_add_ps(r3, r2);
					r3 = _mm_add_ps(r3, _mm_set1_ps(1e-15f));

					__m128 sharp = _mm_loadl_pi(_mm_setzero_ps(), (__m64*) &wsharpen[w]);
					__m128 r7 = _mm_mul_ps(_mm_load1_ps(&sharpen), _mm_unpacklo_ps(sharp, sharp));
					__m128 r4 = _mm_mul_ps(r3, _mm_load1_ps(&sigmaSquaredSharpenMax));
					__m128 r5 = _mm_add_ps(r3, _mm_load1_ps(&sigmaSquaredSharpenMin));
					__m128 r6 = _mm_add_ps(r3, _mm_load1_ps(&sigmaSquaredSharpenMax));
					r5 = _mm_mul_ps(r5, r6);
					r4 = _mm_div_ps(r4, r5);
					r4 = _mm_sqrt_ps(r4);
					r4 = _mm_mul_ps(r4, r7);

					r2 = _mm_sub_ps(r3, _mm_load1_ps(&sigmaSquaredNoiseNormed));
					r3 = _mm_div_ps(r2, r3);
					r3 = _mm_max_ps(r3, _mm_load1_ps(&lowlimit));
					r3 = _mm_shuffle_ps(r3, r3, _MM_SHUFFLE(2, 2, 0, 0));

					__m128 halo = _mm_loadl_pi(_mm_setzero_ps(), (__m64*) &wdehalo[w]);
					__m128 r8 = _mm_mul_ps(r3, _mm_unpacklo_ps(halo, halo));
					__m128 r9 = _mm_add_ps(r3, _mm_load1_ps(&ht2n));
					r8 = _mm_mul_ps(r8, _mm_load1_ps(&dehalo));
					r8 = _mm_add_ps(r9, r8);
					r9 = _mm_div_ps(r9, r8);

					r4 = _mm_mul_ps(r4, r9);
					r4 = _mm_add_ps(r4, _mm_set1_ps(1));
					r4 = _mm_shuffle_ps(r4, r4, _MM_SHUFFLE(2, 2, 0, 0));
					r3 = _mm_mul_ps(r3, r4);
					r1 = _mm_mul_ps(r1, r3);
					_mm_store_ps(outcur[w], r1);
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

//-----------------------------------------------------------------------------------------
void WienerFilter::ApplyWiener2D_degrid_SSE() noexcept
{
	if (sharpen == 0 && dehalo == 0)// no sharpen, no dehalo
	{
		for (int block = start_block; block < blocks; block++)
		{
			const __m128 gridfraction4 = _mm_set1_ps(degrid * outcur[0][0] / gridsample[0][0]);
			for (int h = 0; h < bh; h++) // middle
			{
				for (int w = 0; w < outwidth; w = w + 2) // not skip first v.1.2
				{
					const __m128 gridcorrection = _mm_mul_ps(gridfraction4, _mm_load_ps(gridsample[w])); //gridcorrection
					__m128 r1 = _mm_load_ps(outcur[w]);
					__m128 r4 = _mm_sub_ps(r1, gridcorrection); //corrected
					__m128 r3 = _mm_mul_ps(r4, r4); //corrected^2
					__m128 r2 = _mm_shuffle_ps(r3, r3, _MM_SHUFFLE(0, 3, 0, 1));
					r3 = _mm_add_ps(r3, r2); //psd
					r3 = _mm_add_ps(r3, _mm_set1_ps(1e-15f)); //psd
					r2 = _mm_sub_ps(r3, _mm_load1_ps(&sigmaSquaredNoiseNormed));
					r3 = _mm_div_ps(r2, r3);
					r3 = _mm_max_ps(r3, _mm_load1_ps(&lowlimit)); //wienerfactor
					r3 = _mm_shuffle_ps(r3, r3, _MM_SHUFFLE(2, 2, 0, 0));
					r1 = _mm_mul_ps(r4, r3);//corrected
					r1 = _mm_add_ps(gridcorrection, r1);// final
					_mm_store_ps(outcur[w], r1);
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
			const __m128 gridfraction4 = _mm_set1_ps(degrid * outcur[0][0] / gridsample[0][0]);
			for (int h = 0; h < bh; h++) // middle
			{
				for (int w = 0; w < outwidth; w = w + 2) // not skip first
				{
					const __m128 gridcorrection = _mm_mul_ps(gridfraction4, _mm_load_ps(gridsample[w])); //gridcorrection
					__m128 r1 = _mm_load_ps(outcur[w]);
					__m128 r4 = _mm_sub_ps(r1, gridcorrection); //corrected
					__m128 r3 = _mm_mul_ps(r4, r4); //corrected^2
					__m128 r2 = _mm_shuffle_ps(r3, r3, _MM_SHUFFLE(0, 3, 0, 1));
					r3 = _mm_add_ps(r3, r2); //psd
					__m128 psd4 = _mm_add_ps(r3, _mm_set1_ps(1e-15f)); //psd [0,2]
					psd4 = _mm_shuffle_ps(psd4, psd4, _MM_SHUFFLE(2, 2, 0, 0));
					r2 = _mm_sub_ps(psd4, _mm_load1_ps(&sigmaSquaredNoiseNormed));
					r3 = _mm_div_ps(r2, psd4);
					__m128 wienerfactor4 = _mm_max_ps(r3, _mm_load1_ps(&lowlimit)); //wienerfactor

					__m128 r5 = _mm_mul_ps(_mm_load1_ps(&sharpen), _mm_loadl_pi(_mm_setzero_ps(), (__m64*)&wsharpen[w]));
					__m128 r6 = _mm_add_ps(psd4, _mm_load1_ps(&sigmaSquaredSharpenMax));
					__m128 r7 = _mm_add_ps(psd4, _mm_load1_ps(&sigmaSquaredSharpenMin));
					__m128 r8 = _mm_mul_ps(psd4, _mm_load1_ps(&sigmaSquaredSharpenMax));

					r7 = _mm_mul_ps(r6, r7);
					r8 = _mm_div_ps(r8, r7);
					r6 = _mm_sqrt_ps(r8);

					r5 = _mm_shuffle_ps(r5, r5, _MM_SHUFFLE(1, 1, 0, 0));
					r5 = _mm_mul_ps(r5, r6);
					r5 = _mm_add_ps(_mm_set1_ps(1.0f), r5);
					wienerfactor4 = _mm_mul_ps(r5, wienerfactor4);

					r1 = _mm_mul_ps(r4, wienerfactor4);//corrected
					r1 = _mm_add_ps(gridcorrection, r1);// final
					_mm_store_ps(outcur[w], r1);
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
			const __m128 gridfraction4 = _mm_set1_ps(degrid * outcur[0][0] / gridsample[0][0]);
			for (int h = 0; h < bh; h++) // middle
			{
				for (int w = 0; w < outwidth; w = w + 2) // not skip first
				{
					const __m128 gridcorrection = _mm_mul_ps(gridfraction4, _mm_load_ps(gridsample[w])); //gridcorrection
					__m128 r1 = _mm_load_ps(outcur[w]);
					__m128 r4 = _mm_sub_ps(r1, gridcorrection); //corrected
					__m128 r3 = _mm_mul_ps(r4, r4); //corrected^2
					__m128 r2 = _mm_shuffle_ps(r3, r3, _MM_SHUFFLE(0, 3, 0, 1));
					r3 = _mm_add_ps(r3, r2); //psd
					__m128 psd4 = _mm_add_ps(r3, _mm_set1_ps(1e-15f)); //psd [0,2]
					psd4 = _mm_shuffle_ps(psd4, psd4, _MM_SHUFFLE(2, 2, 0, 0));
					r2 = _mm_sub_ps(psd4, _mm_load1_ps(&sigmaSquaredNoiseNormed));
					r3 = _mm_div_ps(r2, psd4);
					__m128 wienerfactor4 = _mm_max_ps(r3, _mm_load1_ps(&lowlimit)); //wienerfactor

					__m128 r5 = _mm_add_ps(psd4, _mm_load1_ps(&ht2n));
					__m128 r6 = _mm_mul_ps(psd4, _mm_load1_ps(&dehalo));
					__m128 r7 = _mm_loadl_pi(_mm_setzero_ps(), (__m64*)&wdehalo[w]);
					r6 = _mm_mul_ps(r6, _mm_shuffle_ps(r7, r7, _MM_SHUFFLE(1, 1, 0, 0)));
					r6 = _mm_add_ps(r5, r6);

					r5 = _mm_div_ps(r5, r6);
					wienerfactor4 = _mm_mul_ps(r5, wienerfactor4);

					r1 = _mm_mul_ps(r4, wienerfactor4);//corrected
					r1 = _mm_add_ps(gridcorrection, r1);// final
					_mm_store_ps(outcur[w], r1);
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
			const __m128 gridfraction4 = _mm_set1_ps(degrid * outcur[0][0] / gridsample[0][0]);
			for (int h = 0; h < bh; h++) // middle
			{
				for (int w = 0; w < outwidth; w = w + 2) // not skip first
				{
					const __m128 gridcorrection = _mm_mul_ps(gridfraction4, _mm_load_ps(gridsample[w])); //gridcorrection
					__m128 r1 = _mm_load_ps(outcur[w]);
					__m128 r4 = _mm_sub_ps(r1, gridcorrection); //corrected
					__m128 r3 = _mm_mul_ps(r4, r4); //corrected^2
					__m128 r2 = _mm_shuffle_ps(r3, r3, _MM_SHUFFLE(0, 3, 0, 1));
					r3 = _mm_add_ps(r3, r2); //psd
					__m128 psd4 = _mm_add_ps(r3, _mm_set1_ps(1e-15f)); //psd [0,2]
					psd4 = _mm_shuffle_ps(psd4, psd4, _MM_SHUFFLE(2, 2, 0, 0));
					r2 = _mm_sub_ps(psd4, _mm_load1_ps(&sigmaSquaredNoiseNormed));
					r3 = _mm_div_ps(r2, psd4);
					__m128 wienerfactor4 = _mm_max_ps(r3, _mm_load1_ps(&lowlimit)); //wienerfactor

					__m128 r5 = _mm_mul_ps(_mm_load1_ps(&sharpen), _mm_loadl_pi(_mm_setzero_ps(), (__m64*)&wsharpen[w]));
					__m128 r6 = _mm_add_ps(psd4, _mm_load1_ps(&sigmaSquaredSharpenMax));
					__m128 r7 = _mm_add_ps(psd4, _mm_load1_ps(&sigmaSquaredSharpenMin));
					__m128 r8 = _mm_mul_ps(psd4, _mm_load1_ps(&sigmaSquaredSharpenMax));

					r7 = _mm_mul_ps(r6, r7);
					r8 = _mm_div_ps(r8, r7);
					r6 = _mm_sqrt_ps(r8);

					r5 = _mm_shuffle_ps(r5, r5, _MM_SHUFFLE(1, 1, 0, 0));
					__m128 sharp4 = _mm_mul_ps(r5, r6);

					r5 = _mm_add_ps(psd4, _mm_load1_ps(&ht2n));
					r6 = _mm_mul_ps(psd4, _mm_load1_ps(&dehalo));
					r7 = _mm_loadl_pi(_mm_setzero_ps(), (__m64*)&wdehalo[w]);
					r6 = _mm_mul_ps(r6, _mm_shuffle_ps(r7, r7, _MM_SHUFFLE(1, 1, 0, 0)));
					r6 = _mm_add_ps(r5, r6);
					__m128 dehalo4 = _mm_div_ps(r5, r6);

					r5 = _mm_mul_ps(sharp4, dehalo4);

					r5 = _mm_add_ps(_mm_set1_ps(1.0f), r5);
					wienerfactor4 = _mm_mul_ps(r5, wienerfactor4);

					r1 = _mm_mul_ps(r4, wienerfactor4);//corrected
					r1 = _mm_add_ps(gridcorrection, r1);// final
					_mm_store_ps(outcur[w], r1);
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

void WienerFilter::ApplyWiener3D2_SSE() noexcept
{
	// dft 3d (very short - 2 points)
	for (int block = start_block; block < blocks; block++)
	{
		for (int h = 0; h < bh; h++)
		{
			__m128 r2 = _mm_load_ps(outprev[0]);
			for (int w = 0; w < outwidth; w = w + 2)
			{
				__m128 r1 = _mm_load_ps(outcur[w]);
				__m128 sum4 = _mm_add_ps(r1, r2);
				__m128 dif4 = _mm_sub_ps(r1, r2);
				r1 = _mm_mul_ps(sum4, sum4);
				r2 = _mm_mul_ps(dif4, dif4);

				__m128 r3 = _mm_shuffle_ps(r1, r1, _MM_SHUFFLE(0, 3, 0, 1));
				__m128 r4 = _mm_shuffle_ps(r2, r2, _MM_SHUFFLE(0, 3, 0, 1));

				__m128 psds4 = _mm_add_ps(r1, r3);
				__m128 psdd4 = _mm_add_ps(r2, r4);

				psds4 = _mm_add_ps(psds4, _mm_set1_ps(1e-15f));
				psdd4 = _mm_add_ps(psdd4, _mm_set1_ps(1e-15f));

				r1 = _mm_sub_ps(psds4, _mm_load1_ps(&sigmaSquaredNoiseNormed));
				r2 = _mm_sub_ps(psdd4, _mm_load1_ps(&sigmaSquaredNoiseNormed));
				r1 = _mm_div_ps(r1, psds4);
				r2 = _mm_div_ps(r2, psdd4);

				__m128 WienerFactors4 = _mm_max_ps(r1, _mm_load1_ps(&lowlimit));
				__m128 WienerFactord4 = _mm_max_ps(r2, _mm_load1_ps(&lowlimit));

				r1 = _mm_shuffle_ps(WienerFactors4, WienerFactors4, _MM_SHUFFLE(2, 2, 0, 0));
				r2 = _mm_shuffle_ps(WienerFactord4, WienerFactord4, _MM_SHUFFLE(2, 2, 0, 0));

				sum4 = _mm_mul_ps(sum4, r1);
				dif4 = _mm_mul_ps(dif4, r2);

				r1 = _mm_add_ps(sum4, dif4);
				r2 = _mm_load_ps(outprev[w + 2]);
				r1 = _mm_mul_ps(r1, _mm_set1_ps(0.5f));
				_mm_store_ps(outprev[w], r1);
			}
			_mm_prefetch((const char*)(outprev + outpitch), _MM_HINT_T0);
			_mm_prefetch((const char*)(outcur + outpitch), _MM_HINT_T0);

			outcur += outpitch;
			outprev += outpitch;
		}
	}
}

void WienerFilter::ApplyWiener3D2_degrid_SSE() noexcept
{
	// dft 3d (very short - 2 points)
	for (int block = start_block; block < blocks; block++)
	{
		const __m128 gridfraction4 = _mm_set1_ps(degrid * outcur[0][0] / gridsample[0][0]);
		for (int h = 0; h < bh; h++)
		{
			__m128 r2 = _mm_load_ps(outprev[0]);
			for (int w = 0; w < outwidth; w = w + 2)
			{
				__m128 gridcorrection = _mm_mul_ps(gridfraction4, _mm_load_ps(gridsample[w])); //gridcorrection
				gridcorrection = _mm_mul_ps(gridcorrection, _mm_set1_ps(2.0f));
				__m128 r1 = _mm_load_ps(outcur[w]);
				__m128 sum4 = _mm_add_ps(r1, r2);
				sum4 = _mm_sub_ps(sum4, gridcorrection);
				__m128 dif4 = _mm_sub_ps(r1, r2);
				r1 = _mm_mul_ps(sum4, sum4);
				r2 = _mm_mul_ps(dif4, dif4);

				__m128 r3 = _mm_shuffle_ps(r1, r1, _MM_SHUFFLE(0, 3, 0, 1));
				__m128 r4 = _mm_shuffle_ps(r2, r2, _MM_SHUFFLE(0, 3, 0, 1));

				__m128 psds4 = _mm_add_ps(r1, r3);
				__m128 psdd4 = _mm_add_ps(r2, r4);

				psds4 = _mm_add_ps(psds4, _mm_set1_ps(1e-15f));
				psdd4 = _mm_add_ps(psdd4, _mm_set1_ps(1e-15f));

				r1 = _mm_sub_ps(psds4, _mm_load1_ps(&sigmaSquaredNoiseNormed));
				r2 = _mm_sub_ps(psdd4, _mm_load1_ps(&sigmaSquaredNoiseNormed));
				r1 = _mm_div_ps(r1, psds4);
				r2 = _mm_div_ps(r2, psdd4);

				__m128 WienerFactors4 = _mm_max_ps(r1, _mm_load1_ps(&lowlimit));
				__m128 WienerFactord4 = _mm_max_ps(r2, _mm_load1_ps(&lowlimit));

				r1 = _mm_shuffle_ps(WienerFactors4, WienerFactors4, _MM_SHUFFLE(2, 2, 0, 0));
				r2 = _mm_shuffle_ps(WienerFactord4, WienerFactord4, _MM_SHUFFLE(2, 2, 0, 0));

				sum4 = _mm_mul_ps(sum4, r1);
				dif4 = _mm_mul_ps(dif4, r2);

				r1 = _mm_add_ps(sum4, dif4);
				r1 = _mm_add_ps(gridcorrection, r1);
				r2 = _mm_load_ps(outprev[w + 2]);
				r1 = _mm_mul_ps(r1, _mm_set1_ps(0.5f));
				_mm_store_ps(outprev[w], r1);
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

//-----------------------------------------------------------------------------------------
//
void WienerFilter::ApplyWiener3D3_SSE() noexcept
{
	// dft 3d (very short - 3 points)
	constexpr float sin120 = 0.86602540378443864676372317075294f;//sqrtf(3.0f)*0.5f;

	for (int block = start_block; block < blocks; block++)
	{
		for (int h = 0; h < bh; h++) // first half
		{
			__m128 r2 = _mm_load_ps(outnext[0]);
			__m128 r3 = _mm_load_ps(outcur[0]);
			for (int w = 0; w < outwidth; w = w + 2) // 
			{
				__m128 r1 = _mm_load_ps(outprev[w]);
				__m128 pn4 = _mm_add_ps(r1, r2); //r,i,r,i
				__m128 fc4 = _mm_add_ps(pn4, r3); //r,i,r,i

				__m128 d4 = _mm_sub_ps(r1, r2); //r,i,r,i!
				d4 = _mm_mul_ps(d4, _mm_set_ps(1.0f, -1.0f, 1.0f, -1.0f));
				d4 = _mm_mul_ps(d4, _mm_load1_ps(&sin120));
				d4 = _mm_shuffle_ps(d4, d4, _MM_SHUFFLE(2, 3, 0, 1)); //d4 = i,r,i,r
				r1 = _mm_mul_ps(pn4, _mm_set1_ps(0.5f));
				r1 = _mm_sub_ps(r3, r1);
				__m128 fp4 = _mm_add_ps(r1, d4);
				__m128 fn4 = _mm_sub_ps(r1, d4);

				__m128 psdc4 = _mm_mul_ps(fc4, fc4);
				__m128 psdp4 = _mm_mul_ps(fp4, fp4);
				__m128 psdn4 = _mm_mul_ps(fn4, fn4);

				r1 = _mm_shuffle_ps(psdc4, psdc4, _MM_SHUFFLE(0, 3, 0, 1));//psd,-,psd,-
				r2 = _mm_shuffle_ps(psdp4, psdp4, _MM_SHUFFLE(0, 3, 0, 1));
				r3 = _mm_shuffle_ps(psdn4, psdn4, _MM_SHUFFLE(0, 3, 0, 1));

				psdc4 = _mm_add_ps(psdc4, r1);
				psdp4 = _mm_add_ps(psdp4, r2);
				psdn4 = _mm_add_ps(psdn4, r3);

				psdc4 = _mm_add_ps(psdc4, _mm_set1_ps(1e-15f));
				psdp4 = _mm_add_ps(psdp4, _mm_set1_ps(1e-15f));
				psdn4 = _mm_add_ps(psdn4, _mm_set1_ps(1e-15f));

				r1 = _mm_sub_ps(psdc4, _mm_load1_ps(&sigmaSquaredNoiseNormed));
				r2 = _mm_sub_ps(psdp4, _mm_load1_ps(&sigmaSquaredNoiseNormed));
				r3 = _mm_sub_ps(psdn4, _mm_load1_ps(&sigmaSquaredNoiseNormed));

				r1 = _mm_div_ps(r1, psdc4);
				r2 = _mm_div_ps(r2, psdp4);
				r3 = _mm_div_ps(r3, psdn4);

				r1 = _mm_max_ps(r1, _mm_load1_ps(&lowlimit));
				r2 = _mm_max_ps(r2, _mm_load1_ps(&lowlimit));
				r3 = _mm_max_ps(r3, _mm_load1_ps(&lowlimit));

				r1 = _mm_shuffle_ps(r1, r1, _MM_SHUFFLE(2, 2, 0, 0));
				r2 = _mm_shuffle_ps(r2, r2, _MM_SHUFFLE(2, 2, 0, 0));
				r3 = _mm_shuffle_ps(r3, r3, _MM_SHUFFLE(2, 2, 0, 0));

				fc4 = _mm_mul_ps(r1, fc4);
				fp4 = _mm_mul_ps(r2, fp4);
				fn4 = _mm_mul_ps(r3, fn4);

				r1 = _mm_add_ps(fc4, fp4);
				r1 = _mm_add_ps(r1, fn4);
				r2 = _mm_load_ps(outnext[w + 2]);
				r3 = _mm_load_ps(outcur[w + 2]);
				r1 = _mm_mul_ps(r1, _mm_set1_ps(0.33333333333f));

				_mm_store_ps(outprev[w], r1);
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

//-----------------------------------------------------------------------------------------
//
void WienerFilter::ApplyWiener3D3_degrid_SSE() noexcept
{
	// dft 3d (very short - 3 points)
	float sin120 = 0.86602540378443864676372317075294f;//sqrtf(3.0f)*0.5f;

	for (int block = start_block; block < blocks; block++)
	{
		const __m128 gridfraction4 = _mm_set1_ps(degrid * outcur[0][0] / gridsample[0][0]);
		for (int h = 0; h < bh; h++) // first half
		{
			__m128 r2 = _mm_load_ps(outnext[0]);
			__m128 r3 = _mm_load_ps(outcur[0]);
			for (int w = 0; w < outwidth; w = w + 2) // 
			{
				__m128 gridcorrection = _mm_mul_ps(gridfraction4, _mm_load_ps(gridsample[w])); //gridcorrection
				gridcorrection = _mm_mul_ps(gridcorrection, _mm_set1_ps(3.0f));
				__m128 r1 = _mm_load_ps(outprev[w]);
				__m128 pn4 = _mm_add_ps(r1, r2); //r,i,r,i
				__m128 fc4 = _mm_add_ps(pn4, r3); //r,i,r,i
				fc4 = _mm_sub_ps(fc4, gridcorrection);

				__m128 d4 = _mm_sub_ps(r1, r2); //r,i,r,i!
				d4 = _mm_mul_ps(d4, _mm_set_ps(1.0f, -1.0f, 1.0f, -1.0f));
				d4 = _mm_mul_ps(d4, _mm_load1_ps(&sin120));
				d4 = _mm_shuffle_ps(d4, d4, _MM_SHUFFLE(2, 3, 0, 1)); //d4 = i,r,i,r
				r1 = _mm_mul_ps(pn4, _mm_set1_ps(0.5f));
				r1 = _mm_sub_ps(r3, r1);
				__m128 fp4 = _mm_add_ps(r1, d4);
				__m128 fn4 = _mm_sub_ps(r1, d4);

				__m128 psdc4 = _mm_mul_ps(fc4, fc4);
				__m128 psdp4 = _mm_mul_ps(fp4, fp4);
				__m128 psdn4 = _mm_mul_ps(fn4, fn4);

				r1 = _mm_shuffle_ps(psdc4, psdc4, _MM_SHUFFLE(0, 3, 0, 1));//psd,-,psd,-
				r2 = _mm_shuffle_ps(psdp4, psdp4, _MM_SHUFFLE(0, 3, 0, 1));
				r3 = _mm_shuffle_ps(psdn4, psdn4, _MM_SHUFFLE(0, 3, 0, 1));

				psdc4 = _mm_add_ps(psdc4, r1);
				psdp4 = _mm_add_ps(psdp4, r2);
				psdn4 = _mm_add_ps(psdn4, r3);

				psdc4 = _mm_add_ps(psdc4, _mm_set1_ps(1e-15f));
				psdp4 = _mm_add_ps(psdp4, _mm_set1_ps(1e-15f));
				psdn4 = _mm_add_ps(psdn4, _mm_set1_ps(1e-15f));

				r1 = _mm_sub_ps(psdc4, _mm_load1_ps(&sigmaSquaredNoiseNormed));
				r2 = _mm_sub_ps(psdp4, _mm_load1_ps(&sigmaSquaredNoiseNormed));
				r3 = _mm_sub_ps(psdn4, _mm_load1_ps(&sigmaSquaredNoiseNormed));

				r1 = _mm_div_ps(r1, psdc4);
				r2 = _mm_div_ps(r2, psdp4);
				r3 = _mm_div_ps(r3, psdn4);

				r1 = _mm_max_ps(r1, _mm_load1_ps(&lowlimit));
				r2 = _mm_max_ps(r2, _mm_load1_ps(&lowlimit));
				r3 = _mm_max_ps(r3, _mm_load1_ps(&lowlimit));

				r1 = _mm_shuffle_ps(r1, r1, _MM_SHUFFLE(2, 2, 0, 0));
				r2 = _mm_shuffle_ps(r2, r2, _MM_SHUFFLE(2, 2, 0, 0));
				r3 = _mm_shuffle_ps(r3, r3, _MM_SHUFFLE(2, 2, 0, 0));

				fc4 = _mm_mul_ps(r1, fc4);
				fp4 = _mm_mul_ps(r2, fp4);
				fn4 = _mm_mul_ps(r3, fn4);

				r1 = _mm_add_ps(fc4, fp4);
				r1 = _mm_add_ps(r1, fn4);
				r2 = _mm_load_ps(outnext[w + 2]);
				r3 = _mm_load_ps(outcur[w + 2]);
				r1 = _mm_add_ps(r1, gridcorrection);
				r1 = _mm_mul_ps(r1, _mm_set1_ps(0.33333333333f));

				_mm_store_ps(outprev[w], r1);
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

void WienerFilter::ApplyWiener3D4_SSE() noexcept
{
	// dft with 4 points
	for (int block = start_block; block < blocks; block++)
	{
		for (int h = 0; h < bh; h++) // first half
		{
			__m128 r3 = _mm_load_ps(&outcur[0][0]);
			__m128 r4 = _mm_load_ps(&outnext[0][0]);
			for (int w = 0; w < outwidth; w = w + 2)
			{
				__m128 r1 = _mm_load_ps(&outprev2[w][0]);
				__m128 r2 = _mm_load_ps(&outprev[w][0]);
				//outcur[w][0] - outnext[w][1] - outprev2[w][0] + outprev[w][1]
				//outcur[w][1] + outnext[w][0] - outprev2[w][1] - outprev[w][0]
				__m128 r5 = _mm_shuffle_ps(r4, r4, _MM_SHUFFLE(1, 0, 3, 2)); //r5: swapped outnext
				__m128 r6 = _mm_shuffle_ps(r2, r2, _MM_SHUFFLE(1, 0, 3, 2)); //r6: swapped outprev
				r5 = _mm_mul_ps(r5, _mm_set_ps(-1.0f, 1.0f, -1.0f, 1.0f)); //r5,6: negation
				r6 = _mm_mul_ps(r6, _mm_set_ps(1.0f, -1.0f, 1.0f, -1.0f));

				__m128 r7 = _mm_add_ps(r3, r5); //outcur + outnext
				r7 = _mm_sub_ps(r7, r1); //-outprev2
				r7 = _mm_add_ps(r7, r6); // fpr1, fpi1, fpr2, fpi2 = +outprev

										 //outcur[w][0] + outnext[w][1] - outprev2[w][0] - outprev[w][1]
										 //outcur[w][1] - outnext[w][0] - outprev2[w][1] + outprev[w][0]
				r5 = _mm_mul_ps(r5, _mm_set_ps(-1.0f, -1.0f, -1.0f, -1.0f));
				r6 = _mm_mul_ps(r6, _mm_set_ps(-1.0f, -1.0f, -1.0f, -1.0f));

				__m128 r8 = _mm_add_ps(r3, r5); //outcur + outnext
				r8 = _mm_sub_ps(r8, r1); //-outprev2
				r8 = _mm_add_ps(r8, r6); // fnr1, fni1, fnr2, fni2 = +outprev
										 //
										 //
				r5 = _mm_add_ps(r1, r2);
				r6 = _mm_add_ps(r3, r4);
				r5 = _mm_add_ps(r5, r6); // fcr1, fci1, fcr2, fci2

				r1 = _mm_sub_ps(r1, r2);
				r2 = _mm_sub_ps(r3, r4);
				r6 = _mm_add_ps(r1, r2); // fp2r1, fp2i1, fp2r2, fp2i2
										 //r7: fpr1, fpi1, fpr2, fpi2
										 //r8: fnr1, fni1, fnr2, fni2
										 //r5: fcr1, fci1, fcr2, fci2
										 //r6: fp2r1, fp2i1, fp2r2, fp2i2

				r1 = _mm_mul_ps(r7, r7);
				r2 = _mm_mul_ps(r8, r8);
				r3 = _mm_mul_ps(r5, r5);
				r4 = _mm_mul_ps(r6, r6);
				//r1: fpr1, fpi1, fpr2, fpi2
				//r2: fnr1, fni1, fnr2, fni2
				//r3: fcr1, fci1, fcr2, fci2
				//r4: fp2r1, fp2i1, fp2r2, fp2i2
				//
				//
				//t1: fpr1, fpr2, fnr1, fnr2
				//t2: fpi1, fpi2, fni1, fni2
				//t3: fcr1, fcr2, fp2r1, fp2r2
				//t4: fci1, fci2, fp2i1, fp2i2
				__m128 sh1 = _mm_shuffle_ps(r1, r2, _MM_SHUFFLE(2, 0, 2, 0));
				__m128 sh2 = _mm_shuffle_ps(r1, r2, _MM_SHUFFLE(3, 1, 3, 1));
				__m128 sh3 = _mm_shuffle_ps(r3, r4, _MM_SHUFFLE(2, 0, 2, 0));
				__m128 sh4 = _mm_shuffle_ps(r3, r4, _MM_SHUFFLE(3, 1, 3, 1));

				r1 = _mm_add_ps(sh1, sh2);
				r2 = _mm_add_ps(sh3, sh4);
				r1 = _mm_add_ps(r1, _mm_set1_ps(1e-15f));
				r2 = _mm_add_ps(r2, _mm_set1_ps(1e-15f));
				//r1: psd_fp1, psd_fp2, psd_fn1, psd_fn2
				//r2: psd_fc1, psd_fc2, psd_fp2_1, psd_fp2_2
				r3 = _mm_sub_ps(r1, _mm_load1_ps(&sigmaSquaredNoiseNormed));
				r4 = _mm_sub_ps(r2, _mm_load1_ps(&sigmaSquaredNoiseNormed));
				r1 = _mm_div_ps(r3, r1);
				r2 = _mm_div_ps(r4, r2);
				r1 = _mm_max_ps(r1, _mm_load1_ps(&lowlimit));
				r2 = _mm_max_ps(r2, _mm_load1_ps(&lowlimit));
				//r1: wf_fp1, wf_fp2, wf_fn1, wf_fn2
				//r2: wf_fc1, wf_fc2, wf_fp2_1, wf_fp2_2
				//r5: fcr1, fci1, fcr2, fci2 ; fc1, fc2
				//r6: fp2r1, fp2i1, fp2r2, fp2i2 ; fp2_1, fp2_2
				//r7: fpr1, fpi1, fpr2, fpi2 ; fp1, fp2
				//r8: fnr1, fni1, fnr2, fni2 ; fn1, fn2

				r7 = _mm_mul_ps(r7, _mm_unpacklo_ps(r1, r1));
				r8 = _mm_mul_ps(r8, _mm_unpackhi_ps(r1, r1));
				r5 = _mm_mul_ps(r5, _mm_unpacklo_ps(r2, r2));
				r6 = _mm_mul_ps(r6, _mm_unpackhi_ps(r2, r2));

				r1 = _mm_add_ps(r5, r6);
				r2 = _mm_add_ps(r7, r8);
				r1 = _mm_add_ps(r1, r2);
				r1 = _mm_mul_ps(r1, _mm_set1_ps(0.25f));

				r3 = _mm_load_ps(&outcur[w + 2][0]);
				r4 = _mm_load_ps(&outnext[w + 2][0]);
				_mm_store_ps(&outprev2[w][0], r1);	 // Attention! return filtered "out" in "outprev2" to preserve "out" for next step
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

//-----------------------------------------------------------------------------------------
//
void WienerFilter::ApplyWiener3D4_degrid_SSE() noexcept
{
	// dft with 4 points
	_mm_prefetch((const char*)outprev2, _MM_HINT_T0);
	_mm_prefetch((const char*)outprev, _MM_HINT_T0);
	_mm_prefetch((const char*)outcur, _MM_HINT_T0);
	_mm_prefetch((const char*)outnext, _MM_HINT_T0);

	for (int block = start_block; block < blocks; block++)
	{
		const __m128 gridfraction4 = _mm_set1_ps(degrid * outcur[0][0] / gridsample[0][0]);
		for (int h = 0; h < bh; h++) // first half
		{
			__m128 r3 = _mm_load_ps(&outcur[0][0]);
			__m128 r4 = _mm_load_ps(&outnext[0][0]);
			for (int w = 0; w < outwidth; w = w + 2)
			{
				__m128 gridcorrection = _mm_mul_ps(_mm_load_ps(&gridsample[w][0]), gridfraction4);
				gridcorrection = _mm_mul_ps(gridcorrection, _mm_set1_ps(4));
				__m128 r1 = _mm_load_ps(&outprev2[w][0]);
				__m128 r2 = _mm_load_ps(&outprev[w][0]);
				//outcur[w][0] - outnext[w][1] - outprev2[w][0] + outprev[w][1]
				//outcur[w][1] + outnext[w][0] - outprev2[w][1] - outprev[w][0]
				__m128 r5 = _mm_shuffle_ps(r4, r4, _MM_SHUFFLE(1, 0, 3, 2)); //r5: swapped outnext
				__m128 r6 = _mm_shuffle_ps(r2, r2, _MM_SHUFFLE(1, 0, 3, 2)); //r6: swapped outprev
				r5 = _mm_mul_ps(r5, _mm_set_ps(-1.0f, 1.0f, -1.0f, 1.0f)); //r5,6: negation
				r6 = _mm_mul_ps(r6, _mm_set_ps(1.0f, -1.0f, 1.0f, -1.0f));

				__m128 r7 = _mm_add_ps(r3, r5); //outcur + outnext
				r7 = _mm_sub_ps(r7, r1); //-outprev2
				r7 = _mm_add_ps(r7, r6); // fpr1, fpi1, fpr2, fpi2 = +outprev

				//outcur[w][0] + outnext[w][1] - outprev2[w][0] - outprev[w][1]
				//outcur[w][1] - outnext[w][0] - outprev2[w][1] + outprev[w][0]
				r5 = _mm_mul_ps(r5, _mm_set_ps(-1.0f, -1.0f, -1.0f, -1.0f));
				r6 = _mm_mul_ps(r6, _mm_set_ps(-1.0f, -1.0f, -1.0f, -1.0f));

				__m128 r8 = _mm_add_ps(r3, r5); //outcur + outnext
				r8 = _mm_sub_ps(r8, r1); //-outprev2
				r8 = _mm_add_ps(r8, r6); // fnr1, fni1, fnr2, fni2 = +outprev
				//
				//
				r5 = _mm_add_ps(r1, r2);
				r6 = _mm_add_ps(r3, r4);
				r5 = _mm_add_ps(r5, r6); // fcr1, fci1, fcr2, fci2
				r5 = _mm_sub_ps(r5, gridcorrection);

				r1 = _mm_sub_ps(r1, r2);
				r2 = _mm_sub_ps(r3, r4);
				r6 = _mm_add_ps(r1, r2); // fp2r1, fp2i1, fp2r2, fp2i2
				//r7: fpr1, fpi1, fpr2, fpi2
				//r8: fnr1, fni1, fnr2, fni2
				//r5: fcr1, fci1, fcr2, fci2
				//r6: fp2r1, fp2i1, fp2r2, fp2i2

				r1 = _mm_mul_ps(r7, r7);
				r2 = _mm_mul_ps(r8, r8);
				r3 = _mm_mul_ps(r5, r5);
				r4 = _mm_mul_ps(r6, r6);
				//r1: fpr1, fpi1, fpr2, fpi2
				//r2: fnr1, fni1, fnr2, fni2
				//r3: fcr1, fci1, fcr2, fci2
				//r4: fp2r1, fp2i1, fp2r2, fp2i2
				//
				//
				//t1: fpr1, fpr2, fnr1, fnr2
				//t2: fpi1, fpi2, fni1, fni2
				//t3: fcr1, fcr2, fp2r1, fp2r2
				//t4: fci1, fci2, fp2i1, fp2i2
				__m128 sh1 = _mm_shuffle_ps(r1, r2, _MM_SHUFFLE(2, 0, 2, 0));
				__m128 sh2 = _mm_shuffle_ps(r1, r2, _MM_SHUFFLE(3, 1, 3, 1));
				__m128 sh3 = _mm_shuffle_ps(r3, r4, _MM_SHUFFLE(2, 0, 2, 0));
				__m128 sh4 = _mm_shuffle_ps(r3, r4, _MM_SHUFFLE(3, 1, 3, 1));

				r1 = _mm_add_ps(sh1, sh2);
				r2 = _mm_add_ps(sh3, sh4);
				r1 = _mm_add_ps(r1, _mm_set1_ps(1e-15f));
				r2 = _mm_add_ps(r2, _mm_set1_ps(1e-15f));
				//r1: psd_fp1, psd_fp2, psd_fn1, psd_fn2
				//r2: psd_fc1, psd_fc2, psd_fp2_1, psd_fp2_2
				r3 = _mm_sub_ps(r1, _mm_load1_ps(&sigmaSquaredNoiseNormed));
				r4 = _mm_sub_ps(r2, _mm_load1_ps(&sigmaSquaredNoiseNormed));
				r1 = _mm_div_ps(r3, r1);
				r2 = _mm_div_ps(r4, r2);
				r1 = _mm_max_ps(r1, _mm_load1_ps(&lowlimit));
				r2 = _mm_max_ps(r2, _mm_load1_ps(&lowlimit));
				//r1: wf_fp1, wf_fp2, wf_fn1, wf_fn2
				//r2: wf_fc1, wf_fc2, wf_fp2_1, wf_fp2_2
				//r5: fcr1, fci1, fcr2, fci2 ; fc1, fc2
				//r6: fp2r1, fp2i1, fp2r2, fp2i2 ; fp2_1, fp2_2
				//r7: fpr1, fpi1, fpr2, fpi2 ; fp1, fp2
				//r8: fnr1, fni1, fnr2, fni2 ; fn1, fn2

				r7 = _mm_mul_ps(r7, _mm_unpacklo_ps(r1, r1));
				r8 = _mm_mul_ps(r8, _mm_unpackhi_ps(r1, r1));
				r5 = _mm_mul_ps(r5, _mm_unpacklo_ps(r2, r2));
				r6 = _mm_mul_ps(r6, _mm_unpackhi_ps(r2, r2));

				r1 = _mm_add_ps(r5, r6);
				r2 = _mm_add_ps(r7, r8);
				r1 = _mm_add_ps(r1, r2);
				r1 = _mm_mul_ps(_mm_add_ps(r1, gridcorrection), _mm_set1_ps(0.25f));

				r3 = _mm_load_ps(&outcur[w + 2][0]);
				r4 = _mm_load_ps(&outnext[w + 2][0]);
				_mm_store_ps(&outprev2[w][0], r1);	 // Attention! return filtered "out" in "outprev2" to preserve "out" for next step
			}
			_mm_prefetch((const char*)(outprev2 + outpitch), _MM_HINT_T0);
			_mm_prefetch((const char*)(outprev + outpitch), _MM_HINT_T0);
			_mm_prefetch((const char*)(outcur + outpitch), _MM_HINT_T0);
			_mm_prefetch((const char*)(outnext + outpitch), _MM_HINT_T0);

			outcur += outpitch;
			outprev2 += outpitch;
			outprev += outpitch;
			outnext += outpitch;
			gridsample += outpitch;
		}
		gridsample -= outpitch * bh; // restore pointer to only valid first block
	}
}

void WienerFilter::ApplyWiener3D5_SSE() noexcept
{
	// dft with 5 points
	const __m128 sincos72 = _mm_set_ps(0.30901699437494742410229341718282f, 0.95105651629515357211643933337938, 0.30901699437494742410229341718282f, 0.95105651629515357211643933337938);
	const __m128 cossin72 = _mm_set_ps(0.95105651629515357211643933337938, 0.30901699437494742410229341718282f, 0.95105651629515357211643933337938, 0.30901699437494742410229341718282f);
	const __m128 sincos144 = _mm_set_ps(-0.80901699437494742410229341718282f, 0.58778525229247312916870595463907f, -0.80901699437494742410229341718282f, 0.58778525229247312916870595463907f);
	const __m128 cossin144 = _mm_set_ps(0.58778525229247312916870595463907f, -0.80901699437494742410229341718282f, 0.58778525229247312916870595463907f, -0.80901699437494742410229341718282f);

	for (int block = start_block; block < blocks; block++)
	{
		for (int h = 0; h < bh; h++) // first half
		{
			__m128 r3 = _mm_load_ps(outprev[0]);
			__m128 r4 = _mm_load_ps(outnext[0]);
			for (int w = 0; w < outwidth; w = w + 2) // 
			{
				__m128 r1 = _mm_load_ps(outprev2[w]);
				__m128 r2 = _mm_load_ps(outnext2[w]);
				__m128 r5 = _mm_load_ps(outcur[w]);

				__m128 r6 = _mm_add_ps(_mm_mul_ps(r1, _mm_set_ps(-1.0f, 1.0f, -1.0f, 1.0f)), r2); //sum, dif, sum, dif
				__m128 r7 = _mm_add_ps(r3, _mm_mul_ps(r4, _mm_set_ps(-1.0f, 1.0f, -1.0f, 1.0f))); //sum, dif, sum, dif
				r6 = _mm_mul_ps(r6, cossin72);
				r7 = _mm_mul_ps(r7, cossin144);
				r7 = _mm_add_ps(r6, r7);
				r7 = _mm_add_ps(r7, _mm_andnot_ps(_mm_set_ps(~0, 0.0f, ~0, 0.0f), r5));
				r6 = _mm_shuffle_ps(r7, r7, _MM_SHUFFLE(0, 3, 0, 1));
				__m128 fp2r = _mm_add_ps(r7, r6);
				__m128 fn2r = _mm_sub_ps(r7, r6);

				r6 = _mm_add_ps(r1, _mm_mul_ps(r2, _mm_set_ps(1.0f, -1.0f, 1.0f, -1.0f))); //dif, sum, dif, sum
				r7 = _mm_add_ps(_mm_mul_ps(r3, _mm_set_ps(1.0f, -1.0f, 1.0f, -1.0f)), r4); //dif, sum, dif, sum
				r6 = _mm_mul_ps(r6, sincos72);
				r7 = _mm_mul_ps(r7, sincos144);
				r7 = _mm_add_ps(r6, r7);
				r7 = _mm_add_ps(r7, _mm_andnot_ps(_mm_set_ps(0.0f, ~0, 0.0f, ~0), r5));
				r6 = _mm_shuffle_ps(r7, r7, _MM_SHUFFLE(0, 3, 0, 1)); //dif, sum -> r6 == sum!
				__m128 fp2i = _mm_add_ps(r7, r6);
				__m128 fn2i = _mm_sub_ps(r6, r7);

				r6 = _mm_add_ps(r1, _mm_mul_ps(r2, _mm_set_ps(-1.0f, 1.0f, -1.0f, 1.0f))); //sum, dif, sum, dif
				r7 = _mm_add_ps(r3, _mm_mul_ps(r4, _mm_set_ps(-1.0f, 1.0f, -1.0f, 1.0f))); //sum, dif, sum, dif
				r6 = _mm_mul_ps(r6, cossin144);
				r7 = _mm_mul_ps(r7, cossin72);
				r7 = _mm_add_ps(r6, r7);
				r7 = _mm_add_ps(r7, _mm_andnot_ps(_mm_set_ps(~0, 0.0f, ~0, 0.0f), r5));
				r6 = _mm_shuffle_ps(r7, r7, _MM_SHUFFLE(0, 3, 0, 1));
				__m128 fpr = _mm_add_ps(r7, r6);
				__m128 fnr = _mm_sub_ps(r7, r6);

				r6 = _mm_add_ps(_mm_mul_ps(r1, _mm_set_ps(1.0f, -1.0f, 1.0f, -1.0f)), r2); //dif, sum, dif, sum
				r7 = _mm_add_ps(_mm_mul_ps(r3, _mm_set_ps(1.0f, -1.0f, 1.0f, -1.0f)), r4); //dif, sum, dif, sum
				r6 = _mm_mul_ps(r6, sincos144);
				r7 = _mm_mul_ps(r7, sincos72);
				r7 = _mm_add_ps(r6, r7);
				r7 = _mm_add_ps(r7, _mm_andnot_ps(_mm_set_ps(0.0f, ~0, 0.0f, ~0), r5));
				r6 = _mm_shuffle_ps(r7, r7, _MM_SHUFFLE(0, 3, 0, 1)); //dif, sum -> r6 == sum!
				__m128 fpi = _mm_add_ps(r7, r6);
				__m128 fni = _mm_sub_ps(r6, r7);

				r6 = _mm_add_ps(r1, r2);
				r7 = _mm_add_ps(r3, r4);
				r6 = _mm_add_ps(r5, r6);
				__m128 fc = _mm_add_ps(r6, r7);

				__m128 psd = _mm_mul_ps(fc, fc);
				r1 = _mm_shuffle_ps(psd, psd, _MM_SHUFFLE(0, 3, 0, 1));//psd,-,psd,-
				psd = _mm_add_ps(psd, r1);
				psd = _mm_add_ps(psd, _mm_set1_ps(1e-15f));
				r1 = _mm_sub_ps(psd, _mm_load1_ps(&sigmaSquaredNoiseNormed));
				r1 = _mm_div_ps(r1, psd);
				r1 = _mm_max_ps(r1, _mm_load1_ps(&lowlimit));
				r1 = _mm_shuffle_ps(r1, r1, _MM_SHUFFLE(2, 2, 0, 0));
				fc = _mm_mul_ps(r1, fc);

				r1 = _mm_mul_ps(fp2r, fp2r);
				r2 = _mm_mul_ps(fp2i, fp2i);
				fp2r = _mm_shuffle_ps(fp2r, fp2i, _MM_SHUFFLE(2, 0, 2, 0));
				fp2r = _mm_shuffle_ps(fp2r, fp2r, _MM_SHUFFLE(3, 1, 2, 0));
				psd = _mm_add_ps(r1, r2);
				psd = _mm_add_ps(psd, _mm_set1_ps(1e-15f));
				r1 = _mm_sub_ps(psd, _mm_load1_ps(&sigmaSquaredNoiseNormed));
				r1 = _mm_div_ps(r1, psd);
				r1 = _mm_max_ps(r1, _mm_load1_ps(&lowlimit));
				r1 = _mm_shuffle_ps(r1, r1, _MM_SHUFFLE(2, 2, 0, 0));
				fp2r = _mm_mul_ps(r1, fp2r);

				r1 = _mm_mul_ps(fpr, fpr);
				r2 = _mm_mul_ps(fpi, fpi);
				fpr = _mm_shuffle_ps(fpr, fpi, _MM_SHUFFLE(2, 0, 2, 0));
				fpr = _mm_shuffle_ps(fpr, fpr, _MM_SHUFFLE(3, 1, 2, 0));
				psd = _mm_add_ps(r1, r2);
				psd = _mm_add_ps(psd, _mm_set1_ps(1e-15f));
				r1 = _mm_sub_ps(psd, _mm_load1_ps(&sigmaSquaredNoiseNormed));
				r1 = _mm_div_ps(r1, psd);
				r1 = _mm_max_ps(r1, _mm_load1_ps(&lowlimit));
				r1 = _mm_shuffle_ps(r1, r1, _MM_SHUFFLE(2, 2, 0, 0));
				fpr = _mm_mul_ps(r1, fpr);

				r1 = _mm_mul_ps(fnr, fnr);
				r2 = _mm_mul_ps(fni, fni);
				fnr = _mm_shuffle_ps(fnr, fni, _MM_SHUFFLE(2, 0, 2, 0));
				fnr = _mm_shuffle_ps(fnr, fnr, _MM_SHUFFLE(3, 1, 2, 0));
				psd = _mm_add_ps(r1, r2);
				psd = _mm_add_ps(psd, _mm_set1_ps(1e-15f));
				r1 = _mm_sub_ps(psd, _mm_load1_ps(&sigmaSquaredNoiseNormed));
				r1 = _mm_div_ps(r1, psd);
				r1 = _mm_max_ps(r1, _mm_load1_ps(&lowlimit));
				r1 = _mm_shuffle_ps(r1, r1, _MM_SHUFFLE(2, 2, 0, 0));
				fnr = _mm_mul_ps(r1, fnr);

				r1 = _mm_mul_ps(fn2r, fn2r);
				r2 = _mm_mul_ps(fn2i, fn2i);
				fn2r = _mm_shuffle_ps(fn2r, fn2i, _MM_SHUFFLE(2, 0, 2, 0));
				fn2r = _mm_shuffle_ps(fn2r, fn2r, _MM_SHUFFLE(3, 1, 2, 0));
				psd = _mm_add_ps(r1, r2);
				psd = _mm_add_ps(psd, _mm_set1_ps(1e-15f));
				r1 = _mm_sub_ps(psd, _mm_load1_ps(&sigmaSquaredNoiseNormed));
				r1 = _mm_div_ps(r1, psd);
				r1 = _mm_max_ps(r1, _mm_load1_ps(&lowlimit));
				r1 = _mm_shuffle_ps(r1, r1, _MM_SHUFFLE(2, 2, 0, 0));
				fn2r = _mm_mul_ps(r1, fn2r);

				r1 = _mm_add_ps(fp2r, fpr);
				r2 = _mm_add_ps(fc, fnr);
				r1 = _mm_add_ps(r1, fn2r);
				r1 = _mm_add_ps(r1, r2);
				r3 = _mm_load_ps(outprev[w + 2]);
				r4 = _mm_load_ps(outnext[w + 2]);
				r1 = _mm_mul_ps(r1, _mm_set1_ps(0.2f));
				_mm_store_ps(outprev2[w], r1);
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

void WienerFilter::ApplyWiener3D5_degrid_SSE() noexcept
{
	// dft with 5 points
	const __m128 sincos72 = _mm_set_ps(0.30901699437494742410229341718282f, 0.95105651629515357211643933337938, 0.30901699437494742410229341718282f, 0.95105651629515357211643933337938);
	const __m128 cossin72 = _mm_set_ps(0.95105651629515357211643933337938, 0.30901699437494742410229341718282f, 0.95105651629515357211643933337938, 0.30901699437494742410229341718282f);
	const __m128 sincos144 = _mm_set_ps(-0.80901699437494742410229341718282f, 0.58778525229247312916870595463907f, -0.80901699437494742410229341718282f, 0.58778525229247312916870595463907f);
	const __m128 cossin144 = _mm_set_ps(0.58778525229247312916870595463907f, -0.80901699437494742410229341718282f, 0.58778525229247312916870595463907f, -0.80901699437494742410229341718282f);

	for (int block = start_block; block < blocks; block++)
	{
		__m128 gridfraction4 = _mm_set1_ps(degrid * outcur[0][0] / gridsample[0][0]);
		for (int h = 0; h < bh; h++) // first half
		{
			__m128 r3 = _mm_load_ps(outprev[0]);
			__m128 r4 = _mm_load_ps(outnext[0]);
			for (int w = 0; w < outwidth; w = w + 2) // 
			{
				__m128 gridcorrection = _mm_mul_ps(_mm_load_ps(&gridsample[w][0]), gridfraction4);
				gridcorrection = _mm_mul_ps(gridcorrection, _mm_set1_ps(5));
				__m128 r1 = _mm_load_ps(outprev2[w]);
				__m128 r2 = _mm_load_ps(outnext2[w]);
				__m128 r5 = _mm_load_ps(outcur[w]);

				__m128 r6 = _mm_add_ps(_mm_mul_ps(r1, _mm_set_ps(-1.0f, 1.0f, -1.0f, 1.0f)), r2); //sum, dif, sum, dif
				__m128 r7 = _mm_add_ps(r3, _mm_mul_ps(r4, _mm_set_ps(-1.0f, 1.0f, -1.0f, 1.0f))); //sum, dif, sum, dif
				r6 = _mm_mul_ps(r6, cossin72);
				r7 = _mm_mul_ps(r7, cossin144);
				r7 = _mm_add_ps(r6, r7);
				r7 = _mm_add_ps(r7, _mm_andnot_ps(_mm_set_ps(~0, 0.0f, ~0, 0.0f), r5));
				r6 = _mm_shuffle_ps(r7, r7, _MM_SHUFFLE(0, 3, 0, 1));
				__m128 fp2r = _mm_add_ps(r7, r6);
				__m128 fn2r = _mm_sub_ps(r7, r6);

				r6 = _mm_add_ps(r1, _mm_mul_ps(r2, _mm_set_ps(1.0f, -1.0f, 1.0f, -1.0f))); //dif, sum, dif, sum
				r7 = _mm_add_ps(_mm_mul_ps(r3, _mm_set_ps(1.0f, -1.0f, 1.0f, -1.0f)), r4); //dif, sum, dif, sum
				r6 = _mm_mul_ps(r6, sincos72);
				r7 = _mm_mul_ps(r7, sincos144);
				r7 = _mm_add_ps(r6, r7);
				r7 = _mm_add_ps(r7, _mm_andnot_ps(_mm_set_ps(0.0f, ~0, 0.0f, ~0), r5));
				r6 = _mm_shuffle_ps(r7, r7, _MM_SHUFFLE(0, 3, 0, 1)); //dif, sum -> r6 == sum!
				__m128 fp2i = _mm_add_ps(r7, r6);
				__m128 fn2i = _mm_sub_ps(r6, r7);

				r6 = _mm_add_ps(r1, _mm_mul_ps(r2, _mm_set_ps(-1.0f, 1.0f, -1.0f, 1.0f))); //sum, dif, sum, dif
				r7 = _mm_add_ps(r3, _mm_mul_ps(r4, _mm_set_ps(-1.0f, 1.0f, -1.0f, 1.0f))); //sum, dif, sum, dif
				r6 = _mm_mul_ps(r6, cossin144);
				r7 = _mm_mul_ps(r7, cossin72);
				r7 = _mm_add_ps(r6, r7);
				r7 = _mm_add_ps(r7, _mm_andnot_ps(_mm_set_ps(~0, 0.0f, ~0, 0.0f), r5));
				r6 = _mm_shuffle_ps(r7, r7, _MM_SHUFFLE(0, 3, 0, 1));
				__m128 fpr = _mm_add_ps(r7, r6);
				__m128 fnr = _mm_sub_ps(r7, r6);

				r6 = _mm_add_ps(_mm_mul_ps(r1, _mm_set_ps(1.0f, -1.0f, 1.0f, -1.0f)), r2); //dif, sum, dif, sum
				r7 = _mm_add_ps(_mm_mul_ps(r3, _mm_set_ps(1.0f, -1.0f, 1.0f, -1.0f)), r4); //dif, sum, dif, sum
				r6 = _mm_mul_ps(r6, sincos144);
				r7 = _mm_mul_ps(r7, sincos72);
				r7 = _mm_add_ps(r6, r7);
				r7 = _mm_add_ps(r7, _mm_andnot_ps(_mm_set_ps(0.0f, ~0, 0.0f, ~0), r5));
				r6 = _mm_shuffle_ps(r7, r7, _MM_SHUFFLE(0, 3, 0, 1)); //dif, sum -> r6 == sum!
				__m128 fpi = _mm_add_ps(r7, r6);
				__m128 fni = _mm_sub_ps(r6, r7);

				r6 = _mm_add_ps(r1, r2);
				r7 = _mm_add_ps(r3, r4);
				r6 = _mm_add_ps(r5, r6);
				__m128 fc = _mm_add_ps(r6, r7);
				fc = _mm_sub_ps(fc, gridcorrection);

				__m128 psd = _mm_mul_ps(fc, fc);
				r1 = _mm_shuffle_ps(psd, psd, _MM_SHUFFLE(0, 3, 0, 1));//psd,-,psd,-
				psd = _mm_add_ps(psd, r1);
				psd = _mm_add_ps(psd, _mm_set1_ps(1e-15f));
				r1 = _mm_sub_ps(psd, _mm_load1_ps(&sigmaSquaredNoiseNormed));
				r1 = _mm_div_ps(r1, psd);
				r1 = _mm_max_ps(r1, _mm_load1_ps(&lowlimit));
				r1 = _mm_shuffle_ps(r1, r1, _MM_SHUFFLE(2, 2, 0, 0));
				fc = _mm_mul_ps(r1, fc);

				r1 = _mm_mul_ps(fp2r, fp2r);
				r2 = _mm_mul_ps(fp2i, fp2i);
				fp2r = _mm_shuffle_ps(fp2r, fp2i, _MM_SHUFFLE(2, 0, 2, 0));
				fp2r = _mm_shuffle_ps(fp2r, fp2r, _MM_SHUFFLE(3, 1, 2, 0));
				psd = _mm_add_ps(r1, r2);
				psd = _mm_add_ps(psd, _mm_set1_ps(1e-15f));
				r1 = _mm_sub_ps(psd, _mm_load1_ps(&sigmaSquaredNoiseNormed));
				r1 = _mm_div_ps(r1, psd);
				r1 = _mm_max_ps(r1, _mm_load1_ps(&lowlimit));
				r1 = _mm_shuffle_ps(r1, r1, _MM_SHUFFLE(2, 2, 0, 0));
				fp2r = _mm_mul_ps(r1, fp2r);

				r1 = _mm_mul_ps(fpr, fpr);
				r2 = _mm_mul_ps(fpi, fpi);
				fpr = _mm_shuffle_ps(fpr, fpi, _MM_SHUFFLE(2, 0, 2, 0));
				fpr = _mm_shuffle_ps(fpr, fpr, _MM_SHUFFLE(3, 1, 2, 0));
				psd = _mm_add_ps(r1, r2);
				psd = _mm_add_ps(psd, _mm_set1_ps(1e-15f));
				r1 = _mm_sub_ps(psd, _mm_load1_ps(&sigmaSquaredNoiseNormed));
				r1 = _mm_div_ps(r1, psd);
				r1 = _mm_max_ps(r1, _mm_load1_ps(&lowlimit));
				r1 = _mm_shuffle_ps(r1, r1, _MM_SHUFFLE(2, 2, 0, 0));
				fpr = _mm_mul_ps(r1, fpr);

				r1 = _mm_mul_ps(fnr, fnr);
				r2 = _mm_mul_ps(fni, fni);
				fnr = _mm_shuffle_ps(fnr, fni, _MM_SHUFFLE(2, 0, 2, 0));
				fnr = _mm_shuffle_ps(fnr, fnr, _MM_SHUFFLE(3, 1, 2, 0));
				psd = _mm_add_ps(r1, r2);
				psd = _mm_add_ps(psd, _mm_set1_ps(1e-15f));
				r1 = _mm_sub_ps(psd, _mm_load1_ps(&sigmaSquaredNoiseNormed));
				r1 = _mm_div_ps(r1, psd);
				r1 = _mm_max_ps(r1, _mm_load1_ps(&lowlimit));
				r1 = _mm_shuffle_ps(r1, r1, _MM_SHUFFLE(2, 2, 0, 0));
				fnr = _mm_mul_ps(r1, fnr);

				r1 = _mm_mul_ps(fn2r, fn2r);
				r2 = _mm_mul_ps(fn2i, fn2i);
				fn2r = _mm_shuffle_ps(fn2r, fn2i, _MM_SHUFFLE(2, 0, 2, 0));
				fn2r = _mm_shuffle_ps(fn2r, fn2r, _MM_SHUFFLE(3, 1, 2, 0));
				psd = _mm_add_ps(r1, r2);
				psd = _mm_add_ps(psd, _mm_set1_ps(1e-15f));
				r1 = _mm_sub_ps(psd, _mm_load1_ps(&sigmaSquaredNoiseNormed));
				r1 = _mm_div_ps(r1, psd);
				r1 = _mm_max_ps(r1, _mm_load1_ps(&lowlimit));
				r1 = _mm_shuffle_ps(r1, r1, _MM_SHUFFLE(2, 2, 0, 0));
				fn2r = _mm_mul_ps(r1, fn2r);

				r1 = _mm_add_ps(fp2r, fpr);
				r2 = _mm_add_ps(fc, fnr);
				r1 = _mm_add_ps(r1, fn2r);
				r1 = _mm_add_ps(r1, r2);
				r3 = _mm_load_ps(outprev[w + 2]);
				r4 = _mm_load_ps(outnext[w + 2]);
				r1 = _mm_add_ps(r1, gridcorrection);
				r1 = _mm_mul_ps(r1, _mm_set1_ps(0.2f));
				_mm_store_ps(outprev2[w], r1);
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