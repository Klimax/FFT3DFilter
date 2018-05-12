//
//	FFT3DFilter plugin for Avisynth 2.6 - 3D Frequency Domain filter
//  AVX512 version of filtering functions
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
#include "WienerPattern.h"

void PatternFilter::ApplyPattern2D_AVX512() noexcept
{
	int w(0);
	float psd(0.0f), patternfactor(0.0f);
	const int outwidth8 = outwidth - outwidth % 8;

	for (int block = start_block; block < blocks; block++)
	{
		float *pattern2d = pattern3d;
		for (int h = 0; h < bh; h++) // middle
		{
			for (w = 0; w < outwidth8; w = w + 8)
			{
				__m512 cur = _mm512_load_ps(outcur[w]);
				__m512 r1 = _mm512_mul_ps(cur, cur);
				__m512 r2 = _mm512_permute_ps(r1, _MM_SHUFFLE(0, 3, 0, 1));
				r1 = _mm512_add_ps(r1, r2);
				r1 = _mm512_add_ps(r1, _mm512_set1_ps(1e-15f));

				__m512 pf = _mm512_castps256_ps512(_mm256_load_ps(&pattern2d[w]));
				pf = _mm512_permutexvar_ps(_mm512_set_epi32(7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0), pf);
				pf = _mm512_mul_ps(pf, _mm512_set1_ps(pfactor));
				r2 = _mm512_sub_ps(r1, pf);
				r1 = _mm512_div_ps(r2, r1);
				r1 = _mm512_max_ps(r1, _mm512_set1_ps(lowlimit));
				r1 = _mm512_moveldup_ps(r1);
				_mm512_store_ps(outcur[w], _mm512_mul_ps(cur, r1));
			}
			_mm_prefetch((const char*)(outcur + outpitch), _MM_HINT_T0);
			_mm_prefetch((const char*)(pattern2d + outpitch), _MM_HINT_T0);
			for (; w < outwidth; w++)
			{
				psd = (outcur[w][0] * outcur[w][0] + outcur[w][1] * outcur[w][1]) + 1e-15f;
				patternfactor = max((psd - pfactor * pattern2d[w]) / psd, lowlimit);
				outcur[w][0] *= patternfactor;
				outcur[w][1] *= patternfactor;
			}
			outcur += outpitch;
			pattern2d += outpitch;
		}
	}
}

void PatternFilter::ApplyPattern2D_degrid_AVX512() noexcept
{
	int w(0);
	float psd(0.0f), WienerFactor(0.0f);
	const int outwidth8 = outwidth - outwidth % 8;

	for (int block = start_block; block < blocks; block++)
	{
		const float gridfraction = degrid * outcur[0][0] / gridsample[0][0];
		float *pattern2d = pattern3d;
		const __m512 gridfraction8 = _mm512_set1_ps(gridfraction);
		for (int h = 0; h < bh; h++) // middle
		{
			for (w = 0; w < outwidth8; w = w + 8)
			{
				__m512 gridcorrection8 = _mm512_mul_ps(gridfraction8, _mm512_load_ps(gridsample[w])); //gridcorrection8
				__m512 cur = _mm512_load_ps(outcur[w]);
				cur = _mm512_sub_ps(cur, gridcorrection8); //corrected
				__m512 r1 = _mm512_mul_ps(cur, cur);
				__m512 r2 = _mm512_permute_ps(r1, _MM_SHUFFLE(0, 3, 0, 1));
				r1 = _mm512_add_ps(r1, r2);
				r1 = _mm512_add_ps(r1, _mm512_set1_ps(1e-15f));

				__m512 pf = _mm512_castps256_ps512(_mm256_load_ps(&pattern2d[w]));
				pf = _mm512_permutexvar_ps(_mm512_set_epi32(7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0), pf);
				pf = _mm512_mul_ps(pf, _mm512_set1_ps(pfactor));
				r2 = _mm512_sub_ps(r1, pf);
				r1 = _mm512_div_ps(r2, r1);
				r1 = _mm512_max_ps(r1, _mm512_set1_ps(lowlimit));
				r1 = _mm512_moveldup_ps(r1);
				r1 = _mm512_mul_ps(cur, r1);
				r1 = _mm512_add_ps(r1, gridcorrection8);// final
				_mm512_store_ps(outcur[w], r1);
			}
			_mm_prefetch((const char*)(outcur + outpitch), _MM_HINT_T0);
			_mm_prefetch((const char*)(pattern2d + outpitch), _MM_HINT_T0);
			_mm_prefetch((const char*)(gridsample + outpitch), _MM_HINT_T0);
			for (; w < outwidth; w++)
			{
				float gridcorrection0 = gridfraction * gridsample[w][0];
				float corrected0 = outcur[w][0] - gridcorrection0;
				float gridcorrection1 = gridfraction * gridsample[w][1];
				float corrected1 = outcur[w][1] - gridcorrection1;
				psd = (corrected0*corrected0 + corrected1 * corrected1) + 1e-15f;// power spectrum density
																			   //					psd = (outcur[w][0]*outcur[w][0] + outcur[w][1]*outcur[w][1]) + 1e-15f;
				WienerFactor = max((psd - pfactor * pattern2d[w]) / psd, lowlimit); // limited Wiener filter
				corrected0 *= WienerFactor; // apply filter on real  part	
				corrected1 *= WienerFactor; // apply filter on imaginary part
				outcur[w][0] = corrected0 + gridcorrection0;
				outcur[w][1] = corrected1 + gridcorrection1;
			}
			outcur += outpitch;
			pattern2d += outpitch;
			gridsample += outpitch;
		}
		gridsample -= outpitch * bh; // restore pointer to only valid first block
	}
}

//-----------------------------------------------------------------------------------------
//
void PatternFilter::ApplyPattern3D2_AVX512() noexcept
{
	// dft 3d (very short - 2 points)
	float psd(0.0f), WienerFactor(0.0f);
	float f3d0r(0.0f), f3d1r(0.0f), f3d0i(0.0f), f3d1i(0.0f);
	int w(0);

	const int outwidth8 = outwidth - outwidth % 8;

	for (int block = start_block; block < blocks; block++)
	{
		for (int h = 0; h < bh; h++)
		{
			__m512 r2 = _mm512_load_ps(outprev[0]);
			for (w = 0; w < outwidth8; w = w + 8)
			{
				__m512 r1 = _mm512_load_ps(outcur[w]);
				__m512 sum8 = _mm512_add_ps(r1, r2);
				__m512 dif8 = _mm512_sub_ps(r1, r2);
				r1 = _mm512_mul_ps(sum8, sum8);
				r2 = _mm512_mul_ps(dif8, dif8);

				__m512 r3 = _mm512_permute_ps(r1, _MM_SHUFFLE(0, 3, 0, 1));
				__m512 r4 = _mm512_permute_ps(r2, _MM_SHUFFLE(0, 3, 0, 1));

				__m512 psds8 = _mm512_add_ps(r1, r3);
				__m512 psdd8 = _mm512_add_ps(r2, r4);

				psds8 = _mm512_add_ps(psds8, _mm512_set1_ps(1e-15f));
				psdd8 = _mm512_add_ps(psdd8, _mm512_set1_ps(1e-15f));

				__m512 pf = _mm512_castps256_ps512(_mm256_load_ps(&pattern3d[w]));
				pf = _mm512_permutexvar_ps(_mm512_set_epi32(7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0), pf);
				r1 = _mm512_sub_ps(psds8, pf);
				r2 = _mm512_sub_ps(psdd8, pf);
				r1 = _mm512_div_ps(r1, psds8);
				r2 = _mm512_div_ps(r2, psdd8);

				__m512 WienerFactors8 = _mm512_max_ps(r1, _mm512_set1_ps(lowlimit));
				__m512 WienerFactord8 = _mm512_max_ps(r2, _mm512_set1_ps(lowlimit));

				r1 = _mm512_moveldup_ps(WienerFactors8);
				r2 = _mm512_moveldup_ps(WienerFactord8);

				sum8 = _mm512_mul_ps(sum8, r1);
				dif8 = _mm512_mul_ps(dif8, r2);

				r1 = _mm512_add_ps(sum8, dif8);
				r2 = _mm512_load_ps(outprev[w + 8]);
				r1 = _mm512_mul_ps(r1, _mm512_set1_ps(0.5f));
				_mm512_store_ps(outprev[w], r1);
			}
			_mm_prefetch((const char*)(outprev + outpitch), _MM_HINT_T0);
			_mm_prefetch((const char*)(outcur + outpitch), _MM_HINT_T0);
			_mm_prefetch((const char*)(pattern3d + outpitch), _MM_HINT_T0);
			for (; w < outwidth; w++)
			{
				f3d0r = outcur[w][0] + outprev[w][0]; // real 0 (sum)
				f3d0i = outcur[w][1] + outprev[w][1]; // im 0 (sum)
				f3d1r = outcur[w][0] - outprev[w][0]; // real 1 (dif)
				f3d1i = outcur[w][1] - outprev[w][1]; // im 1 (dif)
				psd = f3d0r * f3d0r + f3d0i * f3d0i + 1e-15f; // power spectrum density 0
				WienerFactor = max((psd - pattern3d[w]) / psd, lowlimit); // limited Wiener filter
				f3d0r *= WienerFactor; // apply filter on real  part	
				f3d0i *= WienerFactor; // apply filter on imaginary part
				psd = f3d1r * f3d1r + f3d1i * f3d1i + 1e-15f; // power spectrum density 1
				WienerFactor = max((psd - pattern3d[w]) / psd, lowlimit); // limited Wiener filter
				f3d1r *= WienerFactor; // apply filter on real  part	
				f3d1i *= WienerFactor; // apply filter on imaginary part
									   // reverse dft for 2 points
				outprev[w][0] = (f3d0r + f3d1r)*0.5f; // get  real  part	
				outprev[w][1] = (f3d0i + f3d1i)*0.5f; // get imaginary part
													  // Attention! return filtered "out" in "outprev" to preserve "out" for next step
			}
			outcur += outpitch;
			outprev += outpitch;
			pattern3d += outpitch;
		}
		pattern3d -= outpitch * bh; // restore pointer for new block
	}
}

void PatternFilter::ApplyPattern3D2_degrid_AVX512() noexcept
{
	// dft 3d (very short - 2 points)
	float psd(0.0f), WienerFactor(0.0f), f3d0r(0.0f), f3d1r(0.0f), f3d0i(0.0f), f3d1i(0.0f);
	int w(0);

	const int outwidth8 = outwidth - outwidth % 8;

	for (int block = start_block; block < blocks; block++)
	{
		const float gridfraction = degrid * outcur[0][0] / gridsample[0][0];
		const __m512 gridfraction8 = _mm512_set1_ps(gridfraction);
		for (int h = 0; h < bh; h++)
		{
			__m512 r2 = _mm512_load_ps(outprev[0]);
			for (w = 0; w < outwidth8; w = w + 8)
			{
				__m512 gridcorrection8 = _mm512_mul_ps(gridfraction8, _mm512_load_ps(gridsample[w])); //gridcorrection8
				gridcorrection8 = _mm512_mul_ps(gridcorrection8, _mm512_set1_ps(2.0f));
				__m512 r1 = _mm512_load_ps(outcur[w]);
				__m512 sum8 = _mm512_add_ps(r1, r2);
				sum8 = _mm512_sub_ps(sum8, gridcorrection8);
				__m512 dif8 = _mm512_sub_ps(r1, r2);
				r1 = _mm512_mul_ps(sum8, sum8);
				r2 = _mm512_mul_ps(dif8, dif8);

				__m512 r3 = _mm512_permute_ps(r1, _MM_SHUFFLE(0, 3, 0, 1));
				__m512 r4 = _mm512_permute_ps(r2, _MM_SHUFFLE(0, 3, 0, 1));

				__m512 psds8 = _mm512_add_ps(r1, r3);
				__m512 psdd8 = _mm512_add_ps(r2, r4);

				psds8 = _mm512_add_ps(psds8, _mm512_set1_ps(1e-15f));
				psdd8 = _mm512_add_ps(psdd8, _mm512_set1_ps(1e-15f));

				__m512 pf = _mm512_castps256_ps512(_mm256_load_ps(&pattern3d[w]));
				pf = _mm512_permutexvar_ps(_mm512_set_epi32(7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0), pf);
				r1 = _mm512_sub_ps(psds8, pf);
				r2 = _mm512_sub_ps(psdd8, pf);
				r1 = _mm512_div_ps(r1, psds8);
				r2 = _mm512_div_ps(r2, psdd8);

				__m512 WienerFactors8 = _mm512_max_ps(r1, _mm512_set1_ps(lowlimit));
				__m512 WienerFactord8 = _mm512_max_ps(r2, _mm512_set1_ps(lowlimit));

				r1 = _mm512_moveldup_ps(WienerFactors8);
				r2 = _mm512_moveldup_ps(WienerFactord8);

				sum8 = _mm512_mul_ps(sum8, r1);
				dif8 = _mm512_mul_ps(dif8, r2);

				r1 = _mm512_add_ps(sum8, dif8);
				r1 = _mm512_add_ps(gridcorrection8, r1);
				r2 = _mm512_load_ps(outprev[w + 8]);
				r1 = _mm512_mul_ps(r1, _mm512_set1_ps(0.5f));
				_mm512_store_ps(outprev[w], r1);
			}
			_mm_prefetch((const char*)(outprev + outpitch), _MM_HINT_T0);
			_mm_prefetch((const char*)(outcur + outpitch), _MM_HINT_T0);
			_mm_prefetch((const char*)(gridsample + outpitch), _MM_HINT_T0);
			_mm_prefetch((const char*)(pattern3d + outpitch), _MM_HINT_T0);
			for (; w < outwidth; w++)
			{
				float gridcorrection0_2 = gridfraction * gridsample[w][0] * 2; // grid correction
				float gridcorrection1_2 = gridfraction * gridsample[w][1] * 2;
				f3d0r = outcur[w][0] + outprev[w][0] - gridcorrection0_2; // real 0 (sum)
				f3d0i = outcur[w][1] + outprev[w][1] - gridcorrection1_2; // im 0 (sum)
				f3d1r = outcur[w][0] - outprev[w][0]; // real 1 (dif)
				f3d1i = outcur[w][1] - outprev[w][1]; // im 1 (dif)
				psd = f3d0r * f3d0r + f3d0i * f3d0i + 1e-15f; // power spectrum density 0
				WienerFactor = max((psd - pattern3d[w]) / psd, lowlimit); // limited Wiener filter
				f3d0r *= WienerFactor; // apply filter on real  part	
				f3d0i *= WienerFactor; // apply filter on imaginary part
				psd = f3d1r * f3d1r + f3d1i * f3d1i + 1e-15f; // power spectrum density 1
				WienerFactor = max((psd - pattern3d[w]) / psd, lowlimit); // limited Wiener filter
				f3d1r *= WienerFactor; // apply filter on real  part	
				f3d1i *= WienerFactor; // apply filter on imaginary part
									   // reverse dft for 2 points
				outprev[w][0] = (f3d0r + f3d1r + gridcorrection0_2)*0.5f; // get  real  part	
				outprev[w][1] = (f3d0i + f3d1i + gridcorrection1_2)*0.5f; // get imaginary part
																		  // Attention! return filtered "out" in "outprev" to preserve "out" for next step
			}
			outcur += outpitch;
			outprev += outpitch;
			pattern3d += outpitch;
			gridsample += outpitch;
		}
		pattern3d -= outpitch * bh; // restore pointer for new block
		gridsample -= outpitch * bh; // restore pointer to only valid first block
	}
}

//-----------------------------------------------------------------------------------------
//
void PatternFilter::ApplyPattern3D3_AVX512() noexcept
{
	// dft 3d (very short - 3 points)
	float fcr(0.0f), fci(0.0f), fpr(0.0f), fpi(0.0f), fnr(0.0f), fni(0.0f), WienerFactor(0.0f), psd(0.0f);
	const float sin120 = 0.86602540378443864676372317075294f;//sqrtf(3.0f)*0.5f;
	int w(0);
	const int outwidth8 = outwidth - outwidth % 8;
	constexpr __mmask16 k1 = 0X55;

	for (int block = start_block; block < blocks; block++)
	{
		for (int h = 0; h < bh; h++) // first half
		{
			__m512 r2 = _mm512_load_ps(outnext[0]);
			__m512 r3 = _mm512_load_ps(outcur[0]);
			for (w = 0; w < outwidth8; w = w + 8) // 
			{
				__m512 r1 = _mm512_load_ps(outprev[w]);
				__m512 pn8 = _mm512_add_ps(r1, r2); //r,i,r,i
				__m512 fc8 = _mm512_add_ps(pn8, r3); //r,i,r,i

				__m512 d8 = _mm512_sub_ps(r1, r2); //r,i,r,i!
				d8 = _mm512_mask_sub_ps(d8, k1, r2, r1); //r,i,r,i!
				d8 = _mm512_mul_ps(d8, _mm512_set1_ps(sin120));
				d8 = _mm512_permute_ps(d8, _MM_SHUFFLE(2, 3, 0, 1));
				r1 = _mm512_mul_ps(pn8, _mm512_set1_ps(0.5f));
				r1 = _mm512_sub_ps(r3, r1);
				__m512 fp8 = _mm512_add_ps(r1, d8);
				__m512 fn8 = _mm512_sub_ps(r1, d8);

				__m512 psdc8 = _mm512_mul_ps(fc8, fc8);
				__m512 psdp8 = _mm512_mul_ps(fp8, fp8);
				__m512 psdn8 = _mm512_mul_ps(fn8, fn8);

				r1 = _mm512_permute_ps(psdc8, _MM_SHUFFLE(0, 3, 0, 1));//psd,-,psd,-
				r2 = _mm512_permute_ps(psdp8, _MM_SHUFFLE(0, 3, 0, 1));
				r3 = _mm512_permute_ps(psdn8, _MM_SHUFFLE(0, 3, 0, 1));

				psdc8 = _mm512_add_ps(psdc8, r1);
				psdp8 = _mm512_add_ps(psdp8, r2);
				psdn8 = _mm512_add_ps(psdn8, r3);

				psdc8 = _mm512_add_ps(psdc8, _mm512_set1_ps(1e-15f));
				psdp8 = _mm512_add_ps(psdp8, _mm512_set1_ps(1e-15f));
				psdn8 = _mm512_add_ps(psdn8, _mm512_set1_ps(1e-15f));

				__m512 pf = _mm512_castps256_ps512(_mm256_load_ps(&pattern3d[w]));
				pf = _mm512_permutexvar_ps(_mm512_set_epi32(7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0), pf);
				r1 = _mm512_sub_ps(psdc8, pf);
				r2 = _mm512_sub_ps(psdp8, pf);
				r3 = _mm512_sub_ps(psdn8, pf);

				r1 = _mm512_div_ps(r1, psdc8);
				r2 = _mm512_div_ps(r2, psdp8);
				r3 = _mm512_div_ps(r3, psdn8);

				r1 = _mm512_max_ps(r1, _mm512_set1_ps(lowlimit));
				r2 = _mm512_max_ps(r2, _mm512_set1_ps(lowlimit));
				r3 = _mm512_max_ps(r3, _mm512_set1_ps(lowlimit));

				r1 = _mm512_moveldup_ps(r1);
				r2 = _mm512_moveldup_ps(r2);
				r1 = _mm512_moveldup_ps(r3);

				fc8 = _mm512_mul_ps(r1, fc8);
				fp8 = _mm512_mul_ps(r2, fp8);
				fn8 = _mm512_mul_ps(r3, fn8);

				r1 = _mm512_add_ps(fc8, fp8);
				r1 = _mm512_add_ps(r1, fn8);
				r2 = _mm512_load_ps(outnext[w + 8]);
				r3 = _mm512_load_ps(outcur[w + 8]);
				r1 = _mm512_mul_ps(r1, _mm512_set1_ps(0.33333333333f));

				_mm512_store_ps(outprev[w], r1);
			}
			_mm_prefetch((const char*)(outprev + outpitch), _MM_HINT_T0);
			_mm_prefetch((const char*)(outcur + outpitch), _MM_HINT_T0);
			_mm_prefetch((const char*)(outnext + outpitch), _MM_HINT_T0);
			_mm_prefetch((const char*)(pattern3d + outpitch), _MM_HINT_T0);
			for (; w < outwidth; w++) // 
			{
				float pnr = outprev[w][0] + outnext[w][0];
				float pni = outprev[w][1] + outnext[w][1];
				fcr = outcur[w][0] + pnr; // real cur
				fci = outcur[w][1] + pni; // im cur
				float di = sin120 * (outprev[w][1] - outnext[w][1]);
				float dr = sin120 * (outnext[w][0] - outprev[w][0]);
				fpr = outcur[w][0] - 0.5f*pnr + di; // real prev
				fnr = outcur[w][0] - 0.5f*pnr - di; // real next
				fpi = outcur[w][1] - 0.5f*pni + dr; // im prev
				fni = outcur[w][1] - 0.5f*pni - dr; // im next
				psd = fcr * fcr + fci * fci + 1e-15f; // power spectrum density cur
				WienerFactor = max((psd - pattern3d[w]) / psd, lowlimit); // limited Wiener filter
				fcr *= WienerFactor; // apply filter on real  part	
				fci *= WienerFactor; // apply filter on imaginary part
				psd = fpr * fpr + fpi * fpi + 1e-15f; // power spectrum density prev
				WienerFactor = max((psd - pattern3d[w]) / psd, lowlimit); // limited Wiener filter
				fpr *= WienerFactor; // apply filter on real  part	
				fpi *= WienerFactor; // apply filter on imaginary part
				psd = fnr * fnr + fni * fni + 1e-15f; // power spectrum density next
				WienerFactor = max((psd - pattern3d[w]) / psd, lowlimit); // limited Wiener filter
				fnr *= WienerFactor; // apply filter on real  part	
				fni *= WienerFactor; // apply filter on imaginary part
									 // reverse dft for 3 points
				outprev[w][0] = (fcr + fpr + fnr)*0.33333333333f; // get  real  part	
				outprev[w][1] = (fci + fpi + fni)*0.33333333333f; // get imaginary part
																  // Attention! return filtered "out" in "outprev" to preserve "out" for next step
			}
			outcur += outpitch;
			outprev += outpitch;
			outnext += outpitch;
			pattern3d += outpitch;
		}
		pattern3d -= outpitch * bh; // restore pointer for new block
	}
}

void PatternFilter::ApplyPattern3D3_degrid_AVX512() noexcept
{
	// dft 3d (very short - 3 points)
	float fcr(0.0f), fci(0.0f), fpr(0.0f), fpi(0.0f), fnr(0.0f), fni(0.0f), WienerFactor(0.0f), psd(0.0f);
	const float sin120 = 0.86602540378443864676372317075294f;//sqrtf(3.0f)*0.5f;
	int w(0);
	const int outwidth8 = outwidth - outwidth % 8;
	constexpr __mmask16 k1 = 0X55;


	for (int block = start_block; block < blocks; block++)
	{
		const float gridfraction = degrid * outcur[0][0] / gridsample[0][0];
		const __m512 gridfraction8 = _mm512_set1_ps(gridfraction);
		for (int h = 0; h < bh; h++) // first half
		{
			__m512 r2 = _mm512_load_ps(outnext[0]);
			__m512 r3 = _mm512_load_ps(outcur[0]);
			for (w = 0; w < outwidth8; w = w + 8) // 
			{
				__m512 gridcorrection8 = _mm512_mul_ps(gridfraction8, _mm512_load_ps(gridsample[w])); //gridcorrection8
				gridcorrection8 = _mm512_mul_ps(gridcorrection8, _mm512_set1_ps(3.0f));
				__m512 r1 = _mm512_load_ps(outprev[w]);
				__m512 pn8 = _mm512_add_ps(r1, r2); //r,i,r,i
				__m512 fc8 = _mm512_add_ps(pn8, r3); //r,i,r,i
				fc8 = _mm512_sub_ps(fc8, gridcorrection8);

				__m512 d8 = _mm512_sub_ps(r1, r2); //r,i,r,i!
				d8 = _mm512_mask_sub_ps(d8, k1, r2, r1); //r,i,r,i!
				d8 = _mm512_mul_ps(d8, _mm512_set1_ps(sin120));
				d8 = _mm512_permute_ps(d8, _MM_SHUFFLE(2, 3, 0, 1));
				r1 = _mm512_mul_ps(pn8, _mm512_set1_ps(0.5f));
				r1 = _mm512_sub_ps(r3, r1);
				__m512 fp8 = _mm512_add_ps(r1, d8);
				__m512 fn8 = _mm512_sub_ps(r1, d8);

				__m512 psdc8 = _mm512_mul_ps(fc8, fc8);
				__m512 psdp8 = _mm512_mul_ps(fp8, fp8);
				__m512 psdn8 = _mm512_mul_ps(fn8, fn8);

				r1 = _mm512_permute_ps(psdc8, _MM_SHUFFLE(0, 3, 0, 1));//psd,-,psd,-
				r2 = _mm512_permute_ps(psdp8, _MM_SHUFFLE(0, 3, 0, 1));
				r3 = _mm512_permute_ps(psdn8, _MM_SHUFFLE(0, 3, 0, 1));

				psdc8 = _mm512_add_ps(psdc8, r1);
				psdp8 = _mm512_add_ps(psdp8, r2);
				psdn8 = _mm512_add_ps(psdn8, r3);

				psdc8 = _mm512_add_ps(psdc8, _mm512_set1_ps(1e-15f));
				psdp8 = _mm512_add_ps(psdp8, _mm512_set1_ps(1e-15f));
				psdn8 = _mm512_add_ps(psdn8, _mm512_set1_ps(1e-15f));

				__m512 pf = _mm512_castps256_ps512(_mm256_load_ps(&pattern3d[w]));
				pf = _mm512_permutexvar_ps(_mm512_set_epi32(7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0), pf);
				r1 = _mm512_sub_ps(psdc8, pf);
				r2 = _mm512_sub_ps(psdp8, pf);
				r3 = _mm512_sub_ps(psdn8, pf);

				r1 = _mm512_div_ps(r1, psdc8);
				r2 = _mm512_div_ps(r2, psdp8);
				r3 = _mm512_div_ps(r3, psdn8);

				r1 = _mm512_max_ps(r1, _mm512_set1_ps(lowlimit));
				r2 = _mm512_max_ps(r2, _mm512_set1_ps(lowlimit));
				r3 = _mm512_max_ps(r3, _mm512_set1_ps(lowlimit));

				r1 = _mm512_moveldup_ps(r1);
				r2 = _mm512_moveldup_ps(r2);
				r1 = _mm512_moveldup_ps(r3);

				fc8 = _mm512_mul_ps(r1, fc8);
				fp8 = _mm512_mul_ps(r2, fp8);
				fn8 = _mm512_mul_ps(r3, fn8);

				r1 = _mm512_add_ps(fc8, fp8);
				r1 = _mm512_add_ps(r1, fn8);
				r1 = _mm512_add_ps(r1, gridcorrection8);
				r2 = _mm512_load_ps(outnext[w + 8]);
				r3 = _mm512_load_ps(outcur[w + 8]);
				r1 = _mm512_mul_ps(r1, _mm512_set1_ps(0.33333333333f));

				_mm512_store_ps(outprev[w], r1);
			}
			_mm_prefetch((const char*)(outprev + outpitch), _MM_HINT_T0);
			_mm_prefetch((const char*)(outcur + outpitch), _MM_HINT_T0);
			_mm_prefetch((const char*)(outnext + outpitch), _MM_HINT_T0);
			_mm_prefetch((const char*)(gridsample + outpitch), _MM_HINT_T0);
			_mm_prefetch((const char*)(pattern3d + outpitch), _MM_HINT_T0);
			for (; w < outwidth; w++) // 
			{
				float gridcorrection0_3 = gridfraction * gridsample[w][0] * 3;
				float gridcorrection1_3 = gridfraction * gridsample[w][1] * 3;
				float pnr = outprev[w][0] + outnext[w][0];
				float pni = outprev[w][1] + outnext[w][1];
				fcr = outcur[w][0] + pnr; // real cur
				fcr -= gridcorrection0_3;
				fci = outcur[w][1] + pni; // im cur
				fci -= gridcorrection1_3;
				float di = sin120 * (outprev[w][1] - outnext[w][1]);
				float dr = sin120 * (outnext[w][0] - outprev[w][0]);
				fpr = outcur[w][0] - 0.5f*pnr + di; // real prev
				fnr = outcur[w][0] - 0.5f*pnr - di; // real next
				fpi = outcur[w][1] - 0.5f*pni + dr; // im prev
				fni = outcur[w][1] - 0.5f*pni - dr; // im next
				psd = fcr * fcr + fci * fci + 1e-15f; // power spectrum density cur
				WienerFactor = max((psd - pattern3d[w]) / psd, lowlimit); // limited Wiener filter
				fcr *= WienerFactor; // apply filter on real  part	
				fci *= WienerFactor; // apply filter on imaginary part
				psd = fpr * fpr + fpi * fpi + 1e-15f; // power spectrum density prev
				WienerFactor = max((psd - pattern3d[w]) / psd, lowlimit); // limited Wiener filter
				fpr *= WienerFactor; // apply filter on real  part	
				fpi *= WienerFactor; // apply filter on imaginary part
				psd = fnr * fnr + fni * fni + 1e-15f; // power spectrum density next
				WienerFactor = max((psd - pattern3d[w]) / psd, lowlimit); // limited Wiener filter
				fnr *= WienerFactor; // apply filter on real  part	
				fni *= WienerFactor; // apply filter on imaginary part
									 // reverse dft for 3 points
				outprev[w][0] = (fcr + fpr + fnr + gridcorrection0_3)*0.33333333333f; // get  real  part	
				outprev[w][1] = (fci + fpi + fni + gridcorrection1_3)*0.33333333333f; // get imaginary part
																					  // Attention! return filtered "out" in "outprev" to preserve "out" for next step
			}
			outcur += outpitch;
			outprev += outpitch;
			outnext += outpitch;
			pattern3d += outpitch;
			gridsample += outpitch;
		}
		pattern3d -= outpitch * bh; // restore pointer for new block
		gridsample -= outpitch * bh; // restore pointer to only valid first block
	}
}

void PatternFilter::ApplyPattern3D4_AVX512() noexcept
{
	// dft with 4 points
	// return result in outprev
	float fcr(0.0f), fci(0.0f), fpr(0.0f), fpi(0.0f), fnr(0.0f), fni(0.0f), fp2r(0.0f), fp2i(0.0f), WienerFactor(0.0f), psd(0.0f);
	int w(0);
	int outwidth8 = outwidth - outwidth % 8;
	const __m512 lowlimit8 = _mm512_set1_ps(lowlimit);
	constexpr __mmask16 k1 = 0x5555;
	constexpr __mmask16 k2 = 0xAAAA;

	for (int block = start_block; block < blocks; block++)
	{
		for (int h = 0; h < bh; h++) // first half
		{
			__m512 r3 = _mm512_load_ps(&outcur[0][0]);
			__m512 r4 = _mm512_load_ps(&outnext[0][0]);
			for (w = 0; w < outwidth8; w = w + 8)
			{
				__m512 r1 = _mm512_load_ps(&outprev2[w][0]);
				__m512 r2 = _mm512_load_ps(&outprev[w][0]);
				//outcur[w][0] - outnext[w][1] - outprev2[w][0] + outprev[w][1]
				//outcur[w][1] + outnext[w][0] - outprev2[w][1] - outprev[w][0]

				__m512 r5 = _mm512_permute_ps(r4, _MM_SHUFFLE(1, 0, 3, 2)); //r5: swapped outnext
				__m512 r6 = _mm512_permute_ps(r2, _MM_SHUFFLE(1, 0, 3, 2)); //r6: swapped outprev
				__m512 r8 = _mm512_add_ps(r3, r5);
				__m512 r7 = _mm512_mask_sub_ps(r8, k1, r3, r5);//outcur + outnext

				r7 = _mm512_sub_ps(r7, r1); //-outprev2
				r8 = _mm512_add_ps(r7, r6);
				r7 = _mm512_mask_sub_ps(r8, k1, r7, r6);

				//outcur[w][0] + outnext[w][1] - outprev2[w][0] - outprev[w][1]
				//outcur[w][1] - outnext[w][0] - outprev2[w][1] + outprev[w][0]
				r8 = _mm512_maskz_add_ps(k1, r3, r5);
				r8 = _mm512_mask_sub_ps(r8, k2, r3, r5);

				r8 = _mm512_sub_ps(r8, r1); //-outprev2
				r5 = _mm512_add_ps(r8, r6);
				r8 = _mm512_mask_sub_ps(r5, k2, r8, r6);

				r5 = _mm512_add_ps(r1, r2);
				r6 = _mm512_add_ps(r3, r4);
				r5 = _mm512_add_ps(r5, r6); // fcr1, fci1, fcr2, fci2

				r1 = _mm512_sub_ps(r1, r2);
				r2 = _mm512_sub_ps(r3, r4);
				r6 = _mm512_add_ps(r1, r2); // fp2r1, fp2i1, fp2r2, fp2i2
											//r7: fpr1, fpi1, fpr2, fpi2
											//r8: fnr1, fni1, fnr2, fni2
											//r5: fcr1, fci1, fcr2, fci2
											//r6: fp2r1, fp2i1, fp2r2, fp2i2

				r1 = _mm512_mul_ps(r7, r7);
				r2 = _mm512_mul_ps(r8, r8);
				r3 = _mm512_mul_ps(r5, r5);
				r4 = _mm512_mul_ps(r6, r6);

				__m512 sh1 = _mm512_shuffle_ps(r1, r2, _MM_SHUFFLE(2, 0, 2, 0));
				__m512 sh2 = _mm512_shuffle_ps(r1, r2, _MM_SHUFFLE(3, 1, 3, 1));
				__m512 sh3 = _mm512_shuffle_ps(r3, r4, _MM_SHUFFLE(2, 0, 2, 0));
				__m512 sh4 = _mm512_shuffle_ps(r3, r4, _MM_SHUFFLE(3, 1, 3, 1));

				r1 = _mm512_add_ps(sh1, sh2);
				r2 = _mm512_add_ps(sh3, sh4);

				r1 = _mm512_add_ps(r1, _mm512_set1_ps(1e-15f));
				r2 = _mm512_add_ps(r2, _mm512_set1_ps(1e-15f));
				//r1: psd_fp1, psd_fp2, psd_fn1, psd_fn2
				//r2: psd_fc1, psd_fc2, psd_fp2_1, psd_fp2_2
				__m512 pf = _mm512_castps256_ps512(_mm256_load_ps(&pattern3d[w]));
				pf = _mm512_permutexvar_ps(_mm512_set_epi32(7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0), pf);
				r3 = _mm512_sub_ps(r1, pf);
				r4 = _mm512_sub_ps(r2, pf);
				r1 = _mm512_div_ps(r3, r1);
				r2 = _mm512_div_ps(r4, r2);
				r1 = _mm512_max_ps(r1, lowlimit8);
				r2 = _mm512_max_ps(r2, lowlimit8);
				//r1: wf_fp1, wf_fp2, wf_fn1, wf_fn2
				//r2: wf_fc1, wf_fc2, wf_fp2_1, wf_fp2_2
				//r5: fcr1, fci1, fcr2, fci2 ; fc1, fc2
				//r6: fp2r1, fp2i1, fp2r2, fp2i2 ; fp2_1, fp2_2
				//r7: fpr1, fpi1, fpr2, fpi2 ; fp1, fp2
				//r8: fnr1, fni1, fnr2, fni2 ; fn1, fn2

				r8 = _mm512_mul_ps(r8, _mm512_unpackhi_ps(r1, r1));
				r6 = _mm512_mul_ps(r6, _mm512_unpackhi_ps(r2, r2));

				r6 = _mm512_fmadd_ps(r5, _mm512_unpacklo_ps(r2, r2), r6);
				r8 = _mm512_fmadd_ps(r7, _mm512_unpacklo_ps(r1, r1), r8);
				r3 = _mm512_load_ps(&outcur[w + 8][0]);
				r4 = _mm512_load_ps(&outnext[w + 8][0]);

				r6 = _mm512_add_ps(r6, r8);
				r6 = _mm512_mul_ps(r6, _mm512_set1_ps(0.25f));

				_mm512_store_ps(&outprev2[w][0], r6);
				// Attention! return filtered "out" in "outprev2" to preserve "out" for next step
			}
			_mm_prefetch((const char*)(outprev2 + outpitch), _MM_HINT_T0);
			_mm_prefetch((const char*)(outprev + outpitch), _MM_HINT_T0);
			_mm_prefetch((const char*)(outcur + outpitch), _MM_HINT_T0);
			_mm_prefetch((const char*)(outnext + outpitch), _MM_HINT_T0);
			_mm_prefetch((const char*)(pattern3d + outpitch), _MM_HINT_T0);
			for (; w < outwidth; w++)
			{
				fp2r = outprev2[w][0] - outprev[w][0] + outcur[w][0] - outnext[w][0]; // real prev2
				fp2i = outprev2[w][1] - outprev[w][1] + outcur[w][1] - outnext[w][1]; // im cur
				fpr = -outprev2[w][0] + outprev[w][1] + outcur[w][0] - outnext[w][1]; // real prev
				fpi = -outprev2[w][1] - outprev[w][0] + outcur[w][1] + outnext[w][0]; // im cur
				fcr = outprev2[w][0] + outprev[w][0] + outcur[w][0] + outnext[w][0]; // real cur
				fci = outprev2[w][1] + outprev[w][1] + outcur[w][1] + outnext[w][1]; // im cur
				fnr = -outprev2[w][0] - outprev[w][1] + outcur[w][0] + outnext[w][1]; // real next
				fni = -outprev2[w][1] + outprev[w][0] + outcur[w][1] - outnext[w][0]; // im next

				psd = fp2r * fp2r + fp2i * fp2i + 1e-15f; // power spectrum density prev2
				WienerFactor = max((psd - pattern3d[w]) / psd, lowlimit); // limited Wiener filter
				fp2r *= WienerFactor; // apply filter on real  part	
				fp2i *= WienerFactor; // apply filter on imaginary part

				psd = fpr * fpr + fpi * fpi + 1e-15f; // power spectrum density prev
				WienerFactor = max((psd - pattern3d[w]) / psd, lowlimit); // limited Wiener filter
				fpr *= WienerFactor; // apply filter on real  part	
				fpi *= WienerFactor; // apply filter on imaginary part

				psd = fcr * fcr + fci * fci + 1e-15f; // power spectrum density cur
				WienerFactor = max((psd - pattern3d[w]) / psd, lowlimit); // limited Wiener filter
				fcr *= WienerFactor; // apply filter on real  part	
				fci *= WienerFactor; // apply filter on imaginary part

				psd = fnr * fnr + fni * fni + 1e-15f; // power spectrum density next
				WienerFactor = max((psd - pattern3d[w]) / psd, lowlimit); // limited Wiener filter
				fnr *= WienerFactor; // apply filter on real  part	
				fni *= WienerFactor; // apply filter on imaginary part

									 // reverse dft for 4 points
				outprev2[w][0] = (fp2r + fpr + fcr + fnr)*0.25f; // get  real  part	
				outprev2[w][1] = (fp2i + fpi + fci + fni)*0.25f; // get imaginary part
																 // Attention! return filtered "out" in "outprev2" to preserve "out" for next step
			}
			outcur += outpitch;
			outprev2 += outpitch;
			outprev += outpitch;
			outnext += outpitch;
			pattern3d += outpitch;
		}
		pattern3d -= outpitch * bh; // restore pointer
	}
}

void PatternFilter::ApplyPattern3D4_degrid_AVX512() noexcept
{
	// dft with 4 points
	// return result in outprev
	float fcr(0.0f), fci(0.0f), fpr(0.0f), fpi(0.0f), fnr(0.0f), fni(0.0f), fp2r(0.0f), fp2i(0.0f);
	float WienerFactor(0.0f), psd(0.0f);
	int w(0);
	const int outwidth8 = outwidth - outwidth % 8;
	const __m512 lowlimit8 = _mm512_set1_ps(lowlimit);
	constexpr __mmask16 k1 = 0x5555;
	constexpr __mmask16 k2 = 0xAAAA;


	for (int block = start_block; block < blocks; block++)
	{
		const float gridfraction = degrid * outcur[0][0] / gridsample[0][0];
		const __m512 gridfraction8 = _mm512_set1_ps(gridfraction);
		for (int h = 0; h < bh; h++) // first half
		{
			__m512 r3 = _mm512_load_ps(&outcur[0][0]);
			__m512 r4 = _mm512_load_ps(&outnext[0][0]);
			for (w = 0; w < outwidth8; w = w + 8)
			{
				__m512 gridcorrection8 = _mm512_mul_ps(_mm512_mul_ps(_mm512_load_ps(&gridsample[w][0]), gridfraction8), _mm512_set1_ps(4));
				__m512 r1 = _mm512_load_ps(&outprev2[w][0]);
				__m512 r2 = _mm512_load_ps(&outprev[w][0]);
				//outcur[w][0] - outnext[w][1] - outprev2[w][0] + outprev[w][1]
				//outcur[w][1] + outnext[w][0] - outprev2[w][1] - outprev[w][0]

				__m512 r5 = _mm512_permute_ps(r4, _MM_SHUFFLE(1, 0, 3, 2)); //r5: swapped outnext
				__m512 r6 = _mm512_permute_ps(r2, _MM_SHUFFLE(1, 0, 3, 2)); //r6: swapped outprev
				__m512 r8 = _mm512_add_ps(r3, r5);
				__m512 r7 = _mm512_mask_sub_ps(r8, k1, r3, r5);//outcur + outnext

				r7 = _mm512_sub_ps(r7, r1); //-outprev2
				r8 = _mm512_add_ps(r7, r6);
				r7 = _mm512_mask_sub_ps(r8, k1, r7, r6);

				//outcur[w][0] + outnext[w][1] - outprev2[w][0] - outprev[w][1]
				//outcur[w][1] - outnext[w][0] - outprev2[w][1] + outprev[w][0]
				r8 = _mm512_maskz_add_ps(k1, r3, r5);
				r8 = _mm512_mask_sub_ps(r8, k2, r3, r5);

				r8 = _mm512_sub_ps(r8, r1); //-outprev2
				r5 = _mm512_add_ps(r8, r6);
				r8 = _mm512_mask_sub_ps(r5, k2, r8, r6);

				r5 = _mm512_add_ps(r1, r2);
				r6 = _mm512_add_ps(r3, r4);
				r5 = _mm512_add_ps(r5, r6); // fcr1, fci1, fcr2, fci2
				r5 = _mm512_sub_ps(r5, gridcorrection8);

				r1 = _mm512_sub_ps(r1, r2);
				r2 = _mm512_sub_ps(r3, r4);
				r6 = _mm512_add_ps(r1, r2); // fp2r1, fp2i1, fp2r2, fp2i2
											//r7: fpr1, fpi1, fpr2, fpi2
											//r8: fnr1, fni1, fnr2, fni2
											//r5: fcr1, fci1, fcr2, fci2
											//r6: fp2r1, fp2i1, fp2r2, fp2i2

				r1 = _mm512_mul_ps(r7, r7);
				r2 = _mm512_mul_ps(r8, r8);
				r3 = _mm512_mul_ps(r5, r5);
				r4 = _mm512_mul_ps(r6, r6);

				__m512 sh1 = _mm512_shuffle_ps(r1, r2, _MM_SHUFFLE(2, 0, 2, 0));
				__m512 sh2 = _mm512_shuffle_ps(r1, r2, _MM_SHUFFLE(3, 1, 3, 1));
				__m512 sh3 = _mm512_shuffle_ps(r3, r4, _MM_SHUFFLE(2, 0, 2, 0));
				__m512 sh4 = _mm512_shuffle_ps(r3, r4, _MM_SHUFFLE(3, 1, 3, 1));

				r1 = _mm512_add_ps(sh1, sh2);
				r2 = _mm512_add_ps(sh3, sh4);

				r1 = _mm512_add_ps(r1, _mm512_set1_ps(1e-15f));
				r2 = _mm512_add_ps(r2, _mm512_set1_ps(1e-15f));
				//r1: psd_fp1, psd_fp2, psd_fn1, psd_fn2
				//r2: psd_fc1, psd_fc2, psd_fp2_1, psd_fp2_2
				__m512 pf = _mm512_castps256_ps512(_mm256_load_ps(&pattern3d[w]));
				pf = _mm512_permutexvar_ps(_mm512_set_epi32(7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0), pf);
				r3 = _mm512_sub_ps(r1, pf);
				r4 = _mm512_sub_ps(r2, pf);
				r1 = _mm512_div_ps(r3, r1);
				r2 = _mm512_div_ps(r4, r2);
				r1 = _mm512_max_ps(r1, lowlimit8);
				r2 = _mm512_max_ps(r2, lowlimit8);
				//r1: wf_fp1, wf_fp2, wf_fn1, wf_fn2
				//r2: wf_fc1, wf_fc2, wf_fp2_1, wf_fp2_2
				//r5: fcr1, fci1, fcr2, fci2 ; fc1, fc2
				//r6: fp2r1, fp2i1, fp2r2, fp2i2 ; fp2_1, fp2_2
				//r7: fpr1, fpi1, fpr2, fpi2 ; fp1, fp2
				//r8: fnr1, fni1, fnr2, fni2 ; fn1, fn2

				r8 = _mm512_mul_ps(r8, _mm512_unpackhi_ps(r1, r1));
				r6 = _mm512_mul_ps(r6, _mm512_unpackhi_ps(r2, r2));

				r6 = _mm512_fmadd_ps(r5, _mm512_unpacklo_ps(r2, r2), r6);
				r8 = _mm512_fmadd_ps(r7, _mm512_unpacklo_ps(r1, r1), r8);
				r3 = _mm512_load_ps(&outcur[w + 8][0]);
				r4 = _mm512_load_ps(&outnext[w + 8][0]);

				r6 = _mm512_add_ps(r6, r8);
				r6 = _mm512_mul_ps(_mm512_add_ps(r6, gridcorrection8), _mm512_set1_ps(0.25f));

				_mm512_store_ps(&outprev2[w][0], r6);
				// Attention! return filtered "out" in "outprev2" to preserve "out" for next step
			}
			_mm_prefetch((const char*)(outprev2 + outpitch), _MM_HINT_T0);
			_mm_prefetch((const char*)(outprev + outpitch), _MM_HINT_T0);
			_mm_prefetch((const char*)(outcur + outpitch), _MM_HINT_T0);
			_mm_prefetch((const char*)(outnext + outpitch), _MM_HINT_T0);
			_mm_prefetch((const char*)(gridsample + outpitch), _MM_HINT_T0);
			_mm_prefetch((const char*)(pattern3d + outpitch), _MM_HINT_T0);
			for (; w < outwidth; w++)
			{
				float gridcorrection0_4 = gridfraction * gridsample[w][0] * 4;
				float gridcorrection1_4 = gridfraction * gridsample[w][1] * 4;
				fp2r = outprev2[w][0] - outprev[w][0] + outcur[w][0] - outnext[w][0]; // real prev2
				fp2i = outprev2[w][1] - outprev[w][1] + outcur[w][1] - outnext[w][1]; // im cur
				fpr = -outprev2[w][0] + outprev[w][1] + outcur[w][0] - outnext[w][1]; // real prev
				fpi = -outprev2[w][1] - outprev[w][0] + outcur[w][1] + outnext[w][0]; // im cur
				fcr = outprev2[w][0] + outprev[w][0] + outcur[w][0] + outnext[w][0]; // real cur
				fcr -= gridcorrection0_4;
				fci = outprev2[w][1] + outprev[w][1] + outcur[w][1] + outnext[w][1]; // im cur
				fci -= gridcorrection1_4;
				fnr = -outprev2[w][0] - outprev[w][1] + outcur[w][0] + outnext[w][1]; // real next
				fni = -outprev2[w][1] + outprev[w][0] + outcur[w][1] - outnext[w][0]; // im next

				psd = fp2r * fp2r + fp2i * fp2i + 1e-15f; // power spectrum density prev2
				WienerFactor = max((psd - pattern3d[w]) / psd, lowlimit); // limited Wiener filter
				fp2r *= WienerFactor; // apply filter on real  part	
				fp2i *= WienerFactor; // apply filter on imaginary part

				psd = fpr * fpr + fpi * fpi + 1e-15f; // power spectrum density prev
				WienerFactor = max((psd - pattern3d[w]) / psd, lowlimit); // limited Wiener filter
				fpr *= WienerFactor; // apply filter on real  part	
				fpi *= WienerFactor; // apply filter on imaginary part

				psd = fcr * fcr + fci * fci + 1e-15f; // power spectrum density cur
				WienerFactor = max((psd - pattern3d[w]) / psd, lowlimit); // limited Wiener filter
				fcr *= WienerFactor; // apply filter on real  part	
				fci *= WienerFactor; // apply filter on imaginary part

				psd = fnr * fnr + fni * fni + 1e-15f; // power spectrum density next
				WienerFactor = max((psd - pattern3d[w]) / psd, lowlimit); // limited Wiener filter
				fnr *= WienerFactor; // apply filter on real  part	
				fni *= WienerFactor; // apply filter on imaginary part

									 // reverse dft for 4 points
				outprev2[w][0] = (fp2r + fpr + fcr + fnr + gridcorrection0_4)*0.25f; // get  real  part	
				outprev2[w][1] = (fp2i + fpi + fci + fni + gridcorrection1_4)*0.25f; // get imaginary part
																					 // Attention! return filtered "out" in "outprev2" to preserve "out" for next step
			}
			outcur += outpitch;
			outprev2 += outpitch;
			outprev += outpitch;
			outnext += outpitch;
			pattern3d += outpitch;
			gridsample += outpitch;
		}
		pattern3d -= outpitch * bh; // restore pointer
		gridsample -= outpitch * bh; // restore pointer to only valid first block
	}
}

void PatternFilter::ApplyPattern3D5_AVX512() noexcept
{
	// dft with 5 points
	// return result in outprev2
	const float sin72 = 0.95105651629515357211643933337938f;// 2*pi/5
	const float cos72 = 0.30901699437494742410229341718282f;
	const float sin144 = 0.58778525229247312916870595463907f;
	const float cos144 = -0.80901699437494742410229341718282f;

	const __m512 sincos72 = _mm512_set_ps(0.30901699437494742410229341718282f, 0.95105651629515357211643933337938, 0.30901699437494742410229341718282f, 0.95105651629515357211643933337938,
		0.30901699437494742410229341718282f, 0.95105651629515357211643933337938, 0.30901699437494742410229341718282f, 0.95105651629515357211643933337938,
		0.30901699437494742410229341718282f, 0.95105651629515357211643933337938, 0.30901699437494742410229341718282f, 0.95105651629515357211643933337938,
		0.30901699437494742410229341718282f, 0.95105651629515357211643933337938, 0.30901699437494742410229341718282f, 0.95105651629515357211643933337938);
	const __m512 cossin72 = _mm512_set_ps(0.95105651629515357211643933337938, 0.30901699437494742410229341718282f, 0.95105651629515357211643933337938, 0.30901699437494742410229341718282f,
		0.95105651629515357211643933337938, 0.30901699437494742410229341718282f, 0.95105651629515357211643933337938, 0.30901699437494742410229341718282f,
		0.95105651629515357211643933337938, 0.30901699437494742410229341718282f, 0.95105651629515357211643933337938, 0.30901699437494742410229341718282f,
		0.95105651629515357211643933337938, 0.30901699437494742410229341718282f, 0.95105651629515357211643933337938, 0.30901699437494742410229341718282f);
	const __m512 sincos144 = _mm512_set_ps(-0.80901699437494742410229341718282f, 0.58778525229247312916870595463907f, -0.80901699437494742410229341718282f, 0.58778525229247312916870595463907f,
		-0.80901699437494742410229341718282f, 0.58778525229247312916870595463907f, -0.80901699437494742410229341718282f, 0.58778525229247312916870595463907f,
		-0.80901699437494742410229341718282f, 0.58778525229247312916870595463907f, -0.80901699437494742410229341718282f, 0.58778525229247312916870595463907f,
		-0.80901699437494742410229341718282f, 0.58778525229247312916870595463907f, -0.80901699437494742410229341718282f, 0.58778525229247312916870595463907f);
	const __m512 cossin144 = _mm512_set_ps(0.58778525229247312916870595463907f, -0.80901699437494742410229341718282f, 0.58778525229247312916870595463907f, -0.80901699437494742410229341718282f,
		0.58778525229247312916870595463907f, -0.80901699437494742410229341718282f, 0.58778525229247312916870595463907f, -0.80901699437494742410229341718282f,
		0.58778525229247312916870595463907f, -0.80901699437494742410229341718282f, 0.58778525229247312916870595463907f, -0.80901699437494742410229341718282f,
		0.58778525229247312916870595463907f, -0.80901699437494742410229341718282f, 0.58778525229247312916870595463907f, -0.80901699437494742410229341718282f);

	int w(0);
	const int outwidth8 = outwidth - outwidth % 8;
	constexpr __mmask16 k1 = 0x5555;
	constexpr __mmask16 k2 = 0xAAAA;

	for (int block = start_block; block < blocks; block++)
	{
		for (int h = 0; h < bh; h++) // first half
		{
			__m512 r3 = _mm512_load_ps(outprev[0]);
			__m512 r4 = _mm512_load_ps(outnext[0]);
			for (w = 0; w < outwidth8; w = w + 8) // 
			{
				__m512 r1 = _mm512_load_ps(outprev2[w]);
				__m512 r2 = _mm512_load_ps(outnext2[w]);
				__m512 r5 = _mm512_load_ps(outcur[w]);

				__m512 r6 = _mm512_add_ps(r1, r2);
				r6 = _mm512_mask_sub_ps(r6, k2, r1, r2);
				__m512 r7 = _mm512_add_ps(r4, r3);
				r7 = _mm512_mask_sub_ps(r7, k2, r4, r3);
				r6 = _mm512_mul_ps(r6, cossin72);
				r7 = _mm512_mul_ps(r7, cossin144);
				r7 = _mm512_add_ps(r6, r7);
				r7 = _mm512_mask_add_ps(r5, k1, r7, r5);
				r6 = _mm512_permute_ps(r7, _MM_SHUFFLE(0, 3, 0, 1)); //dif, sum -> r6 == sum!
				__m512 fp2r = _mm512_add_ps(r7, r6);
				__m512 fn2r = _mm512_sub_ps(r7, r6);

				r6 = _mm512_add_ps(r2, r1);
				r6 = _mm512_mask_sub_ps(r6, k1, r2, r1);
				r7 = _mm512_add_ps(r3, r4);
				r7 = _mm512_mask_sub_ps(r7, k1, r3, r4);
				r6 = _mm512_mul_ps(r6, sincos72);
				r7 = _mm512_mul_ps(r7, sincos144);
				r7 = _mm512_add_ps(r6, r7);
				r7 = _mm512_mask_add_ps(r5, k2, r7, r5);
				r6 = _mm512_permute_ps(r7, _MM_SHUFFLE(0, 3, 0, 1)); //dif, sum -> r6 == sum!
				__m512 fp2i = _mm512_add_ps(r7, r6);
				__m512 fn2i = _mm512_sub_ps(r6, r7);

				r6 = _mm512_add_ps(r2, r1);
				r6 = _mm512_mask_sub_ps(r6, k2, r2, r1);
				r7 = _mm512_add_ps(r4, r3);
				r7 = _mm512_mask_sub_ps(r7, k2, r4, r3);
				r6 = _mm512_mul_ps(r6, cossin144);
				r7 = _mm512_mul_ps(r7, cossin72);
				r7 = _mm512_add_ps(r6, r7);
				r7 = _mm512_mask_add_ps(r5, k1, r7, r5);
				r6 = _mm512_permute_ps(r7, _MM_SHUFFLE(0, 3, 0, 1));
				__m512 fpr = _mm512_add_ps(r7, r6);
				__m512 fnr = _mm512_sub_ps(r7, r6);

				r6 = _mm512_add_ps(r1, r2);
				r6 = _mm512_mask_sub_ps(r6, k1, r1, r2);
				r7 = _mm512_add_ps(r3, r4);
				r7 = _mm512_mask_sub_ps(r7, k1, r3, r4);
				r6 = _mm512_mul_ps(r6, sincos144);
				r7 = _mm512_mul_ps(r7, sincos72);
				r7 = _mm512_add_ps(r6, r7);
				r7 = _mm512_mask_add_ps(r5, k2, r7, r5);
				r6 = _mm512_permute_ps(r7, _MM_SHUFFLE(0, 3, 0, 1)); //dif, sum -> r6 == sum!
				__m512 fpi = _mm512_add_ps(r7, r6);
				__m512 fni = _mm512_sub_ps(r6, r7);

				r6 = _mm512_add_ps(r1, r2);
				r7 = _mm512_add_ps(r3, r4);
				r6 = _mm512_add_ps(r5, r6);
				__m512 fc = _mm512_add_ps(r6, r7);

				__m512 pf = _mm512_castps256_ps512(_mm256_load_ps(&pattern3d[w]));
				pf = _mm512_permutexvar_ps(_mm512_set_epi32(7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0), pf);

				__m512 psd = _mm512_mul_ps(fc, fc);
				r1 = _mm512_permute_ps(psd, _MM_SHUFFLE(0, 3, 0, 1)); //psd,-,psd,-
				psd = _mm512_add_ps(psd, r1);
				psd = _mm512_add_ps(psd, _mm512_set1_ps(1e-15f));
				r1 = _mm512_sub_ps(psd, pf);
				r1 = _mm512_div_ps(r1, psd);
				r1 = _mm512_max_ps(r1, _mm512_set1_ps(lowlimit));
				r1 = _mm512_moveldup_ps(r1);
				fc = _mm512_mul_ps(r1, fc);

				r1 = _mm512_mul_ps(fp2r, fp2r);
				r2 = _mm512_mul_ps(fp2i, fp2i);
				fp2r = _mm512_shuffle_ps(fp2r, fp2i, _MM_SHUFFLE(2, 0, 2, 0));
				fp2r = _mm512_permute_ps(fp2r, _MM_SHUFFLE(3, 1, 2, 0));
				psd = _mm512_add_ps(r1, r2);
				psd = _mm512_add_ps(psd, _mm512_set1_ps(1e-15f));
				r1 = _mm512_sub_ps(psd, pf);
				r1 = _mm512_div_ps(r1, psd);
				r1 = _mm512_max_ps(r1, _mm512_set1_ps(lowlimit));
				r1 = _mm512_moveldup_ps(r1);
				fp2r = _mm512_mul_ps(r1, fp2r);

				r1 = _mm512_mul_ps(fpr, fpr);
				r2 = _mm512_mul_ps(fpi, fpi);
				fpr = _mm512_shuffle_ps(fpr, fpi, _MM_SHUFFLE(2, 0, 2, 0));
				fpr = _mm512_permute_ps(fpr, _MM_SHUFFLE(3, 1, 2, 0));
				psd = _mm512_add_ps(r1, r2);
				psd = _mm512_add_ps(psd, _mm512_set1_ps(1e-15f));
				r1 = _mm512_sub_ps(psd, pf);
				r1 = _mm512_div_ps(r1, psd);
				r1 = _mm512_max_ps(r1, _mm512_set1_ps(lowlimit));
				r1 = _mm512_moveldup_ps(r1);
				fpr = _mm512_mul_ps(r1, fpr);

				r1 = _mm512_mul_ps(fnr, fnr);
				r2 = _mm512_mul_ps(fni, fni);
				fnr = _mm512_shuffle_ps(fnr, fni, _MM_SHUFFLE(2, 0, 2, 0));
				fnr = _mm512_permute_ps(fnr, _MM_SHUFFLE(3, 1, 2, 0));
				psd = _mm512_add_ps(r1, r2);
				psd = _mm512_add_ps(psd, _mm512_set1_ps(1e-15f));
				r1 = _mm512_sub_ps(psd, pf);
				r1 = _mm512_div_ps(r1, psd);
				r1 = _mm512_max_ps(r1, _mm512_set1_ps(lowlimit));
				r1 = _mm512_moveldup_ps(r1);
				fnr = _mm512_mul_ps(r1, fnr);

				r1 = _mm512_mul_ps(fn2r, fn2r);
				r2 = _mm512_mul_ps(fn2i, fn2i);
				fn2r = _mm512_shuffle_ps(fn2r, fn2i, _MM_SHUFFLE(2, 0, 2, 0));
				fn2r = _mm512_permute_ps(fn2r, _MM_SHUFFLE(3, 1, 2, 0));
				psd = _mm512_add_ps(r1, r2);
				psd = _mm512_add_ps(psd, _mm512_set1_ps(1e-15f));
				r1 = _mm512_sub_ps(psd, pf);
				r1 = _mm512_div_ps(r1, psd);
				r1 = _mm512_max_ps(r1, _mm512_set1_ps(lowlimit));
				r1 = _mm512_moveldup_ps(r1);
				fn2r = _mm512_mul_ps(r1, fn2r);

				r1 = _mm512_add_ps(fp2r, fpr);
				r2 = _mm512_add_ps(fc, fnr);
				r1 = _mm512_add_ps(r1, fn2r);
				r1 = _mm512_add_ps(r1, r2);
				r3 = _mm512_load_ps(outprev[w + 8]);
				r4 = _mm512_load_ps(outnext[w + 8]);
				r1 = _mm512_mul_ps(r1, _mm512_set1_ps(0.2f));
				_mm512_store_ps(outprev2[w], r1);
			}
			_mm_prefetch((const char*)(outprev2 + outpitch), _MM_HINT_T0);
			_mm_prefetch((const char*)(outprev + outpitch), _MM_HINT_T0);
			_mm_prefetch((const char*)(outcur + outpitch), _MM_HINT_T0);
			_mm_prefetch((const char*)(outnext + outpitch), _MM_HINT_T0);
			_mm_prefetch((const char*)(outnext2 + outpitch), _MM_HINT_T0);
			_mm_prefetch((const char*)(pattern3d + outpitch), _MM_HINT_T0);
			for (; w < outwidth; w++) // 
			{
				float fcr, fci, fpr, fpi, fnr, fni, fp2r, fp2i, fn2r, fn2i, WienerFactor, psd;
				float sum = (outprev2[w][0] + outnext2[w][0])*cos72 + (outprev[w][0] + outnext[w][0])*cos144 + +outcur[w][0];
				float dif = (-outprev2[w][1] + outnext2[w][1])*sin72 + (outprev[w][1] - outnext[w][1])*sin144;
				fp2r = sum + dif; // real prev2
				fn2r = sum - dif; // real next2
				sum = (outprev2[w][1] + outnext2[w][1])*cos72 + (outprev[w][1] + outnext[w][1])*cos144 + outcur[w][1];
				dif = (outprev2[w][0] - outnext2[w][0])*sin72 + (-outprev[w][0] + outnext[w][0])*sin144;
				fp2i = sum + dif; // im prev2
				fn2i = sum - dif; // im next2
				sum = (outprev2[w][0] + outnext2[w][0])*cos144 + (outprev[w][0] + outnext[w][0])*cos72 + outcur[w][0];
				dif = (outprev2[w][1] - outnext2[w][1])*sin144 + (outprev[w][1] - outnext[w][1])*sin72;
				fpr = sum + dif; // real prev
				fnr = sum - dif; // real next
				sum = (outprev2[w][1] + outnext2[w][1])*cos144 + (outprev[w][1] + outnext[w][1])*cos72 + outcur[w][1];
				dif = (-outprev2[w][0] + outnext2[w][0])*sin144 + (-outprev[w][0] + outnext[w][0])*sin72;
				fpi = sum + dif; // im prev
				fni = sum - dif; // im next
				fcr = outprev2[w][0] + outprev[w][0] + outcur[w][0] + outnext[w][0] + outnext2[w][0]; // real cur
				fci = outprev2[w][1] + outprev[w][1] + outcur[w][1] + outnext[w][1] + outnext2[w][1]; // im cur

				psd = fp2r * fp2r + fp2i * fp2i + 1e-15f; // power spectrum density prev2
				WienerFactor = max((psd - pattern3d[w]) / psd, lowlimit); // limited Wiener filter
				fp2r *= WienerFactor; // apply filter on real  part	
				fp2i *= WienerFactor; // apply filter on imaginary part

				psd = fpr * fpr + fpi * fpi + 1e-15f; // power spectrum density prev
				WienerFactor = max((psd - pattern3d[w]) / psd, lowlimit); // limited Wiener filter
				fpr *= WienerFactor; // apply filter on real  part	
				fpi *= WienerFactor; // apply filter on imaginary part

				psd = fcr * fcr + fci * fci + 1e-15f; // power spectrum density cur
				WienerFactor = max((psd - pattern3d[w]) / psd, lowlimit); // limited Wiener filter
				fcr *= WienerFactor; // apply filter on real  part	
				fci *= WienerFactor; // apply filter on imaginary part

				psd = fnr * fnr + fni * fni + 1e-15f; // power spectrum density next
				WienerFactor = max((psd - pattern3d[w]) / psd, lowlimit); // limited Wiener filter
				fnr *= WienerFactor; // apply filter on real  part	
				fni *= WienerFactor; // apply filter on imaginary part

				psd = fn2r * fn2r + fn2i * fn2i + 1e-15f; // power spectrum density next
				WienerFactor = max((psd - pattern3d[w]) / psd, lowlimit); // limited Wiener filter
				fn2r *= WienerFactor; // apply filter on real  part	
				fn2i *= WienerFactor; // apply filter on imaginary part

									  // reverse dft for 5 points
				outprev2[w][0] = (fp2r + fpr + fcr + fnr + fn2r)*0.2f; // get  real  part	
				outprev2[w][1] = (fp2i + fpi + fci + fni + fn2i)*0.2f; // get imaginary part
																	   // Attention! return filtered "out" in "outprev2" to preserve "out" for next step
			}
			outcur += outpitch;
			outprev2 += outpitch;
			outprev += outpitch;
			outnext += outpitch;
			outnext2 += outpitch;
			pattern3d += outpitch;
		}
		pattern3d -= outpitch * bh; // restore pointer
	}
}

void PatternFilter::ApplyPattern3D5_degrid_AVX512() noexcept
{
	// dft with 5 points
	// return result in outprev2
	const float sin72 = 0.95105651629515357211643933337938f;// 2*pi/5
	const float cos72 = 0.30901699437494742410229341718282f;
	const float sin144 = 0.58778525229247312916870595463907f;
	const float cos144 = -0.80901699437494742410229341718282f;


	const __m512 sincos72 = _mm512_set_ps(0.30901699437494742410229341718282f, 0.95105651629515357211643933337938, 0.30901699437494742410229341718282f, 0.95105651629515357211643933337938,
		0.30901699437494742410229341718282f, 0.95105651629515357211643933337938, 0.30901699437494742410229341718282f, 0.95105651629515357211643933337938,
		0.30901699437494742410229341718282f, 0.95105651629515357211643933337938, 0.30901699437494742410229341718282f, 0.95105651629515357211643933337938,
		0.30901699437494742410229341718282f, 0.95105651629515357211643933337938, 0.30901699437494742410229341718282f, 0.95105651629515357211643933337938);
	const __m512 cossin72 = _mm512_set_ps(0.95105651629515357211643933337938, 0.30901699437494742410229341718282f, 0.95105651629515357211643933337938, 0.30901699437494742410229341718282f,
		0.95105651629515357211643933337938, 0.30901699437494742410229341718282f, 0.95105651629515357211643933337938, 0.30901699437494742410229341718282f,
		0.95105651629515357211643933337938, 0.30901699437494742410229341718282f, 0.95105651629515357211643933337938, 0.30901699437494742410229341718282f,
		0.95105651629515357211643933337938, 0.30901699437494742410229341718282f, 0.95105651629515357211643933337938, 0.30901699437494742410229341718282f);
	const __m512 sincos144 = _mm512_set_ps(-0.80901699437494742410229341718282f, 0.58778525229247312916870595463907f, -0.80901699437494742410229341718282f, 0.58778525229247312916870595463907f,
		-0.80901699437494742410229341718282f, 0.58778525229247312916870595463907f, -0.80901699437494742410229341718282f, 0.58778525229247312916870595463907f,
		-0.80901699437494742410229341718282f, 0.58778525229247312916870595463907f, -0.80901699437494742410229341718282f, 0.58778525229247312916870595463907f,
		-0.80901699437494742410229341718282f, 0.58778525229247312916870595463907f, -0.80901699437494742410229341718282f, 0.58778525229247312916870595463907f);
	const __m512 cossin144 = _mm512_set_ps(0.58778525229247312916870595463907f, -0.80901699437494742410229341718282f, 0.58778525229247312916870595463907f, -0.80901699437494742410229341718282f,
		0.58778525229247312916870595463907f, -0.80901699437494742410229341718282f, 0.58778525229247312916870595463907f, -0.80901699437494742410229341718282f,
		0.58778525229247312916870595463907f, -0.80901699437494742410229341718282f, 0.58778525229247312916870595463907f, -0.80901699437494742410229341718282f,
		0.58778525229247312916870595463907f, -0.80901699437494742410229341718282f, 0.58778525229247312916870595463907f, -0.80901699437494742410229341718282f);

	int w(0);
	const int outwidth8 = outwidth - outwidth % 8;
	constexpr __mmask16 k1 = 0x5555;
	constexpr __mmask16 k2 = 0xAAAA;

	for (int block = start_block; block < blocks; block++)
	{
		const float gridfraction = degrid * outcur[0][0] / gridsample[0][0];
		const __m512 gridfraction8 = _mm512_set1_ps(gridfraction);
		for (int h = 0; h < bh; h++) // first half
		{
			__m512 r3 = _mm512_load_ps(outprev[0]);
			__m512 r4 = _mm512_load_ps(outnext[0]);
			for (w = 0; w < outwidth8; w = w + 8) // 
			{
				__m512 gridcorrection8 = _mm512_mul_ps(_mm512_load_ps(&gridsample[w][0]), gridfraction8);
				gridcorrection8 = _mm512_mul_ps(gridcorrection8, _mm512_set1_ps(5));
				__m512 r1 = _mm512_load_ps(outprev2[w]);
				__m512 r2 = _mm512_load_ps(outnext2[w]);
				__m512 r5 = _mm512_load_ps(outcur[w]);

				__m512 r6 = _mm512_add_ps(r1, r2);
				r6 = _mm512_mask_sub_ps(r6, k2, r1, r2);
				__m512 r7 = _mm512_add_ps(r4, r3);
				r7 = _mm512_mask_sub_ps(r7, k2, r4, r3);
				r6 = _mm512_mul_ps(r6, cossin72);
				r7 = _mm512_mul_ps(r7, cossin144);
				r7 = _mm512_add_ps(r6, r7);
				r7 = _mm512_mask_add_ps(r5, k1, r7, r5);
				r6 = _mm512_permute_ps(r7, _MM_SHUFFLE(0, 3, 0, 1)); //dif, sum -> r6 == sum!
				__m512 fp2r = _mm512_add_ps(r7, r6);
				__m512 fn2r = _mm512_sub_ps(r7, r6);

				r6 = _mm512_add_ps(r2, r1);
				r6 = _mm512_mask_sub_ps(r6, k1, r2, r1);
				r7 = _mm512_add_ps(r3, r4);
				r7 = _mm512_mask_sub_ps(r7, k1, r3, r4);
				r6 = _mm512_mul_ps(r6, sincos72);
				r7 = _mm512_mul_ps(r7, sincos144);
				r7 = _mm512_add_ps(r6, r7);
				r7 = _mm512_mask_add_ps(r5, k2, r7, r5);
				r6 = _mm512_permute_ps(r7, _MM_SHUFFLE(0, 3, 0, 1)); //dif, sum -> r6 == sum!
				__m512 fp2i = _mm512_add_ps(r7, r6);
				__m512 fn2i = _mm512_sub_ps(r6, r7);

				r6 = _mm512_add_ps(r2, r1);
				r6 = _mm512_mask_sub_ps(r6, k2, r2, r1);
				r7 = _mm512_add_ps(r4, r3);
				r7 = _mm512_mask_sub_ps(r7, k2, r4, r3);
				r6 = _mm512_mul_ps(r6, cossin144);
				r7 = _mm512_mul_ps(r7, cossin72);
				r7 = _mm512_add_ps(r6, r7);
				r7 = _mm512_mask_add_ps(r5, k1, r7, r5);
				r6 = _mm512_permute_ps(r7, _MM_SHUFFLE(0, 3, 0, 1));
				__m512 fpr = _mm512_add_ps(r7, r6);
				__m512 fnr = _mm512_sub_ps(r7, r6);

				r6 = _mm512_add_ps(r1, r2);
				r6 = _mm512_mask_sub_ps(r6, k1, r1, r2);
				r7 = _mm512_add_ps(r3, r4);
				r7 = _mm512_mask_sub_ps(r7, k1, r3, r4);
				r6 = _mm512_mul_ps(r6, sincos144);
				r7 = _mm512_mul_ps(r7, sincos72);
				r7 = _mm512_add_ps(r6, r7);
				r7 = _mm512_mask_add_ps(r5, k2, r7, r5);
				r6 = _mm512_permute_ps(r7, _MM_SHUFFLE(0, 3, 0, 1)); //dif, sum -> r6 == sum!
				__m512 fpi = _mm512_add_ps(r7, r6);
				__m512 fni = _mm512_sub_ps(r6, r7);

				r6 = _mm512_add_ps(r1, r2);
				r7 = _mm512_add_ps(r3, r4);
				r6 = _mm512_add_ps(r5, r6);
				__m512 fc = _mm512_add_ps(r6, r7);
				fc = _mm512_sub_ps(fc, gridcorrection8);

				__m512 pf = _mm512_castps256_ps512(_mm256_load_ps(&pattern3d[w]));
				pf = _mm512_permutexvar_ps(_mm512_set_epi32(7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0), pf);

				__m512 psd = _mm512_mul_ps(fc, fc);
				r1 = _mm512_permute_ps(psd, _MM_SHUFFLE(0, 3, 0, 1)); //psd,-,psd,-
				psd = _mm512_add_ps(psd, r1);
				psd = _mm512_add_ps(psd, _mm512_set1_ps(1e-15f));
				r1 = _mm512_sub_ps(psd, pf);
				r1 = _mm512_div_ps(r1, psd);
				r1 = _mm512_max_ps(r1, _mm512_set1_ps(lowlimit));
				r1 = _mm512_moveldup_ps(r1);
				fc = _mm512_mul_ps(r1, fc);

				r1 = _mm512_mul_ps(fp2r, fp2r);
				r2 = _mm512_mul_ps(fp2i, fp2i);
				fp2r = _mm512_shuffle_ps(fp2r, fp2i, _MM_SHUFFLE(2, 0, 2, 0));
				fp2r = _mm512_permute_ps(fp2r, _MM_SHUFFLE(3, 1, 2, 0));
				psd = _mm512_add_ps(r1, r2);
				psd = _mm512_add_ps(psd, _mm512_set1_ps(1e-15f));
				r1 = _mm512_sub_ps(psd, pf);
				r1 = _mm512_div_ps(r1, psd);
				r1 = _mm512_max_ps(r1, _mm512_set1_ps(lowlimit));
				r1 = _mm512_moveldup_ps(r1);
				fp2r = _mm512_mul_ps(r1, fp2r);

				r1 = _mm512_mul_ps(fpr, fpr);
				r2 = _mm512_mul_ps(fpi, fpi);
				fpr = _mm512_shuffle_ps(fpr, fpi, _MM_SHUFFLE(2, 0, 2, 0));
				fpr = _mm512_permute_ps(fpr, _MM_SHUFFLE(3, 1, 2, 0));
				psd = _mm512_add_ps(r1, r2);
				psd = _mm512_add_ps(psd, _mm512_set1_ps(1e-15f));
				r1 = _mm512_sub_ps(psd, pf);
				r1 = _mm512_div_ps(r1, psd);
				r1 = _mm512_max_ps(r1, _mm512_set1_ps(lowlimit));
				r1 = _mm512_moveldup_ps(r1);
				fpr = _mm512_mul_ps(r1, fpr);

				r1 = _mm512_mul_ps(fnr, fnr);
				r2 = _mm512_mul_ps(fni, fni);
				fnr = _mm512_shuffle_ps(fnr, fni, _MM_SHUFFLE(2, 0, 2, 0));
				fnr = _mm512_permute_ps(fnr, _MM_SHUFFLE(3, 1, 2, 0));
				psd = _mm512_add_ps(r1, r2);
				psd = _mm512_add_ps(psd, _mm512_set1_ps(1e-15f));
				r1 = _mm512_sub_ps(psd, pf);
				r1 = _mm512_div_ps(r1, psd);
				r1 = _mm512_max_ps(r1, _mm512_set1_ps(lowlimit));
				r1 = _mm512_moveldup_ps(r1);
				fnr = _mm512_mul_ps(r1, fnr);

				r1 = _mm512_mul_ps(fn2r, fn2r);
				r2 = _mm512_mul_ps(fn2i, fn2i);
				fn2r = _mm512_shuffle_ps(fn2r, fn2i, _MM_SHUFFLE(2, 0, 2, 0));
				fn2r = _mm512_permute_ps(fn2r, _MM_SHUFFLE(3, 1, 2, 0));
				psd = _mm512_add_ps(r1, r2);
				psd = _mm512_add_ps(psd, _mm512_set1_ps(1e-15f));
				r1 = _mm512_sub_ps(psd, pf);
				r1 = _mm512_div_ps(r1, psd);
				r1 = _mm512_max_ps(r1, _mm512_set1_ps(lowlimit));
				r1 = _mm512_moveldup_ps(r1);
				fn2r = _mm512_mul_ps(r1, fn2r);

				r1 = _mm512_add_ps(fp2r, fpr);
				r2 = _mm512_add_ps(fc, fnr);
				r1 = _mm512_add_ps(r1, fn2r);
				r1 = _mm512_add_ps(r1, r2);
				r3 = _mm512_load_ps(outprev[w + 8]);
				r4 = _mm512_load_ps(outnext[w + 8]);
				r1 = _mm512_add_ps(r1, gridcorrection8);
				r1 = _mm512_mul_ps(r1, _mm512_set1_ps(0.2f));
				_mm512_store_ps(outprev2[w], r1);
			}
			_mm_prefetch((const char*)(outprev2 + outpitch), _MM_HINT_T0);
			_mm_prefetch((const char*)(outprev + outpitch), _MM_HINT_T0);
			_mm_prefetch((const char*)(outcur + outpitch), _MM_HINT_T0);
			_mm_prefetch((const char*)(outnext + outpitch), _MM_HINT_T0);
			_mm_prefetch((const char*)(outnext2 + outpitch), _MM_HINT_T0);
			_mm_prefetch((const char*)(gridsample + outpitch), _MM_HINT_T0);
			_mm_prefetch((const char*)(pattern3d + outpitch), _MM_HINT_T0);
			for (; w < outwidth; w++) // 
			{
				float fcr, fci, fpr, fpi, fnr, fni, fp2r, fp2i, fn2r, fn2i, WienerFactor, psd;
				float gridcorrection0_5 = gridfraction * gridsample[w][0] * 5;
				float gridcorrection1_5 = gridfraction * gridsample[w][1] * 5;
				float sum = (outprev2[w][0] + outnext2[w][0])*cos72 + (outprev[w][0] + outnext[w][0])*cos144 + +outcur[w][0];
				float dif = (-outprev2[w][1] + outnext2[w][1])*sin72 + (outprev[w][1] - outnext[w][1])*sin144;
				fp2r = sum + dif; // real prev2
				fn2r = sum - dif; // real next2
				sum = (outprev2[w][1] + outnext2[w][1])*cos72 + (outprev[w][1] + outnext[w][1])*cos144 + outcur[w][1];
				dif = (outprev2[w][0] - outnext2[w][0])*sin72 + (-outprev[w][0] + outnext[w][0])*sin144;
				fp2i = sum + dif; // im prev2
				fn2i = sum - dif; // im next2
				sum = (outprev2[w][0] + outnext2[w][0])*cos144 + (outprev[w][0] + outnext[w][0])*cos72 + outcur[w][0];
				dif = (outprev2[w][1] - outnext2[w][1])*sin144 + (outprev[w][1] - outnext[w][1])*sin72;
				fpr = sum + dif; // real prev
				fnr = sum - dif; // real next
				sum = (outprev2[w][1] + outnext2[w][1])*cos144 + (outprev[w][1] + outnext[w][1])*cos72 + outcur[w][1];
				dif = (-outprev2[w][0] + outnext2[w][0])*sin144 + (-outprev[w][0] + outnext[w][0])*sin72;
				fpi = sum + dif; // im prev
				fni = sum - dif; // im next
				fcr = outprev2[w][0] + outprev[w][0] + outcur[w][0] + outnext[w][0] + outnext2[w][0]; // real cur
				fcr -= gridcorrection0_5;
				fci = outprev2[w][1] + outprev[w][1] + outcur[w][1] + outnext[w][1] + outnext2[w][1]; // im cur
				fci -= gridcorrection1_5;

				psd = fp2r * fp2r + fp2i * fp2i + 1e-15f; // power spectrum density prev2
				WienerFactor = max((psd - pattern3d[w]) / psd, lowlimit); // limited Wiener filter
				fp2r *= WienerFactor; // apply filter on real  part	
				fp2i *= WienerFactor; // apply filter on imaginary part

				psd = fpr * fpr + fpi * fpi + 1e-15f; // power spectrum density prev
				WienerFactor = max((psd - pattern3d[w]) / psd, lowlimit); // limited Wiener filter
				fpr *= WienerFactor; // apply filter on real  part	
				fpi *= WienerFactor; // apply filter on imaginary part

				psd = fcr * fcr + fci * fci + 1e-15f; // power spectrum density cur
				WienerFactor = max((psd - pattern3d[w]) / psd, lowlimit); // limited Wiener filter
				fcr *= WienerFactor; // apply filter on real  part	
				fci *= WienerFactor; // apply filter on imaginary part

				psd = fnr * fnr + fni * fni + 1e-15f; // power spectrum density next
				WienerFactor = max((psd - pattern3d[w]) / psd, lowlimit); // limited Wiener filter
				fnr *= WienerFactor; // apply filter on real  part	
				fni *= WienerFactor; // apply filter on imaginary part

				psd = fn2r * fn2r + fn2i * fn2i + 1e-15f; // power spectrum density next
				WienerFactor = max((psd - pattern3d[w]) / psd, lowlimit); // limited Wiener filter
				fn2r *= WienerFactor; // apply filter on real  part	
				fn2i *= WienerFactor; // apply filter on imaginary part

									  // reverse dft for 5 points
				outprev2[w][0] = (fp2r + fpr + fcr + fnr + fn2r + gridcorrection0_5)*0.2f; // get  real  part	
				outprev2[w][1] = (fp2i + fpi + fci + fni + fn2i + gridcorrection1_5)*0.2f; // get imaginary part
																						   // Attention! return filtered "out" in "outprev2" to preserve "out" for next step
			}
			outcur += outpitch;
			outprev2 += outpitch;
			outprev += outpitch;
			outnext += outpitch;
			outnext2 += outpitch;
			gridsample += outpitch;
			pattern3d += outpitch;
		}
		gridsample -= outpitch * bh; // restore pointer to only valid first block
		pattern3d -= outpitch * bh; // restore pointer
	}
}