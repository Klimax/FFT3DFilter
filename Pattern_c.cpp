//
//	FFT3DFilter plugin for Avisynth 2.6 - 3D Frequency Domain filter
//  pure C++ filtering functions
//
//	Copyright(C)2004-2006 A.G.Balakhnin aka Fizick, bag@hotmail.ru, http://avisynth.org.ru
//  Copyright(C)2018 Klimax
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
#include "WienerPattern.h"

// since v1.7 we use outpitch instead of outwidth
#ifndef SSE2BUILD
//-------------------------------------------------------------------------------------------
//
void PatternFilter::ApplyPattern2D_C() noexcept
{
	float psd(0.0f), patternfactor(0.0f);
	float *pattern2d(nullptr);

	if (pfactor != 0)
	{
		for (int block = start_block; block < blocks; block++)
		{
			pattern2d = pattern3d;
			for (int h = 0; h < bh; h++) // middle
			{
				for (int w = 0; w < outwidth; w++)
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
}

void PatternFilter::ApplyPattern2D_degrid_C() noexcept
{
	float psd(0.0f), WienerFactor(0.0f);
	for (int block = start_block; block < blocks; block++)
	{
		float gridfraction = degrid * outcur[0][0] / gridsample[0][0];
		float *pattern2d = pattern3d;
		for (int h = 0; h < bh; h++) // middle
		{
			for (int w = 0; w < outwidth; w++)
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
//
//-----------------------------------------------------------------------------------------
//
void PatternFilter::ApplyPattern3D2_C() noexcept
{
	// return result in outprev
	float psd(0.0f), WienerFactor(0.0f);
	float f3d0r(0.0f), f3d1r(0.0f), f3d0i(0.0f), f3d1i(0.0f);

	for (int block = start_block; block < blocks; block++)
	{
		for (int h = 0; h < bh; h++)
		{
			for (int w = 0; w < outwidth; w++)
			{
				// dft 3d (very short - 2 points)
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

void PatternFilter::ApplyPattern3D2_degrid_C() noexcept
{
	// return result in outprev
	float psd(0.0f), WienerFactor(0.0f);
	float f3d0r(0.0f), f3d1r(0.0f), f3d0i(0.0f), f3d1i(0.0f);


	for (int block = start_block; block < blocks; block++)
	{
		const float gridfraction = degrid * outcur[0][0] / gridsample[0][0];
		for (int h = 0; h < bh; h++)
		{
			for (int w = 0; w < outwidth; w++)
			{
				float gridcorrection0_2 = gridfraction * gridsample[w][0] * 2; // grid correction
				float gridcorrection1_2 = gridfraction * gridsample[w][1] * 2;
				// dft 3d (very short - 2 points)
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
void PatternFilter::ApplyPattern3D3_C() noexcept
{
	// return result in outprev
	float fcr(0.0f), fci(0.0f), fpr(0.0f), fpi(0.0f), fnr(0.0f), fni(0.0f);
	float WienerFactor(0.0f), psd(0.0f);
	constexpr float sin120 = 0.86602540378443864676372317075294f;//sqrtf(3.0f)*0.5f;

	for (int block = start_block; block < blocks; block++)
	{
		for (int h = 0; h < bh; h++) // first half
		{
			for (int w = 0; w < outwidth; w++) // 
			{
				// dft 3d (very short - 3 points)
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

void PatternFilter::ApplyPattern3D3_degrid_C() noexcept
{
	// return result in outprev
	float fcr(0.0f), fci(0.0f), fpr(0.0f), fpi(0.0f), fnr(0.0f), fni(0.0f);
	float WienerFactor(0.0f), psd(0.0f);
	constexpr float sin120 = 0.86602540378443864676372317075294f;//sqrtf(3.0f)*0.5f;


	for (int block = start_block; block < blocks; block++)
	{
		const float gridfraction = degrid * outcur[0][0] / gridsample[0][0];
		for (int h = 0; h < bh; h++) // first half
		{
			for (int w = 0; w < outwidth; w++) // 
			{
				float gridcorrection0_3 = gridfraction * gridsample[w][0] * 3;
				float gridcorrection1_3 = gridfraction * gridsample[w][1] * 3;
				// dft 3d (very short - 3 points)
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
//-----------------------------------------------------------------------------------------
//
void PatternFilter::ApplyPattern3D4_C() noexcept
{
	// dft with 4 points
	// return result in outprev
	float fcr(0.0f), fci(0.0f), fpr(0.0f), fpi(0.0f), fnr(0.0f), fni(0.0f), fp2r(0.0f), fp2i(0.0f);
	float WienerFactor(0.0f), psd(0.0f);

	for (int block = start_block; block < blocks; block++)
	{
		for (int h = 0; h < bh; h++) // first half
		{
			for (int w = 0; w < outwidth; w++) // 
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

void PatternFilter::ApplyPattern3D4_degrid_C() noexcept
{
	// dft with 4 points
	// return result in outprev
	float fcr(0.0f), fci(0.0f), fpr(0.0f), fpi(0.0f), fnr(0.0f), fni(0.0f), fp2r(0.0f), fp2i(0.0f);
	float WienerFactor(0.0f), psd(0.0f);

	for (int block = start_block; block < blocks; block++)
	{
		const float gridfraction = degrid * outcur[0][0] / gridsample[0][0];
		for (int h = 0; h < bh; h++) // first half
		{
			for (int w = 0; w < outwidth; w++) // 
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
//-----------------------------------------------------------------------------------------
//
void PatternFilter::ApplyPattern3D5_C() noexcept
{
	// dft with 5 points
	// return result in outprev2
	float fcr(0.0f), fci(0.0f), fpr(0.0f), fpi(0.0f), fnr(0.0f), fni(0.0f), fp2r(0.0f), fp2i(0.0f), fn2r(0.0f), fn2i(0.0f);
	float WienerFactor(0.0f), psd(0.0f);

	constexpr float sin72 = 0.95105651629515357211643933337938f;// 2*pi/5
	constexpr float cos72 = 0.30901699437494742410229341718282f;
	constexpr float sin144 = 0.58778525229247312916870595463907f;
	constexpr float cos144 = -0.80901699437494742410229341718282f;

	for (int block = start_block; block < blocks; block++)
	{
		for (int h = 0; h < bh; h++) // first half
		{
			for (int w = 0; w < outwidth; w++) // 
			{
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

void PatternFilter::ApplyPattern3D5_degrid_C() noexcept
{
	// dft with 5 points
	// return result in outprev2
	float fcr(0.0f), fci(0.0f), fpr(0.0f), fpi(0.0f), fnr(0.0f), fni(0.0f), fp2r(0.0f), fp2i(0.0f), fn2r(0.0f), fn2i(0.0f);
	float WienerFactor(0.0f), psd(0.0f);

	constexpr float sin72 = 0.95105651629515357211643933337938f;// 2*pi/5
	constexpr float cos72 = 0.30901699437494742410229341718282f;
	constexpr float sin144 = 0.58778525229247312916870595463907f;
	constexpr float cos144 = -0.80901699437494742410229341718282f;

	for (int block = start_block; block < blocks; block++)
	{
		float gridfraction = degrid * outcur[0][0] / gridsample[0][0];
		for (int h = 0; h < bh; h++) // first half
		{
			for (int w = 0; w < outwidth; w++) // 
			{
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

void FindPatternBlock_C(fftwf_complex *outcur0, int outwidth, int outpitch, int bh, int nox, int noy, int &px, int &py, float *pwin, float degrid, fftwf_complex *gridsample) noexcept
{
	// since v1.7 outwidth must be really an outpitch
	fftwf_complex *outcur(nullptr);
	float psd(0.0f), sigmaSquaredcur(0.0f), sigmaSquared(1e15f);

	for (int by = 2; by < noy - 2; by++)
	{
		for (int bx = 2; bx < nox - 2; bx++)
		{
			outcur = outcur0 + nox * by*bh*outpitch + bx * bh*outpitch;
			sigmaSquaredcur = 0;
			float gcur = degrid * outcur[0][0] / gridsample[0][0]; // grid (windowing) correction factor
			for (int h = 0; h < bh; h++)
			{
				for (int w = 0; w < outwidth; w++)
				{
					float grid0 = gcur * gridsample[w][0];
					float grid1 = gcur * gridsample[w][1];
					float corrected0 = outcur[w][0] - grid0;
					float corrected1 = outcur[w][1] - grid1;
					psd = corrected0 * corrected0 + corrected1 * corrected1;
					sigmaSquaredcur += psd * pwin[w]; // windowing
				}
				outcur += outpitch;
				pwin += outpitch;
				gridsample += outpitch;
			}
			pwin -= outpitch * bh; // restore
			gridsample -= outpitch * bh; // restore
			if (sigmaSquaredcur < sigmaSquared)
			{
				px = bx;
				py = by;
				sigmaSquared = sigmaSquaredcur;
			}
		}
	}
}
//-------------------------------------------------------------------------------------------

#endif
//-------------------------------------------------------------------------------------------
void Pattern2Dto3D_C(const float *pattern2d, int bh, int outpitch, float mult, float *pattern3d) noexcept
{
	// slow, but executed once only per clip
	const int size = bh * outpitch;
	for (int i = 0; i < size; i++)
	{ // get 3D pattern
		pattern3d[i] = pattern2d[i] * mult;
	}
}

void SetPattern_C(fftwf_complex *outcur, int outwidth, int outpitch, int bh, int nox, int px, int py, float *pwin,
	float *pattern2d, float &psigma, float degrid, fftwf_complex *gridsample) noexcept
{
	outcur += nox * py*bh*outpitch + px * bh*outpitch;
	float psd(0.0f), sigmaSquared(0.0f), weight(0.0f);

	for (int h = 0; h < bh; h++)
	{
		for (int w = 0; w < outwidth; w++)
		{
			weight += pwin[w];
		}
		pwin += outpitch;
	}
	pwin -= outpitch * bh; // restore

	const float gcur = degrid * outcur[0][0] / gridsample[0][0]; // grid (windowing) correction factor

	for (int h = 0; h < bh; h++)
	{
		for (int w = 0; w < outwidth; w++)
		{
			float grid0 = gcur * gridsample[w][0];
			float grid1 = gcur * gridsample[w][1];
			float corrected0 = outcur[w][0] - grid0;
			float corrected1 = outcur[w][1] - grid1;
			psd = corrected0 * corrected0 + corrected1 * corrected1;
			//			psd = outcur[w][0]*outcur[w][0] + outcur[w][1]*outcur[w][1];
			pattern2d[w] = psd * pwin[w]; // windowing
			sigmaSquared += pattern2d[w]; // sum
		}
		outcur += outpitch;
		pattern2d += outpitch;
		pwin += outpitch;
		gridsample += outpitch;
	}
	psigma = sqrtf(sigmaSquared / (weight*bh*outwidth)); // mean std deviation (sigma)
}