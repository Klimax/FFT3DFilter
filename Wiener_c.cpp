//
//	FFT3DFilter plugin for Avisynth 2.6 - 3D Frequency Domain filter
//  pure C++ filtering functions
//
//	Copyright(C)2004-2006 A.G.Balakhnin aka Fizick, bag@hotmail.ru, http://avisynth.org.ru
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
#include "math.h" // for sqrtf
#include "Wiener.h"

// since v1.7 we use outpitch instead of outwidth

//-----------------------------------------------------------------------------------------
//
void WienerFilter::ApplyWiener2D_C() noexcept
{
	float psd(0.0f);
	float WienerFactor(0.0f);

	if (sharpen == 0 && dehalo == 0)// no sharpen, no dehalo
	{
		for (int block = start_block; block < blocks; block++)
		{
			for (int h = 0; h < bh; h++) // middle
			{
				for (int w = 0; w < outwidth; w++) // not skip first v.1.2
				{
					psd = (outcur[w][0] * outcur[w][0] + outcur[w][1] * outcur[w][1]) + 1e-15f;// power spectrum density
					WienerFactor = max((psd - sigmaSquaredNoiseNormed) / psd, lowlimit); // limited Wiener filter
					outcur[w][0] *= WienerFactor; // apply filter on real  part	
					outcur[w][1] *= WienerFactor; // apply filter on imaginary part
				}
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
				for (int w = 0; w < outwidth; w++) // not skip first
				{
					psd = (outcur[w][0] * outcur[w][0] + outcur[w][1] * outcur[w][1]) + 1e-15f;// power spectrum density
					WienerFactor = max((psd - sigmaSquaredNoiseNormed) / psd, lowlimit); // limited Wiener filter
					WienerFactor *= 1 + sharpen * wsharpen[w] * sqrtf(psd*sigmaSquaredSharpenMax / ((psd + sigmaSquaredSharpenMin)*(psd + sigmaSquaredSharpenMax))); // sharpen factor - changed in v.1.1
					outcur[w][0] *= WienerFactor; // apply filter on real  part	
					outcur[w][1] *= WienerFactor; // apply filter on imaginary part
				}
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
				for (int w = 0; w < outwidth; w++) // not skip first
				{
					psd = (outcur[w][0] * outcur[w][0] + outcur[w][1] * outcur[w][1]) + 1e-15f;// power spectrum density
					WienerFactor = max((psd - sigmaSquaredNoiseNormed) / psd, lowlimit); // limited Wiener filter
					WienerFactor *= (psd + ht2n) / ((psd + ht2n) + dehalo * wdehalo[w] * psd);
					outcur[w][0] *= WienerFactor; // apply filter on real  part	
					outcur[w][1] *= WienerFactor; // apply filter on imaginary part
				}
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
				for (int w = 0; w < outwidth; w++) // not skip first
				{
					psd = (outcur[w][0] * outcur[w][0] + outcur[w][1] * outcur[w][1]) + 1e-15f;// power spectrum density
					WienerFactor = max((psd - sigmaSquaredNoiseNormed) / psd, lowlimit); // limited Wiener filter
					WienerFactor *= 1 + sharpen * wsharpen[w] * sqrtf(psd*sigmaSquaredSharpenMax / ((psd + sigmaSquaredSharpenMin)*(psd + sigmaSquaredSharpenMax))) *
						(psd + ht2n) / ((psd + ht2n) + dehalo * wdehalo[w] * psd);
					outcur[w][0] *= WienerFactor; // apply filter on real  part	
					outcur[w][1] *= WienerFactor; // apply filter on imaginary part
				}
				outcur += outpitch;
				wsharpen += outpitch;
				wdehalo += outpitch;
			}
			wsharpen -= outpitch * bh;
			wdehalo -= outpitch * bh;
		}
	}
}

void WienerFilter::ApplyWiener2D_degrid_C() noexcept
{
	float psd(0.0f), WienerFactor(0.0f);

	if (sharpen == 0 && dehalo == 0)// no sharpen, no dehalo
	{
		for (int block = start_block; block < blocks; block++)
		{
			const float gridfraction = degrid * outcur[0][0] / gridsample[0][0];
			for (int h = 0; h < bh; h++) // middle
			{
				for (int w = 0; w < outwidth; w++) // not skip first v.1.2
				{
					float gridcorrection0 = gridfraction * gridsample[w][0];
					float corrected0 = outcur[w][0] - gridcorrection0;
					float gridcorrection1 = gridfraction * gridsample[w][1];
					float corrected1 = outcur[w][1] - gridcorrection1;
					psd = (corrected0*corrected0 + corrected1 * corrected1) + 1e-15f;// power spectrum density
																					 //					psd = (outcur[w][0]*outcur[w][0] + outcur[w][1]*outcur[w][1]) + 1e-15f;
					WienerFactor = max((psd - sigmaSquaredNoiseNormed) / psd, lowlimit); // limited Wiener filter
					corrected0 *= WienerFactor; // apply filter on real  part	
					corrected1 *= WienerFactor; // apply filter on imaginary part
					outcur[w][0] = corrected0 + gridcorrection0;
					outcur[w][1] = corrected1 + gridcorrection1;
				}
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
			const float gridfraction = degrid * outcur[0][0] / gridsample[0][0];
			for (int h = 0; h < bh; h++) // middle
			{
				for (int w = 0; w < outwidth; w++) // not skip first
				{
					//					psd = (outcur[w][0]*outcur[w][0] + outcur[w][1]*outcur[w][1]) + 1e-15f;// power spectrum density
					float gridcorrection0 = gridfraction * gridsample[w][0];
					float corrected0 = outcur[w][0] - gridcorrection0;
					float gridcorrection1 = gridfraction * gridsample[w][1];
					float corrected1 = outcur[w][1] - gridcorrection1;
					psd = (corrected0*corrected0 + corrected1 * corrected1) + 1e-15f;// power spectrum density
					WienerFactor = max((psd - sigmaSquaredNoiseNormed) / psd, lowlimit); // limited Wiener filter
					WienerFactor *= 1 + sharpen * wsharpen[w] * sqrtf(psd*sigmaSquaredSharpenMax / ((psd + sigmaSquaredSharpenMin)*(psd + sigmaSquaredSharpenMax))); // sharpen factor - changed in v.1.1
																																									 //					outcur[w][0] *= WienerFactor; // apply filter on real  part	
																																									 //					outcur[w][1] *= WienerFactor; // apply filter on imaginary part
					corrected0 *= WienerFactor; // apply filter on real  part	
					corrected1 *= WienerFactor; // apply filter on imaginary part
					outcur[w][0] = corrected0 + gridcorrection0;
					outcur[w][1] = corrected1 + gridcorrection1;
				}
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
			const float gridfraction = degrid * outcur[0][0] / gridsample[0][0];
			for (int h = 0; h < bh; h++) // middle
			{
				for (int w = 0; w < outwidth; w++) // not skip first
				{
					//					psd = (outcur[w][0]*outcur[w][0] + outcur[w][1]*outcur[w][1]) + 1e-15f;// power spectrum density
					float gridcorrection0 = gridfraction * gridsample[w][0];
					float corrected0 = outcur[w][0] - gridcorrection0;
					float gridcorrection1 = gridfraction * gridsample[w][1];
					float corrected1 = outcur[w][1] - gridcorrection1;
					psd = (corrected0*corrected0 + corrected1 * corrected1) + 1e-15f;// power spectrum density
					WienerFactor = max((psd - sigmaSquaredNoiseNormed) / psd, lowlimit); // limited Wiener filter
					WienerFactor *= (psd + ht2n) / ((psd + ht2n) + dehalo * wdehalo[w] * psd);
					//					outcur[w][0] *= WienerFactor; // apply filter on real  part	
					//					outcur[w][1] *= WienerFactor; // apply filter on imaginary part
					corrected0 *= WienerFactor; // apply filter on real  part	
					corrected1 *= WienerFactor; // apply filter on imaginary part
					outcur[w][0] = corrected0 + gridcorrection0;
					outcur[w][1] = corrected1 + gridcorrection1;
				}
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
			const float gridfraction = degrid * outcur[0][0] / gridsample[0][0];
			for (int h = 0; h < bh; h++) // middle
			{
				for (int w = 0; w < outwidth; w++) // not skip first
				{
					//					psd = (outcur[w][0]*outcur[w][0] + outcur[w][1]*outcur[w][1]) + 1e-15f;// power spectrum density
					float gridcorrection0 = gridfraction * gridsample[w][0];
					float corrected0 = outcur[w][0] - gridcorrection0;
					float gridcorrection1 = gridfraction * gridsample[w][1];
					float corrected1 = outcur[w][1] - gridcorrection1;
					psd = (corrected0*corrected0 + corrected1 * corrected1) + 1e-15f;// power spectrum density
					WienerFactor = max((psd - sigmaSquaredNoiseNormed) / psd, lowlimit); // limited Wiener filter
					WienerFactor *= 1 + sharpen * wsharpen[w] * sqrtf(psd*sigmaSquaredSharpenMax / ((psd + sigmaSquaredSharpenMin)*(psd + sigmaSquaredSharpenMax))) *
						(psd + ht2n) / ((psd + ht2n) + dehalo * wdehalo[w] * psd);
					//					outcur[w][0] *= WienerFactor; // apply filter on real  part	
					//					outcur[w][1] *= WienerFactor; // apply filter on imaginary part
					corrected0 *= WienerFactor; // apply filter on real  part	
					corrected1 *= WienerFactor; // apply filter on imaginary part
					outcur[w][0] = corrected0 + gridcorrection0;
					outcur[w][1] = corrected1 + gridcorrection1;
				}
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

//
//-----------------------------------------------------------------------------------------
//
void WienerFilter::ApplyWiener3D2_C() noexcept
{
	// return result in outprev
	float psd(0.0f);
	float WienerFactor(0.0f);
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
				WienerFactor = max((psd - sigmaSquaredNoiseNormed) / psd, lowlimit); // limited Wiener filter
				f3d0r *= WienerFactor; // apply filter on real  part	
				f3d0i *= WienerFactor; // apply filter on imaginary part
				psd = f3d1r * f3d1r + f3d1i * f3d1i + 1e-15f; // power spectrum density 1
				WienerFactor = max((psd - sigmaSquaredNoiseNormed) / psd, lowlimit); // limited Wiener filter
				f3d1r *= WienerFactor; // apply filter on real  part	
				f3d1i *= WienerFactor; // apply filter on imaginary part
				// reverse dft for 2 points
				outprev[w][0] = (f3d0r + f3d1r)*0.5f; // get  real  part	
				outprev[w][1] = (f3d0i + f3d1i)*0.5f; // get imaginary part
				// Attention! return filtered "out" in "outprev" to preserve "out" for next step
			}
			outcur += outpitch;
			outprev += outpitch;
		}
	}
}

void WienerFilter::ApplyWiener3D2_degrid_C() noexcept
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
				// dft 3d (very short - 2 points)
				float gridcorrection0_2 = gridfraction * gridsample[w][0] * 2; // grid correction
				float gridcorrection1_2 = gridfraction * gridsample[w][1] * 2;
				f3d0r = outcur[w][0] + outprev[w][0] - gridcorrection0_2; // real 0 (sum)
				f3d0i = outcur[w][1] + outprev[w][1] - gridcorrection1_2; // im 0 (sum)
				psd = f3d0r * f3d0r + f3d0i * f3d0i + 1e-15f; // power spectrum density 0
				WienerFactor = max((psd - sigmaSquaredNoiseNormed) / psd, lowlimit); // limited Wiener filter
				f3d0r *= WienerFactor; // apply filter on real  part	
				f3d0i *= WienerFactor; // apply filter on imaginary part

				f3d1r = outcur[w][0] - outprev[w][0]; // real 1 (dif)
				f3d1i = outcur[w][1] - outprev[w][1]; // im 1 (dif)
				psd = f3d1r * f3d1r + f3d1i * f3d1i + 1e-15f; // power spectrum density 1
				WienerFactor = max((psd - sigmaSquaredNoiseNormed) / psd, lowlimit); // limited Wiener filter
				f3d1r *= WienerFactor; // apply filter on real  part	
				f3d1i *= WienerFactor; // apply filter on imaginary part
									   // reverse dft for 2 points
				outprev[w][0] = (f3d0r + f3d1r + gridcorrection0_2)*0.5f; // get  real  part	
				outprev[w][1] = (f3d0i + f3d1i + gridcorrection1_2)*0.5f; // get imaginary part
																		  // Attention! return filtered "out" in "outprev" to preserve "out" for next step
			}
			outcur += outpitch;
			outprev += outpitch;
			gridsample += outpitch;
		}
		gridsample -= outpitch * bh; // restore pointer to only valid first block
	}
}
//
//-----------------------------------------------------------------------------------------
//
void WienerFilter::ApplyWiener3D3_C() noexcept
{
	// return result in outprev
	float fcr(0.0f), fci(0.0f), fpr(0.0f), fpi(0.0f), fnr(0.0f), fni(0.0f);
	float WienerFactor(0.0f);
	float psd(0.0f);
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
				fnr = fpr - di - di; //v1.8.1
				fpi = outcur[w][1] - 0.5f*pni + dr; // im prev
				fni = fpi - dr - dr; //v1.8.1
				psd = fcr * fcr + fci * fci + 1e-15f; // power spectrum density cur
				WienerFactor = max((psd - sigmaSquaredNoiseNormed) / psd, lowlimit); // limited Wiener filter
				fcr *= WienerFactor; // apply filter on real  part	
				fci *= WienerFactor; // apply filter on imaginary part
				psd = fpr * fpr + fpi * fpi + 1e-15f; // power spectrum density prev
				WienerFactor = max((psd - sigmaSquaredNoiseNormed) / psd, lowlimit); // limited Wiener filter
				fpr *= WienerFactor; // apply filter on real  part	
				fpi *= WienerFactor; // apply filter on imaginary part
				psd = fnr * fnr + fni * fni + 1e-15f; // power spectrum density next
				WienerFactor = max((psd - sigmaSquaredNoiseNormed) / psd, lowlimit); // limited Wiener filter
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
		}
	}
}

void WienerFilter::ApplyWiener3D3_degrid_C() noexcept
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
				WienerFactor = max((psd - sigmaSquaredNoiseNormed) / psd, lowlimit); // limited Wiener filter
				fcr *= WienerFactor; // apply filter on real  part	
				fci *= WienerFactor; // apply filter on imaginary part
				psd = fpr * fpr + fpi * fpi + 1e-15f; // power spectrum density prev
				WienerFactor = max((psd - sigmaSquaredNoiseNormed) / psd, lowlimit); // limited Wiener filter
				fpr *= WienerFactor; // apply filter on real  part	
				fpi *= WienerFactor; // apply filter on imaginary part
				psd = fnr * fnr + fni * fni + 1e-15f; // power spectrum density next
				WienerFactor = max((psd - sigmaSquaredNoiseNormed) / psd, lowlimit); // limited Wiener filter
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
			gridsample += outpitch;
		}
		gridsample -= outpitch * bh; // restore pointer to only valid first block
	}
}
//-----------------------------------------------------------------------------------------
//
void WienerFilter::ApplyWiener3D4_C() noexcept
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
				// dft 3d (very short - 4 points)
				fpr = -outprev2[w][0] + outprev[w][1] + outcur[w][0] - outnext[w][1]; // real prev
				fpi = -outprev2[w][1] - outprev[w][0] + outcur[w][1] + outnext[w][0]; // im cur
				fcr = outprev2[w][0] + outprev[w][0] + outcur[w][0] + outnext[w][0]; // real cur
				fci = outprev2[w][1] + outprev[w][1] + outcur[w][1] + outnext[w][1]; // im cur
				fnr = -outprev2[w][0] - outprev[w][1] + outcur[w][0] + outnext[w][1]; // real next
				fni = -outprev2[w][1] + outprev[w][0] + outcur[w][1] - outnext[w][0]; // im next
				fp2r = outprev2[w][0] - outprev[w][0] + outcur[w][0] - outnext[w][0]; // real prev2
				fp2i = outprev2[w][1] - outprev[w][1] + outcur[w][1] - outnext[w][1]; // im cur

				psd = fp2r * fp2r + fp2i * fp2i + 1e-15f; // power spectrum density prev2
				WienerFactor = max((psd - sigmaSquaredNoiseNormed) / psd, lowlimit); // limited Wiener filter
				fp2r *= WienerFactor; // apply filter on real  part	
				fp2i *= WienerFactor; // apply filter on imaginary part

				psd = fpr * fpr + fpi * fpi + 1e-15f; // power spectrum density prev
				WienerFactor = max((psd - sigmaSquaredNoiseNormed) / psd, lowlimit); // limited Wiener filter
				fpr *= WienerFactor; // apply filter on real  part	
				fpi *= WienerFactor; // apply filter on imaginary part

				psd = fcr * fcr + fci * fci + 1e-15f; // power spectrum density cur
				WienerFactor = max((psd - sigmaSquaredNoiseNormed) / psd, lowlimit); // limited Wiener filter
				fcr *= WienerFactor; // apply filter on real  part	
				fci *= WienerFactor; // apply filter on imaginary part

				psd = fnr * fnr + fni * fni + 1e-15f; // power spectrum density next
				WienerFactor = max((psd - sigmaSquaredNoiseNormed) / psd, lowlimit); // limited Wiener filter
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
		}
	}
}

void WienerFilter::ApplyWiener3D4_degrid_C() noexcept
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
				// dft 3d (very short - 4 points)
				fpr = -outprev2[w][0] + outprev[w][1] + outcur[w][0] - outnext[w][1]; // real prev
				fpi = -outprev2[w][1] - outprev[w][0] + outcur[w][1] + outnext[w][0]; // im cur
				fcr = outprev2[w][0] + outprev[w][0] + outcur[w][0] + outnext[w][0]; // real cur
				fcr -= gridcorrection0_4;
				fci = outprev2[w][1] + outprev[w][1] + outcur[w][1] + outnext[w][1]; // im cur
				fci -= gridcorrection1_4;
				fnr = -outprev2[w][0] - outprev[w][1] + outcur[w][0] + outnext[w][1]; // real next
				fni = -outprev2[w][1] + outprev[w][0] + outcur[w][1] - outnext[w][0]; // im next
				fp2r = outprev2[w][0] - outprev[w][0] + outcur[w][0] - outnext[w][0]; // real prev2
				fp2i = outprev2[w][1] - outprev[w][1] + outcur[w][1] - outnext[w][1]; // im cur

				psd = fp2r * fp2r + fp2i * fp2i + 1e-15f; // power spectrum density prev2
				WienerFactor = max((psd - sigmaSquaredNoiseNormed) / psd, lowlimit); // limited Wiener filter
				fp2r *= WienerFactor; // apply filter on real  part	
				fp2i *= WienerFactor; // apply filter on imaginary part

				psd = fpr * fpr + fpi * fpi + 1e-15f; // power spectrum density prev
				WienerFactor = max((psd - sigmaSquaredNoiseNormed) / psd, lowlimit); // limited Wiener filter
				fpr *= WienerFactor; // apply filter on real  part	
				fpi *= WienerFactor; // apply filter on imaginary part

				psd = fcr * fcr + fci * fci + 1e-15f; // power spectrum density cur
				WienerFactor = max((psd - sigmaSquaredNoiseNormed) / psd, lowlimit); // limited Wiener filter
				fcr *= WienerFactor; // apply filter on real  part	
				fci *= WienerFactor; // apply filter on imaginary part

				psd = fnr * fnr + fni * fni + 1e-15f; // power spectrum density next
				WienerFactor = max((psd - sigmaSquaredNoiseNormed) / psd, lowlimit); // limited Wiener filter
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
			gridsample += outpitch;
		}
		gridsample -= outpitch * bh; // restore pointer to only valid first block
	}
}
//-----------------------------------------------------------------------------------------
//
void WienerFilter::ApplyWiener3D5_C() noexcept
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
				// dft 3d (very short - 5 points)
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
				WienerFactor = max((psd - sigmaSquaredNoiseNormed) / psd, lowlimit); // limited Wiener filter
				fp2r *= WienerFactor; // apply filter on real  part	
				fp2i *= WienerFactor; // apply filter on imaginary part

				psd = fpr * fpr + fpi * fpi + 1e-15f; // power spectrum density prev
				WienerFactor = max((psd - sigmaSquaredNoiseNormed) / psd, lowlimit); // limited Wiener filter
				fpr *= WienerFactor; // apply filter on real  part	
				fpi *= WienerFactor; // apply filter on imaginary part

				psd = fcr * fcr + fci * fci + 1e-15f; // power spectrum density cur
				WienerFactor = max((psd - sigmaSquaredNoiseNormed) / psd, lowlimit); // limited Wiener filter
				fcr *= WienerFactor; // apply filter on real  part	
				fci *= WienerFactor; // apply filter on imaginary part

				psd = fnr * fnr + fni * fni + 1e-15f; // power spectrum density next
				WienerFactor = max((psd - sigmaSquaredNoiseNormed) / psd, lowlimit); // limited Wiener filter
				fnr *= WienerFactor; // apply filter on real  part	
				fni *= WienerFactor; // apply filter on imaginary part

				psd = fn2r * fn2r + fn2i * fn2i + 1e-15f; // power spectrum density next2
				WienerFactor = max((psd - sigmaSquaredNoiseNormed) / psd, lowlimit); // limited Wiener filter
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
		}
	}
}
//-----------------------------------------------------------------------------------------
//
void WienerFilter::ApplyWiener3D5_degrid_C() noexcept
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
		const float gridfraction = degrid * outcur[0][0] / gridsample[0][0];
		for (int h = 0; h < bh; h++) // first half
		{
			for (int w = 0; w < outwidth; w++) // 
			{
				float gridcorrection0_5 = gridfraction * gridsample[w][0] * 5;
				float gridcorrection1_5 = gridfraction * gridsample[w][1] * 5;
				// dft 3d (very short - 5 points)
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
				WienerFactor = max((psd - sigmaSquaredNoiseNormed) / psd, lowlimit); // limited Wiener filter
				fp2r *= WienerFactor; // apply filter on real  part	
				fp2i *= WienerFactor; // apply filter on imaginary part

				psd = fpr * fpr + fpi * fpi + 1e-15f; // power spectrum density prev
				WienerFactor = max((psd - sigmaSquaredNoiseNormed) / psd, lowlimit); // limited Wiener filter
				fpr *= WienerFactor; // apply filter on real  part	
				fpi *= WienerFactor; // apply filter on imaginary part

				psd = fcr * fcr + fci * fci + 1e-15f; // power spectrum density cur
				WienerFactor = max((psd - sigmaSquaredNoiseNormed) / psd, lowlimit); // limited Wiener filter
				fcr *= WienerFactor; // apply filter on real  part	
				fci *= WienerFactor; // apply filter on imaginary part

				psd = fnr * fnr + fni * fni + 1e-15f; // power spectrum density next
				WienerFactor = max((psd - sigmaSquaredNoiseNormed) / psd, lowlimit); // limited Wiener filter
				fnr *= WienerFactor; // apply filter on real  part	
				fni *= WienerFactor; // apply filter on imaginary part

				psd = fn2r * fn2r + fn2i * fn2i + 1e-15f; // power spectrum density next
				WienerFactor = max((psd - sigmaSquaredNoiseNormed) / psd, lowlimit); // limited Wiener filter
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
		}
		gridsample -= outpitch * bh; // restore pointer to only valid first block
	}
}