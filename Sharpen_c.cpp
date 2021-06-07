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
#include <math.h>
#include "Sharpen.h"

// since v1.7 we use outpitch instead of outwidth

//-------------------------------------------------------------------------------------------
//
void SharpenFilter::Sharpen_C() noexcept
{
	float psd(0.0f), sfact(0.0f);

	if (sharpen != 0 && dehalo == 0)
	{
		for (int block = start_block; block < blocks; block++)
		{
			for (int h = 0; h < bh; h++) // middle
			{
				for (int w = 0; w < outwidth; w++)
				{
					psd = (outcur[w][0] * outcur[w][0] + outcur[w][1] * outcur[w][1]);
					//improved sharpen mode to prevent grid artifactes and to limit sharpening both fo low and high amplitudes
					sfact = (1 + sharpen * wsharpen[w] * sqrtf(psd*sigmaSquaredSharpenMax / ((psd + sigmaSquaredSharpenMin)*(psd + sigmaSquaredSharpenMax))));
					// sharpen factor - changed in v1.1c
					outcur[w][0] *= sfact;
					outcur[w][1] *= sfact;
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
				for (int w = 0; w < outwidth; w++)
				{
					psd = (outcur[w][0] * outcur[w][0] + outcur[w][1] * outcur[w][1]);
					//improved sharpen mode to prevent grid artifactes and to limit sharpening both fo low and high amplitudes
					sfact = (psd + ht2n) / ((psd + ht2n) + dehalo * wdehalo[w] * psd);
					outcur[w][0] *= sfact;
					outcur[w][1] *= sfact;
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
				for (int w = 0; w < outwidth; w++)
				{
					psd = (outcur[w][0] * outcur[w][0] + outcur[w][1] * outcur[w][1]);
					//improved sharpen mode to prevent grid artifactes and to limit sharpening both fo low and high amplitudes
					sfact = (1 + sharpen * wsharpen[w] * sqrtf(psd*sigmaSquaredSharpenMax / ((psd + sigmaSquaredSharpenMin)*(psd + sigmaSquaredSharpenMax)))) *
						(psd + ht2n) / ((psd + ht2n) + dehalo * wdehalo[w] * psd);
					outcur[w][0] *= sfact;
					outcur[w][1] *= sfact;
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
//-----------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------
// DEGRID
//-----------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
//
void SharpenFilter::Sharpen_degrid_C() noexcept
{
	float psd(0.0f), sfact(0.0f);

	if (sharpen != 0 && dehalo == 0)
	{
		for (int block = start_block; block < blocks; block++)
		{
			const float gridfraction = degrid * outcur[0][0] / gridsample[0][0];
			for (int h = 0; h < bh; h++) // middle
			{
				for (int w = 0; w < outwidth; w++)
				{
					float gridcorrection0 = gridfraction * gridsample[w][0];
					float re = outcur[w][0] - gridcorrection0;
					float gridcorrection1 = gridfraction * gridsample[w][1];
					float im = outcur[w][1] - gridcorrection1;
					psd = (re*re + im * im) + 1e-15f;// power spectrum density
//					psd = (outcur[w][0]*outcur[w][0] + outcur[w][1]*outcur[w][1]);
					//improved sharpen mode to prevent grid artifactes and to limit sharpening both fo low and high amplitudes
					sfact = (1 + sharpen * wsharpen[w] * sqrtf(psd*sigmaSquaredSharpenMax / ((psd + sigmaSquaredSharpenMin)*(psd + sigmaSquaredSharpenMax))));
					// sharpen factor - changed in v1.1c
					re *= sfact; // apply filter on real  part	
					im *= sfact; // apply filter on imaginary part
					outcur[w][0] = re + gridcorrection0;
					outcur[w][1] = im + gridcorrection1;
				}
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
		for (int block = start_block; block < blocks; block++)
		{
			const float gridfraction = degrid * outcur[0][0] / gridsample[0][0];
			for (int h = 0; h < bh; h++) // middle
			{
				for (int w = 0; w < outwidth; w++)
				{
					float gridcorrection0 = gridfraction * gridsample[w][0];
					float re = outcur[w][0] - gridcorrection0;
					float gridcorrection1 = gridfraction * gridsample[w][1];
					float im = outcur[w][1] - gridcorrection1;
					psd = (re*re + im * im) + 1e-15f;// power spectrum density
//					psd = (outcur[w][0]*outcur[w][0] + outcur[w][1]*outcur[w][1]);
					//improved sharpen mode to prevent grid artifactes and to limit sharpening both fo low and high amplitudes
					sfact = (psd + ht2n) / ((psd + ht2n) + dehalo * wdehalo[w] * psd);
					re *= sfact; // apply filter on real  part	
					im *= sfact; // apply filter on imaginary part
					outcur[w][0] = re + gridcorrection0;
					outcur[w][1] = im + gridcorrection1;
				}
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
		for (int block = start_block; block < blocks; block++)
		{
			const float gridfraction = degrid * outcur[0][0] / gridsample[0][0];
			for (int h = 0; h < bh; h++) // middle
			{
				for (int w = 0; w < outwidth; w++)
				{
					float gridcorrection0 = gridfraction * gridsample[w][0];
					float re = outcur[w][0] - gridcorrection0;
					float gridcorrection1 = gridfraction * gridsample[w][1];
					float im = outcur[w][1] - gridcorrection1;
					psd = (re*re + im * im) + 1e-15f;// power spectrum density
//					psd = (outcur[w][0]*outcur[w][0] + outcur[w][1]*outcur[w][1]);
					//improved sharpen mode to prevent grid artifactes and to limit sharpening both fo low and high amplitudes
					sfact = (1 + sharpen * wsharpen[w] * sqrtf(psd*sigmaSquaredSharpenMax / ((psd + sigmaSquaredSharpenMin)*(psd + sigmaSquaredSharpenMax)))) *
						(psd + ht2n) / ((psd + ht2n) + dehalo * wdehalo[w] * psd);
					re *= sfact; // apply filter on real  part	
					im *= sfact; // apply filter on imaginary part
					outcur[w][0] = re + gridcorrection0;
					outcur[w][1] = im + gridcorrection1;
				}
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