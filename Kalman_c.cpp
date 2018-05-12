//
//	FFT3DFilter plugin for Avisynth 2.5 - 3D Frequency Domain filter
//  pure C++ filtering functions
//
//  Copyright(C)2004-2006 A.G.Balakhnin aka Fizick, bag@hotmail.ru, http://avisynth.org.ru
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
#include "Kalman.h"

// since v1.7 we use outpitch instead of outwidth

//
//-----------------------------------------------------------------------------------------
//
void KalmanFilter::ApplyKalmanPattern_C() noexcept
{
	// return result in outLast
	int w(0);
	fftwf_complex *__restrict covar = covar_in, *__restrict covarProcess = covarProcess_in;
	for (int block = start_block; block < blocks; block++)
	{
		for (int h = 0; h < bh; h++) // 
		{
			for (w = 0; w < outwidth; w++)
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
//-----------------------------------------------------------------------------------------
//
void KalmanFilter::ApplyKalman_C() noexcept
{
	// return result in outLast
	int w(0);
	fftwf_complex *__restrict covar = covar_in, *__restrict covarProcess = covarProcess_in;
	float sigmaSquaredMotionNormed = covarNoiseNormed * kratio2;

	for (int block = start_block; block < blocks; block++)
	{
		for (int h = 0; h < bh; h++) // 
		{
			for (w = 0; w < outwidth; w++)
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