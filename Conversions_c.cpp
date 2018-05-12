/*
	FFT3DFilter plugin for Avisynth 2.5 - 3D Frequency Domain filter

	Copyright(C)2004-2006 A.G.Balakhnin aka Fizick, bag@hotmail.ru, http://avisynth.org.ru
	2018 Daniel Klíma aka Klimax

	This program is free software; you can redistribute it and/or modify
	it under the terms of the GNU General Public License version 2 as published by
	the Free Software Foundation.

	This program is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with this program; if not, write to the Free Software
	Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
*/

#include "fft3dfilter.h"

//
//-----------------------------------------------------------------------------------------
// make destination frame plane from overlaped blocks
// use synthesis windows wsynxl, wsynxr, wsynyl, wsynyr

#ifndef SSE2BUILD
void FFT3DFilter::DecodeOverlapPlane_C(const float *__restrict inp0, BYTE *__restrict dstp0) noexcept
{
	int w(0);
	BYTE *__restrict dstp = dstp0;// + (hrest/2)*coverpitch + wrest/2; // centered
	const float *__restrict inp = inp0;
	const int dbwow = bw - ow;

	const int xoffset = bh * bw - dbwow;
	const int yoffset = bw * nox*bh - bw * (bh - oh); // vertical offset of same block (overlap)

	// first top big non-overlapped) part
	{
		for (int h = 0; h < bh - oh; h++)
		{
			inp = inp0 + h * bw;
			for (w = 0; w < dbwow; w++)   // first half line of first block
			{   // Copy each byte from float array to dest with windows
				dstp[w] = (BYTE)min(255, max(0, (int)(inp[w] * norm) + planeBase));
			}
			for (; w < dbwow; w++)   // first half line of first block
			{   // Copy each byte from float array to dest with windows
				dstp[w] = (BYTE)min(255, max(0, (int)(inp[w] * norm) + planeBase));
			}
			inp += dbwow;
			dstp += dbwow;
			for (int ihx = 1; ihx < nox; ihx++) // middle horizontal half-blocks
			{
				for (w = 0; w < ow; w++)   // half line of block
				{
					dstp[w] = (BYTE)min(255, max(0, ((inp[w] * wsynxr[w] + inp[w + xoffset] * wsynxl[w])*norm) + planeBase));   // overlapped Copy
				}
				for (; w < ow; w++)   // half line of block
				{
					dstp[w] = (BYTE)min(255, max(0, ((inp[w] * wsynxr[w] + inp[w + xoffset] * wsynxl[w])*norm) + planeBase));   // overlapped Copy
				}
				inp += xoffset + ow;
				dstp += ow;
				for (w = 0; w < dbwow - ow; w++)   // first half line of first block
				{
					dstp[w] = (BYTE)min(255, max(0, (inp[w] * norm) + planeBase));   // Copy each byte from float array to dest with windows
				}
				inp += dbwow - ow;
				dstp += dbwow - ow;
			}
			for (w = 0; w < ow; w++)   // last half line of last block
			{
				dstp[w] = (BYTE)min(255, max(0, (inp[w] * norm) + planeBase));
			}
			for (; w < ow; w++)   // last half line of last block
			{
				dstp[w] = (BYTE)min(255, max(0, (inp[w] * norm) + planeBase));
			}
			inp += ow;
			dstp += ow;

			dstp += (coverpitch - coverwidth);  // Add the pitch of one line (in bytes) to the dest image.
		}
	}

	for (int ihy = 1; ihy < noy; ihy += 1) // middle vertical
	{
		for (int h = 0; h < oh; h++) // top overlapped part
		{
			inp = inp0 + (ihy - 1)*(yoffset + (bh - oh)*bw) + (bh - oh)*bw + h * bw;

			const float wsynyrh = wsynyr[h] * norm; // remove from cycle for speed
			const float wsynylh = wsynyl[h] * norm;

			for (w = 0; w < dbwow; w++)   // first half line of first block
			{
				dstp[w] = (BYTE)min(255, max(0, ((inp[w] * wsynyrh + inp[w + yoffset] * wsynylh)) + planeBase));   //
			}
			for (; w < dbwow; w++)   // first half line of first block
			{
				dstp[w] = (BYTE)min(255, max(0, ((inp[w] * wsynyrh + inp[w + yoffset] * wsynylh)) + planeBase));   //
			}
			inp += dbwow;
			dstp += dbwow;
			for (int ihx = 1; ihx < nox; ihx++) // middle blocks
			{
				for (w = 0; w < ow; w++)   // half overlapped line of block
				{
					dstp[w] = (BYTE)min(255, max(0, (((inp[w] * wsynxr[w] + inp[w + xoffset] * wsynxl[w])*wsynyrh
						+ (inp[w + yoffset] * wsynxr[w] + inp[w + xoffset + yoffset] * wsynxl[w])*wsynylh)) + planeBase));   // x overlapped
				}
				inp += xoffset + ow;
				dstp += ow;
				for (w = 0; w < dbwow - ow; w++)   // double minus - half non-overlapped line of block
				{
					dstp[w] = (BYTE)min(255, max(0, ((inp[w] * wsynyrh + inp[w + yoffset] * wsynylh)) + planeBase));
				}
				inp += dbwow - ow;
				dstp += dbwow - ow;
			}
			for (w = 0; w < ow; w++)   // last half line of last block
			{
				dstp[w] = (BYTE)min(255, max(0, ((inp[w] * wsynyrh + inp[w + yoffset] * wsynylh)) + planeBase));
			}
			for (; w < ow; w++)   // last half line of last block
			{
				dstp[w] = (BYTE)min(255, max(0, ((inp[w] * wsynyrh + inp[w + yoffset] * wsynylh)) + planeBase));
			}
			inp += ow;
			dstp += ow;

			dstp += (coverpitch - coverwidth);  // Add the pitch of one line (in bytes) to the source image.
		}
		// middle  vertical non-ovelapped part
		for (int h = 0; h < (bh - oh - oh); h++)
		{
			inp = inp0 + (ihy - 1)*(yoffset + (bh - oh)*bw) + (bh)*bw + h * bw + yoffset;
			for (w = 0; w < dbwow; w++)   // first half line of first block
			{
				dstp[w] = (BYTE)min(255, max(0, ((inp[w])*norm) + planeBase));
			}
			inp += dbwow;
			dstp += dbwow;
			for (int ihx = 1; ihx < nox; ihx++) // middle blocks
			{
				for (w = 0; w < ow; w++)   // half overlapped line of block
				{
					dstp[w] = (BYTE)min(255, max(0, ((inp[w] * wsynxr[w] + inp[w + xoffset] * wsynxl[w])*norm) + planeBase));   // x overlapped
				}
				for (; w < ow; w++)   // half overlapped line of block
				{
					dstp[w] = (BYTE)min(255, max(0, ((inp[w] * wsynxr[w] + inp[w + xoffset] * wsynxl[w])*norm) + planeBase));   // x overlapped
				}
				inp += xoffset + ow;
				dstp += ow;
				for (w = 0; w < dbwow - ow; w++)   // half non-overlapped line of block
				{
					dstp[w] = (BYTE)min(255, max(0, ((inp[w])*norm) + planeBase));
				}
				inp += dbwow - ow;
				dstp += dbwow - ow;
			}
			for (w = 0; w < ow; w++)   // last half line of last block
			{
				dstp[w] = (BYTE)min(255, max(0, ((inp[w])*norm) + planeBase));
			}
			for (; w < ow; w++)   // last half line of last block
			{
				dstp[w] = (BYTE)min(255, max(0, ((inp[w])*norm) + planeBase));
			}
			inp += ow;
			dstp += ow;

			dstp += (coverpitch - coverwidth);  // Add the pitch of one line (in bytes) to the source image.
		}

	}

	const int ihy = noy; // last bottom part
	{
		for (int h = 0; h < oh; h++)
		{
			inp = inp0 + (ihy - 1)*(yoffset + (bh - oh)*bw) + (bh - oh)*bw + h * bw;
			for (w = 0; w < dbwow; w++)   // first half line of first block
			{
				dstp[w] = (BYTE)min(255, max(0, (inp[w] * norm) + planeBase));
			}
			for (; w < dbwow; w++)   // first half line of first block
			{
				dstp[w] = (BYTE)min(255, max(0, (inp[w] * norm) + planeBase));
			}
			inp += dbwow;
			dstp += dbwow;
			for (int ihx = 1; ihx < nox; ihx++) // middle blocks
			{
				for (w = 0; w < ow; w++)   // half line of block
				{
					dstp[w] = (BYTE)min(255, max(0, ((inp[w] * wsynxr[w] + inp[w + xoffset] * wsynxl[w])*norm) + planeBase));   // overlapped Copy
				}
				for (; w < ow; w++)   // half line of block
				{
					dstp[w] = (BYTE)min(255, max(0, ((inp[w] * wsynxr[w] + inp[w + xoffset] * wsynxl[w])*norm) + planeBase));   // overlapped Copy
				}
				inp += xoffset + ow;
				dstp += ow;
				for (w = 0; w < dbwow - ow; w++)   // half line of block
				{
					dstp[w] = (BYTE)min(255, max(0, ((inp[w])*norm) + planeBase));
				}
				inp += dbwow - ow;
				dstp += dbwow - ow;
			}
			for (w = 0; w < ow; w++)   // last half line of last block
			{
				dstp[w] = (BYTE)min(255, max(0, (inp[w] * norm) + planeBase));
			}
			for (; w < ow; w++)   // last half line of last block
			{
				dstp[w] = (BYTE)min(255, max(0, (inp[w] * norm) + planeBase));
			}
			inp += ow;
			dstp += ow;

			dstp += (coverpitch - coverwidth);  // Add the pitch of one line (in bytes) to the source image.
		}
	}
}
#endif