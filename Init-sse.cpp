/*
	FFT3DFilter plugin for Avisynth 2.6 - 3D Frequency Domain filter

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
#include <intrin.h>

void FFT3DFilter::InitOverlapPlane_SSE(float *__restrict inp0, const BYTE *__restrict srcp0) noexcept
{
	int w(0);
	const BYTE *__restrict srcp = srcp0;// + (hrest/2)*coverpitch + wrest/2; // centered
	const int xoffset = bh * bw - (bw - ow); // skip frames
	const int yoffset = bw * nox*bh - bw * (bh - oh); // vertical offset of same block (overlap)

	float *__restrict inp = inp0;

	// first top (big non-overlapped) part
	{
		for (int h = 0; h < oh; h++)
		{
			inp = inp0 + h * bw;
			for (w = 0; w < ow; w++)   // left part  (non-overlapped) row of first block
			{
				inp[w] = float(wanxl[w] * wanyl[h] * (srcp[w] - planeBase));   // Copy each byte from source to float array
			}
			for (w = ow; w < bw - ow; w++)   // left part  (non-overlapped) row of first block
			{
				inp[w] = float(wanyl[h] * (srcp[w] - planeBase));   // Copy each byte from source to float array
			}
			inp += bw - ow;
			srcp += bw - ow;
			for (int ihx = 1; ihx < nox; ihx += 1) // middle horizontal blocks
			{
				for (w = 0; w < ow; w++)   // first part (overlapped) row of block
				{
					const float ftmp = float(wanyl[h] * (srcp[w] - planeBase));   // Copy each byte from source to float array
					inp[w] = ftmp * wanxr[w]; // cur block
					inp[w + xoffset] = ftmp * wanxl[w];   // overlapped Copy - next block
				}
				inp += ow;
				inp += xoffset;
				srcp += ow;
				for (w = 0; w < bw - ow -ow; w++)   // center part  (non-overlapped) row of first block
				{
					inp[w] = float(wanyl[h] * (srcp[w] - planeBase));   // Copy each byte from source to float array
				}
				inp += bw - ow -ow;
				srcp += bw - ow -ow;
			}
			for (w = 0; w < ow; w++)   // last part (non-overlapped) of line of last block
			{
				inp[w] = float(wanxr[w] * wanyl[h] * (srcp[w] - planeBase));   // Copy each byte from source to float array
			}
			inp += ow;
			srcp += ow;
			srcp += (coverpitch - coverwidth);  // Add the pitch of one line (in bytes) to the source image.
		}
		for (int h = oh; h < bh - oh; h++)
		{
			inp = inp0 + h * bw;
			for (w = 0; w < ow; w++)   // left part  (non-overlapped) row of first block
			{
				inp[w] = float(wanxl[w] * (srcp[w] - planeBase));   // Copy each byte from source to float array
			}
			for (w = ow; w < bw - ow; w++)   // left part  (non-overlapped) row of first block
			{
				inp[w] = float((srcp[w] - planeBase));   // Copy each byte from source to float array
			}
			inp += bw - ow;
			srcp += bw - ow;
			for (int ihx = 1; ihx < nox; ihx += 1) // middle horizontal blocks
			{
				for (w = 0; w < ow; w++)   // first part (overlapped) row of block
				{
					const float ftmp = float((srcp[w] - planeBase));   // Copy each byte from source to float array
					inp[w] = ftmp * wanxr[w]; // cur block
					inp[w + xoffset] = ftmp * wanxl[w];   // overlapped Copy - next block
				}
				inp += ow;
				inp += xoffset;
				srcp += ow;
				for (w = 0; w < bw - ow -ow; w++)   // center part  (non-overlapped) row of first block
				{
					inp[w] = float((srcp[w] - planeBase));   // Copy each byte from source to float array
				}
				inp += bw - ow -ow;
				srcp += bw - ow -ow;
			}
			for (w = 0; w < ow; w++)   // last part (non-overlapped) line of last block
			{
				inp[w] = float(wanxr[w] * (srcp[w] - planeBase));   // Copy each byte from source to float array
			}
			inp += ow;
			srcp += ow;

			srcp += (coverpitch - coverwidth);  // Add the pitch of one line (in bytes) to the source image.
		}
	}

	for (int ihy = 1; ihy < noy; ihy += 1) // middle vertical
	{
		for (int h = 0; h < oh; h++) // top overlapped part
		{
			inp = inp0 + (ihy - 1)*(yoffset + (bh - oh)*bw) + (bh - oh)*bw + h * bw;
			for (w = 0; w < ow; w++)   // first half line of first block
			{
				const float ftmp = float(wanxl[w] * (srcp[w] - planeBase));
				inp[w] = ftmp * wanyr[h];   // Copy each byte from source to float array
				inp[w + yoffset] = ftmp * wanyl[h];   // y overlapped
			}
			for (w = ow; w < bw - ow; w++)   // first half line of first block
			{
				const float ftmp = float((srcp[w] - planeBase));
				inp[w] = ftmp * wanyr[h];   // Copy each byte from source to float array
				inp[w + yoffset] = ftmp * wanyl[h];   // y overlapped
			}
			inp += bw - ow;
			srcp += bw - ow;
			for (int ihx = 1; ihx < nox; ihx++) // middle blocks
			{
				for (w = 0; w < ow; w++)   // half overlapped line of block
				{
					const float ftmp = float((srcp[w] - planeBase));   // Copy each byte from source to float array
					inp[w] = ftmp * wanxr[w] * wanyr[h];
					inp[w + xoffset] = ftmp * wanxl[w] * wanyr[h];   // x overlapped
					inp[w + yoffset] = ftmp * wanxr[w] * wanyl[h];
					inp[w + xoffset + yoffset] = ftmp * wanxl[w] * wanyl[h];   // x overlapped
				}
				inp += ow;
				inp += xoffset;
				srcp += ow;
				for (w = 0; w < bw - ow -ow; w++)   // half non-overlapped line of block
				{
					const float ftmp = float((srcp[w] - planeBase));   // Copy each byte from source to float array
					inp[w] = ftmp * wanyr[h];
					inp[w + yoffset] = ftmp * wanyl[h];
				}
				inp += bw - ow -ow;
				srcp += bw - ow -ow;
			}
			for (w = 0; w < ow; w++)   // last half line of last block
			{
				const float ftmp = float(wanxr[w] * (srcp[w] - planeBase));// Copy each byte from source to float array
				inp[w] = ftmp * wanyr[h];
				inp[w + yoffset] = ftmp * wanyl[h];
			}
			inp += ow;
			srcp += ow;

			srcp += (coverpitch - coverwidth);  // Add the pitch of one line (in bytes) to the source image.
		}
		// middle  vertical nonovelapped part
		for (int h = 0; h < bh - oh - oh; h++)
		{
			inp = inp0 + (ihy - 1)*(yoffset + (bh - oh)*bw) + (bh)*bw + h * bw + yoffset;
			for (w = 0; w < ow; w++)   // first half line of first block
			{
				const float ftmp = float(wanxl[w] * (srcp[w] - planeBase));
				inp[w] = ftmp;   // Copy each byte from source to float array
			}
			for (w = ow; w < bw - ow; w++)   // first half line of first block
			{
				const float ftmp = float((srcp[w] - planeBase));
				inp[w] = ftmp;   // Copy each byte from source to float array
			}
			inp += bw - ow;
			srcp += bw - ow;
			for (int ihx = 1; ihx < nox; ihx++) // middle blocks
			{
				for (w = 0; w < ow; w++)   // half overlapped line of block
				{
					const float ftmp = float((srcp[w] - planeBase));   // Copy each byte from source to float array
					inp[w] = ftmp * wanxr[w];
					inp[w + xoffset] = ftmp * wanxl[w];   // x overlapped
				}
				inp += ow;
				inp += xoffset;
				srcp += ow;
				for (w = 0; w < bw - ow -ow; w++)   // half non-overlapped line of block
				{
					const float ftmp = float((srcp[w] - planeBase));   // Copy each byte from source to float array
					inp[w] = ftmp;
				}
				inp += bw - ow -ow;
				srcp += bw - ow -ow;
			}
			for (w = 0; w < ow; w++)   // last half line of last block
			{
				const float ftmp = float(wanxr[w] * (srcp[w] - planeBase));// Copy each byte from source to float array
				inp[w] = ftmp;
			}
			inp += ow;
			srcp += ow;

			srcp += (coverpitch - coverwidth);  // Add the pitch of one line (in bytes) to the source image.
		}

	}

	int ihy = noy; // last bottom  part
	{
		for (int h = 0; h < oh; h++)
		{
			inp = inp0 + (ihy - 1)*(yoffset + (bh - oh)*bw) + (bh - oh)*bw + h * bw;
			for (w = 0; w < ow; w++)   // first half line of first block
			{
				const float ftmp = float(wanxl[w] * wanyr[h] * (srcp[w] - planeBase));
				inp[w] = ftmp;   // Copy each byte from source to float array
			}
			for (w = ow; w < bw - ow; w++)   // first half line of first block
			{
				const float ftmp = float(wanyr[h] * (srcp[w] - planeBase));
				inp[w] = ftmp;   // Copy each byte from source to float array
			}
			inp += bw - ow;
			srcp += bw - ow;
			for (int ihx = 1; ihx < nox; ihx++) // middle blocks
			{
				for (w = 0; w < ow; w++)   // half line of block
				{
					const float ftmp = float(wanyr[h] * (srcp[w] - planeBase));   // Copy each byte from source to float array
					inp[w] = ftmp * wanxr[w];
					inp[w + xoffset] = ftmp * wanxl[w];   // overlapped Copy
				}
				inp += ow;
				inp += xoffset;
				srcp += ow;
				for (w = 0; w < bw - ow -ow; w++)   // center part  (non-overlapped) row of first block
				{
					inp[w] = float(wanyr[h] * (srcp[w] - planeBase));   // Copy each byte from source to float array
				}
				inp += bw - ow -ow;
				srcp += bw - ow -ow;
			}
			for (w = 0; w < ow; w++)   // last half line of last block
			{
				const float ftmp = float(wanxr[w] * wanyr[h] * (srcp[w] - planeBase));
				inp[w] = ftmp;   // Copy each byte from source to float array
			}
			inp += ow;
			srcp += ow;

			srcp += (coverpitch - coverwidth);  // Add the pitch of one line (in bytes) to the source image.
		}
	}
}

void FFT3DFilter::InitOverlapPlane_wt2_SSE(float *__restrict inp0, const BYTE *__restrict srcp0) noexcept
{
	int w(0);
	const BYTE *__restrict srcp = srcp0;// + (hrest/2)*coverpitch + wrest/2; // centered
	const int xoffset = bh * bw - (bw - ow); // skip frames
	const int yoffset = bw * nox*bh - bw * (bh - oh); // vertical offset of same block (overlap)

	const int dbwow = bw - ow;
	const int dbwow4 = dbwow - dbwow % 4;
	const int d2bwow = dbwow - ow;
	const int d2bwow4 = d2bwow - d2bwow % 4;
	const int ow4 = ow - ow % 4;

	float *__restrict inp = inp0;

	// first top (big non-overlapped) part
	{
		for (int h = 0; h < oh; h++)
		{
			inp = inp0 + h * bw;
			for (w = 0; w < ow4; w = w + 4)   // left part  (non-overlapped) row of first block
			{
				inp[w] = float((srcp[w] - planeBase));   // Copy each byte from source to float array
				inp[w + 1] = float((srcp[w + 1] - planeBase));   // Copy each byte from source to float array
				inp[w + 2] = float((srcp[w + 2] - planeBase));   // Copy each byte from source to float array
				inp[w + 3] = float((srcp[w + 3] - planeBase));   // Copy each byte from source to float array
			}
			for (; w < ow; w++)   // left part  (non-overlapped) row of first block
			{
				inp[w] = float((srcp[w] - planeBase));   // Copy each byte from source to float array
			}
			for (w = ow; w < dbwow4; w = w + 4)   // left part  (non-overlapped) row of first block
			{
				inp[w] = float((srcp[w] - planeBase));   // Copy each byte from source to float array
				inp[w + 1] = float((srcp[w + 1] - planeBase));   // Copy each byte from source to float array
				inp[w + 2] = float((srcp[w + 2] - planeBase));   // Copy each byte from source to float array
				inp[w + 3] = float((srcp[w + 3] - planeBase));   // Copy each byte from source to float array

			}
			for (; w < dbwow; w++)   // left part  (non-overlapped) row of first block
			{
				inp[w] = float((srcp[w] - planeBase));   // Copy each byte from source to float array
			}
			inp += bw - ow;
			srcp += bw - ow;
			for (int ihx = 1; ihx < nox; ihx += 1) // middle horizontal blocks
			{
				for (w = 0; w < ow; w++)   // first part (overlapped) row of block
				{
					const float ftmp = float((srcp[w] - planeBase));   // Copy each byte from source to float array
					inp[w] = ftmp; // cur block
					inp[w + xoffset] = ftmp;   // overlapped Copy - next block
				}
				inp += ow;
				inp += xoffset;
				srcp += ow;
				for (w = 0; w < d2bwow; w++)   // center part  (non-overlapped) row of first block
				{
					inp[w] = float((srcp[w] - planeBase));   // Copy each byte from source to float array
				}
				inp += d2bwow;
				srcp += d2bwow;
			}
			for (w = 0; w < ow4; w = w + 4)   // left part  (non-overlapped) row of first block
			{
				inp[w] = float((srcp[w] - planeBase));   // Copy each byte from source to float array
				inp[w + 1] = float((srcp[w + 1] - planeBase));   // Copy each byte from source to float array
				inp[w + 2] = float((srcp[w + 2] - planeBase));   // Copy each byte from source to float array
				inp[w + 3] = float((srcp[w + 3] - planeBase));   // Copy each byte from source to float array
			}
			for (; w < ow; w++)   // left part  (non-overlapped) row of first block
			{
				inp[w] = float((srcp[w] - planeBase));   // Copy each byte from source to float array
			}
			inp += ow;
			srcp += ow;
			srcp += (coverpitch - coverwidth);  // Add the pitch of one line (in bytes) to the source image.
		}
		for (int h = oh; h < bh - oh; h++)
		{
			inp = inp0 + h * bw;
			for (w = 0; w < ow4; w = w + 4)   // left part  (non-overlapped) row of first block
			{
				inp[w] = float((srcp[w] - planeBase));   // Copy each byte from source to float array
				inp[w + 1] = float((srcp[w + 1] - planeBase));   // Copy each byte from source to float array
				inp[w + 2] = float((srcp[w + 2] - planeBase));   // Copy each byte from source to float array
				inp[w + 3] = float((srcp[w + 3] - planeBase));   // Copy each byte from source to float array
			}
			for (; w < ow; w++)   // left part  (non-overlapped) row of first block
			{
				inp[w] = float((srcp[w] - planeBase));   // Copy each byte from source to float array
			}
			for (w = ow; w < dbwow4; w = w + 4)   // left part  (non-overlapped) row of first block
			{
				inp[w] = float((srcp[w] - planeBase));   // Copy each byte from source to float array
				inp[w + 1] = float((srcp[w + 1] - planeBase));   // Copy each byte from source to float array
				inp[w + 2] = float((srcp[w + 2] - planeBase));   // Copy each byte from source to float array
				inp[w + 3] = float((srcp[w + 3] - planeBase));   // Copy each byte from source to float array

			}
			for (; w < dbwow; w++)   // left part  (non-overlapped) row of first block
			{
				inp[w] = float((srcp[w] - planeBase));   // Copy each byte from source to float array
			}
			inp += bw - ow;
			srcp += bw - ow;
			for (int ihx = 1; ihx < nox; ihx += 1) // middle horizontal blocks
			{
				for (w = 0; w < ow; w++)   // first part (overlapped) row of block
				{
					const float ftmp = float((srcp[w] - planeBase));   // Copy each byte from source to float array
					inp[w] = ftmp; // cur block
					inp[w + xoffset] = ftmp;   // overlapped Copy - next block
				}
				inp += ow;
				inp += xoffset;
				srcp += ow;
				for (w = 0; w < d2bwow; w++)   // center part  (non-overlapped) row of first block
				{
					inp[w] = float((srcp[w] - planeBase));   // Copy each byte from source to float array
				}
				inp += d2bwow;
				srcp += d2bwow;
			}
			for (w = 0; w < ow4; w = w + 4)   // left part  (non-overlapped) row of first block
			{
				inp[w] = float((srcp[w] - planeBase));   // Copy each byte from source to float array
				inp[w + 1] = float((srcp[w + 1] - planeBase));   // Copy each byte from source to float array
				inp[w + 2] = float((srcp[w + 2] - planeBase));   // Copy each byte from source to float array
				inp[w + 3] = float((srcp[w + 3] - planeBase));   // Copy each byte from source to float array
			}
			for (; w < ow; w++)   // left part  (non-overlapped) row of first block
			{
				inp[w] = float((srcp[w] - planeBase));   // Copy each byte from source to float array
			}
			inp += ow;
			srcp += ow;

			srcp += (coverpitch - coverwidth);  // Add the pitch of one line (in bytes) to the source image.
		}
	}

	for (int ihy = 1; ihy < noy; ihy += 1) // middle vertical
	{
		for (int h = 0; h < oh; h++) // top overlapped part
		{
			inp = inp0 + (ihy - 1)*(yoffset + (bh - oh)*bw) + (bh - oh)*bw + h * bw;
			for (w = 0; w < ow; w++)   // first half line of first block
			{
				const float ftmp = float((srcp[w] - planeBase));
				inp[w] = ftmp;   // Copy each byte from source to float array
				inp[w + yoffset] = ftmp;   // y overlapped
			}
			for (w = ow; w < dbwow4; w = w + 4)   // left part  (non-overlapped) row of first block
			{
				inp[w] = float((srcp[w] - planeBase));   // Copy each byte from source to float array
				inp[w + 1] = float((srcp[w + 1] - planeBase));   // Copy each byte from source to float array
				inp[w + 2] = float((srcp[w + 2] - planeBase));   // Copy each byte from source to float array
				inp[w + 3] = float((srcp[w + 3] - planeBase));   // Copy each byte from source to float array

			}
			for (; w < dbwow; w++)   // left part  (non-overlapped) row of first block
			{
				inp[w] = float((srcp[w] - planeBase));   // Copy each byte from source to float array
			}
			inp += bw - ow;
			srcp += bw - ow;
			for (int ihx = 1; ihx < nox; ihx++) // middle blocks
			{
				for (w = 0; w < ow; w++)   // half overlapped line of block
				{
					const float ftmp = float((srcp[w] - planeBase));   // Copy each byte from source to float array
					inp[w] = ftmp;
					inp[w + xoffset] = ftmp;   // x overlapped
					inp[w + yoffset] = ftmp;
					inp[w + xoffset + yoffset] = ftmp;   // x overlapped
				}
				inp += ow;
				inp += xoffset;
				srcp += ow;
				for (w = 0; w < d2bwow; w++)   // half non-overlapped line of block
				{
					const float ftmp = float((srcp[w] - planeBase));   // Copy each byte from source to float array
					inp[w] = ftmp;
					inp[w + yoffset] = ftmp;
				}
				inp += d2bwow;
				srcp += d2bwow;
			}
			for (w = 0; w < ow; w++)   // last half line of last block
			{
				const float ftmp = float((srcp[w] - planeBase));// Copy each byte from source to float array
				inp[w] = ftmp;
				inp[w + yoffset] = ftmp;
			}
			inp += ow;
			srcp += ow;

			srcp += (coverpitch - coverwidth);  // Add the pitch of one line (in bytes) to the source image.
		}
		// middle  vertical nonovelapped part
		for (int h = 0; h < bh - oh - oh; h++)
		{
			inp = inp0 + (ihy - 1)*(yoffset + (bh - oh)*bw) + (bh)*bw + h * bw + yoffset;
			for (w = 0; w < ow4; w = w + 4)   // left part  (non-overlapped) row of first block
			{
				inp[w] = float((srcp[w] - planeBase));   // Copy each byte from source to float array
				inp[w + 1] = float((srcp[w + 1] - planeBase));   // Copy each byte from source to float array
				inp[w + 2] = float((srcp[w + 2] - planeBase));   // Copy each byte from source to float array
				inp[w + 3] = float((srcp[w + 3] - planeBase));   // Copy each byte from source to float array
			}
			for (; w < ow; w++)   // left part  (non-overlapped) row of first block
			{
				inp[w] = float((srcp[w] - planeBase));   // Copy each byte from source to float array
			}
			for (w = ow; w < dbwow4; w = w + 4)   // left part  (non-overlapped) row of first block
			{
				inp[w] = float((srcp[w] - planeBase));   // Copy each byte from source to float array
				inp[w + 1] = float((srcp[w + 1] - planeBase));   // Copy each byte from source to float array
				inp[w + 2] = float((srcp[w + 2] - planeBase));   // Copy each byte from source to float array
				inp[w + 3] = float((srcp[w + 3] - planeBase));   // Copy each byte from source to float array

			}
			for (; w < dbwow; w++)   // left part  (non-overlapped) row of first block
			{
				inp[w] = float((srcp[w] - planeBase));   // Copy each byte from source to float array
			}
			inp += bw - ow;
			srcp += bw - ow;
			for (int ihx = 1; ihx < nox; ihx++) // middle blocks
			{
				for (w = 0; w < ow; w++)   // half overlapped line of block
				{
					const float ftmp = float((srcp[w] - planeBase));   // Copy each byte from source to float array
					inp[w] = ftmp;
					inp[w + xoffset] = ftmp;   // x overlapped
				}
				inp += ow;
				inp += xoffset;
				srcp += ow;
				for (w = 0; w < d2bwow; w++)   // half non-overlapped line of block
				{
					const float ftmp = float((srcp[w] - planeBase));   // Copy each byte from source to float array
					inp[w] = ftmp;
				}
				inp += d2bwow;
				srcp += d2bwow;
			}
			for (w = 0; w < ow4; w = w + 4)   // left part  (non-overlapped) row of first block
			{
				inp[w] = float((srcp[w] - planeBase));   // Copy each byte from source to float array
				inp[w + 1] = float((srcp[w + 1] - planeBase));   // Copy each byte from source to float array
				inp[w + 2] = float((srcp[w + 2] - planeBase));   // Copy each byte from source to float array
				inp[w + 3] = float((srcp[w + 3] - planeBase));   // Copy each byte from source to float array
			}
			for (; w < ow; w++)   // left part  (non-overlapped) row of first block
			{
				inp[w] = float((srcp[w] - planeBase));   // Copy each byte from source to float array
			}
			inp += ow;
			srcp += ow;

			srcp += (coverpitch - coverwidth);  // Add the pitch of one line (in bytes) to the source image.
		}

	}

	int ihy = noy; // last bottom  part
	{
		for (int h = 0; h < oh; h++)
		{
			inp = inp0 + (ihy - 1)*(yoffset + (bh - oh)*bw) + (bh - oh)*bw + h * bw;
			for (w = 0; w < ow4; w = w + 4)   // left part  (non-overlapped) row of first block
			{
				inp[w] = float((srcp[w] - planeBase));   // Copy each byte from source to float array
				inp[w + 1] = float((srcp[w + 1] - planeBase));   // Copy each byte from source to float array
				inp[w + 2] = float((srcp[w + 2] - planeBase));   // Copy each byte from source to float array
				inp[w + 3] = float((srcp[w + 3] - planeBase));   // Copy each byte from source to float array
			}
			for (; w < ow; w++)   // left part  (non-overlapped) row of first block
			{
				inp[w] = float((srcp[w] - planeBase));   // Copy each byte from source to float array
			}
			for (w = ow; w < dbwow4; w = w + 4)   // left part  (non-overlapped) row of first block
			{
				inp[w] = float((srcp[w] - planeBase));   // Copy each byte from source to float array
				inp[w + 1] = float((srcp[w + 1] - planeBase));   // Copy each byte from source to float array
				inp[w + 2] = float((srcp[w + 2] - planeBase));   // Copy each byte from source to float array
				inp[w + 3] = float((srcp[w + 3] - planeBase));   // Copy each byte from source to float array

			}
			for (; w < dbwow; w++)   // left part  (non-overlapped) row of first block
			{
				inp[w] = float((srcp[w] - planeBase));   // Copy each byte from source to float array
			}
			inp += bw - ow;
			srcp += bw - ow;
			for (int ihx = 1; ihx < nox; ihx++) // middle blocks
			{
				for (w = 0; w < ow; w++)   // half line of block
				{
					const float ftmp = float((srcp[w] - planeBase));   // Copy each byte from source to float array
					inp[w] = ftmp;
					inp[w + xoffset] = ftmp;   // overlapped Copy
				}
				inp += ow;
				inp += xoffset;
				srcp += ow;
				for (w = 0; w < d2bwow; w++)   // center part  (non-overlapped) row of first block
				{
					inp[w] = float((srcp[w] - planeBase));   // Copy each byte from source to float array
				}
				inp += d2bwow;
				srcp += d2bwow;
			}
			for (w = 0; w < ow4; w = w + 4)   // left part  (non-overlapped) row of first block
			{
				inp[w] = float((srcp[w] - planeBase));   // Copy each byte from source to float array
				inp[w + 1] = float((srcp[w + 1] - planeBase));   // Copy each byte from source to float array
				inp[w + 2] = float((srcp[w + 2] - planeBase));   // Copy each byte from source to float array
				inp[w + 3] = float((srcp[w + 3] - planeBase));   // Copy each byte from source to float array
			}
			for (; w < ow; w++)   // left part  (non-overlapped) row of first block
			{
				inp[w] = float((srcp[w] - planeBase));   // Copy each byte from source to float array
			}
			inp += ow;
			srcp += ow;

			srcp += (coverpitch - coverwidth);  // Add the pitch of one line (in bytes) to the source image.
		}
	}
}
