/*
FFT3DFilter plugin for Avisynth 2.6 - 3D Frequency Domain filter

Derived from C version of function. (Copyright(C)2004-2006 A.G.Balakhnin aka Fizick, bag@hotmail.ru, http://avisynth.org.ru)
Copyright(C) 2018 Daniel Klíma aka Klimax

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

//
//-----------------------------------------------------------------------------------------
// make destination frame plane from overlaped blocks
// use synthesis windows wsynxl, wsynxr, wsynyl, wsynyr
void FFT3DFilter::DecodeOverlapPlane_AVX(const float *__restrict inp0, BYTE *__restrict dstp0) noexcept
{
	int w(0);
	BYTE *__restrict dstp = dstp0;// + (hrest/2)*coverpitch + wrest/2; // centered
	const float *__restrict inp = inp0;
	const int dbwow = bw - ow;
	const int dbwow4 = dbwow - dbwow % 4;
	const int d2bwow = dbwow - ow;
	const int d2bwow4 = d2bwow - d2bwow % 4;
	const int ow4 = ow - ow % 4;
	const __m128 norm4 = _mm_load1_ps(&norm);
	const __m128i planebase4 = _mm_set1_epi32(planeBase);

	const int xoffset = bh * bw - dbwow;
	const int yoffset = bw * nox*bh - bw * (bh - oh); // vertical offset of same block (overlap)

  // first top big non-overlapped) part
	{
		for (int h = 0; h < bh - oh; h++)
		{
			inp = inp0 + h * bw;
			for (w = 0; w < dbwow4; w = w + 4)   // first half line of first block
			{   // Copy each byte from float array to dest with windows
				const __m128 r1 = _mm_mul_ps(_mm_load_ps(&inp[w]), norm4);
				__m128i r1i = _mm_add_epi32(_mm_cvtps_epi32(r1), planebase4);
				r1i = _mm_packus_epi32(r1i, _mm_setzero_si128());
				r1i = _mm_packus_epi16(r1i, _mm_setzero_si128());
				const int out_t = _mm_cvtsi128_si32(r1i);
				memcpy(&dstp[w], &out_t, 4);
			}
			for (; w < dbwow; w++)   // first half line of first block
			{   // Copy each byte from float array to dest with windows
				dstp[w] = (BYTE)min(255, max(0, (int)(inp[w] * norm) + planeBase));
			}
			inp += dbwow;
			dstp += dbwow;
			for (int ihx = 1; ihx < nox; ihx++) // middle horizontal half-blocks
			{
				for (w = 0; w < ow4; w = w + 4)   // half line of block
				{
					__m128 r1 = _mm_mul_ps(_mm_load_ps(&inp[w]), _mm_load_ps(&wsynxr[w]));
					const __m128 r3 = _mm_mul_ps(_mm_load_ps(&inp[w + xoffset]), _mm_load_ps(&wsynxl[w]));
					r1 = _mm_add_ps(r1, r3);
					r1 = _mm_mul_ps(r1, norm4);
					__m128i r1i = _mm_add_epi32(_mm_cvtps_epi32(r1), planebase4);
					r1i = _mm_packus_epi32(r1i, _mm_setzero_si128());
					r1i = _mm_packus_epi16(r1i, _mm_setzero_si128());
					const int out_t = _mm_cvtsi128_si32(r1i);
					memcpy(&dstp[w], &out_t, 4);
				}
				for (; w < ow; w++)   // half line of block
				{
					dstp[w] = (BYTE)min(255, max(0, ((inp[w] * wsynxr[w] + inp[w + xoffset] * wsynxl[w])*norm) + planeBase));   // overlapped Copy
				}
				inp += xoffset + ow;
				dstp += ow;

				for (w = 0; w < d2bwow4; w = w + 4)   // last half line of last block
				{
					const __m128 r1 = _mm_mul_ps(_mm_load_ps(&inp[w]), norm4);
					__m128i r1i = _mm_add_epi32(_mm_cvtps_epi32(r1), planebase4);
					r1i = _mm_packus_epi32(r1i, _mm_setzero_si128());
					r1i = _mm_packus_epi16(r1i, _mm_setzero_si128());
					const int out_t = _mm_cvtsi128_si32(r1i);
					memcpy(&dstp[w], &out_t, 4);
				}
				for (; w < d2bwow; w++)   // first half line of first block
				{
					dstp[w] = (BYTE)min(255, max(0, (inp[w] * norm) + planeBase));   // Copy each byte from float array to dest with windows
				}
				inp += d2bwow;
				dstp += d2bwow;
			}
			for (w = 0; w < ow4; w = w + 4)   // last half line of last block
			{
				const __m128 r1 = _mm_mul_ps(_mm_load_ps(&inp[w]), norm4);
				__m128i r1i = _mm_add_epi32(_mm_cvtps_epi32(r1), planebase4);
				r1i = _mm_packus_epi32(r1i, _mm_setzero_si128());
				r1i = _mm_packus_epi16(r1i, _mm_setzero_si128());
				const int out_t = _mm_cvtsi128_si32(r1i);
				memcpy(&dstp[w], &out_t, 4);
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

			const __m128 wsynyrh4 = _mm_load1_ps(&wsynyrh), wsynylh4 = _mm_load1_ps(&wsynylh);

			for (w = 0; w < dbwow4; w = w + 4)   // first half line of first block
			{
				__m128 r1 = _mm_mul_ps(_mm_load_ps(&inp[w]), wsynyrh4);
				const __m128 r3 = _mm_mul_ps(_mm_load_ps(&inp[w + yoffset]), wsynylh4);
				r1 = _mm_add_ps(r1, r3);
				__m128i r1i = _mm_add_epi32(_mm_cvtps_epi32(r1), planebase4);
				r1i = _mm_packus_epi32(r1i, _mm_setzero_si128());
				r1i = _mm_packus_epi16(r1i, _mm_setzero_si128());
				const int out_t = _mm_cvtsi128_si32(r1i);
				memcpy(&dstp[w], &out_t, 4);
			}
			for (; w < dbwow; w++)   // first half line of first block
			{
				dstp[w] = (BYTE)min(255, max(0, ((inp[w] * wsynyrh + inp[w + yoffset] * wsynylh)) + planeBase));   //
			}
			inp += dbwow;
			dstp += dbwow;
			for (int ihx = 1; ihx < nox; ihx++) // middle blocks
			{
				for (w = 0; w < ow4; w = w + 4)   // half overlapped line of block
				{
					__m128 r1 = _mm_mul_ps(_mm_load_ps(&inp[w]), _mm_load_ps(&wsynxr[w]));
					const __m128 r3 = _mm_mul_ps(_mm_load_ps(&inp[w + xoffset]), _mm_load_ps(&wsynxl[w]));
					__m128 r6 = _mm_mul_ps(_mm_load_ps(&inp[w + yoffset]), _mm_load_ps(&wsynxr[w]));
					const __m128 r8 = _mm_mul_ps(_mm_load_ps(&inp[w + xoffset + yoffset]), _mm_load_ps(&wsynxl[w]));
					r1 = _mm_add_ps(r1, r3);
					r6 = _mm_add_ps(r6, r8);
					r1 = _mm_mul_ps(r1, wsynyrh4);
					r6 = _mm_mul_ps(r6, wsynylh4);
					r1 = _mm_add_ps(r1, r6);

					__m128i r1i = _mm_add_epi32(_mm_cvtps_epi32(r1), planebase4);
					r1i = _mm_packus_epi32(r1i, _mm_setzero_si128());
					r1i = _mm_packus_epi16(r1i, _mm_setzero_si128());
					const int out_t = _mm_cvtsi128_si32(r1i);
					memcpy(&dstp[w], &out_t, 4);
				}
				for (; w < ow; w++)   // half overlapped line of block
				{
					dstp[w] = (BYTE)min(255, max(0, (((inp[w] * wsynxr[w] + inp[w + xoffset] * wsynxl[w])*wsynyrh
						+ (inp[w + yoffset] * wsynxr[w] + inp[w + xoffset + yoffset] * wsynxl[w])*wsynylh)) + planeBase));   // x overlapped
				}
				inp += xoffset + ow;
				dstp += ow;

				for (w = 0; w < d2bwow4; w = w + 4)   // last half line of last block
				{
					__m128 r1 = _mm_mul_ps(_mm_load_ps(&inp[w]), wsynyrh4);
					const __m128 r2 = _mm_mul_ps(_mm_load_ps(&inp[w + yoffset]), wsynylh4);
					r1 = _mm_add_ps(r1, r2);
					__m128i r1i = _mm_add_epi32(_mm_cvtps_epi32(r1), planebase4);
					r1i = _mm_packus_epi32(r1i, _mm_setzero_si128());
					r1i = _mm_packus_epi16(r1i, _mm_setzero_si128());
					const int out_t = _mm_cvtsi128_si32(r1i);
					memcpy(&dstp[w], &out_t, 4);
				}
				for (; w < d2bwow; w++)   // double minus - half non-overlapped line of block
				{
					dstp[w] = (BYTE)min(255, max(0, ((inp[w] * wsynyrh + inp[w + yoffset] * wsynylh)) + planeBase));
				}
				inp += d2bwow;
				dstp += d2bwow;
			}
			for (w = 0; w < ow4; w = w + 4)   // last half line of last block
			{
				__m128 r1 = _mm_mul_ps(_mm_load_ps(&inp[w]), wsynyrh4);
				const __m128 r3 = _mm_mul_ps(_mm_load_ps(&inp[w + yoffset]), wsynylh4);
				r1 = _mm_add_ps(r1, r3);
				__m128i r1i = _mm_add_epi32(_mm_cvtps_epi32(r1), planebase4);
				r1i = _mm_packus_epi32(r1i, _mm_setzero_si128());
				r1i = _mm_packus_epi16(r1i, _mm_setzero_si128());
				const int out_t = _mm_cvtsi128_si32(r1i);
				memcpy(&dstp[w], &out_t, 4);
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
				for (w = 0; w < ow4; w = w + 4)   // half overlapped line of block
				{
					__m128 r1 = _mm_mul_ps(_mm_load_ps(&inp[w]), _mm_load_ps(&wsynxr[w]));
					const __m128 r3 = _mm_mul_ps(_mm_load_ps(&inp[w + xoffset]), _mm_load_ps(&wsynxl[w]));
					r1 = _mm_add_ps(r1, r3);
					r1 = _mm_mul_ps(r1, norm4);
					__m128i r1i = _mm_add_epi32(_mm_cvtps_epi32(r1), planebase4);
					r1i = _mm_packus_epi32(r1i, _mm_setzero_si128());
					r1i = _mm_packus_epi16(r1i, _mm_setzero_si128());
					const int out_t = _mm_cvtsi128_si32(r1i);
					memcpy(&dstp[w], &out_t, 4);
				}
				for (; w < ow; w++)   // half overlapped line of block
				{
					dstp[w] = (BYTE)min(255, max(0, ((inp[w] * wsynxr[w] + inp[w + xoffset] * wsynxl[w])*norm) + planeBase));   // x overlapped
				}
				inp += xoffset + ow;
				dstp += ow;

				for (w = 0; w < d2bwow4; w = w + 4)   // last half line of last block
				{
					const __m128 r1 = _mm_mul_ps(_mm_load_ps(&inp[w]), norm4);
					__m128i r1i = _mm_add_epi32(_mm_cvtps_epi32(r1), planebase4);
					r1i = _mm_packus_epi32(r1i, _mm_setzero_si128());
					r1i = _mm_packus_epi16(r1i, _mm_setzero_si128());
					const int out_t = _mm_cvtsi128_si32(r1i);
					memcpy(&dstp[w], &out_t, 4);
				}
				for (; w < d2bwow; w++)   // half non-overlapped line of block
				{
					dstp[w] = (BYTE)min(255, max(0, ((inp[w])*norm) + planeBase));
				}
				inp += d2bwow;
				dstp += d2bwow;
			}
			for (w = 0; w < ow4; w = w + 4)   // last half line of last block
			{
				const __m128 r1 = _mm_mul_ps(_mm_load_ps(&inp[w]), norm4);
				__m128i r1i = _mm_add_epi32(_mm_cvtps_epi32(r1), planebase4);
				r1i = _mm_packus_epi32(r1i, _mm_setzero_si128());
				r1i = _mm_packus_epi16(r1i, _mm_setzero_si128());
				const int out_t = _mm_cvtsi128_si32(r1i);
				memcpy(&dstp[w], &out_t, 4);
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
			for (w = 0; w < dbwow4; w = w + 4)   // first half line of first block
			{
				const __m128 r1 = _mm_mul_ps(_mm_load_ps(&inp[w]), norm4);
				__m128i r1i = _mm_add_epi32(_mm_cvtps_epi32(r1), planebase4);
				r1i = _mm_packus_epi32(r1i, _mm_setzero_si128());
				r1i = _mm_packus_epi16(r1i, _mm_setzero_si128());
				const int out_t = _mm_cvtsi128_si32(r1i);
				memcpy(&dstp[w], &out_t, 4);
			}
			for (; w < dbwow; w++)   // first half line of first block
			{
				dstp[w] = (BYTE)min(255, max(0, (inp[w] * norm) + planeBase));
			}
			inp += dbwow;
			dstp += dbwow;
			for (int ihx = 1; ihx < nox; ihx++) // middle blocks
			{
				for (w = 0; w < ow4; w = w + 4)   // half line of block
				{
					__m128 r1 = _mm_mul_ps(_mm_load_ps(&inp[w]), _mm_load_ps(&wsynxr[w]));
					const __m128 r3 = _mm_mul_ps(_mm_load_ps(&inp[w + xoffset]), _mm_load_ps(&wsynxl[w]));
					r1 = _mm_add_ps(r1, r3);
					r1 = _mm_mul_ps(r1, norm4);
					__m128i r1i = _mm_add_epi32(_mm_cvtps_epi32(r1), planebase4);
					r1i = _mm_packus_epi32(r1i, _mm_setzero_si128());
					r1i = _mm_packus_epi16(r1i, _mm_setzero_si128());
					const int out_t = _mm_cvtsi128_si32(r1i);
					memcpy(&dstp[w], &out_t, 4);
				}
				for (; w < ow; w++)   // half line of block
				{
					dstp[w] = (BYTE)min(255, max(0, ((inp[w] * wsynxr[w] + inp[w + xoffset] * wsynxl[w])*norm) + planeBase));   // overlapped Copy
				}
				inp += xoffset + ow;
				dstp += ow;

				for (w = 0; w < d2bwow4; w = w + 4)   // last half line of last block
				{
					const __m128 r1 = _mm_mul_ps(_mm_load_ps(&inp[w]), norm4);
					__m128i r1i = _mm_add_epi32(_mm_cvtps_epi32(r1), planebase4);
					r1i = _mm_packus_epi32(r1i, _mm_setzero_si128());
					r1i = _mm_packus_epi16(r1i, _mm_setzero_si128());
					const int out_t = _mm_cvtsi128_si32(r1i);
					memcpy(&dstp[w], &out_t, 4);
				}
				for (; w < d2bwow; w++)   // half line of block
				{
					dstp[w] = (BYTE)min(255, max(0, ((inp[w])*norm) + planeBase));
				}
				inp += d2bwow;
				dstp += d2bwow;
			}
			for (w = 0; w < ow4; w = w + 4)   // last half line of last block
			{
				const __m128 r1 = _mm_mul_ps(_mm_load_ps(&inp[w]), norm4);
				__m128i r1i = _mm_add_epi32(_mm_cvtps_epi32(r1), planebase4);
				r1i = _mm_packus_epi32(r1i, _mm_setzero_si128());
				r1i = _mm_packus_epi16(r1i, _mm_setzero_si128());
				const int out_t = _mm_cvtsi128_si32(r1i);
				memcpy(&dstp[w], &out_t, 4);
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