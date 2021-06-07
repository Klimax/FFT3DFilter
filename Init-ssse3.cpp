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

void FFT3DFilter::InitOverlapPlane_SSSE3(float *__restrict inp0, const BYTE *__restrict srcp0) noexcept
{
	int in_t(0), w(0);
	const BYTE *__restrict srcp = srcp0;// + (hrest/2)*coverpitch + wrest/2; // centered
	const int xoffset = bh * bw - (bw - ow); // skip frames
	const int yoffset = bw * nox*bh - bw * (bh - oh); // vertical offset of same block (overlap)

	const int dbwow = bw - ow;
	const int dbwow4 = dbwow - dbwow % 4;

	const int d2bwow = dbwow - ow;
	const int d2bwow4 = d2bwow - d2bwow % 4;

	const int ow4 = ow - ow % 4;

	float *__restrict inp = inp0;
	const __m128i planebase4 = _mm_set1_epi32(planeBase);
	// first top (big non-overlapped) part
	{
		for (int h = 0; h < oh; h++)
		{
			inp = inp0 + h * bw;
			const float wanyl_ = wanyl[h];
			const __m128 wanyl4 = _mm_set1_ps(wanyl_);
			for (w = 0; w < ow4; w = w + 4)   // left part  (non-overlapped) row of first block
			{
				memcpy(&in_t, &srcp[w], 4);
				__m128i r1i = _mm_cvtsi32_si128(in_t);
				r1i = _mm_shuffle_epi8(r1i, _mm_set_epi8(-1, -1, -1, 3, -1, -1, -1, 2, -1, -1, -1, 1, -1, -1, -1, 0));
				r1i = _mm_sub_epi32(r1i, planebase4);
				__m128 r1 = _mm_cvtepi32_ps(r1i);
				r1 = _mm_mul_ps(r1, wanyl4);
				r1 = _mm_mul_ps(r1, _mm_load_ps(&wanxl[w]));
				_mm_store_ps(&inp[w], r1);
			}
			for (; w < ow; w++)   // left part  (non-overlapped) row of first block
			{
				inp[w] = float(wanxl[w] * wanyl_ * (srcp[w] - planeBase));
			}

			for (; w < dbwow4; w = w + 4)   // left part  (non-overlapped) row of first block
			{
				memcpy(&in_t, &srcp[w], 4);
				__m128i r1i = _mm_cvtsi32_si128(in_t);
				r1i = _mm_shuffle_epi8(r1i, _mm_set_epi8(-1, -1, -1, 3, -1, -1, -1, 2, -1, -1, -1, 1, -1, -1, -1, 0));
				r1i = _mm_sub_epi32(r1i, planebase4);
				__m128 r1 = _mm_cvtepi32_ps(r1i);
				r1 = _mm_mul_ps(r1, wanyl4);
				_mm_store_ps(&inp[w], r1);
			}
			for (; w < dbwow; w++)   // left part  (non-overlapped) row of first block
			{
				inp[w] = float(wanyl_ * (srcp[w] - planeBase));
			}
			inp += bw - ow;
			srcp += bw - ow;
			for (int ihx = 1; ihx < nox; ihx++) // middle horizontal blocks
			{
				for (w = 0; w < ow4; w = w + 4)   // first part (overlapped) row of block
				{
					memcpy(&in_t, &srcp[w], 4);
					__m128i r1i = _mm_cvtsi32_si128(in_t);
					r1i = _mm_shuffle_epi8(r1i, _mm_set_epi8(-1, -1, -1, 3, -1, -1, -1, 2, -1, -1, -1, 1, -1, -1, -1, 0));
					r1i = _mm_sub_epi32(r1i, planebase4);
					__m128 r1 = _mm_cvtepi32_ps(r1i);
					r1 = _mm_mul_ps(r1, wanyl4);
					_mm_store_ps(&inp[w], _mm_mul_ps(r1, _mm_load_ps(&wanxr[w])));
					_mm_store_ps(&inp[w + xoffset], _mm_mul_ps(r1, _mm_load_ps(&wanxl[w])));
				}
				for (; w < ow; w++)   // first part (overlapped) row of block
				{
					const float ftmp = float(wanyl_ * (srcp[w] - planeBase));
					inp[w] = ftmp * wanxr[w]; // cur block
					inp[w + xoffset] = ftmp * wanxl[w];   // overlapped Copy - next block
				}
				inp += ow;
				inp += xoffset;
				srcp += ow;

				for (w = 0; w < d2bwow4; w = w + 4)   // center part  (non-overlapped) row of first block
				{
					memcpy(&in_t, &srcp[w], 4);
					__m128i r1i = _mm_cvtsi32_si128(in_t);
					r1i = _mm_shuffle_epi8(r1i, _mm_set_epi8(-1, -1, -1, 3, -1, -1, -1, 2, -1, -1, -1, 1, -1, -1, -1, 0));
					r1i = _mm_sub_epi32(r1i, planebase4);
					__m128 r1 = _mm_cvtepi32_ps(r1i);
					r1 = _mm_mul_ps(r1, wanyl4);
					_mm_store_ps(&inp[w], r1);
				}
				for (; w < d2bwow; w++)   // center part  (non-overlapped) row of first block
				{
					inp[w] = float(wanyl_ * (srcp[w] - planeBase));
				}
				inp += d2bwow;
				srcp += d2bwow;
			}

			for (w = 0; w < ow4; w = w + 4)   // last part (non-overlapped) of line of last block
			{
				memcpy(&in_t, &srcp[w], 4);
				__m128i r1i = _mm_cvtsi32_si128(in_t);
				r1i = _mm_shuffle_epi8(r1i, _mm_set_epi8(-1, -1, -1, 3, -1, -1, -1, 2, -1, -1, -1, 1, -1, -1, -1, 0));
				r1i = _mm_sub_epi32(r1i, planebase4);
				__m128 r1 = _mm_cvtepi32_ps(r1i);
				r1 = _mm_mul_ps(r1, wanyl4);
				r1 = _mm_mul_ps(r1, _mm_load_ps(&wanxr[w]));
				_mm_store_ps(&inp[w], r1);
			}
			for (; w < ow; w++)   // last part (non-overlapped) of line of last block
			{
				inp[w] = float(wanxr[w] * wanyl_ * (srcp[w] - planeBase));
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
				memcpy(&in_t, &srcp[w], 4);
				__m128i r1i = _mm_cvtsi32_si128(in_t);
				r1i = _mm_shuffle_epi8(r1i, _mm_set_epi8(-1, -1, -1, 3, -1, -1, -1, 2, -1, -1, -1, 1, -1, -1, -1, 0));
				r1i = _mm_sub_epi32(r1i, planebase4);
				__m128 r1 = _mm_cvtepi32_ps(r1i);
				r1 = _mm_mul_ps(r1, _mm_load_ps(&wanxl[w]));
				_mm_store_ps(&inp[w], r1);
			}
			for (; w < ow; w++)   // left part  (non-overlapped) row of first block
			{
				inp[w] = float(wanxl[w] * (srcp[w] - planeBase));
			}

			for (; w < dbwow4; w = w + 4)   // left part  (non-overlapped) row of first block
			{
				memcpy(&in_t, &srcp[w], 4);
				__m128i r1i = _mm_cvtsi32_si128(in_t);
				r1i = _mm_shuffle_epi8(r1i, _mm_set_epi8(-1, -1, -1, 3, -1, -1, -1, 2, -1, -1, -1, 1, -1, -1, -1, 0));
				r1i = _mm_sub_epi32(r1i, planebase4);
				_mm_store_ps(&inp[w], _mm_cvtepi32_ps(r1i));
			}
			for (; w < dbwow; w++)   // left part  (non-overlapped) row of first block
			{
				inp[w] = float((srcp[w] - planeBase));   // Copy each byte from source to float array
			}
			inp += bw - ow;
			srcp += bw - ow;
			for (int ihx = 1; ihx < nox; ihx++) // middle horizontal blocks
			{
				for (w = 0; w < ow4; w = w + 4)   // first part (overlapped) row of block
				{
					memcpy(&in_t, &srcp[w], 4);
					__m128i r1i = _mm_cvtsi32_si128(in_t);
					r1i = _mm_shuffle_epi8(r1i, _mm_set_epi8(-1, -1, -1, 3, -1, -1, -1, 2, -1, -1, -1, 1, -1, -1, -1, 0));
					r1i = _mm_sub_epi32(r1i, planebase4);
					const __m128 r1 = _mm_cvtepi32_ps(r1i);
					_mm_store_ps(&inp[w], _mm_mul_ps(r1, _mm_load_ps(&wanxr[w])));
					_mm_store_ps(&inp[w + xoffset], _mm_mul_ps(r1, _mm_load_ps(&wanxl[w])));
				}
				for (; w < ow; w++)   // first part (overlapped) row of block
				{
					const float ftmp = float((srcp[w] - planeBase));   // Copy each byte from source to float array
					inp[w] = ftmp * wanxr[w]; // cur block
					inp[w + xoffset] = ftmp * wanxl[w];   // overlapped Copy - next block
				}
				inp += ow;
				inp += xoffset;
				srcp += ow;

				for (w = 0; w < d2bwow4; w = w + 4)   // left part  (non-overlapped) row of first block
				{
					memcpy(&in_t, &srcp[w], 4);
					__m128i r1i = _mm_cvtsi32_si128(in_t);
					r1i = _mm_shuffle_epi8(r1i, _mm_set_epi8(-1, -1, -1, 3, -1, -1, -1, 2, -1, -1, -1, 1, -1, -1, -1, 0));
					r1i = _mm_sub_epi32(r1i, planebase4);
					_mm_store_ps(&inp[w], _mm_cvtepi32_ps(r1i));
				}
				for (; w < d2bwow; w++)   // center part  (non-overlapped) row of first block
				{
					inp[w] = float((srcp[w] - planeBase));   // Copy each byte from source to float array
				}
				inp += d2bwow;
				srcp += d2bwow;
			}

			for (w = 0; w < ow4; w = w + 4)   // left part  (non-overlapped) row of first block
			{
				memcpy(&in_t, &srcp[w], 4);
				__m128i r1i = _mm_cvtsi32_si128(in_t);
				r1i = _mm_shuffle_epi8(r1i, _mm_set_epi8(-1, -1, -1, 3, -1, -1, -1, 2, -1, -1, -1, 1, -1, -1, -1, 0));
				r1i = _mm_sub_epi32(r1i, planebase4);
				__m128 r1 = _mm_cvtepi32_ps(r1i);
				r1 = _mm_mul_ps(r1, _mm_load_ps(&wanxr[w]));
				_mm_store_ps(&inp[w], r1);
			}
			for (; w < ow; w++)   // last part (non-overlapped) line of last block
			{
				inp[w] = float(wanxr[w] * (srcp[w] - planeBase));   // Copy each byte from source to float array
			}
			inp += ow;
			srcp += ow;

			srcp += (coverpitch - coverwidth);  // Add the pitch of one line (in bytes) to the source image.
		}
	}

	for (int ihy = 1; ihy < noy; ihy++) // middle vertical
	{
		for (int h = 0; h < oh; h++) // top overlapped part
		{
			inp = inp0 + (ihy - 1)*(yoffset + (bh - oh)*bw) + (bh - oh)*bw + h * bw;
			const float wanyl_ = wanyl[h];
			const float wanyr_ = wanyr[h];
			const __m128 wanyl4 = _mm_set1_ps(wanyl_);
			const __m128 wanyr4 = _mm_set1_ps(wanyr_);

			for (w = 0; w < ow4; w = w + 4)   // left part  (non-overlapped) row of first block
			{
				memcpy(&in_t, &srcp[w], 4);
				__m128i r1i = _mm_cvtsi32_si128(in_t);
				r1i = _mm_shuffle_epi8(r1i, _mm_set_epi8(-1, -1, -1, 3, -1, -1, -1, 2, -1, -1, -1, 1, -1, -1, -1, 0));
				r1i = _mm_sub_epi32(r1i, planebase4);
				__m128 r1 = _mm_cvtepi32_ps(r1i);
				r1 = _mm_mul_ps(r1, _mm_load_ps(&wanxl[w]));
				_mm_store_ps(&inp[w], _mm_mul_ps(r1, wanyr4));
				_mm_store_ps(&inp[w + yoffset], _mm_mul_ps(r1, wanyl4));
			}
			for (; w < ow; w++)   // first half line of first block
			{
				const float ftmp = float(wanxl[w] * (srcp[w] - planeBase));
				inp[w] = ftmp * wanyr_;   // Copy each byte from source to float array
				inp[w + yoffset] = ftmp * wanyl_;   // y overlapped
			}

			for (; w < dbwow4; w = w + 4)   // left part  (non-overlapped) row of first block
			{
				memcpy(&in_t, &srcp[w], 4);
				__m128i r1i = _mm_cvtsi32_si128(in_t);
				r1i = _mm_shuffle_epi8(r1i, _mm_set_epi8(-1, -1, -1, 3, -1, -1, -1, 2, -1, -1, -1, 1, -1, -1, -1, 0));
				r1i = _mm_sub_epi32(r1i, planebase4);
				const __m128 r1 = _mm_cvtepi32_ps(r1i);
				_mm_store_ps(&inp[w], _mm_mul_ps(r1, wanyr4));
				_mm_store_ps(&inp[w + yoffset], _mm_mul_ps(r1, wanyl4));
			}
			for (; w < dbwow; w++)   // first half line of first block
			{
				const float ftmp = float((srcp[w] - planeBase));
				inp[w] = ftmp * wanyr_;   // Copy each byte from source to float array
				inp[w + yoffset] = ftmp * wanyl_;   // y overlapped
			}
			inp += bw - ow;
			srcp += bw - ow;
			for (int ihx = 1; ihx < nox; ihx++) // middle blocks
			{

				for (w = 0; w < ow4; w = w + 4)   // left part  (non-overlapped) row of first block
				{
					memcpy(&in_t, &srcp[w], 4);
					__m128i r1i = _mm_cvtsi32_si128(in_t);
					r1i = _mm_shuffle_epi8(r1i, _mm_set_epi8(-1, -1, -1, 3, -1, -1, -1, 2, -1, -1, -1, 1, -1, -1, -1, 0));
					r1i = _mm_sub_epi32(r1i, planebase4);
					const __m128 r1 = _mm_cvtepi32_ps(r1i);

					__m128 r2 = _mm_mul_ps(r1, wanyr4);
					r2 = _mm_mul_ps(r2, _mm_load_ps(&wanxr[w]));

					__m128 r3 = _mm_mul_ps(r1, wanyr4);
					r3 = _mm_mul_ps(r3, _mm_load_ps(&wanxl[w]));

					__m128 r4 = _mm_mul_ps(r1, wanyl4);
					r4 = _mm_mul_ps(r4, _mm_load_ps(&wanxr[w]));

					__m128 r5 = _mm_mul_ps(r1, wanyl4);
					r5 = _mm_mul_ps(r5, _mm_load_ps(&wanxl[w]));

					_mm_store_ps(&inp[w], r2);
					_mm_store_ps(&inp[w + xoffset], r3);
					_mm_store_ps(&inp[w + yoffset], r4);
					_mm_store_ps(&inp[w + +xoffset + yoffset], r5);
				}
				for (; w < ow; w++)   // half overlapped line of block
				{
					const float ftmp = float((srcp[w] - planeBase));   // Copy each byte from source to float array
					inp[w] = ftmp * wanxr[w] * wanyr_;
					inp[w + xoffset] = ftmp * wanxl[w] * wanyr_;   // x overlapped
					inp[w + yoffset] = ftmp * wanxr[w] * wanyl_;
					inp[w + xoffset + yoffset] = ftmp * wanxl[w] * wanyl_;   // x overlapped
				}
				inp += ow;
				inp += xoffset;
				srcp += ow;

				for (w = 0; w < d2bwow4; w = w + 4)   // left part  (non-overlapped) row of first block
				{
					memcpy(&in_t, &srcp[w], 4);
					__m128i r1i = _mm_cvtsi32_si128(in_t);
					r1i = _mm_shuffle_epi8(r1i, _mm_set_epi8(-1, -1, -1, 3, -1, -1, -1, 2, -1, -1, -1, 1, -1, -1, -1, 0));
					r1i = _mm_sub_epi32(r1i, planebase4);
					const __m128 r1 = _mm_cvtepi32_ps(r1i);
					_mm_store_ps(&inp[w], _mm_mul_ps(r1, wanyr4));
					_mm_store_ps(&inp[w + yoffset], _mm_mul_ps(r1, wanyl4));
				}
				for (; w < d2bwow; w++)   // half non-overlapped line of block
				{
					const float ftmp = float((srcp[w] - planeBase));   // Copy each byte from source to float array
					inp[w] = ftmp * wanyr_;
					inp[w + yoffset] = ftmp * wanyl_;
				}
				inp += d2bwow;
				srcp += d2bwow;
			}
			for (w = 0; w < ow4; w = w + 4)   // left part  (non-overlapped) row of first block
			{
				memcpy(&in_t, &srcp[w], 4);
				__m128i r1i = _mm_cvtsi32_si128(in_t);
				r1i = _mm_shuffle_epi8(r1i, _mm_set_epi8(-1, -1, -1, 3, -1, -1, -1, 2, -1, -1, -1, 1, -1, -1, -1, 0));
				r1i = _mm_sub_epi32(r1i, planebase4);
				__m128 r1 = _mm_cvtepi32_ps(r1i);
				r1 = _mm_mul_ps(r1, _mm_load_ps(&wanxr[w]));
				_mm_store_ps(&inp[w], _mm_mul_ps(r1, wanyr4));
				_mm_store_ps(&inp[w + yoffset], _mm_mul_ps(r1, wanyl4));
			}
			for (; w < ow; w++)   // last half line of last block
			{
				const float ftmp = float(wanxr[w] * (srcp[w] - planeBase));// Copy each byte from source to float array
				inp[w] = ftmp * wanyr_;
				inp[w + yoffset] = ftmp * wanyl_;
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
				memcpy(&in_t, &srcp[w], 4);
				__m128i r1i = _mm_cvtsi32_si128(in_t);
				r1i = _mm_shuffle_epi8(r1i, _mm_set_epi8(-1, -1, -1, 3, -1, -1, -1, 2, -1, -1, -1, 1, -1, -1, -1, 0));
				r1i = _mm_sub_epi32(r1i, planebase4);
				__m128 r1 = _mm_cvtepi32_ps(r1i);
				r1 = _mm_mul_ps(r1, _mm_load_ps(&wanxl[w]));
				_mm_store_ps(&inp[w], r1);
			}
			for (; w < ow; w++)   // first half line of first block
			{
				inp[w] = float(wanxl[w] * (srcp[w] - planeBase));   // Copy each byte from source to float array
			}

			for (; w < dbwow4; w = w + 4)   // left part  (non-overlapped) row of first block
			{
				memcpy(&in_t, &srcp[w], 4);
				__m128i r1i = _mm_cvtsi32_si128(in_t);
				r1i = _mm_shuffle_epi8(r1i, _mm_set_epi8(-1, -1, -1, 3, -1, -1, -1, 2, -1, -1, -1, 1, -1, -1, -1, 0));
				r1i = _mm_sub_epi32(r1i, planebase4);
				_mm_store_ps(&inp[w], _mm_cvtepi32_ps(r1i));
			}
			for (; w < bw - ow; w++)   // first half line of first block
			{
				inp[w] = float((srcp[w] - planeBase));   // Copy each byte from source to float array
			}
			inp += bw - ow;
			srcp += bw - ow;
			for (int ihx = 1; ihx < nox; ihx++) // middle blocks
			{
				for (w = 0; w < ow4; w = w + 4)   // left part  (non-overlapped) row of first block
				{
					memcpy(&in_t, &srcp[w], 4);
					__m128i r1i = _mm_cvtsi32_si128(in_t);
					r1i = _mm_shuffle_epi8(r1i, _mm_set_epi8(-1, -1, -1, 3, -1, -1, -1, 2, -1, -1, -1, 1, -1, -1, -1, 0));
					r1i = _mm_sub_epi32(r1i, planebase4);
					const __m128 r1 = _mm_cvtepi32_ps(r1i);
					_mm_store_ps(&inp[w], _mm_mul_ps(r1, _mm_load_ps(&wanxr[w])));
					_mm_store_ps(&inp[w + xoffset], _mm_mul_ps(r1, _mm_load_ps(&wanxl[w])));
				}
				for (; w < ow; w++)   // half overlapped line of block
				{
					const float ftmp = float((srcp[w] - planeBase));   // Copy each byte from source to float array
					inp[w] = ftmp * wanxr[w];
					inp[w + xoffset] = ftmp * wanxl[w];   // x overlapped
				}
				inp += ow;
				inp += xoffset;
				srcp += ow;

				for (w = 0; w < d2bwow4; w = w + 4)   // left part  (non-overlapped) row of first block
				{
					memcpy(&in_t, &srcp[w], 4);
					__m128i r1i = _mm_cvtsi32_si128(in_t);
					r1i = _mm_shuffle_epi8(r1i, _mm_set_epi8(-1, -1, -1, 3, -1, -1, -1, 2, -1, -1, -1, 1, -1, -1, -1, 0));
					r1i = _mm_sub_epi32(r1i, planebase4);
					_mm_store_ps(&inp[w], _mm_cvtepi32_ps(r1i));
				}
				for (; w < d2bwow; w++)   // half non-overlapped line of block
				{
					inp[w] = float((srcp[w] - planeBase));
				}
				inp += d2bwow;
				srcp += d2bwow;
			}

			for (w = 0; w < ow4; w = w + 4)   // left part  (non-overlapped) row of first block
			{
				memcpy(&in_t, &srcp[w], 4);
				__m128i r1i = _mm_cvtsi32_si128(in_t);
				r1i = _mm_shuffle_epi8(r1i, _mm_set_epi8(-1, -1, -1, 3, -1, -1, -1, 2, -1, -1, -1, 1, -1, -1, -1, 0));
				r1i = _mm_sub_epi32(r1i, planebase4);
				__m128 r1 = _mm_cvtepi32_ps(r1i);
				r1 = _mm_mul_ps(r1, _mm_load_ps(&wanxr[w]));
				_mm_store_ps(&inp[w], r1);
			}
			for (; w < ow; w++)   // last half line of last block
			{
				inp[w] = float(wanxr[w] * (srcp[w] - planeBase));
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
			const float wanyr_ = wanyr[h];
			const __m128 wanyr4 = _mm_set1_ps(wanyr_);

			for (w = 0; w < ow4; w = w + 4)   // left part  (non-overlapped) row of first block
			{
				memcpy(&in_t, &srcp[w], 4);
				__m128i r1i = _mm_cvtsi32_si128(in_t);
				r1i = _mm_shuffle_epi8(r1i, _mm_set_epi8(-1, -1, -1, 3, -1, -1, -1, 2, -1, -1, -1, 1, -1, -1, -1, 0));
				r1i = _mm_sub_epi32(r1i, planebase4);
				__m128 r1 = _mm_cvtepi32_ps(r1i);
				r1 = _mm_mul_ps(r1, _mm_load_ps(&wanxl[w]));
				r1 = _mm_mul_ps(r1, wanyr4);
				_mm_store_ps(&inp[w], r1);
			}
			for (; w < ow; w++)   // first half line of first block
			{
				inp[w] = float(wanxl[w] * wanyr_ * (srcp[w] - planeBase));   // Copy each byte from source to float array
			}

			for (; w < dbwow4; w = w + 4)   // left part  (non-overlapped) row of first block
			{
				memcpy(&in_t, &srcp[w], 4);
				__m128i r1i = _mm_cvtsi32_si128(in_t);
				r1i = _mm_shuffle_epi8(r1i, _mm_set_epi8(-1, -1, -1, 3, -1, -1, -1, 2, -1, -1, -1, 1, -1, -1, -1, 0));
				r1i = _mm_sub_epi32(r1i, planebase4);
				__m128 r1 = _mm_cvtepi32_ps(r1i);
				r1 = _mm_mul_ps(r1, wanyr4);
				_mm_store_ps(&inp[w], r1);
			}
			for (; w < dbwow; w++)   // first half line of first block
			{
				const float ftmp = float(wanyr_ * (srcp[w] - planeBase));
				inp[w] = ftmp;   // Copy each byte from source to float array
			}
			inp += bw - ow;
			srcp += bw - ow;
			for (int ihx = 1; ihx < nox; ihx++) // middle blocks
			{
				for (w = 0; w < ow4; w = w + 4)   // left part  (non-overlapped) row of first block
				{
					memcpy(&in_t, &srcp[w], 4);
					__m128i r1i = _mm_cvtsi32_si128(in_t);
					r1i = _mm_shuffle_epi8(r1i, _mm_set_epi8(-1, -1, -1, 3, -1, -1, -1, 2, -1, -1, -1, 1, -1, -1, -1, 0));
					r1i = _mm_sub_epi32(r1i, planebase4);
					__m128 r1 = _mm_cvtepi32_ps(r1i);
					r1 = _mm_mul_ps(r1, wanyr4);
					_mm_store_ps(&inp[w], _mm_mul_ps(r1, _mm_load_ps(&wanxr[w])));
					_mm_store_ps(&inp[w + xoffset], _mm_mul_ps(r1, _mm_load_ps(&wanxl[w])));
				}
				for (; w < ow; w++)   // half line of block
				{
					const float ftmp = float(wanyr_ * (srcp[w] - planeBase));   // Copy each byte from source to float array
					inp[w] = ftmp * wanxr[w];
					inp[w + xoffset] = ftmp * wanxl[w];   // overlapped Copy
				}
				inp += ow;
				inp += xoffset;
				srcp += ow;

				for (w = 0; w < d2bwow4; w = w + 4)   // left part  (non-overlapped) row of first block
				{
					memcpy(&in_t, &srcp[w], 4);
					__m128i r1i = _mm_cvtsi32_si128(in_t);
					r1i = _mm_shuffle_epi8(r1i, _mm_set_epi8(-1, -1, -1, 3, -1, -1, -1, 2, -1, -1, -1, 1, -1, -1, -1, 0));
					r1i = _mm_sub_epi32(r1i, planebase4);
					const __m128 r1 = _mm_cvtepi32_ps(r1i);
					_mm_store_ps(&inp[w], _mm_mul_ps(r1, wanyr4));
				}
				for (; w < d2bwow; w++)   // center part  (non-overlapped) row of first block
				{
					inp[w] = float(wanyr_ * (srcp[w] - planeBase));   // Copy each byte from source to float array
				}
				inp += d2bwow;
				srcp += d2bwow;
			}

			for (w = 0; w < ow4; w = w + 4)   // left part  (non-overlapped) row of first block
			{
				memcpy(&in_t, &srcp[w], 4);
				__m128i r1i = _mm_cvtsi32_si128(in_t);
				r1i = _mm_shuffle_epi8(r1i, _mm_set_epi8(-1, -1, -1, 3, -1, -1, -1, 2, -1, -1, -1, 1, -1, -1, -1, 0));
				r1i = _mm_sub_epi32(r1i, planebase4);
				__m128 r1 = _mm_cvtepi32_ps(r1i);
				r1 = _mm_mul_ps(r1, wanyr4);
				r1 = _mm_mul_ps(r1, _mm_load_ps(&wanxr[w]));
				_mm_store_ps(&inp[w], r1);
			}
			for (; w < ow; w++)   // last half line of last block
			{
				inp[w] = float(wanxr[w] * wanyr_ * (srcp[w] - planeBase));   // Copy each byte from source to float array
			}
			inp += ow;
			srcp += ow;

			srcp += (coverpitch - coverwidth);  // Add the pitch of one line (in bytes) to the source image.
		}
	}
}

void FFT3DFilter::InitOverlapPlane_wt2_SSSE3(float *__restrict inp0, const BYTE *__restrict srcp0) noexcept
{
	int in_t(0), w(0);
	const BYTE *__restrict srcp = srcp0;// + (hrest/2)*coverpitch + wrest/2; // centered
	const int xoffset = bh * bw - (bw - ow); // skip frames
	const int yoffset = bw * nox*bh - bw * (bh - oh); // vertical offset of same block (overlap)

	const int dbwow = bw - ow;
	const int dbwow4 = dbwow - dbwow % 4;
	const int d2bwow = dbwow - ow;
	const int d2bwow4 = d2bwow - d2bwow % 4;
	const int ow4 = ow - ow % 4;
	const int dbwow_ow = dbwow + ow;
	const int dbwow_ow4 = dbwow_ow - dbwow_ow % 4;

	float *__restrict inp = inp0;
	const __m128i planebase4 = _mm_set1_epi32(planeBase);

	// first top (big non-overlapped) part
	{
		for (int h = 0; h < oh; h++)
		{
			inp = inp0 + h * bw;
			for (w = 0; w < dbwow_ow4; w = w + 4)   // left part  (non-overlapped) row of first block
			{
				memcpy(&in_t, &srcp[w], 4);
				__m128i r1i = _mm_cvtsi32_si128(in_t);
				r1i = _mm_shuffle_epi8(r1i, _mm_set_epi8(-1, -1, -1, 3, -1, -1, -1, 2, -1, -1, -1, 1, -1, -1, -1, 0));
				r1i = _mm_sub_epi32(r1i, planebase4);
				_mm_store_ps(&inp[w], _mm_cvtepi32_ps(r1i));
			}
			for (; w < dbwow_ow; w++)   // left part  (non-overlapped) row of first block
			{
				inp[w] = float(srcp[w] - planeBase);   // Copy each byte from source to float array
			}
			inp += bw - ow;
			srcp += bw - ow;
			for (int ihx = 1; ihx < nox; ihx++) // middle horizontal blocks
			{
				for (w = 0; w < ow4; w = w + 4)   // first part (overlapped) row of block
				{
					memcpy(&in_t, &srcp[w], 4);
					__m128i r1i = _mm_cvtsi32_si128(in_t);
					r1i = _mm_shuffle_epi8(r1i, _mm_set_epi8(-1, -1, -1, 3, -1, -1, -1, 2, -1, -1, -1, 1, -1, -1, -1, 0));
					r1i = _mm_sub_epi32(r1i, planebase4);
					_mm_store_ps(&inp[w], _mm_cvtepi32_ps(r1i));
					_mm_store_ps(&inp[w + xoffset], _mm_cvtepi32_ps(r1i));
				}
				for (; w < ow; w++)   // first part (overlapped) row of block
				{
					const float ftmp = float(srcp[w] - planeBase);   // Copy each byte from source to float array
					inp[w] = ftmp; // cur block
					inp[w + xoffset] = ftmp;   // overlapped Copy - next block
				}

				inp += ow;
				inp += xoffset;
				srcp += ow;

				for (w = 0; w < d2bwow4; w = w + 4)   // left part  (non-overlapped) row of first block
				{
					memcpy(&in_t, &srcp[w], 4);
					__m128i r1i = _mm_cvtsi32_si128(in_t);
					r1i = _mm_shuffle_epi8(r1i, _mm_set_epi8(-1, -1, -1, 3, -1, -1, -1, 2, -1, -1, -1, 1, -1, -1, -1, 0));
					r1i = _mm_sub_epi32(r1i, planebase4);
					const __m128 r1 = _mm_cvtepi32_ps(r1i);
					_mm_store_ps(&inp[w], r1);
				}
				for (; w < d2bwow; w++)   // center part  (non-overlapped) row of first block
				{
					inp[w] = float(srcp[w] - planeBase);   // Copy each byte from source to float array
				}
				inp += d2bwow;
				srcp += d2bwow;
			}
			for (w = 0; w < ow4; w = w + 4)   // left part  (non-overlapped) row of first block
			{
				memcpy(&in_t, &srcp[w], 4);
				__m128i r1i = _mm_cvtsi32_si128(in_t);
				r1i = _mm_shuffle_epi8(r1i, _mm_set_epi8(-1, -1, -1, 3, -1, -1, -1, 2, -1, -1, -1, 1, -1, -1, -1, 0));
				r1i = _mm_sub_epi32(r1i, planebase4);
				_mm_store_ps(&inp[w], _mm_cvtepi32_ps(r1i));
			}
			for (; w < ow; w++)   // left part  (non-overlapped) row of first block
			{
				inp[w] = float(srcp[w] - planeBase);   // Copy each byte from source to float array
			}
			inp += ow;
			srcp += ow;
			srcp += (coverpitch - coverwidth);  // Add the pitch of one line (in bytes) to the source image.
		}
		for (int h = oh; h < bh - oh; h++)
		{
			inp = inp0 + h * bw;
			for (w = 0; w < dbwow_ow4; w = w + 4)   // left part  (non-overlapped) row of first block
			{
				memcpy(&in_t, &srcp[w], 4);
				__m128i r1i = _mm_cvtsi32_si128(in_t);
				r1i = _mm_shuffle_epi8(r1i, _mm_set_epi8(-1, -1, -1, 3, -1, -1, -1, 2, -1, -1, -1, 1, -1, -1, -1, 0));
				r1i = _mm_sub_epi32(r1i, planebase4);
				_mm_store_ps(&inp[w], _mm_cvtepi32_ps(r1i));
			}
			for (; w < dbwow_ow; w++)   // left part  (non-overlapped) row of first block
			{
				inp[w] = float(srcp[w] - planeBase);   // Copy each byte from source to float array
			}
			inp += bw - ow;
			srcp += bw - ow;
			for (int ihx = 1; ihx < nox; ihx++) // middle horizontal blocks
			{
				for (w = 0; w < ow4; w = w + 4)   // first part (overlapped) row of block
				{
					memcpy(&in_t, &srcp[w], 4);
					__m128i r1i = _mm_cvtsi32_si128(in_t);
					r1i = _mm_shuffle_epi8(r1i, _mm_set_epi8(-1, -1, -1, 3, -1, -1, -1, 2, -1, -1, -1, 1, -1, -1, -1, 0));
					r1i = _mm_sub_epi32(r1i, planebase4);
					_mm_store_ps(&inp[w], _mm_cvtepi32_ps(r1i));
					_mm_store_ps(&inp[w + xoffset], _mm_cvtepi32_ps(r1i));
				}
				for (; w < ow; w++)   // first part (overlapped) row of block
				{
					const float ftmp = float(srcp[w] - planeBase);   // Copy each byte from source to float array
					inp[w] = ftmp; // cur block
					inp[w + xoffset] = ftmp;   // overlapped Copy - next block
				}
				inp += ow;
				inp += xoffset;
				srcp += ow;

				for (w = 0; w < d2bwow4; w = w + 4)   // left part  (non-overlapped) row of first block
				{
					memcpy(&in_t, &srcp[w], 4);
					__m128i r1i = _mm_cvtsi32_si128(in_t);
					r1i = _mm_shuffle_epi8(r1i, _mm_set_epi8(-1, -1, -1, 3, -1, -1, -1, 2, -1, -1, -1, 1, -1, -1, -1, 0));
					r1i = _mm_sub_epi32(r1i, planebase4);
					const __m128 r1 = _mm_cvtepi32_ps(r1i);
					_mm_store_ps(&inp[w], r1);
				}
				for (; w < d2bwow; w++)   // center part  (non-overlapped) row of first block
				{
					inp[w] = float(srcp[w] - planeBase);   // Copy each byte from source to float array
				}
				inp += d2bwow;
				srcp += d2bwow;
			}
			for (w = 0; w < ow4; w = w + 4)   // left part  (non-overlapped) row of first block
			{
				memcpy(&in_t, &srcp[w], 4);
				__m128i r1i = _mm_cvtsi32_si128(in_t);
				r1i = _mm_shuffle_epi8(r1i, _mm_set_epi8(-1, -1, -1, 3, -1, -1, -1, 2, -1, -1, -1, 1, -1, -1, -1, 0));
				r1i = _mm_sub_epi32(r1i, planebase4);
				_mm_store_ps(&inp[w], _mm_cvtepi32_ps(r1i));
			}
			for (; w < ow; w++)   // left part  (non-overlapped) row of first block
			{
				inp[w] = float(srcp[w] - planeBase);   // Copy each byte from source to float array
			}
			inp += ow;
			srcp += ow;

			srcp += (coverpitch - coverwidth);  // Add the pitch of one line (in bytes) to the source image.
		}
	}

	for (int ihy = 1; ihy < noy; ihy++) // middle vertical
	{
		int poffset = (ihy - 1)*(yoffset + (bh - oh)*bw) + (bh - oh)*bw;
		for (int h = 0; h < oh; h++) // top overlapped part
		{
			inp = inp0 + poffset + h * bw;
			for (w = 0; w < ow4; w = w + 4)   // first half line of first block
			{
				memcpy(&in_t, &srcp[w], 4);
				__m128i r1i = _mm_cvtsi32_si128(in_t);
				r1i = _mm_shuffle_epi8(r1i, _mm_set_epi8(-1, -1, -1, 3, -1, -1, -1, 2, -1, -1, -1, 1, -1, -1, -1, 0));
				r1i = _mm_sub_epi32(r1i, planebase4);
				_mm_store_ps(&inp[w], _mm_cvtepi32_ps(r1i));
				_mm_store_ps(&inp[w + yoffset], _mm_cvtepi32_ps(r1i));
			}
			for (; w < ow; w++)   // first half line of first block
			{
				const float ftmp = float(srcp[w] - planeBase);
				inp[w] = ftmp;   // Copy each byte from source to float array
				inp[w + yoffset] = ftmp;   // y overlapped
			}

			for (w = ow; w < dbwow4; w = w + 4)   // left part  (non-overlapped) row of first block
			{
				memcpy(&in_t, &srcp[w], 4);
				__m128i r1i = _mm_cvtsi32_si128(in_t);
				r1i = _mm_shuffle_epi8(r1i, _mm_set_epi8(-1, -1, -1, 3, -1, -1, -1, 2, -1, -1, -1, 1, -1, -1, -1, 0));
				r1i = _mm_sub_epi32(r1i, planebase4);
				_mm_store_ps(&inp[w], _mm_cvtepi32_ps(r1i));
			}
			for (; w < dbwow; w++)   // left part  (non-overlapped) row of first block
			{
				inp[w] = float(srcp[w] - planeBase);   // Copy each byte from source to float array
			}
			inp += bw - ow;
			srcp += bw - ow;
			for (int ihx = 1; ihx < nox; ihx++) // middle blocks
			{
				for (w = 0; w < ow4; w = w + 4)   // half overlapped line of block
				{
					memcpy(&in_t, &srcp[w], 4);
					__m128i r1i = _mm_cvtsi32_si128(in_t);
					r1i = _mm_shuffle_epi8(r1i, _mm_set_epi8(-1, -1, -1, 3, -1, -1, -1, 2, -1, -1, -1, 1, -1, -1, -1, 0));
					r1i = _mm_sub_epi32(r1i, planebase4);
					_mm_store_ps(&inp[w], _mm_cvtepi32_ps(r1i));
					_mm_store_ps(&inp[w + xoffset], _mm_cvtepi32_ps(r1i));
					_mm_store_ps(&inp[w + yoffset], _mm_cvtepi32_ps(r1i));
					_mm_store_ps(&inp[w + xoffset + yoffset], _mm_cvtepi32_ps(r1i));
				}
				for (; w < ow; w++)   // half overlapped line of block
				{
					const float ftmp = float(srcp[w] - planeBase);   // Copy each byte from source to float array
					inp[w] = ftmp;
					inp[w + xoffset] = ftmp;   // x overlapped
					inp[w + yoffset] = ftmp;
					inp[w + xoffset + yoffset] = ftmp;   // x overlapped
				}

				inp += ow;
				inp += xoffset;
				srcp += ow;

				for (w = 0; w < d2bwow4; w = w + 4)   // left part  (non-overlapped) row of first block
				{
					memcpy(&in_t, &srcp[w], 4);
					__m128i r1i = _mm_cvtsi32_si128(in_t);
					r1i = _mm_shuffle_epi8(r1i, _mm_set_epi8(-1, -1, -1, 3, -1, -1, -1, 2, -1, -1, -1, 1, -1, -1, -1, 0));
					r1i = _mm_sub_epi32(r1i, planebase4);
					const __m128 r1 = _mm_cvtepi32_ps(r1i);
					_mm_store_ps(&inp[w], r1);
					_mm_store_ps(&inp[w + yoffset], r1);
				}
				for (; w < d2bwow; w++)   // half non-overlapped line of block
				{
					const float ftmp = float(srcp[w] - planeBase);   // Copy each byte from source to float array
					inp[w] = ftmp;
					inp[w + yoffset] = ftmp;
				}
				inp += d2bwow;
				srcp += d2bwow;
			}
			for (w = 0; w < ow4; w = w + 4)   // first half line of first block
			{
				memcpy(&in_t, &srcp[w], 4);
				__m128i r1i = _mm_cvtsi32_si128(in_t);
				r1i = _mm_shuffle_epi8(r1i, _mm_set_epi8(-1, -1, -1, 3, -1, -1, -1, 2, -1, -1, -1, 1, -1, -1, -1, 0));
				r1i = _mm_sub_epi32(r1i, planebase4);
				_mm_store_ps(&inp[w], _mm_cvtepi32_ps(r1i));
				_mm_store_ps(&inp[w + yoffset], _mm_cvtepi32_ps(r1i));
			}
			for (; w < ow; w++)   // first half line of first block
			{
				const float ftmp = float(srcp[w] - planeBase);
				inp[w] = ftmp;   // Copy each byte from source to float array
				inp[w + yoffset] = ftmp;   // y overlapped
			}
			inp += ow;
			srcp += ow;

			srcp += (coverpitch - coverwidth);  // Add the pitch of one line (in bytes) to the source image.
		}
		// middle  vertical nonovelapped part
		for (int h = 0; h < bh - oh - oh; h++)
		{
			poffset = (ihy - 1)*(yoffset + (bh - oh)*bw) + (bh)*bw + yoffset;
			inp = inp0 + poffset + h * bw;
			for (w = 0; w < dbwow_ow4; w = w + 4)   // left part  (non-overlapped) row of first block
			{
				memcpy(&in_t, &srcp[w], 4);
				__m128i r1i = _mm_cvtsi32_si128(in_t);
				r1i = _mm_shuffle_epi8(r1i, _mm_set_epi8(-1, -1, -1, 3, -1, -1, -1, 2, -1, -1, -1, 1, -1, -1, -1, 0));
				r1i = _mm_sub_epi32(r1i, planebase4);
				_mm_store_ps(&inp[w], _mm_cvtepi32_ps(r1i));
			}
			for (; w < dbwow_ow; w++)   // left part  (non-overlapped) row of first block
			{
				inp[w] = float(srcp[w] - planeBase);   // Copy each byte from source to float array
			}
			inp += bw - ow;
			srcp += bw - ow;
			for (int ihx = 1; ihx < nox; ihx++) // middle blocks
			{
				for (w = 0; w < ow4; w = w + 4)   // first part (overlapped) row of block
				{
					memcpy(&in_t, &srcp[w], 4);
					__m128i r1i = _mm_cvtsi32_si128(in_t);
					r1i = _mm_shuffle_epi8(r1i, _mm_set_epi8(-1, -1, -1, 3, -1, -1, -1, 2, -1, -1, -1, 1, -1, -1, -1, 0));
					r1i = _mm_sub_epi32(r1i, planebase4);
					_mm_store_ps(&inp[w], _mm_cvtepi32_ps(r1i));
					_mm_store_ps(&inp[w + xoffset], _mm_cvtepi32_ps(r1i));
				}
				for (; w < ow; w++)   // first part (overlapped) row of block
				{
					const float ftmp = float(srcp[w] - planeBase);   // Copy each byte from source to float array
					inp[w] = ftmp; // cur block
					inp[w + xoffset] = ftmp;   // overlapped Copy - next block
				}
				inp += ow;
				inp += xoffset;
				srcp += ow;

				for (w = 0; w < d2bwow4; w = w + 4)   // left part  (non-overlapped) row of first block
				{
					memcpy(&in_t, &srcp[w], 4);
					__m128i r1i = _mm_cvtsi32_si128(in_t);
					r1i = _mm_shuffle_epi8(r1i, _mm_set_epi8(-1, -1, -1, 3, -1, -1, -1, 2, -1, -1, -1, 1, -1, -1, -1, 0));
					r1i = _mm_sub_epi32(r1i, planebase4);
					const __m128 r1 = _mm_cvtepi32_ps(r1i);
					_mm_store_ps(&inp[w], r1);
				}
				for (; w < d2bwow; w++)   // half non-overlapped line of block
				{
					inp[w] = float(srcp[w] - planeBase);   // Copy each byte from source to float array
				}
				inp += d2bwow;
				srcp += d2bwow;
			}
			for (w = 0; w < ow4; w = w + 4)   // left part  (non-overlapped) row of first block
			{
				memcpy(&in_t, &srcp[w], 4);
				__m128i r1i = _mm_cvtsi32_si128(in_t);
				r1i = _mm_shuffle_epi8(r1i, _mm_set_epi8(-1, -1, -1, 3, -1, -1, -1, 2, -1, -1, -1, 1, -1, -1, -1, 0));
				r1i = _mm_sub_epi32(r1i, planebase4);
				_mm_store_ps(&inp[w], _mm_cvtepi32_ps(r1i));
			}
			for (; w < ow; w++)   // left part  (non-overlapped) row of first block
			{
				inp[w] = float(srcp[w] - planeBase);   // Copy each byte from source to float array
			}
			inp += ow;
			srcp += ow;

			srcp += (coverpitch - coverwidth);  // Add the pitch of one line (in bytes) to the source image.
		}

	}

	const int ihy = noy; // last bottom  part
	{
		int poffset = (ihy - 1)*(yoffset + (bh - oh)*bw) + (bh - oh)*bw;
		for (int h = 0; h < oh; h++)
		{
			inp = inp0 + poffset + h * bw;
			for (w = 0; w < dbwow_ow4; w = w + 4)   // left part  (non-overlapped) row of first block
			{
				memcpy(&in_t, &srcp[w], 4);
				__m128i r1i = _mm_cvtsi32_si128(in_t);
				r1i = _mm_shuffle_epi8(r1i, _mm_set_epi8(-1, -1, -1, 3, -1, -1, -1, 2, -1, -1, -1, 1, -1, -1, -1, 0));
				r1i = _mm_sub_epi32(r1i, planebase4);
				_mm_store_ps(&inp[w], _mm_cvtepi32_ps(r1i));
			}
			for (; w < dbwow_ow; w++)   // left part  (non-overlapped) row of first block
			{
				inp[w] = float(srcp[w] - planeBase);   // Copy each byte from source to float array
			}
			inp += bw - ow;
			srcp += bw - ow;
			for (int ihx = 1; ihx < nox; ihx++) // middle blocks
			{
				for (w = 0; w < ow4; w = w + 4)   // first part (overlapped) row of block
				{
					memcpy(&in_t, &srcp[w], 4);
					__m128i r1i = _mm_cvtsi32_si128(in_t);
					r1i = _mm_shuffle_epi8(r1i, _mm_set_epi8(-1, -1, -1, 3, -1, -1, -1, 2, -1, -1, -1, 1, -1, -1, -1, 0));
					r1i = _mm_sub_epi32(r1i, planebase4);
					_mm_store_ps(&inp[w], _mm_cvtepi32_ps(r1i));
					_mm_store_ps(&inp[w + xoffset], _mm_cvtepi32_ps(r1i));
				}
				for (; w < ow; w++)   // first part (overlapped) row of block
				{
					const float ftmp = float(srcp[w] - planeBase);   // Copy each byte from source to float array
					inp[w] = ftmp; // cur block
					inp[w + xoffset] = ftmp;   // overlapped Copy - next block
				}
				inp += ow;
				inp += xoffset;
				srcp += ow;

				for (w = 0; w < d2bwow4; w = w + 4)   // left part  (non-overlapped) row of first block
				{
					memcpy(&in_t, &srcp[w], 4);
					__m128i r1i = _mm_cvtsi32_si128(in_t);
					r1i = _mm_shuffle_epi8(r1i, _mm_set_epi8(-1, -1, -1, 3, -1, -1, -1, 2, -1, -1, -1, 1, -1, -1, -1, 0));
					r1i = _mm_sub_epi32(r1i, planebase4);
					const __m128 r1 = _mm_cvtepi32_ps(r1i);
					_mm_store_ps(&inp[w], r1);
				}
				for (; w < d2bwow; w++)   // center part  (non-overlapped) row of first block
				{
					inp[w] = float(srcp[w] - planeBase);   // Copy each byte from source to float array
				}
				inp += d2bwow;
				srcp += d2bwow;
			}
			for (w = 0; w < ow4; w = w + 4)   // left part  (non-overlapped) row of first block
			{
				memcpy(&in_t, &srcp[w], 4);
				__m128i r1i = _mm_cvtsi32_si128(in_t);
				r1i = _mm_shuffle_epi8(r1i, _mm_set_epi8(-1, -1, -1, 3, -1, -1, -1, 2, -1, -1, -1, 1, -1, -1, -1, 0));
				r1i = _mm_sub_epi32(r1i, planebase4);
				_mm_store_ps(&inp[w], _mm_cvtepi32_ps(r1i));
			}
			for (; w < ow; w++)   // left part  (non-overlapped) row of first block
			{
				inp[w] = float(srcp[w] - planeBase);   // Copy each byte from source to float array
			}
			inp += ow;
			srcp += ow;

			srcp += (coverpitch - coverwidth);  // Add the pitch of one line (in bytes) to the source image.
		}
	}
}
