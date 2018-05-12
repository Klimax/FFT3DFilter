/*
	FFT3DFilter plugin for Avisynth 2.5 - 3D Frequency Domain filter

	Copyright(C)2004-2006 A.G.Balakhnin aka Fizick, bag@hotmail.ru, http://avisynth.org.ru

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

//-------------------------------------------------------------------------------------------
void GetAndSubtactMean(float *in, int howmanyblocks, int bw, int bh, int ow, int oh, float *wxl, float *wxr, float *wyl, float *wyr, float *mean)
{
	int h,w, block;
	float meanblock;
	float norma;

	// calculate norma
	norma = 0;
		for (h=0; h<oh; h++)
		{
			for (w=0; w<ow; w++)
			{
				norma += wxl[w]*wyl[h];
			}
			for (w=ow; w<bw-ow; w++)
			{
				norma += wyl[h];
			}
			for (w=bw-ow; w<bw; w++)
			{
				norma += wxr[w-bw+ow]*wyl[h];
			}
		}
		for (h=oh; h<bh-oh; h++)
		{
			for (w=0; w<ow; w++)
			{
				norma += wxl[w];
			}
			for (w=ow; w<bw-ow; w++)
			{
				norma += 1;
			}
			for (w=bw-ow; w<bw; w++)
			{
				norma += wxr[w-bw+ow];
			}
		}
		for (h=bh-oh; h<bh; h++)
		{
			for (w=0; w<ow; w++)
			{
				norma += wxl[w]*wyr[h-bh+oh];
			}
			for (w=ow; w<bw-ow; w++)
			{
				norma += wyr[h-bh+oh];
			}
			for (w=bw-ow; w<bw; w++)
			{
				norma += wxr[w-bw+ow]*wyr[h-bh+oh];
			}
		}


		for (block =0; block <howmanyblocks; block++)
		{
			meanblock = 0;
			for (h=0; h<bh; h++)
			{
				for (w=0; w<bw; w++)
				{
					meanblock += in[w];
				}
				in += bw;
			}
			meanblock /= (bw*bh);
			mean[block] = meanblock;

			in -= bw*bh; // restore pointer
			for (h=0; h<bh; h++)
			{
				for (w=0; w<bw; w++)
				{
					in[w] -= meanblock;
				}
				in += bw;
			}

		}
}
//-------------------------------------------------------------------------------------------
void RestoreMean(float *in, int howmanyblocks, int bw, int bh, float *mean)
{
	int h,w, block;
	float meanblock;

		for (block =0; block <howmanyblocks; block++)
		{
			meanblock = mean[block]*(bw*bh);

			for (h=0; h<bh; h++)
			{
				for (w=0; w<bw; w++)
				{
					in[w] += meanblock;
				}
				in += bw;
			}
		}
}