/*
	FFT3DFilter plugin for Avisynth 2.6 - 3D Frequency Domain filter
	Functions for converting video buffers

	Copyright(C)2004-2006 A.G.Balakhnin aka Fizick, bag@hotmail.ru, http://avisynth.org.ru
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

//-----------------------------------------------------------------------
//
void PlanarPlaneToCovebuf(const BYTE *srcp, int src_width, int src_height, int src_pitch, BYTE *coverbuf, int coverwidth, int coverheight, int coverpitch, int mirw, int mirh, bool interlaced, IScriptEnvironment* env) noexcept
{
	int w(0);
	const int width2 = src_width + src_width + mirw + mirw - 2;
	BYTE * coverbuf1(coverbuf + coverpitch * mirh);
	const int mirw_left = mirw % 4;
	const int coverwidth_left = coverwidth % 4;

	if (!interlaced) //progressive
	{
		for (int h = mirh; h < src_height + mirh; h++)
		{
			env->BitBlt(coverbuf1 + mirw, coverpitch, srcp, src_pitch, src_width, 1); // copy line
			for (w = 0; w < mirw - mirw_left; w += 4)
			{
				coverbuf1[w] = coverbuf1[mirw + mirw - w + 1]; // mirror left border
				coverbuf1[w + 1] = coverbuf1[mirw + mirw - w + 2]; // mirror left border
				coverbuf1[w + 2] = coverbuf1[mirw + mirw - w + 3]; // mirror left border
				coverbuf1[w + 3] = coverbuf1[mirw + mirw - w + 4]; // mirror left border
			}
			for (w = mirw - mirw_left; w < mirw; w++)
			{
				coverbuf1[w] = coverbuf1[mirw + mirw - w]; // mirror left border
			}

			for (w = src_width + mirw; w < coverwidth - coverwidth_left; w += 4)
			{
				coverbuf1[w] = coverbuf1[width2 - w]; // mirror right border
				coverbuf1[w + 1] = coverbuf1[width2 - w + 1]; // mirror right border
				coverbuf1[w + 2] = coverbuf1[width2 - w + 2]; // mirror right border
				coverbuf1[w + 3] = coverbuf1[width2 - w + 3]; // mirror right border
			}
			for (w = coverwidth - coverwidth_left; w < coverwidth; w++)
			{
				coverbuf1[w] = coverbuf1[width2 - w]; // mirror right border
			}
			coverbuf1 += coverpitch;
			srcp += src_pitch;
		}
	}
	else // interlaced
	{
		for (int h = mirh; h < src_height / 2 + mirh; h++) // first field
		{
			env->BitBlt(coverbuf1 + mirw, coverpitch, srcp, src_pitch, src_width, 1); // copy line
			for (w = 0; w < mirw; w++)
			{
				coverbuf1[w] = coverbuf1[mirw + mirw - w]; // mirror left border
			}
			for (w = src_width + mirw; w < coverwidth; w++)
			{
				coverbuf1[w] = coverbuf1[width2 - w]; // mirror right border
			}
			coverbuf1 += coverpitch;
			srcp += src_pitch * 2;
		}

		srcp -= src_pitch;
		for (int h = src_height / 2 + mirh; h < src_height + mirh; h++) // flip second field
		{
			env->BitBlt(coverbuf1 + mirw, coverpitch, srcp, src_pitch, src_width, 1); // copy line
			for (w = 0; w < mirw; w++)
			{
				coverbuf1[w] = coverbuf1[mirw + mirw - w]; // mirror left border
			}
			for (w = src_width + mirw; w < coverwidth; w++)
			{
				coverbuf1[w] = coverbuf1[width2 - w]; // mirror right border
			}
			coverbuf1 += coverpitch;
			srcp -= src_pitch * 2;
		}
	}

	BYTE * pmirror = coverbuf1 - coverpitch * 2; // pointer to vertical mirror
	for (int h = src_height + mirh; h < coverheight; h++)
	{
		env->BitBlt(coverbuf1, coverpitch, pmirror, coverpitch, coverwidth, 1); // mirror bottom line by line
		coverbuf1 += coverpitch;
		pmirror -= coverpitch;
	}
	coverbuf1 = coverbuf;
	pmirror = coverbuf1 + coverpitch * mirh * 2; // pointer to vertical mirror
	for (int h = 0; h < mirh; h++)
	{
		env->BitBlt(coverbuf1, coverpitch, pmirror, coverpitch, coverwidth, 1); // mirror bottom line by line
		coverbuf1 += coverpitch;
		pmirror -= coverpitch;
	}
}
//-----------------------------------------------------------------------
//
void CoverbufToPlanarPlane(const BYTE *coverbuf, int coverpitch, BYTE *dstp, int dst_width, int dst_height, int dst_pitch, int mirw, int mirh, bool interlaced, IScriptEnvironment* env) noexcept
{
	const BYTE *coverbuf1 = coverbuf + coverpitch * mirh + mirw;
	if (!interlaced) // progressive
	{
		for (int h = 0; h < dst_height; h++)
		{
			env->BitBlt(dstp, dst_pitch, coverbuf1, coverpitch, dst_width, 1); // copy pure frame size only
			dstp += dst_pitch;
			coverbuf1 += coverpitch;
		}
	}
	else // interlaced
	{
		for (int h = 0; h < dst_height; h += 2)
		{
			env->BitBlt(dstp, dst_pitch, coverbuf1, coverpitch, dst_width, 1); // copy pure frame size only
			dstp += dst_pitch * 2;
			coverbuf1 += coverpitch;
		}
		// second field is flipped
		dstp -= dst_pitch;
		for (int h = 0; h < dst_height; h += 2)
		{
			env->BitBlt(dstp, dst_pitch, coverbuf1, coverpitch, dst_width, 1); // copy pure frame size only
			dstp -= dst_pitch * 2;
			coverbuf1 += coverpitch;
		}
	}
}
//-----------------------------------------------------------------------
// not planar
void YUY2PlaneToCoverbuf(int plane, const BYTE *srcp, int src_width, int src_height, int src_pitch, BYTE *coverbuf, int coverwidth, int coverheight, int coverpitch, int mirw, int mirh, bool interlaced) noexcept
{
	int w(0);
	int src_width_plane(0);
	int width2(0);
	BYTE * coverbuf1(coverbuf + coverpitch * mirh + mirw); // start of image (not mirrored) v.1.0.1

	if (!interlaced)
	{
		if (plane == 0) // Y
		{
			src_width_plane = src_width / 2;
			for (int h = mirh; h < src_height + mirh; h++)
			{
				for (w = 0; w < src_width_plane; w++)
				{
					coverbuf1[w] = srcp[w << 1];// copy image line
				}
				coverbuf1 += coverpitch;
				srcp += src_pitch;
			}

		}
		else if (plane == 1) // U
		{
			src_width_plane = src_width / 4;
			for (int h = mirh; h < src_height + mirh; h++)
			{
				for (w = 0; w < src_width_plane; w++)
				{
					coverbuf1[w] = srcp[(w << 2) + 1];// copy line
				}
				coverbuf1 += coverpitch;
				srcp += src_pitch;
			}
		}
		else if (plane == 2) // V
		{
			src_width_plane = src_width / 4;
			for (int h = mirh; h < src_height + mirh; h++)
			{
				for (w = 0; w < src_width_plane; w++)
				{
					coverbuf1[w] = srcp[(w << 2) + 3];// copy line
				}
				coverbuf1 += coverpitch;
				srcp += src_pitch;
			}
		}
	}
	else // interlaced
	{
		if (plane == 0) // Y
		{
			src_width_plane = src_width / 2;
			for (int h = mirh; h < src_height / 2 + mirh; h++)
			{
				for (w = 0; w < src_width_plane; w++)
				{
					coverbuf1[w] = srcp[w << 1];// copy image line
				}
				coverbuf1 += coverpitch;
				srcp += src_pitch * 2;
			}
			srcp -= src_pitch;
			for (int h = mirh; h < src_height / 2 + mirh; h++)
			{
				for (w = 0; w < src_width_plane; w++)
				{
					coverbuf1[w] = srcp[w << 1];// copy image line
				}
				coverbuf1 += coverpitch;
				srcp -= src_pitch * 2;
			}

		}
		else if (plane == 1) // U
		{
			src_width_plane = src_width / 4;
			for (int h = mirh; h < src_height / 2 + mirh; h++)
			{
				for (w = 0; w < src_width_plane; w++)
				{
					coverbuf1[w] = srcp[(w << 2) + 1];// copy line
				}
				coverbuf1 += coverpitch;
				srcp += src_pitch * 2;
			}
			srcp -= src_pitch;
			for (int h = mirh; h < src_height / 2 + mirh; h++)
			{
				for (w = 0; w < src_width_plane; w++)
				{
					coverbuf1[w] = srcp[(w << 2) + 1];// copy line
				}
				coverbuf1 += coverpitch;
				srcp -= src_pitch * 2;
			}
		}
		else if (plane == 2) // V
		{
			src_width_plane = src_width / 4;
			for (int h = mirh; h < src_height / 2 + mirh; h++)
			{
				for (w = 0; w < src_width_plane; w++)
				{
					coverbuf1[w] = srcp[(w << 2) + 3];// copy line
				}
				coverbuf1 += coverpitch;
				srcp += src_pitch * 2;
			}
			srcp -= src_pitch;
			for (int h = mirh; h < src_height / 2 + mirh; h++)
			{
				for (w = 0; w < src_width_plane; w++)
				{
					coverbuf1[w] = srcp[(w << 2) + 3];// copy line
				}
				coverbuf1 += coverpitch;
				srcp -= src_pitch * 2;
			}
		}
	}

	// make mirrors
	coverbuf1 = coverbuf + coverpitch * mirh; //  start of first image line
	width2 = src_width_plane * 2 + mirw * 2 - 2; // for right position

	for (int h = mirh; h < src_height + mirh; h++)
	{
		for (w = 0; w < mirw; w++)
		{
			coverbuf1[w] = coverbuf1[(mirw + mirw - w)]; // mirror left border
		}
		for (w = src_width_plane + mirw; w < coverwidth; w++)
		{
			coverbuf1[w] = coverbuf1[width2 - w]; // mirror right border
		}
		coverbuf1 += coverpitch;
		//			srcp += src_pitch;
	}
	// make bottom mirror
	BYTE * pmirror = coverbuf1 - coverpitch * 2; // pointer to vertical mirror
	for (int h = src_height + mirh; h < coverheight; h++)
	{
		for (w = 0; w < coverwidth; w++)
		{
			coverbuf1[w] = pmirror[w];// copy line
		}
		coverbuf1 += coverpitch;
		pmirror -= coverpitch;
	}
	// make top mirror
	coverbuf1 = coverbuf;
	pmirror = coverbuf1 + coverpitch * mirh * 2; // pointer to vertical mirror
	for (int h = 0; h < mirh; h++)
	{
		for (w = 0; w < coverwidth; w++)
		{
			coverbuf1[w] = pmirror[w];// copy line
		}
		coverbuf1 += coverpitch;
		pmirror -= coverpitch;
	}


}
//-----------------------------------------------------------------------
// not planar
void CoverbufToYUY2Plane(int plane, const BYTE *coverbuf, int coverpitch, BYTE *dstp, int dst_width, int dst_height, int dst_pitch, int mirw, int mirh, bool interlaced) noexcept
{
	int w(0), dst_width_plane(0);
	const BYTE *coverbuf1 = coverbuf + coverpitch * mirh + mirw;

	if (!interlaced)
	{
		if (plane == 0) // Y
		{
			dst_width_plane = dst_width / 2;
			for (int h = 0; h < dst_height; h++)
			{
				for (w = 0; w < dst_width_plane; w++)
				{
					dstp[w << 1] = coverbuf1[w];// copy line
				}
				coverbuf1 += coverpitch;
				dstp += dst_pitch;
			}
		}
		else if (plane == 1) // U
		{
			dst_width_plane = dst_width / 4;
			for (int h = 0; h < dst_height; h++)
			{
				for (w = 0; w < dst_width_plane; w++)
				{
					dstp[(w << 2) + 1] = coverbuf1[w];// copy line
				}
				coverbuf1 += coverpitch;
				dstp += dst_pitch;
			}
		}
		else if (plane == 2) // V
		{
			dst_width_plane = dst_width / 4;
			for (int h = 0; h < dst_height; h++)
			{
				for (w = 0; w < dst_width_plane; w++)
				{
					dstp[(w << 2) + 3] = coverbuf1[w];// copy line
				}
				coverbuf1 += coverpitch;
				dstp += dst_pitch;
			}
		}
	}
	else //progressive
	{
		if (plane == 0) // Y
		{
			dst_width_plane = dst_width / 2;
			for (int h = 0; h < dst_height; h += 2)
			{
				for (w = 0; w < dst_width_plane; w++)
				{
					dstp[w << 1] = coverbuf1[w];// copy line
				}
				coverbuf1 += coverpitch;
				dstp += dst_pitch * 2;
			}
			dstp -= dst_pitch;
			for (int h = 0; h < dst_height; h += 2)
			{
				for (w = 0; w < dst_width_plane; w++)
				{
					dstp[w << 1] = coverbuf1[w];// copy line
				}
				coverbuf1 += coverpitch;
				dstp -= dst_pitch * 2;
			}
		}
		else if (plane == 1) // U
		{
			dst_width_plane = dst_width / 4;
			for (int h = 0; h < dst_height; h += 2)
			{
				for (w = 0; w < dst_width_plane; w++)
				{
					dstp[(w << 2) + 1] = coverbuf1[w];// copy line
				}
				coverbuf1 += coverpitch;
				dstp += dst_pitch * 2;
			}
			dstp -= dst_pitch;
			for (int h = 0; h < dst_height; h += 2)
			{
				for (w = 0; w < dst_width_plane; w++)
				{
					dstp[(w << 2) + 1] = coverbuf1[w];// copy line
				}
				coverbuf1 += coverpitch;
				dstp -= dst_pitch * 2;
			}
		}
		else if (plane == 2) // V
		{
			dst_width_plane = dst_width / 4;
			for (int h = 0; h < dst_height; h += 2)
			{
				for (w = 0; w < dst_width_plane; w++)
				{
					dstp[(w << 2) + 3] = coverbuf1[w];// copy line
				}
				coverbuf1 += coverpitch;
				dstp += dst_pitch * 2;
			}
			dstp -= dst_pitch;
			for (int h = 0; h < dst_height; h += 2)
			{
				for (w = 0; w < dst_width_plane; w++)
				{
					dstp[(w << 2) + 3] = coverbuf1[w];// copy line
				}
				coverbuf1 += coverpitch;
				dstp -= dst_pitch * 2;
			}
		}
	}
}
//-----------------------------------------------------------------------
//
void FramePlaneToCoverbuf(int plane, const PVideoFrame &src, VideoInfo vi1, BYTE *coverbuf, int coverwidth, int coverheight, int coverpitch, int mirw, int mirh, bool interlaced, IScriptEnvironment* env) noexcept
{
	const BYTE *srcp(nullptr);
	int src_width(0), src_height(0), src_pitch(0), planarNum(0);
	if (vi1.IsPlanar()) // YV12
	{
		planarNum = 1 << plane;
		srcp = src->GetReadPtr(planarNum);
		src_height = src->GetHeight(planarNum);
		src_width = src->GetRowSize(planarNum);
		src_pitch = src->GetPitch(planarNum);
		PlanarPlaneToCovebuf(srcp, src_width, src_height, src_pitch, coverbuf, coverwidth, coverheight, coverpitch, mirw, mirh, interlaced, env);
	}
	else // YUY2
	{
		srcp = src->GetReadPtr();
		src_height = src->GetHeight();
		src_width = src->GetRowSize();
		src_pitch = src->GetPitch();
		YUY2PlaneToCoverbuf(plane, srcp, src_width, src_height, src_pitch, coverbuf, coverwidth, coverheight, coverpitch, mirw, mirh, interlaced);
	}
}
//-----------------------------------------------------------------------
//
void CoverbufToFramePlane(int plane, const BYTE *coverbuf, int, int, int coverpitch, const PVideoFrame &dst, VideoInfo vi1, int mirw, int mirh, bool interlaced, IScriptEnvironment* env) noexcept
{
	BYTE *dstp(nullptr);
	int dst_width(0), dst_height(0), dst_pitch(0), planarNum(0);
	if (vi1.IsPlanar()) // YV12
	{
		planarNum = 1 << plane;
		dstp = dst->GetWritePtr(planarNum);
		dst_height = dst->GetHeight(planarNum);
		dst_width = dst->GetRowSize(planarNum);
		dst_pitch = dst->GetPitch(planarNum);
		CoverbufToPlanarPlane(coverbuf, coverpitch, dstp, dst_width, dst_height, dst_pitch, mirw, mirh, interlaced, env);
	}
	else // YUY2
	{
		dstp = dst->GetWritePtr();
		dst_height = dst->GetHeight();
		dst_width = dst->GetRowSize();
		dst_pitch = dst->GetPitch();
		CoverbufToYUY2Plane(plane, coverbuf, coverpitch, dstp, dst_width, dst_height, dst_pitch, mirw, mirh, interlaced);
	}
}