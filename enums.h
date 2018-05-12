//
//	FFT3DFilter plugin for Avisynth 2.6 - 3D Frequency Domain filter
//  Enums for feature detetction independent of Avisynth enums
//
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
//

#pragma once

enum {
	/* oldest CPU to support extension */
	CPUK_MMX = 1,
	CPUK_SSE = 2,
	CPUK_3DNOW = 4,
	CPUK_3DNOW_EXT = 8,
	CPUK_SSE2 = 16,
	CPUK_SSE3 = 32,
	CPUK_SSSE3 = 64,
	CPUK_SSE4_1 = 128,
	CPUK_SSE4_2 = 256,
	CPUK_AVX = 512,
	CPUK_AVX2 = 1024,
	CPUK_AVX512 = 2048,
};