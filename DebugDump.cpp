//
//	FFT3DFilter plugin for Avisynth 2.6 - 3D Frequency Domain filter
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

#include "DebugDump.h"
#include <fstream>
#include "enums.h"
#include <vector>

void DebugDump::SaveData(float* data, int size, std::wstring function)
{
	std::wofstream file;
	std::vector<wchar_t> TempBuffer;
	std::wstring TempString;

	DWORD retsize = GetCurrentDirectory(0, nullptr);
	TempBuffer.resize(retsize, 0);
	retsize = GetCurrentDirectory(retsize, TempBuffer.data());
	if (retsize > 0) {

		TempString.assign(TempBuffer.data());
		TempString += L"\\";
	}

	file.open(TempString + function + L"-" + maxinstruction + L".data", std::ios::out | std::ios::trunc);

	for (int i = 0; i < size; i++)
	{
		file << data[i] << L" ";
	}
	file << std::endl;

	file.close();
}

void DebugDump::SaveData(BYTE* data, int size, std::wstring function)
{
	std::wofstream file;
	std::vector<wchar_t> TempBuffer;
	std::wstring TempString;

	DWORD retsize = GetCurrentDirectory(0, nullptr);
	TempBuffer.resize(retsize, 0);
	retsize = GetCurrentDirectory(retsize, TempBuffer.data());
	if (retsize > 0) {

		TempString.assign(TempBuffer.data());
		TempString += L"\\";
	}

	file.open(TempString + function + L"-" + maxinstruction + L".data", std::ios::out | std::ios::trunc);

	for (int i = 0; i < size; i++)
	{
		file << data[i] << L" ";
	}
	file << std::endl;

	file.close();
}

void DebugDump::SetMaxinstruction(int MaxFeatures)
{
	switch (MaxFeatures) {
	case CPUK_AVX512:
		maxinstruction = L"AVX512";
		break;
	case CPUK_AVX2:
		maxinstruction = L"AVX2";
		break;
	case CPUK_AVX:
		maxinstruction = L"AVX";
		break;
	case CPUK_SSE4_2:
		maxinstruction = L"SSE4";
		break;
	case CPUK_SSE4_1:
		maxinstruction = L"SSE4";
		break;
	case CPUK_SSSE3:
		maxinstruction = L"SSSE3";
		break;
	case CPUK_SSE3:
		maxinstruction = L"SSE3";
		break;
	case CPUK_SSE2:
		maxinstruction = L"SSE2";
		break;
	case CPUK_3DNOW_EXT:
		maxinstruction = L"3DNow_ext";
		break;
	case CPUK_3DNOW:
		maxinstruction = L"3DNow";
		break;
	case CPUK_SSE:
		maxinstruction = L"SSE";
		break;
	case CPUK_MMX:
		maxinstruction = L"MMX";
		break;
	default:
		maxinstruction = L"x86";
	}
}
