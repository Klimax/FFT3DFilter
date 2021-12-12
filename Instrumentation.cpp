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

#include "Instrumentation.h"
#include <fstream>
#include "enums.h"

void Instrumentation::SetFrequency(LONGLONG newfrequency) noexcept
{
	frequency = newfrequency;
}

void Instrumentation::AddInstance(std::string function, LONGLONG start, LONGLONG end)
{
	const auto frame = --(data.end());

	header htemp = { function, 1 };

	auto funcit = frame->headers.find(function);
	if (funcit == frame->headers.end()) { funcit = frame->headers.insert(std::make_pair(function, htemp)).first; }
	else { funcit->second.instances = funcit->second.instances + 1; }

	const auto funcit2 = all_headers.find(function);
	if (funcit2 == all_headers.end())
	{
		all_headers.insert(std::make_pair(function, htemp));
		headers_output.push_back(function);
	}
	else
	{
		if (funcit2->second.instances < funcit->second.instances)
		{
			funcit2->second.instances = funcit2->second.instances + 1;
			headers_output.push_back(function);
		}
	}

	const FrameInstanceData instancedata = { function, funcit->second.instances,start,end, (end - start) / frequency };
	frame->framedata.emplace_back(instancedata);
}

void Instrumentation::SaveData()
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

#ifdef FFTW_ATOM
	maxinstruction += L"-atom";
#endif

	file.open(TempString + L"perf-" + maxinstruction + L".csv", std::ios::out | std::ios::trunc);

	file << "Frame";
	for (auto it = headers_output.begin(); it < headers_output.end(); it++)
	{
		file << "" << " | " << it->c_str();
	}
	file << std::endl;

	for (auto it = data.begin(); it < data.end(); it++)
	{
		std::vector<long long int> outputdata(headers_output.size());
		file << it->framenum;

		for (auto it2 = it->framedata.begin(); it2 < it->framedata.end(); it2++)
		{
			for (unsigned int i = 0; i < headers_output.size(); i++)
			{
				if (headers_output[i] == it2->function && outputdata[i] == 0)
				{
					outputdata[i] = it2->delta; break;
				}
			}
		}

		for (auto it2 = outputdata.begin(); it2 < outputdata.end(); it2++)
		{
			file << "" << " | " << *it2;
		}
		file << std::endl;
	}

	file.close();
}

void Instrumentation::FrameInstrumentation(long unsigned int framenum)
{
	data.emplace_back(framenum);
}

void Instrumentation::SetMaxinstruction(int MaxFeatures)
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
