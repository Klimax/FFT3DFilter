//
//	FFT3DFilter plugin for Avisynth 2.6 - 3D Frequency Domain filter
//  Class for recording performance data
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
#include <string>
#include <vector>
#include "windows.h"
#include <unordered_map>

struct FrameInstanceData
{
	std::string function;
	int funcinst = 0;
	LONGLONG start = {0};
	LONGLONG end = {0};
	LONGLONG delta = {0};
};

struct header
{
	std::string function;
	int instances = 0;
};

class FrameData
{
public:
	long unsigned int framenum = 0;
	std::vector<FrameInstanceData> framedata;
	std::unordered_map<std::string, header> headers;

	FrameData(long unsigned int framenum) : framenum(framenum) {}
};

class Instrumentation
{
public:
	void SetFrequency(LONGLONG newfrequency) noexcept;

	void AddInstance(std::string function, LONGLONG start, LONGLONG end);
	void SaveData();
	void FrameInstrumentation(long unsigned int framenum);
	void SetMaxinstruction(int MaxFeatures);

private:
	std::vector<FrameData> data;
	std::unordered_map<std::string, header> all_headers;
	std::vector<std::string> headers_output;
	LONGLONG frequency = {0};
	std::wstring maxinstruction;
};

