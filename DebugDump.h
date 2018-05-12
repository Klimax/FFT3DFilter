//
//	FFT3DFilter plugin for Avisynth 2.6 - 3D Frequency Domain filter
//  Class for dumping arrays for further analysis
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
#include "windows.h"

class DebugDump
{
public:
	void SaveData(float* data, int size, std::wstring function);
	void SaveData(BYTE* data, int size, std::wstring function);
	void SetMaxinstruction(int MaxFeatures);

private:
	std::wstring maxinstruction;
};

