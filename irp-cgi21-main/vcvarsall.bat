@echo off
if "%1" == "" goto x86
if not "%2" == "" goto usage

if /i %1 == x86       goto x86
if /i %1 == amd64     goto amd64
if /i %1 == x64       goto amd64
if /i %1 == arm       goto arm
if /i %1 == x86_arm   goto x86_arm
if /i %1 == x86_amd64 goto x86_amd64
goto usage

:x86
if not exist "C:\Program Files (x86)\Microsoft Visual Studio 11.0\VC\bin\vcvars32.bat" goto missing
call "C:\Program Files (x86)\Microsoft Visual Studio 11.0\VC\bin\vcvars32.bat"
goto :SetVisualStudioVersion

:amd64
if not exist "C:\Program Files (x86)\Microsoft Visual Studio 11.0\VC\bin\amd64\vcvars64.bat" goto missing
call "C:\Program Files (x86)\Microsoft Visual Studio 11.0\VC\bin\amd64\vcvars64.bat"
goto :SetVisualStudioVersion

:arm
if not exist "C:\Program Files (x86)\Microsoft Visual Studio 11.0\VC\bin\arm\vcvarsarm.bat" goto missing
call "C:\Program Files (x86)\Microsoft Visual Studio 11.0\VC\bin\arm\vcvarsarm.bat"
goto :SetVisualStudioVersion

:x86_amd64
if not exist "C:\Program Files (x86)\Microsoft Visual Studio 11.0\VC\bin\x86_amd64\vcvarsx86_amd64.bat" goto missing
call "C:\Program Files (x86)\Microsoft Visual Studio 11.0\VC\bin\x86_amd64\vcvarsx86_amd64.bat"
goto :SetVisualStudioVersion

:x86_arm
if not exist "C:\Program Files (x86)\Microsoft Visual Studio 11.0\VC\bin\x86_arm\vcvarsx86_arm.bat" goto missing
call "C:\Program Files (x86)\Microsoft Visual Studio 11.0\VC\bin\x86_arm\vcvarsx86_arm.bat"
goto :SetVisualStudioVersion

:SetVisualStudioVersion
set VisualStudioVersion=11.0
goto :eof

:usage
echo Error in script usage. The correct usage is:
echo     %0 [option]
echo where [option] is: x86 ^| amd64 ^| arm ^| x86_amd64 ^| x86_arm
echo:
echo For example:
echo     %0 x86_amd64
goto :eof

:missing
echo The specified configuration type is missing.  The tools for the
echo configuration might not be installed.
goto :eof

