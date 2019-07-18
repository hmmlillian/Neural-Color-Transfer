
//#include "stdafx.h"

#include "CmdLine.h"

using namespace std;
using namespace Utility;

CmdLine::CmdLine()
{

}

CmdLine::~CmdLine()
{
    for (size_t i = 0; i < m_Params.size(); ++i) {
        delete m_Params[i];
    }
}

bool CmdLine::Parse(int argc, char* argv[])
{
    CheckUnique();

    int iArg = 1; // skip the program name
    while (iArg < argc) {
        if (!Parameter::IsArg(argv[iArg])) {
            m_Files.push_back(argv[iArg]);
            ++iArg;
        }
        else if (IsHelp(argv[iArg])) {
            DoHelp(*argv);
            return false;
        }
        else {
            bool processed = false;
            for (size_t iParm = 0; iParm < m_Params.size(); ++iParm) {
                if (m_Params[iParm]->Matches(argv[iArg])) {
                    if (argv[iArg + 1] && m_Params[iParm]->Parse(argv[iArg + 1])) {
                        ++iArg;
                    }
                    ++iArg;
                    processed = true;
                    break;
                }
            }
            if (!processed) {
                // Didn't find this parameter. Ooops. Something horribly wrong.
                std::cout << "Unrecognized parameter: " << argv[iArg] << std::endl << std::endl;
                DoHelp(*argv);
                return false;
            }
        }
    }
    return true;
}

string CmdLine::GetFileWithoutExtension(string fullpath)
{
    string ext = GetExtension(fullpath);
    string path = GetPath(fullpath);

    string name = fullpath.substr(path.length(), fullpath.length() - path.length() - ext.length());
    return name;
}

string CmdLine::GetExtension(string fullpath)
{
    string::size_type dotPos = fullpath.find_last_of('.');
    string ext("");
    if (dotPos != string::size_type(-1)) {
        ext = fullpath.substr(dotPos);
    }
    return ext;
}

string CmdLine::GetPath(string fullpath)
{
    string::size_type forePos = fullpath.find_last_of('/');
    string::size_type backPos = fullpath.find_last_of('\\');
    string::size_type delimPos = forePos != -1
        ? backPos != -1
		? (forePos > backPos ? forePos:backPos)
            : backPos
        : backPos;
    string path("");
    if (delimPos != -1) {
        path = fullpath.substr(0, delimPos);
    }
    return path;
}

bool CmdLine::IsHelp(const char* arg) const
{
    if (!Parameter::IsArg(arg)) {
        return false;
    }
    const string sarg(arg + 1);
    return sarg == "h" || sarg == "?" || sarg == "help";
}

bool CmdLine::DoHelp(const char* prog) const
{
	std::cout << "Running: " << prog << std::endl;
    for (size_t iParm = 0; iParm < m_Params.size(); ++iParm) {
        m_Params[iParm]->Print(std::cout);
    }
    return true;
}


void Utility::CmdLine::Dump(std::ostream& outstream) const
{
    for (size_t iParm = 0; iParm < m_Params.size(); ++iParm) {
        m_Params[iParm]->Dump(outstream);
    }
    outstream << std::flush;
}

bool CmdLine::CheckUnique() const
{
    bool errors = false;
    for (size_t iParm = 1; iParm < m_Params.size(); ++iParm) {
        int prev = CheckUnique(iParm);
        if (prev >= 0) {
            std::cout << "Parameter obscured by earlier parameter:" << std::endl;
            m_Params[iParm]->Print(std::cout);
            std::cout << "will be ignored, earlier paramter:" << std::endl;
            m_Params[prev]->Print(std::cout);
            std::cout << "will be used." << std::endl;
            errors = true;
        }
    }
    return errors;
}

int CmdLine::CheckUnique(size_t idx) const
{
    if (!m_Params[idx]->Arg().empty()) {
        for (int prev = 0; prev < idx; ++prev) {
            if (m_Params[prev]->Arg() == m_Params[idx]->Arg()) {
                return prev;
            }
        }
    }
    return -1;
}
