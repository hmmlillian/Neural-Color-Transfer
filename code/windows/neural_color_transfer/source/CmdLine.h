
#ifndef CMD_LINE_H
#define CMD_LINE_H

#include <string>
#include <cstdlib>
#include <cstdio>
#include <sstream>
#include <vector>
#include <iostream>

namespace Utility {

    class CmdLine {
    public:
        using string = std::string;

        CmdLine();
        ~CmdLine();

        CmdLine(CmdLine&& src)                    { Swap(src); }
        CmdLine& operator=(CmdLine&& src)         { Swap(src); return *this; }

        void Param(const string& arg, bool& target, const string& comment) { m_Params.push_back(new BoolParm(arg, target, comment)); }
        void Param(const string& arg, int& target, const string& comment) { m_Params.push_back(new TParm<int>(arg, target, comment)); }
        void Param(const string& arg, float& target, const string& comment) { m_Params.push_back(new TParm<float>(arg, target, comment)); }
        void Param(const string& arg, double& target, const string& comment) { m_Params.push_back(new TParm<double>(arg, target, comment)); }
        void Param(const string& arg, string& target, const string& comment) { m_Params.push_back(new TParm<string>(arg, target, comment)); }

        void Comment(const string& comment) { m_Params.push_back(new ParamComment(comment)); }

        bool Parse(int argc, char* argv[]);

        int NumFiles() const { return int(m_Files.size()); }
        string GetFile(int i) const { return m_Files[i]; }
        static string GetFileWithoutExtension(string fullpath);
        static string GetExtension(string fullpath);
        static string GetPath(string fullpath);

        void Dump(std::ostream& outstream) const;

    private:
        CmdLine(const CmdLine&  src);
        CmdLine& operator=(const CmdLine& src);

        bool IsHelp(const char* arg) const;
        bool DoHelp(const char* prog) const;

        bool CheckUnique() const;
        int CheckUnique(size_t idx) const;

        void Swap(CmdLine& src) { m_Params.swap(src.m_Params); m_Files.swap(src.m_Files); }

        class Parameter {
        public:
            Parameter(const string& arg, const string& comment) : m_Arg(arg), m_Comment(comment)  {}

            bool Matches(const char* arg) const {
                return IsArg(arg) && (string(arg + 1) == m_Arg);
            }

            virtual bool Parse(const char* val) = 0;

            virtual void Print(std::ostream& outstream) const = 0;
            virtual void Dump(std::ostream& outstream) const = 0;

            static bool IsArg(const char* arg) {
                return arg && (*arg == '-' || *arg == '/');
            }

            string Arg() const { return m_Arg; }
        protected:

            string m_Arg;
            string m_Comment;
        };

        class ParamComment : public Parameter {
        public:
            ParamComment(const string& comment) : Parameter("", comment) {}

			virtual bool Parse(const char*) { return true; }

            virtual void Print(std::ostream& outstream) const {
                outstream << m_Comment << std::endl;
            }
            virtual void Dump(std::ostream& outstream) const {
            }
        };

        class BoolParm : public Parameter {
        public:
            BoolParm(const string& arg, bool& targ, const string& comment)
                : Parameter(arg, comment), m_Target(targ) {}

            virtual bool Parse(const char* val) {
                if (IsArg(val)) {
                    m_Target = true;
                    return false;
                }
                string sarg(val);
                if (sarg == "true") {
                    m_Target = true;
                }
                else if (sarg == "false") {
                    m_Target = false;
                }
                else {
                    m_Target = true;
                    return false;
                }
                return true;
            }

            virtual void Print(std::ostream& outstream) const {
                outstream << "-" << m_Arg << ": " << "(default=" << (m_Target ? "true) " : "false) ") << m_Comment << std::endl;
            }
            virtual void Dump(std::ostream& outstream) const {
                outstream << "-" << m_Arg << " " << (m_Target ? "true :" : "false :") << m_Comment << std::endl;
            }

        private:
            bool& m_Target;
        };

        template <typename T>
        class TParm : public Parameter {
        public:
            TParm(const string& arg, T& targ, const string& comment)
                : Parameter(arg, comment), m_Target(targ) {}

            virtual bool Parse(const char* val) {
				if (!(val && *val && !IsArg(val)))
				{
					return false;
				}
                std::istringstream istr(val);
                istr >> m_Target;
                return true;
            }

            virtual void Print(std::ostream& outstream) const {
                outstream << "-" << m_Arg << ": " << "(default=" << m_Target << ") " << m_Comment << std::endl;
            }
            virtual void Dump(std::ostream& outstream) const {
                outstream << "-" << m_Arg << " " << m_Target << ": " << m_Comment << std::endl;
            }
        private:
            T& m_Target;
        };

        std::vector<Parameter*> m_Params;
        std::vector<string> m_Files;
    };

};

#endif // CMD_LINE_H
