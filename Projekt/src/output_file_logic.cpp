//
// Created by aronl on 2025-05-27.
//

#include "output_file_logic.hpp"

#include <sys/stat.h>
#include <dirent.h>
#include <errno.h>
#include <cstring>
#include <sstream>


bool output_dir_check(const std::string &path)
{
  if (mkdir(path.c_str(), 0755) != 0 && errno != EEXIST)
  {
    return false;
  }
  return true;
}

std::string generate_outputfile_name(const std::string& path, int runs)
{

  int max_suf = 0;
  DIR *dp = opendir(path.c_str());
  if (dp)
  {
    struct dirent *ent;
    while ((ent = readdir(dp)) != nullptr)
    {
      int file_runs, suf;
      if (std::sscanf(ent->d_name,
                      "sequentialruns_%d_%d.csv",
                      &file_runs, &suf) == 2 &&
          file_runs == runs)
      {
        if (suf > max_suf)
          max_suf = suf;
      }
    }
    closedir(dp);
  }
  int next_suf = max_suf + 1;

  std::ostringstream oss;
  oss << path << "/sequentialruns_" << runs << "_" << next_suf << ".csv";
  return oss.str();
}