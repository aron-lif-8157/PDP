//
// Created by aronl on 2025-05-27.
//

#ifndef OUTPUT_FILE_LOGIC_HPP
#define OUTPUT_FILE_LOGIC_HPP

#include <string>

bool output_dir_check(const std::string& path);

std::string generate_outputfile_name(const std::string& path, int runs);

#endif //OUTPUT_FILE_LOGIC_HPP
