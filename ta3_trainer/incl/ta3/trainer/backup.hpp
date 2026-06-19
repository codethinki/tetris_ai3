#pragma once
#include <filesystem>


namespace ta3::trainer {
constexpr size_t DEFAULT_MAX_BACKUPS = 5;


std::filesystem::path genBackupDirName(std::filesystem::path const& base);
std::filesystem::path genBackupFileName(std::filesystem::path const& base);
void backup(std::filesystem::path const& base, size_t max_backups = DEFAULT_MAX_BACKUPS);

}
