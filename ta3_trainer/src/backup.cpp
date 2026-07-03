#include "ta3/trainer/backup.hpp"

#include <ranges>

#include <cth/io/log.hpp>


namespace ta3::trainer {
std::filesystem::path genBackupDirName(std::filesystem::path const& base) {
    std::filesystem::path path{base};
    path.remove_filename();
    path.append("backups/");
    return path;
}
std::filesystem::path genBackupFileName(std::filesystem::path const& base) {
    return genBackupDirName(base).replace_filename(
        std::format("{:%Y_%m_%d-%H_%M_%S}_{}", std::chrono::system_clock::now(), base.filename().string())
    );
}

void backup(std::filesystem::path const& base, size_t max_backups) {
    auto const baseStr = base.string();

    auto const backupDir = genBackupDirName(base);
    auto const backupFile = genBackupFileName(base);

    std::filesystem::create_directories(backupDir);
    std::filesystem::copy(base, backupFile, std::filesystem::copy_options::update_existing);
    cth::log::msg("saved backup to: {}", backupFile.string());

    auto files = std::filesystem::directory_iterator{backupDir}
        | std::views::filter([](std::filesystem::directory_entry const& path) { return path.is_regular_file(); })
        | std::views::transform(
            [&baseStr](auto& entry) {
                using tp_t = std::chrono::system_clock::time_point;
                using path_t = std::filesystem::path;
                using opt_t = std::optional<std::pair<path_t, tp_t>>;
                path_t path = entry.path();
                std::string const pathStr = path.string();

                if(pathStr.substr(pathStr.size() - baseStr.size()) != baseStr)
                    return opt_t{};

                std::stringstream tsmpSs{pathStr.substr(0, pathStr.size() - baseStr.size() - 1)};
                tp_t tp;
                std::chrono::from_stream(tsmpSs, "%Y_%m_%d-%H_%M_%S", tp);
                if(tsmpSs.fail())
                    return opt_t{};
                return opt_t{{path, tp}};
            }
        )
        | std::views::filter([](auto const& opt) { return opt.has_value(); })
        | std::views::transform([](auto const& opt) { return *opt; })
        | std::ranges::to<std::vector>();
    if(files.size() < max_backups)
        return;

    std::ranges::sort(files, [](auto const& left, auto const& right) { return left.second > right.second; });

    for(auto& entry : files | std::views::keys | std::views::drop(max_backups))
        std::filesystem::remove(entry);
}
}
