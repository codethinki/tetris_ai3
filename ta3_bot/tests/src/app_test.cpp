#include "test.hpp"

#include "ta3/bot/app.hpp"

#include <cth/coro/sync.hpp>

#include <thread>


namespace ta3::bot {

// async-by-default sanity: the root task must not run on the caller's thread.
BOT_TEST(BotApp, root_task_runs_off_caller_thread) {
    BotApp app{ai::model_t{}, {.workers = 1}};

    auto const caller = std::this_thread::get_id();

    auto probe = []() -> cth::co::executor_task<std::thread::id> { co_return std::this_thread::get_id(); };

    auto const worker = cth::co::sync(app.executor().spawn(probe()));

    EXPECT_NE(caller, worker);
}

}
