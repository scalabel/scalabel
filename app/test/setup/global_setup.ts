import * as child from "child_process"

import Logger from "../../src/server/logger"

module.exports = async () => {
  Logger.info(
    "Info logger is muted for concise test status report. " +
      "The switch is in test/setup/local_setup.ts"
  )
  // Make sure this matches the redis port in test_config.yml
  const redisProc = child.spawn("redis-server", [
    "--appendonly",
    "no",
    "--save",
    "",
    "--port",
    "6378",
    "--bind",
    "127.0.0.1",
    "--protected-mode",
    "yes",
    "--loglevel",
    "warning"
  ])
  redisProc.stdout.on("data", (data) => {
    process.stdout.write(data)
  })

  redisProc.stderr.on("data", (data) => {
    process.stdout.write(data)
  })
}
