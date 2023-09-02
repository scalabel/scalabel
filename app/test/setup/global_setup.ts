import * as child from "child_process"
import * as redis from "redis"

import Logger from "../../src/server/logger"
import { getTestConfig } from "../server/util/util"

/**
 * Launch redis server
 */
function launchRedisServer(): void {
  const redisProc = child.spawn("redis-server", [
    "--appendonly",
    "no",
    "--save",
    "",
    "--port",
    String(getTestConfig().redis.port),
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

module.exports = async () => {
  Logger.info(
    "Info logger is muted for concise test status report. " +
      "The switch is in test/setup/local_setup.ts"
  )
  launchRedisServer()
  const client = redis.createClient({
    socket: {
      port: getTestConfig().redis.port
    }
  })
  await client.connect()
  await client.ping()
  await client.quit()
}
