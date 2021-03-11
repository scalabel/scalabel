import * as fs from "fs-extra"
import _ from "lodash"

import { index2str } from "../../src/common/util"
import { FileStorage } from "../../src/server/file_storage"
import { getFileKey, getRedisMetaKey, getTestDir } from "../../src/server/path"
import { RedisCache } from "../../src/server/redis_cache"
import { RedisClient } from "../../src/server/redis_client"
import { ServerConfig } from "../../src/types/config"
import { StateMetadata } from "../../src/types/project"
import { sleep } from "../project/util"
import { getTestConfig } from "./util/util"

let defaultStore: RedisCache
let storage: FileStorage
let dataDir: string
let config: ServerConfig
let client: RedisClient
let numWrites: number

beforeAll(async () => {
  config = getTestConfig()
  dataDir = getTestDir("test-data-redis")
  storage = new FileStorage(dataDir)
  client = new RedisClient(config.redis)
  defaultStore = new RedisCache(config.redis, storage, client)
  // NumWrites used as a counter across all tests that spawn files
  numWrites = 0
})

afterAll(async () => {
  fs.removeSync(dataDir)
  await client.close()
})

describe("Test redis cache", () => {
  test("Set and get and delete", async () => {
    const keys = _.range(5).map((v) => `test${v}`)
    const values = _.range(5).map((v) => `value${v}`)

    for (let i = 0; i < 5; i++) {
      await defaultStore.set(keys[i], values[i])
      const value = await defaultStore.get(keys[i])
      expect(value).toBe(values[i])
    }

    // This also cleans up for the other tests
    for (let i = 0; i < 5; i++) {
      await defaultStore.del(keys[i])
      const value = await defaultStore.get(keys[i])
      expect(value).toBe(null)
    }
  })

  test("Writes back on timeout", async () => {
    const timeoutConfig = _.clone(config)
    timeoutConfig.redis.writebackTime = 0.2
    const store = new RedisCache(timeoutConfig.redis, storage, client)

    const key = "testKey1"
    await store.set(key, "testvalue")

    await checkFileCount()
    await sleep(1000)
    await checkFileWritten()
  })

  test("Writes back after action limit with 1 action at a time", async () => {
    const actionConfig = _.clone(config)
    actionConfig.redis.writebackCount = 5
    const store = new RedisCache(actionConfig.redis, storage, client)

    const key = "testKey2"
    for (let i = 0; i < 4; i++) {
      await store.set(key, `value${i}`)
      // Make sure no new files are created yet
      await checkFileCount()
    }
    await store.set(key, "value4")
    await checkFileWritten()
  })

  test("Writes back after action limit with multi action packet", async () => {
    const actionConfig = _.clone(config)
    actionConfig.redis.writebackCount = 5
    const store = new RedisCache(actionConfig.redis, storage, client)
    await checkFileCount()
    for (let i = 0; i < 5; i += 1) {
      await store.set("key", "value")
    }
    await checkFileWritten()
  })

  test("Set atomic executes all ops", async () => {
    const keys = _.range(5).map((v) => `test${v}`)
    const values = _.range(5).map((v) => `value${v}`)

    await defaultStore.setMulti(keys, values)

    for (let i = 0; i < 5; i++) {
      const value = await defaultStore.get(keys[i])
      expect(value).toBe(values[i])
    }
  })

  test("Metadata is saved correctly", async () => {
    const keys = _.range(5).map((v) => `test${v}`)
    const values = _.range(5).map((v) => `value${v}`)
    const metadata = _.range(5).map((v) => makeMetadata(v))
    for (let i = 0; i < 5; i++) {
      await defaultStore.set(keys[i], values[i])
      const metakey = getRedisMetaKey(keys[i])
      await defaultStore.set(metakey, metadata[i])
      const metavalue = await defaultStore.get(metakey)
      expect(metavalue).toBe(metadata[i])
    }
  })

  test("Check storage if key is not in redis store", async () => {
    const keys = _.range(5).map((v) => `testGet${v}`)
    const values = _.range(5).map((v) => `value${v}`)

    for (let i = 0; i < 5; i++) {
      await defaultStore.set(keys[i], values[i])
      const fileKey = getFileKey(keys[i])
      await storage.save(fileKey, values[i])
      await defaultStore.del(keys[i])
    }
    for (let i = 0; i < 5; i++) {
      const value = await defaultStore.get(keys[i])
      expect(value).toBe(values[i])
    }

    // This also cleans up for the other tests
    for (let i = 0; i < 5; i++) {
      await defaultStore.del(keys[i])
    }
  })
})

/** Check that expected number of files exist */
async function checkFileCount(): Promise<void> {
  const savedKeys = await storage.listKeys("")
  expect(savedKeys.length).toBe(numWrites)
}

/** Check that expected number of files have been written */
async function checkFileWritten(): Promise<void> {
  numWrites += 1
  await checkFileCount()
}

/**
 * Makes some dummy metadata
 *
 * @param taskIndex
 */
function makeMetadata(taskIndex: number): string {
  const metadata: StateMetadata = {
    projectName: "project",
    taskId: index2str(taskIndex),
    actionIds: {}
  }
  return JSON.stringify(metadata)
}
