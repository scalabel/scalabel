import * as fs from 'fs-extra'
import _ from 'lodash'
import { index2str } from '../../src/common/util'
import { FileStorage } from '../../src/server/file_storage'
import { getFileKey, getRedisMetaKey, getTestDir } from '../../src/server/path'
import { RedisClient } from '../../src/server/redis_client'
import { RedisStore } from '../../src/server/redis_store'
import { ServerConfig } from '../../src/types/config'
import { StateMetadata } from '../../src/types/project'
import { sleep } from '../project/util'
import { getTestConfig } from './util/util'

let defaultStore: RedisStore
let storage: FileStorage
let dataDir: string
let config: ServerConfig
let metadataString: string
let client: RedisClient
let numWrites: number

beforeAll(async () => {
  config = getTestConfig()
  dataDir = getTestDir('test-data-redis')
  storage = new FileStorage(dataDir)
  client = new RedisClient(config.redis)
  defaultStore = new RedisStore(config.redis, storage, client)
  metadataString = makeMetadata(1)
  // NumWrites used as a counter across all tests that spawn files
  numWrites = 0
})

afterAll(async () => {
  fs.removeSync(dataDir)
  await client.close()
})

describe('Test redis cache', () => {
  test('Set and get and delete', async () => {
    const keys = _.range(5).map((v) => `test${v}`)
    const values = _.range(5).map((v) => `value${v}`)

    for (let i = 0; i < 5; i++) {
      await defaultStore.setExWithReminder(
        keys[i], values[i], metadataString, 1)
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

  test('Writes back on timeout', async () => {
    const timeoutConfig = _.clone(config)
    timeoutConfig.redis.writebackTime = 0.2
    timeoutConfig.redis.timeout = 2.0
    const store = new RedisStore(timeoutConfig.redis, storage, client)

    const key = 'testKey1'
    await store.setExWithReminder(key, 'testvalue', metadataString, 1)

    await checkFileCount()
    await sleep((timeoutConfig.redis.timeout + 0.5) * 1000)
    await checkFileWritten()
  })

  test('Writes back after action limit with 1 action at a time', async () => {
    const actionConfig = _.clone(config)
    actionConfig.redis.writebackActions = 5
    const store = new RedisStore(actionConfig.redis, storage, client)

    const key = 'testKey2'
    for (let i = 0; i < 4; i++) {
      await store.setExWithReminder(key, `value${i}`, metadataString, 1)
      // Make sure no new files are created yet
      await checkFileCount()
    }
    await store.setExWithReminder(key, 'value4', metadataString, 1)
    await checkFileWritten()
  })

  test('Writes back after action limit with multi action packet', async () => {
    const actionConfig = _.clone(config)
    actionConfig.redis.writebackActions = 5
    const store = new RedisStore(actionConfig.redis, storage, client)
    await checkFileCount()
    await store.setExWithReminder('key', 'value', metadataString, 5)
    await checkFileWritten()
  })

  test('Set atomic executes all ops', async () => {
    const keys = _.range(5).map((v) => `test${v}`)
    const values = _.range(5).map((v) => `value${v}`)

    await defaultStore.setAtomic(keys, values, 60)

    for (let i = 0; i < 5; i++) {
      const value = await defaultStore.get(keys[i])
      expect(value).toBe(values[i])
    }
  })

  test('Metadata is saved correctly', async () => {
    const keys = _.range(5).map((v) => `test${v}`)
    const values = _.range(5).map((v) => `value${v}`)
    const metadata = _.range(5).map((v) => makeMetadata(v))
    for (let i = 0; i < 5; i++) {
      await defaultStore.setExWithReminder(keys[i], values[i], metadata[i], 1)
      const metakey = getRedisMetaKey(keys[i])
      const metavalue = await defaultStore.get(metakey)
      expect(metavalue).toBe(metadata[i])
    }
  })

  test('Check storage if key is not in redis store', async () => {
    const keys = _.range(5).map((v) => `testGet${v}`)
    const values = _.range(5).map((v) => `value${v}`)

    for (let i = 0; i < 5; i++) {
      await defaultStore.setExWithReminder(
        keys[i], values[i], metadataString, 1)
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
async function checkFileCount () {
  const savedKeys = await storage.listKeys('')
  expect(savedKeys.length).toBe(numWrites)
}

/** Check that expected number of files have been written */
async function checkFileWritten () {
  numWrites += 1
  await checkFileCount()
}

/** Makes some dummy metadata */
function makeMetadata (taskIndex: number): string {
  const metadata: StateMetadata = {
    projectName: 'project',
    taskId: index2str(taskIndex),
    actionIds: {}
  }
  return JSON.stringify(metadata)
}
