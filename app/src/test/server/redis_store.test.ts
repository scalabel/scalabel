import * as fs from 'fs-extra'
import _ from 'lodash'
import { sprintf } from 'sprintf-js'
import { FileStorage } from '../../js/server/file_storage'
import { getRedisMetaKey, getTestDir } from '../../js/server/path'
import { RedisClient } from '../../js/server/redis_client'
import { RedisStore } from '../../js/server/redis_store'
import { ServerConfig, StateMetadata } from '../../js/server/types'
import { index2str } from '../../js/server/util'
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
  client = new RedisClient(config)
  defaultStore = new RedisStore(config, storage, client)
  metadataString = makeMetadata(1)
  // numWrites used as a counter across all tests that spawn files
  numWrites = 0
})

afterAll(async () => {
  fs.removeSync(dataDir)
  await client.close()
})

describe('Test redis cache', () => {
  test('Set and get and delete', async () => {
    const keys = _.range(5).map((v) => sprintf('test%s', v))
    const values = _.range(5).map((v) => sprintf('value%s', v))

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
    timeoutConfig.timeForWrite = 0.2
    timeoutConfig.redisTimeout = 2.0
    const store = new RedisStore(timeoutConfig, storage, client)

    const key = 'testKey1'
    await store.setExWithReminder(key, 'testvalue', metadataString, 1)

    await checkFileCount()
    await sleep((timeoutConfig.redisTimeout + 0.5) * 1000)
    await checkFileWritten()
  })

  test('Writes back after action limit with 1 action at a time', async () => {
    const actionConfig = _.clone(config)
    actionConfig.numActionsForWrite = 5
    const store = new RedisStore(actionConfig, storage, client)

    const key = 'testKey2'
    for (let i = 0; i < 4; i++) {
      await store.setExWithReminder(
        key, sprintf('value%s', i), metadataString, 1)
      // make sure no new files are created yet
      await checkFileCount()
    }
    await store.setExWithReminder(key, 'value4', metadataString, 1)
    await checkFileWritten()
  })

  test('Writes back after action limit with multi action packet', async () => {
    const actionConfig = _.clone(config)
    actionConfig.numActionsForWrite = 5
    const store = new RedisStore(actionConfig, storage, client)
    await checkFileCount()
    await store.setExWithReminder('key', 'value', metadataString, 5)
    await checkFileWritten()
  })

  test('Set atomic executes all ops', async () => {
    const keys = _.range(5).map((v) => sprintf('test%s', v))
    const values = _.range(5).map((v) => sprintf('value%s', v))

    await defaultStore.setAtomic(keys, values, 60)

    for (let i = 0; i < 5; i++) {
      const value = await defaultStore.get(keys[i])
      expect(value).toBe(values[i])
    }
  })

  test('Metadata is saved correctly', async () => {
    const keys = _.range(5).map((v) => sprintf('test%s', v))
    const values = _.range(5).map((v) => sprintf('value%s', v))
    const metadata = _.range(5).map((v) => makeMetadata(v))
    for (let i = 0; i < 5; i++) {
      await defaultStore.setExWithReminder(keys[i], values[i], metadata[i], 1)
      const metakey = getRedisMetaKey(keys[i])
      const metavalue = await defaultStore.get(metakey)
      expect(metavalue).toBe(metadata[i])
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
