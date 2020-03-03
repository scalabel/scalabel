import * as child from 'child_process'
import * as fs from 'fs-extra'
import _ from 'lodash'
import { sprintf } from 'sprintf-js'
import * as defaults from '../../js/server/defaults'
import { FileStorage } from '../../js/server/file_storage'
import { getRedisMetaKey, getTestDir } from '../../js/server/path'
import { RedisClient } from '../../js/server/redis_client'
import { RedisStore } from '../../js/server/redis_store'
import { ServerConfig, StateMetadata } from '../../js/server/types'
import { index2str } from '../../js/server/util'
import { sleep } from '../project/util'

let redisProc: child.ChildProcessWithoutNullStreams

let defaultStore: RedisStore
let storage: FileStorage
let dataDir: string
let config: ServerConfig
let metadataString: string
let client: RedisClient

beforeAll(async () => {
  // Avoid default port 6379 and port 6377 used in box2d integration test
  config = _.clone(defaults.serverConfig)
  config.redisPort = 6378

  redisProc = child.spawn('redis-server',
    ['--appendonly', 'no', '--save', '', '--port', config.redisPort.toString(),
      '--bind', '127.0.0.1', '--protected-mode', 'yes'])

  // Buffer period for redis to launch
  await sleep(1000)
  dataDir = getTestDir('test-data-redis')
  storage = new FileStorage(dataDir)
  client = new RedisClient(config)
  defaultStore = new RedisStore(config, storage, client)
  metadataString = makeMetadata(1)
})

afterAll(async () => {
  redisProc.kill()
  fs.removeSync(dataDir)
})

describe('Test redis cache', () => {
  test('Set and get', async () => {
    const keys = _.range(5).map((v) => sprintf('test%s', v))
    const values = _.range(5).map((v) => sprintf('value%s', v))

    for (let i = 0; i < 5; i++) {
      await defaultStore.setExWithReminder(keys[i], values[i], metadataString)
      const value = await defaultStore.get(keys[i])
      expect(value).toBe(values[i])
    }
  })

  test('Delete', async () => {
    const keys = _.range(5).map((v) => sprintf('test%s', v))
    for (let i = 0; i < 5; i++) {
      defaultStore.del(keys[i])
      const value = await defaultStore.get(keys[i])
      expect(value).toBe(null)
    }
  })

  test('Writes back on timeout', async () => {
    const timeoutEnv = _.clone(config)
    timeoutEnv.timeForWrite = 0.2
    const store = new RedisStore(timeoutEnv, storage, client)

    const key = 'testKey1'
    await store.setExWithReminder(key, 'testvalue', metadataString)

    const savedKeys = await storage.listKeys('')
    expect(savedKeys.length).toBe(0)
    await sleep(800)

    const savedKeysFinal = await storage.listKeys('')
    expect(savedKeysFinal.length).toBe(1)
  })

  test('Writes back after action limit', async () => {
    const actionEnv = _.clone(config)
    actionEnv.numActionsForWrite = 5
    const store = new RedisStore(actionEnv, storage, client)

    const key = 'testKey2'
    for (let i = 0; i < 4; i++) {
      await store.setExWithReminder(key, sprintf('value%s', i), metadataString)
      const savedKeys = await storage.listKeys('')
      expect(savedKeys.length).toBe(1)
    }
    await store.setExWithReminder(key, 'value4', metadataString)
    const savedKeysFinal = await storage.listKeys('')
    expect(savedKeysFinal.length).toBe(2)
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
      await defaultStore.setExWithReminder(keys[i], values[i], metadata[i])
      const metakey = getRedisMetaKey(keys[i])
      const metavalue = await defaultStore.get(metakey)
      expect(metavalue).toBe(metadata[i])
    }
  })
})

/** Makes some dummy metadata */
function makeMetadata (taskIndex: number): string {
  const metadata: StateMetadata = {
    projectName: 'project',
    taskId: index2str(taskIndex),
    actionIds: {}
  }
  return JSON.stringify(metadata)
}
