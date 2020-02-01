import * as child from 'child_process'
import * as fs from 'fs-extra'
import _ from 'lodash'
import { sprintf } from 'sprintf-js'
import { FileStorage } from '../../js/server/file_storage'
import { RedisStore } from '../../js/server/redis_store'
import { sleep } from '../project/util'

let redisProc: child.ChildProcessWithoutNullStreams

let defaultStore: RedisStore
let storage: FileStorage
// Default port 6379 is used in box2d integration test, so change port here
let redisPort: number
let redisTimeout: number
let timeForWrite: number
let numActionsForWrite: number

beforeAll(async () => {
  redisPort = 6378
  redisTimeout = 3600
  timeForWrite = 600
  numActionsForWrite = 10

  redisProc = child.spawn('redis-server',
    ['--appendonly', 'no', '--save', '', '--port', redisPort.toString()])

  // Buffer period for redis to launch
  await sleep(1000)
  defaultStore = new RedisStore(redisPort, redisTimeout,
    timeForWrite, numActionsForWrite)
  storage = new FileStorage('test-data-redis')
  defaultStore.initialize(storage)
})

afterAll(() => {
  redisProc.kill()
  fs.removeSync('test-data-redis')
})

describe('Test redis cache', () => {
  test('Set and get', async () => {
    const keys = _.range(5).map((v) => sprintf('test%s', v))
    const values = _.range(5).map((v) => sprintf('value%s', v))

    for (let i = 0; i < 5; i++) {
      await defaultStore.setEx(keys[i], values[i], 1000)
      const value = await defaultStore.get(keys[i])
      expect(value).toBe(values[i])
    }
  })

  test('Delete', async () => {
    const keys = _.range(5).map((v) => sprintf('test%s', v))
    for (let i = 0; i < 5; i++) {
      await defaultStore.del(keys[i])
      const value = await defaultStore.get(keys[i])
      expect(value).toBe(null)
    }
  })

  test('Writes back on timeout', async () => {
    const store = new RedisStore(redisPort, redisTimeout,
      0.2, numActionsForWrite)
    store.initialize(storage)

    const key = 'testKey1'
    await store.setExWithReminder(key, 'testvalue')
    const savedKeys = await storage.listKeys('')
    expect(savedKeys.length).toBe(0)

    await sleep(800)
    const savedKeysFinal = await storage.listKeys('')
    expect(savedKeysFinal.length).toBe(1)
  })

  test('Writes back after action limit', async () => {
    const store = new RedisStore(redisPort, redisTimeout,
      timeForWrite, 5)
    store.initialize(storage)

    const key = 'testKey2'
    for (let i = 0; i < 4; i++) {
      await store.setExWithReminder(key, sprintf('value%s', i))
      const savedKeys = await storage.listKeys('')
      expect(savedKeys.length).toBe(1)
    }
    await store.setExWithReminder(key, 'value4')
    const savedKeysFinal = await storage.listKeys('')
    expect(savedKeysFinal.length).toBe(2)
  })
})
