import * as child from 'child_process'
import * as fs from 'fs-extra'
import _ from 'lodash'
import { sprintf } from 'sprintf-js'
import { FileStorage } from '../../js/server/file_storage'
import { RedisStore } from '../../js/server/redis_store'
import { defaultEnv, Env } from '../../js/server/types'
import { sleep } from '../project/util'

let redisProc: child.ChildProcessWithoutNullStreams

let defaultStore: RedisStore
let storage: FileStorage
let env: Env

beforeAll(async () => {
  // Default port 6379 is used in box2d integration test, so change port here
  env = _.clone(defaultEnv)
  env.redisPort = 6378

  redisProc = child.spawn('redis-server',
    ['--appendonly', 'no', '--save', '', '--port', env.redisPort.toString()])

  // Buffer period for redis to launch
  await sleep(1000)
  storage = new FileStorage('test-data-redis')
  defaultStore = new RedisStore(env, storage)
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
      await defaultStore.setEx(keys[i], values[i], 1)
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
    const timeoutEnv = _.clone(env)
    timeoutEnv.timeForWrite = 0.2
    const store = new RedisStore(timeoutEnv, storage)

    const key = 'testKey1'
    await store.setExWithReminder(key, 'testvalue')

    const savedKeys = await storage.listKeys('')
    expect(savedKeys.length).toBe(0)
    await sleep(800)

    const savedKeysFinal = await storage.listKeys('')
    expect(savedKeysFinal.length).toBe(1)
  })

  test('Writes back after action limit', async () => {
    const actionEnv = _.clone(env)
    actionEnv.numActionsForWrite = 5
    const store = new RedisStore(actionEnv, storage)

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
