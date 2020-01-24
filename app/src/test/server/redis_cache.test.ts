import * as child from 'child_process'
import * as fs from 'fs-extra'
import _ from 'lodash'
import { sprintf } from 'sprintf-js'
import { FileStorage } from '../../js/server/file_storage'
import { RedisCache } from '../../js/server/redis_cache'
import Session from '../../js/server/server_session'
import { defaultEnv, Env } from '../../js/server/types'
import { sleep } from '../project/util'

let redisProc: child.ChildProcessWithoutNullStreams

beforeAll(async () => {
  redisProc = child.spawn('redis-server', ['--appendonly', 'no', '--save', ''])

  // Buffer period for redis to launch
  await sleep(1000)
})

afterAll(() => {
  redisProc.kill()
  fs.removeSync('test-data-redis')
})

describe('test redis cache', () => {
  const defaultCache = new RedisCache()
  const storage = new FileStorage('test-data-redis')
  Session.setStorage(storage)

  test('set and get', async () => {
    const keys = _.range(5).map((v) => sprintf('test%s', v))
    const values = _.range(5).map((v) => sprintf('value%s', v))

    for (let i = 0; i < 5; i++) {
      await defaultCache.setEx(keys[i], values[i], 1000)
      const value = await defaultCache.get(keys[i])
      expect(value).toBe(values[i])
    }
  })

  test('delete', async () => {
    const keys = _.range(5).map((v) => sprintf('test%s', v))
    for (let i = 0; i < 5; i++) {
      await defaultCache.del(keys[i])
      const value = await defaultCache.get(keys[i])
      expect(value).toBe(null)
    }
  })

  test('writes back on timeout', async () => {
    const timeLimitConfig: Env = defaultEnv
    timeLimitConfig.timeForWrite = 0.2
    Session.setEnv(timeLimitConfig)
    const cache = new RedisCache()

    const key = 'testKey1'
    await cache.setExWithReminder(key, 'testvalue')
    const savedKeys = await storage.listKeys('')
    expect(savedKeys.length).toBe(0)
    await sleep(800)
    const savedKeysFinal = await storage.listKeys('')
    expect(savedKeysFinal.length).toBe(1)
  })

  test('writes back after action limit', async () => {
    const actionLimitConfig: Env = defaultEnv
    actionLimitConfig.numActionsForWrite = 5
    Session.setEnv(actionLimitConfig)
    const cache = new RedisCache()

    const key = 'testKey2'
    for (let i = 0; i < 4; i++) {
      await cache.setExWithReminder(key, sprintf('value%s', i))
      const savedKeys = await storage.listKeys('')
      expect(savedKeys.length).toBe(1)
    }
    await cache.setExWithReminder(key, 'value4')
    const savedKeysFinal = await storage.listKeys('')
    expect(savedKeysFinal.length).toBe(2)
  })
})
