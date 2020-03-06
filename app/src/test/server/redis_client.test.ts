import _ from 'lodash'
import { sprintf } from 'sprintf-js'
import { RedisClient } from '../../js/server/redis_client'
import { getTestConfig } from '../util'

let client: RedisClient

beforeAll(async () => {
  client = new RedisClient(getTestConfig())
})

afterAll(async () => {
  await client.close()
})

describe('Test redis functions that are not tested elsewhere', () => {
  test('Test redis sets', async () => {
    const setName = 'redisSet'
    const memberKeys = _.range(5).map((v) => sprintf('key%s', v))
    for (const memberKey of memberKeys) {
      await client.setAdd(setName, memberKey)
    }

    const actualMemberKeys = await client.getSetMembers(setName)
    expect(actualMemberKeys.sort()).toEqual(memberKeys.sort())

    for (const memberKey of memberKeys) {
      await client.setRemove(setName, memberKey)
    }
    expect(await client.getSetMembers(setName)).toEqual([])
  })
})
