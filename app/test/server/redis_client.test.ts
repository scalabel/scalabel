import _ from "lodash"

import { RedisClient } from "../../src/server/redis_client"
import { getTestConfig } from "./util/util"

let client: RedisClient

beforeAll(async () => {
  client = new RedisClient(getTestConfig().redis)
})

afterAll(async () => {
  await client.close()
})

describe("Test that redis clients catch errors", () => {
  test("Test connecting to non-existent redis server", async () => {
    const config = getTestConfig()
    config.redis.port = 6385
    const failClient = new RedisClient(config.redis)
    await failClient.close()
  })
})

describe("Test redis functions that are not tested elsewhere", () => {
  test("Test redis sets", async () => {
    const setName = "redisSet"
    const memberKeys = _.range(5).map((v) => `key${v}`)
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

  test("Test redis exists", async () => {
    const key = "testExistsKey"
    await client.set(key, "value")
    expect(await client.exists(key)).toBe(true)
    await client.del(key)
    expect(await client.exists(key)).toBe(false)
  })
})
