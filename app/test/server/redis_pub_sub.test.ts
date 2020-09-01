import { RedisClient } from "../../src/server/redis_client"
import { RedisPubSub } from "../../src/server/redis_pub_sub"
import { RegisterMessageType } from "../../src/types/message"
import { getTestConfig } from "./util/util"

let pubClient: RedisClient
let publisher: RedisPubSub
let subClient: RedisClient
let subscriber: RedisPubSub

beforeAll(async () => {
  const config = getTestConfig()
  pubClient = new RedisClient(config.redis)
  subClient = new RedisClient(config.redis)
  publisher = new RedisPubSub(pubClient)
  subscriber = new RedisPubSub(subClient)
})

afterAll(async () => {
  await pubClient.close()
  await subClient.close()
})

describe("Test redis publish/subscribe functionality", () => {
  test("Test register event", async (done) => {
    await subscriber.subscribeRegisterEvent(
      (_channel: string, message: string) => {
        const receivedData = JSON.parse(message) as RegisterMessageType
        expect(receivedData).toEqual(sentData)
        done()
      }
    )

    const sentData: RegisterMessageType = {
      projectName: "projectName",
      taskIndex: 0,
      sessionId: "sessionId",
      userId: "userId",
      address: "address",
      bot: false
    }
    publisher.publishRegisterEvent(sentData)
  })
})
