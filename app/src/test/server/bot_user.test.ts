import { Bot } from '../../js/server/bot'
import { BotData } from '../../js/server/types'

let botData: BotData

beforeAll(async () => {
  botData = {
    webId: 'webId',
    botId: 'botId',
    serverAddress: 'address'
  }
})

describe('Test bot user', () => {
  test('Test data access', async () => {
    const bot = new Bot(botData)
    expect(bot.getData()).toEqual(botData)
  })

  test('Test management of sessions', async () => {
    const bot = new Bot(botData)
    for (let taskIndex = 0; taskIndex < 5; taskIndex++) {
      const sess = bot.makeSession('projectName', taskIndex)
      sess.actionCount = taskIndex
    }
    expect(bot.getActionCount()).toBe(10)
    bot.resetActionCount()
    expect(bot.getActionCount()).toBe(0)
  })
})
