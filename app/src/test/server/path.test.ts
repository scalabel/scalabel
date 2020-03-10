import * as path from '../../js/server/path'
import { BotData } from '../../js/server/types'

test('Test redis keys', () => {
  const baseKey = 'baseRedisKey'
  const botData: BotData = {
    projectName: baseKey,
    taskIndex: 0,
    address: 'address',
    botId: 'botId'
  }
  const extendedKeys = [
    path.getRedisMetaKey(baseKey),
    path.getRedisReminderKey(baseKey),
    path.getRedisBotKey(botData)
  ]
  // condition 1- should get original key back with getRedisBaseKey
  for (const extendedKey of extendedKeys) {
    expect(path.getRedisBaseKey(extendedKey)).toBe(baseKey)
  }

  // condition 2- no two keys should be  the same
  const allKeys = [...extendedKeys, baseKey, path.getRedisBotSet()]
  // set will filter to unique elements
  expect(allKeys.length).toBe(new Set(allKeys).size)

})
