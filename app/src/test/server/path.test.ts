import * as path from '../../js/server/path'

test('Test redis keys', () => {
  const baseKey = 'baseRedisKey'
  const extendedKeys = [
    path.getRedisMetaKey(baseKey),
    path.getRedisReminderKey(baseKey),
    path.getRedisBotKey(baseKey)
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
