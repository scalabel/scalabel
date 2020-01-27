import * as path from 'path'
import { sprintf } from 'sprintf-js'
import { S3Storage } from '../../js/server/s3_storage'
import { getProjectKey, getTaskKey, index2str } from '../../js/server/util'

beforeAll(async () => {
  await storage.makeBucket()

  // add some keys to set up the bucket
  let keys = [
    'project',
    'tasks/000000',
    'tasks/000001'
  ]

  keys = keys.map((key) => path.join('myProject', key))
  const fakeData = '{"testField": "testValue"}'

  for (const key of keys) {
    await storage.save(key, fakeData)
  }
})

const storage = new S3Storage('us-west-2:scalabel-unit-testing/data')
const projectName = 'myProject'

describe('test s3 storage', () => {
  test('key existence', () => {
    return Promise.all([
      checkTaskKey(0, true),
      checkTaskKey(1, true),
      checkTaskKey(2, false),
      checkProjectKey()
    ])
  })

  test('load', () => {
    const taskId = index2str(0)
    const key = getTaskKey(projectName, taskId)
    return storage.load(key).then((data: string) => {
      const loadedData = JSON.parse(data)
      expect(loadedData.testField).toBe('testValue')
    })
  })

  test('save then load', () => {
    const taskId = index2str(2)
    const key = getTaskKey(projectName, taskId)
    const fakeData = '{"testField": "testValue2"}'
    return storage.save(key, fakeData).then(() => {
      return Promise.all([
        checkTaskKey(2, true),
        checkTaskKey(3, false),
        checkLoad(2)
      ])
    })
  })

  test('multiple saves multiple loads', () => {
    const checkPromises = []
    for (let i = 3; i < 7; i++) {
      checkPromises.push(checkTaskKey(i, false))
    }

    return Promise.all(checkPromises).then(() => {
      const savePromises = []
      for (let i = 3; i < 7; i++) {
        savePromises.push(
          storage.save(getTaskKey(projectName, index2str(i)),
           sprintf('{"testField": "testValue%d"}', i)
          )
        )
      }
      savePromises.push(
        storage.save('fakeFile', `fake content`)
      )

      return Promise.all(savePromises).then(() => {
        const loadPromises = []
        for (let j = 3; j < 7; j++) {
          loadPromises.push(checkTaskKey(j, true))
          loadPromises.push(checkLoad(j))
        }
        loadPromises.push(
          storage.load('fakeFile').then((data: string) => {
            expect(data).toBe(`fake content`)
          })
        )
        return Promise.all(loadPromises)
      })
    })
  })

  test('delete', () => {
    const key = 'myProject/tasks'
    return Promise.all([
      checkTaskKey(1, true),
      checkTaskKey(0, true)
    ]).then(() => {
      return storage.delete(key).then(() => {
        return Promise.all([
          checkTaskKey(1, false),
          checkTaskKey(0, false)
        ])
      })
    })

  })

})

afterAll(async () => {
  // cleanup: delete all keys that were created
  await storage.delete('')
})

/**
 * tests if task with index exists
 */
function checkTaskKey (index: number, shouldExist: boolean): Promise<void> {
  const taskId = index2str(index)
  const key = getTaskKey(projectName, taskId)
  return storage.hasKey(key).then((exists) => {
    expect(exists).toBe(shouldExist)
  })
}

/**
 * tests if project key exists
 */
function checkProjectKey (): Promise<void> {
  const key = getProjectKey(projectName)
  return storage.hasKey(key).then((exists) => {
    expect(exists).toBe(true)
  })
}

/**
 * tests if load on an index works
 */
function checkLoad (index: number): Promise<void> {
  return storage.load(getTaskKey(projectName, index2str(index)))
  .then((data: string) => {
    const loadedData = JSON.parse(data)
    expect(loadedData.testField).toBe(sprintf('testValue%d', index))
  })
}
