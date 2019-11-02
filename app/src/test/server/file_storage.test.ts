import mockfs from 'mock-fs'
import { sprintf } from 'sprintf-js'
import { FileStorage } from '../../js/server/file_storage'
import { getProjectKey, getTaskKey, index2str } from '../../js/server/util'

beforeAll(() => {
  // mock the file system for testing storage
  mockfs({
    'data/myProject': {
      '.config': 'config contents',
      'project.json': 'project contents',
      'tasks': {
        '000000.json': '{"testField": "testValue"}',
        '000001.json': 'contents 1'
      }
    }
  })
})

const storage = new FileStorage('data')
const projectName = 'myProject'

describe('test local file storage', () => {
  test('key existence', () => {
    return Promise.all([
      checkTaskKey(0, true),
      checkTaskKey(1, true),
      checkTaskKey(2, false),
      checkProjectKey()
    ])
  })

  /* TODO: find a way to test listKeys
   without using mock-fs, since it doesn't support dirEnt */

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
    const savePromises = []
    for (let i = 3; i < 7; i++) {
      savePromises.push(checkTaskKey(i, false))
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

afterAll(() => {
  mockfs.restore()
})
