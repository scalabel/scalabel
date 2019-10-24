import mockfs from 'mock-fs'
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

 /* TODO:
  test saving- there seems to be some issue with mock-fs and writeFile
  test saving then loading
  test deletion
  */
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

afterAll(() => {
  mockfs.restore()
})
