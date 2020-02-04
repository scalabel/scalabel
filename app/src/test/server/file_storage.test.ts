import * as fs from 'fs-extra'
import * as path from 'path'
import { sprintf } from 'sprintf-js'
import { FileStorage } from '../../js/server/file_storage'
import { getProjectKey, getTaskKey } from '../../js/server/path'
import { index2str } from '../../js/server/util'

let storage: FileStorage
let projectName: string

beforeAll(() => {
  projectName = 'myProject'

  /* Configure file system with following structure:
    'test-fs-data/myProject': {
      '.config': 'config contents',
      'project.json': 'project contents',
      'tasks': {
        '000000.json': '{"testField": "testValue"}',
        '000001.json': 'content1'
      }
    }
    This is necessary because mock-fs doesn't implement the withFileTypes
      option of fs.readDir correctly
  */

  const dataDir = 'test-fs-data'
  storage = new FileStorage(dataDir)
  const projectDir = path.join(dataDir, projectName)
  const taskDir = path.join(projectDir, 'tasks')
  fs.ensureDirSync(projectDir)
  fs.ensureDirSync(taskDir)
  fs.writeFileSync(path.join(projectDir, '.config'), 'config contents')
  fs.writeFileSync(path.join(projectDir, 'project.json'), 'project contents')
  const content0 = JSON.stringify({ testField: 'testValue' })
  fs.writeFileSync(path.join(taskDir, '000000.json'), content0)
  fs.writeFileSync(path.join(taskDir, '000001.json'), 'content1')
})

afterAll(() => {
  fs.removeSync('test-fs-data')
})

describe('test local file storage', () => {
  test('key existence', () => {
    return Promise.all([
      checkTaskKey(0, true),
      checkTaskKey(1, true),
      checkTaskKey(2, false),
      checkProjectKey()
    ])
  })

  test('list keys', async () => {
    const keys = await storage.listKeys('myProject/tasks')
    expect(keys.length).toBe(2)
    expect(keys).toContain('myProject/tasks/000000')
    expect(keys).toContain('myProject/tasks/000001')
  })

  test('list keys dir only', async () => {
    const keys = await storage.listKeys('myProject', true)
    expect(keys.length).toBe(1)
    expect(keys).toContain('myProject/tasks')
  })

  test('load', () => {
    const taskId = index2str(0)
    const key = getTaskKey(projectName, taskId)
    return storage.load(key).then((data: string) => {
      const loadedData = JSON.parse(data)
      expect(loadedData.testField).toBe('testValue')
    })
  })

  test('save then load', async () => {
    const taskId = index2str(2)
    const key = getTaskKey(projectName, taskId)
    const fakeData = '{"testField": "testValue2"}'
    await storage.save(key, fakeData)
    return Promise.all([
      checkTaskKey(2, true),
      checkTaskKey(3, false),
      checkLoad(2)
    ])
  })

  test('multiple saves multiple loads', async () => {
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
    await Promise.all(savePromises)

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

  test('delete', async () => {
    const key = 'myProject/tasks'
    await Promise.all([
      checkTaskKey(1, true),
      checkTaskKey(0, true)
    ])

    await storage.delete(key)

    return Promise.all([
      checkTaskKey(1, false),
      checkTaskKey(0, false)
    ])
  })

})

/**
 * tests if task with index exists
 */
async function checkTaskKey (index: number, shouldExist: boolean) {
  const taskId = index2str(index)
  const key = getTaskKey(projectName, taskId)
  const exists = await storage.hasKey(key)
  expect(exists).toBe(shouldExist)
}

/**
 * tests if project key exists
 */
async function checkProjectKey () {
  const key = getProjectKey(projectName)
  const exists = await storage.hasKey(key)
  expect(exists).toBe(true)
}

/**
 * tests if load on an index works
 */
async function checkLoad (index: number) {
  const data = await storage.load(getTaskKey(projectName, index2str(index)))
  const loadedData = JSON.parse(data)
  expect(loadedData.testField).toBe(sprintf('testValue%d', index))
}
