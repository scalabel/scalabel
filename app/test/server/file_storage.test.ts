import * as fs from "fs-extra"

import { index2str } from "../../src/common/util"
import { STORAGE_FOLDERS, StorageStructure } from "../../src/const/storage"
import { FileStorage } from "../../src/server/file_storage"
import { getProjectKey, getTaskKey, getTestDir } from "../../src/server/path"
import { makeProjectDir } from "./util/util"

let storage: FileStorage
let projectName: string
let dataDir: string

beforeAll(() => {
  projectName = "myProject"
  dataDir = getTestDir("test-fs-data")
  makeProjectDir(dataDir, projectName)
  storage = new FileStorage(dataDir)
})

afterAll(() => {
  fs.removeSync(dataDir)
})

test("make dir", async () => {
  for (const f of STORAGE_FOLDERS) {
    await storage.mkdir(f)
    expect(await fs.pathExists(storage.fullDir(f))).toBe(true)
    const file = `${f}/test`
    await storage.save(file, "")
    expect(await storage.hasKey(file)).toBe(true)
  }
})

test("key existence", async () => {
  return await Promise.all([
    checkTaskKey(0, true),
    checkTaskKey(1, true),
    checkTaskKey(2, false),
    checkProjectKey()
  ])
})

test("list keys", async () => {
  const keys = await storage.listKeys(
    `${StorageStructure.PROJECT}/myProject/tasks`
  )
  expect(keys.length).toBe(2)
  expect(keys).toContain(`projects/myProject/tasks/000000`)
  expect(keys).toContain(`projects/myProject/tasks/000001`)
})

test("list keys dir only", async () => {
  const keys = await storage.listKeys(
    `${StorageStructure.PROJECT}/myProject`,
    true
  )
  expect(keys.length).toBe(1)
  expect(keys).toContain(`projects/myProject/tasks`)
})

test("load", async () => {
  const taskId = index2str(0)
  const key = getTaskKey(projectName, taskId)
  return await storage.load(key).then((data: string) => {
    const loadedData = JSON.parse(data)
    expect(loadedData.testField).toBe("testValue")
  })
})

test("save then load", async () => {
  const taskId = index2str(2)
  const key = getTaskKey(projectName, taskId)
  const fakeData = '{"testField": "testValue2"}'
  await storage.save(key, fakeData)
  return await Promise.all([
    checkTaskKey(2, true),
    checkTaskKey(3, false),
    checkLoad(2)
  ])
})

test("multiple saves multiple loads", async () => {
  const savePromises = []
  for (let i = 3; i < 7; i++) {
    savePromises.push(checkTaskKey(i, false))
    savePromises.push(
      storage.save(
        getTaskKey(projectName, index2str(i)),
        `{"testField": "testValue${i}"}`
      )
    )
  }
  savePromises.push(storage.save("fakeFile", `fake content`))
  await Promise.all(savePromises)

  const loadPromises = []
  for (let j = 3; j < 7; j++) {
    loadPromises.push(checkTaskKey(j, true))
    loadPromises.push(checkLoad(j))
  }
  loadPromises.push(
    storage.load("fakeFile").then((data: string) => {
      expect(data).toBe(`fake content`)
    })
  )
  return Promise.all(loadPromises)
})

test("delete", async () => {
  const key = `${StorageStructure.PROJECT}/myProject/tasks`
  await Promise.all([checkTaskKey(1, true), checkTaskKey(0, true)])

  await storage.delete(key)

  return await Promise.all([checkTaskKey(1, false), checkTaskKey(0, false)])
})

/**
 * tests if task with index exists
 *
 * @param index
 * @param shouldExist
 */
async function checkTaskKey(
  index: number,
  shouldExist: boolean
): Promise<void> {
  const taskId = index2str(index)
  const key = getTaskKey(projectName, taskId)
  const exists = await storage.hasKey(key)
  expect(exists).toBe(shouldExist)
}

/**
 * tests if project key exists
 */
async function checkProjectKey(): Promise<void> {
  const key = getProjectKey(projectName)
  const exists = await storage.hasKey(key)
  expect(exists).toBe(true)
}

/**
 * tests if load on an index works
 *
 * @param index
 */
async function checkLoad(index: number): Promise<void> {
  const data = await storage.load(getTaskKey(projectName, index2str(index)))
  const loadedData = JSON.parse(data)
  expect(loadedData.testField).toBe(`testValue${index}`)
}
