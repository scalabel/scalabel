import AWS from "aws-sdk"
import * as path from "path"

import { index2str } from "../../src/common/util"
import { STORAGE_FOLDERS, StorageStructure } from "../../src/const/storage"
import { getProjectKey, getTaskKey, hostname, now } from "../../src/server/path"
import { S3Storage } from "../../src/server/s3_storage"

const s3 = new AWS.S3()
const projectName = "test"
const storageName = `${hostname()}_${now()}`
const bucketRegion = "us-west-2"
const bucketName = `scalabel-test-tmp-${Date.now()}`
let storage: S3Storage

beforeAll(async () => {
  storage = new S3Storage(`${bucketRegion}:${bucketName}/${storageName}`)
  await storage.makeBucket()

  // Add keys to set up the bucket
  let keys = ["project", "tasks/000000", "tasks/000001"]

  keys = keys.map((key) =>
    path.join(StorageStructure.PROJECT, projectName, key)
  )
  const fakeData = '{"testField": "testValue"}'

  for (const key of keys) {
    await storage.save(key, fakeData)
  }
})

/**
 * Get relative project directory in the storage
 *
 * @param projectName
 * @param name
 */
function getProjectDir(name: string): string {
  return `${StorageStructure.PROJECT}/${name}`
}

/**
 * Check wether the path exist on s3
 *
 * @param key
 */
async function pathExists(key: string): Promise<boolean> {
  const params = {
    Bucket: bucketName,
    Key: key
  }
  try {
    await s3.headObject(params).promise()
    return true
  } catch (_error) {
    return false
  }
}

describe.skip("test s3 storage", () => {
  test("make dir", async () => {
    for (const f of STORAGE_FOLDERS) {
      await storage.mkdir(f)
      expect(await pathExists(storage.fullDir(f) + "/")).toBe(true)
      const file = `${f}/empty`
      await storage.save(file, "test")
      expect(await storage.hasKey(file)).toBe(true)
      await storage.delete(file)
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
    // Top level keys
    let keys = await storage.listKeys(getProjectDir("test"))
    expect(keys).toStrictEqual([
      getProjectDir("test/project"),
      getProjectDir("test/tasks")
    ])

    // Top level (dir only)
    keys = await storage.listKeys(getProjectDir("test"), true)
    expect(keys).toStrictEqual([getProjectDir("test/tasks")])

    // Task keys
    keys = await storage.listKeys(getProjectDir("test/tasks"))
    expect(keys).toStrictEqual([
      getProjectDir("test/tasks/000000"),
      getProjectDir("test/tasks/000001")
    ])
  })

  test("load", async () => {
    const taskId = index2str(0)
    const key = getTaskKey(projectName, taskId)
    return await storage.load(key).then((data: string) => {
      const loadedData = JSON.parse(data)
      expect(loadedData.testField).toBe("testValue")
    })
  })

  test("loading nonexistent key", async () => {
    await storage.load("not_a_real_key").catch((e: Error) => {
      expect(e.message).toEqual(
        `Key '${storageName}/not_a_real_key.json' does not exist`
      )
    })
  })

  test("save then load", async () => {
    const taskId = index2str(2)
    const key = getTaskKey(projectName, taskId)
    const fakeData = '{"testField": "testValue2"}'
    return await storage.save(key, fakeData).then(async () => {
      return await Promise.all([
        checkTaskKey(2, true),
        checkTaskKey(3, false),
        checkLoad(2)
      ])
    })
  })

  test("multiple saves multiple loads", async () => {
    const checkPromises = []
    for (let i = 3; i < 7; i++) {
      checkPromises.push(checkTaskKey(i, false))
    }

    return await Promise.all(checkPromises).then(async () => {
      const savePromises = []
      for (let i = 3; i < 7; i++) {
        savePromises.push(
          storage.save(
            getTaskKey(projectName, index2str(i)),
            `{"testField": "testValue${i}"}`
          )
        )
      }
      savePromises.push(storage.save("fakeFile", `fake content`))

      return await Promise.all(savePromises).then(async () => {
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
    })
  })

  test("delete", async () => {
    const key = getProjectDir(`${projectName}/tasks`)
    return await Promise.all([
      checkTaskKey(1, true),
      checkTaskKey(0, true)
    ]).then(async () => {
      return await storage.delete(key).then(async () => {
        return await Promise.all([
          checkTaskKey(1, false),
          checkTaskKey(0, false)
        ])
      })
    })
  })

  /**
   * Expensive test, so disabled by default
   */
  test.skip("list more than 1000 items", async () => {
    // First save the items
    const startInd = 100
    const subDir = "bigDir"
    const prefix = path.join(projectName, subDir)

    const promises = []
    const fileNames = []
    for (let i = startInd; i < startInd + 1500; i++) {
      const taskId = index2str(i)
      fileNames.push(taskId)
      const key = path.join(prefix, taskId)
      const fakeData = '{"testField": "testValue"}'
      promises.push(storage.save(key, fakeData))
    }
    await Promise.all(promises)

    const keys = await storage.listKeys(prefix)
    expect(keys).toStrictEqual(fileNames.map((name) => path.join(prefix, name)))
  }, 40000)
})

afterAll(async () => {
  // Cleanup: delete all keys that were created
  await storage.delete("")
  // Delete the temporary folder
  let folders: string[] = STORAGE_FOLDERS.map((f) => f.toString())
  folders = folders.map((f) => path.join(storageName, f) + "/")
  folders.push(storageName + "/")
  for (const folder of folders) {
    const params = {
      Bucket: bucketName,
      Key: folder
    }
    await s3.deleteObject(params).promise()
  }
  await storage.removeBucket()
}, 20000)

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
  return await storage.hasKey(key).then((exists) => {
    expect(exists).toBe(shouldExist)
  })
}

/**
 * tests if project key exists
 */
async function checkProjectKey(): Promise<void> {
  const key = getProjectKey(projectName)
  return await storage.hasKey(key).then((exists) => {
    expect(exists).toBe(true)
  })
}

/**
 * tests if load on an index works
 *
 * @param index
 */
async function checkLoad(index: number): Promise<void> {
  return await storage
    .load(getTaskKey(projectName, index2str(index)))
    .then((data: string) => {
      const loadedData = JSON.parse(data)
      expect(loadedData.testField).toBe(`testValue${index}`)
    })
}
