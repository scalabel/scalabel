import { readdir } from "fs"
import * as fs from "fs-extra"
import * as path from "path"
import * as util from "util"

import Logger from "./logger"
import { Storage } from "./storage"

/**
 * Implements local file storage
 */
export class FileStorage extends Storage {
  /**
   * Constructor
   *
   * @param dataDir
   */
  constructor(dataDir: string) {
    super(dataDir)
    // Do this synchronously (only once)
    Logger.info(
      `Using scalabel data dir ${dataDir}. ` +
        `If it doesn't exist, it will be created`
    )
    fs.ensureDirSync(this._dataDir)
  }

  /**
   * Check if specified file exists
   *
   * @param {string} key: relative path of file
   * @param key
   */
  public async hasKey(key: string): Promise<boolean> {
    return await fs.pathExists(this.fullFile(key))
  }

  /**
   * Lists keys of files at directory specified by prefix
   *
   * @param {string} prefix: relative path of directory
   * @param {boolean} onlyDir: whether to only return keys that are directories
   * @param prefix
   * @param onlyDir
   */
  public async listKeys(
    prefix: string,
    onlyDir: boolean = false
  ): Promise<string[]> {
    const dir = this.fullDir(prefix)
    if (!(await fs.pathExists(dir))) {
      return await Promise.resolve([])
    }
    const readdirPromise = util.promisify(readdir)
    const dirEnts = await readdirPromise(dir, { withFileTypes: true })
    const keys: string[] = []
    for (const dirEnt of dirEnts) {
      // If only directories, check if it's a directory
      if (!onlyDir || dirEnt.isDirectory()) {
        const dirName = dirEnt.name
        // Remove any file extension and prepend prefix
        const keyName = path.join(prefix, path.parse(dirName).name)
        keys.push(keyName)
      }
    }
    keys.sort()
    return keys
  }

  /**
   * Saves json to at location specified by key
   *
   * @param {string} key: relative path of file
   * @param {string} json: data to save
   * @param key
   * @param json
   */
  public async save(key: string, json: string): Promise<void> {
    // TODO: Make sure undefined is the right dir options
    try {
      await fs.ensureDir(this.fullDir(path.dirname(key)), undefined)
    } catch (error) {
      // No need to reject if dir existed
      if (error.code !== "EEXIST") {
        throw error
      }
    }
    return await fs.writeFile(this.fullFile(key), json)
  }

  /**
   * Loads fields stored at a key
   *
   * @param {string} key: relative path of file
   * @param key
   */
  public async load(key: string): Promise<string> {
    return (await fs.readFile(this.fullFile(key))).toString()
  }

  /**
   * Deletes values at the key
   *
   * @param {string} key: relative path of directory
   * @param key
   */
  public async delete(key: string): Promise<void> {
    const fullDir = this.fullDir(key)
    if (fullDir === "") {
      // Don't delete everything
      throw new Error("Delete failed: tried to delete home dir")
    }
    return await fs.remove(this.fullDir(key))
  }

  /**
   * make an empty folder object on s3
   *
   * @param key
   */
  public async mkdir(key: string): Promise<void> {
    await fs.ensureDir(this.fullDir(key))
  }
}
