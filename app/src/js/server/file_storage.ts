import * as fs from 'fs-extra'
import * as path from 'path'
import { Storage } from './storage'

/**
 * Implements local file storage
 */
export class FileStorage extends Storage {
  /**
   * Constructor
   */
  constructor (dataDir: string) {
    super(dataDir)
    // do this synchronously (only once)
    fs.ensureDirSync(this.dataDir)
  }

  /**
   * Check if specified file exists
   * @param {string} key: relative path of file
   */
  public async hasKey (key: string): Promise<boolean> {
    return fs.pathExists(this.fullFile(key))
  }

  /**
   * Lists keys of files at directory specified by prefix
   * @param {string} prefix: relative path of directory
   * @param {boolean} onlyDir: whether to only return keys that are directories
   */
  public async listKeys (
    prefix: string, onlyDir: boolean = false): Promise<string[]> {
    const dirEnts = await fs.promises.readdir(
      this.fullDir(prefix), { withFileTypes: true })
    const keys: string[] = []
    for (const dirEnt of dirEnts) {
      // if only directories, check if it's a directory
      if (!onlyDir || dirEnt.isDirectory()) {
        const dirName = dirEnt.name
        // remove any file extension and prepend prefix
        const keyName = path.join(prefix, path.parse(dirName).name)
        keys.push(keyName)
      }
    }
    keys.sort()
    return keys
  }

  /**
   * Saves json to at location specified by key
   * @param {string} key: relative path of file
   * @param {string} json: data to save
   */
  public async save (key: string, json: string): Promise<void> {
    // TODO: Make sure undefined is the right dir options
    try {
      await fs.ensureDir(this.fullDir(path.dirname(key)), undefined)
    } catch (error) {
      // no need to reject if dir existed
      if (error && error.code !== 'EEXIST') {
        throw error
      }
    }
    return fs.writeFile(this.fullFile(key), json)
  }

  /**
   * Loads fields stored at a key
   * @param {string} key: relative path of file
   */
  public async load (key: string): Promise<string> {
    return new Promise((resolve, reject) => {
      fs.readFile(this.fullFile(key), (error: Error, buf: Buffer) => {
        if (error) {
          reject(error)
          return
        }
        resolve(buf.toString())
      })
    })
  }

  /**
   * Deletes values at the key
   * @param {string} key: relative path of directory
   */
  public async delete (key: string): Promise<void> {
    const fullDir = this.fullDir(key)
    if (fullDir === '') {
      // Don't delete everything
      throw new Error('Delete failed: tried to delete home dir')
    }
    return fs.remove(this.fullDir(key))
  }
}
