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
  constructor (dataPath: string) {
    super(dataPath)
    // do this synchronously (only once)
    fs.ensureDirSync(this.dataDir)
  }

  /**
   * Check if specified file exists
   * @param {string} key: relative path of file
   */
  public hasKey (key: string): Promise<boolean> {
    return new Promise<boolean>((resolve, reject) => {
      fs.pathExists(this.fullFile(key), (err, exists) => {
        err ? reject(err) : resolve(exists)
      })
    })
  }

  /**
   * Lists keys of files at directory specified by prefix
   * @param {string} prefix: relative path of directory
   * @param {boolean} onlyDir: whether to only return keys that are directories
   */
  public listKeys (
    prefix: string, onlyDir: boolean = false): Promise<string[]> {
    return fs.promises.readdir(this.fullDir(prefix), { withFileTypes: true })
      .then((dirEnts: fs.Dirent[]) => {
        const keys = []
        for (const dirEnt of dirEnts) {
          // if only directories, check if it's a directory
          if (!onlyDir || dirEnt.isDirectory()) {
            const dirName = dirEnt.name
            // remove any file extension and prepend prefix
            const keyName = path.join(prefix, path.parse(dirName).name)
            keys.push(keyName)
          }
        }
        return keys
      })
  }

  /**
   * Saves json to at location specified by key
   * @param {string} key: relative path of file
   * @param {string} json: data to save
   */
  public save (key: string, json: string): Promise<void> {
    return new Promise((resolve, reject) => {
      const dir = this.fullDir(path.dirname(key))
      fs.ensureDir(dir, undefined,
        (ensureErr: NodeJS.ErrnoException | null) => {
          // no need to reject if dir existed
          if (ensureErr && ensureErr.code !== 'EEXIST') {
            reject(ensureErr)
            return
          }
          fs.writeFile(this.fullFile(key), json, (writeErr: Error) => {
            if (writeErr) {
              reject(writeErr)
              return
            }
            resolve()
          })
        })
    })
  }

  /**
   * Loads fields stored at a key
   * @param {string} key: relative path of file
   */
  public load (key: string): Promise<string> {
    return new Promise((resolve, reject) => {
      fs.readFile(this.fullFile(key), (err: Error, buf: Buffer) => {
        if (err) {
          reject(err)
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
  public delete (key: string): Promise<void> {
    return new Promise((resolve, reject) => {
      const fullDir = this.fullDir(key)
      if (fullDir !== '') {
        fs.remove(this.fullDir(key), (err) => {
          if (err) {
            reject(err)
            return
          }
          resolve()
        })
      } else {
        // Don't delete everything
        reject(Error('Delete failed- tried to delete home dir'))
      }
    })
  }

  /**
   * Makes relative path into full path
   */
  private fullDir (key: string): string {
    return path.join(this.dataDir, key)
  }

  /**
   * Makes relative path into full filename
   */
  private fullFile (key: string): string {
    return this.fullDir(key + '.json')
  }
}
