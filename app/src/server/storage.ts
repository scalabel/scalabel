import * as path from "path"

import { MAX_HISTORIES } from "../const/storage"
import Logger from "./logger"
import * as pathUtil from "./path"

/**
 * Abstract class for storage
 */
export abstract class Storage {
  /** the data directory */
  protected _dataDir: string
  /** the file extension to use */
  protected _extension: string

  /**
   * General constructor
   *
   * @param basePath
   */
  protected constructor(basePath: string) {
    this._dataDir = basePath
    this._extension = ".json"
  }

  /**
   * Get the internal data dir
   */
  public get dataDir(): string {
    return this._dataDir
  }

  /**
   * Change the file extension for different file types
   * Can set to empty string if the keys already include extensions
   *
   * @param newExt
   */
  public setExt(newExt: string): void {
    this._extension = newExt
  }

  /**
   * Extension of the key when it is stored
   * Default is json, but it can be modified with setExt
   */
  public keyExt(): string {
    return this._extension
  }

  /**
   * Check if storage has key
   */
  public abstract async hasKey(key: string): Promise<boolean>

  /**
   * Lists keys in storage
   *
   * @param {string} prefix: relative path of directory
   * @param {boolean} onlyDir: whether to only return keys that are directories
   * @returns {Promise<string[]>} The keys are returned in lexical order
   */
  public abstract async listKeys(
    prefix: string,
    onlyDir: boolean
  ): Promise<string[]>

  /**
   * Saves json to a key
   */
  public abstract async save(key: string, json: string): Promise<void>

  /**
   * Loads json stored at a key
   */
  public abstract async load(key: string): Promise<string>

  /**
   * Deletes values at the key
   */
  public abstract async delete(key: string): Promise<void>

  /**
   * Create a new folder
   *
   * @param key
   */
  public abstract async mkdir(key: string): Promise<void>

  /**
   * Loads the JSON if it exists
   * Otherwise return something that evaluates to false
   *
   * @param key
   */
  public async safeLoad(key: string): Promise<string> {
    if (await this.hasKey(key)) {
      return await this.load(key)
    }
    return ""
  }

  /**
   * Save the key with history backup. The key is treated as a folder or prefix
   * With this function, we can save MAX_HISTORIES in the folder indexed by key
   * The goal is to be robust to possible failures when using disks or
   * network drives. If database is used as the storage, this backup may not be
   * useful.
   *
   * @param key
   * @param value
   */
  public async saveWithBackup(key: string, value: string): Promise<void> {
    const fileKey = pathUtil.getFileKey(key)
    await this.save(fileKey, value)
    // Check whether there are more than MAX_HISTORIES entries in the folder
    // If so, delete the old ones
    const keys = await this.listKeys(key, false)
    for (let i = 0; i < keys.length - MAX_HISTORIES; i += 1) {
      await this.delete(keys[i] + this.keyExt())
    }
    Logger.info(
      `Saving ${fileKey} and ` +
        `deleting ${Math.max(keys.length - MAX_HISTORIES, 0)} historical keys`
    )
  }

  /**
   * Get the value from
   *
   * @param key
   */
  public async getWithBackup(key: string): Promise<string | null> {
    const keys = await this.listKeys(key, false)
    if (keys.length > 0) {
      return await this.load(keys[keys.length - 1])
    } else {
      return null
    }
  }

  /**
   * Makes relative path into full path
   *
   * @param key
   */
  public fullDir(key: string): string {
    return path.join(this._dataDir, key)
  }

  /**
   * Makes relative path into full filename
   *
   * @param key
   */
  public fullFile(key: string): string {
    return this.fullDir(key + this.keyExt())
  }
}
