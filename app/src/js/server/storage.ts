import * as path from 'path'

export enum StorageStructure {
  PROJECT = 'projects',
  USER = 'users'
}

export const STORAGE_FOLDERS = [StorageStructure.PROJECT, StorageStructure.USER]

/**
 * Abstract class for storage
 */
export abstract class Storage {
  /** the data directory */
  protected _dataDir: string

  /**
   * General constructor
   */
  protected constructor (basePath: string) {
    this._dataDir = basePath
  }

  /**
   * Get the internal data dir
   */
  public get dataDir (): string {
    return this._dataDir
  }

  /**
   * Extension of the key when it is stored
   * Only file and s3 storages are supported now, so return .json directly
   */
  public keyExt (): string {
    return '.json'
  }

  /**
   * Check if storage has key
   */
  public abstract async hasKey (key: string): Promise<boolean>

  /**
   * Lists keys in storage
   * @param {string} prefix: relative path of directory
   * @param {boolean} onlyDir: whether to only return keys that are directories
   * @returns {Promise<string[]>} The keys are returned in lexical order
   */
  public abstract async listKeys (
    prefix: string, onlyDir: boolean): Promise<string[]>

  /**
   * Saves json to a key
   */
  public abstract async save (key: string, json: string): Promise<void>

  /**
   * Loads json stored at a key
   */
  public abstract async load (key: string): Promise<string>

  /**
   * Deletes values at the key
   */
  public abstract async delete (key: string): Promise<void>

  /**
   * Create a new folder
   * @param key
   */
  public abstract async mkdir (key: string): Promise<void>

  /**
   * Loads the JSON if it exists
   * Otherwise return something that evaluates to false
   */
  public async safeLoad (key: string): Promise<string> {
    if (await this.hasKey(key)) {
      return this.load(key)
    }
    return ''
  }

  /**
   * Makes relative path into full path
   */
  public fullDir (key: string): string {
    return path.join(this._dataDir, key)
  }

  /**
   * Makes relative path into full filename
   */
  public fullFile (key: string): string {
    return this.fullDir(key + this.keyExt())
  }
}
