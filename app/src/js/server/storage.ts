import * as path from 'path'

/**
 * Abstract class for storage
 */
export abstract class Storage {
  /** the data directory */
  protected dataDir: string

  /**
   * General constructor
   */
  protected constructor (basePath: string) {
    this.dataDir = basePath
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
   * Makes relative path into full path
   */
  protected fullDir (key: string): string {
    return path.join(this.dataDir, key)
  }

  /**
   * Makes relative path into full filename
   */
  protected fullFile (key: string): string {
    return this.fullDir(key + '.json')
  }
}
