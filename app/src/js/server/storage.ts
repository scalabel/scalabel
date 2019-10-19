/**
 * Abstract class for storage
 */
export abstract class Storage {
  /** the data directory */
  protected dataDir: string

  /**
   * General constructor
   */
  protected constructor (path: string) {
    this.dataDir = path
  }

  /**
   * Check if storage has key
   */
  public abstract hasKey (key: string): Promise<boolean>

  /**
   * Lists keys in storage
   */
  public abstract listKeys (prefix: string, onlyDir: boolean): Promise<string[]>

  /**
   * Saves json to a key
   */
  public abstract save (key: string, json: string): Promise<void>

  /**
   * Loads json stored at a key
   */
  public abstract load (key: string): Promise<string>

  /**
   * Deletes valus at the key
   */
  public abstract delete (key: string): Promise<void>
}
