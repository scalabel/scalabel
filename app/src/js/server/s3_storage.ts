import * as aws from 'aws-sdk'
import * as path from 'path'
import { Storage } from './storage'

/**
 * Implements local file storage
 */
export class S3Storage extends Storage {
  /** the region name */
  protected region: string
  /** the bucket name */
  protected bucketName: string
  /** the path within the bucket */
  protected dataPath: string
  /** the aws s3 client */
  protected s3: aws.s3

  /**
   * Constructor
   */
  async constructor (dataDir: string): Promise<void> {
    super(dataDir)

    // data path should have format region:bucket/path
    info = dataDir.split(':')
    this.region = info[0]
    bucketPath = info[1].split('/')
    this.bucketName = bucketPath[0]
    this.dataPath = path.join(bucketPath[1:], '/')
    this.s3 = new aws.s3()

    //create new bucket if there isn't one already (wait until it exists)
    const hasBucket = await this.hasBucket()
    if (!hasBucket) {
      const bucketParams = {
        Bucket: this.bucketName,
        CreateBucketConfiguration: {
          LocationConstraint: this.region,
        }
      }

      this.createBucket.then()
      const createBucketPromise = this.s3.createBucket(bucketParams).promise()
      const checkBucketPromise = this.s3.waitFor('bucketExists', bucketParams).promise()

      createBucketPromise.then((_data: string) => {
        checkBucketPromise
      })

      const bucketCreation = new Promise ((resolve, reject) => {
        this.s3.createBucket(bucketParams, (error: Error, data: string) => {
        if (error) {
          reject(error)
          return
        }
        resolve()
      })

      const bucketCreated = new Promise ((resolve, reject) => {
        this.s3.waitFor('bucketExists', params, (err: Error, data: string) => {
          if (err) {
            reject(error)
            return
          }
          resolve()
        })
      })

    } else {
      return Promise.resolve()
    }
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

  /**
   * Checks if bucket exists
   */
  private async hasBucket (): Promise<bool> {
    const params = {
      Bucket: this.bucketName
    }

    return new Promise((resolve, reject) => {
      this.s3.HeadBucket(params, (error: Error, data: string) => {
        if (error) {
          resolve(false)
        } else {
          resolve(true)
        }
      })
    })
      }
}
