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
  constructor (dataDir: string): Promise<void> {
    super(dataDir)

    // data path should have format region:bucket/path
    const info = dataDir.split(':')
    this.region = info[0]
    const bucketPath = info[1].split('/')
    this.bucketName = bucketPath[0]
    this.dataPath = path.join(...bucketPath.splice(1), '/')
    this.s3 = new aws.s3()
  }

  /**
   * Init bucket
   */
  public async makeBucket (): Promise<void> {
    //create new bucket if there isn't one already (wait until it exists)
    const hasBucket = await this.hasBucket()
    if (!hasBucket) {
      const bucketParams = {
        Bucket: this.bucketName,
        CreateBucketConfiguration: {
          LocationConstraint: this.region,
        }
      }

      await (this.s3.createBucket(bucketParams).promise())
      await (this.s3.waitFor('bucketExists', bucketParams).promise())
    }
    return Promise.resolve()
  }

  /**
   * Check if specified file exists
   * @param {string} key: relative path of file
   */
  public async hasKey (key: string): Promise<boolean> {
    const params = {
      Bucket: this.bucketName, 
      Key: key 
     }
    const [err, _data] = await (this.s3.headObject(params).promise())
    return err === null
  }

  /**
   * Lists keys of files at directory specified by prefix
   * @param {string} prefix: relative path of directory
   * @param {boolean} onlyDir: whether to only return keys that are directories
   */
  public async listKeys (
    prefix: string, onlyDir: boolean = false): Promise<string[]> {
    const fullKey = this.fullDir(prefix)
    let continuationToken = ''

    let keys = []
    for (;;) {
      let params = {
        Bucket: this.bucketName,
        Key: fullKey,
        ContinuationToken: continuationToken
      }

      let [err, data] = await (this.s3.listObjectsV2(params).promise())
      if (err !== null) {
        return Promise.reject(err) 
      }
      //TODO- deal with directories vs files
      for (const key of data.Contents) {
        // remove any file extension and prepend prefix
        const keyName = path.join(prefix, path.parse(key.key).name)
        keys.push(keyName)
      }

      if (!data.IsTruncated) {
        break
      }

      continuationToken = data.NextContinuationToken
    }

    keys.sort()
    return Promise.resolve(keys)
  }

  /**
   * Saves json to at location specified by key
   * @param {string} key: relative path of file
   * @param {string} json: data to save
   */
  public async save (key: string, json: string): Promise<void> {
    const params = {
      Bucket: this.bucketName,
      Key: this.fullFile(key)
    }
    await (this.s3.putObject(params, json).promise())
    
    return Promise.resolve()
  }

  /**
   * Loads fields stored at a key
   * @param {string} key: relative path of file
   */
  public async load (key: string): Promise<string> {
    const params = {
      Bucket: this.bucketName, 
      Key: this.fullFile(key) 
     }
    const [err, data] = await (this.s3.getObject(params).promise())
    if (err !== null) {
      return Promise.reject(err)
    } else {
      return Promise.resolve(data)
    }
  }

  /**
   * Deletes values at the key
   * @param {string} key: relative path of directory
   */
  public async delete (key: string): Promise<void> {
    const params = {
      Bucket: this.bucketName,
      Key: this.fullDir(key)
    }
    await (this.s3.deleteObject(params).promise())
    await (this.s3.waitFor('objectNotExists', params).promise())

    return Promise.resolve()
  }

  /**
   * Checks if bucket exists
   */
  private async hasBucket (): Promise<boolean> {
    const params = {
      Bucket: this.bucketName
    }

    return new Promise((resolve, _reject) => {
      this.s3.HeadBucket(params, (error: Error, _data: string) => {
        if (error) {
          resolve(false)
        } else {
          resolve(true)
        }
      })
    })
      }
}
