import AWS from 'aws-sdk'
import * as path from 'path'
import Logger from './logger'
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
  protected s3: AWS.S3

  /**
   * Constructor
   */
  constructor (dataDir: string) {
    super(dataDir)

    // data path should have format region:bucket/path
    const info = dataDir.split(':')
    this.region = info[0]
    const bucketPath = info[1].split('/')
    this.bucketName = bucketPath[0]
    this.dataPath = path.join(...bucketPath.splice(1), '/')
    this.s3 = new AWS.S3()
  }

  /**
   * Init bucket
   */
  public async makeBucket (): Promise<void> {
    // create new bucket if there isn't one already (wait until it exists)
    const hasBucket = await this.hasBucket()
    if (!hasBucket) {
      const bucketParams = {
        Bucket: this.bucketName,
        CreateBucketConfiguration: {
          LocationConstraint: this.region
        }
      }
      try {
        await (this.s3.createBucket(bucketParams).promise())
        Logger.info('Waiting for bucket to be created.')
        await (this.s3.waitFor('bucketExists', bucketParams).promise())
      } catch (error) {
        return Promise.reject(error)
      }
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
    try {
      await (this.s3.headObject(params).promise())
      return true
    } catch (_error) {
      return false
    }
  }

  /**
   * Lists keys of files at directory specified by prefix
   * @param {string} prefix: relative path of directory
   * @param {boolean} onlyDir: whether to only return keys that are directories
   */
  public async listKeys (
    prefix: string, _onlyDir: boolean = false): Promise<string[]> {
    const fullKey = this.fullDir(prefix)
    let continuationToken = ''

    const keys = []
    for (;;) {
      const params = {
        Bucket: this.bucketName,
        Key: fullKey,
        ContinuationToken: continuationToken
      }

      try {
        const data = await (this.s3.listObjectsV2(params).promise())

        // TODO- deal with directories vs files
        if (data.Contents) {
          for (const key of data.Contents) {
            // remove any file extension and prepend prefix
            if (key.Key) {
              const keyName = path.join(prefix, path.parse(key.Key).name)
              keys.push(keyName)
            }
          }
        }

        if (!data.IsTruncated) {
          break
        }

        if (data.NextContinuationToken) {
          continuationToken = data.NextContinuationToken
        } else {
          continuationToken = ''
        }

      } catch (error) {
        return Promise.reject(error)
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
    const params = {
      Body: json,
      Bucket: this.bucketName,
      Key: this.fullFile(key)
    }
    try {
      await (this.s3.putObject(params).promise())
      return Promise.resolve()
    } catch (error) {
      return Promise.reject(error)
    }
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
    try {
      const data = (await (this.s3.getObject(params).promise())).Body
      if (data) {
        return data.toString()
      } else {
        return Promise.reject(Error('No data at key'))
      }
    } catch (error) {
      return Promise.reject(error)
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
    try {
      await (this.s3.deleteObject(params).promise())
      await (this.s3.waitFor('objectNotExists', params).promise())
      return Promise.resolve()
    } catch (error) {
      return Promise.reject(error)
    }
  }

  /**
   * Checks if bucket exists
   */
  private async hasBucket (): Promise<boolean> {
    const params = {
      Bucket: this.bucketName
    }

    try {
      await (this.s3.headBucket(params).promise())
      return true
    } catch (error) {
      return false
    }
  }
}
