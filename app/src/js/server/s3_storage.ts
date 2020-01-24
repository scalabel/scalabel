import AWS from 'aws-sdk'
import _ from 'lodash'
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
  /** the aws s3 client */
  protected s3: AWS.S3

  /**
   * Constructor
   */
  constructor (dataPath: string) {
    // data path should have format region:bucket/path
    const info = dataPath.split(':')
    const bucketPath = info[1].split('/')
    const dataDir = path.join(...bucketPath.splice(1), '/')
    super(dataDir)

    this.region = info[0]
    this.bucketName = bucketPath[0]
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
      await this.s3.createBucket(bucketParams).promise()
      Logger.info('Waiting for bucket to be created.')
      const waitParams = {
        Bucket: this.bucketName
      }
      await this.s3.waitFor('bucketExists', waitParams).promise()
    }
    return
  }

  /**
   * Check if specified file exists
   * @param {string} key: relative path of file
   */
  public async hasKey (key: string): Promise<boolean> {
    const params = {
      Bucket: this.bucketName,
      Key: this.fullFile(key)
    }
    try {
      await this.s3.headObject(params).promise()
      return true
    } catch (_error) {
      return false
    }
  }

  /**
   * Lists keys of files at a directory specified by prefix
   * Split by files and directories
   * @param {string} prefix: relative path of directory
   */
  public async listKeysOrganized (
    prefix: string): Promise<[string[], string[]]> {
    const fullPrefix = this.fullDir(prefix)
    let continuationToken = ''

    const dirKeys = []
    const fileKeys = []
    for (;;) {
      let data
      if (continuationToken.length > 0) {
        const params = {
          Bucket: this.bucketName,
          Prefix: fullPrefix,
          ContinuationToken: continuationToken
        }
        data = await this.s3.listObjectsV2(params).promise()
      } else {
        const params = {
          Bucket: this.bucketName,
          Prefix: fullPrefix
        }
        data = await this.s3.listObjectsV2(params).promise()
      }

      if (data.Contents) {
        for (const key of data.Contents) {
          // remove any file extension and prepend prefix
          if (key.Key) {
            const noPrefix = key.Key.substr(fullPrefix.length)

            // Parse to get the top level dir or file after prefix
            const parsed = path.parse(noPrefix)
            let keyName = parsed.name
            if (parsed.dir.length > 0 && parsed.dir !== '/') {
              keyName = parsed.dir.split('/')[0]
            }
            const finalKey = path.join(prefix, keyName)
            if (parsed.ext === '') {
              dirKeys.push(finalKey)
            } else {
              fileKeys.push(finalKey)
            }
          }
        }
      }

      if (!data.IsTruncated) {
        break
      }

      if (data.NextContinuationToken) {
        continuationToken = data.NextContinuationToken
      }
    }

    _.uniq(dirKeys)
    _.uniq(fileKeys)
    dirKeys.sort()
    fileKeys.sort()
    return [dirKeys, fileKeys]
  }

  /**
   * Lists keys of files at directory specified by prefix
   * @param {string} prefix: relative path of directory
   * @param {boolean} onlyDir: whether to only return keys that are directories
   */
  public async listKeys (
    prefix: string, onlyDir: boolean = false): Promise<string[]> {
    const [dirKeys, fileKeys] = await this.listKeysOrganized(prefix)
    if (onlyDir) {
      return dirKeys
    } else {
      const mergeKeys = dirKeys.concat(fileKeys)
      mergeKeys.sort()
      return mergeKeys
    }
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
    return this.s3.putObject(params).promise().then()
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

    if (!this.hasKey(key)) {
      return Promise.reject('Key does not exist')
    }
    const data = await this.s3.getObject(params).promise()
    if (!data || !data.Body) {
      return Promise.reject(Error('No data at key'))
    }

    return data.Body.toString()
  }

  /**
   * Deletes values at the key
   * @param {string} key: relative path of directory
   */
  public async delete (key: string): Promise<void> {
    const [dirKeys, fileKeys] = await this.listKeysOrganized(key)

    const promises = []

    // delete files
    for (const subKey of fileKeys) {
      const params = {
        Bucket: this.bucketName,
        Key: this.fullFile(subKey)
      }
      const deletePromise = this.s3.deleteObject(params).promise()
      promises.push(deletePromise.then(async () => {
        await this.s3.waitFor('objectNotExists', params).promise()
      }))
    }

    // recursively delete subdirectories
    for (const subKey of dirKeys) {
      promises.push(this.delete(subKey))
    }

    return Promise.all(promises).then(() => { return })
  }

  /**
   * Checks if bucket exists
   */
  private async hasBucket (): Promise < boolean > {
    const params = {
      Bucket: this.bucketName
    }

    try {
      await this.s3.headBucket(params).promise()
      return true
    } catch (error) {
      return false
    }
  }
}
