import AWS from "aws-sdk"
import _ from "lodash"
import * as path from "path"

import Logger from "./logger"
import { Storage } from "./storage"

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
   *
   * @param dataPath
   */
  constructor(dataPath: string) {
    // Check s3 data path
    const errorMsg = `s3 data path format is incorrect:
       Got:       ${dataPath}
       Should be: region:bucket/path`
    const error = Error(errorMsg)
    const info = dataPath.split(":")
    if (info.length < 2) {
      throw error
    }
    const bucketPath = info[1].split("/")
    if (bucketPath.length < 2) {
      throw error
    }
    const dataDir = path.join(...bucketPath.splice(1), "/")
    super(dataDir)

    this.region = info[0]
    this.bucketName = bucketPath[0]

    /**
     * To prevent requests hanging from invalid credentials,
     * Only check local credential services (not EC2/ECS ones)
     * See here for the difference from default:
     * https://docs.aws.amazon.com/AWSJavaScriptSDK/
     * latest/AWS/CredentialProviderChain.html
     */
    const chain = new AWS.CredentialProviderChain()
    chain.providers = [
      new AWS.EnvironmentCredentials("AWS"),
      new AWS.EnvironmentCredentials("AMAZON"),
      new AWS.SharedIniFileCredentials(),
      new AWS.ProcessCredentials()
    ]

    this.s3 = new AWS.S3({
      credentialProvider: chain,
      httpOptions: { connectTimeout: 10000 },
      maxRetries: 5
    })
  }

  /**
   * Init bucket
   */
  public async makeBucket(): Promise<void> {
    // Create new bucket if there isn't one already (wait until it exists)
    const hasBucket = await this.hasBucket()
    if (!hasBucket) {
      const bucketParams = {
        Bucket: this.bucketName,
        CreateBucketConfiguration: {
          LocationConstraint: this.region
        }
      }
      Logger.info(`Creating Bucket ${this.bucketName}`)
      try {
        await this.s3.createBucket(bucketParams).promise()
      } catch (error) {
        Logger.error(error)
      }
    }
  }

  /**
   * Remove the bucket
   */
  public async removeBucket(): Promise<void> {
    const params = {
      Bucket: this.bucketName
    }
    Logger.info(`Deleting Bucket ${this.bucketName}`)
    await this.s3.deleteBucket(params).promise()
  }

  /**
   * Check if specified file exists
   *
   * @param {string} key: relative path of file
   * @param key
   */
  public async hasKey(key: string): Promise<boolean> {
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
   *
   * @param {string} prefix: relative path of directory
   * @param prefix
   */
  public async listKeysOrganized(
    prefix: string
  ): Promise<[string[], string[]]> {
    const fullPrefix = this.fullDir(prefix)
    let continuationToken = ""

    let dirKeys = []
    let fileKeys = []
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

      if (data.Contents !== undefined) {
        for (const key of data.Contents) {
          // Remove any file extension and prepend prefix
          if (key.Key !== undefined) {
            const noPrefix = key.Key.substr(fullPrefix.length)

            // Parse to get the top level dir or file after prefix
            const parsed = path.parse(noPrefix)
            let keyName = parsed.name
            let isDir = false
            if (parsed.dir.length > 0 && parsed.dir !== "/") {
              const split = parsed.dir.split("/")
              keyName = split[0]
              if (keyName === "") {
                // This handles the case of an extra leading slash: '/dirname'
                keyName = split[1]
              }
              isDir = true
            }
            const finalKey = path.join(prefix, keyName)
            if (isDir) {
              dirKeys.push(finalKey)
            } else {
              fileKeys.push(finalKey)
            }
          }
        }
      }

      if (data.IsTruncated === undefined || !data.IsTruncated) {
        break
      }

      if (data.NextContinuationToken !== undefined) {
        continuationToken = data.NextContinuationToken
      }
    }

    dirKeys = _.uniq(dirKeys)
    fileKeys = _.uniq(fileKeys)
    dirKeys.sort()
    fileKeys.sort()
    return [dirKeys, fileKeys]
  }

  /**
   * Lists keys of files at directory specified by prefix
   *
   * @param {string} prefix: relative path of directory
   * @param {boolean} onlyDir: whether to only return keys that are directories
   * @param prefix
   * @param onlyDir
   */
  public async listKeys(
    prefix: string,
    onlyDir: boolean = false
  ): Promise<string[]> {
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
   *
   * @param {string} key: relative path of file
   * @param {string} json: data to save
   * @param key
   * @param json
   */
  public async save(key: string, json: string): Promise<void> {
    const params = {
      Body: json,
      Bucket: this.bucketName,
      Key: this.fullFile(key)
    }
    await this.s3.putObject(params).promise()
  }

  /**
   * Loads fields stored at a key
   *
   * @param {string} key: relative path of file
   * @param key
   */
  public async load(key: string): Promise<string> {
    const params: AWS.S3.GetObjectRequest = {
      Bucket: this.bucketName,
      Key: this.fullFile(key)
    }

    if (!(await this.hasKey(key))) {
      throw new Error(`Key '${params.Key}' does not exist`)
    }
    const data = await this.s3.getObject(params).promise()
    if (data.Body === undefined) {
      throw new Error(`No data at key '${params.Key}'`)
    } else {
      // This eslint problem seems to be a type error in s3
      // eslint-disable-next-line @typescript-eslint/no-base-to-string
      return data.Body.toString()
    }
  }

  /**
   * Deletes values at the key
   *
   * @param {string} key: relative path of directory
   * @param key
   */
  public async delete(key: string): Promise<void> {
    const [dirKeys, fileKeys] = await this.listKeysOrganized(key)

    const promises = []

    // Delete files
    for (const subKey of fileKeys) {
      const params = {
        Bucket: this.bucketName,
        Key: this.fullFile(subKey)
      }
      const deletePromise = this.s3.deleteObject(params).promise()
      promises.push(
        deletePromise.then(async () => {
          await this.s3.waitFor("objectNotExists", params).promise()
        })
      )
    }

    // Recursively delete subdirectories
    for (const subKey of dirKeys) {
      promises.push(this.delete(subKey))
    }

    await Promise.all(promises)
  }

  /**
   * make an empty folder object on s3
   *
   * @param key
   */
  public async mkdir(key: string): Promise<void> {
    const params = {
      Bucket: this.bucketName,
      Key: this.fullDir(key) + "/"
    }
    await this.s3.putObject(params).promise()
  }

  /**
   * Checks if bucket exists
   */
  private async hasBucket(): Promise<boolean> {
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
