import { File } from 'formidable'
import * as fs from 'fs-extra'
import * as yaml from 'js-yaml'
import { sprintf } from 'sprintf-js'
import * as yargs from 'yargs'
import { BundleFile, HandlerUrl, ItemTypeName, LabelTypeName } from '../common/types'
import { FileStorage } from './file_storage'
import Logger from './logger'
import { S3Storage } from './s3_storage'
import { Storage } from './storage'
import { CreationForm, DatabaseType, defaultEnv, Env } from './types'

/**
 * Initializes backend environment variables
 */
export function makeEnv (): Env {
  /**
   * Creates config, using defaults for missing fields
   * Make sure user env come last to override defaults
   */
  const userEnv = readEnv()
  const fullEnv = {
    ...defaultEnv,
    ...userEnv
  }
  return fullEnv
}

/**
 * Gets values for environment from user-specified file
 */
export function readEnv (): Partial<Env> {
  // read the config file name from argv
  const configFlag = 'config'
  const argv = yargs
    .option(configFlag, {
      describe: 'Config file path.'
    })
    .demandOption(configFlag)
    .string(configFlag)
    .argv
  const configDir: string = argv.config

  // load the config file
  return yaml.load(fs.readFileSync(configDir, 'utf8'))
}

/**
 * Initialize storage
 * @param {string} database: type of storage
 * @param {string} dir: directory name to save at
 */
export async function makeStorage (
  database: string, dir: string): Promise<Storage> {
  switch (database) {
    case DatabaseType.S3:
      const s3Store = new S3Storage(dir)
      try {
        await s3Store.makeBucket()
        return s3Store
      } catch (error) {
        // if s3 fails, default to file storage
        Logger.error(Error('s3 failed, using file storage'))
        Logger.error(error)
        return new FileStorage(dir)
      }
    case DatabaseType.DYNAMO_DB: {
      Logger.error(Error(sprintf(
        '%s storage not implemented yet, using file storage', database)))
      return new FileStorage(dir)
    }
    case DatabaseType.LOCAL: {
      return new FileStorage(dir)
    }
    default: {
      Logger.error(Error(sprintf(
        '%s is an unknown database format, using file storage', database)))
      return new FileStorage(dir)
    }
  }
}

/**
 * Builds creation form
 * With empty arguments, default is created
 */
export function makeCreationForm (
  projectName = '', itemType = '', labelType = '',
  pageTitle = '', taskSize = 0, instructions = '', demoMode = false
): CreationForm {
  const form: CreationForm = {
    projectName, itemType, labelType, pageTitle,
    instructions, taskSize, demoMode
  }
  return form
}

/**
 * Converts index into a filename of size 6 with
 * trailing zeroes
 */
export function index2str (index: number) {
  return index.toString().padStart(6, '0')
}

/**
 * Check whether form file exists
 */
export function formFileExists (file: File | undefined): boolean {
  return (file !== undefined && file.size !== 0)
}

/**
 * Gets the handler url for the project
 */
export function getHandlerUrl (itemType: string, labelType: string): string {
  switch (itemType) {
    case ItemTypeName.IMAGE:
      return HandlerUrl.LABEL
    case ItemTypeName.VIDEO:
      if (labelType === LabelTypeName.BOX_2D
        || labelType === LabelTypeName.POLYGON_2D
        || labelType === LabelTypeName.CUSTOM_2D) {
        return HandlerUrl.LABEL
      }
      return HandlerUrl.INVALID
    case ItemTypeName.POINT_CLOUD:
    case ItemTypeName.POINT_CLOUD_TRACKING:
      if (labelType === LabelTypeName.BOX_3D) {
        return HandlerUrl.LABEL
      }
      return HandlerUrl.INVALID
    case ItemTypeName.FUSION:
      if (labelType === LabelTypeName.BOX_3D) {
        return HandlerUrl.LABEL
      }
      return HandlerUrl.INVALID
    default:
      return HandlerUrl.INVALID
  }
}

/**
 * Get the bundle file for the project
 */
export function getBundleFile (labelType: string): string {
  // depends on redux progress
  if (labelType === LabelTypeName.TAG || labelType === LabelTypeName.BOX_2D) {
    return BundleFile.V2
  } else {
    return BundleFile.V1
  }
}

/**
 * Get whether tracking is on
 * and change item type accordingly
 */
export function getTracking (itemType: string): [string, boolean] {
  switch (itemType) {
    case ItemTypeName.VIDEO:
      return [ItemTypeName.IMAGE, true]
    case ItemTypeName.POINT_CLOUD_TRACKING:
      return [ItemTypeName.POINT_CLOUD, true]
    case ItemTypeName.FUSION:
      return [ItemTypeName.FUSION, true]
    default:
      return [itemType, false]
  }
}
