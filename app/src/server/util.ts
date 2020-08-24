
import _ from 'lodash'
import { configureStore } from '../common/configure_store'
import { uid } from '../common/uid'
import {
  BundleFile, HandlerUrl, ItemTypeName,
  LabelTypeName, TrackPolicyType
} from '../const/common'
import { StorageType } from '../const/config'
import { STORAGE_FOLDERS } from '../const/storage'
import { BaseAction } from '../types/action'
import { ItemExport } from '../types/bdd'
import { CreationForm, UserData, UserMetadata } from '../types/project'
import { Label2DTemplateType, State } from '../types/state'
import { FileStorage } from './file_storage'
import Logger from './logger'
import { S3Storage } from './s3_storage'
import { Storage } from './storage'

/**
 * Initialize storage
 * @param {string} database: type of storage
 * @param {string} dir: directory name to save at
 */
export async function makeStorage (
  database: string, dir: string): Promise<Storage> {
  let storage: Storage
  switch (database) {
    case StorageType.S3:
      try {
        const s3Store = new S3Storage(dir)
        await s3Store.makeBucket()
        storage = s3Store
      } catch (error) {
        // If s3 fails, default to file storage
        error.message = `s3 failed, using file storage ${error.message}`
        Logger.error(error)
        storage = new FileStorage(dir)
      }
      break
    case StorageType.DYNAMO_DB: {
      Logger.error(Error(
        `${database} storage not implemented yet, using file storage`))
      storage = new FileStorage(dir)
      break
    }
    case StorageType.LOCAL: {
      storage = new FileStorage(dir)
      break
    }
    default: {
      Logger.error(Error(
        `${database} is an unknown database format, using file storage`))
      storage = new FileStorage(dir)
    }
  }
  // Create the initial folder structure
  STORAGE_FOLDERS.map((f) => storage.mkdir(f))
  return storage
}

/**
 * Builds creation form
 * With empty arguments, default is created
 */
export function makeCreationForm (
  projectName = '', itemType = '', labelType = '',
  pageTitle = '', taskSize = 0, instructionUrl = '', demoMode = false
): CreationForm {
  const form: CreationForm = {
    projectName, itemType, labelType, pageTitle,
    instructionUrl, taskSize, demoMode
  }
  return form
}

/**
 * Initialize new session id if its a new load
 * If its a reconnection, keep the old session id
 */
export function initSessionId (sessionId: string) {
  return (sessionId ? sessionId : uid())
}

/**
 * Gets the handler url for the project
 */
export function getHandlerUrl (
  itemType: string, labelType: string): string {
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
  // Depends on redux progress
  if (labelType === LabelTypeName.TAG || labelType === LabelTypeName.BOX_2D) {
    return BundleFile.V2
  } else {
    return BundleFile.V1
  }
}

/**
 * Chooses the policy type and label types based on item and label types
 */
export function getPolicy (
  itemType: string, labelTypes: string[],
  policyTypes: string[], templates2d: { [name: string]: Label2DTemplateType }):
  [string[], string[]] {
  // TODO: Move this to be in front end after implementing label selector
  switch (itemType) {
    case ItemTypeName.IMAGE:
    case ItemTypeName.VIDEO:
      if (labelTypes.length === 1) {
        switch (labelTypes[0]) {
          case LabelTypeName.BOX_2D:
            return [[TrackPolicyType.LINEAR_INTERPOLATION], labelTypes]
          case LabelTypeName.POLYGON_2D:
            return [[TrackPolicyType.LINEAR_INTERPOLATION], labelTypes]
          case LabelTypeName.CUSTOM_2D:
            labelTypes[0] = Object.keys(templates2d)[0]
            return [[TrackPolicyType.LINEAR_INTERPOLATION], labelTypes]
        }
      }
      return [policyTypes, labelTypes]
    case ItemTypeName.POINT_CLOUD:
    case ItemTypeName.POINT_CLOUD_TRACKING:
      if (labelTypes.length === 1 &&
        labelTypes[0] === LabelTypeName.BOX_3D) {
        return [[TrackPolicyType.LINEAR_INTERPOLATION], labelTypes]
      }
      return [policyTypes, labelTypes]
    case ItemTypeName.FUSION:
      return [[TrackPolicyType.LINEAR_INTERPOLATION],
      [LabelTypeName.BOX_3D, LabelTypeName.PLANE_3D]]
    default:
      return [policyTypes, labelTypes]
  }
}

/**
 * Loads JSON and logs error if invalid
 */
export function safeParseJSON (data: string) {
  try {
    const parsed = JSON.parse(data)
    return parsed
  } catch (e) {
    Logger.error(Error('JSON parsed failed'))
    Logger.error(e)
  }
}

/**
 * Updates a state with a series of timestamped actions
 */
export function updateStateTimestamp (
  state: State, actions: BaseAction[]): [State, number[]] {
  const stateStore = configureStore(state)

  // For each action, update the store
  const timestamps = []
  for (const action of actions) {
    const time = Date.now()
    timestamps.push(time)
    action.timestamp = time
    stateStore.dispatch(action)
  }

  return [stateStore.getState().present, timestamps]
}

/**
 * Updates a state with a series of actions
 */
export function updateState (
  state: State, actions: BaseAction[]): State {
  const stateStore = configureStore(state)

  // For each action, update the store
  for (const action of actions) {
    stateStore.dispatch(action)
  }

  return stateStore.getState().present
}

/**
 * Builds empty user data except for project name
 */
export function makeUserData (projectName: string): UserData {
  return {
    projectName,
    socketToUser: {},
    userToSockets: {}
  }
}

/**
 * Builds empty user metadata
 */
export function makeUserMetadata (): UserMetadata {
  return {
    socketToProject: {}
  }
}

/**
 * Get item timestamp, or 0 if undefined
 */
export function getItemTimestamp (item: Partial<ItemExport>): number {
  const timestamp = item.timestamp
  if (timestamp !== undefined) {
    return timestamp
  } else {
    return 0
  }
}

/**
 * Parse the project name into internal format
 */
export function parseProjectName (projectName: string): string {
  return projectName.replace(' ', '_')
}

/**
 * Get connection failed error message for http request to python
 */
export function getPyConnFailedMsg (endpoint: string, message: string): string {
  return `Make sure endpoint is correct and python server is \
running; query to \"${endpoint}\" failed with message: ${message}`
}

/**
 * helper function to force javascript to sleep
 * @param milliseconds
 */
export function sleep (milliseconds: number): Promise<object> {
  return new Promise((resolve) => setTimeout(resolve, milliseconds))
}
