import { BaseAction } from '../action/types'
import { ItemExport } from '../functional/bdd_types'
import { Attribute, ConfigType, Label2DTemplateType, SensorType } from '../functional/types'

/**
 * Stores specifications of project
 */
export interface Project {
  /** array of items */
  items: Array<Partial<ItemExport>>
  /** frontend config */
  config: ConfigType
  /** map between data source id and data sources */
  sensors: {[id: number]: SensorType}
}

/**
 * Information for backend environment variables
 * Populated using configuration file
 */
export interface ServerConfig {
  /** Port that server listens on */
  port: number
  /** Where annotation logs and submissions are saved and loaded */
  data: string
  /** Directory of local images and point clouds for annotation */
  itemDir: string
  /** Database storage method */
  database: string
  /** Flag to enable user management */
  userManagement: boolean
  /** Flag to enable session synchronization */
  sync: boolean
  /** Hostname for synchronization socket */
  syncHost: string
  /** whether to save automatically */
  autosave: boolean
  /** timeout (seconds) for clearing value from redis cache */
  redisTimeout: number
  /** write to disk after this time interval (seconds) since last update */
  timeForWrite: number
  /** write to disk every time this number of actions occurs */
  numActionsForWrite: number
  /** Port that redis runs on */
  redisPort: number
}

/**
 * Form data for project creation
 */
export interface CreationForm {
  /** name of project */
  projectName: string
  /** item type */
  itemType: string
  /** label type */
  labelType: string
  /** title of page */
  pageTitle: string
  /** task size */
  taskSize: number
  /** instructions link */
  instructions: string
  /** whether demo mode is true */
  demoMode: boolean
}

/* file data parsed from form */
export interface FormFileData {
  /** categories parsed from form file */
  categories: string[]
  /** sensors */
  sensors: SensorType[]
  /** custom label template */
  templates: Label2DTemplateType[]
  /** attributes parsed from form file */
  attributes: Attribute[]
  /** items parsed from form file (may be incomplete) */
  items: Array<Partial<ItemExport>>
}

export interface RegisterMessageType {
  /** Project name of the socket connection */
  projectName: string
  /** Task index of the socket connection */
  taskIndex: number
  /** Current session Id */
  sessionId: string
  /** Current user Id */
  userId: string
}

/** action type for synchronization between front and back ends */
export interface SyncActionMessageType {
  /** Task Id. It is supposed to be index2str(taskIndex) */
  taskId: string
  /** Project name */
  projectName: string
  /** Session Id */
  sessionId: string
  /** List of actions for synchronization */
  actions: ActionPacketType
}

/** type for transmitted packet of actions */
export interface ActionPacketType {
  /** list of actions in the pakcket */
  actions: BaseAction[]
  /** id of the packet */
  id: string
}

/** metadata associated with a state */
export interface StateMetadata {
  /** project name */
  projectName: string
  /** task id */
  taskId: string
  /** map from processed action ids to their timestamps */
  actionIds: { [key: string]: number[] }
}

/** user data for a project */
export interface UserData {
  /** project name */
  projectName: string
  /** map from socket to user */
  socketToUser: { [key: string]: string }
  /** map from user to list of socket */
  userToSockets: { [key: string]: string[] }
}

/** metadata for all users for all projects */
export interface UserMetadata {
  /** map from socket to project */
  socketToProject: { [key: string]: string }
}

/**
 * Defining the types of some general callback functions
 */
export type MaybeError = Error | null | undefined
/* socket.io event names */
export const enum EventName {
  ACTION_BROADCAST = 'actionBroadcast',
  ACTION_SEND = 'actionSend',
  REGISTER_ACK = 'registerAck',
  REGISTER = 'register',
  CONNECTION = 'connection',
  CONNECT = 'connect',
  DISCONNECT = 'disconnect'
}

/* database types for storage */
export const enum DatabaseType {
  S3 = 's3',
  DYNAMO_DB = 'dynamodb',
  LOCAL = 'local'
}

// TODO: change constants from post to get once go code is removed
/* endpoint names for http server */
export const enum Endpoint {
  POST_PROJECT = '/postProject',
  GET_PROJECT_NAMES = '/postProjectNames',
  EXPORT = '/export',
  DASHBOARD = '/postDashboardContents'
}

/* form field names */
export const enum FormField {
  PROJECT_NAME = 'project_name',
  ITEM_TYPE = 'item_type',
  TRACKING = 'tracking',
  LABEL_TYPE = 'label_type',
  PAGE_TITLE = 'page_title',
  TASK_SIZE = 'task_size',
  INSTRUCTIONS_URL = 'instructions',
  DEMO_MODE = 'demo_mode',
  CATEGORIES = 'categories',
  ATTRIBUTES = 'attributes',
  ITEMS = 'item_file',
  SENSORS = 'sensors',
  LABEL_SPEC = 'label_spec'
}
