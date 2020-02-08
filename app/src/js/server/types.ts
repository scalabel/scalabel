import { BaseAction } from '../action/types'
import { AttributeToolType } from '../common/types'
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
export interface Env {
  /** Port that server listens on */
  port: number
  /** Directory data is saved to and loaded from */
  data: string
  /** Base directory path */
  src: string
  /** Path from base dir to compiled src code */
  appSubDir: string
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
  actions: BaseAction[]
}

/** user data for a project */
export interface UserData {
  /** map from socket to user */
  socketToUser: { [key: string]: string }
  /** map from user to list of socket */
  userToSockets: { [key: string]: string[] }
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

/* default config for env */
export const defaultEnv: Env = {
  port: 8686,
  data: './data',
  src: '.',
  appSubDir: 'app/dist',
  database: 'local',
  userManagement: false,
  sync: false,
  syncHost: 'http://localhost',
  autosave: true,
  redisTimeout: 3600,
  timeForWrite: 600,
  numActionsForWrite: 10,
  redisPort: 6379
}

/* default categories when file is missing and label is box2D or box3D */
export const defaultBoxCategories = [
  'person',
  'rider',
  'car',
  'truck',
  'bus',
  'train',
  'motor',
  'bike',
  'traffic sign',
  'traffic light'
]

/* default categories when file is missing and label is polyline2d */
export const defaultPolyline2DCategories = [
  'road curb',
  'double white',
  'double yellow',
  'double other',
  'single white',
  'single yellow',
  'single other',
  'crosswalk'
]

// TODO: add default seg2d categories once nested categories are supported

/* default attributes when file is missing and label is box2D */
export const defaultBox2DAttributes = [
  {
    name: 'Occluded',
    toolType: AttributeToolType.SWITCH,
    tagText: 'o',
    tagSuffixes: [],
    tagPrefix: '',
    values: [],
    buttonColors: []
  },
  {
    name: 'Truncated',
    toolType: AttributeToolType.SWITCH,
    tagText: 't',
    tagSuffixes: [],
    tagPrefix: '',
    values: [],
    buttonColors: []
  },
  {
    name: 'Traffic Color Light',
    toolType: AttributeToolType.LIST,
    tagText: 't',
    tagSuffixes: ['', 'g', 'y', 'r'],
    tagPrefix: '',
    values: ['NA', 'G', 'Y', 'R'],
    buttonColors: ['white', 'green', 'yellow', 'red']
  }
]

/* default attributes when file is missing and no other defaults exist */
export const dummyAttributes = [{
  name: '',
  toolType: AttributeToolType.NONE,
  tagText: '',
  tagSuffixes: [],
  values: [],
  tagPrefix: '',
  buttonColors: []
}]
