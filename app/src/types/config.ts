import { StorageType } from "../const/config"

export interface RedisConfig {
  /**
   * Write to disk after this time interval (seconds) since last update.
   * If there is a new update, the countdown will restart.
   */
  writebackTime: number
  /**
   * Write to disk every time this number of saving (setting) to redis on
   * a key.
   */
  writebackCount: number
  /** Port that redis runs on */
  port: number
}

export interface BotConfig {
  /** Whether to use virtual sessions/bots for assistance */
  on: boolean
  /** host of python model server */
  host: string
  /** port of python model server */
  port: number
}

/**
 * Cognito server config
 *
 * @export
 * @interface CognitoConfig
 */
export interface CognitoConfig {
  /** region of cognito service */
  region: string
  /** user pool id of cognito */
  userPool: string
  /** client id of cognito */
  clientId: string
  /** user pool base uri */
  userPoolBaseUri: string
  /** callback uri */
  callbackUri: string
}

export interface ModeConfig {
  /** Flag to enable session synchronization */
  sync: boolean
  /** whether to save automatically */
  autosave: boolean
  /** turn on developer mode */
  dev: boolean
  /** launch server in demo mode */
  demo: boolean
}

/**
 * Configuration for user management system
 */
export interface UserConfig {
  /** turn on user management system */
  on: boolean
}

/**
 * Configuration for file and database storage
 */
export interface StorageConfig {
  /** type of the storage */
  type: StorageType
  /** storage path for scalabel system output */
  data: string
  /** Directory of local images and point clouds for annotation */
  itemDir: string
}

export interface HttpConfig {
  /** Port that server listens on */
  port: number
}

/**
 * Information for backend environment variables
 * Populated using configuration file
 */
export interface ServerConfig {
  /** http server configuratoin */
  http: HttpConfig
  /** storage configuration */
  storage: StorageConfig
  /** User management config */
  user: UserConfig
  /** server mode configuration */
  mode: ModeConfig
  /** redis config */
  redis: RedisConfig
  /** Bot config */
  bot: BotConfig
  /** cognito settings */
  cognito?: CognitoConfig

  /**
   * Port that server listens on
   * DEPRECATED: use http.port
   */
  port?: number
  /**
   * Where annotation logs and submissions are saved and loaded.
   * DEPRECATED: use storage.data
   */
  data?: string
  /**
   * Directory of local images and point clouds for annotation
   * DEPRECATED: use storage.itemDir
   */
  itemDir?: string
  /**
   * Database storage method
   * DEPRECATED: use storage.type
   */
  database?: StorageType
}
