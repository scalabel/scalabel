export interface RedisConfig {
  /** timeout (seconds) for clearing value from redis cache */
  timeout: number
  /** write to disk after this time interval (seconds) since last update */
  timeForWrite: number
  /** write to disk every time this number of actions occurs */
  numActionsForWrite: number
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
  clientId: string,
  /** user pool base uri */
  userPoolBaseUri: string,
  /** callback uri */
  callbackUri: string
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
  /** whether to save automatically */
  autosave: boolean
  /** turn on developer mode */
  dev: boolean
  /** redis config */
  redis: RedisConfig
  /** Bot config */
  bot: BotConfig
  /** cognito settings */
  cognito?: CognitoConfig
}
